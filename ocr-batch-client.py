#!/usr/bin/env python3
"""ocr-batch-client.py — drive a page-image OCR pass against an OpenAI-compatible
vision endpoint, submitting pages CONCURRENTLY so a batching server can overlap them.

WHY CONCURRENCY IS THE WHOLE POINT
----------------------------------
A vision server's throughput on a document pass depends far more on HOW you submit than
on the model. Measured on one MI300X (gfx942), 8 distinct pages, same GPU:

    engine            c=1      c=4      c=8     shape
    llama.cpp:rocm    5.4s     5.5s     7.0s    FLAT — encode serializes in one loop
    vLLM (batching)   6.6s     2.0s     3.5s    SCALES — 3.3x at c=4 over its own c=1

So a sequential client (submit page, wait, submit next) leaves a batching server idle
between requests and you get llama.cpp-shaped numbers even on vLLM. This client keeps N
requests in flight at once (--concurrency), which is the only way vLLM's continuous
batching turns into wall-clock. On a non-batching backend it costs nothing (still correct,
just no speedup), so the same client is safe against either.

Pick --concurrency from a short sweep on YOUR endpoint+model+page size: throughput rises
with concurrency until the server's KV/compute saturates, then flattens or regresses (c=8
above was already past the knee for this model). Start at 4, sweep 1/2/4/8/16, keep the
best docs/hour.

CONTRACT
--------
- Endpoint: any OpenAI-compatible POST {endpoint}/v1/chat/completions that accepts an
  image_url data: URI (vLLM `vllm serve <vision-model>`, or lemonade with a vision recipe).
- Input: a directory of page images (png/jpg/jpeg/webp/gif), processed in name order.
- Output: JSONL, one line per page {file, ok, text, latency_s, error}, plus a summary.
- Resumable: pages already present in the output JSONL are skipped, so a re-run continues.

STDLIB ONLY — no pip install on the box. Auth via --api-key or $OCR_API_KEY (sent as
Bearer) when the endpoint requires it.

DATA-HANDLING — the NO-DATA-AT-REST pattern (proven in production, 1531 client pages):
run THIS CLIENT ON YOUR OWN MACHINE, reading images from your own disk, and reach the
remote server over an SSH tunnel — do NOT copy the images to the rented box. The server
serves on the host (a `--network host` container binds the host port), so a plain
`ssh -N -L 8000:localhost:<server-port> user@box` forwards it, and you point
`--endpoint http://127.0.0.1:8000` at the tunnel. Net effect: each page's pixels exist on
the rented box only transiently in the vLLM process during its own request; NO client file
is ever written to the box's disk. This makes "the data never leaves infrastructure we
control" much closer to true than an rsync-the-images-over approach (which lands them on
disk). Still a risk-owner decision, but this is the pattern that satisfies the stronger
claim. NOTE the tunnel LOCAL port and the server's actual `--port` need not match; set
--endpoint to whatever you forwarded to. (If you instead run the client ON the box, the
images do land on its disk — the weaker posture.)
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = "Transcribe ALL text in this image exactly. Output only the transcription."
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def encode(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def transcribe(endpoint, model, path, api_key, max_tokens, timeout, prompt):
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode(path)}},
        ]}],
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        text = d["choices"][0]["message"]["content"]
        return {"file": os.path.basename(path), "ok": True, "text": text,
                "latency_s": round(time.monotonic() - t0, 2), "error": None}
    except (urllib.error.URLError, KeyError, ValueError) as e:
        detail = e.read().decode()[:300] if isinstance(e, urllib.error.HTTPError) else str(e)
        return {"file": os.path.basename(path), "ok": False, "text": None,
                "latency_s": round(time.monotonic() - t0, 2), "error": detail}


def main():
    p = argparse.ArgumentParser(description="Parallel OCR client for an OpenAI-compatible vision endpoint.")
    p.add_argument("--endpoint", required=True, help="e.g. http://127.0.0.1:8000 (vLLM) or http://127.0.0.1:13305 (lemonade)")
    p.add_argument("--model", required=True, help="served model name, e.g. Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
    p.add_argument("--input-dir", required=True, help="directory of page images")
    p.add_argument("--out", default="ocr-results.jsonl", help="JSONL output (append + resume)")
    p.add_argument("--concurrency", type=int, default=4, help="requests in flight (the batching knob; sweep it)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=600, help="per-request seconds")
    p.add_argument("--api-key", default=os.getenv("OCR_API_KEY"), help="Bearer token if the endpoint needs one")
    p.add_argument("--prompt", default=PROMPT)
    a = p.parse_args()

    pages = sorted(os.path.join(a.input_dir, f) for f in os.listdir(a.input_dir)
                   if f.lower().endswith(EXTS))
    if not pages:
        sys.exit(f"no page images ({'/'.join(EXTS)}) in {a.input_dir}")

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("ok"):
                        done.add(row["file"])
                except ValueError:
                    pass
    todo = [p for p in pages if os.path.basename(p) not in done]
    print(f"{len(pages)} pages, {len(done)} already done, {len(todo)} to do, "
          f"concurrency={a.concurrency}", flush=True)
    if not todo:
        return

    t0 = time.monotonic()
    ok = err = 0
    with open(a.out, "a") as out, ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(transcribe, a.endpoint, a.model, pg, a.api_key,
                          a.max_tokens, a.timeout, a.prompt): pg for pg in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            ok += row["ok"]
            err += not row["ok"]
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.monotonic() - t0) * 3600
                print(f"  {i}/{len(todo)}  ok={ok} err={err}  {rate:.0f} pages/hr", flush=True)

    elapsed = time.monotonic() - t0
    print(f"done: {ok} ok, {err} error in {elapsed:.1f}s "
          f"({ok / elapsed * 3600:.0f} pages/hr at concurrency={a.concurrency})")
    if err:
        print(f"  {err} failed — see error lines in {a.out}; re-run to retry them (resume skips ok pages)")
        sys.exit(1)


if __name__ == "__main__":
    main()
