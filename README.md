# Cloud-Init Templates

Example cloud-init templates for HotManager VM provisioning.

## vllm-docker.yaml

Starts vLLM Docker container with GPU acceleration on first boot.

## lemonade-server.yaml

Installs [Lemonade Server](https://lemonade-server.ai) from its
[PPA](https://launchpad.net/~lemonade-team/+archive/ubuntu/stable) and brings up an
OpenAI-compatible endpoint on first boot.

Where `vllm-docker.yaml` stands a serving stack up on the box, this one installs one: the
server, model manager and web UI arrive as a signed Debian package, so there is no
container to pull and no ROCm/PyTorch reconciliation to get right. The server then fetches
a prebuilt ROCm llama.cpp backend for the GPU it finds — nothing is compiled on the box.

The package creates a `lemonade` system user, adds it to `render` for GPU access, and
enables `lemond.service`, which listens on `127.0.0.1:13305`.

```bash
curl -X 'POST' \
  "${API_BASE}/teams/${TEAM_SLUG}/virtual_machines/" \
  -H 'accept: application/json' \
  -H "Authorization: Token ${API_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{
    "gpus": [
      {
        "count": 1,
        "manufacturer": "AMD",
        "model": "MI300X"
      }
    ],
    "user_data_url": "https://raw.githubusercontent.com/hotaisle/cloud-init-templates/master/lemonade-server.yaml"
  }'
```

### Accessing Lemonade

ufw permits port 22 only, so forward the port over SSH:

```bash
ssh -L 13305:localhost:13305 hotaisle@<vm-ip>
```

Then, from your machine — the API is OpenAI-compatible, so any OpenAI client works by
pointing `base_url` at it:

```bash
curl http://localhost:13305/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B-GGUF",
    "messages": [{"role": "user", "content": "Write a haiku about artificial intelligence"}],
    "max_tokens": 128
  }'
```

The web UI is on the same port, at <http://localhost:13305>.

On the VM itself, `lemonade` is a CLI client for the running server:

```bash
lemonade status        # what is loaded
lemonade list          # available models
lemonade pull <model>  # fetch another model
lemonade backends      # backends and the GPU they resolved to
```

## API Usage

### Environment Variables

```bash
export API_TOKEN="your-api-token-here"
export API_BASE="https://admin.hotaisle.app/api"
export TEAM_SLUG="your-team"
```

### Example API Call

```bash
curl -X 'POST' \
  "${API_BASE}/teams/${TEAM_SLUG}/virtual_machines/" \
  -H 'accept: application/json' \
  -H "Authorization: Token ${API_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{
    "gpus": [
      {
        "count": 1,
        "manufacturer": "AMD",
        "model": "MI300X"
      }
    ],
    "user_data_url": "https://raw.githubusercontent.com/hotaisle/cloud-init-templates/master/vllm-docker.yaml"
  }'
```

## Accessing vLLM

```bash
# SSH to VM
ssh hotaisle@<vm-ip>

# Test vLLM API from inside VM
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "prompt": "Write a haiku about artificial intelligence",
    "max_tokens": 128,
    "top_p": 0.95,
    "top_k": 20,
    "description": "Test vLLM API call"
  }'
```

## Important Note on VM Availability

When provisioning a VM with cloud-init, the VM will reboot and may not be available over SSH until cloud-init completes. This is different from VMs provisioned without cloud-init, which are available immediately.

VMs with cloud-init typically take 1-2 minutes to become fully available as cloud-init runs on first boot. During this time, the VM will be rebooting and may not respond to SSH attempts.

## Accessing vLLM

VMs have ufw firewall enabled (port 22 only). Access vLLM via SSH:

```bash
# SSH to VM
ssh hotaisle@<vm-ip>

# Test vLLM API from inside VM
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "prompt": "Write a hatiku about artificial intelligence",
    "max_tokens": 128,
    "top_p": 0.95,
    "top_k": 20,
    "temperature": 0.8
  }'
```

## Creating Your Own Cloud-Init Files

All you need is a URL to a cloud-init user-data file. Here are some options for hosting your files:

### Fork This Repository

Fork the repository to create your own templates or edit the existing templates.

### GitHub Gist

Create GitHub gists with your cloud-init content. Gists are perfect for quick sharing and provide version control automatically.

## Hot Aisle API Reference

For the complete Hot Aisle API documentation and Swagger reference:
<https://admin.hotaisle.app/api/docs/>

## Cloud-Init Reference

For complete cloud-init documentation and reference:
<https://cloudinit.readthedocs.io/en/latest/reference/index.html>

## ocr-batch-client.py

Drives a document OCR pass against an OpenAI-compatible vision endpoint (either template's
serving stack), submitting pages **concurrently** so a batching server (vLLM) overlaps
them. Sequential submission gives you llama.cpp-shaped numbers even on vLLM; `--concurrency`
is what turns continuous batching into wall-clock. Measured on one MI300X, 8 pages: vLLM
3.3x faster at concurrency=4 than at 1, while llama.cpp stayed flat. Stdlib-only, JSONL
output, resumable. Sweep `--concurrency` (1/2/4/8/16) on your endpoint and keep the best
pages/hour.

```bash
./ocr-batch-client.py --endpoint http://127.0.0.1:8000 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-dir ./pages --concurrency 4 --out results.jsonl
```

Note: a rented GPU box is third-party infrastructure; the images are transported to it and
sit on its disk. Whether that fits a given data-handling posture is a risk-owner decision.
