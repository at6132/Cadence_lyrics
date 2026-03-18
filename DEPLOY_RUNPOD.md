# Deploy Lyric Pipeline on RunPod Serverless

Run the full lyric pipeline (draft → evaluate → rewrite → score) as a serverless GPU endpoint on RunPod.

## Prerequisites

- Trained LoRA adapter in `Lyric_model/adapters/lyric-human/` (from finetune).
- Docker (for building the image).
- RunPod account and serverless endpoint set up.

## Build the image

From the **Lyric_model** directory (so `adapters/` is in context):

```bash
cd Lyric_model
docker build -f Dockerfile.runpod -t your-registry/lyric-pipeline:latest .
```

If your adapter is elsewhere, copy it in first:

```bash
cp -r /path/to/lyric-human adapters/
docker build -f Dockerfile.runpod -t your-registry/lyric-pipeline:latest .
```

Push to a registry RunPod can pull from (Docker Hub, GHCR, etc.):

```bash
docker push your-registry/lyric-pipeline:latest
```

## Create the serverless endpoint on RunPod

1. RunPod Console → Serverless → Create Endpoint.
2. Choose **GPU type** (e.g. RTX 4090, A100) that fits your model (7B 4-bit ~6–8GB VRAM; 70B 4-bit ~40GB).
3. **Container image**: `your-registry/lyric-pipeline:latest`.
4. **Container disk**: ≥ 20 GB (for model + adapter).
5. **Environment variables** (optional):
   - `HUGGING_FACE_HUB_TOKEN`: if the base model is gated or private.
   - `LYRIC_DEVICE=4xa100`: if using the 4×A100 70B preset.
6. **Max workers**: as needed; **Idle timeout**: e.g. 5–10 minutes.
7. Create the endpoint and note the **Endpoint ID** and **API key**.

## Request format

**Sync request** (wait for result):

```json
POST https://api.runpod.ai/v2/{endpoint_id}/runsync
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "input": {
    "prompt": "Write a pop song about missing someone after a fight, with a short repeatable chorus.",
    "debug": false,
    "config": { "max_rewrite_passes": 1, "pass_threshold": 85 }
  }
}
```

**Input fields**

| Field    | Type   | Required | Description |
|----------|--------|----------|-------------|
| `prompt` | string | Yes      | Lyric request (e.g. "Write a pop song about..."). |
| `debug`  | bool   | No       | If `true`, response includes `score` and full `debug` object. |
| `config` | object | No       | Pipeline overrides: `max_rewrite_passes`, `pass_threshold`, etc. |

**Response (success)**

```json
{
  "id": "...",
  "status": "COMPLETED",
  "output": {
    "lyrics": "Verse 1:\n...\n\nChorus:\n..."
  }
}
```

With `"debug": true`:

```json
{
  "output": {
    "lyrics": "...",
    "score": 78.5,
    "debug": { "final_lyrics": "...", "final_score_breakdown": {...}, "constraint_violations": [], ... }
  }
}
```

**Response (error)**

```json
{
  "output": {
    "error": "Missing 'prompt' in input. Send e.g. {\"prompt\": \"Write a pop song about...\"}",
    "lyrics": ""
  }
}
```

## Local test (no Docker)

Install RunPod SDK and run the handler with test input:

```bash
pip install runpod
python runpod_handler.py --test_input test_input.json
```

Or with inline JSON:

```bash
python runpod_handler.py --test_input '{"input": {"prompt": "Write a verse and chorus about heartbreak.", "debug": false}}'
```

Model will load on first run; the handler will execute once and exit.

## Notes

- **Cold start**: First request after idle loads the model (can take 30–90 s depending on size). Use a short **Idle timeout** if you want to avoid paying for long idle GPUs.
- **Payload limit**: RunPod `/runsync` has a 20 MB response limit. Normal lyrics + debug stay well under that.
- **Adapters**: The image must include `adapters/lyric-human/`. Build the image from a context that contains this directory (or copy it in before `docker build`).
