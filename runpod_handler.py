"""
RunPod serverless worker for the lyric pipeline.
Loads model once at startup; each job runs the full pipeline (draft → eval → rewrite → score) and returns lyrics or debug payload.

Input (job["input"]):
  - prompt (str, required): Lyric request (e.g. "Write a pop song about a breakup...")
  - debug (bool, optional): If true, return full debug object (score, breakdown, constraint_violations, etc.)
  - config (dict, optional): Override pipeline config (max_rewrite_passes, pass_threshold, etc.)

Output:
  - lyrics (str): Final lyrics (always present)
  - score (float): Total score when debug was true
  - debug (dict): Full debug payload when debug was true
  - error (str): Present only on failure
"""
from __future__ import annotations

import os
import sys

# If this module is imported before handler.py (e.g. tests), keep Xet off by default.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path

# RunPod worker runs from /app; ensure Lyric_model is on path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

def _ensure_adapter_from_hub() -> None:
    """
    If LYRIC_ADAPTER_REPO (or HF_ADAPTER_REPO) is set, download the LoRA adapter from Hugging Face
    into adapters/lyric-human. Use this when the adapter is too large for GitHub/Docker context.

    Upload your adapter folder to a HF model repo (e.g. Private), then set:
      LYRIC_ADAPTER_REPO=yourusername/your-adapter-repo
      HUGGING_FACE_HUB_TOKEN=hf_...   (read access for private repos)
    """
    repo_id = (os.environ.get("LYRIC_ADAPTER_REPO") or os.environ.get("HF_ADAPTER_REPO") or "").strip()
    if not repo_id:
        return
    from config import ADAPTERS_DIR, ADAPTER_NAME

    dest = ADAPTERS_DIR / ADAPTER_NAME
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "adapter_config.json").exists():
        return
    from huggingface_hub import snapshot_download

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(repo_id=repo_id, local_dir=str(dest), token=token)


# Load model once at worker startup (before runpod.serverless.start)
def _load_model():
    _ensure_adapter_from_hub()
    from generate import load_model_and_tokenizer
    load_model_and_tokenizer()


def handler(job):
    job_id = job.get("id", "?")
    inp = job.get("input") or {}
    prompt = inp.get("prompt") or inp.get("message")
    if not prompt or not str(prompt).strip():
        return {"error": "Missing 'prompt' in input. Send e.g. {\"prompt\": \"Write a pop song about...\"}"}

    debug = inp.get("debug", False)
    config = inp.get("config") or {}

    try:
        from pipeline import run_pipeline
        result = run_pipeline(
            prompt.strip(),
            debug=debug,
            config=config,
        )
        if debug and isinstance(result, dict):
            return {
                "lyrics": result.get("final_lyrics", ""),
                "score": result.get("score"),
                "debug": result,
            }
        return {"lyrics": result if isinstance(result, str) else (result.get("final_lyrics", "") if isinstance(result, dict) else "")}
    except Exception as e:
        return {"error": str(e), "lyrics": ""}


if __name__ == "__main__":
    # Local / tests: same as handler.py entrypoint
    import runpod

    _load_model()
    runpod.serverless.start({"handler": handler})
