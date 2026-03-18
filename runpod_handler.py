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
from pathlib import Path

# RunPod worker runs from /app; ensure Lyric_model is on path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Load model once at worker startup (before runpod.serverless.start)
def _load_model():
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
    import runpod
    # Load model before starting the server so first request is fast
    _load_model()
    runpod.serverless.start({"handler": handler})
