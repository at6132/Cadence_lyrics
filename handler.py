"""
RunPod serverless entrypoint (Queue workers scan the default branch for this file).
Logic lives in runpod_handler.py.
"""
import os

# Before huggingface_hub is imported (via snapshot_download / transformers). Xet backend
# can raise "Background writer channel closed" on some runners; classic HTTP is slower but stable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import runpod

from runpod_handler import _load_model, handler

if __name__ == "__main__":
    _load_model()
    runpod.serverless.start({"handler": handler})
