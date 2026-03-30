"""
RunPod serverless entrypoint (Queue workers scan the default branch for this file).
Logic lives in runpod_handler.py.
"""
import runpod

from runpod_handler import _load_model, handler

if __name__ == "__main__":
    _load_model()
    runpod.serverless.start({"handler": handler})
