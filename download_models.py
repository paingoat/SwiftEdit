"""
Pre-download all HuggingFace models required by SwiftEdit so the app
can run fully offline afterwards.

Usage:
    python download_models.py

Configuration is read from .env (copy .env.example -> .env and fill in):

    STORAGE=E:/hf_cache        # where to cache models (optional)
    HF_TOKEN=hf_xxxxxxxxxxxx   # required for gated models

The gated model stabilityai/stable-diffusion-2-1-base requires:
  1. A HuggingFace account
  2. License acceptance at https://huggingface.co/stabilityai/stable-diffusion-2-1
  3. An access token from https://huggingface.co/settings/tokens
"""

import env_config  # noqa: F401 — applies STORAGE -> HF_HOME before any HF import

import os
import sys

# hf_transfer accelerates downloads but is optional; disable if not installed
# to avoid a crash when HF_HUB_ENABLE_HF_TRANSFER=1 is set in the environment.
try:
    import hf_transfer  # noqa: F401
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

# Only the subfolders actually loaded by InverseModel / AuxiliaryModel
MODELS = [
    (
        "stabilityai/sd-turbo",
        # InverseModel: DDPMScheduler, AutoencoderKL, AutoTokenizer, CLIPTextModel
        ["scheduler/**", "tokenizer/**", "text_encoder/**", "vae/**", "*.json", "*.txt"],
        "SD-Turbo  (used by InverseModel)",
    ),
    (
        "stabilityai/stable-diffusion-2-1-base",
        # AuxiliaryModel: DDPMScheduler, AutoencoderKL, AutoTokenizer, CLIPTextModel
        ["scheduler/**", "tokenizer/**", "text_encoder/**", "vae/**", "*.json", "*.txt"],
        "SD-2.1-base  (used by AuxiliaryModel)",
    ),
    (
        "h94/IP-Adapter",
        # AuxiliaryModel: CLIPVisionModelWithProjection
        ["models/image_encoder/**"],
        "IP-Adapter image encoder  (used by AuxiliaryModel)",
    ),
]


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Tip: set HF_TOKEN=hf_xxxx for gated models "
            "(stabilityai/stable-diffusion-2-1-base requires login + license acceptance)."
        )

    print("SwiftEdit — pre-downloading HuggingFace model weights\n")
    ok = True
    for repo_id, allow_patterns, desc in MODELS:
        print(f"{'─'*60}")
        print(f"  {desc}")
        print(f"  {repo_id}")
        try:
            path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                token=token or None,
            )
            print(f"  cached -> {path}")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            ok = False

    print(f"{'─'*60}")
    if ok:
        print("\nAll models cached. Run the app with:")
        print("  streamlit run streamlit_app.py")
    else:
        print("\nSome downloads failed — check your connection / VPN and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
