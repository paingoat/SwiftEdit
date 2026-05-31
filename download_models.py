"""
Pre-download all HuggingFace models required by SwiftEdit so the app
can run fully offline afterwards.

Usage:
    python download_models.py

Set STORAGE in .env (or the environment) to redirect the HF cache, e.g.:
    STORAGE=E:/hf_cache
"""

import env_config  # noqa: F401 — applies STORAGE -> HF_HOME before any HF import

import sys
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
