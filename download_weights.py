# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Download all required weights for SwiftEdit.
# Run this script BEFORE launching app.py or infer.py.
#
# Usage:
#   1. Copy .env.example to .env and fill in your HF_TOKEN
#   2. python download_weights.py
#

import os
import tarfile

from dotenv import load_dotenv

load_dotenv()

# ── Environment setup ──────────────────────────────────────────────
_storage = os.getenv("STORAGE", "./hf_cache")
os.environ["HF_HOME"] = _storage
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

HF_TOKEN = os.getenv("HF_TOKEN", None)

from huggingface_hub import hf_hub_download, snapshot_download

SWIFTEDIT_WEIGHTS_DIR = "swiftedit_weights"


def download_swiftedit_weights():
    """Download and extract SwiftEdit pretrained weights from HF Hub."""
    tar_filename = "swiftedit_weights.tar.gz"

    if os.path.isdir(SWIFTEDIT_WEIGHTS_DIR):
        print(f"[✓] '{SWIFTEDIT_WEIGHTS_DIR}/' already exists, skipping SwiftEdit weights download.")
        return

    print(f"[↓] Downloading {tar_filename} from paingoat/swiftedit-pretrain ...")
    local_path = hf_hub_download(
        repo_id="paingoat/swiftedit-pretrain",
        filename=tar_filename,
        token=HF_TOKEN,
        local_dir=".",
    )
    print(f"[✓] Downloaded to {local_path}")

    print(f"[…] Extracting {tar_filename} ...")
    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall(".")
    print(f"[✓] Extracted to '{SWIFTEDIT_WEIGHTS_DIR}/'")

    # Clean up the tar.gz after extraction
    if os.path.isfile(tar_filename):
        os.remove(tar_filename)
        print(f"[✓] Removed {tar_filename}")


def precache_base_models():
    """Pre-download HuggingFace base models used by models.py so they are cached in STORAGE."""
    models = [
        {
            "repo_id": "stabilityai/sd-turbo",
            "desc": "SD-Turbo (used by InverseModel)",
        },
        {
            "repo_id": "Manojb/stable-diffusion-2-1-base",
            "desc": "SD 2.1 Base (used by AuxiliaryModel)",
        },
        {
            "repo_id": "h94/IP-Adapter",
            "desc": "IP-Adapter image encoder (used by AuxiliaryModel)",
        },
    ]

    for m in models:
        print(f"\n[↓] Pre-caching {m['desc']}  ({m['repo_id']}) ...")
        try:
            snapshot_download(
                repo_id=m["repo_id"],
                token=HF_TOKEN,
            )
            print(f"[✓] Cached {m['repo_id']}")
        except Exception as e:
            print(f"[✗] Failed to cache {m['repo_id']}: {e}")
            raise


if __name__ == "__main__":
    print("=" * 60)
    print("  SwiftEdit — Weight Downloader")
    print(f"  HF_HOME  = {os.environ['HF_HOME']}")
    print(f"  HF_TOKEN = {'***' + HF_TOKEN[-4:] if HF_TOKEN else '(not set)'}")
    print("=" * 60)

    download_swiftedit_weights()
    precache_base_models()

    print("\n" + "=" * 60)
    print("  All weights downloaded successfully!")
    print("  You can now run:  python app.py")
    print("=" * 60)
