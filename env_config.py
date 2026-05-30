# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Load .env and map STORAGE to Hugging Face cache directories."""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(REPO_ROOT, ".env")


def _load_dotenv(path: str) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def configure_hf_cache_from_env():
    """
    Read STORAGE from .env / environment and set HF_HOME + HUGGINGFACE_HUB_CACHE.
    Returns the resolved storage path, or None if STORAGE is unset.
    """
    _load_dotenv(ENV_FILE)
    storage = os.environ.get("STORAGE", "").strip()
    if not storage:
        return None
    storage = os.path.abspath(os.path.expanduser(storage))
    os.makedirs(storage, exist_ok=True)
    hub_cache = os.path.join(storage, "hub")
    os.makedirs(hub_cache, exist_ok=True)
    os.environ["HF_HOME"] = storage
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    return storage


# Apply on import, before huggingface_hub / diffusers download anything.
configure_hf_cache_from_env()
