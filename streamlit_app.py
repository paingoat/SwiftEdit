# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Streamlit UI for SwiftEdit — wraps infer.edit_image without changing core logic."""

import env_config  # noqa: F401 — STORAGE from .env -> HF cache

import io
import os
import re
import tempfile
import time

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from infer import SWIFTEDIT_WEIGHTS_ROOT, edit_image
from models import AuxiliaryModel, InverseModel, IPSBV2Model

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(REPO_ROOT, "result")

APP_CSS = """
<style>
    .swiftedit-header { text-align: center; margin-bottom: 1.5rem; }
    .swiftedit-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
    .swiftedit-authors { font-size: 1rem; color: #444; margin-bottom: 0.25rem; }
    .swiftedit-venue { font-size: 1rem; color: #444; }
    .image-panel-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.35rem;
    }
    div[data-testid="stImage"] {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 4px;
        background: #fafafa;
    }
    .placeholder-box {
        border: 1px dashed #ccc;
        border-radius: 6px;
        padding: 2rem 1rem;
        text-align: center;
        color: #888;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
"""


def get_weights_root() -> str:
    return os.environ.get("SWIFTEDIT_WEIGHTS_ROOT", SWIFTEDIT_WEIGHTS_ROOT)


def validate_runtime(weights_root: str) -> list[str]:
    errors = []
    if not torch.cuda.is_available():
        errors.append(
            "CUDA GPU is not available. SwiftEdit requires a CUDA-capable GPU (~24GB VRAM)."
        )
    inverse_ckpt = os.path.join(weights_root, "inverse_ckpt-120k")
    path_unet_sb = os.path.join(weights_root, "sbv2_0.5")
    ip_ckpt = os.path.join(weights_root, "ip_adapter_ckpt-90k", "ip_adapter.bin")
    for label, path in [
        ("Inversion checkpoint", inverse_ckpt),
        ("SwiftBrush v2 UNet", path_unet_sb),
        ("IP-Adapter weights", ip_ckpt),
    ]:
        if not os.path.exists(path):
            errors.append(f"Missing {label}: `{path}`")
    return errors


@st.cache_resource(show_spinner="Loading SwiftEdit models (first run may take a few minutes)...")
def load_swiftedit_models(weights_root: str):
    inverse_ckpt = os.path.join(weights_root, "inverse_ckpt-120k")
    path_unet_sb = os.path.join(weights_root, "sbv2_0.5")
    ip_ckpt = os.path.join(weights_root, "ip_adapter_ckpt-90k", "ip_adapter.bin")
    inverse_model = InverseModel(inverse_ckpt)
    aux_model = AuxiliaryModel()
    ip_sb_model = IPSBV2Model(
        path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True
    )
    return inverse_model, aux_model, ip_sb_model


def infer_result_basename(src_p: str, edit_p: str) -> str:
    """Same naming pattern as infer.py: result_{src_p}->{edit_p}.png"""
    return f"result_{src_p}->{edit_p}.png"


def save_result_like_infer(result: torch.Tensor, src_p: str, edit_p: str) -> str:
    """Save full batch grid like infer.py save_image(result, ...)."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    filename = infer_result_basename(src_p, edit_p)
    # Sanitize only characters invalid on common filesystems; keep infer naming otherwise.
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", filename)
    out_path = os.path.join(RESULT_DIR, safe_name)
    save_image(result, out_path)
    return out_path


def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name or "image.png")[1] or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="swiftedit_src_")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def run_edit(
    img_path: str,
    src_p: str,
    edit_p: str,
    inverse_model,
    aux_model,
    ip_sb_model,
    scale_edit: float,
    scale_non_edit: float,
    scale_ta: float,
    clamp_rate: float,
    mask_threshold: float,
):
    start = time.time()
    result = edit_image(
        img_path,
        src_p,
        edit_p,
        inverse_model,
        aux_model,
        ip_sb_model,
        scale_ta=scale_ta,
        scale_edit=scale_edit,
        scale_non_edit=scale_non_edit,
        clamp_rate=clamp_rate,
        mask_threshold=mask_threshold,
    )
    elapsed = time.time() - start
    saved_path = save_result_like_infer(result.cpu(), src_p, edit_p)
    edited = to_pil_image(result[1].clamp(0, 1).cpu())
    return edited, elapsed, saved_path


def render_header():
    st.markdown(
        """
        <div class="swiftedit-header">
            <div class="swiftedit-title">
                SwiftEdit: Lightning Fast Text-guided Image Editing via One-step Diffusion
            </div>
            <div class="swiftedit-authors">
                Trong-Tung Nguyen, Quang Nguyen, Khoi Nguyen, Anh Tran, Cuong Pham
            </div>
            <div class="swiftedit-venue">
                Accepted at CVPR 2025, Nashville
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_advanced_settings(default_weights_root: str):
    with st.sidebar:
        st.header("Advanced settings")
        weights_root = st.text_input(
            "Weights directory",
            value=default_weights_root,
            help="Path to swiftedit_weights folder (or set SWIFTEDIT_WEIGHTS_ROOT).",
        )
        scale_non_edit = st.slider(
            "Background preservation",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="scale_non_edit — IP-Adapter strength outside the edit mask.",
        )
        scale_ta = st.slider(
            "Text attention scale",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="scale_ta — cross-attention scaling on the edit branch.",
        )
        clamp_rate = st.slider(
            "Mask clamp rate",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Controls how aggressively the self-guided mask is normalized.",
        )
        mask_threshold = st.slider(
            "Mask threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Binarization threshold for the editing mask.",
        )
    return weights_root, scale_non_edit, scale_ta, clamp_rate, mask_threshold


def main():
    st.set_page_config(
        page_title="SwiftEdit",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    render_header()

    default_weights = get_weights_root()
    weights_root, scale_non_edit, scale_ta, clamp_rate, mask_threshold = (
        render_advanced_settings(default_weights)
    )

    runtime_errors = validate_runtime(weights_root)
    if runtime_errors:
        for msg in runtime_errors:
            st.error(msg)
        st.info(
            "Download checkpoints per README, extract to `swiftedit_weights/`, "
            "then set the weights directory in the sidebar."
        )
        st.stop()

    try:
        inverse_model, aux_model, ip_sb_model = load_swiftedit_models(weights_root)
    except Exception as exc:
        st.error(f"Failed to load models: {exc}")
        st.stop()

    col_src, col_edit = st.columns(2)

    with col_src:
        st.markdown('<p class="image-panel-label">Source Image</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload source image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            source_pil = Image.open(uploaded).convert("RGB")
            st.image(source_pil, use_container_width=True)
        else:
            st.markdown(
                '<div class="placeholder-box">Upload a source image</div>',
                unsafe_allow_html=True,
            )
        src_p = st.text_input(
            "Source Prompt",
            value="",
            placeholder="e.g. german shepherd dog on grass field",
        )

    with col_edit:
        st.markdown('<p class="image-panel-label">Edited Image</p>', unsafe_allow_html=True)
        if "edited_image" in st.session_state:
            st.image(st.session_state.edited_image, use_container_width=True)
            buf = io.BytesIO()
            st.session_state.edited_image.save(buf, format="PNG")
            st.download_button(
                label="Download edited image",
                data=buf.getvalue(),
                file_name="swiftedit_edited.png",
                mime="image/png",
            )
        else:
            st.markdown(
                '<div class="placeholder-box">Run Edit to generate</div>',
                unsafe_allow_html=True,
            )
        edit_p = st.text_area(
            "Edit Prompt",
            value="",
            placeholder="e.g. german shepherd dog with mouth opened on grass field",
            height=80,
        )
        scale_edit = st.number_input(
            "Edit Strength",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.05,
            help="scale_edit — editing strength in the masked region (default 0.2).",
        )

    run_clicked = st.button("Run Edit", type="primary", use_container_width=True)

    if run_clicked:
        if uploaded is None:
            st.warning("Please upload a source image.")
            st.stop()
        if not edit_p.strip():
            st.warning("Please enter an Edit Prompt.")
            st.stop()

        temp_path = None
        try:
            temp_path = save_upload_to_temp(uploaded)
            with st.spinner("Running SwiftEdit..."):
                src_prompt = src_p.strip()
                edit_prompt = edit_p.strip()
                edited_pil, elapsed, saved_path = run_edit(
                    temp_path,
                    src_prompt,
                    edit_prompt,
                    inverse_model,
                    aux_model,
                    ip_sb_model,
                    scale_edit=float(scale_edit),
                    scale_non_edit=float(scale_non_edit),
                    scale_ta=float(scale_ta),
                    clamp_rate=float(clamp_rate),
                    mask_threshold=float(mask_threshold),
                )
            st.session_state.edited_image = edited_pil
            st.session_state.last_saved_path = saved_path
            st.success(
                f"Edit completed in {elapsed:.2f}s. "
                f"Saved to `{saved_path}` (same format as infer.py: source + edit grid)."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Editing failed: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    main()
