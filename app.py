# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Gradio UI for SwiftEdit
#
# Usage:  python app.py
#

import os, re, tempfile

from dotenv import load_dotenv
load_dotenv()

# Set HF cache directory from .env STORAGE variable (if available)
_storage = os.getenv("STORAGE")
if _storage:
    os.environ["HF_HOME"] = _storage

import torch
import gradio as gr
from PIL import Image
from torchvision.utils import save_image

from infer import edit_image, SWIFTEDIT_WEIGHTS_ROOT
from models import InverseModel, AuxiliaryModel, IPSBV2Model

# ── Results directory ──────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load models (once at startup) ─────────────────────────────────
print("Loading SwiftEdit models …")

inverse_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
inverse_model = InverseModel(inverse_ckpt)
aux_model = AuxiliaryModel()

path_unet_sb = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5")
ip_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True)

print("Models loaded ✓")


# ── Helpers ────────────────────────────────────────────────────────
def _sanitize(text: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', text)


# ── Inference callback ─────────────────────────────────────────────
def run_edit(source_image: Image.Image, src_prompt: str, edit_prompt: str, edit_strength: float):
    if source_image is None:
        raise gr.Error("Please upload a source image.")
    if not edit_prompt.strip():
        raise gr.Error("Please enter an edit prompt.")

    # Fallback to 1.0 if the user cleared the number field in the UI
    if edit_strength is None:
        edit_strength = 1.0

    # Save uploaded image to a temp file (edit_image expects a path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        source_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        result_tensor = edit_image(
            img_path=tmp_path,
            src_p=src_prompt,
            edit_p=edit_prompt,
            inverse_model=inverse_model,
            aux_model=aux_model,
            ip_sb_model=ip_sb_model,
            scale_ta=edit_strength,
        )
    finally:
        os.unlink(tmp_path)

    # Save result in a folder named after the source prompt
    safe_src = _sanitize(src_prompt).strip() if src_prompt and src_prompt.strip() else "none"
    safe_edit = _sanitize(edit_prompt).strip()
    target_dir = os.path.join(RESULTS_DIR, safe_src)
    os.makedirs(target_dir, exist_ok=True)

    save_name = f"{safe_edit}_SY_{edit_strength}.png"
    save_path = os.path.join(target_dir, save_name)
    save_image(result_tensor, save_path)
    print(f"Saved to {save_path}")

    # Convert tensor → PIL for Gradio display (index 1 is the edited image)
    result_pil = result_tensor[1].clamp(0, 1).cpu()
    result_pil = result_pil.permute(1, 2, 0).numpy()
    result_pil = Image.fromarray((result_pil * 255).astype("uint8"))

    return result_pil


# ── Gradio UI ──────────────────────────────────────────────────────
with gr.Blocks(title="SwiftEdit", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>⚡ SwiftEdit: Lightning Fast Text-guided<br>Image Editing via One-step Diffusion.</h1>
            <p>Trong-Tung Nguyen, Quang Nguyen, Khoi Nguyen,<br>Anh Tran, Cuong Pham.</p>
            <p>🔥 Accepted at CVPR 2025, Nashville 🔥</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            source_image = gr.Image(label="Source Image", type="pil", height=400)
            src_prompt = gr.Textbox(label="Source Prompt", placeholder="e.g. german shepherd dog on grass field")
        with gr.Column():
            edited_image = gr.Image(label="Edited Image", type="pil", height=400, interactive=False)
            edit_prompt = gr.Textbox(label="Edit Prompt", placeholder="e.g. german shepherd dog with mouth opened on grass field")
            edit_strength = gr.Number(
                label="Edit Strength",
                value=1,
            )

    edit_btn = gr.Button("⚡ Edit", variant="primary", size="lg")

    edit_btn.click(
        fn=run_edit,
        inputs=[source_image, src_prompt, edit_prompt, edit_strength],
        outputs=[edited_image],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
