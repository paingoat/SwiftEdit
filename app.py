# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# app.py — Gradio UI for SwiftEdit
#
# Two editing modes:
#   ⚡ Auto Mask     — Original SwiftEdit behaviour (noise diff → binary mask)
#   ✏️ Semi-Auto     — User brush + Δε intersection + Distance Transform soft mask
#
# Usage:  python app.py
#

import os
import re
import tempfile

from dotenv import load_dotenv
load_dotenv()

_storage = os.getenv("STORAGE")
if _storage:
    os.environ["HF_HOME"] = _storage

import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision.utils import save_image
import torch.nn.functional as F

from infer import edit_image, edit_image_with_mask, edit_image_user_mask, SWIFTEDIT_WEIGHTS_ROOT
from models import InverseModel, AuxiliaryModel, IPSBV2Model
from src.mask_refine import (
    extract_noise_map,
    extract_user_brush,
    refine_mask,
    distance_transform_mask,
    prepare_mask_tensor,
    make_heatmap_overlay,
    make_soft_mask_visual,
    make_red_overlay,
    apply_poisson_blend,
)

# ── Results directory ─────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading SwiftEdit models …")

inverse_ckpt  = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
inverse_model = InverseModel(inverse_ckpt)
aux_model     = AuxiliaryModel()

path_unet_sb = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5")
ip_ckpt      = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
ip_sb_model  = IPSBV2Model(path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True)

print("Models loaded ✓")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _sanitize(text: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", text)


def _mask_to_overlay(source_pil: Image.Image, mask_tensor) -> Image.Image:
    """Original binary mask overlay (used in Auto tab)."""
    img      = source_pil.resize((512, 512)).convert("RGBA")
    mask_4d  = mask_tensor.unsqueeze(0).unsqueeze(0).float()
    mask_512 = F.interpolate(mask_4d, size=(512, 512), mode="nearest").squeeze().numpy()
    overlay  = np.zeros((512, 512, 4), dtype=np.uint8)
    fg       = mask_512 > 0.5
    overlay[fg]  = [255, 80, 80, 120]
    overlay[~fg] = [30,  30, 30,  80]
    overlay_pil  = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(img, overlay_pil).convert("RGB")


def _save_result(result_tensor, src_prompt: str, edit_prompt: str, strength: float) -> str:
    safe_src  = _sanitize(src_prompt).strip()  if src_prompt  and src_prompt.strip()  else "none"
    safe_edit = _sanitize(edit_prompt).strip() if edit_prompt and edit_prompt.strip() else "edit"
    target_dir = os.path.join(RESULTS_DIR, safe_src)
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, f"{safe_edit}_SY_{strength}.png")
    save_image(result_tensor, save_path)
    return save_path


def _tensor_to_pil(result_tensor) -> Image.Image:
    arr = result_tensor[1].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype("uint8"))


# ═════════════════════════════════════════════════════════════════════════════
# Callback — Auto Mask (original behaviour)
# ═════════════════════════════════════════════════════════════════════════════

def run_auto_edit(source_image: Image.Image, src_prompt: str, edit_prompt: str, edit_strength: float):
    if source_image is None:
        raise gr.Error("Please upload a source image.")
    if not edit_prompt.strip():
        raise gr.Error("Please enter an edit prompt.")
    edit_strength = edit_strength or 1.0

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        source_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        result_tensor, mask_tensor = edit_image(
            img_path=tmp_path, src_p=src_prompt, edit_p=edit_prompt,
            inverse_model=inverse_model, aux_model=aux_model, ip_sb_model=ip_sb_model,
            scale_ta=edit_strength,
        )
    finally:
        os.unlink(tmp_path)

    save_path = _save_result(result_tensor, src_prompt, edit_prompt, edit_strength)
    print(f"Saved → {save_path}")

    result_pil       = _tensor_to_pil(result_tensor)
    mask_overlay_pil = _mask_to_overlay(source_image, mask_tensor)
    return result_pil, mask_overlay_pil


# ═════════════════════════════════════════════════════════════════════════════
# Callback — Preview Mask only (Semi-Auto, no generation)
# ═════════════════════════════════════════════════════════════════════════════

def run_preview_mask(
    editor_output,
    src_prompt: str,
    edit_prompt: str,
    threshold: float,
    use_otsu: bool,
    dilate_kernel: int,
):
    """
    Run the full mask pipeline (Δε + brush intersect + DT) WITHOUT running
    the diffusion generation step. Returns three debug images in < 1 second.
    """
    if editor_output is None or editor_output.get("background") is None:
        raise gr.Error("Please upload an image first.")

    background = editor_output.get("background")
    if isinstance(background, np.ndarray):
        source_pil = Image.fromarray(background)
    else:
        source_pil = background
    source_pil = source_pil.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        source_pil.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        # Encode image
        from torchvision.transforms.functional import to_tensor as tv_to_tensor
        pil_512 = source_pil.resize((512, 512))
        processed = tv_to_tensor(pil_512).unsqueeze(0).to("cuda") * 2 - 1
        latents = inverse_model.vae.encode(
            processed.to(inverse_model.weight_dtype)
        ).latent_dist.sample()
        latents = latents * inverse_model.vae.config.scaling_factor

        # Compute Δε soft map
        soft_map_64, _ = extract_noise_map(
            inverse_model, latents,
            src_prompt or "", edit_prompt or "",
            clamp_rate=3.0, t=500,
        )
        noise_map_512 = F.interpolate(
            soft_map_64.unsqueeze(0).unsqueeze(0),
            size=(512, 512), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # Extract brush ROI
        roi_mask = extract_user_brush(editor_output, dilate_kernel=dilate_kernel)

        if roi_mask is not None:
            refined  = refine_mask(noise_map_512, roi_mask,
                                   threshold=threshold, use_otsu=use_otsu)
            binary   = refined["binary_mask"]
        else:
            binary = ((noise_map_512 > threshold) * 255).astype(np.uint8)

        # Distance Transform → soft mask
        soft_mask = distance_transform_mask(binary, alpha=0.6, gamma=0.7)

        img_np = np.array(pil_512)
        heatmap_pil  = Image.fromarray(make_heatmap_overlay(img_np, noise_map_512))
        softmask_pil = Image.fromarray(make_soft_mask_visual(soft_mask))
        overlay_pil  = Image.fromarray(make_red_overlay(img_np, soft_mask))

        report = (
            f"Mask coverage: {(soft_mask > 0.5).mean() * 100:.1f}% of image\n"
            f"Noise map peak: {noise_map_512.max():.3f}  mean: {noise_map_512.mean():.3f}\n"
            f"Mode: {'Semi-Auto (brush + Δε)' if roi_mask is not None else 'Auto (Δε only)'}\n"
            f"Threshold: {threshold:.2f}  Otsu: {use_otsu}  Dilate: {dilate_kernel}px\n"
            "✅ Preview ready — tune parameters then click ⚡ Edit."
        )
    finally:
        os.unlink(tmp_path)

    return heatmap_pil, softmask_pil, overlay_pil, report


# ═════════════════════════════════════════════════════════════════════════════
# Callback — Semi-Auto Edit (with mask + generation)
# ═════════════════════════════════════════════════════════════════════════════

def run_semi_edit(
    editor_output,
    src_prompt: str,
    edit_prompt: str,
    edit_strength: float,
    threshold: float,
    use_otsu: bool,
    dilate_kernel: int,
    use_poisson: bool,
):
    if editor_output is None or editor_output.get("background") is None:
        raise gr.Error("Please upload an image first.")
    if not edit_prompt.strip():
        raise gr.Error("Please enter an edit prompt.")
    edit_strength = edit_strength or 1.0

    background = editor_output.get("background")
    source_pil = Image.fromarray(background) if isinstance(background, np.ndarray) else background
    source_pil = source_pil.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        source_pil.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        result_tensor, soft_mask_cpu, debug = edit_image_with_mask(
            img_path=tmp_path,
            src_p=src_prompt,
            edit_p=edit_prompt,
            inverse_model=inverse_model,
            aux_model=aux_model,
            ip_sb_model=ip_sb_model,
            editor_output=editor_output,
            threshold=threshold,
            use_otsu=use_otsu,
            dilate_kernel=dilate_kernel,
            scale_ta=edit_strength,
            use_poisson=use_poisson,
        )
    finally:
        os.unlink(tmp_path)

    save_path = _save_result(result_tensor, src_prompt, edit_prompt, edit_strength)
    print(f"Saved → {save_path}")

    # Use Poisson-composited result if available, else fall back to raw tensor
    if use_poisson and "poisson" in debug:
        result_pil = debug["poisson"]
    else:
        result_pil = _tensor_to_pil(result_tensor)

    report = (
        f"Mask coverage: {(soft_mask_cpu.numpy() > 0.5).mean() * 100:.1f}% of image\n"
        f"Saved to: {save_path}\n"
        f"Mode: {'Semi-Auto (brush + Δε + DT)' if editor_output else 'Auto (Δε + DT)'}\n"
        f"Poisson blend: {'ON' if use_poisson else 'OFF'}\n"
        "✅ Edit complete."
    )

    return result_pil, debug["heatmap"], debug["soft_mask"], debug["overlay"], report


# ═══════════════════════════════════════════════════════════════════════════════
# Callback — User Mask Preview (no generation)
# ═══════════════════════════════════════════════════════════════════════════════

def run_user_mask_preview(
    editor_output,
    dilate_kernel: int,
    dt_alpha: float,
    dt_gamma: float,
):
    """
    Preview the Distance-Transform mask from user brush strokes.
    Fast (≈ 0 s) — no generation, just mask visualisation.
    """
    if editor_output is None or editor_output.get("background") is None:
        raise gr.Error("Please upload an image first.")

    background = editor_output.get("background")
    source_pil = Image.fromarray(background) if isinstance(background, np.ndarray) else background
    source_pil = source_pil.convert("RGB")
    pil_512 = source_pil.resize((512, 512))

    roi_mask = extract_user_brush(editor_output, dilate_kernel=dilate_kernel)
    if roi_mask is None:
        raise gr.Error("No brush strokes detected. Please paint the region first.")

    soft_mask = distance_transform_mask(roi_mask, alpha=dt_alpha, gamma=dt_gamma)
    img_np = np.array(pil_512)
    softmask_pil = Image.fromarray(make_soft_mask_visual(soft_mask))
    overlay_pil  = Image.fromarray(make_red_overlay(img_np, soft_mask))

    report = (
        f"Mask coverage: {(soft_mask > 0.5).mean() * 100:.1f}% of image\n"
        f"Dilate kernel: {dilate_kernel}px  \u2502  DT alpha: {dt_alpha}  \u2502  gamma: {dt_gamma}\n"
        "✅ Preview ready — adjust parameters then click ⚡ Edit."
    )
    return softmask_pil, overlay_pil, report


# ═══════════════════════════════════════════════════════════════════════════════
# Callback — User Mask Edit (with generation + Poisson)
# ═══════════════════════════════════════════════════════════════════════════════

def run_user_mask_edit(
    editor_output,
    src_prompt: str,
    edit_prompt: str,
    edit_strength: float,
    dilate_kernel: int,
    dt_alpha: float,
    dt_gamma: float,
):
    if editor_output is None or editor_output.get("background") is None:
        raise gr.Error("Please upload an image first.")
    if not edit_prompt.strip():
        raise gr.Error("Please enter an edit prompt.")
    edit_strength = edit_strength or 1.0

    background = editor_output.get("background")
    source_pil = Image.fromarray(background) if isinstance(background, np.ndarray) else background
    source_pil = source_pil.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        source_pil.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        result_tensor, soft_mask_cpu, debug = edit_image_user_mask(
            img_path=tmp_path,
            src_p=src_prompt,
            edit_p=edit_prompt,
            inverse_model=inverse_model,
            aux_model=aux_model,
            ip_sb_model=ip_sb_model,
            editor_output=editor_output,
            dilate_kernel=dilate_kernel,
            dt_alpha=dt_alpha,
            dt_gamma=dt_gamma,
            scale_ta=edit_strength,
        )
    except ValueError as e:
        raise gr.Error(str(e))
    finally:
        os.unlink(tmp_path)

    save_path = _save_result(result_tensor, src_prompt, edit_prompt, edit_strength)
    print(f"Saved → {save_path}")

    result_pil = debug["poisson"]   # Poisson-composited result is the primary output

    report = (
        f"Mask coverage: {(soft_mask_cpu.numpy() > 0.5).mean() * 100:.1f}% of image\n"
        f"Saved to: {save_path}\n"
        "Mode: User Mask (brush → DT → ARAM → Poisson blend)\n"
        "✅ Edit complete."
    )
    return result_pil, debug["soft_mask"], debug["overlay"], report


# ═════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 1400px !important; }
.tab-nav button { font-weight: 600; font-size: 15px; }
"""

with gr.Blocks(
    title="SwiftEdit",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
    css=CSS,
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding:16px 0 8px">
        <h1 style="font-size:2em; margin:0">⚡ SwiftEdit</h1>
        <p style="color:#888; margin:4px 0 0">
            Lightning-Fast Text-guided Image Editing via One-step Diffusion
            &nbsp;·&nbsp; CVPR 2025
        </p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Auto Mask ─────────────────────────────────────────────
        with gr.Tab("⚡ Auto Mask"):
            gr.Markdown(
                "Upload an image, enter prompts, and SwiftEdit will automatically "
                "estimate the edit region from the noise difference map."
            )
            with gr.Row():
                with gr.Column():
                    auto_source = gr.Image(
                        label="Source Image", type="pil", height=400, elem_id="auto_source"
                    )
                    auto_src_prompt  = gr.Textbox(
                        label="Source Prompt",
                        placeholder="e.g. german shepherd dog on grass field",
                    )
                with gr.Column():
                    auto_result = gr.Image(
                        label="Edited Image", type="pil", height=400, interactive=False,
                    )
                    auto_edit_prompt = gr.Textbox(
                        label="Edit Prompt",
                        placeholder="e.g. golden retriever on grass field",
                    )
                    auto_strength = gr.Number(label="Edit Strength", value=1.0)
                with gr.Column():
                    auto_mask_img = gr.Image(
                        label="Predicted Edit Mask 🔴", type="pil", height=400, interactive=False,
                    )
                    gr.Markdown(
                        "🔴 **Red** = edit region · 🔲 **Dark** = background preserved"
                    )

            auto_btn = gr.Button("⚡ Edit", variant="primary", size="lg")
            auto_btn.click(
                fn=run_auto_edit,
                inputs=[auto_source, auto_src_prompt, auto_edit_prompt, auto_strength],
                outputs=[auto_result, auto_mask_img],
            )

        # ── Tab 2: Semi-Auto Mask ────────────────────────────────────────
        with gr.Tab("✏️ Semi-Auto Mask"):
            gr.Markdown(
                "**Draw a rough brush stroke** over the region you want to edit. "
                "SwiftEdit's Δε noise map is refined by your ROI, then smoothed with "
                "Distance Transform to produce a precise gradient mask."
            )

            with gr.Row(equal_height=False):

                # Left: Inputs
                with gr.Column(scale=5):
                    gr.Markdown("### 📥 Input")
                    sa_editor = gr.ImageEditor(
                        label="Upload image & paint edit region (rough strokes are fine!)",
                        height=450, type="pil",
                        brush=gr.Brush(
                            colors=["#ff0000", "#ffffff", "#00ff00"],
                            color_mode="fixed", default_size=20,
                        ),
                        eraser=gr.Eraser(default_size=20),
                        elem_id="sa_editor",
                    )
                    with gr.Row():
                        sa_src_prompt  = gr.Textbox(
                            label="Source Prompt",
                            placeholder="e.g. a cat sitting on a sofa",
                            lines=2,
                        )
                        sa_edit_prompt = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="e.g. a dog sitting on a sofa",
                            lines=2,
                        )

                    gr.Markdown("### ⚙️ Mask Parameters")
                    with gr.Row():
                        sa_threshold = gr.Slider(
                            label="Threshold", minimum=0.10, maximum=0.90, step=0.05, value=0.35,
                            info="Low → bigger mask · High → tighter mask",
                        )
                        sa_dilate = gr.Slider(
                            label="Brush Dilate Kernel (px)", minimum=5, maximum=60, step=5, value=20,
                            info="Expand brush strokes to cover the full object",
                        )
                    with gr.Row():
                        sa_otsu = gr.Checkbox(
                            label="Auto Threshold (Otsu)", value=False,
                            info="Let OpenCV find the optimal threshold automatically",
                        )
                        sa_strength = gr.Number(label="Edit Strength", value=1.0)
                    with gr.Row():
                        sa_poisson = gr.Checkbox(
                            label="🎨 Poisson Blend (fix bleeding)", value=True,
                            info="Seamless-clone paste — eliminates ARAM attention bleeding at pixel level",
                        )

                    with gr.Row():
                        preview_btn = gr.Button("🔍 Preview Mask", variant="secondary", size="lg")
                        edit_btn    = gr.Button("⚡ Edit",          variant="primary",   size="lg")

                # Right: Outputs
                with gr.Column(scale=5):
                    gr.Markdown("### 📤 Outputs")
                    sa_result = gr.Image(
                        label="Edited Image", type="pil", height=350, interactive=False,
                    )
                    with gr.Tabs():
                        with gr.Tab("🌡️ Noise Heatmap"):
                            sa_heatmap = gr.Image(
                                label="Δε Noise Map (JET + source image)",
                                type="pil", height=300,
                            )
                            gr.Markdown(
                                "🔴 **Red** = model detects semantic change here · "
                                "🔵 **Blue** = stable background"
                            )
                        with gr.Tab("⬛ Soft Mask (DT)"):
                            sa_softmask = gr.Image(
                                label="Final Soft Mask after Distance Transform",
                                type="pil", height=300,
                            )
                            gr.Markdown(
                                "**White** = edit region center · "
                                "**Gray gradient** = smooth transition · "
                                "**Black** = background"
                            )
                        with gr.Tab("🔴 Overlay Preview"):
                            sa_overlay = gr.Image(
                                label="Mask overlaid on source — verify region accuracy",
                                type="pil", height=300,
                            )
                    sa_report = gr.Textbox(
                        label="Pipeline Report", lines=6, interactive=False,
                        placeholder="Run Preview or Edit to see results…",
                    )

            # Wiring: Preview (no generation)
            preview_btn.click(
                fn=run_preview_mask,
                inputs=[sa_editor, sa_src_prompt, sa_edit_prompt,
                        sa_threshold, sa_otsu, sa_dilate],
                outputs=[sa_heatmap, sa_softmask, sa_overlay, sa_report],
            )

            # Wiring: Full edit
            edit_btn.click(
                fn=run_semi_edit,
                inputs=[sa_editor, sa_src_prompt, sa_edit_prompt,
                        sa_strength, sa_threshold, sa_otsu, sa_dilate, sa_poisson],
                outputs=[sa_result, sa_heatmap, sa_softmask, sa_overlay, sa_report],
            )

            with gr.Accordion("💡 Tips & Tricks", open=False):
                gr.Markdown("""
| Problem | Solution |
|:---|:---|
| Mask too large / bleeds into background | Increase Threshold or decrease Dilate Kernel |
| Mask too small / misses part of object | Decrease Threshold or increase Dilate Kernel |
| Heatmap diffuse (no clear red region) | Use more distinct prompts (cat→dog vs cat→kitten) |
| Mask doesn't follow object edges | Try Timestep t=400 (lower = sharper edges) |
| Edited region has hard seam artifact | Already solved by Distance Transform soft mask |
| Bleeding still visible | Enable 🎨 Poisson Blend checkbox |
                """)

        # ── Tab 3: User Mask ──────────────────────────────────────────
        with gr.Tab("🖌️ User Mask"):
            gr.Markdown(
                "**Paint precisely** over the region you want to edit. "
                "Your brush stroke is used **directly** as the mask — no noise inference. "
                "Poisson blending is applied automatically to eliminate any bleeding at the boundary."
            )

            with gr.Row(equal_height=False):

                # Left: Inputs
                with gr.Column(scale=5):
                    gr.Markdown("### 📥 Input")
                    um_editor = gr.ImageEditor(
                        label="Upload image & paint EXACTLY the region to edit",
                        height=450, type="pil",
                        brush=gr.Brush(
                            colors=["#ff0000", "#ffffff", "#00ff00"],
                            color_mode="fixed", default_size=20,
                        ),
                        eraser=gr.Eraser(default_size=20),
                        elem_id="um_editor",
                    )
                    with gr.Row():
                        um_src_prompt = gr.Textbox(
                            label="Source Prompt",
                            placeholder="e.g. a cat sitting on a sofa",
                            lines=2,
                        )
                        um_edit_prompt = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="e.g. a dog sitting on a sofa",
                            lines=2,
                        )

                    gr.Markdown("### ⚙️ Mask Parameters")
                    with gr.Row():
                        um_dilate = gr.Slider(
                            label="Brush Dilate Kernel (px)",
                            minimum=0, maximum=40, step=5, value=10,
                            info="Smaller than Semi-Auto — your brush is already precise",
                        )
                        um_strength = gr.Number(label="Edit Strength", value=1.0)
                    with gr.Row():
                        um_dt_alpha = gr.Slider(
                            label="DT Alpha", minimum=0.1, maximum=1.0, step=0.1, value=0.6,
                            info="Blend weight for distance gradient (higher = sharper center)",
                        )
                        um_dt_gamma = gr.Slider(
                            label="DT Gamma", minimum=0.3, maximum=1.5, step=0.1, value=0.7,
                            info="< 1 = wider edit zone · > 1 = tighter edit zone",
                        )

                    with gr.Row():
                        um_preview_btn = gr.Button("🔍 Preview Mask", variant="secondary", size="lg")
                        um_edit_btn    = gr.Button("⚡ Edit",           variant="primary",   size="lg")

                # Right: Outputs
                with gr.Column(scale=5):
                    gr.Markdown("### 📤 Outputs")
                    um_result = gr.Image(
                        label="Edited Image (Poisson Composited)",
                        type="pil", height=350, interactive=False,
                    )
                    with gr.Tabs():
                        with gr.Tab("⬛ Soft Mask (DT)"):
                            um_softmask = gr.Image(
                                label="Mask after Distance Transform",
                                type="pil", height=300,
                            )
                            gr.Markdown(
                                "**White** = edit center · "
                                "**Gray gradient** = transition · "
                                "**Black** = background"
                            )
                        with gr.Tab("🔴 Overlay Preview"):
                            um_overlay = gr.Image(
                                label="Mask overlaid on source — verify region accuracy",
                                type="pil", height=300,
                            )
                    um_report = gr.Textbox(
                        label="Pipeline Report", lines=5, interactive=False,
                        placeholder="Run Preview or Edit to see results…",
                    )

            # Wiring: Preview (no generation)
            um_preview_btn.click(
                fn=run_user_mask_preview,
                inputs=[um_editor, um_dilate, um_dt_alpha, um_dt_gamma],
                outputs=[um_softmask, um_overlay, um_report],
            )

            # Wiring: Full edit
            um_edit_btn.click(
                fn=run_user_mask_edit,
                inputs=[um_editor, um_src_prompt, um_edit_prompt,
                        um_strength, um_dilate, um_dt_alpha, um_dt_gamma],
                outputs=[um_result, um_softmask, um_overlay, um_report],
            )

            with gr.Accordion("💡 Tips & Tricks", open=False):
                gr.Markdown("""
| Problem | Solution |
|:---|:---|
| Bleeding at mask edge | Already handled by Poisson blending. If still visible, reduce Dilate Kernel |
| Edit misses fine detail (hair, fur) | Paint generously then use Eraser to trim brush edges |
| Mask gradient too wide | Increase DT Gamma (> 1) or increase DT Alpha |
| Mask gradient too narrow / hard edge | Decrease DT Gamma (< 0.7) or decrease DT Alpha |
| Want to compare raw vs Poisson result | Check the 🎨 Poisson toggle in Semi-Auto tab |
                """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

