# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os, time, re

from dotenv import load_dotenv
load_dotenv()

# Set HF cache directory from .env STORAGE variable (if available)
_storage = os.getenv("STORAGE")
if _storage:
    os.environ["HF_HOME"] = _storage

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from models import *
from src.mask_refine import (
    extract_noise_map,
    extract_user_brush,
    refine_mask,
    distance_transform_mask,
    prepare_mask_tensor,
)

#
# Configure this path to where you have stored the local copy of the weights:
#
SWIFTEDIT_WEIGHTS_ROOT = 'swiftedit_weights'

def to_binary(pix, threshold=0.5):
    if float(pix) > threshold:
        return 1.0
    else:
        return 0.0


@torch.no_grad()
def edit_image(
    img_path,
    src_p,
    edit_p,
    inverse_model,
    aux_model,
    ip_sb_model,
    scale_ta=1,
    scale_edit=0.2,
    scale_non_edit=1,
    clamp_rate=3.0,
    mask_threshold=0.5,
):
    """
        Edit a source image guided by text prompts.
            + img_path: path to the source image.
            + src_p: Source Prompt that describes source image (could leave it empty).
            + edit_p: Edit Prompt that describes your desired changes.
        Returns:
            (res_gen_img, mask12) — edited-image tensor and the binary editing mask (64×64, on CPU).
    """
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 500
    final_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 999

    # Input Image
    pil_img_cond = Image.open(img_path).resize((512, 512))

    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1

    # Predict inverted noise
    latents = inverse_model.vae.encode(
        processed_image.to(inverse_model.weight_dtype)
    ).latent_dist.sample()
    latents = latents * inverse_model.vae.config.scaling_factor
    dub_latents = torch.cat([latents] * 2, dim=0)

    input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to("cuda")
    encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
        dtype=inverse_model.weight_dtype
    )

    predict_inverted_code = inverse_model.unet_inverse(
        dub_latents, mid_timestep, encoder_hidden_state
    ).sample.to("cuda", dtype=inverse_model.weight_dtype)

    # Estimate editing mask
    inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
    subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
    max_v = (subed.mean() * clamp_rate).item()
    mask12 = subed.clamp(0, max_v) / max_v
    mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to("cuda")

    # Edit images
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, scale_text_hiddenstate=scale_ta, scale_ip_fg=scale_edit, scale_ip_bg=scale_non_edit
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    res_gen_img, _ = ip_sb_model.gen_img(
        pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
    )

    return res_gen_img, mask12.cpu()


@torch.no_grad()
def edit_image_with_mask(
    img_path,
    src_p,
    edit_p,
    inverse_model,
    aux_model,
    ip_sb_model,
    # ── Semi-Auto Mask params ─────────────────────────────────────────────
    editor_output=None,        # dict from gr.ImageEditor, or None → Auto mode
    threshold: float = 0.35,   # Binarization threshold for noise map
    use_otsu: bool = False,     # Use Otsu auto-threshold instead
    dilate_kernel: int = 20,    # Dilation kernel for user brush ROI
    min_area: int = 200,        # Min CC area (px²) to retain
    dist_alpha: float = 0.6,    # Distance Transform blend weight
    dist_gamma: float = 0.7,    # Gamma correction (< 1 = wider edit zone)
    clamp_rate: float = 3.0,    # Noise map clamp aggressiveness
    # ── Edit strength params ──────────────────────────────────────────────
    scale_ta: float = 1.0,
    scale_edit: float = 0.2,
    scale_non_edit: float = 1.0,
):
    """
    Edit a source image with Semi-Automatic Mask Refinement.

    Compared to edit_image(), this function:
    1. Computes a SOFT (continuous) Δε noise map instead of a hard binary mask.
    2. Optionally intersects with user brush strokes (editor_output) to focus
       the mask on the user-intended region.
    3. Applies Connected Components filtering + Distance Transform to produce
       a smooth center-to-edge gradient mask.
    4. Feeds the soft mask into MaskController for ARAM-guided IP-Adapter editing.

    If editor_output is None (no brush strokes), falls back to Auto mode
    using just the noise map — equivalent to edit_image() but with soft mask.

    Args:
        img_path:       Path to source image.
        src_p:          Source prompt.
        edit_p:         Edit prompt.
        inverse_model:  SwiftEdit InverseModel.
        aux_model:      SwiftEdit AuxiliaryModel.
        ip_sb_model:    SwiftEdit IPSBV2Model.
        editor_output:  Gradio ImageEditor dict (optional). None = Auto mode.
        threshold:      Binarization threshold for noise map [0.1, 0.9].
        use_otsu:       Use Otsu's automatic threshold.
        dilate_kernel:  Brush dilation size (px).
        min_area:       Min blob area to keep.
        dist_alpha:     Distance Transform blend weight (0.6 recommended).
        dist_gamma:     Gamma for distance gradient width (0.7 recommended).
        clamp_rate:     Noise map clamp aggressiveness (3.0 = same as original).
        scale_ta:       Edit strength (text attention scale).
        scale_edit:     IP-Adapter scale in foreground (edit region).
        scale_non_edit: IP-Adapter scale in background (preserved region).

    Returns:
        res_gen_img:    Result image tensor (batch, C, H, W).
        soft_mask_cpu:  (64, 64) float32 Tensor [0,1] — final soft mask (CPU).
        debug:          dict with heatmap/overlay PIL Images for UI display.
    """
    # ── Encode image ──────────────────────────────────────────────────────
    pil_img_cond = Image.open(img_path).resize((512, 512))
    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1
    latents = inverse_model.vae.encode(
        processed_image.to(inverse_model.weight_dtype)
    ).latent_dist.sample()
    latents = latents * inverse_model.vae.config.scaling_factor

    # ── Compute Δε noise map (SOFT, continuous) ───────────────────────────
    soft_map_64, inverted_noise_src = extract_noise_map(
        inverse_model, latents, src_p, edit_p,
        clamp_rate=clamp_rate, t=500,
    )

    # Upsample to 512×512 for OpenCV processing
    noise_map_512 = F.interpolate(
        soft_map_64.unsqueeze(0).unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    # ── Build final soft mask ─────────────────────────────────────────────
    roi_mask = extract_user_brush(editor_output, dilate_kernel=dilate_kernel)

    if roi_mask is not None:
        # Semi-Auto mode: intersect noise map with user brush ROI
        refined = refine_mask(
            noise_map_512, roi_mask,
            threshold=threshold, use_otsu=use_otsu, min_area=min_area,
        )
        binary_for_dt = refined["binary_mask"]
    else:
        # Auto mode: just threshold the noise map
        binary_uint8 = ((noise_map_512 > threshold) * 255).astype(np.uint8)
        binary_for_dt = binary_uint8
        refined = {
            "noise_binary": binary_uint8,
            "binary_mask": binary_uint8,
            "gaussian_soft": np.clip(noise_map_512, 0, 1).astype(np.float32),
        }

    # Distance Transform → smooth gradient mask
    soft_mask_512 = distance_transform_mask(binary_for_dt, alpha=dist_alpha, gamma=dist_gamma)

    # Resize to (64, 64) CUDA tensor for MaskController
    mask_tensor = prepare_mask_tensor(soft_mask_512, device="cuda")

    # ── Build debug visualizations ────────────────────────────────────────
    img_np = np.array(pil_img_cond.resize((512, 512)))
    from src.mask_refine import make_heatmap_overlay, make_soft_mask_visual, make_red_overlay
    debug = {
        "heatmap":   Image.fromarray(make_heatmap_overlay(img_np, noise_map_512)),
        "soft_mask": Image.fromarray(make_soft_mask_visual(soft_mask_512)),
        "overlay":   Image.fromarray(make_red_overlay(img_np, soft_mask_512)),
    }

    # ── Run ARAM-guided generation ────────────────────────────────────────
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_src
    mask_controller = MaskController(
        mask_tensor,
        scale_text_hiddenstate=scale_ta,
        scale_ip_fg=scale_edit,
        scale_ip_bg=scale_non_edit,
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    res_gen_img, _ = ip_sb_model.gen_img(
        pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
    )

    return res_gen_img, mask_tensor.cpu(), debug


if __name__ == "__main__":

    # Define model
    inverse_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
    inverse_model = InverseModel(inverse_ckpt)
    aux_model = AuxiliaryModel()

    path_unet_sb = (os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5"))
    ip_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
    ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True)

    # Input

    img_path = "./assets/imgs_demo/woman_face.jpg"
    src_p = "woman"
    edit_p = "Taylor Swift"
    scale_ta = 1

    # img_path = "./assets/imgs_demo/02.jpg"
    # src_p = "dog"
    # edit_p = "dog with mouth opened"

    start_time = time.time()
    result, _ = edit_image(img_path, src_p, edit_p, inverse_model, aux_model, ip_sb_model, scale_ta=scale_ta)
    print(f"Edit {src_p}->{edit_p} in {time.time()-start_time}")

    # Save result with new naming convention
    os.makedirs("results", exist_ok=True)
    safe_src = re.sub(r'[\\/*?:"<>|]', '_', src_p)
    safe_edit = re.sub(r'[\\/*?:"<>|]', '_', edit_p)
    save_name = f"{safe_src}->{safe_edit}_SY_{scale_ta}.png"
    save_image(result, os.path.join("results", save_name))
    print(f"Saved to results/{save_name}")

