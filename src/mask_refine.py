# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# src/mask_refine.py
# ─────────────────────────────────────────────────────────────────────────────
# Semi-Automatic Mask Refinement pipeline for SwiftEdit.
#
# Pipeline:
#   1. extract_noise_map          — Δε soft map from SwiftEdit InverseModel
#   2. extract_user_brush         — ROI mask from Gradio ImageEditor brush
#   3. refine_mask                — Intersect + Morphology + CC Filter + Blur
#   4. distance_transform_mask    — Gradient soft mask (center→edge falloff)
#   5. prepare_mask_tensor        — Resize to (64,64) CUDA tensor for MaskController
#   6. Visualization helpers      — heatmap, soft mask, red overlay
#

from __future__ import annotations

import warnings
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import tokenize_captions


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NOISE MAP (Δε) from SwiftEdit InverseModel
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_noise_map(
    inverse_model,
    latents: torch.Tensor,
    src_p: str,
    edit_p: str,
    clamp_rate: float = 3.0,
    t: int = 500,
):
    """
    Compute the soft Δε noise-difference map using SwiftEdit's own InverseModel.

    Unlike the notebook approach (SD 2.1 UNet + random forward noising),
    this uses the SAME unet_inverse that SwiftEdit uses internally, ensuring
    the noise map is in the correct latent space.

    Args:
        inverse_model: SwiftEdit InverseModel instance.
        latents:       VAE-encoded latents (1, 4, 64, 64) on CUDA.
        src_p:         Source prompt string.
        edit_p:        Edit prompt string.
        clamp_rate:    Controls noise map clamp aggressiveness.
        t:             Timestep for inversion (default 500).

    Returns:
        soft_map:          Tensor (64, 64) float32 in [0, 1], on CPU.
                           CONTINUOUS (not binarized).
        inverted_noise_src: ε_src tensor (for reuse in edit_image_with_mask).
    """
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * t
    dub_latents  = torch.cat([latents] * 2, dim=0)  # (2, 4, 64, 64)

    input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to("cuda")
    encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
        dtype=inverse_model.weight_dtype
    )

    predict_inverted_code = inverse_model.unet_inverse(
        dub_latents, mid_timestep, encoder_hidden_state
    ).sample.to("cuda", dtype=inverse_model.weight_dtype)

    inverted_noise_src, inverted_noise_edit = predict_inverted_code.chunk(2)
    subed = (inverted_noise_src - inverted_noise_edit).abs_().mean(dim=[0, 1])

    # Soft clamp-normalize to [0,1] — NO binarization
    max_v    = (subed.mean() * clamp_rate).clamp(min=1e-6).item()
    soft_map = (subed.clamp(0, max_v) / max_v).detach().cpu()

    return soft_map, inverted_noise_src


# ═══════════════════════════════════════════════════════════════════════════════
# 2. USER BRUSH → ROI MASK
# ═══════════════════════════════════════════════════════════════════════════════

def extract_user_brush(
    editor_output,
    dilate_kernel: int = 20,
    target_size: int = 512,
) -> Optional[np.ndarray]:
    """
    Extract user brush strokes from a Gradio ImageEditor output dict.

    Gradio ImageEditor dict structure:
        { "background": PIL.Image, "layers": [PIL.Image, ...], "composite": PIL.Image }
    Brush strokes live in "layers" as RGBA images (alpha = painted pixels).

    Args:
        editor_output: Output dict from gr.ImageEditor, or None.
        dilate_kernel: Elliptical dilation kernel size to expand ROI.
        target_size:   Output resolution (default 512).

    Returns:
        roi_mask: (target_size, target_size) uint8 {0, 255}, or None if empty.
    """
    if editor_output is None:
        return None

    layers = editor_output.get("layers", [])
    if not layers:
        return None

    merged = np.zeros((target_size, target_size), dtype=np.uint8)
    for layer in layers:
        if layer is None:
            continue
        pil = layer if isinstance(layer, Image.Image) else Image.fromarray(layer)
        pil = pil.convert("RGBA").resize((target_size, target_size), Image.LANCZOS)
        arr = np.array(pil)
        painted = (arr[..., 3] > 10) | (arr[..., :3].max(axis=2) > 10)
        merged  = np.maximum(merged, painted.astype(np.uint8) * 255)

    if merged.max() == 0:
        return None

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    roi_mask = cv2.dilate(merged, kernel, iterations=2)
    return roi_mask  # (512, 512) uint8 {0, 255}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MASK REFINEMENT (Intersect + Morphology + CC Filter + Gaussian Blur)
# ═══════════════════════════════════════════════════════════════════════════════

def refine_mask(
    noise_map_512: np.ndarray,
    roi_mask: np.ndarray,
    threshold: float = 0.35,
    use_otsu: bool = False,
    min_area: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Refine the raw noise map using the user's ROI mask (5-step pipeline).

    Steps:
        B1: Binarize noise_map via threshold or Otsu.
        B2: Intersection with roi_mask → focus on user-intended region.
            Fallback to raw noise mask if intersection is empty.
        B3: Morphological Opening (5×5) + Closing (9×9).
        B4: Connected Components Filtering — keep blobs overlapping ROI.
        B5: Gaussian Blur (11×11) → smooth edges.

    Args:
        noise_map_512:  (512, 512) float32 [0,1] — upsampled noise diff map.
        roi_mask:       (512, 512) uint8 {0, 255} — dilated user brush ROI.
        threshold:      Binarization threshold [0, 1] (ignored if use_otsu).
        use_otsu:       Use Otsu's automatic threshold.
        min_area:       Min connected component area (px²) to retain.

    Returns:
        dict:
            'noise_binary':  (512,512) uint8 — raw binarized noise map.
            'binary_mask':   (512,512) uint8 — refined binary mask {0,255}.
            'gaussian_soft': (512,512) float32 [0,1] — Gaussian soft mask.
    """
    diff_uint8 = (noise_map_512 * 255).astype(np.uint8)

    # B1: Binarize
    if use_otsu:
        _, noise_binary = cv2.threshold(
            diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, noise_binary = cv2.threshold(
            diff_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY
        )

    # B2: Intersection
    roi_u8       = roi_mask if roi_mask.dtype == np.uint8 else (roi_mask * 255).astype(np.uint8)
    intersection = cv2.bitwise_and(noise_binary, roi_u8)
    if intersection.sum() == 0:
        warnings.warn(
            "Mask intersection empty — falling back to raw noise mask. "
            "Try increasing Dilate Kernel or decreasing Threshold."
        )
        intersection = noise_binary.copy()

    # B3: Morphological
    opened = cv2.morphologyEx(
        intersection, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    # B4: Connected Components Filtering
    roi_binary = (roi_u8 > 127).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    filtered_mask = np.zeros_like(closed)
    for lid in range(1, n_labels):
        area    = stats[lid, cv2.CC_STAT_AREA]
        comp    = (labels == lid).astype(np.uint8)
        overlap = (comp * roi_binary).sum()
        if area >= min_area and overlap > 0:
            filtered_mask = cv2.bitwise_or(filtered_mask, comp * 255)

    if filtered_mask.sum() == 0:
        filtered_mask = closed  # Fallback

    # B5: Gaussian Blur
    blurred      = cv2.GaussianBlur(filtered_mask.astype(np.float32), (11, 11), 0)
    gaussian_soft = np.clip(blurred / 255.0, 0.0, 1.0).astype(np.float32)

    return {
        "noise_binary":  noise_binary,
        "binary_mask":   filtered_mask,
        "gaussian_soft": gaussian_soft,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DISTANCE TRANSFORM → Gradient Soft Mask
# ═══════════════════════════════════════════════════════════════════════════════

def distance_transform_mask(
    binary_mask: np.ndarray,
    alpha: float = 0.6,
    gamma: float = 0.7,
) -> np.ndarray:
    """
    Convert a binary mask to a smooth gradient via Distance Transform.

    Each foreground pixel receives a value proportional to its distance
    to the nearest background pixel → smooth center-to-edge falloff.

    Formula:
        dist = DistanceTransform(binary_mask)      # L2 distance
        dist_norm = (dist / dist.max()) ** gamma   # gamma widens edit zone
        soft = alpha * dist_norm + (1-alpha) * gaussian_blur(binary)

    Args:
        binary_mask:  (H, W) uint8 {0,255} — refined binary mask.
        alpha:        Weight for distance gradient (0.6 = 60% dist, 40% blur).
        gamma:        Gamma exponent. < 1.0 widens bright region (more edit).
                      0.7 is a good default for face/object editing.

    Returns:
        soft_mask: (H, W) float32 [0, 1] — smooth gradient mask.
    """
    if binary_mask.max() == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)

    bin_u8  = (binary_mask > 127).astype(np.uint8) * 255
    dist    = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 5)
    max_val = dist.max()

    if max_val < 1e-6:
        return (bin_u8 / 255.0).astype(np.float32)

    dist_norm = (dist / max_val).astype(np.float32)
    dist_norm = np.power(dist_norm, gamma)  # Gamma correction

    blurred   = cv2.GaussianBlur(bin_u8.astype(np.float32), (11, 11), 0) / 255.0
    soft_mask = alpha * dist_norm + (1.0 - alpha) * blurred

    return np.clip(soft_mask, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PREPARE MASK TENSOR for MaskController
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_mask_tensor(
    soft_mask_np: np.ndarray,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Resize a (512, 512) float32 soft mask to (64, 64) CUDA tensor.

    Uses bilinear interpolation to preserve gradient smoothness.
    The output is directly compatible with MaskController (expects 64×64).

    Args:
        soft_mask_np:  (512, 512) float32 [0, 1].
        device:        Target device.

    Returns:
        mask_tensor: (64, 64) float32 Tensor on `device`.
    """
    t = torch.from_numpy(soft_mask_np).unsqueeze(0).unsqueeze(0)  # (1,1,512,512)
    t = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).to(device)  # (64, 64)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_heatmap_overlay(
    original_np: np.ndarray,
    diff_map: np.ndarray,
    alpha: float = 0.60,
) -> np.ndarray:
    """
    Blend a JET colormap of the noise map onto the original image.
    Red = high Δε (edit region), Blue = background.
    """
    heatmap_bgr = cv2.applyColorMap((diff_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (
        alpha * heatmap_rgb.astype(np.float32)
        + (1.0 - alpha) * original_np.astype(np.float32)
    )
    return blended.clip(0, 255).astype(np.uint8)


def make_soft_mask_visual(soft_mask: np.ndarray) -> np.ndarray:
    """Convert (H,W) float32 soft mask → (H,W,3) uint8 grayscale for display."""
    gray = (soft_mask * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def make_red_overlay(
    original_np: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay semi-transparent red highlight on the mask region."""
    overlay   = original_np.copy().astype(np.float32)
    mask_bool = (mask > 0.5) if mask.dtype == np.float32 else (mask > 127)
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 0] = 255.0
    overlay[mask_bool] = (
        alpha * red_layer[mask_bool] + (1.0 - alpha) * overlay[mask_bool]
    )
    return overlay.clip(0, 255).astype(np.uint8)
