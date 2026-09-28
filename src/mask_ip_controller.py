# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MaskController:
    """
        Customized controller to incorporate editing mask.
    """
    def __init__(self, mask_s, scale_text_hiddenstate=None, scale_ip_fg=0, scale_ip_bg=1):
        self.mask_s = mask_s

        self.scale_ip_fg = scale_ip_fg
        self.scale_ip_bg = scale_ip_bg
        self.scale_text_hiddenstate = scale_text_hiddenstate

    def attn_batch(self, q, k, v, sim, num_heads, **kwargs):
        """
            Forward batch attention with mask control
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")

        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten().unsqueeze(0).unsqueeze(-1)  # (1, hw, 1) soft [0,1]

            # Binarise for attention gating (soft mask → hard gate via threshold 0.5)
            mask_bin = (mask > 0.5).float()

            # foreground attention: suppress background positions
            sim_fg = sim + mask_bin.masked_fill(mask_bin == 0, torch.finfo(sim.dtype).min)
            # background attention: suppress foreground positions
            sim_bg = sim + mask_bin.masked_fill(mask_bin == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def fwd_no_ip(self, q, k, v, sim, num_heads, **kwargs):
        """
            Forward without ip branch
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))

        if B <= 2:
            out_source = self.attn_batch(
                q[:num_heads], k[:num_heads], v[:num_heads], sim[:num_heads], num_heads, **kwargs
            )

            mask = F.interpolate(self.mask_s.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)

            out_target = self.attn_batch(
                q[-num_heads:],
                k[-num_heads:],
                v[-num_heads:],
                sim[-num_heads:],
                num_heads,
                **kwargs
            )
            if self.scale_text_hiddenstate:
                # scale within foreground mask only
                # out_target = self.scale_text_hiddenstate * out_target * mask + out_target * (1 - mask)
                out_target = self.scale_text_hiddenstate * out_target
            out = torch.cat([out_source, out_target], dim=0)
        elif B == 3:
            out_source = self.attn_batch(
                q[:num_heads], k[:num_heads], v[:num_heads], sim[:num_heads], num_heads, **kwargs
            )

            mask = F.interpolate(self.mask_s.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)

            out_target1 = self.attn_batch(
                q[num_heads : (2 * num_heads) :],
                k[num_heads : (2 * num_heads)],
                v[num_heads : (2 * num_heads)],
                sim[num_heads : (2 * num_heads)],
                num_heads,
                **kwargs
            )
            out_target2 = self.attn_batch(
                q[-num_heads:],
                k[-num_heads:],
                v[-num_heads:],
                sim[-num_heads:],
                num_heads,
                **kwargs
            )
            if self.scale_text_hiddenstate:
                # scale within foreground mask only
                out_target1 = self.scale_text_hiddenstate * out_target1 * mask + out_target1 * (
                    1 - mask
                )
                # out_target = self.scale_text_hiddenstate * out_target * mask + out_source * (1 - mask)
            out = torch.cat([out_source, out_target1, out_target2], dim=0)
        return out

    def fwd_ip(self, q, k, v, sim, num_heads, **kwargs):
        """
            Forward with ip branch
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))

        # no auxiliary condition
        if B <= 2:
            out_source = self.attn_batch(
                q[:num_heads], k[:num_heads], v[:num_heads], sim[:num_heads], num_heads, **kwargs
            )
            out_target = self.attn_batch(
                q[-num_heads:],
                k[-num_heads:],
                v[-num_heads:],
                sim[-num_heads:],
                num_heads,
                is_mask_attn=True,
                **kwargs
            )

            out_target_fg, out_target_bg = out_target.chunk(2, 0)

            mask = F.interpolate(self.mask_s.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)  soft [0,1]

            # Blend: foreground region → edited (low IP scale), background → preserved (high IP scale)
            out_target = (
                self.scale_ip_fg * out_target_fg * mask
                + self.scale_ip_bg * out_target_bg * (1 - mask)
            )

            out = torch.cat([out_source, out_target], dim=0)
        elif B == 3:
            out_source = self.attn_batch(
                q[:num_heads], k[:num_heads], v[:num_heads], sim[:num_heads], num_heads, **kwargs
            )
            out_target1 = self.attn_batch(
                q[num_heads : (2 * num_heads) :],
                k[num_heads : (2 * num_heads)],
                v[num_heads : (2 * num_heads)],
                sim[num_heads : (2 * num_heads)],
                num_heads,
                is_mask_attn=True,
                **kwargs
            )
            out_target2 = self.attn_batch(
                q[-num_heads:],
                k[-num_heads:],
                v[-num_heads:],
                sim[-num_heads:],
                num_heads,
                is_mask_attn=True,
                **kwargs
            )

            out_target_fg1, out_target_bg1 = out_target1.chunk(2, 0)
            out_target_fg2, out_target_bg2 = out_target2.chunk(2, 0)

            mask = F.interpolate(self.mask_s.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)  soft [0,1]

            # Blend: foreground → edited, background → preserved (use bg attention, not source)
            out_target1 = (
                self.scale_ip_fg * ((out_target_fg1 + out_target_fg2) / 2) * mask
                + self.scale_ip_bg * out_target_bg1 * (1 - mask)
            )
            out_target2 = out_target_fg2 * mask + out_target_bg2 * (1 - mask)

            out = torch.cat([out_source, out_target1, out_target2], dim=0)

        return out
