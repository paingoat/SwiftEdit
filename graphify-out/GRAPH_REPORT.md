# Graph Report - SwiftEdit  (2026-09-28)

## Corpus Check
- 10 files · ~58,041 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 4, .example 1)

## Summary
- 87 nodes · 130 edges · 12 communities (7 shown, 5 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `70a77cc9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- app.py
- models.py
- IPSBV2Model
- MaskController
- download_weights.py
- AttnProcessor2_0
- IPAttnProcessor2_0WithIPMaskController
- AuxiliaryModel
- InverseModel
- rules/graphify.md
- workflows/graphify.md

## God Nodes (most connected - your core abstractions)
1. `IPSBV2Model` - 12 edges
2. `edit_image()` - 8 edges
3. `IPAttnProcessor2_0WithIPMaskController` - 7 edges
4. `MaskController` - 7 edges
5. `AttnProcessor2_0` - 6 edges
6. `IPAttnProcessor2_0` - 6 edges
7. `run_edit()` - 4 edges
8. `ImageProjModel` - 4 edges
9. `InverseModel` - 4 edges
10. `AuxiliaryModel` - 4 edges

## Surprising Connections (you probably didn't know these)
- `edit_image()` --calls--> `MaskController`  [INFERRED]
  infer.py → src/mask_ip_controller.py
- `IPSBV2Model` --uses--> `AttnProcessor2_0`  [INFERRED]
  models.py → src/attention_processor.py
- `IPSBV2Model` --uses--> `IPAttnProcessor2_0`  [INFERRED]
  models.py → src/attention_processor.py
- `IPSBV2Model` --uses--> `IPAttnProcessor2_0WithIPMaskController`  [INFERRED]
  models.py → src/mask_attention_processor.py
- `run_edit()` --calls--> `edit_image()`  [EXTRACTED]
  app.py → infer.py

## Import Cycles
- None detected.

## Communities (12 total, 5 thin omitted)

### Community 0 - "app.py"
Cohesion: 0.16
Nodes (16): Replace filesystem-unsafe characters with underscores., run_edit(), _sanitize(), gradio, Image, edit_image(), no_grad, Save keysteps to file. + img_path: path to the source image. + src_p: Source… (+8 more)

### Community 1 - "models.py"
Cohesion: 0.22
Nodes (11): diffusers, einops, numpy, # TODO: add support for attn.scale when we move to Torch 2.1, # TODO: add support for attn.scale when we move to Torch 2.1, # TODO: add support for attn.scale when we move to Torch 2.1, # TODO: add support for attn.scale when we move to Torch 2.1, torch (+3 more)

### Community 2 - "IPSBV2Model"
Cohesion: 0.28
Nodes (4): inference_mode, IPSBV2Model, no_grad, SwiftBrushv2 model with incorporated IP-Adapter.

### Community 3 - "MaskController"
Cohesion: 0.28
Nodes (5): MaskController, Forward with ip branch, Customized controller to incorporate editing mask., Forward batch attention with mask control, Forward without ip branch

### Community 4 - "download_weights.py"
Cohesion: 0.22
Nodes (8): dotenv, download_swiftedit_weights(), precache_base_models(), Download and extract SwiftEdit pretrained weights from HF Hub., Pre-download HuggingFace base models used by models.py so they are cached in…, huggingface_hub, os, tarfile

### Community 5 - "AttnProcessor2_0"
Cohesion: 0.25
Nodes (4): AttnProcessor2_0, IPAttnProcessor2_0, r""" Attention processor for IP-Adapater for PyTorch 2.0. Args: hidden_size…, r""" Processor for implementing scaled dot-product attention (enabled by…

### Community 6 - "IPAttnProcessor2_0WithIPMaskController"
Cohesion: 0.29
Nodes (3): ImageProjModel, IPAttnProcessor2_0WithIPMaskController, r""" Attention processor for IP-Adapater for PyTorch 2.0. Args: hidden_size…

## Knowledge Gaps
- **2 isolated node(s):** `graphify`, `Workflow: graphify`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 45 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `IPSBV2Model` connect `IPSBV2Model` to `app.py`, `models.py`, `AttnProcessor2_0`, `IPAttnProcessor2_0WithIPMaskController`?**
  _High betweenness centrality (0.192) - this node is a cross-community bridge._
- **Why does `MaskController` connect `MaskController` to `app.py`, `models.py`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Why does `edit_image()` connect `app.py` to `MaskController`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `IPSBV2Model` (e.g. with `AttnProcessor2_0` and `IPAttnProcessor2_0`) actually correct?**
  _`IPSBV2Model` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Workflow: graphify` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._