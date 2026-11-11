# Project Context

    project_name       : georl                                                  [HIGH]
    domain             : remote-sensing CV — adaptive tokenization for VFMs      [HIGH]
    framework          : PyTorch 2.x + Hydra + einops + timm                     [HIGH]
    venue              : Pattern Recognition (Elsevier)                          [MED]
    primary_datasets   : 4 datasets (see §6)                                     [HIGH]
    compute_target     : 4 GPUs, FP16, eff. batch 32                             [MED]
    hparams_reference  : §4.1 "Training Configuration" + Table 9                 [HIGH]
    supp_path          : none
    extra_signals      : Algorithm 2 omitted from PDF; 2 theorems; no released ckpts

    NEEDS_USER_DECISION: 0 (defaults locked below; override by editing this file)

---

## 1. project_name → `georl`

- **Derived value**: `georl`
- **Paper anchor**: Title (page 1), Abstract line 7, §1 ¶3 ("a new framework, termed GeoRL").
- **Confidence**: HIGH
- **Rationale**: paper title yields the content words `GeoRL`, `Adaptive`, `Tokenization`, `Remote`, `Sensing`, `Foundation`, `Models`; the lab brand name `GeoRL` is the canonical 1-token identifier and is preferred per the kickoff's "2–4 content words" rule.

## 2. supp_path → `none`

- **Paper anchor**: sibling listing of `paper/`; no `*supp*`, `*_si.*`, `SI.*`, or `appendix*` artifacts present; main text contains no reference to "SI" or "Supplementary".
- **Confidence**: HIGH

## 3. domain → remote-sensing computer vision; adaptive tokenization across detection, instance segmentation, semantic segmentation, scene classification

- **Paper anchor**: Abstract lines 8–17, §1 ¶1–¶4, §4.1 task list.
- **Confidence**: HIGH

## 4. framework → PyTorch 2.x (`torch.nn` + `torch.distributed`) + Hydra 1.3 + einops + timm

- **Paper anchor**: §3.5 (TransformerBlock with manual QKV and scaled-dot-product attention — naturally `torch.nn.functional.scaled_dot_product_attention`); §3.6 (PPO objective Eq. 16–18 implemented in-tree); §4.1 "Network Architecture" (BatchNorm + ReLU + dropout = PyTorch idioms); baseline set DINOv2 / SAM / RemoteCLIP / SatMAE / A-ViT — all PyTorch-native. AdamW + cosine schedule + grad-clip + FP16 (§4.1 Training Configuration).
- **Confidence**: HIGH

## 5. venue → *Pattern Recognition* (Elsevier)

- **Paper anchor**: front-matter style (continuous numeric references in square brackets, alphabetized reference list, ORCID + Scopus IDs); CRediT contributor statement (Elsevier standard since 2018); ~50% of the cited works are in Elsevier journals (*Pattern Recognition*, *ISPRS J. Photogrammetry*, *Remote Sensing of Environment*); single-column long-form layout consistent with the Elsevier `elsarticle` 1p-class.
- **Confidence**: MEDIUM — *ISPRS J. Photogrammetry & Remote Sensing* is also plausible; resolved to *Pattern Recognition* as the most-cited target journal in the bibliography.
- **Override**: edit this section if the manuscript is under review at a different journal.

## 6. primary_datasets — 4 entries

| Name | Version | License | Access URL | Notes |
|------|---------|---------|------------|-------|
| DOTA | v1.0 | non-commercial research use (DOTA license) | https://captain-whu.github.io/DOTA/ | 2,806 images, 188,282 instances, 15 categories, sizes 800×800 to 20,000×20,000. |
| iSAID | v1.0 | non-commercial research use (DOTA-derived) | https://captain-whu.github.io/iSAID/ | 655,451 instance masks across 15 categories on the same 2,806 DOTA images. |
| LoveDA | v1.0 | CC BY-NC 4.0 | https://github.com/Junjue-Wang/LoveDA | 5,987 images, 0.3 m ground resolution, Urban + Rural splits. |
| xView | v1 | CC BY-NC-SA 4.0 | https://xviewdataset.org/ | ~1 M instances, 60 categories, WorldView-3 imagery, 0.3 m. |

Cross-modal transfer (§4.4 Table 10): Sentinel-1 GRD SAR (C-band, VV+VH) co-registered to LoveDA; Sentinel-2 L2A (10 bands, 10 m); AVIRIS hyperspectral (224 bands). Public, no relabelling, per-channel normalisation only.

- **Paper anchor**: §4.1 "Datasets" pp. 18–19; §4.4 "Cross-Modal Transfer" p. 25.
- **Confidence**: HIGH

## 7. compute_target → 4 GPUs, FP16, effective batch 32; wall-clock and GPU model not reported (`COMPUTE_NOT_REPORTED`)

- **Paper anchor**: §4.1 "Training Configuration" — "batch size 32 (8 images per GPU × 4 GPUs), 50 epochs, FP16 mixed precision".
- **Effective batch**: `bs × grad_accum × world_size = 8 × 1 × 4 = 32`. Matches.
- **Upper-bound estimate** (Pattern-Recognition reviewer-grade transparency, since GPU model is not reported):
  - ViT-L/14 forward at 512×512 input ≈ 362 GFLOPs/image (Table 6 row "ViT-L/14 (Uniform)").
  - Policy + value networks ≈ +24 GFLOPs/image (GeoRL-Full row, Table 6).
  - Backward ≈ 2× forward → ≈ 770 GFLOPs/image/step.
  - 50 epochs × (≈12,000 DOTA train images / 32 batch) ≈ 18,750 optimiser steps × 32 batch = 600 K image-passes.
  - Total ≈ 600,000 × 770 GFLOPs ≈ 4.6 × 10⁵ TFLOPs.
  - At 312 TFLOPs/s peak FP16 on a single A100, divided by 4 GPUs and ≈ 50% utilisation: ≈ 4.6 × 10⁵ / (4 × 156) ≈ 740 GPU-hours wall-clock. Reported as **upper bound** in README; actual wall-clock awaits an empirical run.
  - **Storage estimate**: DOTA 35 GB + iSAID 50 GB + LoveDA 20 GB + xView 70 GB = ~175 GB raw imagery; ~250 GB after tiling. Checkpoint per task ≈ 1.3 GB (ViT-L/14 + policy + value).
- **Confidence**: MEDIUM

## 8. hparams_reference → §4.1 "Training Configuration" (prose), §4.1 "Network Architecture" (prose), Table 9 (reward coefficients), §3.4–§3.7 (reward design), Algorithm 1.

| Hparam | Value | Source |
|--------|-------|--------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | §4.1 Training |
| LR | 3 × 10⁻⁴ | §4.1 Training |
| LR schedule | Cosine annealing | §4.1 Training |
| Weight decay | 0.01 | §4.1 Training |
| Batch size | 32 (8/GPU × 4) | §4.1 Training |
| Epochs | 50 | §4.1 Training |
| PPO epochs / iteration (K) | 4 | §4.1 Training |
| Grad clip | 0.5 | §4.1 Training |
| Precision | FP16 mixed | §4.1 Training |
| Input resolution | 512 × 512 | Figure 1 caption + §4.1 Baselines |
| Base patch size P | 16 | Figure 1 caption |
| N candidate regions | 1024 | Figure 1 caption (= 32×32 grid at P=16) |
| α (task) | 1.0 | Table 9 row 2 (best) |
| β (cost) | 0.3 | Table 9 row 2 (best) |
| λ (consistency) | 0.1 | Table 9 row 2 (best) |
| PPO clip ε | 0.2 | standard PPO (§3.6 references Schulman et al. — paper does not over-specify; locked at 0.2) |
| GAE γ | 0.99 | standard (paper §3.1 calls γ ∈ [0,1); locked at 0.99) |
| GAE λ_GAE | 0.95 | standard; locked at 0.95 |
| Value-loss coefficient c₁ | 0.5 | standard PPO |
| Entropy coefficient c₂ | 0.01 | standard PPO |
| State Encoder | 3 conv (64, 128, 256) + BN + ReLU + GAP | §4.1 Network Architecture |
| Transformer Block | 4 blocks, d=256, h=4, FFN=1024, dropout=0.1 | §4.1 Network Architecture |
| Policy Head | linear 256 → 4 | §4.1 Network Architecture |
| Value Head | linear 256 → 128 → 1, ReLU | §4.1 Network Architecture |
| Action token counts | {fine: 16, medium: 4, coarse: 1, skip: 0} | §3.3 + Figure 1 caption |

- **Confidence**: HIGH

## 9. extra_signals

- **Algorithm 1 (Training)**: present (p. 17). Maps 1:1 to `src/georl/training/loop.py::train_one_iter`.
- **Algorithm 2 (Inference)**: announced in §3.9 but not rendered in the PDF body. **Reconstruction**: a deterministic forward pass through state encoder → policy → adaptive tokenizer → frozen VFM backbone → task head, with `arg max` action selection rather than sampling. Implemented in `src/georl/evaluation/inference.py`. Flagged in `docs/deviations.md`.
- **Theorems 1 (Convergence) & 2 (Sample Complexity)**: pp. 14–16. Non-code; restated with assumptions A1–A3 in `docs/theory-notes.md`.
- **Proprietary tokenizer**: none.
- **Released checkpoints**: none announced; README will state "available upon reasonable request".
- **Fairness / ethics section**: none.
- **Code-availability statement** (pre-written in paper): none.

---

Ready to proceed to Turn 1.
