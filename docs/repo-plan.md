# Repo Plan

## Final directory tree

```
georl/
├── pyproject.toml                      PEP-621, ruff + mypy --strict + pytest config
├── requirements.txt                    pinned runtime deps
├── environment.yml                     conda alternative
├── Dockerfile                          nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
├── .gitignore
├── .pre-commit-config.yaml             ruff, black, isort, mypy stub-check
├── .github/workflows/ci.yml            ruff + mypy + pytest on _smoke.yaml
├── README.md
├── LICENSE                             Apache-2.0
├── configs/
│   ├── model/
│   │   ├── state_encoder.yaml
│   │   ├── policy.yaml
│   │   ├── value.yaml
│   │   ├── vit_l14.yaml
│   │   ├── dinov2.yaml
│   │   ├── sam.yaml
│   │   ├── remoteclip.yaml
│   │   └── satmae.yaml
│   ├── data/
│   │   ├── dota.yaml
│   │   ├── isaid.yaml
│   │   ├── loveda.yaml
│   │   ├── xview.yaml
│   │   ├── scene_aid.yaml
│   │   ├── scene_nwpu.yaml
│   │   ├── scene_patternnet.yaml
│   │   ├── sar.yaml
│   │   ├── ms.yaml
│   │   └── hs.yaml
│   ├── train/
│   │   ├── ppo_default.yaml
│   │   └── ddp_4gpu.yaml
│   └── experiment/
│       ├── main_dota.yaml
│       ├── main_isaid.yaml
│       ├── main_loveda.yaml
│       ├── main_xview.yaml
│       ├── main_scene_aid.yaml
│       ├── main_scene_nwpu.yaml
│       ├── main_scene_patternnet.yaml
│       ├── ablation_no_multi_state.yaml
│       ├── ablation_no_progress.yaml
│       ├── ablation_no_consistency.yaml
│       ├── ablation_no_hsa.yaml
│       ├── ablation_no_ppo.yaml
│       ├── ablation_uniform.yaml
│       ├── ablation_action_binary.yaml
│       ├── ablation_action_ternary.yaml
│       ├── ablation_action_quaternary.yaml
│       ├── ablation_action_continuous.yaml
│       ├── ablation_reward_r1.yaml
│       ├── ablation_reward_r2.yaml
│       ├── ablation_reward_r3.yaml
│       ├── ablation_reward_r4.yaml
│       ├── ablation_reward_r5.yaml
│       ├── transfer_sar.yaml
│       ├── transfer_ms.yaml
│       ├── transfer_hs.yaml
│       └── _smoke.yaml
├── src/georl/
│   ├── __init__.py
│   ├── configs/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   ├── dota.py
│   │   ├── isaid.py
│   │   ├── loveda.py
│   │   ├── xview.py
│   │   ├── scene.py
│   │   ├── crossmodal.py
│   │   └── manifests.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones.py
│   │   ├── state_encoder.py
│   │   ├── transformer_block.py
│   │   ├── policy_net.py
│   │   ├── value_net.py
│   │   ├── adaptive_tokenizer.py
│   │   └── heads.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── task.py
│   │   ├── ppo.py
│   │   └── consistency.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── detection.py
│   │   ├── segmentation.py
│   │   ├── classification.py
│   │   └── efficiency.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── mdp.py
│   │   ├── actions.py
│   │   ├── policy.py
│   │   ├── rewards.py
│   │   ├── gae.py
│   │   └── buffer.py
│   ├── hsa/
│   │   ├── __init__.py
│   │   ├── hierarchy.py
│   │   ├── anchor.py
│   │   └── explain.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loop.py
│   │   ├── optim.py
│   │   ├── scheduler.py
│   │   ├── ddp.py
│   │   ├── amp.py
│   │   ├── checkpoint.py
│   │   └── ema.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── detection_eval.py
│   │   ├── segmentation_eval.py
│   │   ├── classification_eval.py
│   │   ├── stats.py
│   │   └── inference.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── logging.py
│   │   ├── distributed.py
│   │   └── timing.py
│   └── cli/
│       ├── __init__.py
│       ├── train.py
│       ├── eval.py
│       ├── infer.py
│       └── export_onnx.py
├── scripts/
│   ├── launch_train.sh
│   ├── launch_eval.sh
│   ├── prepare_data.sh
│   └── compute_manifest.sh
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_state_encoder_shapes.py
│   │   ├── test_transformer_block.py
│   │   ├── test_policy_value_softmax.py
│   │   ├── test_adaptive_tokenizer.py
│   │   ├── test_action_token_counts.py
│   │   ├── test_rewards_signs.py
│   │   ├── test_gae_recurrence.py
│   │   ├── test_ppo_clip_bounds.py
│   │   ├── test_consistency_neighbours.py
│   │   ├── test_task_losses.py
│   │   ├── test_hsa_argmax.py
│   │   ├── test_efficiency_counters.py
│   │   └── test_seed_roundtrip.py
│   ├── integration/
│   │   ├── test_dota_metric_protocol.py
│   │   ├── test_hsa_distribution.py
│   │   ├── test_ckpt_resume_seed.py
│   │   └── test_dataset_loaders.py
│   ├── regression/
│   │   └── test_token_ratio_main.py
│   ├── overfit/
│   │   └── test_single_batch_overfit.py
│   └── smoke/
│       └── test_training_smoke.py
├── docs/
│   ├── project-context.md
│   ├── implementation-map.md
│   ├── repo-plan.md
│   ├── deviations.md
│   └── theory-notes.md
└── assets/
    ├── figures_1_2_architecture.md
    ├── figure_3_pipeline.md
    ├── figure_4_qualitative.md
    └── figure_5_crossdataset.md
```

## Module responsibilities

| Sub-package | Responsibility |
|---|---|
| `configs/` | Hydra structured configs (dataclass schemas) — single source of truth for hparams. |
| `data/` | Dataset classes, 512×512 augmentation pipeline, SHA-256 manifest verification. |
| `models/` | State encoder (Eq. 1–5), transformer block (Eq. 13–14), policy + value networks (Eq. 12, 15), adaptive tokenizer (assembles per-region tokens), backbone Protocol + adapters, task heads. |
| `losses/` | Task losses (CE, IoU, oriented bbox), consistency reward (Eq. 11), PPO clipped surrogate + total objective (Eq. 16–18). |
| `metrics/` | DOTA mAP@0.5, iSAID bg-ignored mIoU, LoveDA mIoU + OA, scene OA, efficiency (GFLOPs, latency, memory, token ratio). |
| `rl/` | MDP tuple, action enum (Eq. 6), policy head (Eq. 7), reward composition (Eq. 8–11), GAE (Eq. 17), rollout buffer. |
| `hsa/` | 3-level semantic hierarchy, policy-semantic alignment (Eq. 19), explanation template (Eq. 20). |
| `training/` | Algorithm 1 training loop, AdamW + cosine, DDP, AMP, atomic checkpointing, EMA. |
| `evaluation/` | Per-task evaluation loops, bootstrap CI + 5-seed std, reconstructed Algorithm 2 (`inference.py`). |
| `utils/` | Seed (round-tripped through checkpoint), logging setup, distributed helpers, timing. |
| `cli/` | Hydra entry points: `train`, `eval`, `infer`, `export_onnx`. |

## Pinned dependencies

```
torch==2.3.1
torchvision==0.18.1
hydra-core==1.3.2
omegaconf==2.3.0
einops==0.8.0
attrs==23.2.0
timm==1.0.7
numpy==1.26.4
pillow==10.3.0
opencv-python-headless==4.10.0.84
pycocotools==2.0.7
shapely==2.0.4
fvcore==0.1.5.post20221221
rich==13.7.1

# dev
mypy==1.10.0
ruff==0.5.0
black==24.4.2
isort==5.13.2
pytest==8.2.2
pytest-cov==5.0.0
pre-commit==3.7.1
```

## Expected coverage (post-Turn 6)

- `pytest -q` exits 0 with at least 25 passing tests covering shape, unit, integration, regression, overfit, smoke.
- `mypy --strict src/georl` exits 0.
- `ruff check .` exits 0.
- `docker build .` exits 0.
- `docs/` contains the four mandatory documents + a theory-notes restatement of Theorems 1 & 2.
