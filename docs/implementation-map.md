# Implementation Map

Each numbered equation, every algorithm box, every reported table, every ablation, every metric is listed below with the file that implements it and the module entry point.

| Paper section | Equation / figure / table | File | Module / symbol | Notes |
|---|---|---|---|---|
| §3.1 ¶3 | MDP tuple `(S, A, P, R, γ)` | `src/georl/rl/mdp.py` | `MDP` | attrs.frozen, γ from cfg |
| §3.2 Eq. 1 | `s_t = φ_state(I, m_{t-1}, t)` | `src/georl/models/state_encoder.py` | `StateEncoder.forward` | concat of three branches |
| §3.2 Eq. 2 | `f_local^{(i)} = Conv₃ₓ₃(I_{r,c,k})` | `src/georl/models/state_encoder.py` | `LocalBranch` | 3 conv blocks 64/128/256 |
| §3.2 Eq. 3 | `f_global = Attn(GAP(Conv₁ₓ₁(I)))` | `src/georl/models/state_encoder.py` | `GlobalBranch` | 1×1 conv → GAP → single-head attn |
| §3.2 Eq. 4 | `f_progress = MLP([m_{t-1}; t/T])` | `src/georl/models/state_encoder.py` | `ProgressBranch` | 2-layer MLP |
| §3.2 Eq. 5 | `s_t = LayerNorm(Linear([f_local; f_global; f_progress]))` | `src/georl/models/state_encoder.py` | `StateEncoder.forward` | LN over feature dim |
| §3.3 Eq. 6 | `A = {a_fine, a_medium, a_coarse, a_skip}` | `src/georl/rl/actions.py` | `Action`, `TOKEN_COUNTS = (16, 4, 1, 0)` | enum + lookup |
| §3.3 Eq. 7 | `π_θ(a|s_t) = Softmax(W_π s_t + b_π)` | `src/georl/rl/policy.py` | `LinearPolicyHead` | also used by simpler baselines |
| §3.4 Eq. 8 | Composite reward `R = α·r_task − β·r_cost + λ·r_consistency` | `src/georl/rl/rewards.py` | `composite_reward` | reads (α, β, λ) from cfg |
| §3.4 Eq. 9 | `r_task = ℒ(y, ŷ_prev) − ℒ(y, ŷ_a)` | `src/georl/rl/rewards.py` | `task_reward` | frozen backbone + head |
| §3.4 Eq. 10 | `r_cost = n_tokens(a) / N_max` | `src/georl/rl/rewards.py` | `cost_reward` | |
| §3.4 Eq. 11 | `r_consistency = Σ_{j∈N(i)} 1[a_t^{(i)} = a_{t-1}^{(j)}] · sim(f_i, f_j)` | `src/georl/rl/rewards.py` | `consistency_reward` | raster-order neighbour set; median fallback for unseen |
| §3.5 Eq. 12 | `π_θ(a|s) = Softmax(MLP_π(TransformerBlock(s)))` | `src/georl/models/policy_net.py` | `PolicyNetwork.forward` | 4 transformer blocks |
| §3.5 Eq. 13 | `Q,K,V = W_Q s, W_K s, W_V s` | `src/georl/models/transformer_block.py` | `MultiHeadSelfAttention.forward` | manual projection |
| §3.5 Eq. 14 | `Attn(Q,K,V) = Softmax(QKᵀ/√d_k) V` | `src/georl/models/transformer_block.py` | same | `F.scaled_dot_product_attention` |
| §3.5 Eq. 15 | `V_φ(s) = MLP_V(TransformerBlock(s))` | `src/georl/models/value_net.py` | `ValueNetwork.forward` | shared block, separate head |
| §3.6 Eq. 16 | PPO clipped surrogate `min(r·Â, clip(r, 1±ε)·Â)` | `src/georl/losses/ppo.py` | `ppo_clip_loss` | |
| §3.6 Eq. 17 | `Â_t = Σ (γλ)^l δ_{t+l}` | `src/georl/rl/gae.py` | `compute_gae` | |
| §3.6 Eq. 18 | `L = L_PPO + c₁·L_value − c₂·H[π]` | `src/georl/losses/ppo.py` | `total_objective` | |
| §3.7 Eq. 19 | `p_a = (1/|D_a|) Σ c(s)·1[a' = a]` | `src/georl/hsa/anchor.py` | `policy_semantic_alignment` | 3-level hierarchy |
| §3.7 Eq. 20 | `Explain(a, s) = Template(arg max_c p_a[c], a)` | `src/georl/hsa/explain.py` | `explain_action` | |
| §3.8 Th. 1 | Convergence `‖∇J‖ ≤ ε in O(L²_R / (ε²(1−γ)⁵))` | `docs/theory-notes.md` | restated under A1–A3 | non-code |
| §3.8 Th. 2 | Sample complexity `N ≥ O(|S||A| H³ / (ε²(1−γ)³) log(|S||A|/δ))` | `docs/theory-notes.md` | restated | non-code |
| §3.9 | Algorithm 1 (Training Procedure, 20 lines) | `src/georl/training/loop.py` | `train_one_iter` | line-by-line mapping |
| §3.9 | Algorithm 2 (Inference Procedure — announced, omitted from PDF) | `src/georl/evaluation/inference.py` | `run_inference` | reconstructed; logged in `docs/deviations.md` |
| §4.1 Datasets | DOTA / iSAID / LoveDA / xView loaders | `src/georl/data/{dota,isaid,loveda,xview}.py` | `Dataset` subclasses | per-paper preprocessing |
| §4.1 Table 1 | DOTA-v1.0 detection mAP per class | `configs/experiment/main_dota.yaml`, `tests/integration/test_dota_metric_protocol.py` | | mAP@IoU=0.50 DOTA protocol |
| §4.1 Table 2 | iSAID mIoU & LoveDA mIoU | `configs/experiment/{main_isaid,main_loveda}.yaml` | | bg-ignored mIoU per iSAID protocol |
| §4.1 Table 3 | LoveDA Urban + Rural | `configs/experiment/main_loveda.yaml` (`split=urban|rural`) | | mIoU + OA |
| §4.1 Table 4 | xView detection | `configs/experiment/main_xview.yaml` | | mAP + AP50 + GFLOPs |
| §4.1 Table 5 | Scene classification on AID / NWPU / PatternNet | `configs/experiment/main_scene_{aid,nwpu,patternnet}.yaml` | | OA |
| §4.2 Table 6 | Computational efficiency | `src/georl/evaluation/efficiency.py` | `measure_gflops`, `measure_latency`, `measure_memory`, `token_ratio` | three rows: GeoRL-Light / GeoRL / GeoRL-Full |
| §4.3.1 Table 7 | Component ablation (6 rows) | `configs/experiment/ablation_no_{multi_state,progress,consistency,hsa,ppo,uniform}.yaml` | | one toggle each |
| §4.3.2 Table 8 | Action-space ablation (4 rows) | `configs/experiment/ablation_action_{binary,ternary,quaternary,continuous}.yaml` | | |
| §4.3.3 Table 9 | Reward coefficient sweep (5 rows) | `configs/experiment/ablation_reward_r{1..5}.yaml` | | α=1.0 fixed; (β, λ) swept |
| §4.4 Table 10 | Cross-modal transfer (SAR / MS / HS, Transfer + Fine-tune rows) | `configs/experiment/transfer_{sar,ms,hs}.yaml` | | freeze policy for "Transfer"; unfreeze for "Fine-tuned" |
| §4.5 Table 11 | HSA per-category action % | `tests/integration/test_hsa_distribution.py` | | regression against the table |
| Figures 1, 2 | Architecture diagrams | `assets/figures_1_2_architecture.md` | | references only |
| Figure 3 | Experiment pipeline | `assets/figure_3_pipeline.md` | | references only |
| Figure 4 | Qualitative uniform vs adaptive | `assets/figure_4_qualitative.md` | | references only |
| Figure 5 | Cross-dataset qualitative comparison | `assets/figure_5_crossdataset.md` | | references only |

## Coverage check

- All 22 numbered equations: covered.
- Algorithm 1: covered (training loop).
- Algorithm 2: reconstructed and logged in `docs/deviations.md`.
- Theorems 1 & 2: restated in `docs/theory-notes.md` (no code).
- Tables 1–11: covered by configs + tests.
- Figures 1–5: indexed as `assets/*.md` reference cards.

## Test taxonomy (per-module)

| Module | Test kind | File |
|---|---|---|
| `models/state_encoder.py` | shape | `tests/unit/test_state_encoder_shapes.py` |
| `models/transformer_block.py` | unit + shape | `tests/unit/test_transformer_block.py` |
| `models/policy_net.py` / `value_net.py` | unit + shape | `tests/unit/test_policy_value_softmax.py` |
| `models/adaptive_tokenizer.py` | unit (grid layout) | `tests/unit/test_adaptive_tokenizer.py` |
| `rl/actions.py` | unit (token counts) | `tests/unit/test_action_token_counts.py` |
| `rl/rewards.py` | unit (signs, scaling) | `tests/unit/test_rewards_signs.py` |
| `rl/gae.py` | unit (recurrence) | `tests/unit/test_gae_recurrence.py` |
| `losses/ppo.py` | unit (clip bounds, entropy bonus) | `tests/unit/test_ppo_clip_bounds.py` |
| `losses/consistency.py` | unit (raster-order fallback) | `tests/unit/test_consistency_neighbours.py` |
| `losses/task.py` | unit | `tests/unit/test_task_losses.py` |
| `hsa/*` | unit (Eq. 19, Eq. 20) | `tests/unit/test_hsa_argmax.py` |
| `metrics/detection.py` | integration (DOTA protocol fixture) | `tests/integration/test_dota_metric_protocol.py` |
| `metrics/efficiency.py` | unit | `tests/unit/test_efficiency_counters.py` |
| `data/*` | integration (tiny fixtures) | `tests/integration/test_dataset_loaders.py` |
| `training/checkpoint.py` + `utils/seed.py` | integration | `tests/integration/test_ckpt_resume_seed.py` |
| `training/loop.py` | smoke | `tests/smoke/test_training_smoke.py` |
| Full pipeline | regression | `tests/regression/test_token_ratio_main.py` |
| Full pipeline | overfit | `tests/overfit/test_single_batch_overfit.py` |
| HSA distribution | regression vs Table 11 | `tests/integration/test_hsa_distribution.py` |
