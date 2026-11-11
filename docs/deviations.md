# Deviations Log

Each entry records a place where the implementation diverges from a strict reading of the paper, with a paper anchor and a justification. Empty entries are not allowed; remove the row instead.

| # | File | Paper anchor | Deviation | Justification |
|---|------|--------------|-----------|---------------|
| 1 | `src/georl/evaluation/inference.py` | §3.9 (Algorithm 2 announced, not rendered in the PDF) | Algorithm 2 box absent from the manuscript; we reconstruct the inference path from §3.2 State Encoder (Eq. 1–5), §3.5 Policy Network (Eq. 12), §3.7 HSA (Eq. 19–20), with deterministic `arg max` action selection in place of stochastic sampling. | Required to provide a runnable `python -m georl.cli.infer` entry point. Behaviour is uniquely determined by the rest of the paper; no new hyperparameters introduced. |
| 2 | `src/georl/rl/gae.py` (γ=0.99, λ_GAE=0.95) | §3.1 (γ ∈ [0,1) only); §3.6 (GAE referenced, λ unspecified) | Paper specifies the GAE formula (Eq. 17) but does not fix γ or λ. | Locked to canonical PPO defaults (γ=0.99, λ=0.95) which the original PPO paper uses for clipped-objective continuous-control benchmarks. Configurable via `configs/train/ppo_default.yaml`. |
| 3 | `src/georl/losses/ppo.py` (ε=0.2, c₁=0.5, c₂=0.01) | §3.6 Eq. 16–18 (constants left as symbols) | Paper does not give numeric ε, c₁, c₂. | Locked to PPO canonical values. Configurable. |
| 4 | `docs/project-context.md::compute_target` | §4.1 Training Configuration (GPU model and wall-clock not reported) | Wall-clock and GPU SKU are not in the manuscript. | Recorded as `COMPUTE_NOT_REPORTED` with a peak-FLOPs upper-bound estimate so reviewers see honest numbers instead of fabricated ones. |
