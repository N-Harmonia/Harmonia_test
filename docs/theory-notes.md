# Theory Notes

Restatement of Theorems 1 and 2 from §3.8 of the manuscript, with the assumptions used in the proofs. No code here; this document is the reviewer-facing pointer for the convergence and sample-complexity statements.

## Assumptions (A1–A3)

- **A1 — Bounded reward.** The composite reward `R(s, a) = α · r_task − β · r_cost + λ · r_consistency` is bounded by a constant `L_R` independent of state and action. Since `r_task` is a difference of bounded task losses, `r_cost ∈ [0, 1]`, and `r_consistency` is a sum of bounded cosine similarities over a fixed-size neighbourhood, this holds with `L_R = α · L_task + β + 8λ` for the 8-neighbour raster set.
- **A2 — Lipschitz policy / value parameterisation.** The policy `π_θ` and the value `V_φ` are `L_π`- and `L_V`-Lipschitz in their parameters. The transformer-block architecture used in `src/georl/models/policy_net.py` and `src/georl/models/value_net.py` satisfies this with constants that depend only on `d`, `h`, and the FFN dimension.
- **A3 — Ergodic MDP.** The induced Markov chain over the candidate-region grid is ergodic under the raster-scan visitation order; mixing time is `O(N)` where `N = 1024` is the number of base regions at 512 × 512 input with `P = 16`.

## Theorem 1 (Policy gradient convergence)

Under A1–A3, the PPO update over the composite reward satisfies

```
min_{k ≤ K} E[ ||∇_θ J(θ_k)||² ] ≤ ε
```

after `K ≥ O( L_R² / ( ε² (1 − γ)⁵ ) )` outer iterations, where `γ = 0.99` is the discount factor used throughout this codebase.

## Theorem 2 (Sample complexity)

Under A1–A3, with probability at least `1 − δ`, the suboptimality of the learned policy is bounded by `ε` provided the number of trajectory samples satisfies

```
N ≥ O( |S| · |A| · H³ / ( ε² (1 − γ)³ ) · log(|S| · |A| / δ) )
```

with horizon `H` equal to the number of decision steps per image (here `T = 4` PPO inner epochs).

These bounds are restated to provide the asymptotic dependency on `ε`, `γ`, `H`, and `(|S|, |A|)`; tight numeric constants are omitted because they do not affect the architectural choices made in the implementation.
