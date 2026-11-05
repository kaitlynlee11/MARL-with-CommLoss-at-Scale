# Paper-guided adaptive optimization: measured results

Completed November 4, 2026. This is a controlled follow-up in the same zero-loss, collision-terminated, 0.4-agent-spacing environment. Both continuation arms received 3,200,000 additional world steps from the identical existing adaptive checkpoint. The original checkpoint is retained as the baseline.

## Sources and implementation

[Method, equations, paper-to-code mapping and limitations](paper-guided-optimization.md). The supplied papers are [ACR-PPO](../../23951_Scalable_Cooperative_Mul.pdf), an anonymous ICLR 2026 review copy, and [CS-MARL](../../s10489-026-07202-4.pdf), Nama (2026), DOI 10.1007/s10489-026-07202-4. Neither paper supplies a safety guarantee for this implementation.
The proposed method separates recurrent scheduling and behavior updates, uses clipped product importance ratios and KL early-stop guards, and adds separate reward, packet-cost and collision-risk critics. A clock is visible only to the critic. It uses finite-horizon returns, packet/risk dual ascent and a soft pre-collision training penalty. Message content, one-step delay, observations, sensing and physics are unchanged. This package adaptation is not an exact reproduction of either paper.
The control uses the original joint PPO architecture and objective, but continues from the same checkpoint with a fresh optimizer, conservative LR 0.0001, two PPO epochs and 25-step recurrent sequences. The proposed method uses the same actor LR/epochs/sequence length, but a larger model and critic LR 0.0003. Equal environment steps are not equal optimizer FLOPs.
Both arms use validation seed 100000 only for selection: packet ratio<20%, then success - 2*collision fraction, then success and lower traffic. The initial checkpoint is a candidate for both arms, and the modular initial law/greedy validation matches the original exactly. Neither held-out cohort selects checkpoints.

## Matched and fresh evaluation

Each cohort contains 200 paired scenarios per policy. Matched uses seeds 400000 onward; fresh uses previously unused seeds 600000 onward. Success requires collision-free survival through step 100 and final full landmark coverage. Attempts/600 always uses the full scheduled episode, while attempts/active step uses executed steps. All starts satisfy the 0.4 radial agent-spacing rule; no episodes are filtered.

| Cohort | Policy | Success (95% Wilson) | Collision episodes | Attempts / 600 | Packets/episode | Attempts/active step | Mean active steps |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| matched | baseline | 56.0% (49.1%-62.7%) | 23.5% | 6.04% | 36.21 | 0.434 | 83.50 |
| matched | control | 56.5% (49.6%-63.2%) | 25.0% | 5.93% | 35.61 | 0.435 | 81.86 |
| matched | proposed | 51.5% (44.6%-58.3%) | 10.5% | 5.74% | 34.43 | 0.373 | 92.35 |
| fresh | baseline | 51.0% (44.1%-57.8%) | 29.5% | 6.10% | 36.61 | 0.462 | 79.19 |
| fresh | control | 55.0% (48.1%-61.7%) | 26.5% | 6.02% | 36.10 | 0.448 | 80.56 |
| fresh | proposed | 54.5% (47.6%-61.3%) | 8.5% | 5.78% | 34.69 | 0.368 | 94.39 |

## Interpretation

- matched: proposed versus baseline changes success by -4.5 pp and collision episodes by -13.0 pp. Compared with the equal-budget control, changes are -5.0 pp success and -14.5 pp collisions.
  Simultaneously higher success, fewer collision episodes and traffic below 20% versus baseline: **False** (point estimates).
- fresh: proposed versus baseline changes success by +3.5 pp and collision episodes by -21.0 pp. Compared with the equal-budget control, changes are -0.5 pp success and -18.0 pp collisions.
  Simultaneously higher success, fewer collision episodes and traffic below 20% versus baseline: **True** (point estimates).

The proposed method's fresh-test collision fraction is 8.5%, versus its 2% empirical training risk target. Its fresh success is 54.5%; the original >90% coverage target must be assessed separately from packet savings.
Any collision reduction with lower coverage is a safety/coverage tradeoff, not an overall improvement. A result from one seed and 200 episodes is not a general algorithm superiority claim; Wilson intervals describe evaluation episodes only. Component ablations and independent training seeds are still needed.
The risk multiplier has a finite cap and the KL guard stops future epochs after detecting excessive divergence; neither certifies collision avoidance. No action shield or hidden global-state actor input is used.

## Artifacts and checks

- [Full numerical summary](../results/paper-guided-adaptive-20261104/summary.json)
- [Manifest and PDF provenance](../results/paper-guided-adaptive-20261104/manifest.json)
- [Selected checkpoint hashes and validation scores](../results/paper-guided-adaptive-20261104/selection.json)
- [Matched replay](../results/paper-guided-adaptive-20261104/matched/comparison.html) and [fresh replay](../results/paper-guided-adaptive-20261104/fresh/comparison.html)
- [Completed progress page](../results/paper-guided-adaptive-20261104/progress.html)

Verified 1200 episode records, both additional-training budgets, unchanged PDF/source/checkpoint hashes, identical starting layouts across methods, zero initial collisions, packet accounting, exact initial validation equivalence and replay endpoints. The 56-test suite and complete two-arm smoke pipeline passed before full training. Historical checkpoints and reports were preserved.
