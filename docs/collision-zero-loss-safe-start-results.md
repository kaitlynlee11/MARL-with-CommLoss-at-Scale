# Collision-terminated comparison with reset collisions excluded

Recalculated November 4, 2026 from saved episode records. The same **30 reset-collision scenarios (15%)** were removed from all three schedulers in both stages. This leaves **170 matched scenarios per condition per stage**, or 1,020 retained episode records. No policies were retrained, no trajectories rerun, and no replacement scenarios added.

Success still requires collision-free survival through step 100 and full landmark coverage at that final step. Collisions after reset remain immediate failures. Packet loss is zero. The communication denominator remains **600 packets per retained scheduled 100-step episode**.

| Stage | Scheduler | Success (95% Wilson interval) | Any collision after reset | Packets / 600 | Packets/episode | Attempts/active step | Mean steps to termination |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| frozen | adaptive | 28.24% (22.01%-35.43%) | 71.76% | 0.14% | 0.86 | 0.024 | 35.94 |
| frozen | fixed_5 | 31.18% (24.69%-38.49%) | 67.06% | 8.48% | 50.89 | 1.239 | 41.08 |
| frozen | full | 30.59% (24.15%-37.88%) | 69.41% | 38.75% | 232.52 | 6.000 | 38.75 |
| retrained | adaptive | 41.18% (34.05%-48.69%) | 6.47% | 2.02% | 12.09 | 0.126 | 95.89 |
| retrained | fixed_5 | 52.35% (44.88%-59.73%) | 1.18% | 19.82% | 118.94 | 1.200 | 99.08 |
| retrained | full | 61.18% (53.68%-68.18%) | 2.35% | 98.65% | 591.92 | 6.000 | 98.65 |

## Counts and termination details

| Stage | Scheduler | Successful episodes | Collision episodes | Horizon survivors | Median steps | Min-max steps | Mean steps among collision failures |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| frozen | adaptive | 48/170 | 122/170 | 48/170 | 13.0 | 2-100 | 10.73 |
| frozen | fixed_5 | 53/170 | 114/170 | 56/170 | 16.0 | 2-100 | 12.13 |
| frozen | full | 52/170 | 118/170 | 52/170 | 13.5 | 2-100 | 11.76 |
| retrained | adaptive | 70/170 | 11/170 | 159/170 | 100.0 | 4-100 | 36.55 |
| retrained | fixed_5 | 89/170 | 2/170 | 168/170 | 100.0 | 17-100 | 22.00 |
| retrained | full | 104/170 | 4/170 | 166/170 | 100.0 | 16-100 | 42.75 |

## What changed

- All excluded episodes had zero active steps, zero packets and no success. Removing them preserves the total successful episodes, attempted packets and executed steps, while changing the number of evaluated episodes from 200 to 170.
- Consequently, success rate, mean packets per episode, packets/600, and mean executed steps increase by a factor of 200/170 relative to the original tables. Collision fractions are recomputed after subtracting the 30 initial failures from each condition.
- Attempts per active step is unchanged: total attempts divided by total executed steps has the same numerator and denominator after zero-step episodes are removed.
- The previous 85% ceiling caused by initial collisions no longer applies to this filtered cohort. These are results conditional on a collision-free start, not results for the original unrestricted reset distribution.
- Policies and checkpoint selections are unchanged. The frozen adaptive policy was originally trained with 25% loss; all evaluations here and the collision-aware retraining used zero loss. One training seed per condition; Wilson intervals describe episode uncertainty only.

## Artifacts

- [Filtered summaries](../results/collision-zero-loss-20261104/safe-starts/summary.json)
- [Excluded scenario seeds](../results/collision-zero-loss-20261104/safe-starts/excluded-seeds.json)
- [Frozen retained episode records](../results/collision-zero-loss-20261104/safe-starts/frozen/episodes.jsonl)
- [Retrained retained episode records](../results/collision-zero-loss-20261104/safe-starts/retrained/episodes.jsonl)
- [Original all-start comparison](../results/collision-zero-loss-20261104/report.md)
- [Original methodology](collision-zero-loss-study.md)

The original files and replays are preserved. Filtering changes aggregate statistics; it does not change retained trajectories. The existing first-scenario comparison replays use seed 400000, which belongs to the retained cohort.
