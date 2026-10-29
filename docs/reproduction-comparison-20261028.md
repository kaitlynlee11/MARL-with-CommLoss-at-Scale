# Reproduced results compared with historical reports

Run date: October 28, 2026 (America/Los_Angeles). Run directory: `results/reproduction-20261028/`.

The new runs completed training, 2,400 evaluation episodes, and HTML generation. They reproduce the documented experimental procedures, but do not reproduce all historical scores or the historical baseline checkpoint lineage.

## Main findings

- Unconstrained baseline full-communication success fell from 76.5% to 69.5% (-7.0 percentage points). The new baseline refinement stopped early, so its training lineage differs materially from the historical retained model.
- With spaced landmarks and no loss, full communication improved from 90.5% to 92.0%; no communication improved from 81.5% to 87.5%; fixed-five-step communication fell from 91.5% to 85.5%. The historical ranking of schedules was not reproduced.
- At 25% loss, new full-communication success was 90.5%, matching the historical loss-condition score, but decreasing 1.5 percentage points relative to its own new loss-free baseline. New fixed-five-step success remained 85.5%, with 21 episodes changing in each direction.
- Fixed-five-step communication still attempts 80% fewer packets than full communication (120 versus 600 per episode). Similar packet accounting does not imply similar navigation performance.

## Sources and comparison rules

Historical values come from [navigation results](navigation-results.md), [spaced-navigation results](spaced-navigation-results.md), and [25% loss results](spaced-navigation-loss25-results.md). Historical runtime details are in [validation](validation.md).

New values come from [the combined summary](../results/reproduction-20261028/combined-summary.json), the per-study summaries and episode records listed below, and saved training configurations/logs. Historical artifacts were unavailable, so historical numbers are taken from the rounded Markdown tables. Historical checkpoint hashes and per-episode outcomes cannot be independently checked here.

All conditions use three agents, three landmarks, a 100-step horizon, sensing radius 0.6, coverage radius 0.1, deterministic learned-policy evaluation, and 200 held-out episodes per condition. Success means all landmarks are covered simultaneously at the final step. Task return excludes shaping, training bonuses, and communication penalties. Collision rate is the fraction of steps with any overlapping agent pair.

`Delta` means new minus historical. Percentage differences are percentage points (pp); return differences are reward units. Larger success, coverage, and task return are better; lower collision rates are better. New-versus-historical deltas use the historical table precision. All intervals below are 95% Wilson episode intervals.

## Training and evaluation comparability

| Setting | Historical | New run |
| --- | --- | --- |
| Baseline design | One full-trained GRU policy ablated under six evaluation conditions | Same design and conditions |
| Baseline landmark spacing | Unconstrained (0) | Unconstrained (0) |
| Baseline test seeds | 300000-300199 | 300000-300199 |
| Baseline retained lineage | 8,640,000 world steps; refinement parent at 8,320,000 | 7,046,400 steps; refinement parent at 7,040,000 |
| Baseline validation success / coverage | 81% / 90% | 81% / 90% |
| Spaced design | Three independently trained policies, each evaluated under its own schedule | Same design |
| Spaced budgets | 8,320,000 initial + 640,000 refinement steps per policy; no early stopping | Same completed budgets |
| Spaced test seeds / spacing | 400000-400199 / minimum center distance 0.4 | Same seeds and spacing |
| Training seed / CPU threads | Seed 42; baseline 2 threads; spaced policies 1 thread each | Same |
| Loss study | Frozen loss-free-trained policies, 25% independent loss per directed link | Same procedure using newly trained policies |
| Python | 3.12.13 documented in validation | 3.12.16 recorded in saved configs |
| Core versions | PyTorch 2.14.1+cpu, NumPy 2.5.3, MPE2 1.1.0 documented | Same recorded versions |

### Why the new baseline lineage is shorter

The runner followed the published commands, including resuming `baseline-initial/best.pt` and leaving early stopping enabled. Initial training completed all 8,320,000 steps, but its selected best checkpoint was at 7,040,000. Refinement resumed that checkpoint, reached 81% validation success after one 6,400-step update, and stopped at 7,046,400 rather than the requested cumulative limit of 8,640,000.

Actual baseline training work was 8,326,400 world steps (8,320,000 initial plus 6,400 refinement); the evaluated checkpoint lineage was 7,046,400. Work spent after the initial selected checkpoint is not part of that lineage. This is a documented-command rerun with different checkpoint selection and early-stop behavior, not an exact recreation of the retained historical baseline. Matching validation scores did not produce matching held-out scores.

Spaced-navigation selection compared validation success and then coverage across both phases, as documented. Selected checkpoint steps can be below the full completed budget:

| Policy | Selected phase | Selected lineage steps | Completed training work | Validation success / coverage |
| --- | --- | ---: | ---: | --- |
| full | refine | 8,640,000 | 8,960,000 | 91.0% / 95.7% |
| none | refine | 8,960,000 | 8,960,000 | 87.0% / 94.3% |
| fixed_5 | initial | 8,320,000 | 8,960,000 | 93.0% / 97.3% |

## Unconstrained baseline: single-policy ablations

| Schedule | Historical success (95% interval) | New success (95% interval) | Delta (pp) |
| --- | --- | --- | ---: |
| full | 76.5% (70.2% to 81.8%) | 69.5% (62.8% to 75.5%) | -7.0 |
| fixed_2 | 75.0% (68.6% to 80.5%) | 67.5% (60.7% to 73.6%) | -7.5 |
| fixed_5 | 78.0% (71.8% to 83.2%) | 60.5% (53.6% to 67.0%) | -17.5 |
| fixed_10 | 68.0% (61.2% to 74.1%) | 71.0% (64.4% to 76.8%) | +3.0 |
| none | 43.5% (36.8% to 50.4%) | 27.5% (21.8% to 34.1%) | -16.0 |
| random_full | 0.0% (0.0% to 1.9%) | 0.0% (0.0% to 1.9%) | +0.0 |

Each metric cell below is **historical -> new (delta)**. Coverage and collision deltas are pp.

| Schedule | Final coverage | Task return | Collision step rate | Attempted packets/episode (historical -> new) |
| --- | --- | --- | --- | --- |
| full | 87.2% -> 85.3% (-1.87) | -35.11 -> -34.27 (+0.84) | 10.60% -> 9.72% (-0.88) | 600 -> 600 |
| fixed_2 | 86.3% -> 84.7% (-1.63) | -35.13 -> -35.07 (+0.06) | 11.10% -> 9.82% (-1.28) | 300 -> 300 |
| fixed_5 | 87.8% -> 83.2% (-4.63) | -35.03 -> -34.15 (+0.88) | 11.00% -> 9.41% (-1.59) | 120 -> 120 |
| fixed_10 | 83.8% -> 84.8% (+1.03) | -37.30 -> -36.06 (+1.24) | 12.30% -> 10.43% (-1.87) | 60 -> 60 |
| none | 71.0% -> 61.8% (-9.17) | -41.74 -> -41.75 (-0.01) | 13.20% -> 12.37% (-0.83) | 0 -> 0 |
| random_full | 0.8% -> 0.8% (+0.03) | -137.24 -> -137.24 (-0.00) | 2.10% -> 2.08% (-0.02) | 600 -> 600 |

The new learned policy is less successful in four of six evaluation conditions; fixed_10 improves by 3.0 pp, while the random reference remains at 0%. Returns improve in several conditions despite lower final-step success, because cumulative navigation reward and terminal simultaneous coverage measure different behavior. The random reference closely matches the historical rounded values, supporting consistent evaluation/accounting without establishing learned-policy equivalence.

## Spaced navigation: no packet loss

| Schedule | Historical success (95% interval) | New success (95% interval) | Delta (pp) |
| --- | --- | --- | ---: |
| full | 90.5% (85.6% to 93.8%) | 92.0% (87.4% to 95.0%) | +1.5 |
| none | 81.5% (75.5% to 86.3%) | 87.5% (82.2% to 91.4%) | +6.0 |
| fixed_5 | 91.5% (86.8% to 94.6%) | 85.5% (80.0% to 89.7%) | -6.0 |

Each metric cell below is **historical -> new (delta)**. Coverage and collision deltas are pp.

| Schedule | Final coverage | Task return | Collision step rate | Attempted packets/episode (historical -> new) |
| --- | --- | --- | --- | --- |
| full | 95.7% -> 96.0% (+0.30) | -29.41 -> -27.77 (+1.64) | 5.50% -> 4.72% (-0.78) | 600 -> 600 |
| none | 92.7% -> 94.2% (+1.47) | -32.39 -> -31.55 (+0.84) | 5.50% -> 6.20% (+0.70) | 0 -> 0 |
| fixed_5 | 95.8% -> 93.3% (-2.47) | -27.50 -> -29.89 (-2.39) | 4.60% -> 4.93% (+0.33) | 120 -> 120 |

Full communication now has the highest observed success (92.0%), followed by no communication (87.5%) and fixed_5 (85.5%). Historically fixed_5 slightly exceeded full communication. These changes do not establish a general ranking across training seeds.

## Spaced navigation: 25% packet loss

| Schedule | Historical success (95% interval) | New success (95% interval) | Delta (pp) |
| --- | --- | --- | ---: |
| full | 90.5% (85.6% to 93.8%) | 90.5% (85.6% to 93.8%) | +0.0 |
| none | 81.5% (75.5% to 86.3%) | 87.5% (82.2% to 91.4%) | +6.0 |
| fixed_5 | 91.5% (86.8% to 94.6%) | 85.5% (80.0% to 89.7%) | -6.0 |

Each metric cell below is **historical -> new (delta)**. Coverage and collision deltas are pp.

| Schedule | Final coverage | Task return | Collision step rate | Attempted packets/episode (historical -> new) |
| --- | --- | --- | --- | --- |
| full | 94.3% -> 96.0% (+1.70) | -30.16 -> -29.17 (+0.99) | 5.90% -> 5.35% (-0.55) | 600 -> 600 |
| none | 92.7% -> 94.2% (+1.47) | -32.39 -> -31.55 (+0.84) | 5.50% -> 6.20% (+0.70) | 0 -> 0 |
| fixed_5 | 95.8% -> 93.5% (-2.30) | -28.45 -> -29.50 (-1.05) | 5.20% -> 4.61% (-0.59) | 120 -> 120 |

## Effect of packet loss within each run

This compares each loss-tested policy with its own loss-free checkpoint evaluation on the same seeds; it separates loss effects from historical-versus-new training differences.

| Schedule | Historical success: no loss -> 25% loss | Historical delta (pp) | New success: no loss -> 25% loss | New delta (pp) |
| --- | --- | ---: | --- | ---: |
| full | 90.5% -> 90.5% | +0.0 | 92.0% -> 90.5% | -1.5 |
| none | 81.5% -> 81.5% | +0.0 | 87.5% -> 87.5% | +0.0 |
| fixed_5 | 91.5% -> 91.5% | +0.0 | 85.5% -> 85.5% | +0.0 |

| Schedule | Historical success -> failure | Historical failure -> success | New success -> failure | New failure -> success |
| --- | ---: | ---: | ---: | ---: |
| full | 11 | 11 | 14 | 11 |
| none | 0 | 0 | 0 | 0 |
| fixed_5 | 12 | 12 | 21 | 21 |

In the new run, full communication loses 1.5 pp success, its return worsens from -27.77 to -29.17, and its collision step rate rises from 4.72% to 5.35%. Fixed_5 retains the same aggregate success, but 42 episode outcomes change; its return improves from -29.89 to -29.50 and collision rate falls from 4.93% to 4.61%. This differs from the historical observation that both communicating policies had worse returns and more collisions under loss. No-communication metrics remain exactly unchanged.

| Schedule | Historical delivered packets/episode | New delivered packets/episode | Historical observed loss | New observed loss |
| --- | ---: | ---: | --- | --- |
| full | 449.55 | 449.55 | 25.08% | 25.08% |
| none | 0.00 | 0.00 | Undefined (no attempts) | Undefined (no attempts) |
| fixed_5 | 90.19 | 90.19 | 24.84% | 24.84% |

Packets carry 16 payload bytes. Attempted traffic remains 600 packets/9,600 bytes for full communication, 120 packets/1,920 bytes for fixed_5, and zero for none. Failed packets still count toward attempted traffic. Displayed delivery means agree with historical rounding.

## Verification and limits

- The test run passed 25 tests, with two upstream environment deprecation warnings.
- The completed artifacts contain 1,200 baseline episodes, 600 spaced-navigation episodes, and 600 loss-study episodes. Each condition has 200 episodes.
- The spaced and loss-study generators check landmark spacing, paired layouts, packet accounting, and replay agreement with episode metrics. The loss study also checks checkpoint hashes and exact no-communication equivalence. Completion means these built-in assertions passed.
- Both studies use one training seed per schedule. Episode intervals do not quantify training-seed variability. Differences here are descriptive; no statistical significance claim is made.
- The same held-out seed ranges allow matched scenario comparisons, but historical per-episode records are absent. Historical-versus-new paired outcome tests cannot be reconstructed from rounded aggregate reports.
- Platform/kernel differences and the Python patch-version change are possible sources of numerical variation; this run does not isolate their contribution. The shorter baseline lineage is an observed procedural difference.
- Unconstrained baseline scores and spaced-navigation scores use different layouts, test seeds, and training designs. Their differences cannot be attributed to spacing or communication alone.
- Communication timing remains externally fixed. These results do not test a learned adaptive scheduler, training with packet loss, or transfer to larger teams.

## Artifacts and provenance

| Artifact | New run |
| --- | --- |
| Runner and exact invoked stages | [run.py](../results/reproduction-20261028/run.py) |
| Final status | [status.json](../results/reproduction-20261028/status.json) |
| Baseline summary / episodes | [summary](../results/reproduction-20261028/baseline-evaluation/summary.json) / [episodes](../results/reproduction-20261028/baseline-evaluation/episodes.jsonl) |
| Spaced summary / episodes | [summary](../results/reproduction-20261028/spaced-navigation/summary.json) / [episodes](../results/reproduction-20261028/spaced-navigation/episodes.jsonl) |
| Loss summary / episodes | [summary](../results/reproduction-20261028/spaced-navigation-loss25/summary.json) / [episodes](../results/reproduction-20261028/spaced-navigation-loss25/episodes.jsonl) |
| Baseline replay (seed 300000) | [baseline.html](../results/reproduction-20261028/baseline.html) |
| Baseline replay (seed 300001) | [baseline-seed300001.html](../results/reproduction-20261028/baseline-seed300001.html) |
| Spaced replay (first held-out seed) | [comparison.html](../results/reproduction-20261028/spaced-navigation/comparison.html) |
| Spaced replay (first all-success seed) | [all-success.html](../results/reproduction-20261028/spaced-navigation/all-success.html) |
| Loss replay (first held-out seed) | [comparison.html](../results/reproduction-20261028/spaced-navigation-loss25/comparison.html) |
| Loss replay (selected loss-affected seed) | [loss-affected.html](../results/reproduction-20261028/spaced-navigation-loss25/loss-affected.html) |

Seed 300001 is preserved as a matching historical scenario; the new filename does not assert that the newly trained policy succeeds on it. First-scenario replays and deliberately selected all-success/loss-affected examples are illustrative, not aggregate evidence.

Selected checkpoint SHA-256 hashes:

| Checkpoint | SHA-256 |
| --- | --- |
| Baseline | `072198d4edcb5560ded11987dcd102773d41f6c371049672994226fc25769655` |
| Spaced full (also used at 25% loss) | `f9278fac17b0acfc1e4fedc50f4fff593c87588c95e69ed59998b0d508a16e12` |
| Spaced none (also used at 25% loss) | `833e03e8a56f1c35fbb03c61f21613341fc77348efe7102c31bfd76671becffd` |
| Spaced fixed_5 (also used at 25% loss) | `5a1a7ca5df76de54b0d25a1cff2c297412cb6ca3066acfb1d5ec9503155f4a74` |

This comparison file is stored in `docs/` for versioning. The linked `results/` files are local, Git-ignored artifacts; preserve or distribute that directory separately to retain the underlying evidence.
