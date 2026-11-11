# Communication-loss tests of frozen schedulers

The test compares the selected adaptive, full-communication and fixed-five-step
policies for every training seed 42-56 at configured losses of 10%, 25% and 50%.
No policy is retrained, reselected or given an additional observation. The rates
are known to the evaluator; the existing actors do not have a loss-rate input.

Every condition uses scenarios 400000-400199 from the original baseline adapter,
for 27,000 episodes in total. Agent starts and static landmarks retain minimum
pairwise radial center spacing of 0.4. Episodes last 100 steps and do not
terminate on collision. Success requires simultaneous full landmark coverage
at the final step; coverage is the final fraction of landmarks covered.

Each attempted directed packet is independently lost at the configured rate.
Channel draws are indexed by scenario seed, step, sender and receiver, including
silent senders, so policies share channel randomness. The same draws pair all
three rates. Checkpoint hashes are verified before and after evaluation; the
simulator and base task-return code are copied from the corrected frozen
baseline source with verified hashes. Adaptive checkpoints are the previously
selected policies; no new held-out outcomes select checkpoints.

Measurements include:

- Success rate and final coverage percentage.
- Collision step rate: fraction of the 100 steps with any overlapping agent pair.
- Collision episode fraction and mean overlapping-pair counts per episode.
- Successfully delivered packets per episode (“completed packets”).
- Attempted packets, dropped packets and realized loss fraction.

Per-seed summaries include episode-level Wilson success intervals. The overall
comparison gives each training seed equal weight and reports seed SDs and
95% percentile bootstrap intervals (20,000 resamples). Paired changes compare
each condition with that policy's saved zero-loss measurements on the same
scenarios. These seed intervals do not quantify independent channel realizations
or support a claim that a small improvement under sampled loss is general.

Silent policies can be insensitive to loss. Where a saved zero-loss episode has
no attempted packets, its deterministic trajectory must match the corresponding
loss-tested episode; this property is verified during evaluation. Low traffic
therefore should not be mistaken for successful adaptation to an observed rate.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_scheduler_losses.py --workers 6
```

Outputs are under `results/scheduler-loss-comparison-20261110/`. The
[comparison report](scheduler-loss-comparison-results.md) updates as complete
training-seed jobs finish. Individual results are in `per-seed.csv`; every
episode is recorded in `seed-N/episodes.jsonl` and concatenated into
`episodes.jsonl` on completion. Checkpoints and settings are in `manifest.json`.
The [live local tracker](http://127.0.0.1:8765/loss/) refreshes every five seconds.

After completion, `scripts/verify_scheduler_losses.py` independently recomputes
summaries, checks checkpoint/source hashes, and verifies exact delivered-packet
counts for all fixed schedules directly from the channel RNG. The audit is saved
as `verification.json`. `scripts/render_loss_comparison.py` exports a standalone
`comparison.html` with selectable policy curves, all requested metrics, original
zero-loss references and 95% training-seed interval tooltips.
