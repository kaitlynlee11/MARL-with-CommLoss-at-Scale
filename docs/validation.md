# Scaffold verification

Verified October 5, 2026 on Windows with Python 3.12.13, MPE2 1.1.0,
PettingZoo 1.27.0, Gymnasium 1.3.0 and NumPy 2.5.3. Dependencies are recorded
in `uv.lock`.

`python -m pytest -q`: **11 passed**. MPE2's legacy environment factory emitted
two upstream deprecation warnings; the pinned version passed the Parallel API
test. Tests cover full/no delivery, charged failed packets, cache expiration,
message age, reproducibility, local observation isolation, reset behavior,
coverage/collision definitions and paired baseline accounting.

```powershell
python -m limited_comm.experiment --agents 3 5 8 --loss 0 0.1 0.3 --episodes 3 --horizon 100 --output results/first-smoke.jsonl
```

Completed **81 episodes**: 3 sizes × 3 losses × 3 schedules × 3 scenario seeds.
Local artifacts: `results/first-smoke.jsonl` and
`results/first-smoke.metadata.json`. Results are ignored by version control.

These runs validate execution and telemetry with random navigation. They do not
measure learning, convergence, adaptive communication or trained-policy quality.
The next milestone is shared-actor MAPPO navigation training at N=3.
