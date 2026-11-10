# Live baseline progress

Open [the local dashboard](http://127.0.0.1:8765/progress.html) while the local
server is running. The dashboard refreshes every five seconds and reads training
logs directly, including partial runs that the study coordinator still marks
pending. It does not change training or evaluation artifacts.

To start it again from the repository directory:

```powershell
.\.venv\Scripts\python.exe scripts/serve_progress.py
```

The page source is `progress.html`; the server is `scripts/serve_progress.py`.
The server listens only on this computer's loopback address. Leave it running
while viewing progress; Ctrl+C stops a foreground server.

The grid covers training seeds 42-56 and all three communication schedules.
All 45 policies are retrained from scratch with minimum pairwise radial center
separation of 0.4 units for both agent starts and landmarks. This applies to
training, curriculum, validation and evaluation resets. Agents may approach each
other after reset. Earlier unrestricted-agent-start runs are excluded.
Artifacts are under `results/baseline-agent-spacing04-15-seeds-20261109/`.
Each policy has an initial budget of
8,320,000 world steps, followed by 640,000 refinement steps and 200 held-out
evaluation episodes. Training progress totals 403,200,000 world steps; evaluation
totals 9,000 episodes. A run becomes complete only when the measured-study
coordinator has verified it. Historical rounded seed-42 scores are excluded.

Click a run for phase progress, validation metrics, held-out results and its
artifact directory. The filter selects seeds with active, completed, queued or
potentially stalled runs. Warnings identify failures and active runs without a
log update for three minutes; a warning does not establish that a process died.

If disconnected, the page retains the last received data and marks it as stale.
Schedule success means are provisional until all 15 training seeds are verified.
