# Exploratory landmark and coverage communication within a 1.0-unit radius

**Status: exploratory — `landmark-radius-v1`.** This experimental extension
includes landmark-location sharing, timestamped coverage reports, radius-limited
delivery and the expanded actor observation. Unit tests and short full/adaptive
smoke runs check implementation correctness only. Its navigation performance,
communication benefit and training-seed robustness have not been established by
a full study. It is not the protocol used in the completed baseline,
scheduler-loss or larger-team results.

Future runs using this mode carry `communication_status: exploratory` in their
training/checkpoint configuration. This label does not change the current mode
selection, transmit decisions, reward coefficients or environment behavior.

The new `landmark` communication mode implements `landmark-radius-v1` in both
the PettingZoo/MPE2 adapter and the batched training environment. Select it with
`--communication-mode landmark`; fresh CLI training now defaults to that mode,
and the communication radius defaults to 1.0. Low-level environment constructors
retain their legacy default for existing callers; pass `communication_mode="landmark"`
to opt in directly. Historical study launchers explicitly select legacy mode,
and existing checkpoint evaluations retain their saved protocol.

A directed packet is eligible only when the sender and receiver centers are at
most 1.0 unit apart radially **at send time**, before movement. There are no
self-links. The adaptive, full, fixed_5 and none transmit decisions still apply:
discovery and coverage changes do not force additional broadcasts. Out-of-range
agents do not count as attempted directed packets. An in-range attempted packet
can still be dropped at the configured loss rate, and dropped attempts are charged.

Each drone remembers the locations of landmarks it has directly observed during
the current episode. Discovery follows the existing local observation: the nearest
three landmarks within sensing radius 0.6. Hidden coordinates are never inserted
into its memory or packets. Static landmark locations remain remembered when the
drone moves away. Every transmitted packet contains all of that sender's directly
discovered landmark records, not just the landmarks currently in sensor range.
Memory is cleared at episode reset, including partial collision-world resets and
curriculum resets. Received discoveries are not automatically relayed; a drone
must directly observe a landmark before including it in its own discovery packets.

The decoded payload contains:

- Sender position and velocity: four float32 values.
- Whether the sender covers any known landmark at send time.
- Send-step timestamp and landmark-record count.
- For each discovered landmark: its ID, absolute x/y coordinates, and whether
  **that sender** is covering it at send time (distance <=0.1).

The fixed header is 28 bytes and each landmark record is 16 bytes, excluding
network headers. A packet with M discovered landmarks therefore contains
`28 + 16*M` payload bytes. Payload cost is charged separately for every attempted
in-range directed recipient, including dropped packets. A send gate with no
in-range recipients produces zero directed attempts.

Delivery remains one simulation step later. Messages stay cached for five steps;
a missing/dropped packet does not refresh them. Coverage is a timestamped report,
not live global truth: a sender may move after sending. A new report from that
sender updates its coverage flag; its old flag otherwise expires with the packet.
Another sender reporting that it is not covering a landmark does not cancel a
still-valid positive coverage report from a different sender.

The actor observation grows from 28 to **44 values** (with three local slots):

| Indices | Meaning |
| --- | --- |
| 0-3 | Self position and velocity |
| 4-12 | Existing local peer slots |
| 13-21 | Existing local landmark slots |
| 22-27 | Existing pooled sender kinematics, cache fraction and cache age |
| 28-42 | Three nearest reported landmarks: relative x/y, known mask, reported-covered flag, report age / TTL |
| 43 | Whether this drone currently covers a directly known landmark |

For a positive reported-covered flag, the age belongs to the newest valid positive
report. Otherwise it belongs to the freshest location report. False means there
is no valid positive report, not that the environment has certified vacancy.
Packets retain all sender discoveries even though the fixed-width actor inbox
shows only the three nearest reported landmarks. Actor input width stays fixed
when evaluating larger teams. The three-agent centralized training state is 81
values; it remains privileged and is not used by actors during inference.

The global full-traffic reference remains N*(N-1)*horizon for historical traffic
ratios and the existing training constraint. Within this radius-limited protocol,
full communication may be below that reference. Evaluation additionally records
`eligible_links` and `eligible_link_attempt_ratio`, so actual range-limited
communication opportunities are distinguishable from the global reference.

Existing 28-input checkpoints cannot use the new information without retraining.
The trainer rejects incompatible warm starts and resumes instead of silently
changing observation semantics. The evaluator and replay infer landmark mode
from new 44-input checkpoints; saved configuration preserves the protocol and
radius. Historical evaluation snapshots, trained policies, reports and replays
remain legacy results and have not been retrained or relabeled.

Example training from scratch:

```powershell
.\.venv\Scripts\python.exe -m limited_comm.train --communication-mode landmark --communication-radius 1.0 --recurrent --hidden 64 --schedule full --min-agent-distance 0.4 --min-landmark-distance 0.4 --output results/landmark-full
```

Adaptive warm starts must use a compatible **landmark-mode** fixed-schedule
checkpoint. Both training and original-environment validation/confirmation use
the same protocol and radius. This implementation does not start a new 15-seed
training study automatically.
