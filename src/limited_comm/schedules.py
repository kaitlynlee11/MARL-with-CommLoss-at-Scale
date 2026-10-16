"""External gates for the independently trained communication baselines."""


def transmit(schedule, step, period=5):
    if schedule not in ("full", "none", "fixed") or period < 1:
        raise ValueError("Require full/none/fixed schedule and a positive period")
    return int(schedule == "full" or (schedule == "fixed" and step % period == 0))
