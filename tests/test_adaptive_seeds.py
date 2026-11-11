"""Replicate selection and reports must preserve the strict target semantics."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "adaptive_seeds", Path(__file__).resolve().parents[1] / "scripts/run_adaptive_seeds.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.mark.parametrize("success,packets,expected", [(.9, 100, False), (.905, 120, False),
                                                     (.905, 119.99, True), (.8, 0, False)])
def test_strict_targets(success, packets, expected):
    assert study.targets_met(success, packets) == expected


def test_training_matches_environment_and_uses_validation_only(tmp_path):
    command = study.training_command(55, tmp_path, tmp_path / "init.pt", 6400, 200)
    def arg(name):
        return command[command.index(name) + 1]
    assert arg("--seed") == "55"
    assert arg("--loss") == "0"
    assert arg("--min-agent-distance") == arg("--min-landmark-distance") == ".4"
    assert arg("--target-success") == ".9" and arg("--packet-ratio-target") == ".2"
    assert "--terminate-on-collision" not in command
    assert "400000" not in command


def test_empty_aggregate_does_not_claim_success(tmp_path):
    study.aggregate(tmp_path, [], [42, 43], smoke=True)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["complete"] and not result["aggregate_targets_met"]
    assert result["metrics"]["success_rate"]["mean"] is None


def test_duplicate_seed_cannot_count_as_an_independent_replicate(tmp_path):
    with pytest.raises(ValueError, match="repeated"):
        study.aggregate(tmp_path, [{"training_seed": 42}] * 2, [42, 43], smoke=True)


def test_analysis_uses_seed_sample_variation():
    summary = study.seed_stats([.8, 1.])
    assert summary["mean"] == pytest.approx(.9)
    assert summary["sample_sd"] == pytest.approx(.2 / 2 ** .5)
    assert summary["seed_bootstrap_95pct"] == pytest.approx([.8, 1.])


def test_atomic_write_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    actual = Path.replace
    attempts = []
    def locked_once(path, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError("Windows reader holds destination")
        return actual(path, target)
    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr(study.time, "sleep", lambda _: None)
    study.write_json(tmp_path / "progress.json", {"complete": True})
    assert len(attempts) == 2
    assert json.loads((tmp_path / "progress.json").read_text()) == {"complete": True}


def test_progress_failure_does_not_stop_training(tmp_path, monkeypatch):
    def locked(*args):
        raise PermissionError("Persistent reader lock")
    monkeypatch.setattr(study, "_progress_snapshot", locked)
    assert study.progress_snapshot(tmp_path, [42], {"42": "training"}, 6400, 0) is False
