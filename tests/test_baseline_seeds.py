"""Analysis must use training replicates, not pool episodes as replicates."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "baseline_seeds", Path(__file__).resolve().parents[1] / "scripts/run_baseline_seeds.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_published_seed_is_retained_without_substituting_reproduction():
    rows = study.historical_rows()
    assert [r["success_rate"] for r in rows] == [.905, .815, .915]
    assert all(r["training_seed"] == 42 for r in rows)
    assert all("rounded" in r["provenance"] for r in rows)


def test_seed_summary_weights_training_replicates_equally(tmp_path):
    rows = study.historical_rows()
    new = []
    for row in rows:
        new.append({**row, "training_seed": 43, "episodes": 10,
                    "success_rate": 0., "provenance": "test fixture"})
    assert study.report(tmp_path, rows + new, expected=2, smoke=True)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["complete"] and result["smoke_only"]
    full = result["conditions"]["full"]
    assert full["training_seeds"] == 2
    assert full["metrics"]["success_rate"]["mean"] == pytest.approx(.905 / 2)
    assert full["metrics"]["success_rate"]["sample_sd"] == pytest.approx(.905 / 2**.5)
    assert len(result["per_seed"]) == 6


def test_partial_study_is_not_reported_as_fifteen_completed_seeds(tmp_path):
    assert not study.report(tmp_path, study.historical_rows(), expected=15, smoke=True)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["complete"]
    assert result["completed_condition_runs"] == 3
    assert "in progress" in (tmp_path / "report.md").read_text()
    assert result["conditions"]["full"]["metrics"]["success_rate"]["sample_sd"] is None


def test_measured_study_excludes_published_estimates(tmp_path):
    with pytest.raises(ValueError, match="rounded"):
        study.report(tmp_path, study.historical_rows(), expected=15,
                     smoke=True, measured_only=True)


def test_empty_measured_study_reports_pending_without_historical_placeholders(tmp_path):
    assert not study.report(tmp_path, [], expected=15, smoke=True, measured_only=True)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["per_seed"] == [] and result["completed_condition_runs"] == 0
    assert result["measured_only"]
    assert result["conditions"]["full"]["metrics"]["success_rate"]["mean"] is None


def test_same_seed_cannot_be_counted_twice(tmp_path):
    rows = study.historical_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        study.report(tmp_path, rows + rows, expected=2, smoke=True)
