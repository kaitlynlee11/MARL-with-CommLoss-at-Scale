import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("scheduler_losses", Path(__file__).resolve().parents[1] / "scripts/evaluate_scheduler_losses.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def sample_row():
    return {"seed": 400000, "loss": .25, "actual_initial_agent_separation": .45,
            "actual_min_separation": .5, "initial_collision": False, "active_steps": 100,
            "terminate_on_collision": False, "delivered_links": 90, "attempted_links": 120,
            "task_reward_version": "legacy-distance-v0", "final_success": True,
            "final_coverage": 1., "collision_step_rate": .03, "task_return": -20}


def test_loss_test_preserves_fixed_attempt_budget_and_counts_delivery():
    row = sample_row()
    study.check_episode(row, {**row, "loss": 0}, "fixed_5", .25)
    row["delivered_links"] = 121
    with pytest.raises(AssertionError):
        study.check_episode(row, row, "fixed_5", .25)


def test_silent_policy_cannot_change_trajectory_due_to_packet_drops():
    row = {**sample_row(), "attempted_links": 0, "delivered_links": 0}
    original = {**row, "loss": 0}
    study.check_episode(row, original, "adaptive", .25)
    row["final_success"] = False
    with pytest.raises(AssertionError):
        study.check_episode(row, original, "adaptive", .25)


def test_summary_separates_steps_collision_episodes_and_packets():
    row = {**sample_row(), "first_success_step": 80, "mean_landmark_distance": .08,
           "payload_bytes": 1920, "any_collision": True, "collision_pairs": 4}
    other = {**row, "any_collision": False, "collision_pairs": 0, "collision_step_rate": 0}
    summary = study.summarize_rows([row, other])
    assert summary["collision_episode_fraction"] == .5
    assert summary["collision_step_rate"] == pytest.approx(.015)
    assert summary["delivered_links"] == 90
    assert summary["lost_links"] == 30


def test_empty_report_does_not_claim_complete(tmp_path):
    study.aggregate(tmp_path, [], 15, smoke=True)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["complete"] and len(result["conditions"]) == 9
