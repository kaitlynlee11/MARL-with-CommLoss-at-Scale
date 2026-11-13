import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("adaptive_ablation", Path(__file__).resolve().parents[1] / "scripts/evaluate_adaptive_ablation.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_intervention_preserves_joint_argmax_moves_and_advances_memory_once():
    class Actor:
        spec = {"adaptive": True}
        calls = 0
        def reset_memory(self):
            self.calls = 0
        def actions(self, obs):
            self.calls += 1
            return np.array([4, 1, 0]), np.array([1, 0, 1])
    actor = Actor()
    silent = study.ForceSilent(actor)
    moves, gates = silent.actions(None)
    np.testing.assert_array_equal(moves, [4, 1, 0])
    np.testing.assert_array_equal(gates, [0, 0, 0])
    assert actor.calls == 1 and silent.suppressed_gates == 2
    silent.reset_memory()
    assert actor.calls == 0 and silent.suppressed_gates == 0


def test_original_reproduction_rejects_different_outcomes():
    study.check_original({"task_return": -20., "final_success": True},
                         {"condition": "adaptive", "task_return": -20., "final_success": True})
    with pytest.raises(AssertionError):
        study.check_original({"final_success": False}, {"final_success": True})


def test_empty_ablation_is_not_a_completed_comparison(tmp_path):
    study.aggregate(tmp_path, [], [42, 43], smoke=True)
    import json
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["complete"]
    assert result["statistics"]["differences_disabled_minus_original"]["success_rate"]["mean"] is None
