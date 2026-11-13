import ast
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("warm_no_comm", Path(__file__).resolve().parents[1] / "scripts/train_warm_no_comm.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_budget_plan_uses_completed_work_not_adaptive_ceiling():
    plan = study.budget_plan(42)
    assert plan["additional_world_steps"] == 12320000
    assert plan["ppo_updates"] == 1925 and plan["optimizer_steps"] == 23100
    assert plan["total_training_work"] == 8960000 + 12320000


def test_command_enforces_silent_fixed_budget_and_adaptive_hyperparameters(tmp_path):
    plan = study.budget_plan(42)
    cmd, steps = study.command(tmp_path, plan)
    def arg(name):
        return cmd[cmd.index(name) + 1]
    assert "--no-early-stop" in cmd
    assert arg("--schedule") == "none" and int(arg("--steps")) == plan["additional_world_steps"]
    assert arg("--initialize-actor") == plan["initializer_path"]
    assert arg("--epochs") == "3" and arg("--sequence-length") == "25"
    assert arg("--minibatch") == "2048" and arg("--loss") == "0.0"
    assert arg("--coverage-bonus") == "1.0" and arg("--success-bonus") == "10.0"


def test_control_patch_is_guarded_and_changes_only_training_module():
    original = (study.ADAPTIVE / "source/limited_comm/train.py").read_text(encoding="utf-8")
    patched = study.patch_trainer(original)
    ast.parse(patched)
    assert 'model.actor = parent.actor' in patched
    assert 'model.spec["adaptive"] = False' in patched
    assert 'args.output / "initial.pt"' in patched
    with pytest.raises(ValueError, match="patch anchor"):
        study.patch_trainer(original.replace('if args.initialize_actor and (not adaptive or args.resume):', '# changed'))
