import numpy as np
import pytest

torch = pytest.importorskip("torch")

from limited_comm.env import CommunicationSpread
from limited_comm.mappo import MAPPO, gae, load_policy
from limited_comm.vector_env import VectorSpread
from limited_comm.layout import minimum_separation


def test_gae_bootstraps_time_limits_without_crossing_resets():
    rewards = np.array([[1.0], [100.0]], dtype=np.float32)
    values = np.zeros_like(rewards)
    next_values = np.array([[2.0], [0.0]], dtype=np.float32)
    boundaries = np.ones_like(rewards)
    advantages, returns = gae(rewards, values, next_values, boundaries, gamma=0.9)
    assert advantages[0, 0] == pytest.approx(2.8)
    assert returns[1, 0] == 100
    advantages, _ = gae(rewards, values, next_values, boundaries, terminals=boundaries, gamma=0.9)
    assert advantages[0, 0] == 1


@pytest.mark.parametrize("schedule", ["full", "fixed", "none"])
def test_vector_matches_original_physics_reward_observation_and_cache(schedule):
    env = CommunicationSpread(horizon=30, comm_cost=0.0)
    vector = VectorSpread(batch=1, horizon=30, seed=4)
    rng = np.random.default_rng(14)
    try:
        obs, _ = env.reset(seed=4)
        vector.pos[0] = np.array([a.state.p_pos for a in env.world.agents])
        vector.vel[0] = np.array([a.state.p_vel for a in env.world.agents])
        vector.landmarks[0] = np.array([l.state.p_pos for l in env.world.landmarks])
        np.testing.assert_allclose(vector.observe()[0], np.stack(list(obs.values())), atol=1e-6)
        for t in range(30):
            moves = rng.integers(5, size=3)
            gate = int(schedule == "full" or (schedule == "fixed" and t % 5 == 0))
            obs, rewards, _, truncated, _ = env.step({a: np.array([moves[i], gate]) for i, a in enumerate(env.possible_agents)})
            vobs, _, reward, boundary, diagnostics = vector.step(moves[None], np.full((1, 3), gate))
            np.testing.assert_allclose(vector.pos[0], [a.state.p_pos for a in env.world.agents], atol=1e-9)
            np.testing.assert_allclose(vobs[0], np.stack(list(obs.values())), atol=1e-6)
            np.testing.assert_allclose(reward[0], list(rewards.values())[0], atol=1e-6)
            assert boundary == all(truncated.values())
            assert diagnostics["success"][0] == env.diagnostics()["success"]
    finally:
        env.close()


def test_checkpoint_roundtrip_preserves_actor(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = MAPPO(hidden=32)
    obs = np.random.default_rng(1).normal(size=(3, 28)).astype(np.float32)
    path = tmp_path / "policy.pt"
    torch.save({"model_spec": model.spec, "model": model.state_dict()}, path)
    restored, _ = load_policy(path)
    np.testing.assert_array_equal(model.moves(obs), restored.moves(obs))
    # Actor accepts only local observations; global state has a separate critic path.
    with pytest.raises(RuntimeError):
        model.actor(torch.zeros(3, 36))


def test_popart_preserves_unnormalized_predictions():
    torch.set_num_threads(1)
    model = MAPPO(hidden=32)
    states = torch.randn(12, 36)
    before = model.value(states).detach().clone()
    targets = torch.linspace(-50, -5, 12)
    normalized = model.normalize_value_targets(targets)
    torch.testing.assert_close(model.value(states), before, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(normalized * model.value_std + model.value_mean, targets)


def test_recurrent_history_and_reset():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = MAPPO(hidden=32, recurrent=True)
    obs = torch.randn(3, 28)
    first, hidden = model.policy(obs)
    second, _ = model.policy(obs, hidden)
    assert not torch.equal(first.logits, second.logits)
    # Replay inference must clear history at each new scenario/schedule.
    model.moves(obs.numpy())
    model.reset_memory()
    model.moves(obs.numpy())
    torch.testing.assert_close(model._memory, hidden)


def test_recurrent_sequence_recomputes_on_policy_log_probabilities():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = MAPPO(hidden=32, recurrent=True)
    observations = torch.randn(8, 2, 3, 28)
    memory = torch.zeros(2, 3, 32)
    history, moves, logps = [], [], []
    with torch.no_grad():
        for t in range(8):
            history.append(memory.clone())
            dist, memory = model.policy(observations[t], memory)
            action = dist.sample()
            moves.append(action)
            logps.append(dist.log_prob(action))
            if t == 3:
                memory.zero_()
    hidden = history[2]
    for t in range(2, 7):
        dist, hidden = model.policy(observations[t], hidden)
        torch.testing.assert_close(dist.log_prob(moves[t]), logps[t])
        if t == 3:
            hidden = hidden * 0


def test_assignment_potential_is_invariant_to_agent_permutation():
    vector = VectorSpread(batch=4)
    before = vector.potential()
    vector.pos = vector.pos[:, [2, 0, 1]]
    np.testing.assert_allclose(vector.potential(), before)


def test_vector_spacing_and_schedules_preserve_paired_worlds():
    worlds = [VectorSpread(batch=64, seed=7, horizon=10) for _ in range(3)]
    for world in worlds:
        assert (minimum_separation(world.landmarks) >= 0.4).all()
    for _ in range(10):
        for world, gate in zip(worlds, [1, 0, 1]):
            world.step(np.zeros((64, 3), dtype=int), np.full((64, 3), gate))
            assert (minimum_separation(world.landmarks) >= 0.4).all()
        np.testing.assert_array_equal(worlds[0].pos, worlds[1].pos)
        np.testing.assert_array_equal(worlds[0].landmarks, worlds[2].landmarks)


