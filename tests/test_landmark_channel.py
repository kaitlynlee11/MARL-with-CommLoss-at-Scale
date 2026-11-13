import numpy as np
import pytest

from limited_comm.landmark_channel import LandmarkChannel, VectorLandmarkChannel, HEADER_BYTES, LANDMARK_BYTES
from limited_comm.env import CommunicationSpread
from limited_comm.vector_env import VectorSpread
from limited_comm.collision_vector_env import CollisionVectorSpread


LANDMARKS = np.array([[-.5, 0.], [10., 10.], [-10., 10.]])


@pytest.mark.parametrize("inside,outside", [([1., 0.], [1.0001, 0.]), ([.6, .8], [.8, .8])])
def test_radial_boundary_and_packets_only_contain_directly_seen_landmarks(inside, outside):
    channel = LandmarkChannel(3, 0, 5, np.random.default_rng(0))
    positions = np.array([[0., 0.], inside, outside])
    traffic = channel.advance([1, 0, 0], positions, np.zeros((3, 2)), LANDMARKS)
    assert traffic.attempted_links == traffic.delivered_links == 1
    assert traffic.payload_bytes == HEADER_BYTES + LANDMARK_BYTES
    assert channel.packet(2, 0) is None
    packet = channel.packet(1, 0)
    assert [r["landmark_id"] for r in packet["landmarks"]] == [0]
    np.testing.assert_allclose(packet["landmarks"][0]["position"], [-.5, 0])


def test_gate_controls_discovery_sends_memory_persists_and_received_records_are_not_relayed():
    channel = LandmarkChannel(3, 0, 5, np.random.default_rng(1))
    positions = np.array([[0., 0.], [.9, 0.], [1.8, 0.]])
    silent = channel.advance([0, 0, 0], positions, np.zeros((3, 2)), LANDMARKS)
    assert silent.attempted_links == 0 and channel.packet(1, 0) is None
    channel.advance([1, 0, 0], positions, np.zeros((3, 2)), LANDMARKS)
    assert channel.packet(1, 0)["landmarks"][0]["landmark_id"] == 0
    heard = channel.observe_all(positions)[1, 6:].reshape(3, 5)
    np.testing.assert_allclose(heard[0], [-1.4, 0, 1, 0, .2], atol=1e-6)
    assert not channel.impl.known[0, 1, 0]  # hearing does not count as direct sight
    channel.advance([0, 1, 0], positions, np.zeros((3, 2)), LANDMARKS)
    assert channel.packet(2, 1)["landmarks"] == []
    positions[0] = [.3, 0]  # original landmark now outside sensing radius
    channel.advance([1, 0, 0], positions, np.zeros((3, 2)), LANDMARKS)
    assert channel.packet(1, 0)["landmarks"][0]["landmark_id"] == 0


def test_coverage_is_a_timestamped_snapshot_and_updates_to_false_when_sender_leaves():
    channel = LandmarkChannel(3, 0, 5, np.random.default_rng(2))
    positions = np.array([[0., 0.], [.8, 0.], [3., 0.]])
    landmarks = np.array([[0., 0.], [10., 10.], [-10., 10.]])
    channel.advance([1, 0, 0], positions, np.zeros((3, 2)), landmarks)
    packet = channel.packet(1, 0)
    assert packet["sender_covering_any"] and packet["landmarks"][0]["sender_covering"]
    assert packet["sent_step"] == 0 and packet["age_steps"] == 1
    positions[0] = [.4, 0]
    channel.advance([0, 0, 0], positions, np.zeros((3, 2)), landmarks)
    assert channel.packet(1, 0)["sender_covering_any"]  # stale report, not live truth
    assert channel.packet(1, 0)["age_steps"] == 2
    channel.advance([1, 0, 0], positions, np.zeros((3, 2)), landmarks)
    assert not channel.packet(1, 0)["sender_covering_any"]
    assert not channel.packet(1, 0)["landmarks"][0]["sender_covering"]
    for _ in range(5):
        channel.advance([0, 0, 0], positions, np.zeros((3, 2)), landmarks)
    assert channel.packet(1, 0) is None
    assert not channel.observe_all(positions)[1, 6:].reshape(3, 5)[:, 2].any()
    assert channel.impl.known[0, 0, 0]
    channel.impl.reset()
    assert not channel.impl.known.any()


def test_another_sender_negative_does_not_erase_a_valid_positive_coverage_report():
    channel = LandmarkChannel(3, 0, 5, np.random.default_rng(2))
    positions = np.array([[0., 0.], [.3, 0.], [.9, 0.]])
    landmarks = np.array([[0., 0.], [10., 10.], [-10., 10.]])
    channel.advance([1, 1, 0], positions, np.zeros((3, 2)), landmarks)
    channel.advance([0, 1, 0], positions, np.zeros((3, 2)), landmarks)
    report = channel.observe_all(positions)[2, 6:].reshape(3, 5)[0]
    assert report[3] == 1 and report[4] == pytest.approx(2/5)


def test_dropped_packets_are_charged_but_out_of_range_recipients_are_not():
    channel = LandmarkChannel(3, 1, 5, np.random.default_rng(3))
    positions = np.array([[0., 0.], [.8, 0.], [3., 0.]])
    traffic = channel.advance([1, 0, 0], positions, np.zeros((3, 2)), LANDMARKS)
    assert traffic.attempted_links == 1 and traffic.delivered_links == 0
    assert traffic.payload_bytes == HEADER_BYTES + LANDMARK_BYTES
    assert channel.packet(1, 0) is None


def test_partial_reset_clears_only_requested_worlds():
    channel = VectorLandmarkChannel(2, 3, 0, 5, np.random.default_rng(4))
    positions = np.repeat(np.array([[[0., 0.], [.8, 0.], [3., 0.]]]), 2, axis=0)
    channel.advance(np.ones((2, 3)), positions, np.zeros_like(positions), np.repeat(LANDMARKS[None], 2, axis=0))
    known = channel.known[1].copy()
    channel.reset(np.array([True, False]))
    assert not channel.known[0].any() and (channel.age[0] == 6).all()
    np.testing.assert_array_equal(channel.known[1], known)
    assert channel.age[1].min() == 1


@pytest.mark.parametrize("loss", [0., .25, 1.])
def test_adapter_and_batched_training_match_new_protocol(loss):
    env = CommunicationSpread(horizon=15, comm_cost=.02, loss=loss, communication_mode="landmark")
    vector = VectorSpread(batch=1, horizon=15, comm_cost=.02, loss=loss, communication_mode="landmark")
    try:
        env.reset(seed=4)
        vector.pos[0] = np.array([a.state.p_pos for a in env.world.agents])
        vector.vel[0] = 0
        vector.landmarks[0] = np.array([l.state.p_pos for l in env.world.landmarks])
        env.channel.impl.reset()
        vector.landmark_channel.reset()
        env.channel.impl.rng = np.random.default_rng(55)
        vector.landmark_channel.rng = np.random.default_rng(55)
        obs = env._observations()
        np.testing.assert_allclose(vector.observe()[0], np.stack(list(obs.values())), atol=1e-6)
        rng = np.random.default_rng(14)
        for _ in range(15):
            moves, gates = rng.integers(5, size=3), rng.integers(2, size=3)
            obs, rewards, _, _, info = env.step({a: np.array([moves[i], gates[i]]) for i, a in enumerate(env.possible_agents)})
            vobs, state, reward, _, diagnostics = vector.step(moves[None], gates[None])
            np.testing.assert_allclose(vobs[0], np.stack(list(obs.values())), atol=1e-6)
            np.testing.assert_allclose(state[0], env.state(), atol=1e-6)
            np.testing.assert_allclose(reward[0], list(rewards.values())[0], atol=1e-6)
            for key in ("attempted_links", "delivered_links", "payload_bytes", "eligible_links"):
                assert diagnostics[key][0] == info[env.possible_agents[0]][key]
    finally:
        env.close()


def test_collision_world_resets_also_clear_discovery_memory():
    env = CollisionVectorSpread(batch=2, communication_mode="landmark")
    env.landmark_channel.known[0] = True
    env.reset_indices(np.array([True, False]))
    assert not env.landmark_channel.received_known[0].any()
    assert not env.landmark_channel.known[0].all()
