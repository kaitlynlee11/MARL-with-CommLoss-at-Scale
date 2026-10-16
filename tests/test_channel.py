import numpy as np
import pytest

from limited_comm.channel import Channel


def test_full_delivery_excludes_self_and_counts_unicast_cost():
    channel = Channel(3, 0.0, 2, np.random.default_rng(1))
    payloads = np.arange(12, dtype=np.float32).reshape(3, 4)
    traffic = channel.advance(np.ones(3), payloads)
    assert (traffic.broadcasts, traffic.attempted_links, traffic.delivered_links) == (3, 6, 6)
    assert traffic.payload_bytes == 96
    np.testing.assert_allclose(channel.observe(0)[:4], payloads[1:].mean(axis=0))
    assert channel.observe(0)[4] == 1
    assert channel.observe(0)[5] == 0.5


def test_total_loss_still_costs_transmissions():
    channel = Channel(3, 1.0, 2, np.random.default_rng(1))
    traffic = channel.advance(np.ones(3), np.ones((3, 4)))
    assert traffic.attempted_links == 6 and traffic.delivered_links == 0
    assert traffic.payload_bytes == 96
    assert not channel.observe(0).any()


def test_silence_preserves_then_expires_old_messages():
    channel = Channel(2, 0.0, 2, np.random.default_rng(1))
    payloads = np.ones((2, 4))
    channel.advance(np.array([1, 0]), payloads)
    assert not channel.observe(0).any()
    channel.advance(np.zeros(2), payloads * 2)
    np.testing.assert_allclose(channel.observe(1)[:4], 1)
    assert channel.observe(1)[5] == 1
    channel.advance(np.zeros(2), payloads)
    assert not channel.observe(1).any()


def test_loss_does_not_refresh_cached_message():
    channel = Channel(2, 0.0, 3, np.random.default_rng(1))
    channel.advance(np.ones(2), np.ones((2, 4)))
    channel.loss = 1.0
    channel.advance(np.ones(2), np.ones((2, 4)) * 10)
    np.testing.assert_allclose(channel.observe(0)[:4], 1)
    assert channel.observe(0)[5] == pytest.approx(2 / 3)


def test_seeded_delivery_reproduces():
    a = Channel(4, 0.5, 3, np.random.default_rng(42))
    b = Channel(4, 0.5, 3, np.random.default_rng(42))
    for _ in range(5):
        assert a.advance(np.ones(4), np.ones((4, 4))) == b.advance(np.ones(4), np.ones((4, 4)))
        np.testing.assert_array_equal(a.observe(0), b.observe(0))


def test_quarter_loss_is_sampled_per_directed_link_each_step():
    channel = Channel(3, 0.25, 5, np.random.default_rng(123))
    reference = np.random.default_rng(123)
    for step in range(20):
        # Include silent steps: their draws must still advance the channel RNG.
        gates = np.array([step % 2, 1, int(step % 5 == 0)])
        draws = reference.random((3, 3))
        expected = {(receiver, sender) for sender in range(3) for receiver in range(3)
                    if sender != receiver and gates[sender] and draws[sender, receiver] >= 0.25}
        traffic = channel.advance(gates, np.full((3, 4), step, dtype=np.float32))
        fresh = {key for key, (_, age) in channel.cache.items() if age == 1}
        assert fresh == expected
        assert traffic.delivered_links == len(expected)
        assert traffic.attempted_links == int(gates.sum()) * 2
        assert traffic.payload_bytes == traffic.attempted_links * 16
