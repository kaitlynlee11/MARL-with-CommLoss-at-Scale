"""Train MAPPO navigation with full, periodic, or absent communication."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .mappo import MAPPO, gae, load_policy
from .vector_env import VectorSpread
from .schedules import transmit


@torch.no_grad()
def validate(model, episodes=100, seed=100_000, horizon=100, radius=0.6,
             schedule="full", period=5, min_landmark_distance=0.4):
    env = VectorSpread(batch=episodes, seed=seed, horizon=horizon, radius=radius,
                       min_landmark_distance=min_landmark_distance)
    obs, _ = env.reset()
    model.reset_memory()
    ever = np.zeros(episodes, dtype=bool)
    for step in range(horizon):
        gates = np.full((episodes, 3), transmit(schedule, step, period), dtype=int)
        obs, _, _, _, info = env.step(model.moves(obs), gates)
        ever |= info["success"]
    return {"success": float(info["success"].mean()), "coverage": float(info["coverage"].mean()),
            "ever_success": float(ever.mean()), "distance": float(info["distance_sum"].mean() / 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/mappo-full"))
    parser.add_argument("--steps", type=int, default=2_000_000, help="Total training simulator steps, across worlds")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--rollout", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--recurrent", action="store_true", help="GRU actor remembering past local observations")
    parser.add_argument("--sequence-length", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=2048, help="Team transitions per minibatch")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--gradient-clipping", choices=["joint", "separate"], default="separate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--schedule", choices=["full", "none", "fixed"], default="full")
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--min-landmark-distance", type=float, default=0.4)
    parser.add_argument("--no-early-stop", action="store_true", help="Use the entire budget for fair baseline comparisons")
    parser.add_argument("--curriculum-steps", type=int, default=0,
                        help="Optional near-landmark reset curriculum, annealed to original resets by this step")
    parser.add_argument("--curriculum-mode", choices=["mixture", "blend"], default="mixture")
    parser.add_argument("--potential-scale", type=float, default=0.0,
                        help="Optional potential-based shaping using optimal assignment distance")
    parser.add_argument("--coverage-bonus", type=float, default=0.0,
                        help="Training-only reward per covered landmark per step")
    parser.add_argument("--success-bonus", type=float, default=0.0,
                        help="Training-only reward per all-landmarks-covered step")
    parser.add_argument("--discovery-bonus", type=float, default=0.0,
                        help="Training-only reward for each previously unseen landmark entering team sensor range")
    parser.add_argument("--resume", type=Path, help="Continue weights/optimizer; restart simulator with a fresh seed")
    args = parser.parse_args()
    if min(args.steps, args.envs, args.rollout, args.horizon, args.hidden, args.epochs,
           args.minibatch, args.threads, args.eval_every, args.validation_episodes) < 1:
        parser.error("budgets and dimensions must be positive")
    if (not 0 < args.target_success <= 1 or args.radius <= 0 or args.lr <= 0
            or args.entropy < 0 or args.curriculum_steps < 0 or args.potential_scale < 0
            or args.coverage_bonus < 0 or args.success_bonus < 0 or args.discovery_bonus < 0):
        parser.error("invalid success target, sensing radius or optimization settings")
    if args.sequence_length < 1:
        parser.error("sequence-length must be positive")
    if args.period < 1 or not np.isfinite(args.min_landmark_distance) or args.min_landmark_distance < 0:
        parser.error("require positive period and finite nonnegative landmark separation")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start_step = 0
    parent_hash = None
    if args.resume:
        parent_bytes = args.resume.read_bytes()
        parent_path = args.output / "parent.pt"
        parent_path.write_bytes(parent_bytes)
        parent_hash = hashlib.sha256(parent_bytes).hexdigest()
        model, previous = load_policy(parent_path)
        if previous["config"]["radius"] != args.radius or previous["config"]["horizon"] != args.horizon:
            parser.error("resume radius/horizon must match the checkpoint")
        parent_config = previous["config"]
        if (parent_config.get("min_landmark_distance", 0.0) != args.min_landmark_distance
                or parent_config.get("training_schedule", "full") != args.schedule
                or (args.schedule == "fixed" and parent_config.get("period", 5) != args.period)):
            parser.error("resume schedule, period and landmark spacing must match the checkpoint")
        start_step = previous["steps"]
    else:
        model = MAPPO(hidden=args.hidden, recurrent=args.recurrent)
    if model.spec["recurrent"] and args.rollout % args.sequence_length:
        parser.error("recurrent rollout must be divisible by sequence-length")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.resume:
        optimizer.load_state_dict(previous["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    config = {**vars(args), "output": str(args.output), "resume": str(args.resume) if args.resume else None,
              "parent_sha256": parent_hash, "parent_steps": start_step if args.resume else None,
              "training_schedule": args.schedule, "loss": 0.0, "comm_cost": 0.0,
              "gamma": 0.99, "gae_lambda": 0.95, "ppo_clip": 0.2,
              "value_normalization": "PopArt EMA 0.01; preserve unnormalized predictions",
              "validation_seed": 100_000, "test_seed_start": 200_000,
              "torch": str(torch.__version__), "numpy": np.__version__,
              "mpe2": importlib.metadata.version("mpe2"), "python": platform.python_version(),
              "device": "cpu", "model_spec": model.spec,
              "resume_semantics": "weights and optimizer retained; simulator and RNG restarted"}
    args.output.joinpath("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = VectorSpread(batch=args.envs, horizon=args.horizon, seed=args.seed + start_step,
                       radius=args.radius, min_landmark_distance=args.min_landmark_distance)
    validation_args = dict(horizon=args.horizon, radius=args.radius, schedule=args.schedule,
                           period=args.period, min_landmark_distance=args.min_landmark_distance)
    def reset_training(step):
        env.reset()
        if args.curriculum_steps > step:
            # Initial-state curriculum only: rewards/actions/observations stay unchanged.
            progress = step / args.curriculum_steps
            easy = (env.rng.random(args.envs) < 1 - progress) if args.curriculum_mode == "mixture" else None
            order = np.argsort(env.rng.random((args.envs, 3)), axis=1)
            goals = np.take_along_axis(env.landmarks, order[..., None], axis=1)
            if args.curriculum_mode == "blend":
                env.pos[:] = ((1 - progress) * goals + progress * env.pos
                              + (1 - progress) * env.rng.uniform(-0.05, 0.05, env.pos.shape))
            else:
                env.pos[easy] = (goals + env.rng.uniform(-0.25, 0.25, env.pos.shape))[easy]
        return env.observe(), env.state()

    obs, state = reset_training(start_step)
    discovered = env.visible_landmarks()
    memory = torch.zeros(args.envs, 3, model.spec["hidden"])
    gates = np.ones((args.envs, 3), dtype=int)
    best_score = (-1.0, -1.0)
    started = time.perf_counter()
    total = start_step
    update = 0

    def save(name, validation):
        torch.save({"model": model.state_dict(), "model_spec": model.spec,
                    "optimizer": optimizer.state_dict(), "steps": total,
                    "config": config, "validation": validation}, args.output / name)

    with args.output.joinpath("learning.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        while total < args.steps:
            update += 1
            buffers = {k: [] for k in ["obs", "state", "moves", "logp", "value", "reward", "next_value", "boundary", "memory"]}
            for _ in range(args.rollout):
                with torch.no_grad():
                    tensor_obs, tensor_state = torch.from_numpy(obs), torch.from_numpy(state)
                    buffers["memory"].append(memory.numpy().copy())
                    dist, next_memory = model.policy(tensor_obs, memory)
                    moves = dist.sample()
                    logp = dist.log_prob(moves).numpy()
                    values = model.value(tensor_state).numpy()
                potential = env.potential() if args.potential_scale else None
                gates.fill(transmit(args.schedule, env.steps, args.period))
                next_obs, next_state, reward, boundary, info = env.step(moves.numpy(), gates)
                if args.potential_scale:
                    reward = (reward + args.potential_scale * (0.99 * env.potential() - potential)).astype(np.float32)
                reward = (reward + args.coverage_bonus * 3 * info["coverage"]
                          + args.success_bonus * info["success"]).astype(np.float32)
                if args.discovery_bonus:
                    visible = env.visible_landmarks()
                    reward += args.discovery_bonus * (visible & ~discovered).sum(axis=1)
                    discovered |= visible
                with torch.no_grad():
                    next_values = model.value(torch.from_numpy(next_state)).numpy()
                for key, value in [("obs", obs), ("state", state), ("moves", moves.numpy()),
                                   ("logp", logp), ("value", values), ("reward", reward),
                                   ("next_value", next_values), ("boundary", np.full(args.envs, boundary))]:
                    buffers[key].append(value)
                obs, state = (reset_training(total + args.envs) if boundary else (next_obs, next_state))
                if boundary:
                    discovered = env.visible_landmarks()
                if model.spec["recurrent"]:
                    memory = torch.zeros_like(memory) if boundary else next_memory
                total += args.envs
            arrays = {k: np.asarray(v) for k, v in buffers.items()}
            advantages, returns = gae(arrays["reward"], arrays["value"], arrays["next_value"], arrays["boundary"])
            explained_variance = 1 - np.var(returns - arrays["value"]) / max(np.var(returns), 1e-8)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            count = args.rollout * args.envs
            observations = torch.from_numpy(arrays["obs"].reshape(count, 3, -1))
            states = torch.from_numpy(arrays["state"].reshape(count, -1))
            actions = torch.from_numpy(arrays["moves"].reshape(count, 3))
            old_logp = torch.from_numpy(arrays["logp"].reshape(count, 3))
            adv = torch.from_numpy(advantages.reshape(count, 1))
            targets = torch.from_numpy(returns.reshape(count))
            normalized_targets = model.normalize_value_targets(targets)
            metrics = []
            for _ in range(args.epochs):
                sequence_length = args.sequence_length if model.spec["recurrent"] else 1
                sequence_count = count // sequence_length
                permutation = rng.permutation(sequence_count)
                sequence_batch = max(1, args.minibatch // sequence_length)
                for start in range(0, sequence_count, sequence_batch):
                    selected = permutation[start:start + sequence_batch]
                    # Flat storage is time-major; each sequence follows a single world.
                    initial_t = (selected // args.envs) * sequence_length
                    worlds = selected % args.envs
                    indices = (initial_t[None] + np.arange(sequence_length)[:, None]) * args.envs + worlds[None]
                    idx = indices.reshape(-1)
                    if model.spec["recurrent"]:
                        hidden = torch.from_numpy(arrays["memory"][initial_t, worlds])
                        logits = []
                        for time_idx in indices:
                            distribution, hidden = model.policy(observations[time_idx], hidden)
                            logits.append(distribution.logits)
                            reset = torch.from_numpy(arrays["boundary"].reshape(count)[time_idx])
                            hidden = hidden * (~reset)[:, None, None]
                        dist = torch.distributions.Categorical(logits=torch.stack(logits).reshape(-1, 3, 5))
                    else:
                        dist = model.distribution(observations[idx])
                    log_ratio = dist.log_prob(actions[idx]) - old_logp[idx]
                    ratio = log_ratio.exp()
                    policy_loss = -torch.minimum(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx]).mean()
                    value_loss = 0.5 * (model.critic(states[idx]).squeeze(-1) - normalized_targets[idx]).square().mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss + value_loss - args.entropy * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    if args.gradient_clipping == "joint":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
                        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
                    optimizer.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - log_ratio).mean().item()
                    metrics.append([policy_loss.item(), value_loss.item(), entropy.item(), kl])
            averaged = np.mean(metrics, axis=0)
            record = {"steps": total, "agent_transitions": total * 3, "update": update,
                      "seconds": time.perf_counter() - started, "mean_step_reward": float(arrays["reward"].mean()),
                      "policy_loss": averaged[0], "value_loss": averaged[1], "entropy": averaged[2],
                      "approx_kl": averaged[3], "explained_variance": float(explained_variance)}
            if update == 1 or update % args.eval_every == 0 or total >= args.steps:
                validation = validate(model, args.validation_episodes, **validation_args)
                record["validation"] = validation
                score = (validation["success"], validation["coverage"])
                if score > best_score:
                    best_score = score
                    save("best.pt", validation)
                save("last.pt", validation)
                print(json.dumps(record), flush=True)
                if validation["success"] >= args.target_success and not args.no_early_stop:
                    log.write(json.dumps(record) + "\n")
                    print("Validation success target reached; evaluate best.pt on held-out episodes.", flush=True)
                    break
            log.write(json.dumps(record) + "\n")
            if (args.output / "STOP").exists():
                validation = validate(model, args.validation_episodes, **validation_args)
                save("last.pt", validation)
                print("Stopped at requested update boundary; last.pt saved.", flush=True)
                break
    print(f"Checkpoints and learning log: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
