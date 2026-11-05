"""Train MAPPO navigation with fixed schedules or learned adaptive transmission."""
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
from .layout import spaced_agents
from .rewards import TASK_REWARD_VERSION


@torch.no_grad()
def validate(model, episodes=100, seed=100_000, horizon=100, radius=0.6,
             schedule="full", period=5, min_landmark_distance=0.4, loss=0.0, terminate_on_collision=False, min_agent_distance=0.4):
    env = VectorSpread(batch=episodes, seed=seed, horizon=horizon, radius=radius,
                       min_landmark_distance=min_landmark_distance, loss=loss, min_agent_distance=min_agent_distance)
    obs, _ = env.reset()
    model.reset_memory()
    ever = np.zeros(episodes, dtype=bool)
    initial = env.diagnostics()
    failed = (initial["collision_pairs"] > 0) if terminate_on_collision else np.zeros(episodes,dtype=bool)
    active_steps = np.zeros(episodes)
    coverage = initial["coverage"].copy()
    distance = initial["distance_sum"].copy()
    success = np.zeros(episodes,dtype=bool)
    attempted = np.zeros(episodes)
    collision_steps = np.zeros(episodes)
    for step in range(horizon):
        if schedule == "adaptive":
            moves, gates = model.actions(obs)
        else:
            moves = model.moves(obs)
            gates = np.full((episodes, 3), transmit(schedule, step, period), dtype=int)
        active = ~failed
        if terminate_on_collision:
            gates = gates * active[:,None]
        obs, _, _, _, info = env.step(moves, gates)
        attempted += info["attempted_links"] * active
        active_steps += active
        collision_steps += (info["collision_pairs"] > 0) & active
        coverage = np.where(active,info["coverage"],coverage)
        distance = np.where(active,info["distance_sum"],distance)
        if terminate_on_collision:
            failed |= (info["collision_pairs"] > 0) & active
        success = info["success"] & ~failed
        ever |= success
    return {"success": float(success.mean()), "coverage": float(coverage.mean()),
            "ever_success": float(ever.mean()), "distance": float(distance.mean() / 3),
            "collision_episode_fraction": float(failed.mean()) if terminate_on_collision else None,
            "mean_active_steps": float(active_steps.mean()),
            "attempts_per_active_step": float(attempted.sum()/max(active_steps.sum(),1)),
            "attempted_packets": float(attempted.mean()), "packet_ratio": float(attempted.mean() / (6 * horizon)),
            "collision_step_rate": float(collision_steps.mean() / horizon)}


def adaptive_reward(info, args, packet_multiplier=0.0):
    return (args.coverage_bonus * 3 * info["coverage"] + args.success_bonus * info["success"]
            - args.collision_penalty * info["collision_pairs"]
            - (args.packet_penalty + packet_multiplier) * info["attempted_links"]
            - args.time_penalty * (~info["success"])
            - args.distance_penalty * np.where(info["coverage"] == 1.0, 0.0,
                                               info["distance_sum"])).astype(np.float32)


def update_packet_multiplier(multiplier, packet_ratio, learning_rate, budget):
    """Projected dual ascent for the mean attempted-traffic constraint."""
    return max(0.0, multiplier + learning_rate * (packet_ratio - budget))


def meets_adaptive_target(metrics, args):
    return metrics["success"] > args.target_success and metrics["packet_ratio"] < args.packet_ratio_target


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
    parser.add_argument("--schedule", choices=["full", "none", "fixed", "adaptive"], default="full")
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
    parser.add_argument("--loss", type=float, default=0.0, help="Known environment per-directed-link packet loss")
    parser.add_argument("--initialize-actor", type=Path, help="Warm-start movement from a fixed-schedule recurrent checkpoint; new optimizer/critic")
    parser.add_argument("--collision-penalty", type=float, default=5.0)
    parser.add_argument("--packet-penalty", type=float, default=0.05)
    parser.add_argument("--time-penalty", type=float, default=0.01)
    parser.add_argument("--distance-penalty", type=float, default=0.5)
    parser.add_argument("--packet-ratio-target", type=float, default=0.2)
    parser.add_argument("--confirmation-episodes", type=int, default=200)
    parser.add_argument("--packet-dual-lr", type=float, default=0.02,
                        help="Projected dual ascent rate for adaptive mean packet budget")
    parser.add_argument("--training-packet-budget", type=float, default=0.18,
                        help="Stochastic training traffic target, with margin below evaluation limit")
    parser.add_argument("--terminate-on-collision", action="store_true")
    parser.add_argument("--failure-penalty", type=float, default=100.0,
                        help="Finite extra terminal collision penalty for collision-aware training")
    parser.add_argument("--min-agent-distance", type=float, default=0.4, help="Minimum pairwise radial agent center distance at reset")
    parser.add_argument("--reset-optimizer", action="store_true", help="Fresh Adam state for a controlled continuation")
    parser.add_argument("--selection-collision-weight", type=float, default=0., help="Optional success-minus-collision validation selection")
    args = parser.parse_args()
    if not np.isfinite(args.selection_collision_weight) or args.selection_collision_weight < 0:
        parser.error("selection-collision-weight must be finite/nonnegative")
    if not np.isfinite(args.min_agent_distance) or args.min_agent_distance < 0:
        parser.error("min-agent-distance must be finite and nonnegative")
    if args.terminate_on_collision and (args.curriculum_steps or args.initialize_actor):
        parser.error("collision training does not use reset curriculum or initialize-actor; resume only compatible collision checkpoints")
    if not np.isfinite(args.failure_penalty) or args.failure_penalty <= 0:
        parser.error("failure penalty must be finite and positive")
    adaptive = args.schedule == "adaptive"
    if (not np.isfinite(args.packet_dual_lr) or args.packet_dual_lr < 0
            or not 0 < args.training_packet_budget < args.packet_ratio_target):
        parser.error("dual rate must be finite/nonnegative and training packet budget below evaluation target")
    if (not np.isfinite(args.loss) or not 0 <= args.loss <= 1
            or not 0 < args.packet_ratio_target < 1 or args.confirmation_episodes < 1
            or not all(np.isfinite(x) and x >= 0 for x in
                       [args.collision_penalty, args.packet_penalty, args.time_penalty, args.distance_penalty])):
        parser.error("invalid loss, reward coefficients, traffic target or confirmation count")
    if adaptive and (not args.recurrent or args.success_bonus <= 3 * args.coverage_bonus
                     or args.coverage_bonus <= 0 or args.packet_penalty <= 0 or args.collision_penalty <= 0
                     or args.target_success >= 1 or (args.resume and not args.terminate_on_collision)):
        parser.error("adaptive requires recurrent actor, positive coverage/packet/collision terms, full bonus > total partial bonus, strict target < 1, and a fresh output (use initialize-actor)")
    if args.initialize_actor and (not adaptive or args.resume):
        parser.error("initialize-actor requires a fresh adaptive run")
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
        if model.spec.get("separated"):
            parser.error("Use limited_comm.paper_train for modular-policy optimization")
        if previous["config"]["radius"] != args.radius or previous["config"]["horizon"] != args.horizon:
            parser.error("resume radius/horizon must match the checkpoint")
        parent_config = previous["config"]
        if parent_config.get("min_agent_distance",0.0) != args.min_agent_distance:
            parser.error("resume agent-start spacing must match checkpoint; use --min-agent-distance 0 for legacy runs")
        if (bool(parent_config.get("terminate_on_collision",False)) != args.terminate_on_collision
                or parent_config.get("loss",0.) != args.loss):
            parser.error("resume must preserve collision termination and packet loss")
        if args.terminate_on_collision:
            for key in ["failure_penalty","collision_penalty","packet_penalty","time_penalty","distance_penalty",
                        "coverage_bonus","success_bonus","training_packet_budget","packet_dual_lr"]:
                if parent_config[key] != getattr(args,key):
                    parser.error(f"collision resume reward/constraint mismatch: {key}")
        if (parent_config.get("min_landmark_distance", 0.0) != args.min_landmark_distance
                or parent_config.get("training_schedule", "full") != args.schedule
                or (args.schedule == "fixed" and parent_config.get("period", 5) != args.period)):
            parser.error("resume schedule, period and landmark spacing must match the checkpoint")
        start_step = previous["steps"]
    else:
        model = MAPPO(hidden=args.hidden, recurrent=args.recurrent, adaptive=adaptive)
    initializer_hash = None
    if args.initialize_actor:
        parent, source = load_policy(args.initialize_actor)
        if (parent.spec.get("adaptive") or not parent.spec["recurrent"]
                or parent.spec["hidden"] != args.hidden or source["config"]["radius"] != args.radius
                or source["config"]["horizon"] != args.horizon
                or source["config"].get("min_landmark_distance", 0) != args.min_landmark_distance):
            parser.error("initializer must be a compatible fixed-schedule recurrent navigation policy")
        weights = parent.actor.state_dict()
        # Ten categorical actions: movement + 5 * transmit. Preserve movement marginals,
        # with an initial stochastic transmit probability of 0.1 for every movement.
        weights["head.weight"] = weights["head.weight"].repeat(2, 1)
        bias = weights["head.bias"]
        weights["head.bias"] = torch.cat([bias + np.log(0.9), bias + np.log(0.1)])
        model.actor.load_state_dict(weights)
        initializer_bytes = args.initialize_actor.read_bytes()
        (args.output / "initializer.pt").write_bytes(initializer_bytes)
        initializer_hash = hashlib.sha256(initializer_bytes).hexdigest()
    if model.spec["recurrent"] and args.rollout % args.sequence_length:
        parser.error("recurrent rollout must be divisible by sequence-length")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.resume and not args.reset_optimizer:
        optimizer.load_state_dict(previous["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    config = {**vars(args), "output": str(args.output), "resume": str(args.resume) if args.resume else None,
              "parent_sha256": parent_hash, "parent_steps": start_step if args.resume else None,
              "training_schedule": args.schedule, "loss": args.loss, "comm_cost": 0.0,
              "initialize_actor": str(args.initialize_actor) if args.initialize_actor else None,
              "initializer_sha256": initializer_hash,
              "confirmation_seed": 500_000, "adaptive_heldout_seed": 400_000,
              "stop_rule": "success > target AND attempted/full < packet target, confirmed in original environment" if adaptive else "baseline validation target",
              "task_reward_version": TASK_REWARD_VERSION,
              "reward_formula": "coverage_bonus*covered_count + success_bonus*success - collision_penalty*pairs - (packet_penalty+packet_multiplier)*attempted_links - time_penalty*(not success) - distance_penalty*distance_sum*(not full_coverage)" if adaptive else "coverage-gated baseline",
              "gamma": 0.99, "gae_lambda": 0.95, "ppo_clip": 0.2,
              "value_normalization": "PopArt EMA 0.01; preserve unnormalized predictions",
              "validation_seed": 100_000, "test_seed_start": 200_000,
              "torch": str(torch.__version__), "numpy": np.__version__,
              "mpe2": importlib.metadata.version("mpe2"), "python": platform.python_version(),
              "device": "cpu", "model_spec": model.spec,
              "resume_semantics": "weights and optimizer retained; simulator and RNG restarted"}
    if args.terminate_on_collision:
        config["reward_formula"] = ("coverage_bonus*covered_count + success_bonus*full_coverage - "
            "collision_penalty*pairs - (packet_penalty+adaptive_packet_multiplier)*attempted_links - "
            "time_penalty*(not full_coverage) - distance_penalty*distance_sum*(not full_coverage) - failure_penalty*collision_terminal")
        config["initial_collision_training_rule"] = "count actionless failures and reset without an action transition; evaluation includes all starts"
        config["stop_rule"] = "matched fixed budget; select by collision-free success, fewer collision episodes, coverage, traffic"
    args.output.joinpath("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.terminate_on_collision:
        from .collision_vector_env import CollisionVectorSpread
        environment_class = CollisionVectorSpread
    else:
        environment_class = VectorSpread
    env = environment_class(batch=args.envs, horizon=args.horizon, seed=args.seed + start_step,
                       radius=args.radius, min_landmark_distance=args.min_landmark_distance, loss=args.loss, min_agent_distance=args.min_agent_distance)
    validation_args = dict(horizon=args.horizon, radius=args.radius, schedule=args.schedule,
                           period=args.period, min_landmark_distance=args.min_landmark_distance, loss=args.loss,
                           terminate_on_collision=args.terminate_on_collision, min_agent_distance=args.min_agent_distance)
    def reset_training(step, mask=None):
        if args.terminate_on_collision:
            env.reset_indices(np.ones(args.envs,dtype=bool) if mask is None else mask)
        else:
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
        env.pos[:] = spaced_agents(env.pos, env.rng, args.min_agent_distance)
        return env.observe(), env.state()

    obs, state = reset_training(start_step)
    discovered = env.visible_landmarks()
    memory = torch.zeros(args.envs, 3, model.spec["hidden"])
    gates = np.ones((args.envs, 3), dtype=int)
    best_score = (-1.0, -1.0)
    started = time.perf_counter()
    total = start_step
    update = 0
    stop_reason = "budget_exhausted"
    packet_multiplier = float(previous.get("packet_multiplier",0.0)) if args.resume else 0.0

    def save(name, validation):
        torch.save({"model": model.state_dict(), "model_spec": model.spec,
                    "optimizer": optimizer.state_dict(), "steps": total,
                    "config": config, "validation": validation, "packet_multiplier": packet_multiplier}, args.output / name)

    with args.output.joinpath("learning.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        while total < args.steps:
            update += 1
            rollout_attempts = 0.0
            multiplier_used = packet_multiplier
            buffers = {k: [] for k in ["obs", "state", "moves", "logp", "value", "reward", "next_value", "boundary", "terminal", "memory"]}
            for _ in range(args.rollout):
                with torch.no_grad():
                    tensor_obs, tensor_state = torch.from_numpy(obs), torch.from_numpy(state)
                    buffers["memory"].append(memory.numpy().copy())
                    dist, next_memory = model.policy(tensor_obs, memory)
                    moves = dist.sample()
                    logp = dist.log_prob(moves).numpy()
                    values = model.value(tensor_state).numpy()
                potential = env.potential() if args.potential_scale else None
                if adaptive:
                    gates = moves.numpy() // 5
                else:
                    if args.terminate_on_collision:
                        gates[:] = ((env.steps % args.period == 0)[:,None] if args.schedule == "fixed"
                                    else int(args.schedule == "full"))
                    else:
                        gates.fill(transmit(args.schedule, env.steps, args.period))
                next_obs, next_state, reward, boundary, info = env.step(moves.numpy() % 5, gates)
                terminals = info["collision_terminal"] if args.terminate_on_collision else np.zeros(args.envs,dtype=bool)
                boundaries = np.broadcast_to(boundary,(args.envs,)).copy()
                if adaptive or args.terminate_on_collision:
                    reward = adaptive_reward(info, args, multiplier_used if adaptive else 0)
                    rollout_attempts += float(info["attempted_links"].sum())
                    if args.terminate_on_collision:
                        reward -= args.failure_penalty * terminals
                if args.potential_scale:
                    reward = (reward + args.potential_scale * (0.99 * env.potential() - potential)).astype(np.float32)
                if not adaptive and not args.terminate_on_collision:
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
                                   ("next_value", next_values), ("boundary", boundaries), ("terminal", terminals)]:
                    buffers[key].append(value)
                if boundaries.any():
                    obs, state = reset_training(total + args.envs, boundaries) if args.terminate_on_collision else reset_training(total + args.envs)
                    discovered = env.visible_landmarks()
                else:
                    obs, state = next_obs, next_state
                if model.spec["recurrent"]:
                    memory = next_memory * torch.from_numpy(~boundaries)[:,None,None]
                total += args.envs
            arrays = {k: np.asarray(v) for k, v in buffers.items()}
            advantages, returns = gae(arrays["reward"], arrays["value"], arrays["next_value"], arrays["boundary"], terminals=arrays["terminal"])
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
                        dist = torch.distributions.Categorical(logits=torch.stack(logits).reshape(-1, 3, model.action_dim))
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
            sampled_packet_ratio = rollout_attempts / (6 * args.envs * args.rollout)
            if adaptive:
                packet_multiplier = update_packet_multiplier(packet_multiplier, sampled_packet_ratio,
                                                             args.packet_dual_lr, args.training_packet_budget)
            averaged = np.mean(metrics, axis=0)
            record = {"steps": total, "agent_transitions": total * 3, "update": update,
                      "seconds": time.perf_counter() - started, "mean_step_reward": float(arrays["reward"].mean()),
                      "policy_loss": averaged[0], "value_loss": averaged[1], "entropy": averaged[2],
                      "approx_kl": averaged[3], "explained_variance": float(explained_variance)}
            if args.terminate_on_collision:
                record["initial_actionless_failures"] = env.initial_failures
                record["terminal_collision_transitions"] = int(arrays["terminal"].sum())
            if adaptive:
                record.update(sampled_packet_ratio=sampled_packet_ratio,
                              effective_packet_penalty=args.packet_penalty + multiplier_used,
                              next_packet_multiplier=packet_multiplier)
            if update == 1 or update % args.eval_every == 0 or total >= args.steps:
                validation = validate(model, args.validation_episodes, **validation_args)
                record["validation"] = validation
                if args.terminate_on_collision:
                    score = (validation["success"], -validation["collision_episode_fraction"],
                             validation["coverage"], -validation["packet_ratio"])
                elif adaptive:
                    score = (int(meets_adaptive_target(validation, args)), validation["success"],
                             -validation["packet_ratio"], validation["coverage"])
                else:
                    score = (validation["success"], validation["coverage"])
                if args.terminate_on_collision and args.selection_collision_weight:
                    score = (int(validation["packet_ratio"] < args.packet_ratio_target),
                             validation["success"]-args.selection_collision_weight*validation["collision_episode_fraction"],
                             validation["success"],-validation["packet_ratio"])
                if score > best_score:
                    best_score = score
                    save("best.pt", validation)
                save("last.pt", validation)
                print(json.dumps(record), flush=True)
                (args.output / "status.json").write_text(json.dumps({
                    "state": "running", "steps": total, "step_limit": args.steps,
                    "validation": validation, "target_met": False}, indent=2), encoding="utf-8")
                reached = validation["success"] >= args.target_success
                if adaptive:
                    reached = meets_adaptive_target(validation, args)
                    if reached:
                        from .evaluate import evaluate_episode, summarize
                        rows = [evaluate_episode(model, 500_000 + i, "adaptive", loss=args.loss,
                                                 horizon=args.horizon, radius=args.radius,
                                                 min_landmark_distance=args.min_landmark_distance,
                                                 terminate_on_collision=args.terminate_on_collision, min_agent_distance=args.min_agent_distance)
                                for i in range(args.confirmation_episodes)]
                        confirmation = summarize(rows)
                        confirmation["success"] = confirmation["success_rate"]
                        confirmation["packet_ratio"] = confirmation["attempted_links"] / (6 * args.horizon)
                        record["confirmation"] = confirmation
                        reached = meets_adaptive_target(confirmation, args)
                        (args.output / f"confirmation-{total}.json").write_text(
                            json.dumps({"summary": confirmation, "episodes": rows}, indent=2), encoding="utf-8")
                        if reached:
                            save("target.pt", confirmation)
                if reached and not args.no_early_stop:
                    stop_reason = "target_reached"
                    log.write(json.dumps(record) + "\n")
                    print("Confirmed adaptive targets reached." if adaptive else "Validation success target reached; evaluate best.pt on held-out episodes.", flush=True)
                    break
            log.write(json.dumps(record) + "\n")
            if (args.output / "STOP").exists():
                validation = validate(model, args.validation_episodes, **validation_args)
                save("last.pt", validation)
                stop_reason = "user_stop"
                print("Stopped at requested update boundary; last.pt saved.", flush=True)
                break
    (args.output / "status.json").write_text(json.dumps({"state": stop_reason, "steps": total,
        "step_limit": args.steps, "validation": validation,
        "target_met": stop_reason == "target_reached"}, indent=2), encoding="utf-8")
    print(f"Checkpoints and learning log: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
