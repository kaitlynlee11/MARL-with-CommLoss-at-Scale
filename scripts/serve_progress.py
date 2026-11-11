"""Serve the local, read-only run dashboard: python scripts/serve_progress.py."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
LABELS = ("full", "none", "fixed_5")
INITIAL, TOTAL, EPISODES = 8_320_000, 8_960_000, 200


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def tail_records(path, limit=65536):
    """Read a bounded tail; tolerate a writer's unfinished final JSON line."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        records = []
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    records.append(row)
            except ValueError:
                continue
        return records
    except OSError:
        return []


def modified(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def episode_count(path):
    try:
        with path.open("rb") as stream:
            return sum(1 for line in stream if line.endswith(b"\n"))
    except OSError:
        return 0


def condition_progress(directory, seed, label, verified=False, error=None, now=None):
    now = time.time() if now is None else now
    initial_log = directory / "initial/learning.jsonl"
    refine_log = directory / "refine/learning.jsonl"
    initial, refine = tail_records(initial_log), tail_records(refine_log)
    last = (refine or initial or [{}])[-1]
    steps = min(TOTAL, max(0, last.get("steps", 0)))
    evaluation_file = directory / "evaluation/episodes.jsonl"
    evaluation = read_json(directory / "evaluation/summary.json")
    done_episodes = episode_count(evaluation_file)
    result = evaluation.get("conditions", {}).get(label) if evaluation else None
    state = "queued"
    if directory.exists():
        state = "starting"
    if (directory / "initial/config.json").exists() or initial:
        state = "initial"
    if (directory / "refine/config.json").exists() or refine:
        state = "refine"
    if steps >= TOTAL or evaluation_file.exists():
        state = "evaluating"
    if result:
        state = "verifying"
    if verified:
        state = "complete"
        steps = TOTAL
    if error:
        state = "failed"
    paths = [initial_log, refine_log, evaluation_file, directory / "evaluation/summary.json"]
    timestamps = [stamp for path in paths if (stamp := modified(path)) is not None]
    updated = max(timestamps) if timestamps else None
    validation = next((r["validation"] for r in reversed(initial + refine) if r.get("validation")), None)
    selection = read_json(directory / "selection.json")
    return {"seed": seed, "condition": label, "state": state, "steps": steps,
            "fraction": steps / TOTAL, "initial_steps": min(steps, INITIAL),
            "refine_steps": max(0, steps - INITIAL), "episodes": done_episodes,
            "updated_unix": updated, "seconds_since_update": max(0, now - updated) if updated else None,
            "stale": bool(updated and now - updated > 180 and state in ("initial", "refine", "evaluating")),
            "latest_training": last, "validation": validation, "result": result,
            "selection": selection, "error": error, "artifact_directory": str(directory)}


def collect_progress(repo=REPO):
    now = time.time()
    original_root = repo / "results/baseline-agent-spacing04-15-seeds-20261109"
    measured_root = original_root
    original = read_json(original_root / "status.json", {})
    measured = read_json(measured_root / "status.json", {})
    summary = read_json(measured_root / "summary.json", {})
    verified = {(r["training_seed"], r["condition"]) for r in summary.get("per_seed", [])}
    runs = []
    for seed in range(42, 57):
        root = measured_root if seed == 42 else original_root
        for label in LABELS:
            error = measured.get("failures", {}).get(f"{seed}/{label}")
            old_state = original.get("jobs", {}).get(f"{seed}/{label}", "")
            if old_state.startswith("failed"):
                error = old_state
            runs.append(condition_progress(root / f"seed-{seed}" / label, seed, label,
                                           (seed, label) in verified, error, now))
    counts = {state: sum(r["state"] == state for r in runs) for state in
              ("queued", "starting", "initial", "refine", "evaluating", "verifying", "complete", "failed")}
    return {"generated_unix": now, "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "runs": runs, "counts": counts, "total_runs": len(runs),
            "training_steps": sum(r["steps"] for r in runs), "training_budget": len(runs) * TOTAL,
            "evaluation_episodes": sum(r["episodes"] for r in runs), "evaluation_budget": len(runs) * EPISODES,
            "initial_budget": INITIAL, "policy_budget": TOTAL,
            "aggregate": summary.get("conditions", {}),
            "coordinator_updated_unix": measured.get("updated_unix"),
            "source_state": original.get("state", "unknown"), "measured_state": measured.get("state", "unknown"),
            "started_unix": original.get("started_unix"), "refresh_seconds": 5}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/progress":
            data = json.dumps(collect_progress(), allow_nan=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif route in ("/adaptive", "/adaptive/", "/adaptive/progress.html",
                       "/adaptive/report.md", "/adaptive/per-seed.csv", "/adaptive/progress.json"):
            filename = route.rsplit("/", 1)[-1]
            if filename not in ("report.md", "per-seed.csv", "progress.json"):
                filename = "progress.html"
            path = REPO / "results/adaptive-zero-loss-15-seeds-20261110" / filename
            if not path.exists():
                self.send_error(404, "Adaptive study has not generated this artifact yet")
                return
            data = path.read_bytes()
            content_type = {"progress.html": "text/html; charset=utf-8", "report.md": "text/plain; charset=utf-8",
                            "per-seed.csv": "text/csv; charset=utf-8", "progress.json": "application/json; charset=utf-8"}[filename]
        elif route in ("/loss", "/loss/", "/loss/progress.html", "/loss/comparison.html", "/loss/report.md",
                       "/loss/per-seed.csv", "/loss/summary.json", "/loss/status.json"):
            filename = route.rsplit("/", 1)[-1]
            if filename not in ("comparison.html", "report.md", "per-seed.csv", "summary.json", "status.json"):
                filename = "progress.html"
            path = REPO / "results/scheduler-loss-comparison-20261110" / filename
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            content_type = ("text/html; charset=utf-8" if filename.endswith(".html") else
                            "application/json; charset=utf-8" if filename.endswith(".json") else
                            "text/csv; charset=utf-8" if filename.endswith(".csv") else "text/plain; charset=utf-8")
        elif route in ("/", "/progress.html"):
            data = (REPO / "progress.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        elif route == "/report":
            path = REPO / "docs/navigation-baseline-15-seeds-results.md"
            if not path.exists():
                self.send_error(404)
                return
            data, content_type = path.read_bytes(), "text/plain; charset=utf-8"
        elif route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Live dashboard: http://127.0.0.1:{args.port}/progress.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
