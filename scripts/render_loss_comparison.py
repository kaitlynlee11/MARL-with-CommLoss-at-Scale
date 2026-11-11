"""Export a standalone interactive comparison from verified loss-test results."""
import argparse
import json
from pathlib import Path

from run_adaptive_seeds import atomic_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "summary.json").read_text())
    assert result["complete"]
    template = Path(__file__).with_name("scheduler-loss-comparison-template.html").read_text(encoding="utf-8")
    atomic_text(root / "comparison.html", template.replace("__COMPARISON_DATA__", json.dumps(result, allow_nan=False).replace("<", "\\u003c")))
    print(root / "comparison.html")


if __name__ == "__main__":
    main()
