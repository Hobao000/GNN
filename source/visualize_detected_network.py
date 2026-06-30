from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Print summary from web_graph.json for AML visualization.")
    parser.add_argument("--web-graph", default="data/outputs/web_graph.json")
    args = parser.parse_args()
    path = Path(args.web_graph)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run source/predict.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(data.get("summary", {}), ensure_ascii=False, indent=2))
    print(f"Nodes for web: {len(data.get('nodes', []))}")
    print(f"Edges for web: {len(data.get('edges', []))}")
    print(f"Groups: {len(data.get('groups', []))}")


if __name__ == "__main__":
    main()
