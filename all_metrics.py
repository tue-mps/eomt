#!/usr/bin/env python3
"""
collect_metrics.py
Collects all metrics_*.json files from a directory and writes a CSV.

Usage:
    python collect_metrics.py --dir /netscratch/billimoria
    python collect_metrics.py --dir /netscratch/billimoria --output results.csv
    python collect_metrics.py --dir /netscratch/billimoria --pattern "metrics_*.json"
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Recursively flatten nested dicts, joining keys with sep."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def collect(directory: str, pattern: str = "metrics_*.json") -> pd.DataFrame:
    base = Path(directory)
    json_files = sorted(base.glob(pattern))

    if not json_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in: {base}"
        )

    rows = {}
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        rows[path.stem] = flatten(data)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model"
    return df


def main():
    p = argparse.ArgumentParser(description="Aggregate metrics JSON files into a CSV.")
    p.add_argument("--dir",     required=True,             help="Directory containing JSON files")
    p.add_argument("--output",  default="metrics_all.csv", help="Output CSV path")
    p.add_argument("--pattern", default="metrics_*.json",  help="Glob pattern for JSON files")
    args = p.parse_args()

    df = collect(args.dir, args.pattern)

    out = Path(args.output)
    df.to_csv(out)

    print(f"Collected {len(df)} models → {out}")
    print(df.to_string())


if __name__ == "__main__":
    main()
