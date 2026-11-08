"""Run selected simulations with the current Python interpreter.

Examples:
  python run_all.py --quick
  python run_all.py --experiments fourier acrobot burgers --seed 10
  python run_all.py --experiments airfoil --seed 10

The subprocess interface keeps each numerical experiment independently usable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+",
                        choices=["fourier", "acrobot", "burgers", "airfoil"],
                        default=["fourier", "acrobot", "burgers", "airfoil"])
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.experiments:
        command = [sys.executable, str(root / "simulations" / (name + ".py")),
                   "--output", str(destination / name), "--seed", str(args.seed)]
        if args.quick:
            command.append("--quick")
        print(f"Running {name} ({'quick' if args.quick else 'default'})", flush=True)
        start = time.perf_counter()
        result = subprocess.run(command, cwd=root, check=False)
        records.append({"experiment": name, "returncode": result.returncode,
                        "elapsed_seconds": time.perf_counter() - start,
                        "command": [Path(command[0]).name, "simulations/" + name + ".py",
                                    "--output", str(destination / name), "--seed", str(args.seed)]
                                   + (["--quick"] if args.quick else [])})
        if result.returncode:
            break
    (destination / "run_manifest.json").write_text(json.dumps({
        "quick": args.quick, "seed": args.seed, "runs": records,
        "note": "Outputs are newly generated results, not the original physical table records."
    }, indent=2) + "\n")
    return 0 if all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
