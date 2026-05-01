"""
run_model.py - Command-line entry point for the ULP cash flow model.

Usage
-----
    python run_model.py                          # uses config.yaml in current directory
    python run_model.py --config path/to/config.yaml
    python run_model.py --config config.yaml --output-dir ./results/run1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ULP Model – Universal Life Policy cash flow projection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Override output_dir from config",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Override compute_device from config",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["summary", "per_policy", "both"],
        help="Override output_mode from config",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    from ulp_model.config import load_config

    config = load_config(config_path)

    # Apply CLI overrides
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.compute_device = args.device
    if args.mode is not None:
        config.output_mode = args.mode

    print(f"ULP Model")
    print(f"  Config          : {config_path}")
    print(f"  Policy inputs   : {config.policy_inputs_dir}")
    print(f"  Param tables    : {config.param_tables_dir}")
    print(f"  Output dir      : {config.output_dir}")
    print(f"  Output mode     : {config.output_mode}")
    print(f"  Device          : {config.compute_device}")
    print(f"  Precision       : {config.float_precision}")
    print(f"  Projection      : {config.MAX_PROJ_YEARS} years ({config.MAX_PROJ_MONTHS} months)")
    print()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    from ulp_model.loader import load_model_inputs

    print("Loading inputs...", end=" ", flush=True)
    t0 = time.perf_counter()
    policies, param_tables = load_model_inputs(config)
    print(f"done ({time.perf_counter() - t0:.2f}s)  |  {policies.policy_id.shape[0]:,} policies")

    # ------------------------------------------------------------------
    # Run model
    # ------------------------------------------------------------------
    from ulp_model.model import ULPModel
    from ulp_model.outputs import print_metrics, write_summary_outputs, write_per_policy_outputs

    model = ULPModel(config)

    print("Running projection...", end=" ", flush=True)
    result = model.run(policies, param_tables)
    print(f"done ({result['elapsed']:.2f}s)")

    # ------------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------------
    print_metrics(result["summary"], policies, scenario_id=1, elapsed_time=result["elapsed"])

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    if config.output_mode in ("summary", "both"):
        summary = result["summary"]

        # Filter to requested time steps
        if config.output_time_steps != "all" and isinstance(config.output_time_steps, list):
            import torch
            ts = config.output_time_steps
            summary_out = {k: v[ts] for k, v in summary.items()}
        else:
            summary_out = summary

        write_summary_outputs(summary_out, scenario_id=1, output_dir=config.output_dir)
        print(f"  Summary output  → {config.output_dir}/summary_scen1.csv")

    if config.output_mode in ("per_policy", "both"):
        write_per_policy_outputs(
            result["part1"],
            result["part2"],
            result["part3"],
            policies.policy_id,
            scenario_id=1,
            output_dir=config.output_dir,
            output_batch_size=config.output_batch_size,
        )
        print(f"  Per-policy output → {config.output_dir}/per_policy_scen*.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
