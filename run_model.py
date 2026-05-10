"""
run_model.py - Command-line entry point for the ULP cash flow model.

Policies are always processed in batches of config.batch_size (set in
config.yaml).  When the total number of policies is less than or equal
to batch_size the model runs as a single batch, which is equivalent to
the previous full-load behaviour.

Usage
-----
    python run_model.py                          # uses config.yaml in current directory
    python run_model.py --config path/to/config.yaml
    python run_model.py --config config.yaml --output-dir ./results/run1
"""
from __future__ import annotations

import argparse
import sys
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override batch_size from config (policies per batch)",
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
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    print(f"ULP Model")
    print(f"  Config          : {config_path}")
    print(f"  Policy inputs   : {config.policy_inputs_file}")
    print(f"  Param tables    : {config.param_tables_dir}")
    print(f"  Output dir      : {config.output_dir}")
    print(f"  Output mode     : {config.output_mode}")
    print(f"  Device          : {config.compute_device}")
    print(f"  Precision       : {config.float_precision}")
    print(f"  Projection      : {config.MAX_PROJ_YEARS} years ({config.MAX_PROJ_MONTHS} months)")
    print(f"  Batch size      : {config.batch_size:,}")
    print()

    # ------------------------------------------------------------------
    # Pre-flight checks for per-policy output mode
    # ------------------------------------------------------------------
    if config.output_mode in ("per_policy", "both"):
        from ulp_model.loader import PolicyBatchIterator
        import torch

        iterator = PolicyBatchIterator(
            config,
            config.batch_size,
            torch.device(config.compute_device),
            torch.float64 if config.float_precision == "float64" else torch.float32,
        )

        # Excel-friendly size validation
        excel_max_rows = 1_000_000
        max_rows = iterator.n_policies * config.MAX_PROJ_MONTHS
        if max_rows > excel_max_rows:
            print(
                f"\n[ERROR] Per-policy output size exceeds Excel capacity:"
                f"\n  Total rows = {iterator.n_policies:,} policies × {config.MAX_PROJ_MONTHS} months = {max_rows:,} rows"
                f"\n  Excel limit = {excel_max_rows:,} rows"
                f"\n  Required max policies = {excel_max_rows // config.MAX_PROJ_MONTHS:,} (for {config.MAX_PROJ_YEARS} year projection)"
                f"\n  TERMINATING: Reduce n_policies or use output_mode='summary' for large portfolios.\n"
            )
            return 1

        if iterator.n_batches > 1:
            print(
                f"\n[ERROR] Per-policy output not supported for multi-batch portfolios:"
                f"\n  Current batches: {iterator.n_batches} batches"
                f"\n  Policies: {iterator.n_policies:,}"
                f"\n  Batch size: {config.batch_size:,}"
                f"\n  TERMINATING: Increase batch_size to >= {iterator.n_policies:,} or use output_mode='summary'.\n"
            )
            return 1

    # ------------------------------------------------------------------
    # Run model (always batched; single batch when n_policies <= batch_size)
    # ------------------------------------------------------------------
    from ulp_model.model import ULPModel
    from ulp_model.outputs import print_metrics, write_summary_outputs
 
    model = ULPModel(config)
    result = model.run_portfolio()
 
    # ------------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------------
    print_metrics(
        result["summary"],
        scenario_id=1,
        elapsed_time=result["elapsed"],
        ape=result["ape"],
    )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    if config.output_mode in ("summary", "both"):
        summary = result["summary"]

        # Filter to requested time steps
        if config.output_time_steps != "all" and isinstance(config.output_time_steps, list):
            ts = config.output_time_steps
            summary_out = {k: v[ts] for k, v in summary.items()}
        else:
            summary_out = summary

        write_summary_outputs(summary_out, scenario_id=1, output_dir=config.output_dir)
        print(f"  Summary output  → {config.output_dir}/summary_scen1.csv")

    if config.output_mode in ("per_policy", "both"):
        # Per-policy output requires the raw [B, T] tensors which are not
        # retained by run_portfolio (by design, to bound GPU memory).
        # Pre-flight checks have already validated Excel limits and single-batch requirement.
        from ulp_model.loader import PolicyBatchIterator, load_param_tables
        from ulp_model.outputs import write_per_policy_outputs
        import torch
 
        iterator = PolicyBatchIterator(
            config,
            config.batch_size,
            torch.device(config.compute_device),
            torch.float64 if config.float_precision == "float64" else torch.float32,
        )
        
        param_tables = load_param_tables(config)
        # Single batch (pre-validated): re-run with full [B, T] retention to obtain
        # per-policy tensors for CSV output
        for policies, _idx, _s, _e in iterator:
            single_result = model.run(
                policies, param_tables,
                retain_full_outputs=True,
            )
            # Get summary variable order: Part 2 then Part 3
            summary_keys = list(single_result["part2"].keys()) + list(single_result["part3"].keys())
            write_per_policy_outputs(
                single_result["part2"],
                single_result["part3"],
                policies.policy_id,
                summary_keys,
                scenario_id=1,
                output_dir=config.output_dir,
                n_scenarios=1,
            )
            print(f"  Per-policy output → {config.output_dir}/per_policy_scen1.csv")
 
    return 0


if __name__ == "__main__":
    sys.exit(main())
