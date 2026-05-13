"""
run_model.py - Command-line entry point for the ULP cash flow model.

Supports three output modes:
  - summary: aggregated portfolio results (fast, low memory)
  - per_policy: disaggregated policy-level results (slower, requires single batch)
  - both: summary + per_policy (requires single batch)

Policies are always processed in batches of config.batch_size (set in
config.yaml).  When the total number of policies is less than or equal
to batch_size the model runs as a single batch, which is equivalent to
the previous full-load behaviour.

Orchestrates multi-scenario execution based on scenario_file in config.yaml.
For each scenario, applies sensitivity factors to base param_tables, then runs model.

Sensitivity factors are applied to param_tables BEFORE the model runs, so
the model calculation formulas remain unchanged and clean.

If scenario_file is not provided or missing, runs with default sensitivity factors
(all 100/0, meaning no adjustment to base parameters).

Usage
-----
    python run_model.py                          # uses config.yaml in current directory
    python run_model.py --config path/to/config.yaml
    python run_model.py --config config.yaml --output-dir ./results/run1
    python run_model.py --config config.yaml --scenario-id 1
"""
from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override batch_size from config (policies per batch)",
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=None,
        metavar="N",
        help="Run only a specific scenario ID (default: run all scenarios)",
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

    print("=" * 80)
    print(f"ULP Model")
    print("=" * 80)
    print(f"  Config          : {config_path}")
    print(f"  Policy inputs   : {config.policy_inputs_file}")
    print(f"  Param tables    : {config.param_tables_dir}")
    print(f"  Scenario file   : {config.scenario_file}")
    print(f"  Output dir      : {config.output_dir}")
    print(f"  Output mode     : {config.output_mode}")
    print(f"  Device          : {config.compute_device}")
    print(f"  Precision       : {config.float_precision}")
    print(f"  Projection      : {config.MAX_PROJ_YEARS} years ({config.MAX_PROJ_MONTHS} months)")
    print(f"  Batch size      : {config.batch_size:,}")
    print()

    # ------------------------------------------------------------------
    # Load scenarios and sensitivity factors
    # ------------------------------------------------------------------
    import torch
    from ulp_model.sensitivity import load_sensitivity_scenarios, apply_sensitivity_factors
    from ulp_model.loader import load_param_tables, PolicyBatchIterator

    device = torch.device(config.compute_device)
    dtype = torch.float64 if config.float_precision == "float64" else torch.float32

    scenarios = {}
    scenario_file_path = None

    # Try to load scenario file if configured
    if hasattr(config, 'scenario_file') and config.scenario_file:
        scenario_file_path = Path(config.scenario_file)
        if scenario_file_path.exists():
            try:
                scenarios = load_sensitivity_scenarios(config.scenario_file)
                print(f"Loaded {len(scenarios)} scenario(s) from {config.scenario_file}")
            except Exception as e:
                print(f"[WARNING] Failed to load scenarios: {e}")
                print(f"          Running with default sensitivity factors (base case only)")
                scenarios = {1: None}  # Default: base case
        else:
            print(f"[WARNING] Scenario file not found: {config.scenario_file}")
            print(f"          Running with default sensitivity factors (base case only)")
            scenarios = {1: None}  # Default: base case
    else:
        print(f"[INFO] No scenario_file configured.")
        print(f"       Running with default sensitivity factors (base case only)")
        scenarios = {1: None}  # Default: base case

    # Filter to single scenario if --scenario-id specified
    if args.scenario_id is not None:
        if args.scenario_id not in scenarios:
            print(f"[ERROR] Scenario {args.scenario_id} not found")
            return 1
        scenarios = {args.scenario_id: scenarios[args.scenario_id]}
        print(f"Running only Scenario {args.scenario_id}")

    print()

    # ------------------------------------------------------------------
    # Pre-flight checks for per-policy output mode
    # ------------------------------------------------------------------
    if config.output_mode in ("per_policy", "both"):
        iterator = PolicyBatchIterator(config, config.batch_size, device, dtype)

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
    # Run model for each scenario (always batched; single batch when n_policies <= batch_size)
    # ------------------------------------------------------------------
    from ulp_model.model import ULPModel
    from ulp_model.outputs import (
        compute_ape,
        compute_metrics,
        print_metrics,
        write_summary_outputs,
        write_per_policy_outputs,
    )

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    scenario_results = []
    all_passed = True

    for scenario_id in sorted(scenarios.keys()):
        sensitivity = scenarios[scenario_id]

        print(f"─" * 80)
        print(f"Scenario {scenario_id}")
        print(f"─" * 80)

        t_scen_start = time.perf_counter()

        # Load base parameter tables
        param_tables = load_param_tables(config)

        # Apply sensitivity factors (if provided; else use defaults)
        if sensitivity is not None:
            apply_sensitivity_factors(param_tables, sensitivity, device, dtype)

        # Run model with sensitivity-adjusted param_tables
        model = ULPModel(config)
        single_result = None
        try:
            if config.output_mode in ("per_policy", "both"):
                # Single batch guaranteed by pre-flight check.
                # Run once with full outputs — no double-run needed.
                run_iterator = PolicyBatchIterator(config, config.batch_size, device, dtype)
                for policies, _idx, _s, _e in run_iterator:
                    single_result = model.run(policies, param_tables, retain_full_outputs=True)
                    ape = float(compute_ape(policies))
                summary = single_result["summary"]
            else:
                portfolio_result = model.run_portfolio(
                    retain_full_outputs=False,
                    param_tables=param_tables,
                )
                summary = portfolio_result["summary"]
                ape = portfolio_result["ape"]
        except Exception as e:
            print(f"[ERROR] Model execution failed for Scenario {scenario_id}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            continue

        t_scen_elapsed = time.perf_counter() - t_scen_start

        metrics = compute_metrics(summary, ape)
        pv_cf = metrics["pv_cf"]
        pv_prem = metrics["pv_prem"]
        pvcf_over_ape = metrics["pvcf_over_ape"]
        pvcf_over_pv_prem = metrics["pvcf_over_pv_prem"]

        # Determine output filename(s)
        n_digits = len(str(len(scenarios)))
        if config.output_mode in ("summary", "both"):
            fname_summary = f"summary_scen{scenario_id:0{n_digits}d}.csv"
        else:
            fname_summary = None

        if config.output_mode in ("per_policy", "both"):
            fname_per_policy = f"per_policy_scen{scenario_id:0{n_digits}d}.csv"
        else:
            fname_per_policy = None
        
        output_filename = fname_summary or fname_per_policy or "N/A"

        # Store results for the summary table
        scenario_results.append({
            "scenario_id": scenario_id,
            "elapsed_time": t_scen_elapsed,
            "ape": ape,
            "pv_cf": pv_cf,
            "pv_prem": pv_prem,
            "pvcf_over_ape": pvcf_over_ape,
            "pvcf_over_pv_prem": pvcf_over_pv_prem,
            "output_file": output_filename,
        })

        # Print per-scenario metrics (will be aggregated into table later)
        print_metrics(
            summary,
            scenario_id=scenario_id,
            elapsed_time=t_scen_elapsed,
            ape=ape,
        )

        # Write summary outputs
        if config.output_mode in ("summary", "both"):
            if config.output_time_steps != "all" and isinstance(config.output_time_steps, list):
                ts = config.output_time_steps
                summary_out = {k: v[ts] for k, v in summary.items()}
            else:
                summary_out = summary
            write_summary_outputs(
                summary_out,
                scenario_id=scenario_id,
                output_dir=config.output_dir,
                n_scenarios=len(scenarios),
            )
            print(f"  Summary output  → {Path(config.output_dir) / fname_summary}")

        # Write per-policy outputs
        if config.output_mode in ("per_policy", "both"):
            all_single_outputs = {
                **single_result.get("part1", {}),
                **single_result["part2"],
                **single_result["part3"],
            }
            if config.additional_output_vars == "all":
                output_keys = list(all_single_outputs.keys())
            else:
                output_keys = (
                    list(single_result["part2"].keys())
                    + list(single_result["part3"].keys())
                )
                if config.additional_output_vars:
                    seen = set(output_keys)
                    for var in config.additional_output_vars:
                        if var not in seen:
                            output_keys.append(var)
            write_per_policy_outputs(
                all_single_outputs,
                policies.policy_id,
                output_keys,
                scenario_id=scenario_id,
                output_dir=config.output_dir,
                n_scenarios=len(scenarios),
            )
            print(f"  Per-policy output → {fname_per_policy}")

    # ------------------------------------------------------------------
    # Print metrics table (console + CSV)
    # ------------------------------------------------------------------
    has_summary_metrics = config.output_mode in ("summary", "both")

    # Determine unit scaling: if any scenario APE exceeds 1 billion, show in millions.
    use_millions = (
        has_summary_metrics
        and any(r["ape"] >= 1_000_000_000 for r in scenario_results)
    )
    scale = 1_000_000 if use_millions else 1
    ape_label  = "APE (in mil)"  if use_millions else "APE"
    pvcf_label = "PV CF (in mil)" if use_millions else "PV CF"

    if has_summary_metrics:
        header     = ["Scenario ID", "Elapsed (s)", "Output File", ape_label, pvcf_label, "PV CF / APE", "PV CF / PV Prem"]
        col_widths = [12,             12,             40,            20,         20,          15,            16]
    else:
        header     = ["Scenario ID", "Elapsed (s)", "Output File"]
        col_widths = [12,             12,             40]

    table_width = sum(col_widths) + 3 * (len(col_widths) - 1)
    print("=" * table_width)
    print("Scenario Metrics Summary")
    print("=" * table_width)

    if scenario_results:
        header_str = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
        print(header_str)
        print("─" * len(header_str))

        for r in scenario_results:
            base_cells = [
                f"{r['scenario_id']}",
                f"{r['elapsed_time']:.2f}",
                r["output_file"],
            ]
            if has_summary_metrics:
                metric_cells = [
                    f"{r['ape'] / scale:,.2f}",
                    f"{r['pv_cf'] / scale:,.2f}",
                    f"{r['pvcf_over_ape']:.4f}",
                    f"{r['pvcf_over_pv_prem']:.4f}",
                ]
            else:
                metric_cells = []
            row_str = " | ".join(
                c.ljust(w) for c, w in zip(base_cells + metric_cells, col_widths)
            )
            print(row_str)

        # Save to CSV (same columns, no thousands separators)
        csv_filename = Path(config.output_dir) / "scenario_metrics_summary.csv"
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in scenario_results:
                base_cells = [r["scenario_id"], f"{r['elapsed_time']:.2f}", r["output_file"]]
                if has_summary_metrics:
                    metric_cells = [
                        f"{r['ape'] / scale:.2f}",
                        f"{r['pv_cf'] / scale:.2f}",
                        f"{r['pvcf_over_ape']:.4f}",
                        f"{r['pvcf_over_pv_prem']:.4f}",
                    ]
                else:
                    metric_cells = []
                writer.writerow(base_cells + metric_cells)
        print(f"\nMetrics saved to: {csv_filename}")

    print()

    # ------------------------------------------------------------------
    # Print execution summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("Execution Summary")
    print("=" * 80)

    if all_passed:
        print(f"✓ All {len(scenarios)} scenario(s) completed successfully")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Metrics summary:  {Path(config.output_dir) / 'scenario_metrics_summary.csv'}")
    else:
        print(f"✗ One or more scenarios failed")
        return 1

    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
