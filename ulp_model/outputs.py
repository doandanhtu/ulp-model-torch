"""
outputs.py - Output writing and reporting utilities for ULP model.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch

from .inputs import PolicyBatch


# ---------------------------------------------------------------------------
# APE computation
# ---------------------------------------------------------------------------

def compute_ape(policies: PolicyBatch) -> float:
    """Compute Annual Premium Equivalent (APE) for the policy batch.

    Rules:
    - Non-single-pay (prem_term > 1 OR prem_freq != 0/annual):
        100% of ACP * init_pols_if
    - Single-pay (prem_term == 1 AND prem_freq == 0/annual):
        10% of ACP * init_pols_if
    - Top-up premium: always 10% of ATP * init_pols_if
    """
    is_single_pay = (
        (policies.prem_term == 1)
        & (policies.prem_freq == 0)
    )
    basic_factor = torch.where(
        is_single_pay,
        torch.full_like(policies.acp, 0.10),
        torch.full_like(policies.acp, 1.00),
    )
    basic_ape = (basic_factor * policies.acp * policies.init_pols_if).sum().item()
    topup_ape = (0.10 * policies.atp * policies.init_pols_if).sum().item()
    return basic_ape + topup_ape


# ---------------------------------------------------------------------------
# Console metrics
# ---------------------------------------------------------------------------

def print_metrics(
    summary_data: dict,
    policies: PolicyBatch,
    scenario_id: int,
    elapsed_time: float,
) -> None:
    """Print a concise summary of key projection metrics."""
    ape = compute_ape(policies)

    # Scalar metrics: value at t=0 (or sum across all months)
    def _t0(key: str) -> float:
        if key in summary_data:
            v = summary_data[key]
            return float(v[0]) if v.ndim >= 1 else float(v)
        return float("nan")

    def _total(key: str) -> float:
        if key in summary_data:
            return float(summary_data[key].sum())
        return float("nan")

    pv_cf = _t0("pv_cf_after_scr")
    pv_prem = _t0("pv_prem_inc")
    total_prem = _total("prem_inc_if")
    total_death = _total("death_outgo")
    total_surr = _total("surr_outgo")

    print(f"\n{'='*60}")
    print(f" Scenario {scenario_id:>4d} | Elapsed: {elapsed_time:.2f}s")
    print(f"{'='*60}")
    print(f"  APE                : {ape:>20,.0f}")
    print(f"  PV Cashflow (t=0)  : {pv_cf:>20,.0f}")
    print(f"  PV Prem Inc (t=0)  : {pv_prem:>20,.0f}")
    print(f"  Total Prem Inc     : {total_prem:>20,.0f}")
    print(f"  Total Death Outgo  : {total_death:>20,.0f}")
    print(f"  Total Surr Outgo   : {total_surr:>20,.0f}")
    if pv_prem != 0.0 and not math.isnan(pv_prem):
        vif_over_ape = pv_cf / ape if ape != 0.0 else float("nan")
        print(f"  VIF / APE          : {vif_over_ape:>20.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def write_summary_outputs(
    summary_data: dict,
    scenario_id: int,
    output_dir: str,
    n_scenarios: int = 1,
) -> None:
    """Write summary (batch-aggregated) outputs to CSV.

    Parameters
    ----------
    summary_data : dict of {key: [T] tensor}
    scenario_id  : integer scenario identifier
    output_dir   : directory path for output files
    n_scenarios  : total number of scenarios (used for file naming)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_digits = len(str(n_scenarios))
    fname = out_path / f"summary_scen{scenario_id:0{n_digits}d}.csv"

    keys = sorted(summary_data.keys())
    T = max(v.shape[0] for v in summary_data.values() if hasattr(v, "shape"))

    with open(fname, "w", newline="", encoding="utf-8") as f:
        # Header
        f.write("t," + ",".join(keys) + "\n")
        for t in range(T):
            row = [str(t)]
            for k in keys:
                v = summary_data[k]
                row.append(f"{float(v[t]):.6f}" if t < v.shape[0] else "0")
            f.write(",".join(row) + "\n")


# ---------------------------------------------------------------------------
# Per-policy output
# ---------------------------------------------------------------------------

def write_per_policy_outputs(
    part1: dict,
    part2: dict,
    part3: dict,
    policy_ids: torch.Tensor,
    scenario_id: int,
    output_dir: str,
    output_batch_size: int = 1000,
) -> None:
    """Write per-policy outputs in batches to avoid memory exhaustion.

    Each output file contains a subset of policies. The key variables
    written are pv_cf_after_scr and pv_prem_inc (at t=0), plus a
    selection of time-series data.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    B = policy_ids.shape[0]
    n_batches = math.ceil(B / output_batch_size)

    ts_keys_p3 = ["prem_inc_if", "cf_before_zv", "cf_after_scr"]

    for batch_idx in range(n_batches):
        start = batch_idx * output_batch_size
        end = min(start + output_batch_size, B)
        pids = policy_ids[start:end]
        T = part3["pv_cf_after_scr"].shape[1]

        fname = out_path / f"per_policy_scen{scenario_id:04d}_batch{batch_idx:04d}.csv"
        with open(fname, "w", newline="", encoding="utf-8") as f:
            header_parts = ["policy_id", "pv_cf_after_scr_t0", "pv_prem_inc_t0"]
            for k in ts_keys_p3:
                for t in range(min(T, 24)):
                    header_parts.append(f"{k}_t{t}")
            f.write(",".join(header_parts) + "\n")

            for i in range(end - start):
                b = start + i
                pid_val = int(pids[i])
                pv_cf = float(part3["pv_cf_after_scr"][b, 0])
                pv_pi = float(part3["pv_prem_inc"][b, 0])
                row = [str(pid_val), f"{pv_cf:.6f}", f"{pv_pi:.6f}"]
                for k in ts_keys_p3:
                    for t in range(min(T, 24)):
                        row.append(f"{float(part3[k][b, t]):.6f}")
                f.write(",".join(row) + "\n")
