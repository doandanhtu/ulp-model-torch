"""
model.py - ULPModel orchestrator.

Two-stage execution:

  Stage 1 — Forward projection
    ForwardProjection runs Part 1 (PAV) + Part 2 (Decrements) +
    Part 3 Pass 1 (forward cashflow components) in a single monthly loop.

  Stage 2 — Cashflow finalisation
    CashflowProjection consumes the Stage 1 outputs and runs the three
    remaining passes (zeroising backward, tax/SCR forward, PV backward).

Storage modes
-------------
ULPModel.run() supports two storage modes via the `retain_full_outputs`
parameter (defaulting to False):

  retain_full_outputs=False (summary mode, default)
    All non-survivor tensors are stored as rolling buffers / [B] scratch.
    Per-month [T] summary accumulators are populated on the fly. The result
    'summary' is built from these accumulators rather than from full [B, T]
    reductions. Memory-efficient — typically ~37% less per-policy storage.

  retain_full_outputs=True (per-policy mode)
    All tensors are full [B, T], identical to the legacy implementation.
    Used for per-policy CSV output and validation. Result 'summary' is
    built from full [B, T] -> [T] reductions.
"""
from __future__ import annotations

import time

import torch

from .forward_projection import ForwardProjection
from .inputs import ParamTables, PolicyBatch
from .loader import PolicyBatchIterator, load_param_tables
from .part3_cashflows import CashflowProjection


class ULPModel:
    """Universal Life Policy cash flow projection model.

    Orchestrates the two-stage pipeline:
      1. ForwardProjection (Parts 1, 2, and Part 3 Pass 1)
      2. CashflowProjection (Part 3 Pass 2/3/4)
    """

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.compute_device)
        self.dtype = (
            torch.float64 if config.float_precision == "float64" else torch.float32
        )

    # -----------------------------------------------------------------------
    # Main entry point — single batch
    # -----------------------------------------------------------------------

    def run(
        self,
        policies: PolicyBatch,
        param_tables: ParamTables,
        retain_full_outputs: bool = False,
    ) -> dict:
        """Run the full model on one batch.

        Parameters
        ----------
        policies            : PolicyBatch with B policies
        param_tables        : ParamTables with all rate/param data
        retain_full_outputs : if True, allocate all tensors as full [B, T]
                              (per-policy mode); if False, use rolling
                              buffers + on-the-fly [T] summary aggregation.

        Returns
        -------
        dict containing:
          - 'part1'      : Part 1 tensor dict
          - 'part2'      : Part 2 tensor dict
          - 'part3'      : Part 3 tensor dict
          - 'summary'    : aggregated outputs summed over batch [T]
                           - in summary mode: built from on-the-fly accumulators
                           - in per-policy mode: built from [B, T] -> [T] reductions
          - 'elapsed'    : float seconds wall-clock time
        """
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1: Forward projection (Part 1 + Part 2 + Part 3 Pass 1)
        # ------------------------------------------------------------------
        forward = ForwardProjection(
            policies, param_tables, self.config,
            retain_full_outputs=retain_full_outputs,
        )
        forward_out = forward.run()
        part1_outputs       = forward_out["part1"]
        part2_outputs       = forward_out["part2"]
        part3_pass1_outputs = forward_out["part3_pass1"]
        forward_summary     = forward_out["summary"]  # empty in retain mode

        # ------------------------------------------------------------------
        # Stage 2: Cashflow finalisation (Pass 2 / 3 / 4)
        # ------------------------------------------------------------------
        cf = CashflowProjection(
            policies,
            param_tables,
            self.config,
            part1_outputs,
            part2_outputs,
            part3_pass1_outputs,
            retain_full_outputs=retain_full_outputs,
        )
        part3_outputs = cf.run()
        cf_summary = cf.collect_summary()  # empty in retain mode

        elapsed = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # Build summary dict
        # ------------------------------------------------------------------
        if retain_full_outputs:
            summary = self._aggregate_outputs_full(part2_outputs, part3_outputs)
        else:
            summary = {**forward_summary, **cf_summary}

        return {
            "part1":   part1_outputs,
            "part2":   part2_outputs,
            "part3":   part3_outputs,
            "summary": summary,
            "elapsed": elapsed,
        }

    # -----------------------------------------------------------------------
    # Aggregation (per-policy / retain_full_outputs mode only)
    # -----------------------------------------------------------------------

    def _aggregate_outputs_full(
        self,
        part2: dict,
        part3: dict,
    ) -> dict:
        """Sum selected Part 2 and Part 3 outputs over batch dimension [B] -> [T].

        Used only in retain_full_outputs=True mode where all tensors are [B, T].
        """
        p2_keys = [
            "no_pols_if",
            "no_pols_ifsm",
            "no_deaths",
            "no_surrs",
            "no_mats",
        ]
        p3_keys = [
            "prem_inc_if", "basic_prem_if", "topup_prem_if",
            "op_init_exp_if", "op_ren_exp_if", "invt_exp_if",
            "comm_if", "ovrd_if",
            "death_outgo", "surr_outgo", "mat_outgo", "cog_term_adj",
            "unit_res_bgn", "unit_res_end", "unit_inc", "non_unit_inc",
            "cf_before_zv",
            "zeroising_res_if", "cf_after_zv", "op_tax", "cf_after_tax",
            "tot_res_if", "solv_cap_req", "scr_inv_inc", "scr_inc_tax",
            "cf_after_scr",
            "pv_cf_after_scr", "pv_prem_inc",
        ]

        summary: dict = {}
        for k in p2_keys:
            if k in part2:
                summary[k] = part2[k].sum(dim=0)  # [T]
        for k in p3_keys:
            if k in part3:
                summary[k] = part3[k].sum(dim=0)  # [T]

        return summary

    # -----------------------------------------------------------------------
    # Portfolio (batched) entry point
    # -----------------------------------------------------------------------

    def run_portfolio(self, retain_full_outputs: bool = False, param_tables=None) -> dict:
        """Run the model over all policies using batched iteration.

        Policies are loaded and processed in chunks of config.batch_size.
        When the total number of policies is <= batch_size this is equivalent
        to a single model.run() call.

        The portfolio-level summary is accumulated on the fly: after each
        batch the per-batch [T] summary vectors are added in-place to a
        running total, so only one batch of [B, *] tensors exists on the
        device at any one time.

        Parameters
        ----------
        retain_full_outputs : if True, run each batch with full [B, T] storage
                              (only feasible for a single-batch portfolio).
                              Default is False (summary mode).

        Returns
        -------
        dict containing:
          - 'summary'     : portfolio-aggregated [T] tensors
          - 'ape'         : total Annual Premium Equivalent across all policies
          - 'n_policies'  : total number of policies processed
          - 'n_batches'   : number of batches used
          - 'elapsed'     : total wall-clock seconds
          - 'batch_times' : list of per-batch elapsed seconds
        """
        from .outputs import compute_ape

        batch_size = self.config.batch_size
        if param_tables is None:
            param_tables = load_param_tables(self.config)
        iterator = PolicyBatchIterator(self.config, batch_size, self.device, self.dtype)

        print(
            f"Portfolio: {iterator.n_policies:,} policies | "
            f"{iterator.n_batches} batch(es) of {batch_size:,}"
        )

        portfolio_summary: dict = {}
        total_ape: float = 0.0
        batch_times: list[float] = []

        t0_total = time.perf_counter()

        for policies, batch_idx, start, end in iterator:
            n_pol = end - start
            print(
                f"  Batch {batch_idx + 1}/{iterator.n_batches} "
                f"(rows {start:,}-{end:,}, {n_pol:,} policies)...",
                end=" ",
                flush=True,
            )

            t0 = time.perf_counter()
            result = self.run(
                policies, param_tables,
                retain_full_outputs=retain_full_outputs,
            )
            elapsed_batch = time.perf_counter() - t0
            batch_times.append(elapsed_batch)

            total_ape += compute_ape(policies)

            self._accumulate_summary(portfolio_summary, result["summary"])
            del result
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            throughput = n_pol / elapsed_batch if elapsed_batch > 0 else float("inf")
            print(f"done ({elapsed_batch:.2f}s, {throughput:,.0f} pol/s)")

        total_elapsed = time.perf_counter() - t0_total
        print(
            f"Portfolio complete: {iterator.n_policies:,} policies in "
            f"{total_elapsed:.2f}s  ({iterator.n_policies / total_elapsed:,.0f} pol/s avg)"
        )

        return {
            "summary":     portfolio_summary,
            "ape":         total_ape,
            "n_policies":  iterator.n_policies,
            "n_batches":   iterator.n_batches,
            "elapsed":     total_elapsed,
            "batch_times": batch_times,
        }

    @staticmethod
    def _accumulate_summary(portfolio: dict, batch_summary: dict) -> None:
        """Add batch_summary [T] tensors into portfolio dict in-place."""
        for key, value in batch_summary.items():
            if key in portfolio:
                portfolio[key] += value
            else:
                portfolio[key] = value.clone()
