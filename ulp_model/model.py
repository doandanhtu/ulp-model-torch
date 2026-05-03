"""
model.py - ULPModel orchestrates all three projection parts.
"""
from __future__ import annotations

import time

import torch

from .inputs import ParamTables, PolicyBatch
from .loader import PolicyBatchIterator, load_param_tables
from .part1_pav import PAVProjection
from .part2_decrements import DecrementProjection
from .part3_cashflows import CashflowProjection


class ULPModel:
    """Universal Life Policy cash flow projection model.

    Orchestrates Part 1 (PAV), Part 2 (Decrements), and Part 3 (Cashflows).
    """

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.compute_device)
        self.dtype = (
            torch.float64 if config.float_precision == "float64" else torch.float32
        )

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        policies: PolicyBatch,
        param_tables: ParamTables,
    ) -> dict:
        """Run the full model and return aggregated outputs.

        Parameters
        ----------
        policies     : PolicyBatch with B policies
        param_tables : ParamTables with all rate/param data

        Returns
        -------
        dict containing:
          - 'part1'      : raw Part 1 tensor dict [B, T]
          - 'part2'      : raw Part 2 tensor dict [B, T]
          - 'part3'      : raw Part 3 tensor dict [B, T]
          - 'summary'    : aggregated outputs summed over batch [T]
          - 'elapsed'    : float seconds wall-clock time
        """
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Part 1: Policy Account Values
        # ------------------------------------------------------------------
        pav = PAVProjection(policies, param_tables, self.config)
        part1_outputs = pav.run()

        # ------------------------------------------------------------------
        # Part 2: Decrements
        # ------------------------------------------------------------------
        dec = DecrementProjection(policies, param_tables, self.config, part1_outputs)
        part2_outputs = dec.run()

        # ------------------------------------------------------------------
        # Part 3: Shareholder Cashflows
        # ------------------------------------------------------------------
        cf = CashflowProjection(
            policies, param_tables, self.config, part1_outputs, part2_outputs
        )
        part3_outputs = cf.run()

        elapsed = time.perf_counter() - t0

        summary = self._aggregate_outputs(part1_outputs, part2_outputs, part3_outputs)

        return {
            "part1":   part1_outputs,
            "part2":   part2_outputs,
            "part3":   part3_outputs,
            "summary": summary,
            "elapsed": elapsed,
        }

    # -----------------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------------

    def _aggregate_outputs(
        self,
        part1: dict,
        part2: dict,
        part3: dict,
    ) -> dict:
        """Sum selected Part 2 and Part 3 outputs over batch dimension [B] → [T]."""

        p2_keys = [
            "no_pols_if",
            "no_pols_ifsm",
            "no_deaths",
            "no_surrs",
            "no_mats",
        ]
        p3_keys = [
            "prem_inc_if",
            "basic_prem_if",
            "topup_prem_if",
            "op_init_exp_if",
            "op_ren_exp_if",
            "invt_exp_if",
            "comm_if",
            "ovrd_if",
            "death_outgo",
            "surr_outgo",
            "mat_outgo",
            "cog_term_adj",
            "unit_res_bgn",
            "unit_res_end",
            "unit_inc",
            "non_unit_inc",
            "cf_before_zv",
            "zeroising_res_if",
            "cf_after_zv",
            "op_tax",
            "cf_after_tax",
            "tot_res_if",
            "solv_cap_req",
            "scr_inv_inc",
            "scr_inc_tax",
            "cf_after_scr",
            "pv_cf_after_scr",
            "pv_prem_inc",
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
 
    def run_portfolio(self) -> dict:
        """Run the model over all policies using batched iteration.
 
        Policies are loaded and processed in chunks of config.batch_size.
        When the total number of policies is <= batch_size this is equivalent
        to a single model.run() call.
 
        The portfolio-level summary is accumulated on the fly: after each
        batch the per-batch [T] summary vectors are added in-place to a
        running total, so only one batch of [B, T] tensors exists on the
        device at any one time.
 
        Returns
        -------
        dict containing:
          - 'summary'     : portfolio-aggregated [T] tensors (sum over all policies)
          - 'ape'         : total Annual Premium Equivalent across all policies
          - 'n_policies'  : total number of policies processed
          - 'n_batches'   : number of batches used
          - 'elapsed'     : total wall-clock seconds
          - 'batch_times' : list of per-batch elapsed seconds
        """
        from .outputs import compute_ape
 
        batch_size = self.config.batch_size
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
            result = self.run(policies, param_tables)
            elapsed_batch = time.perf_counter() - t0
            batch_times.append(elapsed_batch)
 
            # Accumulate APE across batches
            total_ape += compute_ape(policies)
 
            # Accumulate summary in-place: add this batch's [T] vectors to the
            # running portfolio total.  No per-batch [B, T] tensors are retained.
            self._accumulate_summary(portfolio_summary, result["summary"])
            del result                     # drop [B, T] tensors immediately
            if self.device.type == "cuda":
                torch.cuda.empty_cache()  # return freed pages to CUDA allocator, preventing fragmentation
 
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
        """Add batch_summary [T] tensors into portfolio dict in-place.
 
        On the first batch the portfolio dict is empty so tensors are cloned
        in.  On subsequent batches they are added with +=.
        """
        for key, value in batch_summary.items():
            if key in portfolio:
                portfolio[key] += value
            else:
                portfolio[key] = value.clone()