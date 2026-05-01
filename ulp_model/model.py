"""
model.py - ULPModel orchestrates all three projection parts.
"""
from __future__ import annotations

import time

import torch

from .inputs import ParamTables, PolicyBatch
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
