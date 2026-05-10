"""
part3_cashflows.py - Part 3: Shareholder Cashflow Projection.

After the architecture refactor, the model is split into two stages:

  Stage 1 — Forward projection (forward_projection.ForwardProjection)
    Computes every quantity that depends only on past- or current-month state
    in one synchronous monthly walk. Produces:
      - Part 1 outputs (Policy Account Values)
      - Part 2 outputs (Decrements)
      - Part 3 Pass 1 outputs (forward cashflow components)

  Stage 2 — Cashflow finalisation (this module)
    Consumes the Stage 1 outputs and runs the three remaining passes:
      - Pass 2 (backward) : zeroising reserve
      - Pass 3 (forward)  : tax, total reserve, solvency capital requirement,
                            cashflow after SCR
      - Pass 4 (backward) : present value of cashflows after SCR and of
                            premium income

Two storage modes (matching ForwardProjection)
---------------------------------------------
CashflowProjection supports two storage modes via the `retain_full_outputs`
constructor flag:

  retain_full_outputs=False   (summary mode, default)
      Pass 2/3/4 output tensors that are not consumed by downstream passes
      are kept as [B] current-month scratch with [T] accumulators. Tensors
      with backward/forward recursion that need t and t-1 use [B, 2] rolling
      buffers. Tensors needed across the full Pass 2/3/4 dependency chain
      remain [B, T].

  retain_full_outputs=True    (per-policy mode)
      Every tensor is stored as full [B, T], identical to the legacy
      implementation. Used for per-policy CSV output and validation.
"""
from __future__ import annotations

import math

import torch

from .inputs import ParamTables, PolicyBatch
from .utils import lookup_lien_pc, pol_year_at_t


# Pass 2/3/4 keys aggregated into the portfolio summary (sum over batch).
_PASS234_SUMMARY_KEYS = (
    "zeroising_res_if", "cf_after_zv", "op_tax", "cf_after_tax",
    "tot_res_if", "solv_cap_req", "scr_inv_inc", "scr_inc_tax",
    "cf_after_scr", "pv_cf_after_scr", "pv_prem_inc",
)


class CashflowProjection:
    """Executes Passes 2, 3, 4 of the shareholder cashflow projection.

    Pass 1 outputs are passed in via `part3_pass1_outputs` (typically produced
    by ForwardProjection).
    """

    def __init__(
        self,
        policies: PolicyBatch,
        param_tables: ParamTables,
        config,
        part1_outputs: dict,
        part2_outputs: dict,
        part3_pass1_outputs: dict,
        retain_full_outputs: bool = False,
    ) -> None:
        self.policies = policies
        self.param_tables = param_tables
        self.config = config
        self.p1 = part1_outputs
        self.p2 = part2_outputs
        self.retain_full_outputs = retain_full_outputs

        B = policies.policy_id.shape[0]
        T = config.MAX_PROJ_MONTHS
        self.B = B
        self.T = T

        self.device = torch.device(config.compute_device)
        self.dtype = (
            torch.float64 if config.float_precision == "float64" else torch.float32
        )

        # ------------------------------------------------------------------
        # Adopt Part 3 Pass 1 outputs from the forward projection.
        # In retain_full_outputs=True these are full [B, T]; in summary mode
        # only the survivors (prem_inc_if, cf_before_zv, unit_res_end) are
        # [B, T] and the rest may be [B] scratch — Pass 2/3/4 only reads the
        # survivors so this is safe.
        # ------------------------------------------------------------------
        self.prem_inc_if    = part3_pass1_outputs["prem_inc_if"]
        self.basic_prem_if  = part3_pass1_outputs["basic_prem_if"]
        self.topup_prem_if  = part3_pass1_outputs["topup_prem_if"]
        self.op_init_exp_if = part3_pass1_outputs["op_init_exp_if"]
        self.op_ren_exp_if  = part3_pass1_outputs["op_ren_exp_if"]
        self.invt_exp_if    = part3_pass1_outputs["invt_exp_if"]
        self.comm_if        = part3_pass1_outputs["comm_if"]
        self.ovrd_if        = part3_pass1_outputs["ovrd_if"]
        self.death_outgo    = part3_pass1_outputs["death_outgo"]
        self.surr_outgo     = part3_pass1_outputs["surr_outgo"]
        self.mat_outgo      = part3_pass1_outputs["mat_outgo"]
        self.cog_term_adj   = part3_pass1_outputs["cog_term_adj"]
        self.unit_res_bgn   = part3_pass1_outputs["unit_res_bgn"]
        self.unit_res_end   = part3_pass1_outputs["unit_res_end"]
        self.unit_inc       = part3_pass1_outputs["unit_inc"]
        self.non_unit_inc   = part3_pass1_outputs["non_unit_inc"]
        self.cf_before_zv   = part3_pass1_outputs["cf_before_zv"]

        # ------------------------------------------------------------------
        # Allocate Pass 2/3/4 tensors.
        #
        # Pass-by-pass dependency analysis:
        #   Pass 2 backward writes zeroising_res_if[t]; Pass 3 reads
        #     zeroising_res_if[t] and zeroising_res_if[t-1].
        #     => zeroising_res_if needs [B, T] (Pass 3 walks all t).
        #   Pass 3 forward reads solv_cap_req[t-1]; nothing reads it after
        #     => could be [B, 2], but kept [B, T] in summary because it is
        #     summed (the [T] sum is computed on the fly so we can shrink it).
        #   pv_cf_after_scr / pv_prem_inc backward read [t+1] and write [t]
        #     => need [B, 2] in summary mode.
        # ------------------------------------------------------------------
        def _ft(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=self.dtype, device=self.device)

        # Pass 2 output: zeroising_res_if -- read by Pass 3 at t and t-1
        # for arbitrary t. Pass 3 walks the full range, so this needs [B, T].
        self.zeroising_res_if = _ft(B, T)

        if retain_full_outputs:
            # All Pass 3/4 outputs as [B, T]
            self.cf_after_zv     = _ft(B, T)
            self.op_tax          = _ft(B, T)
            self.cf_after_tax    = _ft(B, T)
            self.tot_res_if      = _ft(B, T)
            self.solv_cap_req    = _ft(B, T)
            self.scr_inv_inc     = _ft(B, T)
            self.scr_inc_tax     = _ft(B, T)
            self.cf_after_scr    = _ft(B, T)
            self.pv_cf_after_scr = _ft(B, T)
            self.pv_prem_inc     = _ft(B, T)
            self._summary: dict[str, torch.Tensor] = {}
        else:
            # Summary mode: shrink tensors to minimal storage.
            #   - cf_after_scr: read by Pass 4 backward at any future t; needs [B, T]
            #   - solv_cap_req: read at t-1 inside Pass 3 forward; needs [B, 2]
            #   - all others: written and used only at t (within Pass 3) and
            #     summed to [T] accumulators -> [B] scratch.
            self.cf_after_zv     = _ft(B)
            self.op_tax          = _ft(B)
            self.cf_after_tax    = _ft(B)
            self.tot_res_if      = _ft(B)
            self.solv_cap_req    = _ft(B, 2)   # Pass 3 needs t-1
            self.scr_inv_inc     = _ft(B)
            self.scr_inc_tax     = _ft(B)
            # cf_after_scr: read by Pass 4 backward at arbitrary t -> keep [B, T]
            self.cf_after_scr    = _ft(B, T)
            # pv_cf_after_scr / pv_prem_inc: backward recursion reads t+1 only.
            # Once t=0 is computed, we want only t=0 retained for the summary.
            # Use [B, 2] rolling buffer that holds {t, t+1} during the backward
            # walk, plus [B] tensors holding the t=0 row at the end.
            self.pv_cf_after_scr = _ft(B, 2)
            self.pv_prem_inc     = _ft(B, 2)

            # On-the-fly summary accumulators ([T] for each summarised key)
            self._summary = {k: _ft(T) for k in _PASS234_SUMMARY_KEYS}

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """Run the remaining three passes (2/3/4) and return result dict."""
        self._pass2_backward()
        self._pass3_forward()
        self._pass4_backward()
        return self._collect_outputs()

    # -----------------------------------------------------------------------
    # Pass 2 – backward (zeroising reserve)
    # zeroising_res_if remains [B, T] in both modes because Pass 3 reads it
    # at t and t-1 across the full forward walk.
    # -----------------------------------------------------------------------

    def _pass2_backward(self) -> None:
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B
        T = self.T

        m_vir = (1.0 + pt.ann_vir / 100.0) ** (1.0 / 12.0) - 1.0
        pol_term_months = pol.pol_term * 12  # [B]

        for t in range(T - 1, -1, -1):
            if t == 0:
                self.zeroising_res_if[:, t] = 0.0
                continue

            at_or_past_mat = (t >= pol_term_months).to(dtype)
            is_inforce_end_t = self.p1["is_inforce_end"][:, t]

            if t + 1 < T:
                zr_next = self.zeroising_res_if[:, t + 1]
                cf_next = self.cf_before_zv[:, t + 1]
            else:
                zr_next = torch.zeros(B, dtype=dtype, device=device)
                cf_next = torch.zeros(B, dtype=dtype, device=device)

            zr_val = torch.clamp(
                (zr_next - cf_next) / (1.0 + m_vir), min=0.0
            )
            zr_val = zr_val * (1.0 - at_or_past_mat) * is_inforce_end_t
            self.zeroising_res_if[:, t] = zr_val

    # -----------------------------------------------------------------------
    # Pass 3 – forward (tax, SCR)
    # -----------------------------------------------------------------------

    def _pass3_forward(self) -> None:
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B
        T = self.T
        retain = self.retain_full_outputs

        m_sh_fer = (1.0 + pt.ann_sh_fer / 100.0) ** (1.0 / 12.0) - 1.0
        pol_term_months = pol.pol_term * 12  # [B]

        for t in range(0, T):
            pol_year = pol_year_at_t(t)
            py = min(pol_year, self.config.MAX_PROJ_YEARS)

            is_inforce_end_t = self.p1["is_inforce_end"][:, t]
            no_pols_if_t     = self.p2["no_pols_if"][:, t]

            # ---------------- S3.54: cashflow after zeroising ----------------
            zr_t = self.zeroising_res_if[:, t]
            if t > 0:
                zr_prev = self.zeroising_res_if[:, t - 1]
                # m_vir_prev recomputed inside loop for stochastic compatibility
                m_vir_prev = (1.0 + pt.ann_vir / 100.0) ** (1.0 / 12.0) - 1.0
            else:
                zr_prev = torch.zeros(B, dtype=dtype, device=device)
                m_vir_prev = 0.0

            cf_after_zv = (
                self.cf_before_zv[:, t]
                - zr_t
                + zr_prev * (1.0 + m_vir_prev)
            )
            self._write_t_only(self.cf_after_zv, t, cf_after_zv)

            # ---------------- S3.55-S3.57: operating tax ----------------
            op_tax = (pt.tax_pc / 100.0) * cf_after_zv
            self._write_t_only(self.op_tax, t, op_tax)
            cf_after_tax = cf_after_zv - op_tax
            self._write_t_only(self.cf_after_tax, t, cf_after_tax)

            # ---------------- S3.58: total reserve ----------------
            tot_res_if = self.unit_res_end[:, t] + zr_t
            self._write_t_only(self.tot_res_if, t, tot_res_if)

            # ---------------- S3.59: solvency capital requirement ----------------
            at_or_past_mat = (t >= pol_term_months).to(dtype)

            attained_age   = pol.age_at_entry + max(pol_year - 1, 0)
            lien_pc        = lookup_lien_pc(attained_age, pt.lien_table)
            current_db_opt = self.p1["current_db_opt"][:, t]
            bav_bval_bb_t  = self.p1["bav_bval_bb"][:, t]
            tuav_bval_bb_t = self.p1["tuav_bval_bb"][:, t]

            db1_pp = torch.maximum(pol.sum_assd * lien_pc, bav_bval_bb_t) + tuav_bval_bb_t
            db2_pp = pol.sum_assd * lien_pc + bav_bval_bb_t + tuav_bval_bb_t
            death_ben_pp_t = torch.where(current_db_opt == 1, db1_pp, db2_pp)

            sar_scr = torch.clamp(
                death_ben_pp_t * no_pols_if_t - tot_res_if, min=0.0
            )
            scr_raw = (
                (pt.solv_marg_res / 100.0) * tot_res_if
                + (pt.solv_marg_sar / 100.0) * sar_scr
            ) * is_inforce_end_t

            scr = scr_raw * (1.0 - at_or_past_mat) * float(t > 0)
            # scr lookback: t-1 in next iteration; need [B, 2] in summary, [B, T] in retain.
            if retain:
                self.solv_cap_req[:, t] = scr
            else:
                self.solv_cap_req[:, t % 2] = scr

            # ---------------- S3.62-S3.64: SCR investment income ----------------
            if t > 0:
                if retain:
                    scr_prev = self.solv_cap_req[:, t - 1]
                else:
                    scr_prev = self.solv_cap_req[:, (t - 1) % 2]
            else:
                scr_prev = torch.zeros(B, dtype=dtype, device=device)

            scr_inv_inc = scr_prev * m_sh_fer
            scr_inc_tax = (pt.tax_pc / 100.0) * scr_inv_inc
            self._write_t_only(self.scr_inv_inc, t, scr_inv_inc)
            self._write_t_only(self.scr_inc_tax, t, scr_inc_tax)

            cf_after_scr = (
                cf_after_tax
                + scr_prev
                - scr
                + scr_inv_inc
                - scr_inc_tax
            )
            # cf_after_scr always [B, T] because Pass 4 backward walks all t
            self.cf_after_scr[:, t] = cf_after_scr

            # On-the-fly summary aggregation
            if not retain:
                s = self._summary
                s["zeroising_res_if"][t] = zr_t.sum()
                s["cf_after_zv"][t]      = cf_after_zv.sum()
                s["op_tax"][t]           = op_tax.sum()
                s["cf_after_tax"][t]     = cf_after_tax.sum()
                s["tot_res_if"][t]       = tot_res_if.sum()
                s["solv_cap_req"][t]     = scr.sum()
                s["scr_inv_inc"][t]      = scr_inv_inc.sum()
                s["scr_inc_tax"][t]      = scr_inc_tax.sum()
                s["cf_after_scr"][t]     = cf_after_scr.sum()

    # -----------------------------------------------------------------------
    # Pass 4 – backward (PV of cashflows)
    #
    # Backward recursion: pv_cf_after_scr[t] = (pv_cf_after_scr[t+1] + cf_after_scr[t+1]) / (1+m_rdr)
    # In summary mode pv_* tensors are [B, 2] rolling buffers; the [T]
    # accumulator records the sum at each t as the backward walk passes through.
    # -----------------------------------------------------------------------

    def _pass4_backward(self) -> None:
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B
        T = self.T
        retain = self.retain_full_outputs

        m_rdr = (1.0 + pt.ann_rdr / 100.0) ** (1.0 / 12.0) - 1.0
        pol_term_months = pol.pol_term * 12  # [B]

        for t in range(T - 1, -1, -1):
            at_or_past_mat = (t >= pol_term_months).to(dtype)

            if t + 1 < T:
                if retain:
                    pv_next  = self.pv_cf_after_scr[:, t + 1]
                    pvp_next = self.pv_prem_inc[:, t + 1]
                else:
                    pv_next  = self.pv_cf_after_scr[:, (t + 1) % 2]
                    pvp_next = self.pv_prem_inc[:, (t + 1) % 2]
                cf_next  = self.cf_after_scr[:, t + 1]
                pi_next  = self.prem_inc_if[:, t + 1]
            else:
                pv_next  = torch.zeros(B, dtype=dtype, device=device)
                cf_next  = torch.zeros(B, dtype=dtype, device=device)
                pvp_next = torch.zeros(B, dtype=dtype, device=device)
                pi_next  = torch.zeros(B, dtype=dtype, device=device)

            pv_new  = (pv_next  + cf_next)  / (1.0 + m_rdr)
            pvp_new = (pvp_next + pi_next) / (1.0 + m_rdr)

            pv_t  = pv_new  * (1.0 - at_or_past_mat)
            pvp_t = pvp_new * (1.0 - at_or_past_mat)

            if retain:
                self.pv_cf_after_scr[:, t] = pv_t
                self.pv_prem_inc[:, t]     = pvp_t
            else:
                self.pv_cf_after_scr[:, t % 2] = pv_t
                self.pv_prem_inc[:, t % 2]     = pvp_t
                # Accumulate [T] summary
                self._summary["pv_cf_after_scr"][t] = pv_t.sum()
                self._summary["pv_prem_inc"][t]     = pvp_t.sum()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _write_t_only(self, buf: torch.Tensor, t: int, value: torch.Tensor) -> None:
        """Write at index t for [B, T] buffer, or copy_ for [B] buffer."""
        if buf.dim() == 2 and buf.shape[1] == self.T:
            buf[:, t] = value
        else:
            buf.copy_(value)

    # -----------------------------------------------------------------------
    # Collect results
    # -----------------------------------------------------------------------

    def _collect_outputs(self) -> dict:
        """Return Part 3 result dict.

        In retain_full_outputs=True every key maps to a full [B, T] tensor,
        identical to the legacy CashflowProjection.run() output.
        In summary mode some entries are [B] scratch / [B, 2] rolling buffers
        whose values are not meaningful — downstream consumers should rely
        on the 'summary' dict on ULPModel result for batch-aggregated outputs.
        """
        return {
            # Pass 1 outputs (passed through from ForwardProjection)
            "prem_inc_if":      self.prem_inc_if,
            "basic_prem_if":    self.basic_prem_if,
            "topup_prem_if":    self.topup_prem_if,
            "op_init_exp_if":   self.op_init_exp_if,
            "op_ren_exp_if":    self.op_ren_exp_if,
            "invt_exp_if":      self.invt_exp_if,
            "comm_if":          self.comm_if,
            "ovrd_if":          self.ovrd_if,
            "death_outgo":      self.death_outgo,
            "surr_outgo":       self.surr_outgo,
            "mat_outgo":        self.mat_outgo,
            "cog_term_adj":     self.cog_term_adj,
            "unit_res_bgn":     self.unit_res_bgn,
            "unit_res_end":     self.unit_res_end,
            "unit_inc":         self.unit_inc,
            "non_unit_inc":     self.non_unit_inc,
            "cf_before_zv":     self.cf_before_zv,
            # Pass 2 / 3 / 4 outputs
            "zeroising_res_if": self.zeroising_res_if,
            "cf_after_zv":      self.cf_after_zv,
            "op_tax":           self.op_tax,
            "cf_after_tax":     self.cf_after_tax,
            "tot_res_if":       self.tot_res_if,
            "solv_cap_req":     self.solv_cap_req,
            "scr_inv_inc":      self.scr_inv_inc,
            "scr_inc_tax":      self.scr_inc_tax,
            "cf_after_scr":     self.cf_after_scr,
            "pv_cf_after_scr":  self.pv_cf_after_scr,
            "pv_prem_inc":      self.pv_prem_inc,
        }

    def collect_summary(self) -> dict:
        """Return on-the-fly [T] summary accumulators (summary mode only).

        Returns an empty dict in per-policy mode; the orchestrator should
        compute summaries from the full [B, T] tensors in that case.
        """
        return dict(self._summary)
