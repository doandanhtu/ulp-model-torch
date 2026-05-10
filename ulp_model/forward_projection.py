"""
forward_projection.py - Single-pass monthly forward projection.

Computes every model quantity that depends only on past- or current-month
state, in one synchronous walk through the projection horizon. This covers:

  - Part 1: Policy Account Values (PAV) — premium allocation, deductions,
            unit fund growth with cost-of-guarantee, bonuses, and the
            beginning- and end-of-month account values.
  - Part 2: Decrements — number-of-policies-in-force, deaths, surrenders,
            and maturities.
  - Part 3 Pass 1: Forward cashflow components — premium income, expenses,
                   commissions, benefit outgoes, COG on terminations,
                   unit/non-unit income, and `cf_before_zv`.

The remaining Part 3 work (Pass 2 backward, Pass 3 forward, Pass 4 backward)
runs after this engine completes; see part3_cashflows.CashflowProjection.

Two storage modes
-----------------
ForwardProjection supports two storage modes, controlled by the
`retain_full_outputs` constructor flag:

  retain_full_outputs=False   (summary mode, default)
      All tensors that are NOT consumed by Pass 2/3/4 are stored as either:
        * rolling [B, W] buffers when they have a short lookback window
          (e.g. bav_ab needs only t-1, so W=2; av_ad needs t, t-1, t-2 so W=3;
          bonus-lookback tensors use W=max_lookback_N+1).
        * [B] current-month-only scratch when they are read only at index t.
      Tensors that feed into the portfolio summary are accumulated to [T]
      vectors on the fly via .sum(dim=0) reductions per month.
      Total per-policy storage is ~37% less than legacy. See VRAM analysis.

  retain_full_outputs=True    (per-policy mode)
      Every tensor is stored as full [B, T], exactly as in the legacy
      PAVProjection / DecrementProjection / CashflowProjection._pass1_forward.
      This mode is bit-identical to the legacy implementation and is intended
      for per-policy CSV output and validation. Used when the entire portfolio
      fits in a single batch.

Numerical equivalence
---------------------
Both modes are numerically equivalent to the legacy three-stage forward path.
In retain_full_outputs=True they are bit-identical (no rolling buffer logic
exercised). In retain_full_outputs=False the per-policy [T] outputs collected
from the [T] accumulators are bit-identical to summing the full [B, T]
tensors over the batch dimension.

Outputs
-------
.run() returns a dict with three sub-dicts:
    - 'part1'       : Part 1 [B, T] tensors (or rolling buffers + [T] accums
                      in summary mode) plus the [B, T] survivors needed by
                      Pass 2/3/4.
    - 'part2'       : Part 2 [B, T] tensors (or rolling buffers + [T] accums).
    - 'part3_pass1' : Part 3 Pass 1 [B, T] tensors (or [T] accums).
    - 'summary'     : (summary mode only) [T] aggregates accumulated on the fly.
"""
from __future__ import annotations

import math

import torch

from .inputs import ParamTables, PolicyBatch
from .utils import (
    attained_age_at_t,
    lookup_coi_rate,
    lookup_lien_pc,
    lookup_mortality_rate,
    pol_year_at_t,
    precompute_bonus_schedule,
)


# Tensors that are aggregated to [T] in summary mode (Part 1 portion).
# In legacy these are [B, T] within Part 1's output dict; downstream the model
# orchestrator never sums them (they're not in p1_keys of _aggregate_outputs).
# However m_ulp_fer is the one Part 1 tensor read by Part 3 Pass 1 inside the
# forward loop, so it gets special treatment (kept as [B] current-month).

# Part 1 tensors that survive to Pass 2/3/4 (must remain [B, T]):
_P1_SURVIVORS = ("is_inforce_end", "current_db_opt", "bav_bval_bb", "tuav_bval_bb")

# Part 2 tensors that survive to Pass 2/3/4 (must remain [B, T]):
_P2_SURVIVORS = ("no_pols_if",)

# Part 2 tensors aggregated to [T] in summary mode (sum over batch):
_P2_SUMMARY_KEYS = ("no_pols_if", "no_pols_ifsm", "no_deaths", "no_surrs", "no_mats")

# Part 3 Pass 1 tensors that survive to Pass 2/3/4 (must remain [B, T]):
_P3P1_SURVIVORS = ("prem_inc_if", "cf_before_zv", "unit_res_end")

# Part 3 Pass 1 tensors aggregated to [T] in summary mode:
_P3P1_SUMMARY_KEYS = (
    "prem_inc_if", "basic_prem_if", "topup_prem_if",
    "op_init_exp_if", "op_ren_exp_if", "invt_exp_if",
    "comm_if", "ovrd_if",
    "death_outgo", "surr_outgo", "mat_outgo", "cog_term_adj",
    "unit_res_bgn", "unit_res_end", "unit_inc", "non_unit_inc",
    "cf_before_zv",
)


class ForwardProjection:
    """Single-pass monthly forward projection: PAV + decrements + Pass 1 cashflows.

    Runs a single monthly loop t = 1..T-1 (with t=0 initialisation handled
    explicitly) computing PAV, decrements, and Pass 1 cashflows in lockstep.
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        policies: PolicyBatch,
        param_tables: ParamTables,
        config,
        retain_full_outputs: bool = False,
    ) -> None:
        self.policies = policies
        self.param_tables = param_tables
        self.config = config
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
        # Premium frequency (already stored in months by convention)
        # ------------------------------------------------------------------
        self.prem_freq_mths = policies.prem_freq
        self.topup_freq_mths = policies.topup_freq

        # ------------------------------------------------------------------
        # Bonus schedule precomputation (Part 1)
        # ------------------------------------------------------------------
        self.bonus_schedule = precompute_bonus_schedule(
            param_tables, config.MAX_PROJ_YEARS
        )
        self._basic_lb_months = set(self.bonus_schedule["basic_lb"]["award_months"])
        self._topup_lb_months = set(self.bonus_schedule["topup_lb"]["award_months"])
        self._sb_coi_months = set(self.bonus_schedule["sb_coi"]["award_months"])
        self._sb_acp_months = set(self.bonus_schedule["sb_acp"]["award_months"])

        # Maximum lookback window N over all bonus types and award months.
        # Used to size the rolling buffers for bav_aval_bb / tuav_aval_bb /
        # mort_coi. Add 1 for safety / readability (covers the inclusive
        # endpoint t).
        max_N = 1
        for key in ("basic_lb", "topup_lb", "sb_coi"):
            for n in self.bonus_schedule[key]["lookback_N"].values():
                if n > max_N:
                    max_N = n
        self._lookback_W = max_N  # buffer width for windowed bonus tensors

        # ------------------------------------------------------------------
        # Lapse-table column index per policy (Part 2)
        # cols: 0=monthly, 1=quarterly, 2=semiann, 3=annual
        # ------------------------------------------------------------------
        _m2c = torch.zeros(13, dtype=torch.long, device=self.device)
        _m2c[1] = 0; _m2c[3] = 1; _m2c[6] = 2; _m2c[12] = 3
        self.freq_col = _m2c[policies.prem_freq.long()]  # [B]

        # ------------------------------------------------------------------
        # Allocate output tensors. In retain_full_outputs=True we allocate
        # full [B, T] for every tensor so the output dicts are identical in
        # shape to the legacy implementation. In summary mode we use rolling
        # buffers and [T] accumulators.
        # ------------------------------------------------------------------
        self._allocate_tensors()

    def _allocate_tensors(self) -> None:
        B = self.B
        T = self.T
        W = self._lookback_W
        dtype = self.dtype
        device = self.device
        retain = self.retain_full_outputs

        def _ft(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=dtype, device=device)

        def _lt(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.long, device=device)

        # ==================================================================
        # PART 1 — survivors (always [B, T])
        # ==================================================================
        self.is_inforce_end   = _ft(B, T)
        self.current_db_opt   = _lt(B, T)
        self.bav_bval_bb      = _ft(B, T)
        self.tuav_bval_bb     = _ft(B, T)

        # ==================================================================
        # PART 1 — short-history (1-step lookback): [B, 2] in summary mode,
        # [B, T] in per-policy mode.
        # Read pattern: each is read at t-1 inside the forward loop.
        # ==================================================================
        if retain:
            self.bav_ab      = _ft(B, T)
            self.tuav_ab     = _ft(B, T)
            self.av_ab       = _ft(B, T)
            self.g_bav_ab    = _ft(B, T)
            self.g_tuav_ab   = _ft(B, T)
            self.tot_dedncf  = _ft(B, T)
        else:
            self.bav_ab      = _ft(B, 2)
            self.tuav_ab     = _ft(B, 2)
            self.av_ab       = _ft(B, 2)
            self.g_bav_ab    = _ft(B, 2)
            self.g_tuav_ab   = _ft(B, 2)
            self.tot_dedncf  = _ft(B, 2)

        # ==================================================================
        # PART 1 — 3-month lookback (av_ad: read at t, t-1, t-2)
        # ==================================================================
        if retain:
            self.av_ad = _ft(B, T)
        else:
            self.av_ad = _ft(B, 3)

        # ==================================================================
        # PART 1 — windowed lookback for bonus calculations:
        #   bav_aval_bb, tuav_aval_bb, mort_coi
        # Width W = max lookback N across all bonus-award months (typically 60).
        # ==================================================================
        if retain:
            self.bav_aval_bb  = _ft(B, T)
            self.tuav_aval_bb = _ft(B, T)
            self.mort_coi     = _ft(B, T)
        else:
            self.bav_aval_bb  = _ft(B, W)
            self.tuav_aval_bb = _ft(B, W)
            self.mort_coi     = _ft(B, W)

        # ==================================================================
        # PART 1 — t-only tensors (overwritten each month, no lookback):
        # [B] in summary mode, [B, T] in per-policy mode.
        # ==================================================================
        if retain:
            self.is_inforce_bgn     = _ft(B, T)
            self.basic_prem_pp      = _ft(B, T)
            self.topup_prem_pp      = _ft(B, T)
            self.basic_alloc_chg_pp = _ft(B, T)
            self.topup_alloc_chg_pp = _ft(B, T)
            self.bav_ad             = _ft(B, T)
            self.tuav_ad            = _ft(B, T)
            self.g_bav_bval         = _ft(B, T)
            self.g_tuav_bval        = _ft(B, T)
            self.bonus_alloc        = _ft(B, T)
            self.m_ulp_fer          = _ft(B, T)
            self.tot_dedn_act       = _ft(B, T)
        else:
            self.is_inforce_bgn     = _ft(B)
            self.basic_prem_pp      = _ft(B)
            self.topup_prem_pp      = _ft(B)
            self.basic_alloc_chg_pp = _ft(B)
            self.topup_alloc_chg_pp = _ft(B)
            self.bav_ad             = _ft(B)
            self.tuav_ad            = _ft(B)
            self.g_bav_bval         = _ft(B)
            self.g_tuav_bval        = _ft(B)
            self.bonus_alloc        = _ft(B)
            self.m_ulp_fer          = _ft(B)
            self.tot_dedn_act       = _ft(B)

        # ==================================================================
        # PART 1 — internal working scalars (only ever read at t):
        # [B] in summary mode, [B, T] in per-policy mode.
        # ==================================================================
        if retain:
            self._basic_dedn        = _ft(B, T)
            self._topup_dedn        = _ft(B, T)
            self._basic_dedncf_clrd = _ft(B, T)
            self._topup_dedncf_clrd = _ft(B, T)
        else:
            self._basic_dedn        = _ft(B)
            self._topup_dedn        = _ft(B)
            self._basic_dedncf_clrd = _ft(B)
            self._topup_dedncf_clrd = _ft(B)

        # ==================================================================
        # PART 2 — survivors (always [B, T])
        # ==================================================================
        self.no_pols_if   = _ft(B, T)

        # ==================================================================
        # PART 2 — 1-step lookback: no_mats (read at t-1 inside Part 2)
        # ==================================================================
        if retain:
            self.no_mats = _ft(B, T)
        else:
            self.no_mats = _ft(B, 2)

        # ==================================================================
        # PART 2 — t-only tensors:
        # ==================================================================
        if retain:
            self.no_pols_ifsm = _ft(B, T)
            self.no_deaths    = _ft(B, T)
            self.no_surrs     = _ft(B, T)
        else:
            self.no_pols_ifsm = _ft(B)
            self.no_deaths    = _ft(B)
            self.no_surrs     = _ft(B)

        # ==================================================================
        # PART 3 PASS 1 — survivors (always [B, T])
        # ==================================================================
        self.prem_inc_if    = _ft(B, T)
        self.cf_before_zv   = _ft(B, T)
        self.unit_res_end   = _ft(B, T)

        # ==================================================================
        # PART 3 PASS 1 — t-only tensors:
        # ==================================================================
        if retain:
            self.basic_prem_if  = _ft(B, T)
            self.topup_prem_if  = _ft(B, T)
            self.op_init_exp_if = _ft(B, T)
            self.op_ren_exp_if  = _ft(B, T)
            self.invt_exp_if    = _ft(B, T)
            self.comm_if        = _ft(B, T)
            self.ovrd_if        = _ft(B, T)
            self.death_outgo    = _ft(B, T)
            self.surr_outgo     = _ft(B, T)
            self.mat_outgo      = _ft(B, T)
            self.cog_term_adj   = _ft(B, T)
            self.unit_res_bgn   = _ft(B, T)
            self.unit_inc       = _ft(B, T)
            self.non_unit_inc   = _ft(B, T)
        else:
            self.basic_prem_if  = _ft(B)
            self.topup_prem_if  = _ft(B)
            self.op_init_exp_if = _ft(B)
            self.op_ren_exp_if  = _ft(B)
            self.invt_exp_if    = _ft(B)
            self.comm_if        = _ft(B)
            self.ovrd_if        = _ft(B)
            self.death_outgo    = _ft(B)
            self.surr_outgo     = _ft(B)
            self.mat_outgo      = _ft(B)
            self.cog_term_adj   = _ft(B)
            self.unit_res_bgn   = _ft(B)
            self.unit_inc       = _ft(B)
            self.non_unit_inc   = _ft(B)

        # ==================================================================
        # SUMMARY ACCUMULATORS — only allocated in summary mode.
        # Each is a [T] tensor; on each month we add the .sum(dim=0) of the
        # [B] current-month value (or [B, t] slice for the survivors).
        # ==================================================================
        if not retain:
            self._summary_p2: dict[str, torch.Tensor] = {
                k: _ft(T) for k in _P2_SUMMARY_KEYS
            }
            self._summary_p3p1: dict[str, torch.Tensor] = {
                k: _ft(T) for k in _P3P1_SUMMARY_KEYS
            }
        else:
            self._summary_p2 = {}
            self._summary_p3p1 = {}

    # =========================================================================
    # Helpers for rolling-buffer access
    # =========================================================================

    def _w_idx(self, t: int, W: int) -> int:
        """Position in a width-W rolling buffer for absolute month t."""
        return t % W

    def _read_buf(self, buf: torch.Tensor, t: int, W: int) -> torch.Tensor:
        """Read column for month t from a width-W rolling buffer."""
        if self.retain_full_outputs:
            return buf[:, t]
        return buf[:, t % W]

    def _write_buf(self, buf: torch.Tensor, t: int, W: int, value: torch.Tensor) -> None:
        """Write column for month t into a width-W rolling buffer."""
        if self.retain_full_outputs:
            buf[:, t] = value
        else:
            buf[:, t % W] = value

    def _read_t_only(self, buf: torch.Tensor, t: int) -> torch.Tensor:
        """Read a t-only tensor: [B] in summary mode, [B, t] slice in per-policy mode."""
        if self.retain_full_outputs:
            return buf[:, t]
        return buf

    def _write_t_only(self, buf: torch.Tensor, t: int, value: torch.Tensor) -> None:
        """Write a t-only tensor: [B] copy_ in summary mode, [B, t] in per-policy mode."""
        if self.retain_full_outputs:
            buf[:, t] = value
        else:
            buf.copy_(value)

    def _window_indices(self, t_start: int, t_end_inclusive: int, W: int) -> list[int]:
        """Return rolling-buffer column indices for window [t_start, t_end_inclusive]."""
        return [(t_start + k) % W for k in range(t_end_inclusive - t_start + 1)]

    # =========================================================================
    # Public interface
    # =========================================================================

    def run(self) -> dict:
        """Run the forward projection.

        Returns
        -------
        dict with keys:
            - 'part1'       : Part 1 tensor dict
            - 'part2'       : Part 2 tensor dict
            - 'part3_pass1' : Part 3 Pass 1 tensor dict
            - 'summary'     : (summary mode only) on-the-fly [T] aggregates
                              for all summable Part 2 + Part 3 Pass 1 keys.
                              Empty dict in per-policy mode.
        """
        # ------------------------------------------------------------------
        # t = 0 initialisation (matches PAVProjection.run() and
        # DecrementProjection.run() initial-month behaviour)
        # ------------------------------------------------------------------
        self.is_inforce_end[:, 0] = 1.0
        self.current_db_opt[:, 0] = self.policies.db_opt
        self.no_pols_if[:, 0] = self.policies.init_pols_if

        # In summary mode, contribute t=0 values of the survivors to the
        # accumulators so that the [T] summary matches a legacy
        # part2[k].sum(dim=0) over the FULL [B, T] tensor including t=0.
        if not self.retain_full_outputs:
            # no_pols_if at t=0: init_pols_if (already written above)
            self._summary_p2["no_pols_if"][0] = self.no_pols_if[:, 0].sum()
            # All other Part 2 / Part 3 Pass 1 tensors are zero at t=0; the
            # accumulators are zero-initialised so no extra writes needed.

        # ------------------------------------------------------------------
        # Forward time loop t = 1 .. T-1
        # ------------------------------------------------------------------
        for t in range(1, self.T):
            # 1) Part 1
            self._compute_part1_month(t)
            # 2) Part 2 (uses Part 1 is_inforce_end at t)
            self._compute_part2_month(t)
            # 3) Part 3 Pass 1 (uses Part 1 + Part 2 at t)
            self._compute_part3_pass1_month(t)

            # 4) Update on-the-fly summary accumulators (summary mode only)
            if not self.retain_full_outputs:
                self._accumulate_summary(t)

        return {
            "part1":       self._collect_part1(),
            "part2":       self._collect_part2(),
            "part3_pass1": self._collect_part3_pass1(),
            "summary":     self._collect_summary(),
        }

    # =========================================================================
    # Part 1 - Policy Account Value (identical formulas to PAVProjection)
    # =========================================================================

    def _compute_part1_month(self, t: int) -> None:  # noqa: C901 — by spec
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B
        W = self._lookback_W

        pol_year = pol_year_at_t(t)
        py = pol_year
        py_clamped = min(py, self.config.MAX_PROJ_YEARS)

        # Attained age [B] (long)
        attained_age = attained_age_at_t(pol.age_at_entry, t)

        # ---------------- S1.1: is_inforce_bgn ----------------
        past_term = (t > pol.pol_term * 12).to(dtype)
        is_inforce_bgn = self.is_inforce_end[:, t - 1] * (1.0 - past_term)
        self._write_t_only(self.is_inforce_bgn, t, is_inforce_bgn)

        # ---------------- S1.40: current_db_opt ----------------
        age_changed = (attained_age >= pt.age_db_opt_change).to(torch.long)
        self.current_db_opt[:, t] = torch.where(
            age_changed.bool(),
            torch.ones(B, dtype=torch.long, device=device),
            self.current_db_opt[:, t - 1],
        )
        current_db_opt = self.current_db_opt[:, t]

        # ---------------- S1.4-S1.5: premium per policy ----------------
        basic_due = (
            ((t - 1) % self.prem_freq_mths == 0).to(dtype)
            * (t <= pol.prem_term * 12).to(dtype)
        )
        basic_prem_pp = (
            (pol.acp * self.prem_freq_mths.to(dtype) / 12.0)
            * basic_due * is_inforce_bgn
        )
        self._write_t_only(self.basic_prem_pp, t, basic_prem_pp)

        topup_due = (
            ((t - 1) % self.topup_freq_mths == 0).to(dtype)
            * (t <= pol.topup_term * 12).to(dtype)
        )
        topup_prem_pp = (
            (pol.atp * self.topup_freq_mths.to(dtype) / 12.0)
            * topup_due * is_inforce_bgn
        )
        self._write_t_only(self.topup_prem_pp, t, topup_prem_pp)

        # ---------------- S1.6-S1.9: allocation charges ----------------
        basic_alloc_chg_pc = float(pt.alloc_chg_basic[py_clamped]) / 100.0
        topup_alloc_chg_pc = float(pt.alloc_chg_topup[py_clamped]) / 100.0

        basic_alloc_chg_pp = basic_alloc_chg_pc * basic_prem_pp
        topup_alloc_chg_pp = topup_alloc_chg_pc * topup_prem_pp
        self._write_t_only(self.basic_alloc_chg_pp, t, basic_alloc_chg_pp)
        self._write_t_only(self.topup_alloc_chg_pp, t, topup_alloc_chg_pp)

        # ---------------- S1.10-S1.11: allocated premium b4 cf ----------------
        basic_alloc_prem_b4 = basic_prem_pp - basic_alloc_chg_pp
        topup_alloc_prem_b4 = topup_prem_pp - topup_alloc_chg_pp

        # ---------------- S1.17-S1.18: carry-forward clearing ----------------
        prev_dedncf = self._read_buf(self.tot_dedncf, t - 1, 2)
        basic_dedncf_clrd = torch.minimum(prev_dedncf, basic_alloc_prem_b4)
        remaining_dedncf = prev_dedncf - basic_dedncf_clrd
        topup_dedncf_clrd = torch.minimum(remaining_dedncf, topup_alloc_prem_b4)
        self._write_t_only(self._basic_dedncf_clrd, t, basic_dedncf_clrd)
        self._write_t_only(self._topup_dedncf_clrd, t, topup_dedncf_clrd)

        # ---------------- S1.12-S1.13: allocated premium ----------------
        basic_alloc_prem = basic_alloc_prem_b4 - basic_dedncf_clrd
        topup_alloc_prem = topup_alloc_prem_b4 - topup_dedncf_clrd

        # ---------------- S1.14-S1.16: AV at beginning of deduction (bd) ----
        bav_ab_prev = self._read_buf(self.bav_ab, t - 1, 2)
        tuav_ab_prev = self._read_buf(self.tuav_ab, t - 1, 2)
        bav_bd = (bav_ab_prev + basic_alloc_prem) * is_inforce_bgn
        tuav_bd = (tuav_ab_prev + topup_alloc_prem) * is_inforce_bgn

        # ---------------- S1.28: admin charge ----------------
        admin_chg_pp = min(
            pt.admin_chg_start + pt.admin_chg_inc * (py - 1),
            pt.admin_chg_cap,
        )
        admin_chg = (
            torch.full((B,), admin_chg_pp, dtype=dtype, device=device)
            * is_inforce_bgn
        )

        # ---------------- S1.39: lien percentage ----------------
        lien_pc = lookup_lien_pc(attained_age, pt.lien_table)

        # ---------------- S1.35-S1.37: COI base & SAR ----------------
        bav_coi = torch.clamp(bav_bd - admin_chg, min=0.0)

        db1_coi = torch.maximum(pol.sum_assd * lien_pc, bav_coi)
        db2_coi = pol.sum_assd * lien_pc + bav_coi
        dthben_coi = torch.where(current_db_opt == 1, db1_coi, db2_coi)

        sar_mort = torch.clamp(dthben_coi - bav_coi, min=0.0) * is_inforce_bgn

        # S1.33: annual COI rate (per mille)
        ann_mort_coi = lookup_coi_rate(
            attained_age, pol.sex, pt.coi_table_male, pt.coi_table_female
        )
        # S1.32: monthly COI charge
        mort_coi = (
            (sar_mort / 1000.0) * (ann_mort_coi / 12.0)
            * (1.0 + pol.mort_loading / 100.0)
        )
        self._write_buf(self.mort_coi, t, W, mort_coi)

        # ---------------- S1.23-S1.27: deductions ----------------
        totdedn = mort_coi + admin_chg
        basic_dedn = torch.minimum(totdedn, bav_bd)
        topup_dedn = torch.minimum(totdedn - basic_dedn, tuav_bd)
        dedn = basic_dedn + topup_dedn
        self._write_t_only(self._basic_dedn, t, basic_dedn)
        self._write_t_only(self._topup_dedn, t, topup_dedn)

        # S1.41-S1.43: AV after deductions (ad)
        bav_ad = torch.clamp(bav_bd - basic_dedn, min=0.0)
        tuav_ad = torch.clamp(tuav_bd - topup_dedn, min=0.0)
        av_ad = bav_ad + tuav_ad
        self._write_t_only(self.bav_ad, t, bav_ad)
        self._write_t_only(self.tuav_ad, t, tuav_ad)
        # av_ad needs 3-month lookback (t, t-1, t-2)
        self._write_buf(self.av_ad, t, 3, av_ad)

        # ---------------- S1.2: is_inforce_end ----------------
        if t >= 3:
            av_ad_t  = self._read_buf(self.av_ad, t,     3)
            av_ad_p1 = self._read_buf(self.av_ad, t - 1, 3)
            av_ad_p2 = self._read_buf(self.av_ad, t - 2, 3)
            av_zero_3 = ((av_ad_t + av_ad_p1 + av_ad_p2) == 0.0)
            past_nlg = t > pt.nlg_period + 2
            lapse_zero_av = (av_zero_3 & past_nlg).to(dtype)
        else:
            lapse_zero_av = torch.zeros(B, dtype=dtype, device=device)

        is_inforce_end = torch.where(
            (t > pol.pol_term * 12),
            torch.zeros(B, dtype=dtype, device=device),
            torch.where(
                self.is_inforce_end[:, t - 1] == 0.0,
                torch.zeros(B, dtype=dtype, device=device),
                torch.where(
                    lapse_zero_av.bool(),
                    torch.zeros(B, dtype=dtype, device=device),
                    torch.ones(B, dtype=dtype, device=device),
                ),
            ),
        )
        self.is_inforce_end[:, t] = is_inforce_end

        # ---------------- S1.22: unmet deductions carry-forward ----------------
        dedncf = totdedn - dedn
        tot_dedncf_b4_bonus = (
            (dedncf + prev_dedncf - basic_dedncf_clrd - topup_dedncf_clrd)
            * is_inforce_bgn
        )

        # ---------------- S1.45-S1.57: unit fund growth ----------------
        # m_ulp_fer is per-policy / per-month tensor for stochastic compatibility
        m_ulp_fer = (1.0 + pt.ann_ulp_fer / 100.0) ** (1.0 / 12.0) - 1.0
        m_fmc_pc = pt.ann_fmc_pc / 12.0  # in % (e.g. 0.125)
        m_unit_gth = (1.0 + m_ulp_fer) * (1.0 - m_fmc_pc / 100.0) - 1.0
        m_hard_gtee = (1.0 + float(pt.hard_g_inv[py_clamped]) / 100.0) ** (1.0 / 12.0) - 1.0

        # Write the scalar to every cell of the [B] / [B, T] storage
        if self.retain_full_outputs:
            self.m_ulp_fer[:, t] = m_ulp_fer
        else:
            self.m_ulp_fer.fill_(m_ulp_fer)

        # S1.44 & S1.52: unit growth before COG
        bav_bval_bb = bav_ad * (1.0 + m_unit_gth)
        tuav_bval_bb = tuav_ad * (1.0 + m_unit_gth)
        self.bav_bval_bb[:, t] = bav_bval_bb
        self.tuav_bval_bb[:, t] = tuav_bval_bb

        # S1.54 & S1.57: COG adjustments
        cog_adj_bav = torch.clamp(bav_ad * (m_hard_gtee - m_unit_gth), min=0.0)
        cog_adj_tuav = torch.clamp(tuav_ad * (m_hard_gtee - m_unit_gth), min=0.0)

        bav_aval_bb = bav_bval_bb + cog_adj_bav
        tuav_aval_bb = tuav_bval_bb + cog_adj_tuav
        self._write_buf(self.bav_aval_bb, t, W, bav_aval_bb)
        self._write_buf(self.tuav_aval_bb, t, W, tuav_aval_bb)

        # ---------------- S1.60-S1.64: guaranteed AV tracking ----------------
        g_bav_ab_prev = self._read_buf(self.g_bav_ab, t - 1, 2)
        g_tuav_ab_prev = self._read_buf(self.g_tuav_ab, t - 1, 2)
        g_bav_bval = (g_bav_ab_prev + basic_alloc_prem - basic_dedn) * (1.0 + m_hard_gtee)
        g_tuav_bval = (g_tuav_ab_prev + topup_alloc_prem - topup_dedn) * (1.0 + m_hard_gtee)
        self._write_t_only(self.g_bav_bval, t, g_bav_bval)
        self._write_t_only(self.g_tuav_bval, t, g_tuav_bval)

        cog_adj_g_bav = torch.clamp(bav_bval_bb - g_bav_bval, min=0.0)
        cog_adj_g_tuav = torch.clamp(tuav_bval_bb - g_tuav_bval, min=0.0)
        g_bav_ab = g_bav_bval + cog_adj_g_bav
        self._write_buf(self.g_bav_ab, t, 2, g_bav_ab)

        # ---------------- S1.66-S1.77: Bonuses ----------------
        basic_lb = torch.zeros(B, dtype=dtype, device=device)
        topup_lb = torch.zeros(B, dtype=dtype, device=device)
        sb_coi_bonus = torch.zeros(B, dtype=dtype, device=device)
        sb_acp_bonus = torch.zeros(B, dtype=dtype, device=device)

        if t in self._basic_lb_months:
            N = self.bonus_schedule["basic_lb"]["lookback_N"][t]
            t_start = max(1, t - N + 1)
            basic_lb_pc = float(pt.basic_lb_rate[py_clamped]) / 100.0
            avg_bav = self._read_window(self.bav_aval_bb, t_start, t, W)
            basic_lb = basic_lb_pc * avg_bav

        if t in self._topup_lb_months:
            N = self.bonus_schedule["topup_lb"]["lookback_N"][t]
            t_start = max(1, t - N + 1)
            topup_lb_pc = float(pt.topup_lb_rate[py_clamped]) / 100.0
            avg_tuav = self._read_window(self.tuav_aval_bb, t_start, t, W)
            topup_lb = topup_lb_pc * avg_tuav

        if t in self._sb_coi_months:
            N = self.bonus_schedule["sb_coi"]["lookback_N"][t]
            t_start = max(1, t - N + 1)
            sb_coi_pc = float(pt.sb_coi_rate[py_clamped]) / 100.0
            sum_coi = self._read_window_sum(self.mort_coi, t_start, t, W)
            sb_coi_bonus = sb_coi_pc * sum_coi

        if t in self._sb_acp_months:
            sb_acp_pc = float(pt.sb_acp_rate[py_clamped]) / 100.0
            sb_acp_bonus = sb_acp_pc * pol.acp

        # S1.77: total bonus, only for in-force at end of this month
        tot_bonus = (basic_lb + topup_lb + sb_coi_bonus + sb_acp_bonus) * is_inforce_end

        # S1.21: bonus offset against carry-forward
        bonus_dedncf_clrd = torch.minimum(tot_bonus, tot_dedncf_b4_bonus)
        # S1.19: updated carry-forward
        tot_dedncf = tot_dedncf_b4_bonus - bonus_dedncf_clrd
        self._write_buf(self.tot_dedncf, t, 2, tot_dedncf)

        # S1.78: bonus allocated to AV
        bonus_alloc = tot_bonus - bonus_dedncf_clrd
        self._write_t_only(self.bonus_alloc, t, bonus_alloc)

        # ---------------- S1.65: g_tuav_ab ----------------
        g_tuav_ab = g_tuav_bval + cog_adj_g_tuav + bonus_alloc
        self._write_buf(self.g_tuav_ab, t, 2, g_tuav_ab)

        # ---------------- S1.79-S1.81: AV at end of month (ab) ----------------
        bav_ab_t = bav_aval_bb
        tuav_ab_t = tuav_aval_bb + bonus_alloc
        av_ab_t = bav_ab_t + tuav_ab_t
        self._write_buf(self.bav_ab, t, 2, bav_ab_t)
        self._write_buf(self.tuav_ab, t, 2, tuav_ab_t)
        self._write_buf(self.av_ab, t, 2, av_ab_t)

        # ---------------- S1.27: total actual deductions ----------------
        av_bd = bav_bd + tuav_bd
        tot_dedn_act = av_bd - av_ad + basic_dedncf_clrd + topup_dedncf_clrd
        self._write_t_only(self.tot_dedn_act, t, tot_dedn_act)

    def _read_window(
        self, buf: torch.Tensor, t_start: int, t_end: int, W: int
    ) -> torch.Tensor:
        """Compute mean over window [t_start, t_end] inclusive."""
        if self.retain_full_outputs:
            return buf[:, t_start:t_end + 1].mean(dim=1)
        # Rolling buffer: gather indices and mean
        idxs = self._window_indices(t_start, t_end, W)
        return buf[:, idxs].mean(dim=1)

    def _read_window_sum(
        self, buf: torch.Tensor, t_start: int, t_end: int, W: int
    ) -> torch.Tensor:
        """Compute sum over window [t_start, t_end] inclusive."""
        if self.retain_full_outputs:
            return buf[:, t_start:t_end + 1].sum(dim=1)
        idxs = self._window_indices(t_start, t_end, W)
        return buf[:, idxs].sum(dim=1)

    # =========================================================================
    # Part 2 - Decrements (identical formulas to DecrementProjection)
    # =========================================================================

    def _compute_part2_month(self, t: int) -> None:
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B

        pol_year = pol_year_at_t(t)
        py_clamped = min(pol_year, self.config.MAX_PROJ_YEARS)

        # ---------------- S2.1: no_pols_ifsm ----------------
        no_mats_prev = self._read_buf(self.no_mats, t - 1, 2)
        no_pols_ifsm = torch.clamp(self.no_pols_if[:, t - 1] - no_mats_prev, min=0.0)
        self._write_t_only(self.no_pols_ifsm, t, no_pols_ifsm)

        # ---------------- S2.6-S2.7: annual / monthly death rate ----------------
        ann_death_rate = lookup_mortality_rate(
            pol.age_at_entry,
            pol.sex,
            pol_year,
            pt.mortality_select_period,
            pt.mortality_min_age,
            pt.mortality_male,
            pt.mortality_female,
        ) / 1000.0
        ann_death_rate = torch.clamp(ann_death_rate, max=1.0)
        m_death_rate = 1.0 - torch.pow(1.0 - ann_death_rate, 1.0 / 12.0)

        # ---------------- S2.3: no_deaths ----------------
        no_deaths = no_pols_ifsm * m_death_rate
        self._write_t_only(self.no_deaths, t, no_deaths)

        # ---------------- S2.10-S2.13: m_lapse_rate ----------------
        m_lapse_rate = self._compute_lapse_rate(t, pol_year, py_clamped)

        # ---------------- S2.4: no_surrs ----------------
        no_surrs = (no_pols_ifsm - no_deaths) * m_lapse_rate
        self._write_t_only(self.no_surrs, t, no_surrs)

        # ---------------- S2.5: no_mats ----------------
        at_maturity = (t == pol.pol_term * 12).to(dtype)
        survivors_after_decr = no_pols_ifsm - no_deaths - no_surrs
        no_mats = survivors_after_decr * at_maturity
        self._write_buf(self.no_mats, t, 2, no_mats)

        # ---------------- S2.2: no_pols_if (end of month) ----------------
        self.no_pols_if[:, t] = no_pols_ifsm - no_deaths - no_surrs - no_mats

    def _compute_lapse_rate(
        self, t: int, pol_year: int, py_clamped: int
    ) -> torch.Tensor:
        """Identical to DecrementProjection._compute_lapse_rate."""
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B

        is_inforce_end_t = self.is_inforce_end[:, t]
        pol_term_months = pol.pol_term * 12
        prem_term_months = pol.prem_term * 12

        freq_col = self.freq_col.clamp(0, 3)
        py_idx = torch.full((B,), py_clamped, dtype=torch.long, device=device)
        ann_lapse = pt.lapse_rates[py_idx, freq_col]  # [B]

        m_lapse_case1 = torch.ones(B, dtype=dtype, device=device)
        m_lapse_case2 = torch.zeros(B, dtype=dtype, device=device)
        m_lapse_case3 = 1.0 - torch.pow(1.0 - ann_lapse / 100.0, 1.0 / 12.0)

        prem_due = (
            (t % self.prem_freq_mths == 0).to(dtype)
            * (t <= prem_term_months).to(dtype)
        )
        exponent_case4 = self.prem_freq_mths.to(dtype) / 12.0
        m_lapse_case4 = 1.0 - torch.pow(1.0 - ann_lapse / 100.0, exponent_case4)

        m_lapse_case5 = torch.zeros(B, dtype=dtype, device=device)

        # Priority order: 1 > 2 > 3 > 4 > 5
        m_lapse = m_lapse_case5
        m_lapse = torch.where(prem_due.bool(), m_lapse_case4, m_lapse)
        past_prem_term = (t > prem_term_months).to(torch.bool)
        m_lapse = torch.where(past_prem_term, m_lapse_case3, m_lapse)
        at_maturity = (t == pol_term_months).to(torch.bool)
        m_lapse = torch.where(at_maturity, m_lapse_case2, m_lapse)
        already_lapsed = (is_inforce_end_t == 0.0)
        m_lapse = torch.where(already_lapsed, m_lapse_case1, m_lapse)
        return m_lapse

    # =========================================================================
    # Part 3 Pass 1 - Cashflow components (identical formulas to legacy)
    # =========================================================================

    def _compute_part3_pass1_month(self, t: int) -> None:  # noqa: C901 — by spec
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B

        pol_year = pol_year_at_t(t)
        py = min(pol_year, self.config.MAX_PROJ_YEARS)

        # Pull Part 1 / Part 2 outputs at month t (already populated in this loop)
        no_pols_ifsm = self._read_t_only(self.no_pols_ifsm, t)
        no_pols_if   = self.no_pols_if[:, t]
        no_deaths    = self._read_t_only(self.no_deaths, t)
        no_surrs     = self._read_t_only(self.no_surrs, t)
        no_mats      = self._read_buf(self.no_mats, t, 2)

        # m_sh_fer is computed inside the t-loop because in stochastic mode
        # ann_sh_fer becomes a per-scenario / per-month rate.
        m_sh_fer = (1.0 + pt.ann_sh_fer / 100.0) ** (1.0 / 12.0) - 1.0
        m_fme_pc = pt.ann_fme_pc / 12.0  # in % per month

        # ---------------- S3.1-S3.3: premium income ----------------
        basic_prem_pp = self._read_t_only(self.basic_prem_pp, t)
        topup_prem_pp = self._read_t_only(self.topup_prem_pp, t)
        basic_prem_if = basic_prem_pp * no_pols_ifsm
        topup_prem_if = topup_prem_pp * no_pols_ifsm
        prem_inc_if = basic_prem_if + topup_prem_if
        self._write_t_only(self.basic_prem_if, t, basic_prem_if)
        self._write_t_only(self.topup_prem_if, t, topup_prem_if)
        self.prem_inc_if[:, t] = prem_inc_if

        # ---------------- S3.4: initial year operating expenses ----------------
        if pol_year == 1:
            ie_fixed = float(pt.op_exp_per_pol[0])
            ie_pc    = float(pt.op_exp_per_prem[0])
            op_init_exp_if = (
                (ie_fixed / 12.0) * no_pols_ifsm
                + (ie_pc / 100.0) * basic_prem_if
            )
        else:
            op_init_exp_if = torch.zeros(B, dtype=dtype, device=device)
        self._write_t_only(self.op_init_exp_if, t, op_init_exp_if)

        # ---------------- S3.10: renewal year operating expenses ----------------
        in_pol_term = (t <= pol.pol_term * 12).to(dtype)
        if pol_year > 1:
            re_fixed = float(pt.op_exp_per_pol[1])
            re_pc    = float(pt.op_exp_per_prem[1])
            exp_inflation = (1.0 + pt.inf_pc / 100.0) ** ((t - 1) / 12.0)
            op_ren_exp_if = (
                (re_fixed / 12.0) * exp_inflation * no_pols_ifsm
                + (re_pc / 100.0) * basic_prem_if
            ) * in_pol_term
        else:
            op_ren_exp_if = torch.zeros(B, dtype=dtype, device=device)
        self._write_t_only(self.op_ren_exp_if, t, op_ren_exp_if)

        # ---------------- S3.18-S3.19: investment expense ----------------
        av_ad_t = self._read_buf(self.av_ad, t, 3)
        invt_exp_if = av_ad_t * (m_fme_pc / 100.0) * no_pols_ifsm
        self._write_t_only(self.invt_exp_if, t, invt_exp_if)

        # ---------------- S3.22: commission ----------------
        comm_basic_pc = float(pt.comm_basic[py]) / 100.0
        comm_topup_pc = float(pt.comm_topup[py]) / 100.0
        comm_if = comm_basic_pc * basic_prem_if + comm_topup_pc * topup_prem_if
        self._write_t_only(self.comm_if, t, comm_if)

        # ---------------- S3.26: override commission ----------------
        ovrd_pc = float(pt.ovrd[py]) / 100.0
        ovrd_if = ovrd_pc * basic_prem_if
        self._write_t_only(self.ovrd_if, t, ovrd_if)

        # ---------------- S3.30-S3.31: death benefits ----------------
        attained_age = pol.age_at_entry + max(pol_year - 1, 0)
        lien_pc = lookup_lien_pc(attained_age, pt.lien_table)

        current_db_opt = self.current_db_opt[:, t]
        bav_bval_bb_t  = self.bav_bval_bb[:, t]
        tuav_bval_bb_t = self.tuav_bval_bb[:, t]

        db1_pp = torch.maximum(pol.sum_assd * lien_pc, bav_bval_bb_t) + tuav_bval_bb_t
        db2_pp = pol.sum_assd * lien_pc + bav_bval_bb_t + tuav_bval_bb_t
        death_ben_pp = torch.where(current_db_opt == 1, db1_pp, db2_pp)
        death_outgo  = death_ben_pp * no_deaths
        self._write_t_only(self.death_outgo, t, death_outgo)

        # ---------------- S3.32-S3.33: surrender benefits ----------------
        surr_chg_pp = torch.minimum(
            (float(pt.surr_chg[py]) / 100.0) * pol.acp,
            bav_bval_bb_t,
        )
        surr_ben_pp = bav_bval_bb_t + tuav_bval_bb_t - surr_chg_pp
        surr_outgo  = surr_ben_pp * no_surrs
        self._write_t_only(self.surr_outgo, t, surr_outgo)

        # ---------------- S3.35: maturity benefit ----------------
        av_ab_t = self._read_buf(self.av_ab, t, 2)
        mat_outgo = av_ab_t * no_mats
        self._write_t_only(self.mat_outgo, t, mat_outgo)

        # ---------------- S3.38-S3.41: cost-of-guarantee on terminations ----
        g_bav_bval_t  = self._read_t_only(self.g_bav_bval, t)
        g_tuav_bval_t = self._read_t_only(self.g_tuav_bval, t)
        g_bav_ab_t    = self._read_buf(self.g_bav_ab, t, 2)
        g_tuav_ab_t   = self._read_buf(self.g_tuav_ab, t, 2)

        # S3.38: guaranteed death benefit
        g_db1_pp = torch.maximum(pol.sum_assd * lien_pc, g_bav_bval_t) + g_tuav_bval_t
        g_db2_pp = pol.sum_assd * lien_pc + g_bav_bval_t + g_tuav_bval_t
        g_death_ben_pp = torch.where(current_db_opt == 1, g_db1_pp, g_db2_pp)
        cog_death = torch.clamp(g_death_ben_pp - death_ben_pp, min=0.0) * no_deaths

        # S3.40-S3.39: guaranteed surrender benefit
        g_surr_chg_pp = torch.minimum(
            (float(pt.surr_chg[py]) / 100.0) * pol.acp,
            g_bav_bval_t,
        )
        g_surr_ben_pp = g_bav_bval_t + g_tuav_bval_t - g_surr_chg_pp
        cog_surr = torch.clamp(g_surr_ben_pp - surr_ben_pp, min=0.0) * no_surrs

        # S3.41: guaranteed maturity benefit
        cog_mat = torch.clamp(
            g_bav_ab_t + g_tuav_ab_t - av_ab_t, min=0.0
        ) * no_mats

        cog_term_adj = cog_death + cog_surr + cog_mat
        self._write_t_only(self.cog_term_adj, t, cog_term_adj)

        # ---------------- S3.42-S3.43: unit reserve ----------------
        # unit_res_bgn reads av_ab[:, t-1]; av_ab buffer width = 2
        av_ab_prev = self._read_buf(self.av_ab, t - 1, 2)
        unit_res_bgn = av_ab_prev * no_pols_ifsm
        self._write_t_only(self.unit_res_bgn, t, unit_res_bgn)

        at_maturity = (t == pol.pol_term * 12).to(dtype)
        unit_res_end = av_ab_t * no_pols_if * (1.0 - at_maturity)
        self.unit_res_end[:, t] = unit_res_end

        # ---------------- S3.44: unit income ----------------
        m_ulp_fer_t = self._read_t_only(self.m_ulp_fer, t)
        unit_inc = av_ad_t * m_ulp_fer_t * no_pols_ifsm
        self._write_t_only(self.unit_inc, t, unit_inc)

        # ---------------- S3.45-S3.46: non-unit income ----------------
        basic_alloc_chg_pp = self._read_t_only(self.basic_alloc_chg_pp, t)
        topup_alloc_chg_pp = self._read_t_only(self.topup_alloc_chg_pp, t)
        tot_dedn_act_t     = self._read_t_only(self.tot_dedn_act, t)
        is_inforce_bgn_t   = self._read_t_only(self.is_inforce_bgn, t)

        net_cash_start = (
            (basic_alloc_chg_pp + topup_alloc_chg_pp + tot_dedn_act_t) * no_pols_ifsm
            - op_init_exp_if
            - op_ren_exp_if
            - invt_exp_if
            - comm_if
            - ovrd_if
        )
        non_unit_inc = net_cash_start * m_sh_fer * is_inforce_bgn_t
        self._write_t_only(self.non_unit_inc, t, non_unit_inc)

        # ---------------- S3.49: cashflow before zeroising ----------------
        cf_before_zv = (
            unit_res_bgn
            + prem_inc_if
            + unit_inc
            + non_unit_inc
            - op_init_exp_if
            - op_ren_exp_if
            - invt_exp_if
            - comm_if
            - ovrd_if
            - death_outgo
            - surr_outgo
            - mat_outgo
            - cog_term_adj
            - unit_res_end
        )
        self.cf_before_zv[:, t] = cf_before_zv

    # =========================================================================
    # On-the-fly summary accumulation (summary mode only)
    # =========================================================================

    def _accumulate_summary(self, t: int) -> None:
        """Sum [B] current-month values into [T] accumulators for month t."""
        # Part 2 keys
        # no_pols_if: read from the [B, t] survivor (already written above)
        self._summary_p2["no_pols_if"][t]   = self.no_pols_if[:, t].sum()
        self._summary_p2["no_pols_ifsm"][t] = self.no_pols_ifsm.sum()
        self._summary_p2["no_deaths"][t]    = self.no_deaths.sum()
        self._summary_p2["no_surrs"][t]     = self.no_surrs.sum()
        # no_mats lives in the [B, 2] rolling buffer
        self._summary_p2["no_mats"][t]      = self.no_mats[:, t % 2].sum()

        # Part 3 Pass 1 keys: 'survivors' read from [B, T], others from [B]
        s = self._summary_p3p1
        s["prem_inc_if"][t]    = self.prem_inc_if[:, t].sum()
        s["basic_prem_if"][t]  = self.basic_prem_if.sum()
        s["topup_prem_if"][t]  = self.topup_prem_if.sum()
        s["op_init_exp_if"][t] = self.op_init_exp_if.sum()
        s["op_ren_exp_if"][t]  = self.op_ren_exp_if.sum()
        s["invt_exp_if"][t]    = self.invt_exp_if.sum()
        s["comm_if"][t]        = self.comm_if.sum()
        s["ovrd_if"][t]        = self.ovrd_if.sum()
        s["death_outgo"][t]    = self.death_outgo.sum()
        s["surr_outgo"][t]     = self.surr_outgo.sum()
        s["mat_outgo"][t]      = self.mat_outgo.sum()
        s["cog_term_adj"][t]   = self.cog_term_adj.sum()
        s["unit_res_bgn"][t]   = self.unit_res_bgn.sum()
        s["unit_res_end"][t]   = self.unit_res_end[:, t].sum()
        s["unit_inc"][t]       = self.unit_inc.sum()
        s["non_unit_inc"][t]   = self.non_unit_inc.sum()
        s["cf_before_zv"][t]   = self.cf_before_zv[:, t].sum()

    # =========================================================================
    # Output collection
    # =========================================================================

    def _collect_part1(self) -> dict:
        """Return Part 1 tensor dict.

        In retain_full_outputs=True, all entries are full [B, T] tensors with
        the same keys and shapes as the legacy PAVProjection.run().
        In summary mode, entries that have been shrunk are returned as their
        rolling buffers ([B, 2], [B, 3], [B, W]) or [B] tensors. Downstream
        consumers (CashflowProjection) should not access these keys.
        """
        return {
            "is_inforce_bgn":     self.is_inforce_bgn,
            "is_inforce_end":     self.is_inforce_end,
            "current_db_opt":     self.current_db_opt,
            "bav_ab":             self.bav_ab,
            "tuav_ab":            self.tuav_ab,
            "av_ab":              self.av_ab,
            "bav_ad":             self.bav_ad,
            "tuav_ad":            self.tuav_ad,
            "av_ad":              self.av_ad,
            "bav_bval_bb":        self.bav_bval_bb,
            "tuav_bval_bb":       self.tuav_bval_bb,
            "bav_aval_bb":        self.bav_aval_bb,
            "tuav_aval_bb":       self.tuav_aval_bb,
            "g_bav_bval":         self.g_bav_bval,
            "g_tuav_bval":        self.g_tuav_bval,
            "g_bav_ab":           self.g_bav_ab,
            "g_tuav_ab":          self.g_tuav_ab,
            "tot_dedncf":         self.tot_dedncf,
            "tot_dedn_act":       self.tot_dedn_act,
            "basic_alloc_chg_pp": self.basic_alloc_chg_pp,
            "topup_alloc_chg_pp": self.topup_alloc_chg_pp,
            "basic_prem_pp":      self.basic_prem_pp,
            "topup_prem_pp":      self.topup_prem_pp,
            "mort_coi":           self.mort_coi,
            "m_ulp_fer":          self.m_ulp_fer,
            "bonus_alloc":        self.bonus_alloc,
        }

    def _collect_part2(self) -> dict:
        return {
            "no_pols_if":   self.no_pols_if,
            "no_pols_ifsm": self.no_pols_ifsm,
            "no_deaths":    self.no_deaths,
            "no_surrs":     self.no_surrs,
            "no_mats":      self.no_mats,
        }

    def _collect_part3_pass1(self) -> dict:
        return {
            "prem_inc_if":    self.prem_inc_if,
            "basic_prem_if":  self.basic_prem_if,
            "topup_prem_if":  self.topup_prem_if,
            "op_init_exp_if": self.op_init_exp_if,
            "op_ren_exp_if":  self.op_ren_exp_if,
            "invt_exp_if":    self.invt_exp_if,
            "comm_if":        self.comm_if,
            "ovrd_if":        self.ovrd_if,
            "death_outgo":    self.death_outgo,
            "surr_outgo":     self.surr_outgo,
            "mat_outgo":      self.mat_outgo,
            "cog_term_adj":   self.cog_term_adj,
            "unit_res_bgn":   self.unit_res_bgn,
            "unit_res_end":   self.unit_res_end,
            "unit_inc":       self.unit_inc,
            "non_unit_inc":   self.non_unit_inc,
            "cf_before_zv":   self.cf_before_zv,
        }

    def _collect_summary(self) -> dict:
        """Return on-the-fly [T] summary accumulators (summary mode only).

        Returns an empty dict in per-policy mode; the caller is expected to
        compute summaries from the full [B, T] tensors in that mode.
        """
        if self.retain_full_outputs:
            return {}
        out: dict[str, torch.Tensor] = {}
        out.update(self._summary_p2)
        out.update(self._summary_p3p1)
        return out
