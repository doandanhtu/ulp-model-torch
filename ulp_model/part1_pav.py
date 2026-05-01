"""
part1_pav.py - Part 1: Policy Account Value (PAV) Projection.

Implements steps S1.1 through S1.81.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .inputs import ParamTables, PolicyBatch
from .utils import (
    attained_age_at_t,
    lookup_coi_rate,
    lookup_lien_pc,
    pol_year_at_t,
    precompute_bonus_schedule,
)


class PAVProjection:
    """Projects Policy Account Values (BAV and TUAV) over the full projection horizon."""

    def __init__(
        self,
        policies: PolicyBatch,
        param_tables: ParamTables,
        config,
    ) -> None:
        self.policies = policies
        self.param_tables = param_tables
        self.config = config

        B = policies.policy_id.shape[0]
        T = config.MAX_PROJ_MONTHS
        self.B = B
        self.T = T

        self.device = torch.device(config.compute_device)
        self.dtype = (
            torch.float64 if config.float_precision == "float64" else torch.float32
        )

        # prem_freq values are already in months (12=annual, 6=semi, 3=quarterly, 1=monthly)
        self.prem_freq_mths = policies.prem_freq
        self.topup_freq_mths = policies.topup_freq

        # Pre-compute bonus schedule
        self.bonus_schedule = precompute_bonus_schedule(param_tables, config.MAX_PROJ_YEARS)
        # Build sets for O(1) lookup
        self._basic_lb_months = set(self.bonus_schedule["basic_lb"]["award_months"])
        self._topup_lb_months = set(self.bonus_schedule["topup_lb"]["award_months"])
        self._sb_coi_months = set(self.bonus_schedule["sb_coi"]["award_months"])
        self._sb_acp_months = set(self.bonus_schedule["sb_acp"]["award_months"])

        # ---------------------------------------------------------------
        # Allocate full [B, T] tensors
        # ---------------------------------------------------------------
        def _ft(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=self.dtype, device=self.device)

        def _lt(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.long, device=self.device)

        self.is_inforce_bgn   = _ft(B, T)
        self.is_inforce_end   = _ft(B, T)
        self.current_db_opt   = _lt(B, T)

        self.bav_ab           = _ft(B, T)
        self.tuav_ab          = _ft(B, T)
        self.av_ab            = _ft(B, T)

        self.bav_ad           = _ft(B, T)
        self.tuav_ad          = _ft(B, T)
        self.av_ad            = _ft(B, T)

        self.bav_bval_bb      = _ft(B, T)
        self.tuav_bval_bb     = _ft(B, T)
        self.bav_aval_bb      = _ft(B, T)
        self.tuav_aval_bb     = _ft(B, T)

        self.g_bav_bval       = _ft(B, T)
        self.g_tuav_bval      = _ft(B, T)
        self.g_bav_ab         = _ft(B, T)
        self.g_tuav_ab        = _ft(B, T)

        self.tot_dedncf       = _ft(B, T)  # carry-forward deductions
        self.tot_dedn_act     = _ft(B, T)

        self.basic_alloc_chg_pp = _ft(B, T)
        self.topup_alloc_chg_pp = _ft(B, T)
        self.basic_prem_pp    = _ft(B, T)
        self.topup_prem_pp    = _ft(B, T)

        self.mort_coi         = _ft(B, T)
        self.m_ulp_fer        = _ft(B, T)  # monthly unit fund earned rate (scalar replicated)
        self.bonus_alloc      = _ft(B, T)

        # Internal working tensors (per month, kept for bonus lookback)
        self._basic_dedn      = _ft(B, T)
        self._topup_dedn      = _ft(B, T)
        self._basic_dedncf_clrd = _ft(B, T)
        self._topup_dedncf_clrd = _ft(B, T)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """Run PAV projection and return dict of output tensors."""
        # t=0: initialization
        self.is_inforce_end[:, 0] = 1.0
        self.current_db_opt[:, 0] = self.policies.db_opt

        for t in range(1, self.T):
            self._compute_month(t)

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

    # -----------------------------------------------------------------------
    # Month computation
    # -----------------------------------------------------------------------

    def _compute_month(self, t: int) -> None:  # noqa: C901 – complex by spec
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B

        pol_year = pol_year_at_t(t)
        py = pol_year  # shorthand

        # Clamp pol_year for table lookup (tables have length MAX_PROJ_YEARS+1)
        py_clamped = min(py, self.config.MAX_PROJ_YEARS)

        # attained age [B] (long)
        attained_age = attained_age_at_t(pol.age_at_entry, t)

        # ---------------------------------------------------------------
        # S1.1: is_inforce_bgn
        # ---------------------------------------------------------------
        past_term = (t > pol.pol_term * 12).to(dtype)
        is_inforce_bgn = self.is_inforce_end[:, t - 1] * (1.0 - past_term)
        self.is_inforce_bgn[:, t] = is_inforce_bgn

        # ---------------------------------------------------------------
        # S1.40: current_db_opt
        # db_opt forced to 1 (basic) once attained_age >= age_db_opt_change
        # ---------------------------------------------------------------
        age_changed = (attained_age >= pt.age_db_opt_change).to(torch.long)
        # If age exceeded threshold, set to 1; else keep previous value
        self.current_db_opt[:, t] = torch.where(
            age_changed.bool(),
            torch.ones(B, dtype=torch.long, device=device),
            self.current_db_opt[:, t - 1],
        )
        current_db_opt = self.current_db_opt[:, t]

        # ---------------------------------------------------------------
        # S1.4-S1.5: premium per policy
        # ---------------------------------------------------------------
        # Basic premium due when (t-1) % prem_freq_mths == 0 AND t <= prem_term*12
        basic_due = (
            ((t - 1) % self.prem_freq_mths == 0).to(dtype)
            * (t <= pol.prem_term * 12).to(dtype)
        )
        # Premium amount per payment = ACP / (12 / prem_freq_mths)
        # = ACP * prem_freq_mths / 12
        basic_prem_pp = (pol.acp * self.prem_freq_mths.to(dtype) / 12.0) * basic_due * is_inforce_bgn
        self.basic_prem_pp[:, t] = basic_prem_pp

        # Top-up premium
        topup_due = (
            ((t - 1) % self.topup_freq_mths == 0).to(dtype)
            * (t <= pol.topup_term * 12).to(dtype)
        )
        topup_prem_pp = (pol.atp * self.topup_freq_mths.to(dtype) / 12.0) * topup_due * is_inforce_bgn
        self.topup_prem_pp[:, t] = topup_prem_pp

        # ---------------------------------------------------------------
        # S1.6-S1.9: allocation charges
        # ---------------------------------------------------------------
        basic_alloc_chg_pc = float(pt.alloc_chg_basic[py_clamped]) / 100.0
        topup_alloc_chg_pc = float(pt.alloc_chg_topup[py_clamped]) / 100.0

        basic_alloc_chg_pp = basic_alloc_chg_pc * basic_prem_pp
        topup_alloc_chg_pp = topup_alloc_chg_pc * topup_prem_pp
        self.basic_alloc_chg_pp[:, t] = basic_alloc_chg_pp
        self.topup_alloc_chg_pp[:, t] = topup_alloc_chg_pp

        # ---------------------------------------------------------------
        # S1.10-S1.11: allocated premium before carry-forward deductions
        # ---------------------------------------------------------------
        basic_alloc_prem_b4 = basic_prem_pp - basic_alloc_chg_pp
        topup_alloc_prem_b4 = topup_prem_pp - topup_alloc_chg_pp

        # ---------------------------------------------------------------
        # S1.17-S1.18: carry-forward deduction clearing
        # ---------------------------------------------------------------
        prev_dedncf = self.tot_dedncf[:, t - 1]
        basic_dedncf_clrd = torch.minimum(prev_dedncf, basic_alloc_prem_b4)
        remaining_dedncf = prev_dedncf - basic_dedncf_clrd
        topup_dedncf_clrd = torch.minimum(remaining_dedncf, topup_alloc_prem_b4)
        self._basic_dedncf_clrd[:, t] = basic_dedncf_clrd
        self._topup_dedncf_clrd[:, t] = topup_dedncf_clrd

        # ---------------------------------------------------------------
        # S1.12-S1.13: allocated premium after carry-forward clearing
        # ---------------------------------------------------------------
        basic_alloc_prem = basic_alloc_prem_b4 - basic_dedncf_clrd
        topup_alloc_prem = topup_alloc_prem_b4 - topup_dedncf_clrd

        # ---------------------------------------------------------------
        # S1.14-S1.16: account values at beginning of deduction (bd)
        # ---------------------------------------------------------------
        bav_bd = (self.bav_ab[:, t - 1] + basic_alloc_prem) * is_inforce_bgn
        tuav_bd = (self.tuav_ab[:, t - 1] + topup_alloc_prem) * is_inforce_bgn

        # ---------------------------------------------------------------
        # S1.28: admin charge
        # ---------------------------------------------------------------
        admin_chg_pp = min(
            pt.admin_chg_start + pt.admin_chg_inc * (py - 1),
            pt.admin_chg_cap,
        )
        admin_chg = torch.full((B,), admin_chg_pp, dtype=dtype, device=device) * is_inforce_bgn

        # ---------------------------------------------------------------
        # S1.39: lien percentage
        # ---------------------------------------------------------------
        lien_pc = lookup_lien_pc(attained_age, pt.lien_table)  # [B] fraction 0-1

        # ---------------------------------------------------------------
        # S1.35-S1.37: COI base & SAR
        # ---------------------------------------------------------------
        bav_coi = torch.clamp(bav_bd - admin_chg, min=0.0)

        # Death benefit for COI purposes
        db1_coi = torch.maximum(pol.sum_assd * lien_pc, bav_coi)     # db_opt=1
        db2_coi = pol.sum_assd * lien_pc + bav_coi                    # db_opt=2
        dthben_coi = torch.where(current_db_opt == 1, db1_coi, db2_coi)

        sar_mort = torch.clamp(dthben_coi - bav_coi, min=0.0) * is_inforce_bgn

        # S1.33: annual COI rate (per mille)
        ann_mort_coi = lookup_coi_rate(attained_age, pol.sex, pt.coi_table_male, pt.coi_table_female)

        # S1.32: monthly COI charge
        mort_coi = (sar_mort / 1000.0) * (ann_mort_coi / 12.0) * (1.0 + pol.mort_loading / 100.0)
        self.mort_coi[:, t] = mort_coi

        # ---------------------------------------------------------------
        # S1.23-S1.27: deductions
        # ---------------------------------------------------------------
        totdedn = mort_coi + admin_chg

        basic_dedn = torch.minimum(totdedn, bav_bd)
        topup_dedn = torch.minimum(totdedn - basic_dedn, tuav_bd)
        dedn = basic_dedn + topup_dedn
        self._basic_dedn[:, t] = basic_dedn
        self._topup_dedn[:, t] = topup_dedn

        # S1.41-S1.43: account values after deductions (ad)
        bav_ad = torch.clamp(bav_bd - basic_dedn, min=0.0)
        tuav_ad = torch.clamp(tuav_bd - topup_dedn, min=0.0)
        av_ad = bav_ad + tuav_ad
        self.bav_ad[:, t] = bav_ad
        self.tuav_ad[:, t] = tuav_ad
        self.av_ad[:, t] = av_ad

        # ---------------------------------------------------------------
        # S1.2: is_inforce_end
        # ---------------------------------------------------------------
        # Rule: policy lapses if AV has been zero for 3 consecutive months past NLG
        if t >= 3:
            av_zero_3 = (
                (self.av_ad[:, t] + self.av_ad[:, t - 1] + self.av_ad[:, t - 2]) == 0.0
            )
            past_nlg = t > pt.nlg_period + 2
            lapse_zero_av = (av_zero_3 & past_nlg).to(dtype)
        else:
            lapse_zero_av = torch.zeros(B, dtype=dtype, device=device)

        is_inforce_end = torch.where(
            (t > pol.pol_term * 12),  # past policy term
            torch.zeros(B, dtype=dtype, device=device),
            torch.where(
                self.is_inforce_end[:, t - 1] == 0.0,  # already lapsed
                torch.zeros(B, dtype=dtype, device=device),
                torch.where(
                    lapse_zero_av.bool(),
                    torch.zeros(B, dtype=dtype, device=device),
                    torch.ones(B, dtype=dtype, device=device),
                ),
            ),
        )
        self.is_inforce_end[:, t] = is_inforce_end

        # ---------------------------------------------------------------
        # S1.22: unmet deductions carry-forward
        # ---------------------------------------------------------------
        dedncf = totdedn - dedn  # unmet deductions this month
        tot_dedncf_b4_bonus = (
            (dedncf + prev_dedncf - basic_dedncf_clrd - topup_dedncf_clrd)
            * is_inforce_bgn
        )

        # ---------------------------------------------------------------
        # S1.45-S1.57: unit fund growth
        # ---------------------------------------------------------------
        m_ulp_fer = (1.0 + pt.ann_ulp_fer / 100.0) ** (1.0 / 12.0) - 1.0
        m_fmc_pc = pt.ann_fmc_pc / 12.0  # in % (e.g. 0.125)
        m_unit_gth = (1.0 + m_ulp_fer) * (1.0 - m_fmc_pc / 100.0) - 1.0
        m_hard_gtee = (1.0 + float(pt.hard_g_inv[py_clamped]) / 100.0) ** (1.0 / 12.0) - 1.0

        # Store m_ulp_fer as tensor
        self.m_ulp_fer[:, t] = m_ulp_fer

        # S1.44 & S1.52: unit growth before COG adjustment
        bav_bval_bb = bav_ad * (1.0 + m_unit_gth)
        tuav_bval_bb = tuav_ad * (1.0 + m_unit_gth)
        self.bav_bval_bb[:, t] = bav_bval_bb
        self.tuav_bval_bb[:, t] = tuav_bval_bb

        # S1.54 & S1.57: COG adjustments
        cog_adj_bav = torch.clamp(bav_ad * (m_hard_gtee - m_unit_gth), min=0.0)
        cog_adj_tuav = torch.clamp(tuav_ad * (m_hard_gtee - m_unit_gth), min=0.0)

        bav_aval_bb = bav_bval_bb + cog_adj_bav
        tuav_aval_bb = tuav_bval_bb + cog_adj_tuav
        self.bav_aval_bb[:, t] = bav_aval_bb
        self.tuav_aval_bb[:, t] = tuav_aval_bb

        # ---------------------------------------------------------------
        # S1.60-S1.64: guaranteed AV tracking
        # ---------------------------------------------------------------
        g_bav_bval = (self.g_bav_ab[:, t - 1] + basic_alloc_prem - basic_dedn) * (1.0 + m_hard_gtee)
        g_tuav_bval = (self.g_tuav_ab[:, t - 1] + topup_alloc_prem - topup_dedn) * (1.0 + m_hard_gtee)
        self.g_bav_bval[:, t] = g_bav_bval
        self.g_tuav_bval[:, t] = g_tuav_bval

        cog_adj_g_bav = torch.clamp(bav_bval_bb - g_bav_bval, min=0.0)
        cog_adj_g_tuav = torch.clamp(tuav_bval_bb - g_tuav_bval, min=0.0)
        g_bav_ab = g_bav_bval + cog_adj_g_bav
        self.g_bav_ab[:, t] = g_bav_ab

        # ---------------------------------------------------------------
        # S1.66-S1.77: Bonuses
        # ---------------------------------------------------------------
        basic_lb = torch.zeros(B, dtype=dtype, device=device)
        topup_lb = torch.zeros(B, dtype=dtype, device=device)
        sb_coi_bonus = torch.zeros(B, dtype=dtype, device=device)
        sb_acp_bonus = torch.zeros(B, dtype=dtype, device=device)

        if t in self._basic_lb_months:
            N = self.bonus_schedule["basic_lb"]["lookback_N"][t]
            t_start = max(1, t - N + 1)
            basic_lb_pc = float(pt.basic_lb_rate[py_clamped]) / 100.0
            avg_bav = self.bav_aval_bb[:, t_start: t + 1].mean(dim=1)
            basic_lb = basic_lb_pc * avg_bav

        if t in self._topup_lb_months:
            N = self.bonus_schedule["topup_lb"]["lookback_N"][t]
            t_start = max(1, t - N + 1)
            topup_lb_pc = float(pt.topup_lb_rate[py_clamped]) / 100.0
            avg_tuav = self.tuav_aval_bb[:, t_start: t + 1].mean(dim=1)
            topup_lb = topup_lb_pc * avg_tuav

        if t in self._sb_coi_months:
            N = self.bonus_schedule["sb_coi"]["lookback_N"][t]
            t_start = max(1, t - N + 1)
            sb_coi_pc = float(pt.sb_coi_rate[py_clamped]) / 100.0
            sum_coi = self.mort_coi[:, t_start: t + 1].sum(dim=1)
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
        self.tot_dedncf[:, t] = tot_dedncf

        # S1.78: bonus allocated to AV
        bonus_alloc = tot_bonus - bonus_dedncf_clrd
        self.bonus_alloc[:, t] = bonus_alloc

        # ---------------------------------------------------------------
        # S1.65: g_tuav_ab
        # ---------------------------------------------------------------
        g_tuav_ab = g_tuav_bval + cog_adj_g_tuav + bonus_alloc
        self.g_tuav_ab[:, t] = g_tuav_ab

        # ---------------------------------------------------------------
        # S1.79-S1.81: final account values at end of month (ab)
        # ---------------------------------------------------------------
        self.bav_ab[:, t] = bav_aval_bb
        self.tuav_ab[:, t] = tuav_aval_bb + bonus_alloc
        self.av_ab[:, t] = self.bav_ab[:, t] + self.tuav_ab[:, t]

        # ---------------------------------------------------------------
        # S1.27: total actual deductions
        # ---------------------------------------------------------------
        # av_bd - av_ad + cleared carry-forwards
        av_bd = bav_bd + tuav_bd
        tot_dedn_act = av_bd - av_ad + basic_dedncf_clrd + topup_dedncf_clrd
        self.tot_dedn_act[:, t] = tot_dedn_act
