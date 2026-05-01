"""
part3_cashflows.py - Part 3: Shareholder Cashflow Projection.

Four-pass computation implementing steps S3.1 through S3.69.
"""
from __future__ import annotations

import math

import torch

from .inputs import ParamTables, PolicyBatch
from .utils import lookup_lien_pc, pol_year_at_t


class CashflowProjection:
    """Projects shareholder cashflows using four passes over the time horizon."""

    def __init__(
        self,
        policies: PolicyBatch,
        param_tables: ParamTables,
        config,
        part1_outputs: dict,
        part2_outputs: dict,
    ) -> None:
        self.policies = policies
        self.param_tables = param_tables
        self.config = config
        self.p1 = part1_outputs
        self.p2 = part2_outputs

        B = policies.policy_id.shape[0]
        T = config.MAX_PROJ_MONTHS
        self.B = B
        self.T = T

        self.device = torch.device(config.compute_device)
        self.dtype = (
            torch.float64 if config.float_precision == "float64" else torch.float32
        )

        # ---------------------------------------------------------------
        # Allocate [B, T] tensors
        # ---------------------------------------------------------------
        def _ft(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=self.dtype, device=self.device)

        # Pass 1 outputs
        self.prem_inc_if    = _ft(B, T)
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
        self.unit_res_end   = _ft(B, T)
        self.unit_inc       = _ft(B, T)
        self.non_unit_inc   = _ft(B, T)
        self.cf_before_zv   = _ft(B, T)

        # Pass 2 output
        self.zeroising_res_if = _ft(B, T)

        # Pass 3 outputs
        self.cf_after_zv    = _ft(B, T)
        self.op_tax         = _ft(B, T)
        self.cf_after_tax   = _ft(B, T)
        self.tot_res_if     = _ft(B, T)
        self.solv_cap_req   = _ft(B, T)
        self.scr_inv_inc    = _ft(B, T)
        self.scr_inc_tax    = _ft(B, T)
        self.cf_after_scr   = _ft(B, T)

        # Pass 4 outputs
        self.pv_cf_after_scr = _ft(B, T)
        self.pv_prem_inc     = _ft(B, T)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """Execute all four passes and return result dict."""
        self._pass1_forward()
        self._pass2_backward()
        self._pass3_forward()
        self._pass4_backward()
        return self._collect_outputs()

    # -----------------------------------------------------------------------
    # Pass 1 – forward
    # -----------------------------------------------------------------------

    def _pass1_forward(self) -> None:  # noqa: C901
        pol = self.policies
        pt = self.param_tables
        p1 = self.p1
        p2 = self.p2
        dtype = self.dtype
        device = self.device
        B = self.B
        T = self.T

        m_sh_fer = (1.0 + pt.ann_sh_fer / 100.0) ** (1.0 / 12.0) - 1.0
        m_fme_pc = pt.ann_fme_pc / 12.0  # in % per month

        for t in range(0, T):
            pol_year = pol_year_at_t(t)
            py = min(pol_year, self.config.MAX_PROJ_YEARS)

            no_pols_ifsm = p2["no_pols_ifsm"][:, t]
            no_pols_if   = p2["no_pols_if"][:, t]
            no_deaths    = p2["no_deaths"][:, t]
            no_surrs     = p2["no_surrs"][:, t]
            no_mats      = p2["no_mats"][:, t]

            # -------------------------------------------------------
            # S3.1-S3.3: premium income
            # -------------------------------------------------------
            basic_prem_pp  = p1["basic_prem_pp"][:, t]
            topup_prem_pp  = p1["topup_prem_pp"][:, t]
            basic_prem_if  = basic_prem_pp * no_pols_ifsm
            topup_prem_if  = topup_prem_pp * no_pols_ifsm
            prem_inc_if    = basic_prem_if + topup_prem_if
            self.basic_prem_if[:, t]  = basic_prem_if
            self.topup_prem_if[:, t]  = topup_prem_if
            self.prem_inc_if[:, t]    = prem_inc_if

            # -------------------------------------------------------
            # S3.4: initial year operating expenses
            # -------------------------------------------------------
            if pol_year == 1:
                ie_fixed = float(pt.op_exp_per_pol[0])
                ie_pc    = float(pt.op_exp_per_prem[0])
                op_init_exp_if = (
                    (ie_fixed / 12.0) * no_pols_ifsm
                    + (ie_pc / 100.0) * basic_prem_if
                )
            else:
                op_init_exp_if = torch.zeros(B, dtype=dtype, device=device)
            self.op_init_exp_if[:, t] = op_init_exp_if

            # -------------------------------------------------------
            # S3.10: renewal year operating expenses
            # -------------------------------------------------------
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
            self.op_ren_exp_if[:, t] = op_ren_exp_if

            # -------------------------------------------------------
            # S3.18-S3.19: investment expense
            # -------------------------------------------------------
            av_ad_t = p1["av_ad"][:, t]
            invt_exp_if = av_ad_t * (m_fme_pc / 100.0) * no_pols_ifsm
            self.invt_exp_if[:, t] = invt_exp_if

            # -------------------------------------------------------
            # S3.22: commission
            # -------------------------------------------------------
            comm_basic_pc = float(pt.comm_basic[py]) / 100.0
            comm_topup_pc = float(pt.comm_topup[py]) / 100.0
            comm_if = comm_basic_pc * basic_prem_if + comm_topup_pc * topup_prem_if
            self.comm_if[:, t] = comm_if

            # -------------------------------------------------------
            # S3.26: override commission
            # -------------------------------------------------------
            ovrd_pc = float(pt.ovrd[py]) / 100.0
            ovrd_if = ovrd_pc * basic_prem_if
            self.ovrd_if[:, t] = ovrd_if

            # -------------------------------------------------------
            # S3.30-S3.31: death benefits
            # -------------------------------------------------------
            attained_age = pol.age_at_entry + max(pol_year - 1, 0)
            lien_pc = lookup_lien_pc(attained_age, pt.lien_table)

            current_db_opt   = p1["current_db_opt"][:, t]
            bav_bval_bb_t    = p1["bav_bval_bb"][:, t]
            tuav_bval_bb_t   = p1["tuav_bval_bb"][:, t]

            # db_opt=1: max(SA*lien_pc, bav_bval_bb) + tuav_bval_bb
            # db_opt=2: SA*lien_pc + bav_bval_bb + tuav_bval_bb
            db1_pp = torch.maximum(pol.sum_assd * lien_pc, bav_bval_bb_t) + tuav_bval_bb_t
            db2_pp = pol.sum_assd * lien_pc + bav_bval_bb_t + tuav_bval_bb_t
            death_ben_pp = torch.where(current_db_opt == 1, db1_pp, db2_pp)
            death_outgo  = death_ben_pp * no_deaths
            self.death_outgo[:, t] = death_outgo

            # -------------------------------------------------------
            # S3.32-S3.33: surrender benefits
            # -------------------------------------------------------
            surr_chg_pp = torch.minimum(
                (float(pt.surr_chg[py]) / 100.0) * pol.acp,
                bav_bval_bb_t,
            )
            surr_ben_pp = bav_bval_bb_t + tuav_bval_bb_t - surr_chg_pp
            surr_outgo  = surr_ben_pp * no_surrs
            self.surr_outgo[:, t] = surr_outgo

            # -------------------------------------------------------
            # S3.35: maturity benefit
            # -------------------------------------------------------
            av_ab_t = p1["av_ab"][:, t]
            mat_outgo = av_ab_t * no_mats
            self.mat_outgo[:, t] = mat_outgo

            # -------------------------------------------------------
            # S3.38-S3.41: cost-of-guarantee (cog) on termination benefits
            # -------------------------------------------------------
            g_bav_bval_t  = p1["g_bav_bval"][:, t]
            g_tuav_bval_t = p1["g_tuav_bval"][:, t]
            g_bav_ab_t    = p1["g_bav_ab"][:, t]
            g_tuav_ab_t   = p1["g_tuav_ab"][:, t]

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
            self.cog_term_adj[:, t] = cog_term_adj

            # -------------------------------------------------------
            # S3.42-S3.43: unit reserve
            # -------------------------------------------------------
            if t > 0:
                unit_res_bgn = p1["av_ab"][:, t - 1] * no_pols_ifsm
            else:
                unit_res_bgn = torch.zeros(B, dtype=dtype, device=device)
            self.unit_res_bgn[:, t] = unit_res_bgn

            # unit_res_end = 0 at maturity month, else av_ab * no_pols_if
            at_maturity = (t == pol.pol_term * 12).to(dtype)
            unit_res_end = av_ab_t * no_pols_if * (1.0 - at_maturity)
            self.unit_res_end[:, t] = unit_res_end

            # -------------------------------------------------------
            # S3.44: unit income (investment return on AV)
            # -------------------------------------------------------
            m_ulp_fer_t = float(p1["m_ulp_fer"][:, t].mean()) if t > 0 else 0.0
            unit_inc = av_ad_t * m_ulp_fer_t * no_pols_ifsm
            self.unit_inc[:, t] = unit_inc

            # -------------------------------------------------------
            # S3.45-S3.46: non-unit income (interest on net non-unit cashflow)
            # -------------------------------------------------------
            basic_alloc_chg_pp = p1["basic_alloc_chg_pp"][:, t]
            topup_alloc_chg_pp = p1["topup_alloc_chg_pp"][:, t]
            tot_dedn_act_t     = p1["tot_dedn_act"][:, t]
            is_inforce_bgn_t   = p1["is_inforce_bgn"][:, t]

            net_cash_start = (
                (basic_alloc_chg_pp + topup_alloc_chg_pp + tot_dedn_act_t) * no_pols_ifsm
                - op_init_exp_if
                - op_ren_exp_if
                - invt_exp_if
                - comm_if
                - ovrd_if
            )
            non_unit_inc = net_cash_start * m_sh_fer * is_inforce_bgn_t
            self.non_unit_inc[:, t] = non_unit_inc

            # -------------------------------------------------------
            # S3.49: cashflow before zeroising
            # -------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Pass 2 – backward (zeroising reserve)
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
            # Zero out where at or past maturity, or policy not in force
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

        m_sh_fer = (1.0 + pt.ann_sh_fer / 100.0) ** (1.0 / 12.0) - 1.0
        pol_term_months = pol.pol_term * 12  # [B]

        for t in range(0, T):
            pol_year = pol_year_at_t(t)
            py = min(pol_year, self.config.MAX_PROJ_YEARS)

            is_inforce_end_t = self.p1["is_inforce_end"][:, t]
            no_pols_if_t     = self.p2["no_pols_if"][:, t]

            # -------------------------------------------------------
            # S3.54: cashflow after zeroising
            # -------------------------------------------------------
            zr_t = self.zeroising_res_if[:, t]
            if t > 0:
                zr_prev = self.zeroising_res_if[:, t - 1]
                m_vir_prev = (1.0 + pt.ann_vir / 100.0) ** (1.0 / 12.0) - 1.0
            else:
                zr_prev = torch.zeros(B, dtype=dtype, device=device)
                m_vir_prev = 0.0

            cf_after_zv = (
                self.cf_before_zv[:, t]
                - zr_t
                + zr_prev * (1.0 + m_vir_prev)
            )
            self.cf_after_zv[:, t] = cf_after_zv

            # -------------------------------------------------------
            # S3.55-S3.57: operating tax
            # -------------------------------------------------------
            op_tax = (pt.tax_pc / 100.0) * cf_after_zv
            self.op_tax[:, t] = op_tax
            cf_after_tax = cf_after_zv - op_tax
            self.cf_after_tax[:, t] = cf_after_tax

            # -------------------------------------------------------
            # S3.58: total reserve
            # -------------------------------------------------------
            tot_res_if = self.unit_res_end[:, t] + zr_t
            self.tot_res_if[:, t] = tot_res_if

            # -------------------------------------------------------
            # S3.59: solvency capital requirement
            # -------------------------------------------------------
            at_or_past_mat = (t >= pol_term_months).to(dtype)
            not_inforce    = (1.0 - is_inforce_end_t)

            # death benefit at end of month for SCR
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

            # Zero when at/past maturity or not in force
            scr = scr_raw * (1.0 - at_or_past_mat)
            self.solv_cap_req[:, t] = scr

            # -------------------------------------------------------
            # S3.62-S3.64: SCR investment income & cashflow after SCR
            # -------------------------------------------------------
            if t > 0:
                scr_prev = self.solv_cap_req[:, t - 1]
            else:
                scr_prev = torch.zeros(B, dtype=dtype, device=device)

            scr_inv_inc = scr_prev * m_sh_fer
            scr_inc_tax = (pt.tax_pc / 100.0) * scr_inv_inc
            self.scr_inv_inc[:, t] = scr_inv_inc
            self.scr_inc_tax[:, t] = scr_inc_tax

            cf_after_scr = (
                cf_after_tax
                + scr_prev
                - scr
                + scr_inv_inc
                - scr_inc_tax
            )
            self.cf_after_scr[:, t] = cf_after_scr

    # -----------------------------------------------------------------------
    # Pass 4 – backward (PV of cashflows)
    # -----------------------------------------------------------------------

    def _pass4_backward(self) -> None:
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B
        T = self.T

        m_rdr = (1.0 + pt.ann_rdr / 100.0) ** (1.0 / 12.0) - 1.0
        pol_term_months = pol.pol_term * 12  # [B]

        for t in range(T - 1, -1, -1):
            at_or_past_mat = (t >= pol_term_months).to(dtype)

            if t + 1 < T:
                pv_next  = self.pv_cf_after_scr[:, t + 1]
                cf_next  = self.cf_after_scr[:, t + 1]
                pvp_next = self.pv_prem_inc[:, t + 1]
                pi_next  = self.prem_inc_if[:, t + 1]
            else:
                pv_next  = torch.zeros(B, dtype=dtype, device=device)
                cf_next  = torch.zeros(B, dtype=dtype, device=device)
                pvp_next = torch.zeros(B, dtype=dtype, device=device)
                pi_next  = torch.zeros(B, dtype=dtype, device=device)

            pv_new  = (pv_next  + cf_next)  / (1.0 + m_rdr)
            pvp_new = (pvp_next + pi_next) / (1.0 + m_rdr)

            self.pv_cf_after_scr[:, t] = pv_new  * (1.0 - at_or_past_mat)
            self.pv_prem_inc[:, t]     = pvp_new * (1.0 - at_or_past_mat)

    # -----------------------------------------------------------------------
    # Collect results
    # -----------------------------------------------------------------------

    def _collect_outputs(self) -> dict:
        return {
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
