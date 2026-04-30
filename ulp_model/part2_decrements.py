"""
part2_decrements.py - Part 2: Decrement (Death, Surrender, Maturity) Projection.

Implements steps S2.1 through S2.13.
"""
from __future__ import annotations

import math

import torch

from .config import ModelConfig
from .inputs import ParamTables, PolicyBatch
from .utils import (
    attained_age_at_t,
    lookup_mortality_rate,
    pol_year_at_t,
    prem_freq_to_months,
)


class DecrementProjection:
    """Projects policyholder decrements over the full projection horizon."""

    def __init__(
        self,
        policies: PolicyBatch,
        param_tables: ParamTables,
        config: ModelConfig,
        part1_outputs: dict,
    ) -> None:
        self.policies = policies
        self.param_tables = param_tables
        self.config = config
        self.part1 = part1_outputs

        B = policies.policy_id.shape[0]
        T = config.MAX_PROJ_MONTHS
        self.B = B
        self.T = T

        self.device = torch.device(config.compute_device)
        self.dtype = (
            torch.float64 if config.float_precision == "float64" else torch.float32
        )

        # Pre-compute premium frequency in months [B]
        self.prem_freq_mths = prem_freq_to_months(policies.prem_freq)
        # freq_col: prem_freq=0(annual)->3, 1(semi)->2, 2(quarterly)->1, 3(monthly)->0
        self.freq_col = 3 - policies.prem_freq  # [B] long

        # ---------------------------------------------------------------
        # Allocate [B, T] tensors
        # ---------------------------------------------------------------
        def _ft(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=self.dtype, device=self.device)

        self.no_pols_if   = _ft(B, T)
        self.no_pols_ifsm = _ft(B, T)
        self.no_deaths    = _ft(B, T)
        self.no_surrs     = _ft(B, T)
        self.no_mats      = _ft(B, T)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """Run decrement projection and return dict of output tensors."""
        pol = self.policies

        # t=0 initialisation
        self.no_pols_if[:, 0] = pol.init_pols_if

        for t in range(1, self.T):
            self._compute_month(t)

        return {
            "no_pols_if":   self.no_pols_if,
            "no_pols_ifsm": self.no_pols_ifsm,
            "no_deaths":    self.no_deaths,
            "no_surrs":     self.no_surrs,
            "no_mats":      self.no_mats,
        }

    # -----------------------------------------------------------------------
    # Month computation
    # -----------------------------------------------------------------------

    def _compute_month(self, t: int) -> None:
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B

        pol_year = pol_year_at_t(t)
        py_clamped = min(pol_year, self.config.MAX_PROJ_YEARS)

        # ---------------------------------------------------------------
        # S2.1: no_pols_ifsm (start-of-month survivors)
        # no_pols_if already deducts no_mats in S2.2, so clamping prevents
        # double-subtraction artifacts in months after policy maturity.
        # ---------------------------------------------------------------
        no_pols_ifsm = torch.clamp(
            self.no_pols_if[:, t - 1] - self.no_mats[:, t - 1], min=0.0
        )
        self.no_pols_ifsm[:, t] = no_pols_ifsm

        # ---------------------------------------------------------------
        # S2.6-S2.7: annual death rate from select mortality table
        # ---------------------------------------------------------------
        ann_death_rate = lookup_mortality_rate(
            pol.age_at_entry,
            pol.sex,
            pol_year,
            pt.mortality_select_period,
            pt.mortality_male,
            pt.mortality_female,
        ) / 1000.0  # convert per mille to probability

        ann_death_rate = torch.clamp(ann_death_rate, max=1.0)

        # S2.6: monthly death rate
        m_death_rate = 1.0 - torch.pow(1.0 - ann_death_rate, 1.0 / 12.0)

        # ---------------------------------------------------------------
        # S2.3: no_deaths
        # ---------------------------------------------------------------
        no_deaths = no_pols_ifsm * m_death_rate
        self.no_deaths[:, t] = no_deaths

        # ---------------------------------------------------------------
        # S2.10-S2.13: m_lapse_rate with priority logic
        # ---------------------------------------------------------------
        m_lapse_rate = self._compute_lapse_rate(t, pol_year, py_clamped)

        # ---------------------------------------------------------------
        # S2.4: no_surrs
        # ---------------------------------------------------------------
        no_surrs = (no_pols_ifsm - no_deaths) * m_lapse_rate
        self.no_surrs[:, t] = no_surrs

        # ---------------------------------------------------------------
        # S2.5: no_mats (maturities)
        # Policy matures at exactly t == pol_term * 12
        # ---------------------------------------------------------------
        at_maturity = (t == pol.pol_term * 12).to(dtype)
        no_pols_after = self.no_pols_if[:, t - 1] - self.no_mats[:, t - 1]  # same as ifsm
        # All survivors (after deaths & surrs) mature if this is maturity month
        survivors_after_decr = no_pols_ifsm - no_deaths - no_surrs
        no_mats = survivors_after_decr * at_maturity
        self.no_mats[:, t] = no_mats

        # ---------------------------------------------------------------
        # S2.2: no_pols_if (end of month)
        # ---------------------------------------------------------------
        self.no_pols_if[:, t] = no_pols_ifsm - no_deaths - no_surrs - no_mats

    # -----------------------------------------------------------------------
    # Lapse rate logic (S2.10-S2.13)
    # -----------------------------------------------------------------------

    def _compute_lapse_rate(self, t: int, pol_year: int, py_clamped: int) -> torch.Tensor:
        pol = self.policies
        pt = self.param_tables
        dtype = self.dtype
        device = self.device
        B = self.B

        is_inforce_end_t = self.part1["is_inforce_end"][:, t]  # [B]
        pol_term_months = pol.pol_term * 12                     # [B] long
        prem_term_months = pol.prem_term * 12                   # [B] long

        # ann_lapse from lapse_rates table [MAX_PROJ_YEARS+1, 4]
        # freq_col is per-policy [B]
        freq_col = self.freq_col.clamp(0, 3)
        py_idx = torch.full((B,), py_clamped, dtype=torch.long, device=device)
        # Gather along rows: lapse_rates[py_clamped, freq_col[b]] for each b
        ann_lapse = pt.lapse_rates[py_idx, freq_col]  # [B]

        # Case 1: policy lapsed (is_inforce_end == 0) → full lapse
        m_lapse_case1 = torch.ones(B, dtype=dtype, device=device)

        # Case 2: at maturity → no lapse (handled by no_mats)
        m_lapse_case2 = torch.zeros(B, dtype=dtype, device=device)

        # Case 3: past prem_term → monthly lapse equivalent
        m_lapse_case3 = 1.0 - torch.pow(1.0 - ann_lapse / 100.0, 1.0 / 12.0)

        # Case 4: premium due month within prem_term
        # Frequency: prem_freq=0(annual)->1 payment/yr=12mths; prem_freq=3(monthly)->1mth
        # Number of payments per year = 12 / prem_freq_mths
        # m_lapse = 1 - (1 - ann_lapse/100)^(1/n_payments_per_year)
        #         = 1 - (1 - ann_lapse/100)^(prem_freq_mths/12)
        # prem_due_this_month: (t-1) % prem_freq_mths == 0 AND t <= prem_term_months
        prem_due = (
            ((t - 1) % self.prem_freq_mths == 0).to(dtype)
            * (t <= prem_term_months).to(dtype)
        )
        exponent_case4 = self.prem_freq_mths.to(dtype) / 12.0
        m_lapse_case4 = 1.0 - torch.pow(1.0 - ann_lapse / 100.0, exponent_case4)

        # Case 5: no lapse
        m_lapse_case5 = torch.zeros(B, dtype=dtype, device=device)

        # Priority order: 1 > 2 > 3 > 4 > 5
        # Start from 5 and apply higher priority conditions on top
        m_lapse = m_lapse_case5

        # Case 4: premium due and within prem_term
        m_lapse = torch.where(prem_due.bool(), m_lapse_case4, m_lapse)

        # Case 3: past prem_term
        past_prem_term = (t > prem_term_months).to(torch.bool)
        m_lapse = torch.where(past_prem_term, m_lapse_case3, m_lapse)

        # Case 2: at maturity
        at_maturity = (t == pol_term_months).to(torch.bool)
        m_lapse = torch.where(at_maturity, m_lapse_case2, m_lapse)

        # Case 1: is_inforce_end == 0 (policy already lapsed / zero AV forced lapse)
        already_lapsed = (is_inforce_end_t == 0.0)
        m_lapse = torch.where(already_lapsed, m_lapse_case1, m_lapse)

        return m_lapse
