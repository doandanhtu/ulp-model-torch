"""
utils.py - Helper functions for ULP model.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .inputs import ParamTables


# ---------------------------------------------------------------------------
# Time step helpers
# ---------------------------------------------------------------------------

def prem_freq_to_months(prem_freq: torch.Tensor) -> torch.Tensor:
    """Convert prem_freq code to months between premium payments.

    0 -> 12 (annual)
    1 ->  6 (semi-annual)
    2 ->  3 (quarterly)
    3 ->  1 (monthly)
    """
    mapping = torch.tensor([12, 6, 3, 1], dtype=prem_freq.dtype, device=prem_freq.device)
    return mapping[prem_freq]


def attained_age_at_t(age_at_entry: torch.Tensor, t: int) -> torch.Tensor:
    """Return attained age at time step t (in months).

    attained_age = age_at_entry + max(ceil(t/12) - 1, 0)
    For t=0 this returns age_at_entry (no increment in initialisation month).
    """
    increment = max(math.ceil(t / 12) - 1, 0)
    return age_at_entry + increment


def pol_year_at_t(t: int) -> int:
    """Return policy year at time step t.

    pol_year = ceil(t/12) for t >= 1, returns 1 for t = 0.
    """
    if t <= 0:
        return 1
    return math.ceil(t / 12)


# ---------------------------------------------------------------------------
# Rate lookups
# ---------------------------------------------------------------------------

def lookup_coi_rate(
    attained_age: torch.Tensor,
    sex: torch.Tensor,
    coi_male: torch.Tensor,
    coi_female: torch.Tensor,
) -> torch.Tensor:
    """Look up COI rate (per mille) by attained_age and sex.

    Parameters
    ----------
    attained_age : [B] long tensor, age 0-98
    sex          : [B] long tensor, 0=male 1=female
    coi_male     : [99] per mille
    coi_female   : [99] per mille

    Returns
    -------
    [B] float tensor, per mille
    """
    # Clamp age to valid table range
    age_clamped = attained_age.clamp(0, coi_male.shape[0] - 1)
    rate_male = coi_male[age_clamped]
    rate_female = coi_female[age_clamped]
    # sex==0 -> male rate, sex==1 -> female rate
    return torch.where(sex == 0, rate_male, rate_female)


def lookup_mortality_rate(
    age_at_entry: torch.Tensor,
    sex: torch.Tensor,
    pol_year: int,
    S: int,
    mort_male: torch.Tensor,
    mort_female: torch.Tensor,
) -> torch.Tensor:
    """Look up select mortality rate (per mille) for a given policy year.

    The select-S mortality table has:
      row    = entry_age + max(0, (pol_year-1) - S)
      col    = min(pol_year-1, S)
      row_idx = row + 4   (because table row 0 corresponds to age -4)

    Parameters
    ----------
    age_at_entry : [B] long, entry age in years
    sex          : [B] long, 0=male 1=female
    pol_year     : int, current policy year (1-indexed)
    S            : int, select period
    mort_male    : [121, S+1] float, per mille
    mort_female  : [121, S+1] float, per mille

    Returns
    -------
    [B] float tensor, per mille
    """
    row_offset = 4  # table row 0 corresponds to age -4
    n_rows, n_cols = mort_male.shape

    col = min(pol_year - 1, S)
    row_age_adj = max(0, (pol_year - 1) - S)  # how many years past select period

    row_base = age_at_entry + row_age_adj     # [B]
    row_idx = (row_base + row_offset).clamp(0, n_rows - 1)  # [B]

    rate_male = mort_male[row_idx, col]
    rate_female = mort_female[row_idx, col]
    return torch.where(sex == 0, rate_male, rate_female)


def lookup_lien_pc(
    attained_age: torch.Tensor,
    lien_table: torch.Tensor,
) -> torch.Tensor:
    """Return lien percentage (as a fraction 0-1) for attained age.

    Ages >= 4 use the last entry. Table values are in %.

    Parameters
    ----------
    attained_age : [B] long
    lien_table   : [5] float, percentage values for ages 0,1,2,3,4+

    Returns
    -------
    [B] float, fraction (0-1)
    """
    age_clamped = attained_age.clamp(0, lien_table.shape[0] - 1)
    return lien_table[age_clamped] / 100.0


# ---------------------------------------------------------------------------
# Bonus schedule precomputation
# ---------------------------------------------------------------------------

def precompute_bonus_schedule(
    param_tables: ParamTables,
    max_proj_years: int,
) -> dict:
    """Pre-compute award months and lookback windows for all bonus types.

    Returns
    -------
    dict with keys 'basic_lb', 'topup_lb', 'sb_coi', 'sb_acp'.
    Each entry contains:
      - 'award_months' : sorted list of t values (integer months) where bonus applies
      - 'lookback_N'   : dict {month: N} giving averaging window size (for lb / sb_coi)
    'sb_acp' has only 'award_months' (no lookback averaging needed).
    """
    T = max_proj_years * 12 + 1

    def _award_months_and_lookback(
        rate_table: torch.Tensor,
        first_consid_period: int,
    ) -> tuple[list[int], dict[int, int]]:
        """
        Award month = pol_year * 12.
        Lookback N: first_consid_period for the first award; gap between consecutive
        awards thereafter.
        """
        award_months: list[int] = []
        lookback_N: dict[int, int] = {}
        is_first = True
        prev_award_month = 0

        for py in range(1, max_proj_years + 1):
            award_t = py * 12
            if award_t >= T:
                break
            if float(rate_table[py]) != 0.0:
                award_months.append(award_t)
                N = first_consid_period if is_first else (award_t - prev_award_month)
                lookback_N[award_t] = max(1, N)
                is_first = False
                prev_award_month = award_t

        return award_months, lookback_N

    basic_lb_award, basic_lb_lookback = _award_months_and_lookback(
        param_tables.basic_lb_rate,
        param_tables.basic_lb_first_consid_period,
    )
    topup_lb_award, topup_lb_lookback = _award_months_and_lookback(
        param_tables.topup_lb_rate,
        param_tables.topup_lb_first_consid_period,
    )
    sb_coi_award, sb_coi_lookback = _award_months_and_lookback(
        param_tables.sb_coi_rate,
        param_tables.sb_coi_first_consid_period,
    )

    # sb_acp: award months only (no lookback – it's % of ACP which is constant)
    sb_acp_award: list[int] = []
    for py in range(1, max_proj_years + 1):
        award_t = py * 12
        if award_t >= T:
            break
        if float(param_tables.sb_acp_rate[py]) != 0.0:
            sb_acp_award.append(award_t)

    return {
        "basic_lb": {"award_months": basic_lb_award, "lookback_N": basic_lb_lookback},
        "topup_lb": {"award_months": topup_lb_award, "lookback_N": topup_lb_lookback},
        "sb_coi":   {"award_months": sb_coi_award,   "lookback_N": sb_coi_lookback},
        "sb_acp":   {"award_months": sb_acp_award},
    }
