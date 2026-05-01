"""
inputs.py - PolicyBatch and ParamTables dataclasses for ULP model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PolicyBatch:
    """Holds a batch of B policies as per-field tensors."""

    policy_id: torch.Tensor       # [B] int32
    age_at_entry: torch.Tensor    # [B] int32
    sex: torch.Tensor             # [B] int32; 0=male, 1=female
    pol_term: torch.Tensor        # [B] int32, years
    prem_term: torch.Tensor       # [B] int32, years
    prem_freq: torch.Tensor       # [B] int32; months between payments: 12=annual, 6=semi, 3=quarterly, 1=monthly
    sum_assd: torch.Tensor        # [B] float
    db_opt: torch.Tensor          # [B] int32; 1=basic, 2=escalating
    acp: torch.Tensor             # [B] float (annualised committed premium)
    atp: torch.Tensor             # [B] float (annualised top-up premium)
    topup_term: torch.Tensor      # [B] int32, years
    topup_freq: torch.Tensor      # [B] int32; same as prem_freq: 12=annual, 6=semi, 3=quarterly, 1=monthly
    mort_loading: torch.Tensor    # [B] float, % (0=standard)
    init_pols_if: torch.Tensor    # [B] float


@dataclass
class ParamTables:
    """All parameter tables and scalar assumptions."""

    # ------------------------------------------------------------------
    # 1-D tables indexed by pol_year (length MAX_PROJ_YEARS+1,
    # index 0 unused, last row extended to fill)
    # ------------------------------------------------------------------
    alloc_chg_basic: torch.Tensor   # [MAX_PROJ_YEARS+1] %
    alloc_chg_topup: torch.Tensor   # [MAX_PROJ_YEARS+1] %
    surr_chg: torch.Tensor          # [MAX_PROJ_YEARS+1] %
    hard_g_inv: torch.Tensor        # [MAX_PROJ_YEARS+1] annual %
    comm_basic: torch.Tensor        # [MAX_PROJ_YEARS+1] %
    comm_topup: torch.Tensor        # [MAX_PROJ_YEARS+1] %
    ovrd: torch.Tensor              # [MAX_PROJ_YEARS+1] %
    basic_lb_rate: torch.Tensor     # [MAX_PROJ_YEARS+1] % (0 for non-award years)
    topup_lb_rate: torch.Tensor     # [MAX_PROJ_YEARS+1] %
    sb_coi_rate: torch.Tensor       # [MAX_PROJ_YEARS+1] %
    sb_acp_rate: torch.Tensor       # [MAX_PROJ_YEARS+1] %
    lapse_rates: torch.Tensor       # [MAX_PROJ_YEARS+1, 4] cols: monthly,quarterly,semiann,annual

    # ------------------------------------------------------------------
    # Age-indexed lookup tables
    # ------------------------------------------------------------------
    coi_table_male: torch.Tensor    # [99] per mille, ages 0-98
    coi_table_female: torch.Tensor  # [99] per mille, ages 0-98
    lien_table: torch.Tensor        # [5] % for ages 0,1,2,3,4+

    # ------------------------------------------------------------------
    # Mortality tables (select-S format)
    # ------------------------------------------------------------------
    mortality_male: torch.Tensor    # [121, 5] per mille, row offset=4 (rows age -4 to 116)
    mortality_female: torch.Tensor  # [121, 5] per mille
    mortality_select_period: int    # S=4

    # ------------------------------------------------------------------
    # Operational expense
    # ------------------------------------------------------------------
    op_exp_per_pol: torch.Tensor    # [2]: [initial_annual, renewal_annual] monetary
    op_exp_per_prem: torch.Tensor   # [2]: [initial_%, renewal_%]

    # ------------------------------------------------------------------
    # Scalars from reg_param_tbl
    # ------------------------------------------------------------------
    solv_marg_res: float            # %
    solv_marg_sar: float            # %
    tax_pc: float                   # %

    # ------------------------------------------------------------------
    # Scalars from scalar_inputs.yaml (model assumptions)
    # ------------------------------------------------------------------
    ann_ulp_fer: float
    ann_sh_fer: float
    ann_fmc_pc: float
    ann_fme_pc: float
    ann_vir: float
    ann_rdr: float
    inf_pc: float

    # ------------------------------------------------------------------
    # Scalars from scalar_inputs.yaml (product features)
    # ------------------------------------------------------------------
    nlg_period: int
    age_db_opt_change: int
    basic_lb_first_consid_period: int
    topup_lb_first_consid_period: int
    sb_coi_first_consid_period: int
    admin_chg_start: float
    admin_chg_inc: float
    admin_chg_cap: float
