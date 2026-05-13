"""
sensitivity.py - Sensitivity scenario table loading and application.

Loads the sensitivity scenario table from CSV and provides utilities to apply
sensitivity factors to model assumptions and rates.

Sensitivity factors are applied as follows:
  - Multiplicative factors (e.g., 100% = no change, 150% = +50%):
    new_value = base_value * (factor / 100)
    Includes: ie_pp_sen, re_pp_sen, op_exp_sen, comm_sen, ovrd_sen,
              mort_sen, lapse_sen

  - Additive factors (e.g., 0% = no change, +0.5% = +0.5 percentage points):
    new_value = base_value + factor
    Includes: ie_pc_sen, re_pc_sen, inf_sen, fme_sen, ulp_fer_sen,
              sh_fer_sen, rdr_sen, vir_sen, fmc_sen
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


@dataclass
class SensitivityFactors:
    """Holds sensitivity factors for a single scenario."""

    scenario_id: int

    # Expense sensitivity factors (multiplicative: 100 = base)
    op_exp_sen: float = 100.0         # Multiplicative: all op_exp
    ie_pp_sen: float = 100.0          # Multiplicative: initial per-policy
    ie_pc_sen: float = 0.0            # Additive: initial per-premium %
    re_pp_sen: float = 100.0          # Multiplicative: renewal per-policy
    re_pc_sen: float = 0.0            # Additive: renewal per-premium %
    inf_sen: float = 0.0              # Additive: expense inflation %

    # Fund management and commission (mixed)
    fme_sen: float = 0.0              # Additive: fund mgmt expense %
    fmc_sen: float = 0.0              # Additive: fund mgmt charge %
    comm_sen: float = 100.0           # Multiplicative: commission
    ovrd_sen: float = 100.0           # Multiplicative: overriding

    # Decrement sensitivity (multiplicative)
    mort_sen: float = 100.0           # Multiplicative: mortality
    lapse_sen: float = 100.0          # Multiplicative: lapse

    # Rate assumption sensitivity (additive, in %)
    ulp_fer_sen: float = 0.0          # Additive: ULP FER %
    sh_fer_sen: float = 0.0           # Additive: SH FER %
    rdr_sen: float = 0.0              # Additive: RDR %
    vir_sen: float = 0.0              # Additive: VIR %


def load_sensitivity_scenarios(scenario_file: str | Path) -> dict[int, SensitivityFactors]:
    """Load sensitivity scenario table from CSV.

    Parameters
    ----------
    scenario_file : str or Path
        Path to CSV file with columns: Scen ID (or Scenario ID), then sensitivity factor columns.

    Returns
    -------
    dict[int, SensitivityFactors]
        Dictionary mapping scenario_id to SensitivityFactors object.

    Notes
    -----
    - Scenario 1 is always the base scenario (all factors at default).
    - CSV columns are matched case-insensitively to SensitivityFactors fields.
    - Missing columns are assumed to take their default values (100 for multiplicative,
      0 for additive).
    - First column can be "Scen ID" or "Scenario ID" (case-insensitive).
    """
    df = pd.read_csv(scenario_file)

    # Normalize column names to lowercase for matching
    df.columns = df.columns.str.lower().str.strip()

    scenarios = {}
    for _, row in df.iterrows():
        # Handle both "Scen ID" and "Scenario ID" column names
        if "scen id" in df.columns:
            scenario_id = int(row["scen id"])
        elif "scenario id" in df.columns:
            scenario_id = int(row["scenario id"])
        else:
            raise ValueError(
                f"CSV must have 'Scen ID' or 'Scenario ID' column. "
                f"Found: {list(df.columns)}"
            )

        # Build kwargs from row data, matching column names to SensitivityFactors fields
        kwargs = {"scenario_id": scenario_id}
        for field in SensitivityFactors.__dataclass_fields__.keys():
            if field == "scenario_id":
                continue
            col_name = field.lower()
            if col_name in df.columns:
                kwargs[field] = float(row[col_name])

        scenarios[scenario_id] = SensitivityFactors(**kwargs)

    return scenarios


def apply_sensitivity_factors(
    param_tables,
    sensitivity: SensitivityFactors,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Apply sensitivity factors to param_tables in-place.

    Modifies the ParamTables object to apply the given sensitivity factors.
    After this function, param_tables contains the adjusted assumptions.

    Parameters
    ----------
    param_tables : ParamTables
        Parameter tables to modify (modified in-place).
    sensitivity : SensitivityFactors
        Sensitivity factors to apply.
    device : torch.device
        Torch device (for creating tensor copies if needed).
    dtype : torch.dtype
        Torch dtype (for creating tensor copies if needed).

    Notes
    -----
    Expense factors stack multiplicatively:
      final_expense = base_expense × (op_exp_sen/100) × (ie_pp_sen/100) × ...
    """
    # ------------------------------------------------------------------
    # Operational expense sensitivity (multiplicative stacking)
    # ------------------------------------------------------------------
    # 1. Overall multiplier on all expenses
    mult_op_exp = sensitivity.op_exp_sen / 100.0
    param_tables.op_exp_per_pol = param_tables.op_exp_per_pol * mult_op_exp
    param_tables.op_exp_per_prem = param_tables.op_exp_per_prem * mult_op_exp

    # 2. Individual per-policy multipliers (stack with overall)
    mult_ie_pp = sensitivity.ie_pp_sen / 100.0
    param_tables.op_exp_per_pol[0] = param_tables.op_exp_per_pol[0] * mult_ie_pp

    mult_re_pp = sensitivity.re_pp_sen / 100.0
    param_tables.op_exp_per_pol[1] = param_tables.op_exp_per_pol[1] * mult_re_pp

    # 3. Additive per-premium rates: values in param_tables and sensitivity are both in %, add directly
    param_tables.op_exp_per_prem[0] = param_tables.op_exp_per_prem[0] + sensitivity.ie_pc_sen
    param_tables.op_exp_per_prem[1] = param_tables.op_exp_per_prem[1] + sensitivity.re_pc_sen

    # ------------------------------------------------------------------
    # Expense inflation (additive, both sides in %)
    # ------------------------------------------------------------------
    param_tables.inf_pc = param_tables.inf_pc + sensitivity.inf_sen

    # ------------------------------------------------------------------
    # Fund management expense and charge (additive, both sides in %)
    # ------------------------------------------------------------------
    param_tables.ann_fme_pc = param_tables.ann_fme_pc + sensitivity.fme_sen
    param_tables.ann_fmc_pc = param_tables.ann_fmc_pc + sensitivity.fmc_sen

    # ------------------------------------------------------------------
    # Commission and overriding (multiplicative)
    # ------------------------------------------------------------------
    mult_comm = sensitivity.comm_sen / 100.0
    param_tables.comm_basic = param_tables.comm_basic * mult_comm
    param_tables.comm_topup = param_tables.comm_topup * mult_comm

    mult_ovrd = sensitivity.ovrd_sen / 100.0
    param_tables.ovrd = param_tables.ovrd * mult_ovrd

    # ------------------------------------------------------------------
    # Decrement rates (multiplicative, clamped to max 1.0)
    # ------------------------------------------------------------------
    mult_mort = sensitivity.mort_sen / 100.0
    param_tables.mortality_male = (param_tables.mortality_male * mult_mort).clamp(max=1000.0)
    param_tables.mortality_female = (param_tables.mortality_female * mult_mort).clamp(max=1000.0)

    mult_lapse = sensitivity.lapse_sen / 100.0
    param_tables.lapse_rates = (param_tables.lapse_rates * mult_lapse).clamp(max=100.0)

    # ------------------------------------------------------------------
    # Rate assumptions (additive, both sides in %)
    # ------------------------------------------------------------------
    param_tables.ann_ulp_fer = param_tables.ann_ulp_fer + sensitivity.ulp_fer_sen
    param_tables.ann_sh_fer = param_tables.ann_sh_fer + sensitivity.sh_fer_sen
    param_tables.ann_rdr = param_tables.ann_rdr + sensitivity.rdr_sen
    param_tables.ann_vir = param_tables.ann_vir + sensitivity.vir_sen
