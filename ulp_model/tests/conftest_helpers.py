"""
Shared test helpers for ulp_model tests.

Provides fixtures for a single-policy test case and matching ParamTables.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from ulp_model.config import ModelConfig
from ulp_model.inputs import ParamTables, PolicyBatch
from ulp_model.loader import load_param_tables

# Project root (where CSV files live) relative to this file's location
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent


def make_config() -> ModelConfig:
    return ModelConfig(
        param_tables_dir=str(_PROJ_ROOT),
        policy_inputs_dir=str(_PROJ_ROOT / "policy_inputs"),
        output_dir=str(_PROJ_ROOT / "results"),
    )


def make_param_tables() -> ParamTables:
    return load_param_tables(make_config())


def make_single_policy(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> PolicyBatch:
    """Create a single test policy:
    - Age at entry: 30, male
    - Policy term: 20 years, premium term: 20 years
    - Annual premium frequency (prem_freq=0)
    - SA: 500,000,000 (500M), DB option 2 (escalating)
    - ACP: 20,000,000 (20M), no top-up
    """
    def _l(v: int) -> torch.Tensor:
        return torch.tensor([v], dtype=torch.long, device=device)

    def _f(v: float) -> torch.Tensor:
        return torch.tensor([v], dtype=dtype, device=device)

    return PolicyBatch(
        policy_id=_l(1),
        age_at_entry=_l(30),
        sex=_l(0),           # male
        pol_term=_l(20),
        prem_term=_l(20),
        prem_freq=_l(0),     # annual
        sum_assd=_f(500_000_000.0),
        db_opt=_l(2),        # escalating
        acp=_f(20_000_000.0),
        atp=_f(0.0),
        topup_term=_l(0),
        topup_freq=_l(0),
        mort_loading=_f(0.0),
        init_pols_if=_f(1.0),
    )
