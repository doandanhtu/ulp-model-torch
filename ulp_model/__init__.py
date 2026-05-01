"""
ulp_model - Universal Life Policy cash flow projection model using PyTorch.

Package structure
-----------------
config.py           load_config() — loads config.yaml into a SimpleNamespace
inputs.py           PolicyBatch and ParamTables dataclasses
loader.py           File-loading utilities
utils.py            Helper functions (lookups, schedule, etc.)
part1_pav.py        Part 1: Policy Account Value projection (S1.1-S1.81)
part2_decrements.py Part 2: Decrement projection (S2.1-S2.13)
part3_cashflows.py  Part 3: Shareholder cashflow projection (S3.1-S3.69)
model.py            ULPModel orchestrator
outputs.py          Output writing and reporting utilities
tests/              Unit and integration tests
"""
from .config import load_config
from .inputs import ParamTables, PolicyBatch
from .loader import load_model_inputs, load_param_tables, load_policy_batch
from .model import ULPModel
from .outputs import (
    compute_ape,
    print_metrics,
    write_per_policy_outputs,
    write_summary_outputs,
)
from .part1_pav import PAVProjection
from .part2_decrements import DecrementProjection
from .part3_cashflows import CashflowProjection
from .utils import (
    attained_age_at_t,
    lookup_coi_rate,
    lookup_lien_pc,
    lookup_mortality_rate,
    pol_year_at_t,
    precompute_bonus_schedule,
)

__all__ = [
    # Config
    "load_config",
    # Inputs
    "PolicyBatch",
    "ParamTables",
    # Loader
    "load_param_tables",
    "load_policy_batch",
    "load_model_inputs",
    # Utils
    "attained_age_at_t",
    "pol_year_at_t",
    "lookup_coi_rate",
    "lookup_mortality_rate",
    "lookup_lien_pc",
    "precompute_bonus_schedule",
    # Parts
    "PAVProjection",
    "DecrementProjection",
    "CashflowProjection",
    # Model
    "ULPModel",
    # Outputs
    "compute_ape",
    "print_metrics",
    "write_summary_outputs",
    "write_per_policy_outputs",
]
