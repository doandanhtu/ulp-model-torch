"""
ulp_model - Universal Life Policy cash flow projection model using PyTorch.

Package structure
-----------------
config.py           load_config() — loads config.yaml into a SimpleNamespace
inputs.py           PolicyBatch and ParamTables dataclasses
loader.py           File-loading utilities
utils.py            Helper functions (lookups, schedule, etc.)
forward_projection.py Single-pass monthly forward projection
                      (Part 1 + Part 2 + Part 3 Pass 1)
part3_cashflows.py    Cashflow finalisation (Part 3 Pass 2/3/4); consumes
                      Pass 1 outputs from forward_projection.
model.py              ULPModel orchestrator (uses ForwardProjection)
outputs.py            Output writing and reporting utilities
"""
from .config import load_config
from .forward_projection import ForwardProjection
from .inputs import ParamTables, PolicyBatch
from .loader import load_model_inputs, load_param_tables, load_policy_batch
from .model import ULPModel
from .outputs import (
    compute_ape,
    print_metrics,
    write_per_policy_outputs,
    write_summary_outputs,
)
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
    "load_config",
    "PolicyBatch",
    "ParamTables",
    "load_param_tables",
    "load_policy_batch",
    "load_model_inputs",
    "attained_age_at_t",
    "pol_year_at_t",
    "lookup_coi_rate",
    "lookup_mortality_rate",
    "lookup_lien_pc",
    "precompute_bonus_schedule",
    "ForwardProjection",
    "PAVProjection",
    "DecrementProjection",
    "CashflowProjection",
    "ULPModel",
    "compute_ape",
    "print_metrics",
    "write_summary_outputs",
    "write_per_policy_outputs",
]
