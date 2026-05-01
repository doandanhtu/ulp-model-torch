"""
config.py - Configuration loader for ULP model.

Reads config.yaml and returns a SimpleNamespace so settings are
accessible as attributes (config.MAX_PROJ_MONTHS, config.compute_device, …).
All defaults live in config.yaml — this module only maps YAML keys to
attribute names and derives MAX_PROJ_MONTHS.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def load_config(path: str | Path) -> SimpleNamespace:
    """Load model configuration from a YAML file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    cfg = SimpleNamespace(
        MAX_PROJ_YEARS         = int(raw["max_proj_years"]),
        float_precision        = str(raw["float_precision"]),
        compute_device         = str(raw["compute_device"]),
        batch_size             = int(raw["batch_size"]),
        output_batch_size      = int(raw["output_batch_size"]),
        policy_inputs_file     = str(raw["policy_inputs_file"]),
        param_tables_dir       = str(raw["param_tables_dir"]),
        output_dir             = str(raw["output_dir"]),
        scenario_file          = str(raw["scenario_file"]),
        output_mode            = str(raw["output_mode"]),
        additional_output_vars = raw.get("additional_output_vars"),
        output_time_steps      = raw.get("output_time_steps", "all"),
        n_simulations          = int(raw["n_simulations"]),
    )

    cfg.MAX_PROJ_MONTHS = cfg.MAX_PROJ_YEARS * 12 + 1

    return cfg
