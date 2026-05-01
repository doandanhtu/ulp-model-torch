"""
loader.py - File loading utilities for ULP model parameter tables and policy inputs.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .config import load_config
from .inputs import ParamTables, PolicyBatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: str | Path) -> list[dict]:
    """Read a CSV file and return a list of dicts (header row as keys)."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _build_1d_table(
    rows: list[dict],
    year_col: str,
    value_col: str,
    max_years: int,
    dtype: torch.dtype,
    device: torch.device,
    sparse: bool = False,
) -> torch.Tensor:
    """Build a 1-D table of length max_years+1 indexed by pol_year.

    Index 0 is unused (set to 0).
    Indices 1..last_row filled from CSV data.
    Indices last_row+1..max_years filled with last CSV row value (extension rule).
    If sparse=True, missing years are left as 0 (no extension – for award-rate tables).
    """
    table = torch.zeros(max_years + 1, dtype=dtype, device=device)
    last_year = 0
    last_val = 0.0
    for row in rows:
        yr = int(row[year_col])
        val = float(row[value_col])
        if yr <= max_years:
            table[yr] = val
            if yr > last_year:
                last_year = yr
                last_val = val
    if not sparse and last_year < max_years:
        table[last_year + 1:] = last_val
    return table


def _build_1d_table_multi(
    rows: list[dict],
    year_col: str,
    value_cols: list[str],
    max_years: int,
    dtype: torch.dtype,
    device: torch.device,
    sparse: bool = False,
) -> torch.Tensor:
    """Like _build_1d_table but for multiple value columns; returns [max_years+1, C]."""
    C = len(value_cols)
    table = torch.zeros(max_years + 1, C, dtype=dtype, device=device)
    last_year = 0
    last_vals = [0.0] * C
    for row in rows:
        yr = int(row[year_col])
        vals = [float(row[c]) for c in value_cols]
        if yr <= max_years:
            table[yr] = torch.tensor(vals, dtype=dtype, device=device)
            if yr > last_year:
                last_year = yr
                last_vals = vals
    if not sparse and last_year < max_years:
        ext = torch.tensor(last_vals, dtype=dtype, device=device)
        table[last_year + 1:] = ext
    return table


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_param_tables(config) -> ParamTables:
    """Load all parameter tables from config.param_tables_dir."""
    base = Path(config.param_tables_dir)
    max_years = config.MAX_PROJ_YEARS
    dtype = torch.float64 if config.float_precision == "float64" else torch.float32
    device = torch.device(config.compute_device)

    # ----- scalar_inputs.yaml -----
    with open(base / "scalar_inputs.yaml", encoding="utf-8") as f:
        scalars = yaml.safe_load(f)

    ann_ulp_fer: float = float(scalars["ann_ulp_fer"])
    ann_sh_fer: float = float(scalars["ann_sh_fer"])
    ann_fmc_pc: float = float(scalars["ann_fmc_pc"])
    ann_fme_pc: float = float(scalars["ann_fme_pc"])
    ann_vir: float = float(scalars["ann_vir"])
    ann_rdr: float = float(scalars["ann_rdr"])
    inf_pc: float = float(scalars["inf_pc"])
    nlg_period: int = int(scalars["nlg_period"])
    age_db_opt_change: int = int(scalars["age_db_opt_change"])
    basic_lb_first_consid_period: int = int(scalars["basic_lb_first_consid_period"])
    topup_lb_first_consid_period: int = int(scalars["topup_lb_first_consid_period"])
    sb_coi_first_consid_period: int = int(scalars["sb_coi_first_consid_period"])

    # ----- alloc_chg_tbl.csv -----
    rows = _read_csv(base / "alloc_chg_tbl.csv")
    alloc_chg_basic = _build_1d_table(rows, "pol_year", "basic_prem", max_years, dtype, device)
    alloc_chg_topup = _build_1d_table(rows, "pol_year", "topup_prem", max_years, dtype, device)

    # ----- surr_chg_tbl.csv -----
    rows = _read_csv(base / "surr_chg_tbl.csv")
    surr_chg = _build_1d_table(rows, "pol_year", "sc_rate", max_years, dtype, device)

    # ----- admin_chg_tbl.csv -----
    rows = _read_csv(base / "admin_chg_tbl.csv")
    admin_chg_start = float(rows[0]["admin_chg_start"])
    admin_chg_inc = float(rows[0]["admin_chg_inc"])
    admin_chg_cap = float(rows[0]["admin_chg_cap"])

    # ----- hard_g_inv_tbl.csv -----
    rows = _read_csv(base / "hard_g_inv_tbl.csv")
    hard_g_inv = _build_1d_table(rows, "pol_year", "g_inv", max_years, dtype, device)

    # ----- comm_tbl.csv -----
    rows = _read_csv(base / "comm_tbl.csv")
    comm_basic = _build_1d_table(rows, "pol_year", "basic_prem", max_years, dtype, device)
    comm_topup = _build_1d_table(rows, "pol_year", "topup_prem", max_years, dtype, device)

    # ----- ovrd_tbl.csv -----
    rows = _read_csv(base / "ovrd_tbl.csv")
    ovrd = _build_1d_table(rows, "pol_year", "basic_prem", max_years, dtype, device)

    # ----- basic_lb_rate_tbl.csv (sparse – award years only) -----
    rows = _read_csv(base / "basic_lb_rate_tbl.csv")
    basic_lb_rate = _build_1d_table(rows, "pol_year", "basic_lb_rate", max_years, dtype, device, sparse=True)

    # ----- topup_lb_rate_tbl.csv (sparse) -----
    rows = _read_csv(base / "topup_lb_rate_tbl.csv")
    topup_lb_rate = _build_1d_table(rows, "pol_year", "topup_lb_rate", max_years, dtype, device, sparse=True)

    # ----- sb_coi_rate_tbl.csv (sparse) -----
    rows = _read_csv(base / "sb_coi_rate_tbl.csv")
    sb_coi_rate = _build_1d_table(rows, "pol_year", "sb_coi_rate", max_years, dtype, device, sparse=True)

    # ----- sb_acp_rate_tbl.csv (sparse) -----
    rows = _read_csv(base / "sb_acp_rate_tbl.csv")
    sb_acp_rate = _build_1d_table(rows, "pol_year", "sb_acp_rate", max_years, dtype, device, sparse=True)

    # ----- op_exp_tbl.csv -----
    rows = _read_csv(base / "op_exp_tbl.csv")
    op_exp_per_pol = torch.tensor(
        [float(rows[0]["exp_per_pol"]), float(rows[1]["exp_per_pol"])],
        dtype=dtype, device=device,
    )
    op_exp_per_prem = torch.tensor(
        [float(rows[0]["exp_per_prem"]), float(rows[1]["exp_per_prem"])],
        dtype=dtype, device=device,
    )

    # ----- lapse_tbl.csv -----
    rows = _read_csv(base / "lapse_tbl.csv")
    lapse_rates = _build_1d_table_multi(
        rows, "pol_year", ["monthly", "quarterly", "semiann", "annual"],
        max_years, dtype, device,
    )

    # ----- coi_tbl.csv -----
    rows = _read_csv(base / "coi_tbl.csv")
    coi_male_list = [float(r["male"]) for r in rows]
    coi_female_list = [float(r["female"]) for r in rows]
    coi_table_male = torch.tensor(coi_male_list, dtype=dtype, device=device)
    coi_table_female = torch.tensor(coi_female_list, dtype=dtype, device=device)

    # ----- lien_tbl.csv -----
    rows = _read_csv(base / "lien_tbl.csv")
    lien_list = [float(r["lien_pc"]) for r in rows]
    lien_table = torch.tensor(lien_list, dtype=dtype, device=device)

    # ----- reg_param_tbl.csv -----
    rows = _read_csv(base / "reg_param_tbl.csv")
    solv_marg_res = float(rows[0]["solv_marg_res"])
    solv_marg_sar = float(rows[0]["solv_marg_sar"])
    tax_pc = float(rows[0]["tax_pc"])

    # ----- mortality tables -----
    mortality_male, mort_s_male, min_age_male = _load_mortality_table(
        base, "male", dtype, device
    )
    mortality_female, mort_s_female, min_age_female = _load_mortality_table(
        base, "female", dtype, device
    )
    assert mort_s_male == mort_s_female, "Mortality select periods must match."
    assert min_age_male == min_age_female, "Minimum table ages must match between male and female tables."
    mortality_select_period = mort_s_male
    mortality_min_age = min_age_male

    return ParamTables(
        alloc_chg_basic=alloc_chg_basic,
        alloc_chg_topup=alloc_chg_topup,
        surr_chg=surr_chg,
        hard_g_inv=hard_g_inv,
        comm_basic=comm_basic,
        comm_topup=comm_topup,
        ovrd=ovrd,
        basic_lb_rate=basic_lb_rate,
        topup_lb_rate=topup_lb_rate,
        sb_coi_rate=sb_coi_rate,
        sb_acp_rate=sb_acp_rate,
        lapse_rates=lapse_rates,
        coi_table_male=coi_table_male,
        coi_table_female=coi_table_female,
        lien_table=lien_table,
        mortality_male=mortality_male,
        mortality_female=mortality_female,
        mortality_select_period=mortality_select_period,
        mortality_min_age=mortality_min_age,
        op_exp_per_pol=op_exp_per_pol,
        op_exp_per_prem=op_exp_per_prem,
        solv_marg_res=solv_marg_res,
        solv_marg_sar=solv_marg_sar,
        tax_pc=tax_pc,
        ann_ulp_fer=ann_ulp_fer,
        ann_sh_fer=ann_sh_fer,
        ann_fmc_pc=ann_fmc_pc,
        ann_fme_pc=ann_fme_pc,
        ann_vir=ann_vir,
        ann_rdr=ann_rdr,
        inf_pc=inf_pc,
        nlg_period=nlg_period,
        age_db_opt_change=age_db_opt_change,
        basic_lb_first_consid_period=basic_lb_first_consid_period,
        topup_lb_first_consid_period=topup_lb_first_consid_period,
        sb_coi_first_consid_period=sb_coi_first_consid_period,
        admin_chg_start=admin_chg_start,
        admin_chg_inc=admin_chg_inc,
        admin_chg_cap=admin_chg_cap,
    )


def _load_mortality_table(
    base_dir: Path, gender: str, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, int, int]:
    """Find and parse a mortality CSV with select format.

    Returns (table [n_rows, S+1], S, min_table_age).
    S is auto-detected: S = (data columns) - 1.
    min_table_age is the minimum value in the age[x] column.
    """
    matches = list(base_dir.glob(f"mortality_select*_{gender}.csv"))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No mortality table found for '{gender}' in '{base_dir}'. "
            f"Expected a file matching mortality_select_*_{gender}.csv"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple mortality tables found for '{gender}' in '{base_dir}': {matches}"
        )

    rows = _read_csv(matches[0])
    col_names = list(rows[0].keys())
    value_cols = col_names[1:]  # skip the age[x] column

    S = len(value_cols) - 1
    min_table_age = min(int(row[col_names[0]]) for row in rows)

    n_rows = len(rows)
    table = torch.zeros(n_rows, len(value_cols), dtype=dtype, device=device)
    for i, row in enumerate(rows):
        for j, col in enumerate(value_cols):
            table[i, j] = float(row[col])
    return table, S, min_table_age


def load_policy_batch(
    config, device: torch.device, dtype: torch.dtype
) -> PolicyBatch:
    """Load policies from a single CSV or Parquet file specified in config.policy_inputs_file."""
    import pandas as pd

    pol_file = Path(config.policy_inputs_file)
    if not pol_file.exists():
        raise FileNotFoundError(
            f"Policy input file not found: '{pol_file}'. "
            f"Check policy_inputs_file in your config."
        )

    suffix = pol_file.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(pol_file)
    elif suffix == ".csv":
        df = pd.read_csv(pol_file)
    else:
        raise ValueError(
            f"Unsupported policy file format '{suffix}' in '{pol_file}'. "
            f"Supported formats: .csv, .parquet"
        )

    required_cols = [
        "policy_id", "age_at_entry", "sex", "pol_term", "prem_term",
        "prem_freq", "sum_assd", "db_opt", "acp", "atp",
        "topup_term", "topup_freq", "mort_loading", "init_pols_if",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Policy file '{pol_file}' is missing required columns: {missing}"
        )

    def _int(col: str) -> torch.Tensor:
        return torch.tensor(df[col].values, dtype=torch.int32, device=device)

    def _float(col: str) -> torch.Tensor:
        return torch.tensor(df[col].values, dtype=dtype, device=device)

    return PolicyBatch(
        policy_id=_int("policy_id"),
        age_at_entry=_int("age_at_entry"),
        sex=_int("sex"),
        pol_term=_int("pol_term"),
        prem_term=_int("prem_term"),
        prem_freq=_int("prem_freq"),
        sum_assd=_float("sum_assd"),
        db_opt=_int("db_opt"),
        acp=_float("acp"),
        atp=_float("atp"),
        topup_term=_int("topup_term"),
        topup_freq=_int("topup_freq"),
        mort_loading=_float("mort_loading"),
        init_pols_if=_float("init_pols_if"),
    )


def load_model_inputs(config) -> tuple[PolicyBatch, ParamTables]:
    """Convenience function: load both policies and param tables."""
    dtype = torch.float64 if config.float_precision == "float64" else torch.float32
    device = torch.device(config.compute_device)
    param_tables = load_param_tables(config)
    policies = load_policy_batch(config, device, dtype)
    return policies, param_tables
