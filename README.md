# PyTorch-Based GPU-Accelerated Actuarial Model

PyTorch implementation of Universal Life Policy (ULP) cash flow projections, designed for large portfolios with GPU acceleration.

For background and motivation, see [About this project](docs/about.md).

> **Disclaimer:** This is a personal project. All policy inputs, parameter tables, and assumptions included in this repository are fabricated for illustration purposes only. This project does not represent the practices, methodologies, or views of any organisation.

## Status

| Component | Status |
|---|---|
| Part 1 — PAV Projection | Done |
| Part 2 — Decrements | Done |
| Part 3 — Shareholder Cashflows | Done |
| Sensitivity analysis | Done |
| Summary output mode | Done |
| Per-policy output mode | Done |
| Stochastic mode | **Not yet implemented** |

The full model specification is available at [`docs/Model_specifications_full_v4_5.md`](docs/Model_specifications_full_v4_5.md). Illustrative input files are provided for reference: parameter tables in [`param_tables/`](param_tables/), sample policy inputs in [`policy_data/`](policy_data/), and sensitivity scenarios in [`sen_fac/`](sen_fac/).

---

## Setup

### CPU only
```bash
pip install -r requirements.txt
```

### GPU (NVIDIA CUDA 12.1)
```bash
pip install -r requirements-gpu.txt
```

---

## Usage

```bash
# Run with default config.yaml in the current directory
python run_model.py

# Specify a config file
python run_model.py --config path/to/config.yaml

# Override output directory
python run_model.py --config config.yaml --output-dir ./results/run1

# Run a single scenario only
python run_model.py --config config.yaml --scenario-id 3

# Override device and output mode
python run_model.py --config config.yaml --device cuda --mode summary
```

---

## Performance
| Number of model points | Excel Python Orchestration | Prophet | PyTorch CPU | GTX 1060 6GB | T4 15GB | A100 40GB | A100 80GB | RTX Pro 6000 Blackwell Server Edition 96GB |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.5m | 20,617s | 2,128s | 244s | 120s | 47s | 17s | 11s | 8.5s |
| 3m | — | — | 1,132s | 650s | 252s | 68s | 47s | 35s |
| 5m | — | — | — | 1,035s | 421s | 116s | 76s | 59s |
| 10m | — | — | — | — | 834s | 221s | 151s | 119s |

<details>
<summary>Benchmark notes</summary>

- All figures in the table are measured runtimes, not projections or extrapolations.
- Excel Python Orchestration refers to an Excel-based actuarial model orchestrated through Python to process multiple Excel instances in parallel.
- Prophet benchmark is based on a standard run without MPF batching, which was not available in the tested setup.
- Excel, Prophet, and PyTorch CPU benchmarks were run on Intel i7-14700 with 32GB RAM.
- GTX 1060 6GB is a local consumer GPU.
- T4, A100, and RTX Pro 6000 Blackwell Server Edition 96GB (G4) results were tested through Google Colab GPU environments using Colab Pro subscription.
- The purpose of the benchmark is not to claim optimal performance, but to demonstrate practical viability and accessibility of GPU-native actuarial modeling using modern open-source tooling.

</details>

---

## Architecture

### Module map

| Module | Role |
|---|---|
| [`run_model.py`](run_model.py) | CLI entry point. Orchestrates scenario loop, applies sensitivity, writes all outputs, prints metrics table. |
| [`generate_policies.py`](generate_policies.py) | Standalone script that generates a random portfolio of a specified size for performance testing. |
| [`ulp_model/config.py`](ulp_model/config.py) | Loads `config.yaml` into a `SimpleNamespace`. |
| [`ulp_model/inputs.py`](ulp_model/inputs.py) | `PolicyBatch` and `ParamTables` dataclasses — the typed containers passed between all model components. |
| [`ulp_model/loader.py`](ulp_model/loader.py) | Reads all parameter tables and policy inputs into tensors; exposes `PolicyBatchIterator` for batched policy loading. |
| [`ulp_model/utils.py`](ulp_model/utils.py) | Shared helpers: age and policy-year calculations, COI/mortality/lien rate lookups, and bonus schedule precomputation. |
| [`ulp_model/model.py`](ulp_model/model.py) | `ULPModel`: top-level runner. `run()` projects a single batch; `run_portfolio()` iterates batches and accumulates summary tensors. |
| [`ulp_model/forward_projection.py`](ulp_model/forward_projection.py) | Part 1 (PAV), Part 2 (decrements) and Part 3 Pass 1 forward pass. Contains the main monthly loop and buffer management. |
| [`ulp_model/part3_cashflows.py`](ulp_model/part3_cashflows.py) | The remaining 3 passes (2 backward, 1 forward) for Part 3 (shareholder cashflows). |
| [`ulp_model/sensitivity.py`](ulp_model/sensitivity.py) | `SensitivityFactors` dataclass, scenario CSV loader, and `apply_sensitivity_factors()`. |
| [`ulp_model/outputs.py`](ulp_model/outputs.py) | APE and metrics computation, console reporting, and CSV writers for summary and per-policy outputs. |

All runtime configuration lives in [`config.yaml`](config.yaml) — see the Configuration section below for the full parameter reference.

### Computation structure

At a high level, the model treats millions of policies as large tensor operations evolving simultaneously across projection months, rather than iterating policy-by-policy in nested loops.

The model follows the three-part dependency chain specified in the model specification:

```
Part 1 (PAV Projection)  →  Part 2 (Decrements)  →  Part 3 (SH Cashflows)
```

Each part performs a forward pass over all monthly time steps `t = 0 … MAX_PROJ_MONTHS − 1`. Part 3 additionally requires two backward passes (zeroising reserve, present values). All three parts execute within a single `model.run()` call, operating on a vectorised batch of `B` policies simultaneously.

For portfolios larger than `batch_size`, `model.run_portfolio()` iterates over batches and accumulates portfolio-level `[T]` summary tensors after each batch, keeping only one batch of full `[B, T]` tensors in memory at a time.

### Memory modes

Each variable is allocated in one of two modes depending on `retain_full_outputs`:

- **Summary mode** (`retain_full_outputs=False`): variables needed only for within-step calculations are allocated as small rolling buffers (e.g., `[B, 2]` or `[B, 3]`) rather than full `[B, T]` tensors, keeping GPU memory bounded.
- **Per-policy mode** (`retain_full_outputs=True`): all variables are allocated as full `[B, T]` tensors so they can be written to the output CSV. This mode requires the portfolio to fit in a single batch.

### Sensitivity architecture

> **Implementation note:** Sensitivity factors are applied to assumption parameter tables **before model execution**, rather than as variables within the calculation formulas defined in the model specification. This means `model.run()` and `model.run_portfolio()` receive pre-adjusted tables and their internal formulas remain unchanged — the specification formulas are implemented exactly as written, with no sensitivity branching inside them.

In practice:
1. `load_param_tables()` loads the base assumptions.
2. `apply_sensitivity_factors()` modifies the loaded tables in-place.
3. The adjusted tables are passed to `model.run_portfolio()` or `model.run()`.

This keeps calculation code clean and makes it straightforward to verify that a sensitivity run with all factors at their base values (multiplicative = 100, additive = 0) produces results identical to the unsensitised base case.

---

## Configuration (`config.yaml`)

| Parameter | Description | Default |
|---|---|---|
| `max_proj_years` | Maximum projection horizon in years. Must be ≥ the longest policy term in the portfolio. Total time steps = `max_proj_years × 12 + 1`. | `90` |
| `float_precision` | Numeric precision for tensor computations. `float64` (recommended) or `float32`. | `float64` |
| `compute_device` | Hardware device: `cpu` or `cuda` (first available GPU). | `cpu` |
| `batch_size` | Policies processed simultaneously per vectorised batch. Smaller values reduce peak VRAM at the cost of slightly higher overhead. Typical range: 10,000–100,000 for a GPU with 6–16 GB VRAM at float64. | `70000` |
| `output_batch_size` | Policies per output file flush in per-policy mode (bounds peak memory during writing). | `1000` |
| `policy_inputs_file` | Path to the policy input file (`.csv` or `.parquet`). Parquet is recommended for large portfolios. | — |
| `param_tables_dir` | Directory containing all parameter table CSVs and `scalar_inputs.yaml`. | — |
| `scenario_file` | Path to the sensitivity scenario CSV. Scenario 1 is always the base (all factors at default). If omitted or missing, the model runs a single base case. | — |
| `output_dir` | Directory where result files are written. Created automatically if it does not exist. | — |
| `output_mode` | `summary`, `per_policy`, or `both`. See Output Modes below. | `summary` |
| `output_time_steps` | Time steps to include in summary output. `all` or a YAML list of integer month indices, e.g. `[0, 12, 24, 60, 120]`. | `all` |
| `additional_output_vars` | Extra Part 1 variables to include in per-policy output, in addition to the default Part 2 + Part 3 set. Accepts a list of variable names or the keyword `"all"`. `null` means default variables only. See Per-Policy Output Variables below. | `null` |
| `n_simulations` | Number of stochastic simulations. Applicable in stochastic mode only (not yet implemented). | `5000` |

---

## Input Files

### Policy inputs

One row per policy (or policy group). Supported formats: CSV, Parquet (recommended).

| Column | Description |
|---|---|
| `policy_id` | Unique policy identifier |
| `age` | Age at entry |
| `sex` | `0` = male, `1` = female |
| `pol_term` | Policy term (years) |
| `prem_term` | Basic premium payable term (years) |
| `sum_assd` | Sum assured |
| `db_opt` | Death benefit option: `1` = Basic (DB = max(SA, PAV)), `2` = Escalating (DB = SA + PAV) |
| `acp` | Annualised committed premium |
| `prem_freq` | Basic premium frequency in months per payment: `12` = annual, `6` = semi-annual, `3` = quarterly, `1` = monthly |
| `atp` | Annualised top-up premium |
| `topup_term` | Top-up premium payable term (years) |
| `topup_freq` | Top-up frequency (same coding as `prem_freq`) |
| `mort_loading` | Substandard mortality loading (%). `0` for standard lives. |
| `init_pols_if` | Initial number of policies in force. Normally `1`; use for grouped calculations. |

### Parameter tables (`param_tables/`)

| File | Description |
|---|---|
| `scalar_inputs.yaml` | Scalar assumptions: `ann_ulp_fer`, `ann_sh_fer`, `ann_fmc_pc`, `ann_fme_pc`, `ann_vir`, `ann_rdr`, `inf_pc`, product feature parameters |
| `alloc_chg_tbl.csv` | Allocation charge rates (%) by policy year for basic and top-up premium |
| `surr_chg_tbl.csv` | Surrender charge rates (%) by policy year |
| `admin_chg_tbl.csv` | Monthly admin charge: start level, annual increase, cap |
| `coi_tbl.csv` | COI rates (per mille) by sex and attained age (ultimate format, ages 0–98) |
| `hard_g_inv_tbl.csv` | Hard guaranteed investment rates (%) by policy year |
| `lien_tbl.csv` | LIEN clause — adjusted SA proportion for ages 0–3 (20%, 40%, 60%, 80%); 100% for age 4+ |
| `op_exp_tbl.csv` | Operational expenses: initial per-policy, initial per-premium %, renewal per-policy, renewal per-premium % |
| `comm_tbl.csv` | Commission rates (%) by policy year for basic and top-up premium |
| `ovrd_tbl.csv` | Overriding commission rates (%) by policy year (basic only) |
| `lapse_tbl.csv` | Lapse rates (%) by policy year and premium frequency (monthly, quarterly, semi-annual, annual) |
| `mortality_select_male.csv` | Male select mortality table. Rows: entry age; columns: `q[x]`, `q[x]+1`, …, `qx+S`. Selection period S is auto-detected from column count: `S = total_data_columns − 1`. |
| `mortality_select_female.csv` | Female select mortality table (same format) |
| `basic_lb_rate_tbl.csv` | Basic loyalty bonus rates (%) by policy year |
| `topup_lb_rate_tbl.csv` | Top-up loyalty bonus rates (%) by policy year |
| `sb_coi_rate_tbl.csv` | Special bonus as % of COI rates by policy year |
| `sb_acp_rate_tbl.csv` | Special bonus as % of ACP rates by policy year |
| `reg_param_tbl.csv` | Regulatory parameters: SM on reserve, SM on SAR, tax rate |

**Mortality table notes:**
- Values are stored as per-mille (e.g., `0.24` = 0.24‰).
- The selection period S is detected automatically from the number of data columns.
- To use an ultimate-only table, supply it as a single-column select table (S = 0).
- The COI table is always ultimate format (attained-age lookup); the mortality table is always select format.

**Lapse table note:** Values are whole-number percentages (e.g., `60` = 60%).

---

## Sensitivity Analysis

The scenario file is a CSV where each row defines one scenario. Scenario 1 must always be the base case (all factors at their default values).

| Factor | Type | Default | Description |
|---|---|---|---|
| `op_exp_sen` | Multiplicative | 100 | Overall multiplier on all operational expenses. Stacks with `ie_pp_sen`, `re_pp_sen`, etc. |
| `ie_pp_sen` | Multiplicative | 100 | Initial expense per policy |
| `ie_pc_sen` | Additive (pp) | 0 | Initial expense per premium (percentage points added) |
| `re_pp_sen` | Multiplicative | 100 | Renewal expense per policy |
| `re_pc_sen` | Additive (pp) | 0 | Renewal expense per premium |
| `inf_sen` | Additive (pp) | 0 | Expense inflation rate |
| `fme_sen` | Additive (pp) | 0 | Fund management expense rate |
| `fmc_sen` | Additive (pp) | 0 | Fund management charge rate |
| `comm_sen` | Multiplicative | 100 | Commission (basic premium only) |
| `ovrd_sen` | Multiplicative | 100 | Overriding commission |
| `mort_sen` | Multiplicative | 100 | Mortality rate. Result is capped at 1000 (per-mille units). |
| `lapse_sen` | Multiplicative | 100 | Lapse rate. Result is capped at 100 (percentage-point units). |
| `ulp_fer_sen` | Additive (pp) | 0 | ULP fund earned rate |
| `sh_fer_sen` | Additive (pp) | 0 | SH fund earned rate |
| `rdr_sen` | Additive (pp) | 0 | Risk discount rate |
| `vir_sen` | Additive (pp) | 0 | Valuation interest rate |

Multiplicative factors: `new_value = base_value × (factor / 100)`.  
Additive factors: `new_value = base_value + factor` (both base and sensitivity values are in percentage points).

---

## Output Modes

| Mode | Description |
|---|---|
| `summary` | One aggregated time-series CSV per scenario. Fast, low memory. Supports any number of batches. |
| `per_policy` | One per-policy CSV per scenario. Requires the portfolio to fit within a single batch (enforced by a pre-flight check). |
| `both` | Summary + per-policy in a single run. Single-batch constraint applies. |
| `stochastic` | **Not yet implemented.** |

For `per_policy` and `both`, the model also enforces an Excel row-count limit (1,000,000 rows). Use `summary` for large portfolios.

After all scenarios complete, a `scenario_metrics_summary.csv` is written to `output_dir` containing elapsed time and key metrics (APE, PV CF, ratios) for each scenario. If any APE exceeds 1 billion, monetary columns are automatically displayed in millions with headers labelled `(in mil)`.

---

## Per-Policy Output Variables

The output CSV always contains the **default variables** (Part 2 + Part 3). Additional Part 1 variables can be requested via `additional_output_vars`.

Set `additional_output_vars` to:
- A list of Part 1 variable names (see table below)
- `"all"` — include every variable from Part 1, Part 2, and Part 3

### Default output — Part 2 (decrements)

| Variable | Description |
|---|---|
| `no_pols_if` | Policies in force (start of month) |
| `no_pols_ifsm` | Policies in force (mid-year basis) |
| `no_deaths` | Deaths |
| `no_surrs` | Surrenders |
| `no_mats` | Maturities |

### Default output — Part 3 (shareholder cashflows)

| Variable | Description |
|---|---|
| `prem_inc_if` | Total premium income |
| `basic_prem_if` | Basic premium income |
| `topup_prem_if` | Top-up premium income |
| `op_init_exp_if` | Initial operating expense |
| `op_ren_exp_if` | Renewal operating expense |
| `invt_exp_if` | Investment expense |
| `comm_if` | Commission |
| `ovrd_if` | Overriding commission |
| `death_outgo` | Death benefit outgo |
| `surr_outgo` | Surrender benefit outgo |
| `mat_outgo` | Maturity benefit outgo |
| `cog_term_adj` | Cost of guarantee / termination adjustment |
| `unit_res_bgn` | Unit reserve (beginning of month) |
| `unit_res_end` | Unit reserve (end of month) |
| `unit_inc` | Unit fund income |
| `non_unit_inc` | Non-unit fund income |
| `cf_before_zv` | Cashflow before zeroising reserve |
| `zeroising_res_if` | Zeroising reserve |
| `cf_after_zv` | Cashflow after zeroising reserve |
| `op_tax` | Operating tax |
| `cf_after_tax` | Cashflow after tax |
| `tot_res_if` | Total reserve |
| `solv_cap_req` | Solvency capital requirement |
| `scr_inv_inc` | SCR investment income |
| `scr_inc_tax` | SCR income tax |
| `cf_after_scr` | Cashflow after SCR |
| `pv_cf_after_scr` | Present value of cashflow after SCR |
| `pv_prem_inc` | Present value of premium income |

### Additional variables — Part 1 (PAV projection)

| Variable | Description |
|---|---|
| `is_inforce_bgn` | In-force indicator (beginning of month) |
| `is_inforce_end` | In-force indicator (end of month) |
| `current_db_opt` | Current death benefit option |
| `bav_ab` | Basic account value — after bonus |
| `tuav_ab` | Top-up account value — after bonus |
| `av_ab` | Total account value — after bonus |
| `bav_ad` | Basic account value — after deductions |
| `tuav_ad` | Top-up account value — after deductions |
| `av_ad` | Total account value — after deductions |
| `bav_bval_bb` | Basic account value — before value, before beginning |
| `tuav_bval_bb` | Top-up account value — before value, before beginning |
| `bav_aval_bb` | Basic account value — after value, before beginning |
| `tuav_aval_bb` | Top-up account value — after value, before beginning |
| `g_bav_bval` | Guaranteed basic account value — before value |
| `g_tuav_bval` | Guaranteed top-up account value — before value |
| `g_bav_ab` | Guaranteed basic account value — after bonus |
| `g_tuav_ab` | Guaranteed top-up account value — after bonus |
| `tot_dedncf` | Total deduction cashflow |
| `tot_dedn_act` | Total deduction (actual) |
| `basic_alloc_chg_pp` | Basic allocation charge per policy |
| `topup_alloc_chg_pp` | Top-up allocation charge per policy |
| `basic_prem_pp` | Basic premium per policy |
| `topup_prem_pp` | Top-up premium per policy |
| `mort_coi` | Mortality cost of insurance |
| `m_ulp_fer` | Monthly ULP fund earning rate |
| `bonus_alloc` | Bonus allocation |

### Variables that do NOT exist as outputs

Any name specified in `additional_output_vars` that is not a key in Part 1, Part 2, or Part 3 output dicts will produce a **column of zeros** in the output — no error is raised. Use `additional_output_vars: "all"` to discover every available variable name, or consult the tables above. Common examples of names that do not exist as stored outputs:

- `av_bd` — `bav_bd` / `tuav_bd` are intermediate local variables in the projection loop, not retained. Use `bav_ad` / `tuav_ad` instead.
- `av_bval_bb` — no combined basic + top-up "before value, before beginning" is stored. Use `bav_bval_bb` and `tuav_bval_bb` separately.
