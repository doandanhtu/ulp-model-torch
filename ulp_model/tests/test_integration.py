"""
test_integration.py - Integration tests for the full ULP model pipeline.

Runs the complete pipeline (Part1 → Part2 → Part3 → ULPModel) and verifies
end-to-end consistency between parts and key actuarial identities.
"""
from __future__ import annotations

import math

import pytest
import torch

from ulp_model.model import ULPModel
from ulp_model.outputs import compute_ape
from ulp_model.part1_pav import PAVProjection
from ulp_model.part2_decrements import DecrementProjection
from ulp_model.part3_cashflows import CashflowProjection

from .conftest_helpers import make_config, make_param_tables, make_single_policy


@pytest.fixture(scope="module")
def full_run():
    config = make_config()
    policies = make_single_policy()
    param_tables = make_param_tables()
    model = ULPModel(config)
    result = model.run(policies, param_tables)
    return result, policies, param_tables, config


class TestFullPipeline:

    def test_run_returns_all_keys(self, full_run):
        """Model.run() must return part1, part2, part3, summary, elapsed."""
        result, _, _, _ = full_run
        for key in ("part1", "part2", "part3", "summary", "elapsed"):
            assert key in result, f"Missing key '{key}' in model output"

    def test_elapsed_positive(self, full_run):
        """Elapsed time should be a positive number."""
        result, _, _, _ = full_run
        assert result["elapsed"] > 0.0

    def test_part1_keys_present(self, full_run):
        """Part 1 output should contain key tensors."""
        result, _, _, _ = full_run
        for key in ("bav_ab", "av_ab", "is_inforce_end", "is_inforce_bgn"):
            assert key in result["part1"], f"Missing part1 key '{key}'"

    def test_part2_keys_present(self, full_run):
        """Part 2 output should contain decrement tensors."""
        result, _, _, _ = full_run
        for key in ("no_pols_if", "no_deaths", "no_surrs", "no_mats"):
            assert key in result["part2"], f"Missing part2 key '{key}'"

    def test_part3_keys_present(self, full_run):
        """Part 3 output should contain cashflow tensors."""
        result, _, _, _ = full_run
        for key in ("prem_inc_if", "cf_before_zv", "cf_after_scr", "pv_cf_after_scr"):
            assert key in result["part3"], f"Missing part3 key '{key}'"

    def test_summary_sums_batch(self, full_run):
        """summary['no_pols_if'] should equal sum over batch of part2['no_pols_if']."""
        result, _, _, _ = full_run
        p2_sum = result["part2"]["no_pols_if"].sum(dim=0)
        assert torch.allclose(result["summary"]["no_pols_if"], p2_sum)

    def test_no_pols_if_starts_at_init(self, full_run):
        """Summary no_pols_if at t=0 should equal total init_pols_if."""
        result, policies, _, _ = full_run
        expected = float(policies.init_pols_if.sum())
        assert float(result["summary"]["no_pols_if"][0]) == pytest.approx(expected)

    def test_total_decrements_sum_to_init(self, full_run):
        """Total deaths + surrs + mats across projection = init_pols_if."""
        result, policies, _, _ = full_run
        total_d = float(result["summary"]["no_deaths"].sum())
        total_s = float(result["summary"]["no_surrs"].sum())
        total_m = float(result["summary"]["no_mats"].sum())
        total_out = total_d + total_s + total_m
        init = float(policies.init_pols_if.sum())
        assert total_out == pytest.approx(init, rel=1e-4)

    def test_pv_prem_positive(self, full_run):
        """PV of premium income at t=0 should be positive."""
        result, _, _, _ = full_run
        assert float(result["summary"]["pv_prem_inc"][0]) > 0.0

    def test_pv_cf_finite(self, full_run):
        """PV cashflow should be finite (no NaN/Inf in summary)."""
        result, _, _, _ = full_run
        assert torch.isfinite(result["summary"]["pv_cf_after_scr"]).all()

    def test_cf_after_zv_t0_zero(self, full_run):
        """Aggregate cf_after_zv at t=0 must be 0."""
        result, _, _, _ = full_run
        assert float(result["summary"]["cf_after_zv"][0]) == pytest.approx(0.0, abs=1e-9)

    def test_cf_after_zv_nonneg_from_t2(self, full_run):
        """Aggregate cf_after_zv >= 0 for t >= 2 (t=1 is the only allowed negative)."""
        result, _, _, _ = full_run
        assert (result["summary"]["cf_after_zv"][2:] >= -1e-6).all()

    def test_unit_res_end_matches_av_at_maturity(self, full_run):
        """At maturity month, unit reserve should drop to zero."""
        result, policies, _, _ = full_run
        pol_term = int(policies.pol_term[0])
        mat_t = pol_term * 12
        unit_res = float(result["part3"]["unit_res_end"][0, mat_t])
        assert unit_res == pytest.approx(0.0, abs=1.0)  # AV should be paid out


class TestAPEComputation:

    def test_ape_positive(self):
        """APE should be positive for a non-single-pay policy."""
        policies = make_single_policy()
        ape = compute_ape(policies)
        assert ape > 0.0

    def test_ape_equals_acp_for_annual_20yr(self):
        """For annual, 20-year term: APE = ACP * 1.0 (non single-pay)."""
        policies = make_single_policy()
        acp = float(policies.acp[0])
        ape = compute_ape(policies)
        # No top-up, so APE = 100% * ACP
        assert ape == pytest.approx(acp, rel=1e-9)

    def test_ape_single_pay_is_10pct(self):
        """Single-pay policy APE = 10% of ACP."""
        import torch

        def _l(v):
            return torch.tensor([v], dtype=torch.long)

        def _f(v):
            return torch.tensor([v], dtype=torch.float64)

        from ulp_model.inputs import PolicyBatch

        pol = PolicyBatch(
            policy_id=_l(99),
            age_at_entry=_l(40),
            sex=_l(0),
            pol_term=_l(10),
            prem_term=_l(1),    # single pay
            prem_freq=_l(0),    # annual (single pay check)
            sum_assd=_f(1_000_000.0),
            db_opt=_l(1),
            acp=_f(100_000.0),
            atp=_f(0.0),
            topup_term=_l(0),
            topup_freq=_l(0),
            mort_loading=_f(0.0),
            init_pols_if=_f(1.0),
        )
        ape = compute_ape(pol)
        assert ape == pytest.approx(10_000.0, rel=1e-9)


class TestMultiPolicyBatch:
    """Tests that a 2-policy batch produces consistent results."""

    @pytest.fixture(scope="class")
    def two_policy_run(self):
        import torch
        from ulp_model.inputs import PolicyBatch

        def _l(*vals):
            return torch.tensor(vals, dtype=torch.long)

        def _f(*vals):
            return torch.tensor(vals, dtype=torch.float64)

        policies = PolicyBatch(
            policy_id=_l(1, 2),
            age_at_entry=_l(30, 45),
            sex=_l(0, 1),
            pol_term=_l(20, 15),
            prem_term=_l(20, 15),
            prem_freq=_l(0, 3),     # annual / monthly
            sum_assd=_f(500_000_000.0, 200_000_000.0),
            db_opt=_l(2, 1),
            acp=_f(20_000_000.0, 10_000_000.0),
            atp=_f(0.0, 0.0),
            topup_term=_l(0, 0),
            topup_freq=_l(0, 0),
            mort_loading=_f(0.0, 0.0),
            init_pols_if=_f(1.0, 1.0),
        )
        config = make_config()
        param_tables = make_param_tables()
        model = ULPModel(config)
        result = model.run(policies, param_tables)
        return result, policies

    def test_two_policy_shape(self, two_policy_run):
        """Output tensors should have B=2."""
        result, _ = two_policy_run
        assert result["part2"]["no_pols_if"].shape[0] == 2

    def test_different_maturity_months(self, two_policy_run):
        """Each policy matures at its own pol_term * 12."""
        result, policies = two_policy_run
        for b in range(2):
            pol_term = int(policies.pol_term[b])
            mat_t = pol_term * 12
            mats = float(result["part2"]["no_mats"][b, mat_t])
            assert mats > 0.0, f"Policy {b} has no maturity at t={mat_t}"

    def test_no_crosscontamination(self, two_policy_run):
        """Policy 0 should have zero maturities at policy 1's maturity month."""
        result, policies = two_policy_run
        mat_t1 = int(policies.pol_term[1]) * 12   # 180
        mat_t0 = int(policies.pol_term[0]) * 12   # 240
        # Policy 0 should not mature at t=180
        assert float(result["part2"]["no_mats"][0, mat_t1]) == pytest.approx(0.0, abs=1e-10)
        # Policy 1 should not have policies in-force after its maturity
        if mat_t1 + 1 < result["part2"]["no_pols_if"].shape[1]:
            assert float(result["part2"]["no_pols_if"][1, mat_t1]) == pytest.approx(0.0, abs=1e-10)

    def test_two_prem_freqs_independent(self, two_policy_run):
        """Monthly-premium policy (b=1) should have non-zero prem at t=2, annual (b=0) zero."""
        result, _ = two_policy_run
        # Policy 0 (annual): no prem at t=2
        assert float(result["part3"]["prem_inc_if"][0, 2]) == pytest.approx(0.0, abs=1e-6)
        # Policy 1 (monthly): prem at t=2
        assert float(result["part3"]["prem_inc_if"][1, 2]) > 0.0
