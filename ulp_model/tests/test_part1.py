"""
test_part1.py - Unit tests for Part 1: PAV Projection.

Test policy: age 30, male, 20-year term, annual premium,
             SA=500M, ACP=20M, escalating DB.
"""
from __future__ import annotations

import math

import pytest
import torch

from ulp_model.part1_pav import PAVProjection

from .conftest_helpers import make_config, make_param_tables, make_single_policy


@pytest.fixture(scope="module")
def part1_outputs():
    config = make_config()
    policies = make_single_policy()
    param_tables = make_param_tables()
    pav = PAVProjection(policies, param_tables, config)
    return pav.run(), policies, param_tables


class TestPAVProjection:
    """Tests for PAV projection logic over 24 months."""

    def test_is_inforce_end_t1(self, part1_outputs):
        """Policy should be in force at end of month 1."""
        outputs, _, _ = part1_outputs
        assert float(outputs["is_inforce_end"][0, 1]) == 1.0

    def test_bav_ab_t1_positive(self, part1_outputs):
        """Basic AV after growth (bav_ab) should be positive at t=1."""
        outputs, _, _ = part1_outputs
        assert float(outputs["bav_ab"][0, 1]) > 0.0

    def test_basic_prem_pp_t1(self, part1_outputs):
        """Annual premium is due at t=1 (first annual payment at (t-1)%12==0 → t=1).

        Expected: ACP = 20M (annual, so per payment = 20M * 12/12 = 20M).
        """
        outputs, policies, _ = part1_outputs
        acp = float(policies.acp[0])  # 20_000_000
        assert float(outputs["basic_prem_pp"][0, 1]) == pytest.approx(acp, rel=1e-6)

    def test_basic_prem_pp_t13(self, part1_outputs):
        """Second annual premium is due at t=13 ((13-1)%12 == 0)."""
        outputs, policies, _ = part1_outputs
        acp = float(policies.acp[0])
        assert float(outputs["basic_prem_pp"][0, 13]) == pytest.approx(acp, rel=1e-6)

    def test_basic_prem_pp_t2_zero(self, part1_outputs):
        """No premium is due at t=2 for annual frequency ((2-1)%12 = 1 ≠ 0)."""
        outputs, _, _ = part1_outputs
        assert float(outputs["basic_prem_pp"][0, 2]) == pytest.approx(0.0, abs=1e-6)

    def test_basic_prem_pp_t0_zero(self, part1_outputs):
        """t=0 is initialisation – no premium should be allocated."""
        outputs, _, _ = part1_outputs
        assert float(outputs["basic_prem_pp"][0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_av_nonnegative_all_months(self, part1_outputs):
        """AV should never be negative (clamped to 0)."""
        outputs, _, _ = part1_outputs
        T = 25  # check first 24 active months
        assert (outputs["av_ab"][0, :T] >= 0).all()

    def test_allocation_charge_t1(self, part1_outputs):
        """At pol_year=1, basic alloc charge should be 50% of basic_prem_pp."""
        outputs, policies, param_tables = part1_outputs
        basic_prem = float(outputs["basic_prem_pp"][0, 1])
        alloc_chg = float(outputs["basic_alloc_chg_pp"][0, 1])
        expected = 0.50 * basic_prem  # 50% in year 1
        assert alloc_chg == pytest.approx(expected, rel=1e-6)

    def test_mort_coi_positive_after_prem(self, part1_outputs):
        """Monthly COI should be positive once AV is funded."""
        outputs, _, _ = part1_outputs
        assert float(outputs["mort_coi"][0, 1]) > 0.0

    def test_db_opt_stays_escalating_young(self, part1_outputs):
        """DB option should remain 2 (escalating) for young policy (age < 65)."""
        outputs, policies, param_tables = part1_outputs
        # Age at entry 30, pol_year 1 → attained age 30, well below age_db_opt_change=65
        assert int(outputs["current_db_opt"][0, 1]) == 2

    def test_inforce_bgn_t1_equals_inforce_end_t0(self, part1_outputs):
        """is_inforce_bgn[:,1] should equal is_inforce_end[:,0] (=1)."""
        outputs, _, _ = part1_outputs
        assert float(outputs["is_inforce_bgn"][0, 1]) == float(outputs["is_inforce_end"][0, 0])

    def test_tot_dedn_act_positive_t1(self, part1_outputs):
        """Total actual deductions should be positive at t=1."""
        outputs, _, _ = part1_outputs
        assert float(outputs["tot_dedn_act"][0, 1]) > 0.0

    def test_g_bav_ab_grows(self, part1_outputs):
        """Guaranteed BAV should accumulate over time with premium additions."""
        outputs, _, _ = part1_outputs
        # After first premium allocation g_bav_ab at t=1 should exceed t=0 (0)
        assert float(outputs["g_bav_ab"][0, 1]) > 0.0

    def test_bav_ab_after_12_months(self, part1_outputs):
        """BAV at t=12 should be in a plausible range after 1 year of growth."""
        outputs, policies, _ = part1_outputs
        acp = float(policies.acp[0])
        bav_12 = float(outputs["bav_ab"][0, 12])
        # After 50% alloc charge on first premium: at least some fraction of ACP * 0.5
        assert bav_12 > acp * 0.1
