"""
test_part2.py - Unit tests for Part 2: Decrement Projection.
"""
from __future__ import annotations

import pytest
import torch

from ulp_model.part1_pav import PAVProjection
from ulp_model.part2_decrements import DecrementProjection

from .conftest_helpers import make_config, make_param_tables, make_single_policy


@pytest.fixture(scope="module")
def part2_outputs():
    config = make_config()
    policies = make_single_policy()
    param_tables = make_param_tables()
    pav = PAVProjection(policies, param_tables, config)
    p1 = pav.run()
    dec = DecrementProjection(policies, param_tables, config, p1)
    p2 = dec.run()
    return p2, policies, param_tables, config


class TestDecrementProjection:

    def test_initial_pols_if(self, part2_outputs):
        """no_pols_if at t=0 should equal init_pols_if (1.0)."""
        p2, policies, _, _ = part2_outputs
        assert float(p2["no_pols_if"][0, 0]) == pytest.approx(1.0)

    def test_no_pols_if_t0_equals_init(self, part2_outputs):
        """Shorthand: init_pols_if propagates to no_pols_if[:,0]."""
        p2, policies, _, _ = part2_outputs
        assert float(p2["no_pols_if"][0, 0]) == pytest.approx(float(policies.init_pols_if[0]))

    def test_no_pols_ifsm_t1(self, part2_outputs):
        """no_pols_ifsm[:,1] = no_pols_if[:,0] - no_mats[:,0]."""
        p2, _, _, _ = part2_outputs
        expected = float(p2["no_pols_if"][0, 0]) - float(p2["no_mats"][0, 0])
        assert float(p2["no_pols_ifsm"][0, 1]) == pytest.approx(expected, rel=1e-9)

    def test_no_deaths_positive(self, part2_outputs):
        """Death decrements should be positive for an in-force policy."""
        p2, _, _, _ = part2_outputs
        assert float(p2["no_deaths"][0, 1]) > 0.0

    def test_no_deaths_small(self, part2_outputs):
        """Monthly death rate for age 30 male should be very small (< 0.001)."""
        p2, _, _, _ = part2_outputs
        # 1 policy, monthly deaths << 0.001
        assert float(p2["no_deaths"][0, 1]) < 0.001

    def test_no_surrs_nonnegative(self, part2_outputs):
        """Surrender counts should never be negative."""
        p2, _, _, _ = part2_outputs
        T = p2["no_surrs"].shape[1]
        assert (p2["no_surrs"][0, :T] >= 0.0).all()

    def test_no_pols_if_nonnegative(self, part2_outputs):
        """Policies in force should never go negative."""
        p2, _, _, _ = part2_outputs
        assert (p2["no_pols_if"][0, :] >= 0.0).all()

    def test_conservation_each_month(self, part2_outputs):
        """Accounting identity: no_pols_ifsm = max(no_pols_if_prev - no_mats_prev, 0)."""
        p2, policies, _, _ = part2_outputs
        T = p2["no_pols_if"].shape[1]
        mat_t = int(policies.pol_term[0]) * 12
        for t in range(2, min(T, mat_t)):  # only pre-maturity months
            ifsm = float(p2["no_pols_ifsm"][0, t])
            pols_prev = float(p2["no_pols_if"][0, t - 1])
            mats_prev = float(p2["no_mats"][0, t - 1])
            assert ifsm == pytest.approx(max(pols_prev - mats_prev, 0.0), rel=1e-9, abs=1e-12)

    def test_maturity_at_term(self, part2_outputs):
        """Policy matures at t = pol_term * 12 = 240."""
        p2, policies, _, _ = part2_outputs
        pol_term = int(policies.pol_term[0])
        mat_t = pol_term * 12  # 240
        assert float(p2["no_mats"][0, mat_t]) > 0.0

    def test_no_pols_if_zero_after_maturity(self, part2_outputs):
        """After maturity month, no more policies in force."""
        p2, policies, _, _ = part2_outputs
        pol_term = int(policies.pol_term[0])
        mat_t = pol_term * 12
        T = p2["no_pols_if"].shape[1]
        if mat_t + 1 < T:
            assert float(p2["no_pols_if"][0, mat_t]) == pytest.approx(0.0, abs=1e-10)

    def test_no_mats_zero_before_maturity(self, part2_outputs):
        """No maturities should occur before the maturity month."""
        p2, policies, _, _ = part2_outputs
        pol_term = int(policies.pol_term[0])
        mat_t = pol_term * 12
        assert float(p2["no_mats"][0, :mat_t].sum()) == pytest.approx(0.0, abs=1e-10)

    def test_surrs_annual_only_at_prem_due_months(self, part2_outputs):
        """For annual premium, surrender should occur at end of each policy year
        (t=12, 24, ...) and not within the year."""
        p2, policies, _, _ = part2_outputs
        prem_term_months = int(policies.prem_term[0]) * 12  # 240
        # No lapse at start of year 1 (t=1) or mid-year (t=2..11)
        for t in [1, 2, 5, 11]:
            assert float(p2["no_surrs"][0, t]) == pytest.approx(0.0, abs=1e-10)
        # Lapse at end of year 1 (t=12), within prem term
        if 12 <= prem_term_months:
            assert float(p2["no_surrs"][0, 12]) > 0.0

    def test_pols_if_end_of_month_identity(self, part2_outputs):
        """no_pols_if[:,t] = no_pols_ifsm[:,t] - deaths - surrs - mats at t."""
        p2, _, _, _ = part2_outputs
        for t in range(1, 6):
            pols_if = float(p2["no_pols_if"][0, t])
            expected = (
                float(p2["no_pols_ifsm"][0, t])
                - float(p2["no_deaths"][0, t])
                - float(p2["no_surrs"][0, t])
                - float(p2["no_mats"][0, t])
            )
            assert pols_if == pytest.approx(expected, rel=1e-9, abs=1e-12)
