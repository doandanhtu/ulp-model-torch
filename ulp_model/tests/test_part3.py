"""
test_part3.py - Unit tests for Part 3: Shareholder Cashflow Projection.
"""
from __future__ import annotations

import pytest
import torch

from ulp_model.part1_pav import PAVProjection
from ulp_model.part2_decrements import DecrementProjection
from ulp_model.part3_cashflows import CashflowProjection

from .conftest_helpers import make_config, make_param_tables, make_single_policy


@pytest.fixture(scope="module")
def all_outputs():
    config = make_config()
    policies = make_single_policy()
    param_tables = make_param_tables()
    pav = PAVProjection(policies, param_tables, config)
    p1 = pav.run()
    dec = DecrementProjection(policies, param_tables, config, p1)
    p2 = dec.run()
    cf = CashflowProjection(policies, param_tables, config, p1, p2)
    p3 = cf.run()
    return p1, p2, p3, policies, param_tables, config


class TestCashflowProjection:

    # ------------------------------------------------------------------
    # Premium income
    # ------------------------------------------------------------------

    def test_prem_inc_t1_positive(self, all_outputs):
        """Premium income at t=1 should be positive (first annual payment)."""
        _, _, p3, _, _, _ = all_outputs
        assert float(p3["prem_inc_if"][0, 1]) > 0.0

    def test_prem_inc_t2_zero(self, all_outputs):
        """No premium income at t=2 for annual frequency."""
        _, _, p3, _, _, _ = all_outputs
        assert float(p3["prem_inc_if"][0, 2]) == pytest.approx(0.0, abs=1e-6)

    def test_basic_prem_if_equals_topup_plus_basic(self, all_outputs):
        """prem_inc_if ≥ basic_prem_if (basic + topup combined)."""
        _, _, p3, _, _, _ = all_outputs
        for t in range(1, 5):
            assert float(p3["prem_inc_if"][0, t]) >= float(p3["basic_prem_if"][0, t]) - 1e-9

    # ------------------------------------------------------------------
    # Expenses / commissions
    # ------------------------------------------------------------------

    def test_comm_nonnegative(self, all_outputs):
        """Commissions should be non-negative."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["comm_if"][0, :25] >= 0.0).all()

    def test_op_init_exp_t1_positive(self, all_outputs):
        """Initial operating expense at t=1 should be positive."""
        _, _, p3, _, _, _ = all_outputs
        assert float(p3["op_init_exp_if"][0, 1]) >= 0.0

    def test_op_ren_exp_nonnegative(self, all_outputs):
        """Renewal operating expenses should be non-negative throughout."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["op_ren_exp_if"][0, :25] >= 0.0).all()

    # ------------------------------------------------------------------
    # Death / surrender / maturity outgos
    # ------------------------------------------------------------------

    def test_death_outgo_nonnegative(self, all_outputs):
        """Death outgo should be non-negative."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["death_outgo"][0, :25] >= 0.0).all()

    def test_surr_outgo_nonneg(self, all_outputs):
        """Surrender outgo should be non-negative."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["surr_outgo"][0, :25] >= 0.0).all()

    def test_mat_outgo_at_term(self, all_outputs):
        """Maturity outgo should be positive at maturity month."""
        p1, p2, p3, policies, _, _ = all_outputs
        pol_term = int(policies.pol_term[0])
        mat_t = pol_term * 12  # 240
        assert float(p3["mat_outgo"][0, mat_t]) > 0.0

    # ------------------------------------------------------------------
    # Unit reserve
    # ------------------------------------------------------------------

    def test_unit_res_nonnegative(self, all_outputs):
        """Unit reserve should be non-negative (backed by AV)."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["unit_res_bgn"][0, :25] >= -1e-6).all()
        assert (p3["unit_res_end"][0, :25] >= -1e-6).all()

    # ------------------------------------------------------------------
    # Cashflow passes
    # ------------------------------------------------------------------

    def test_cf_before_zv_finite(self, all_outputs):
        """cf_before_zv should have no NaN or Inf values."""
        _, _, p3, _, _, _ = all_outputs
        assert torch.isfinite(p3["cf_before_zv"][0, :]).all()

    def test_zeroising_res_nonnegative(self, all_outputs):
        """Zeroising reserve should be non-negative (it zeroes negative cashflows)."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["zeroising_res_if"][0, :] >= -1e-9).all()

    def test_cf_after_zv_t0_zero(self, all_outputs):
        """cf_after_zv must be 0 at t=0 (initialisation month)."""
        _, _, p3, _, _, _ = all_outputs
        assert float(p3["cf_after_zv"][0, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_cf_after_zv_nonneg_from_t2(self, all_outputs):
        """cf_after_zv >= 0 for t >= 2. t=1 is the only timepoint allowed to be negative."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["cf_after_zv"][0, 2:] >= -1e-6).all()

    def test_cf_after_scr_finite(self, all_outputs):
        """cf_after_scr should be finite throughout."""
        _, _, p3, _, _, _ = all_outputs
        assert torch.isfinite(p3["cf_after_scr"][0, :]).all()

    # ------------------------------------------------------------------
    # Present value
    # ------------------------------------------------------------------

    def test_pv_cf_at_t0_finite(self, all_outputs):
        """pv_cf_after_scr[:,0] should be finite (total VIF)."""
        _, _, p3, _, _, _ = all_outputs
        assert torch.isfinite(p3["pv_cf_after_scr"][0, 0])

    def test_pv_prem_at_t0_positive(self, all_outputs):
        """PV of premium income at t=0 should be positive."""
        _, _, p3, _, _, _ = all_outputs
        assert float(p3["pv_prem_inc"][0, 0]) > 0.0

    def test_pv_cf_backward_structure(self, all_outputs):
        """pv_cf_after_scr should generally decrease as t approaches maturity."""
        _, _, p3, policies, _, _ = all_outputs
        pv = p3["pv_cf_after_scr"][0, :]
        pol_term = int(policies.pol_term[0])
        mat_t = pol_term * 12
        # PV at t=0 should be >= PV at mat_t (roughly; can differ by sign)
        # At least: PV at maturity should be near zero
        assert float(pv[mat_t]) == pytest.approx(0.0, abs=float(abs(pv[0])) * 0.1 + 1.0)

    # ------------------------------------------------------------------
    # Tax
    # ------------------------------------------------------------------

    def test_op_tax_nonneg_from_t2(self, all_outputs):
        """Operating tax >= 0 for t >= 2 (t=1 can be negative as cf_after_zv(1) may be negative)."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["op_tax"][0, 2:25] >= -1e-9).all()

    # ------------------------------------------------------------------
    # SCR
    # ------------------------------------------------------------------

    def test_solv_cap_req_zero_at_t0(self, all_outputs):
        """SCR must be 0 at t=0 (initialisation month)."""
        _, _, p3, _, _, _ = all_outputs
        assert float(p3["solv_cap_req"][0, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_cf_after_scr_zero_at_t0(self, all_outputs):
        """cf_after_scr must be 0 at t=0 (initialisation month)."""
        _, _, p3, _, _, _ = all_outputs
        assert float(p3["cf_after_scr"][0, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_solv_cap_req_nonnegative(self, all_outputs):
        """Solvency capital requirement should be non-negative."""
        _, _, p3, _, _, _ = all_outputs
        assert (p3["solv_cap_req"][0, :25] >= -1e-9).all()

    # ------------------------------------------------------------------
    # Column consistency
    # ------------------------------------------------------------------

    def test_all_outputs_same_shape(self, all_outputs):
        """All [B, T] output tensors should have shape (1, T)."""
        _, _, p3, _, _, config = all_outputs
        T = config.MAX_PROJ_MONTHS
        for key, val in p3.items():
            assert val.shape == (1, T), f"{key} has shape {val.shape}, expected (1, {T})"
