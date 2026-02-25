"""
Tests for advanced analysis functions in plot_antenna.calculations

Covers:
- Link Budget: FSPL, Friis range, link margin, range-vs-azimuth
- Indoor Propagation: Log-distance PL, ITU P.1238, wall penetration
- Fading: Rayleigh/Rician CDF, fade margin, statistical fading
- MIMO: Combining gain, capacity-vs-SNR, MEG
- Wearable: Body-worn analysis, dense device interference, SAR, WBAN
- Data tables: Protocol presets, environment presets, body positions
"""

import pytest
import numpy as np

from plot_antenna.calculations import (
    # Link Budget
    free_space_path_loss,
    friis_range_estimate,
    min_tx_gain_for_range,
    link_margin,
    range_vs_azimuth,
    # Indoor Propagation
    log_distance_path_loss,
    itu_indoor_path_loss,
    wall_penetration_loss,
    apply_indoor_propagation,
    # Fading
    rayleigh_cdf,
    rician_cdf,
    fade_margin_for_reliability,
    apply_statistical_fading,
    delay_spread_estimate,
    # MIMO
    envelope_correlation_from_patterns,
    combining_gain,
    mimo_capacity_vs_snr,
    mean_effective_gain_mimo,
    # Wearable / Medical
    body_worn_pattern_analysis,
    dense_device_interference,
    sar_exposure_estimate,
    wban_link_budget,
    # Data tables
    PROTOCOL_PRESETS,
    ENVIRONMENT_PRESETS,
    BODY_POSITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_isotropic_pattern(n_theta=37, n_phi=73, gain_dbi=0.0):
    """Create a simple isotropic 2D gain grid for testing."""
    theta = np.linspace(0, 180, n_theta)
    phi = np.linspace(0, 360, n_phi)
    gain_2d = np.full((n_theta, n_phi), gain_dbi)
    return theta, phi, gain_2d


# ---------------------------------------------------------------------------
# 1. Import smoke test
# ---------------------------------------------------------------------------
class TestImports:
    """Verify all new functions are importable."""

    def test_link_budget_imports(self):
        assert callable(free_space_path_loss)
        assert callable(friis_range_estimate)
        assert callable(min_tx_gain_for_range)
        assert callable(link_margin)
        assert callable(range_vs_azimuth)

    def test_indoor_imports(self):
        assert callable(log_distance_path_loss)
        assert callable(itu_indoor_path_loss)
        assert callable(wall_penetration_loss)
        assert callable(apply_indoor_propagation)

    def test_fading_imports(self):
        assert callable(rayleigh_cdf)
        assert callable(rician_cdf)
        assert callable(fade_margin_for_reliability)
        assert callable(apply_statistical_fading)
        assert callable(delay_spread_estimate)

    def test_mimo_imports(self):
        assert callable(envelope_correlation_from_patterns)
        assert callable(combining_gain)
        assert callable(mimo_capacity_vs_snr)
        assert callable(mean_effective_gain_mimo)

    def test_wearable_imports(self):
        assert callable(body_worn_pattern_analysis)
        assert callable(dense_device_interference)
        assert callable(sar_exposure_estimate)
        assert callable(wban_link_budget)

    def test_data_tables(self):
        assert isinstance(PROTOCOL_PRESETS, dict)
        assert isinstance(ENVIRONMENT_PRESETS, dict)
        assert isinstance(BODY_POSITIONS, dict)
        assert "BLE 1Mbps" in PROTOCOL_PRESETS
        assert "Office" in ENVIRONMENT_PRESETS
        assert "wrist" in BODY_POSITIONS


# ---------------------------------------------------------------------------
# 2. Link Budget
# ---------------------------------------------------------------------------
class TestLinkBudget:
    """Validate link budget / Friis calculations."""

    def test_fspl_2450mhz_1m(self):
        """FSPL at 2450 MHz, 1m should be ~40.2 dB."""
        fspl = free_space_path_loss(2450.0, 1.0)
        assert 39.5 < fspl < 41.0

    def test_fspl_increases_with_distance(self):
        fspl_1m = free_space_path_loss(2450.0, 1.0)
        fspl_10m = free_space_path_loss(2450.0, 10.0)
        assert fspl_10m > fspl_1m
        # 10x distance -> +20 dB in free space
        assert abs((fspl_10m - fspl_1m) - 20.0) < 0.5

    def test_friis_range_positive(self):
        """Friis range should return a positive finite distance."""
        d = friis_range_estimate(
            pt_dbm=0.0, pr_dbm=-98.0, gt_dbi=-10.0, gr_dbi=-10.0,
            freq_mhz=2450.0, path_loss_exp=4.0, misc_loss_db=0.0,
        )
        assert d > 0 and np.isfinite(d)

    def test_friis_range_free_space_longer(self):
        """Free-space (n=2) should give longer range than n=4."""
        d_fs = friis_range_estimate(
            pt_dbm=0.0, pr_dbm=-98.0, gt_dbi=0.0, gr_dbi=0.0,
            freq_mhz=2450.0, path_loss_exp=2.0,
        )
        d_indoor = friis_range_estimate(
            pt_dbm=0.0, pr_dbm=-98.0, gt_dbi=0.0, gr_dbi=0.0,
            freq_mhz=2450.0, path_loss_exp=4.0,
        )
        assert d_fs > d_indoor

    def test_friis_more_power_more_range(self):
        """Higher Tx power should yield longer range."""
        d_low = friis_range_estimate(0.0, -98.0, 0.0, 0.0, 2450.0)
        d_high = friis_range_estimate(20.0, -98.0, 0.0, 0.0, 2450.0)
        assert d_high > d_low

    def test_link_margin_positive_close(self):
        """At 1m with decent gain, margin should be positive."""
        margin = link_margin(
            pt_dbm=0.0, gt_dbi=0.0, gr_dbi=0.0, freq_mhz=2450.0,
            distance_m=1.0, path_loss_exp=2.0, misc_loss_db=0.0,
            pr_sensitivity_dbm=-98.0,
        )
        assert margin > 0

    def test_link_margin_decreases_with_distance(self):
        m_1m = link_margin(0.0, 0.0, 0.0, 2450.0, 1.0)
        m_100m = link_margin(0.0, 0.0, 0.0, 2450.0, 100.0)
        assert m_1m > m_100m

    def test_min_tx_gain_for_range_finite(self):
        """Required Gt for 5m target should be finite."""
        gt = min_tx_gain_for_range(
            target_range_m=5.0, pt_dbm=0.0, pr_dbm=-98.0, gr_dbi=0.0,
            freq_mhz=2450.0, path_loss_exp=2.0,
        )
        assert np.isfinite(gt)

    def test_range_vs_azimuth_returns_arrays(self):
        """range_vs_azimuth returns (range_m, horizon_gain) arrays."""
        theta, phi, gain_2d = _make_isotropic_pattern()
        range_m, horizon_gain = range_vs_azimuth(
            gain_2d, theta, phi, 2450.0, 0.0, -98.0, 0.0,
        )
        assert len(range_m) == len(phi)
        assert len(horizon_gain) == len(phi)

    def test_range_vs_azimuth_positive_ranges(self):
        """With reasonable gain, ranges should be positive."""
        theta, phi, gain_2d = _make_isotropic_pattern(gain_dbi=5.0)
        range_m, _ = range_vs_azimuth(
            gain_2d, theta, phi, 2450.0, 0.0, -98.0, 0.0,
        )
        assert np.all(range_m >= 0)


# ---------------------------------------------------------------------------
# 3. Indoor Propagation
# ---------------------------------------------------------------------------
class TestIndoorPropagation:
    """Validate indoor propagation models."""

    def test_log_distance_pl_increases(self):
        """Path loss should increase with distance."""
        pl_1 = log_distance_path_loss(2450.0, 1.0, n=3.0)
        pl_10 = log_distance_path_loss(2450.0, 10.0, n=3.0)
        assert pl_10 > pl_1

    def test_log_distance_pl_exponent_effect(self):
        """Higher exponent -> more loss at same distance."""
        pl_n2 = log_distance_path_loss(2450.0, 10.0, n=2.0)
        pl_n4 = log_distance_path_loss(2450.0, 10.0, n=4.0)
        assert pl_n4 > pl_n2

    def test_itu_indoor_office(self):
        """ITU P.1238 office model should give reasonable loss."""
        pl = itu_indoor_path_loss(2450.0, 10.0, environment="office")
        assert 30 < pl < 100

    def test_wall_penetration_loss_materials(self):
        """Concrete should attenuate more than drywall."""
        loss_drywall = wall_penetration_loss(2450.0, "drywall")
        loss_concrete = wall_penetration_loss(2450.0, "concrete")
        loss_metal = wall_penetration_loss(2450.0, "metal")
        assert loss_concrete > loss_drywall
        assert loss_metal > loss_concrete

    def test_apply_indoor_propagation_shape(self):
        """Output shape should match input pattern shape."""
        theta, phi, gain_2d = _make_isotropic_pattern()
        # Signature: (gain_2d, theta_deg, phi_deg, freq_mhz, pt_dbm, distance_m, ...)
        pr_2d, path_loss_total = apply_indoor_propagation(
            gain_2d, theta, phi, 2450.0, 0.0, 5.0,
        )
        assert pr_2d.shape == gain_2d.shape
        assert path_loss_total > 0


# ---------------------------------------------------------------------------
# 4. Fading
# ---------------------------------------------------------------------------
class TestFading:
    """Validate fading CDF and fade margin calculations."""

    def test_rayleigh_cdf_monotonic(self):
        """CDF should be monotonically non-decreasing."""
        powers = np.linspace(-30, 10, 50)
        cdf_vals = [rayleigh_cdf(p) for p in powers]
        for i in range(1, len(cdf_vals)):
            assert cdf_vals[i] >= cdf_vals[i - 1] - 1e-10

    def test_rayleigh_cdf_bounds(self):
        """CDF should be between 0 and 1."""
        assert 0 <= rayleigh_cdf(-40.0) <= 1
        assert 0 <= rayleigh_cdf(20.0) <= 1

    def test_rician_cdf_callable(self):
        """Rician CDF should return a value between 0 and 1."""
        val = rician_cdf(0.0, mean_power_db=0.0, K_factor=10.0)
        assert 0 <= val <= 1

    def test_rician_cdf_k0_matches_rayleigh(self):
        """Rician with K=0 should approximate Rayleigh."""
        p = 0.0
        # K_factor=0 -> falls back to rayleigh_cdf
        cdf_ric = rician_cdf(p, K_factor=0.0)
        cdf_ray = rayleigh_cdf(p)
        assert abs(cdf_ric - cdf_ray) < 0.01

    def test_fade_margin_rayleigh_99pct(self):
        """99% Rayleigh fade margin should be ~20 dB."""
        fm = fade_margin_for_reliability(99.0, fading="rayleigh")
        assert 15 < fm < 25, f"Expected ~20 dB, got {fm:.1f} dB"

    def test_fade_margin_increases_with_reliability(self):
        """Higher reliability -> higher fade margin."""
        fm_90 = fade_margin_for_reliability(90.0, fading="rayleigh")
        fm_99 = fade_margin_for_reliability(99.0, fading="rayleigh")
        assert fm_99 > fm_90

    def test_apply_statistical_fading_output(self):
        """Monte-Carlo fading should return 3 arrays (mean, std, outage)."""
        theta, phi, gain_2d = _make_isotropic_pattern(n_theta=19, n_phi=37)
        mean_db, std_db, outage_5pct_db = apply_statistical_fading(
            gain_2d, theta, phi, fading="rayleigh", realizations=100,
        )
        assert mean_db.shape == gain_2d.shape
        assert std_db.shape == gain_2d.shape
        assert outage_5pct_db.shape == gain_2d.shape

    def test_apply_statistical_fading_rician_output(self):
        """Rician fading path should also return correctly-shaped outputs."""
        theta, phi, gain_2d = _make_isotropic_pattern(n_theta=19, n_phi=37)
        mean_db, std_db, outage_5pct_db = apply_statistical_fading(
            gain_2d, theta, phi, fading="rician", K=6.0, realizations=120,
        )
        assert mean_db.shape == gain_2d.shape
        assert std_db.shape == gain_2d.shape
        assert outage_5pct_db.shape == gain_2d.shape
        assert np.all(np.isfinite(mean_db))
        assert np.all(np.isfinite(std_db))
        assert np.all(np.isfinite(outage_5pct_db))

    def test_delay_spread_positive(self):
        """Delay spread should be positive for any environment."""
        ds = delay_spread_estimate(10.0, "indoor")
        assert ds > 0


# ---------------------------------------------------------------------------
# 5. MIMO
# ---------------------------------------------------------------------------
class TestMIMO:
    """Validate MIMO / diversity calculations."""

    def test_combining_gain_mrc(self):
        """MRC of two equal-power branches -> ~3 dB gain."""
        gains = np.array([0.0, 0.0])  # two 0 dBi branches
        combined, improvement = combining_gain(gains, method="mrc")
        assert improvement > 2.0  # MRC should give ~3 dB

    def test_combining_gain_sc(self):
        """Selection combining picks the best branch."""
        gains = np.array([5.0, 0.0, -3.0])
        combined, improvement = combining_gain(gains, method="sc")
        assert combined == 5.0

    def test_mimo_capacity_vs_snr_shape(self):
        """Capacity curve returns (snr, siso_cap, awgn_cap, fading_cap)."""
        snr_axis, siso_cap, awgn_cap, fading_cap = mimo_capacity_vs_snr(
            ecc=0.1, snr_range_db=(-5, 25), num_points=31,
        )
        assert len(snr_axis) == 31
        assert len(siso_cap) == 31
        assert len(awgn_cap) == 31
        assert len(fading_cap) == 31
        # Capacity should generally increase with SNR
        assert siso_cap[-1] > siso_cap[0]

    def test_mimo_capacity_low_ecc_better(self):
        """Lower ECC (better isolation) -> higher fading capacity."""
        _, _, _, fading_low = mimo_capacity_vs_snr(
            ecc=0.1, snr_range_db=(20, 20), num_points=1,
        )
        _, _, _, fading_high = mimo_capacity_vs_snr(
            ecc=0.9, snr_range_db=(20, 20), num_points=1,
        )
        assert fading_low[0] >= fading_high[0]

    def test_mean_effective_gain_shape(self):
        """MEG should return one value per antenna element."""
        theta, phi, g1 = _make_isotropic_pattern()
        _, _, g2 = _make_isotropic_pattern(gain_dbi=-3.0)
        meg = mean_effective_gain_mimo([g1, g2], theta, phi, xpr_db=6.0)
        assert len(meg) == 2


# ---------------------------------------------------------------------------
# 6. Wearable / Medical
# ---------------------------------------------------------------------------
class TestWearable:
    """Validate wearable / medical device functions."""

    def test_body_worn_returns_dict(self):
        """Body-worn analysis should return per-position results."""
        theta, phi, gain_2d = _make_isotropic_pattern()
        result = body_worn_pattern_analysis(
            gain_2d, theta, phi, 2450.0,
            body_positions=["wrist", "chest"],
        )
        assert isinstance(result, dict)
        assert "wrist" in result
        assert "chest" in result

    def test_body_worn_shows_degradation(self):
        """Body shadowing should reduce peak gain."""
        theta, phi, gain_2d = _make_isotropic_pattern(gain_dbi=5.0)
        result = body_worn_pattern_analysis(
            gain_2d, theta, phi, 2450.0, body_positions=["chest"],
        )
        # The shadowed pattern should have lower peak than original
        shadowed_peak = np.max(result["chest"]["pattern"])
        original_peak = np.max(gain_2d)
        assert shadowed_peak <= original_peak + 0.1  # small tolerance

    def test_dense_device_interference_returns_tuple(self):
        """dense_device_interference returns (avg_sinr, sinr_dist, noise_floor)."""
        avg_sinr, sinr_dist, noise_floor = dense_device_interference(
            num_devices=20, tx_power_dbm=0.0, freq_mhz=2450.0,
        )
        assert np.isfinite(avg_sinr)
        assert len(sinr_dist) > 0
        assert noise_floor < 0  # noise floor is negative dBm

    def test_sar_exposure_returns_tuple(self):
        """SAR returns (sar, fcc_limit, icnirp_limit, compliant)."""
        sar, fcc_limit, icnirp_limit, compliant = sar_exposure_estimate(
            tx_power_mw=1.0, antenna_gain_dbi=0.0,
            distance_cm=1.0, freq_mhz=2450.0,
        )
        assert sar > 0
        assert fcc_limit == 1.6
        assert icnirp_limit == 2.0
        assert isinstance(compliant, bool)

    def test_sar_low_power_compliant(self):
        """Very low power at distance should be compliant."""
        sar, _, _, compliant = sar_exposure_estimate(
            tx_power_mw=0.01, antenna_gain_dbi=-10.0,
            distance_cm=10.0, freq_mhz=2450.0,
        )
        assert compliant is True

    def test_wban_link_budget_returns_tuple(self):
        """WBAN returns (path_loss_db, received_power_dbm)."""
        path_loss, rx_power = wban_link_budget(
            tx_power_dbm=0.0, freq_mhz=2450.0,
            body_channel="on_body", distance_cm=30,
        )
        assert path_loss > 0
        assert rx_power < 0  # received power should be negative dBm


# ---------------------------------------------------------------------------
# 7. Protocol & Environment Presets
# ---------------------------------------------------------------------------
class TestPresets:
    """Validate preset data tables."""

    def test_protocol_presets_have_tuples(self):
        """Each protocol preset should be a 3-tuple (rx_sens, tx_power, freq)."""
        for name, vals in PROTOCOL_PRESETS.items():
            assert len(vals) == 3, f"Preset '{name}' should have 3 values"

    def test_protocol_ble_1mbps(self):
        rx_sens, tx_pwr, freq = PROTOCOL_PRESETS["BLE 1Mbps"]
        assert rx_sens == -98.0
        assert tx_pwr == 0.0
        assert freq == 2450.0

    def test_environment_presets_have_tuples(self):
        """Each environment preset should be a 5-tuple."""
        for name, vals in ENVIRONMENT_PRESETS.items():
            assert len(vals) == 5, f"Environment '{name}' should have 5 values"

    def test_environment_office(self):
        n, sigma, fading_m, k, walls = ENVIRONMENT_PRESETS["Office"]
        assert n == 3.0
        assert sigma == 5.0
        assert fading_m == "rician"

    def test_body_positions_have_required_keys(self):
        """Each body position should have axis, cone_deg, tissue_cm."""
        for pos, data in BODY_POSITIONS.items():
            assert "axis" in data, f"Position '{pos}' missing 'axis'"
            assert "cone_deg" in data, f"Position '{pos}' missing 'cone_deg'"
            assert "tissue_cm" in data, f"Position '{pos}' missing 'tissue_cm'"
