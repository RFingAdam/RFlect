"""
Pytest configuration and shared fixtures for RFlect tests
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data_dir():
    """Return path to test fixtures directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_passive_data():
    """Return mock passive antenna measurement data

    Returns:
        dict: Dictionary containing phi, theta angles and gain measurements
    """
    return {
        'phi': np.linspace(0, 360, 37),  # 10-degree steps
        'theta': np.linspace(0, 180, 19),  # 10-degree steps
        'h_gain': np.random.randn(19, 37) * 5 + 10,  # HPOL gain (dBi)
        'v_gain': np.random.randn(19, 37) * 5 + 10,  # VPOL gain (dBi)
        'frequency': 2400.0  # MHz
    }


@pytest.fixture
def sample_active_data():
    """Return mock active TRP measurement data

    Returns:
        dict: Dictionary containing phi, theta angles and power measurements
    """
    return {
        'phi': np.linspace(0, 360, 37),
        'theta': np.linspace(0, 180, 19),
        'power_dbm': np.random.randn(19, 37) * 3 + 5,  # Power in dBm
        'frequency': 2400.0  # MHz
    }


@pytest.fixture
def sample_frequencies():
    """Return list of typical measurement frequencies"""
    return [2400.0, 2450.0, 2500.0]  # MHz


@pytest.fixture
def mock_vna_data():
    """Return mock VNA S-parameter data

    Returns:
        dict: Dictionary containing frequency and S-parameter measurements
    """
    freqs = np.linspace(2000, 3000, 101)  # 2-3 GHz, 101 points
    return {
        'frequency_mhz': freqs,
        's11_db': -15 - 10 * np.random.rand(101),  # S11 magnitude in dB
        's11_phase': np.random.randn(101) * 50,  # Phase in degrees
        'vswr': 1.5 + 0.5 * np.random.rand(101)  # VSWR
    }
