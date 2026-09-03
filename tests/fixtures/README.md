# Test fixtures

Sample files used by the pytest suite. They are **synthetic / reference** data
written for tests, not customer chamber captures.

## What is in this tree

| Path | Kind | Used by |
|------|------|---------|
| `cal_drift/*.txt` | Hand-written TRP-cal and summary files (baseline, shifted, missing-freq, HPOL stub) | `tests/test_cal_drift.py`, `tests/test_cal_drift_v52.py` |

There are no bundled WTL HPOL/VPOL pairs, active TRP spheres, or VNA sweeps here.

## What is not in the public repo

Real antenna-chamber files (if you have them) stay on the machine that measured
them. Point optional integration tests at a local folder:

```bash
export RFLECT_TEST_DATA_DIR=/path/to/your/captures
python -m pytest tests/test_real_data_integration.py tests/test_uwb_real_data.py
```

Those tests skip when the env var is unset (including CI).

## Synthetic in-memory data

Most tests never touch this directory. `tests/conftest.py` builds numpy grids
(`sample_passive_data`, `sample_active_data`, `mock_vna_data`) for unit tests.
Golden-reference TRP tests in `tests/test_golden_reference.py` use analytic
isotropic patterns, not measured files.

```python
def test_file_reading(sample_data_dir):
    cal = sample_data_dir / "cal_drift" / "cal_baseline.txt"
    assert cal.exists()
```
