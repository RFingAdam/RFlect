# Test Fixtures

This directory contains sample data files for testing RFlect functionality.

## Files

Sample data files will be added here for:
- Passive measurements (HPOL/VPOL .txt files)
- Active measurements (TRP .txt files)
- VNA measurements (.csv files)
- CST simulation exports

## Usage

Test fixtures are loaded via the `sample_data_dir` fixture in conftest.py:

```python
def test_file_reading(sample_data_dir):
    file_path = sample_data_dir / "sample_hpol.txt"
    data = read_passive_file(file_path)
    assert data is not None
```
