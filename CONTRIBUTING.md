# Contributing to RFlect

Thank you for your interest in contributing to RFlect! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

Be respectful, constructive, and professional in all interactions. We're here to build great RF measurement tools together.

---

## Getting Started

### Prerequisites
- Python 3.11 or 3.12
- Git
- Windows, macOS, or Linux

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/RFlect.git
cd RFlect
git remote add upstream https://github.com/RFingAdam/RFlect.git
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Verify Installation

```bash
# Run tests to verify setup
pytest

# Run the application
python -m plot_antenna.main
```

---

## Code Style

### Python Style Guide

RFlect follows these coding standards:

- **Line Length**: 100 characters maximum
- **Formatter**: Black (configured in pyproject.toml)
- **Linter**: Flake8 with complexity checking
- **Type Hints**: Encouraged but not required
- **Docstrings**: Google style for classes and public methods

### Formatting Code

```bash
# Format code with Black
black plot_antenna/

# Check formatting without changes
black --check plot_antenna/

# Run linter
flake8 plot_antenna --max-line-length=100
```

### Import Order

```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from plot_antenna.calculations import calculate_passive_variables
from .utils import resource_path
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`

### Example Code

```python
"""
Module docstring explaining purpose.
"""

import numpy as np
from typing import Dict, List, Optional


class AntennaAnalyzer:
    """
    Short description of class.

    Longer description with usage example:
        >>> analyzer = AntennaAnalyzer(data, "passive", [2400])
        >>> stats = analyzer.get_gain_statistics()

    Args:
        measurement_data: Dictionary with measurement arrays
        scan_type: One of 'passive', 'active', 'vna'
    """

    def __init__(self, measurement_data: Dict, scan_type: str):
        self.data = measurement_data
        self.scan_type = scan_type

    def get_gain_statistics(self, frequency: Optional[float] = None) -> Dict:
        """
        Calculate gain statistics.

        Args:
            frequency: Target frequency in MHz. Defaults to first frequency.

        Returns:
            Dictionary with keys: max_gain_dBi, min_gain_dBi, avg_gain_dBi

        Raises:
            ValueError: If no measurement data available
        """
        if not self.data:
            raise ValueError("No measurement data loaded")

        # Implementation
        return {"max_gain_dBi": 10.5}
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=plot_antenna --cov-report=html

# Run specific test file
pytest tests/test_calculations.py

# Run specific test
pytest tests/test_calculations.py::TestPassiveCalculations::test_efficiency_calculation
```

### Writing Tests

Place tests in `tests/` directory with `test_` prefix:

```python
# tests/test_my_feature.py
import pytest
from plot_antenna.my_module import my_function


def test_my_function_basic():
    """Test basic functionality of my_function"""
    result = my_function(input_data)
    assert result == expected_output


def test_my_function_edge_case():
    """Test edge case handling"""
    with pytest.raises(ValueError):
        my_function(invalid_input)


@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return {"phi": [0, 90, 180], "gain": [5, 10, 5]}
```

### Test Coverage Goals

- **Current**: 227 tests passing, 22% overall coverage
- **Target Overall**: ≥60% coverage
- **Core modules** (calculations, file_utils): ≥80% coverage
- **GUI modules**: Best effort (GUI testing is harder)

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-123
```

### 2. Make Changes

- Write clear, focused commits
- Follow code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Message Format

```
type: Short description (50 chars or less)

Longer description if needed, explaining:
- What changed
- Why it changed
- Any breaking changes

Closes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance (dependencies, build, etc.)

**Example**:
```
feat: Add batch frequency analysis function

Implements analyze_all_frequencies() for AntennaAnalyzer class
to provide gain trends across all measured frequencies.

- Calculates 3dB bandwidth
- Detects resonance frequency
- Reports frequency stability metrics

Closes #45
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Include:
# - Clear description of changes
# - Link to related issues
# - Screenshots/examples if UI changes
# - Test results
```

### 5. PR Review Process

- Automated tests must pass (GitHub Actions CI)
- Code review by maintainer
- Address review feedback
- Squash commits if requested
- Maintainer merges when approved

---

## Release Process

### Version Numbering

RFlect follows [Semantic Versioning](https://semver.org/):
- **Major** (v4.0.0): Breaking changes, major refactoring
- **Minor** (v4.1.0): New features, backward compatible
- **Patch** (v4.0.1): Bug fixes, backward compatible

### Automated Version Bumping

```bash
# Install bump2version
pip install bump2version

# Bump version (updates all files automatically)
bump2version major  # 4.0.0 → 5.0.0
bump2version minor  # 4.0.0 → 4.1.0
bump2version patch  # 4.0.0 → 4.0.1

# Creates commit and git tag automatically
```

### Release Checklist (Maintainers Only)

1. **Update RELEASE_NOTES.md** with changes
2. **Run full test suite**: `pytest --cov=plot_antenna`
3. **Bump version**: `bump2version minor`
4. **Push with tags**: `git push && git push --tags`
5. **GitHub Actions** builds .exe automatically
6. **Verify release** on GitHub with correct artifacts
7. **Announce release** in discussions/README

---

## Project Structure

```
RFlect/
├── plot_antenna/           # Main package
│   ├── main.py            # Entry point
│   ├── calculations.py    # RF calculations
│   ├── file_utils.py      # File parsing
│   ├── plotting.py        # Visualization
│   ├── save.py            # Report generation
│   ├── ai_analysis.py     # AI analysis logic
│   ├── config.py          # Configuration
│   ├── gui/               # GUI components
│   │   ├── main_window.py
│   │   ├── dialogs_mixin.py
│   │   ├── ai_chat_mixin.py
│   │   ├── tools_mixin.py
│   │   └── callbacks_mixin.py
│   └── __init__.py
├── rflect-mcp/            # MCP server for programmatic access
│   ├── server.py          # FastMCP server entry point
│   ├── tools/
│   │   ├── import_tools.py    # File import tools
│   │   ├── analysis_tools.py  # Analysis tools
│   │   ├── report_tools.py    # Report generation tools
│   │   └── bulk_tools.py      # Batch processing tools
│   ├── templates/
│   │   └── default.yaml       # Report template
│   └── README.md
├── tests/                  # Test suite (227 tests)
│   ├── conftest.py        # Pytest fixtures
│   ├── test_calculations.py
│   ├── test_ai_analysis.py
│   ├── test_file_utils.py
│   ├── test_mcp_tools.py
│   ├── test_mcp_integration.py  # 66 MCP integration tests (all 20 tools)
│   ├── test_real_data_integration.py  # Real BLE/LoRa chamber data tests
│   └── integration/       # Integration tests
├── .github/
│   └── workflows/         # CI/CD workflows
├── assets/                # Images, logos
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml         # Package configuration
├── .bumpversion.cfg       # Version management
└── README.md
```

---

## Areas Needing Contribution

### Completed (v4.0.0)
- ~~HPBW and F/B ratio in pattern analysis~~ (implemented and verified with boundary wrapping fix)
- ~~Test coverage expansion~~ (227 tests achieved, up from 82; 22% overall coverage)

### High Priority
- 🔴 Increase test coverage toward 60% target (currently 22% with 227 tests)
- 🔴 Sidelobe detection and reporting in pattern analysis
- 🔴 Automated figure insertion in DOCX reports
- 🔴 System Fidelity Factor calculation (#31)

### Medium Priority
- 🟡 Add support for additional file formats
- 🟡 Improve error messages and user feedback
- 🟡 Add more antenna benchmarks to AI knowledge base
- 🟡 Create tutorial documentation
- 🟡 Multi-frequency comparison tables in reports

### Future Features
- 🟢 Vision API integration for plot analysis (v4.4+)
- 🟢 MIMO antenna analysis
- 🟢 macOS/Linux .app/.deb packaging

---

## Getting Help

- **Issues**: https://github.com/RFingAdam/RFlect/issues
- **Discussions**: https://github.com/RFingAdam/RFlect/discussions
- **AI Features**: See [AI_STATUS.md](AI_STATUS.md)

---

## License

By contributing to RFlect, you agree that your contributions will be licensed under the GPL-3.0 License.
