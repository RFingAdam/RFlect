"""
Smoke tests for plot_antenna.file_utils module

These are basic smoke tests to verify imports and basic functionality.
Comprehensive file parsing tests to be added in future iterations.

TODO for v4.1:
- Add tests with actual sample data files in fixtures/
- Test all supported file formats (Howland, CST, VNA)
- Test error handling for malformed files
- Test edge cases (empty files, unicode paths, etc.)
"""

import pytest
from pathlib import Path

# Verify imports work
from plot_antenna.file_utils import (
    read_passive_file,
    read_active_file,
    parse_2port_data,
)


class TestImports:
    """Test that all file_utils functions can be imported"""

    def test_imports_successful(self):
        """Verify all file parsing functions import without errors"""
        assert callable(read_passive_file)
        assert callable(read_active_file)
        assert callable(parse_2port_data)


class TestErrorHandling:
    """Test error handling for invalid inputs"""

    def test_nonexistent_passive_file(self):
        """Test that nonexistent passive file raises appropriate error"""
        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            read_passive_file("nonexistent_file_12345.txt")

    def test_nonexistent_active_file(self):
        """Test that nonexistent active file raises appropriate error"""
        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            read_active_file("nonexistent_file_12345.txt")

    def test_nonexistent_vna_file(self):
        """Test that nonexistent VNA file raises appropriate error"""
        with pytest.raises((FileNotFoundError, IOError, ValueError)):
            parse_2port_data("nonexistent_file_12345.csv")


# Mark comprehensive tests as TODO for future implementation
@pytest.mark.skip(reason="TODO: Add sample data files and implement comprehensive tests")
class TestPassiveFileParsingTODO:
    """Placeholder for comprehensive passive file parsing tests"""

    pass


@pytest.mark.skip(reason="TODO: Add sample data files and implement comprehensive tests")
class TestActiveFileParsingTODO:
    """Placeholder for comprehensive active file parsing tests"""

    pass


@pytest.mark.skip(reason="TODO: Add sample data files and implement comprehensive tests")
class TestVNAFileParsingTODO:
    """Placeholder for comprehensive VNA file parsing tests"""

    pass
