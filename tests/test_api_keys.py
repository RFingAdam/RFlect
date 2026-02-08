"""
Comprehensive tests for the api_keys module — v4.1.0

Tests cover:
- Provider registry validation
- Encryption key generation and Fernet compatibility
- Encrypted file storage (save/load round-trip)
- Keyring storage (with mocking)
- Public API (save, get, delete, is_configured)
- Legacy migration (v4.0.0 base64 → v4.1.0 Fernet)
- Initialization and cleanup
"""

import os
import base64
from unittest import mock
from pathlib import Path
import pytest

# Import the module under test
from plot_antenna.api_keys import (
    PROVIDERS,
    get_user_data_dir,
    save_api_key,
    delete_api_key,
    get_api_key,
    is_api_key_configured,
    initialize_keys,
    clear_env_keys,
    _get_encryption_key,
    _get_fallback_key_path,
    _save_to_encrypted_file,
    _load_from_encrypted_file,
    _load_from_keyring,
    _save_to_keyring,
    _delete_from_keyring,
    _load_from_env_var,
    _migrate_legacy_keys,
    CRYPTO_AVAILABLE,
)


# ────────────────────────────────────────────────────────────────────────────
# Test Class 1: Provider Registry
# ────────────────────────────────────────────────────────────────────────────


class TestProviderRegistry:
    """Tests for PROVIDERS dictionary structure and validation."""

    def test_providers_has_openai_and_anthropic(self):
        """Verify PROVIDERS has openai and anthropic entries."""
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS

    def test_provider_has_required_keys(self):
        """Each provider must have required metadata keys."""
        required_keys = {"key_name", "env_vars", "prefix", "display_name", "url"}
        for provider_name, provider_info in PROVIDERS.items():
            assert required_keys.issubset(
                provider_info.keys()
            ), f"{provider_name} missing required keys"
            assert isinstance(provider_info["env_vars"], list)
            assert len(provider_info["env_vars"]) > 0

    def test_unknown_provider_returns_none(self):
        """get_api_key should return None for unknown providers."""
        result = get_api_key("unknown_provider")
        assert result is None


# ────────────────────────────────────────────────────────────────────────────
# Test Class 2: Encryption Key
# ────────────────────────────────────────────────────────────────────────────


class TestEncryptionKey:
    """Tests for _get_encryption_key() and Fernet compatibility."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_encryption_key_returns_bytes_of_correct_length(self):
        """_get_encryption_key() should return 44-byte base64-encoded key."""
        key = _get_encryption_key()
        assert isinstance(key, bytes)
        assert len(key) == 44  # base64 of 32-byte key is 44 chars

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_encryption_key_is_deterministic(self):
        """Key should be deterministic (same on repeated calls)."""
        key1 = _get_encryption_key()
        key2 = _get_encryption_key()
        assert key1 == key2

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_encryption_key_is_valid_for_fernet(self):
        """Key should be usable with Fernet encryption."""
        from cryptography.fernet import Fernet

        key = _get_encryption_key()
        fernet = Fernet(key)  # Should not raise
        encrypted = fernet.encrypt(b"test message")
        decrypted = fernet.decrypt(encrypted)
        assert decrypted == b"test message"


# ────────────────────────────────────────────────────────────────────────────
# Test Class 3: Encrypted File Storage
# ────────────────────────────────────────────────────────────────────────────


class TestEncryptedFileStorage:
    """Tests for encrypted file save/load operations."""

    @pytest.fixture
    def temp_user_dir(self, tmp_path, monkeypatch):
        """Mock get_user_data_dir to use tmp_path for isolated testing."""
        user_dir = tmp_path / "test_rflect_data"
        user_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("plot_antenna.api_keys.get_user_data_dir", lambda: str(user_dir))
        return user_dir

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_save_and_load_round_trip(self, temp_user_dir):
        """Save then load should return the same key."""
        test_key = "sk-test-1234567890abcdef"
        assert _save_to_encrypted_file("openai", test_key) is True
        loaded_key = _load_from_encrypted_file("openai")
        assert loaded_key == test_key

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_per_provider_files(self, temp_user_dir):
        """OpenAI and Anthropic should have different file paths."""
        openai_path = _get_fallback_key_path("openai")
        anthropic_path = _get_fallback_key_path("anthropic")
        assert openai_path != anthropic_path
        assert ".openai_key" in openai_path
        assert ".anthropic_key" in anthropic_path

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_missing_file_returns_none(self, temp_user_dir):
        """Loading from a non-existent file should return None."""
        loaded_key = _load_from_encrypted_file("openai")
        assert loaded_key is None

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_save_loads_from_temp_user_dir(self, temp_user_dir):
        """Verify files are created in mocked user directory."""
        test_key = "sk-test-file-storage"
        _save_to_encrypted_file("openai", test_key)
        expected_file = temp_user_dir / ".openai_key"
        assert expected_file.exists()


# ────────────────────────────────────────────────────────────────────────────
# Test Class 4: Keyring Storage
# ────────────────────────────────────────────────────────────────────────────


class TestKeyringStorage:
    """Tests for OS keyring integration (with mocking)."""

    def test_keyring_save_load_round_trip(self, monkeypatch):
        """Mock keyring save/load round-trip."""
        mock_storage = {}

        def mock_get_password(service, username):
            return mock_storage.get((service, username))

        def mock_set_password(service, username, password):
            mock_storage[(service, username)] = password

        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", True)
        with mock.patch("plot_antenna.api_keys.keyring") as mock_keyring:
            mock_keyring.get_password = mock_get_password
            mock_keyring.set_password = mock_set_password

            test_key = "sk-keyring-test-12345"
            assert _save_to_keyring("openai", test_key) is True
            loaded_key = _load_from_keyring("openai")
            assert loaded_key == test_key

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_fallback_to_file_when_keyring_unavailable(self, tmp_path, monkeypatch):
        """If keyring is unavailable, should fall back to encrypted file."""
        user_dir = tmp_path / "test_rflect_data"
        user_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("plot_antenna.api_keys.get_user_data_dir", lambda: str(user_dir))
        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", False)

        test_key = "sk-fallback-test"
        assert save_api_key("openai", test_key) is True
        # Should be in file, not keyring
        assert (user_dir / ".openai_key").exists()

    def test_handle_keyring_exception_gracefully(self, monkeypatch):
        """Keyring exceptions should not crash, return None instead."""
        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", True)
        with mock.patch("plot_antenna.api_keys.keyring") as mock_keyring:
            mock_keyring.get_password.side_effect = RuntimeError("Keyring locked")
            result = _load_from_keyring("openai")
            assert result is None


# ────────────────────────────────────────────────────────────────────────────
# Test Class 5: Public API
# ────────────────────────────────────────────────────────────────────────────


class TestPublicAPI:
    """Tests for public API functions."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Remove API key env vars and cache before each test."""
        from plot_antenna.api_keys import _key_cache
        _key_cache.clear()
        for provider_info in PROVIDERS.values():
            for var_name in provider_info["env_vars"]:
                monkeypatch.delenv(var_name, raising=False)

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_save_and_get_api_key_round_trip(self, tmp_path, monkeypatch, clean_env):
        """save_api_key + get_api_key should round-trip successfully."""
        user_dir = tmp_path / "test_rflect_data"
        user_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("plot_antenna.api_keys.get_user_data_dir", lambda: str(user_dir))
        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", False)

        test_key = "sk-public-api-test"
        assert save_api_key("openai", test_key) is True
        retrieved_key = get_api_key("openai")
        assert retrieved_key == test_key

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_delete_api_key_removes_key(self, tmp_path, monkeypatch, clean_env):
        """delete_api_key should remove the key from storage."""
        user_dir = tmp_path / "test_rflect_data"
        user_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("plot_antenna.api_keys.get_user_data_dir", lambda: str(user_dir))
        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", False)

        test_key = "sk-delete-test"
        save_api_key("openai", test_key)
        assert is_api_key_configured("openai") is True
        delete_api_key("openai")
        assert is_api_key_configured("openai") is False

    def test_is_api_key_configured_returns_correct_bool(self, monkeypatch, clean_env):
        """is_api_key_configured should check env vars."""
        assert is_api_key_configured("openai") is False
        monkeypatch.setenv("OPENAI_API_KEY", "sk-configured-test")
        assert is_api_key_configured("openai") is True

    def test_clear_env_keys_removes_env_vars(self, monkeypatch):
        """clear_env_keys should delete all provider env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-2")
        clear_env_keys()
        assert "OPENAI_API_KEY" not in os.environ
        assert "ANTHROPIC_API_KEY" not in os.environ


# ────────────────────────────────────────────────────────────────────────────
# Test Class 6: Legacy Migration
# ────────────────────────────────────────────────────────────────────────────


class TestLegacyMigration:
    """Tests for v4.0.0 base64 → v4.1.0 Fernet migration."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_migrate_base64_legacy_file(self, tmp_path, monkeypatch):
        """Detect and migrate base64-encoded legacy key file."""
        user_dir = tmp_path / "test_rflect_data"
        user_dir.mkdir(exist_ok=True)

        def mock_get_user_data_dir():
            return str(user_dir)

        # Patch all references to get_user_data_dir in the api_keys module
        monkeypatch.setattr("plot_antenna.api_keys.get_user_data_dir", mock_get_user_data_dir)
        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", False)

        # Create legacy base64-encoded file
        legacy_key = "sk-legacy-test-1234567890"
        legacy_path = user_dir / ".openai_key"
        encoded = base64.b64encode(legacy_key.encode()).decode()
        legacy_path.write_text(encoded)

        # Verify legacy file exists before migration
        assert legacy_path.exists()

        # Trigger migration
        with mock.patch("plot_antenna.api_keys.save_api_key") as mock_save:
            mock_save.return_value = True
            _migrate_legacy_keys()

            # Verify save_api_key was called with the decoded legacy key
            mock_save.assert_called_once_with("openai", legacy_key)

        # After migration with successful save, legacy file should be removed
        assert not legacy_path.exists()

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_skip_non_legacy_fernet_files(self, tmp_path, monkeypatch):
        """Non-legacy (already Fernet-encrypted) files should not be migrated."""
        user_dir = tmp_path / "test_rflect_data"
        user_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("plot_antenna.api_keys.get_user_data_dir", lambda: str(user_dir))
        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", False)

        # Create Fernet-encrypted file
        test_key = "sk-fernet-test"
        _save_to_encrypted_file("openai", test_key)
        legacy_path = user_dir / ".openai_key"
        original_mtime = legacy_path.stat().st_mtime

        # Trigger migration (should skip)
        _migrate_legacy_keys()

        # File should still exist (not removed)
        assert legacy_path.exists()

        # Content should be unchanged
        loaded_key = _load_from_encrypted_file("openai")
        assert loaded_key == test_key


# ────────────────────────────────────────────────────────────────────────────
# Test Class 7: Initialization
# ────────────────────────────────────────────────────────────────────────────


class TestInitialize:
    """Tests for initialize_keys() startup function."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not installed")
    def test_initialize_keys_loads_all_providers(self, tmp_path, monkeypatch):
        """initialize_keys should load keys for all providers."""
        from plot_antenna.api_keys import _key_cache
        _key_cache.clear()

        user_dir = tmp_path / "test_rflect_data"
        user_dir.mkdir(exist_ok=True)
        monkeypatch.setattr("plot_antenna.api_keys.get_user_data_dir", lambda: str(user_dir))
        monkeypatch.setattr("plot_antenna.api_keys.KEYRING_AVAILABLE", False)

        # Clear env vars
        for provider_info in PROVIDERS.values():
            for var_name in provider_info["env_vars"]:
                monkeypatch.delenv(var_name, raising=False)

        # Save keys for both providers
        _save_to_encrypted_file("openai", "sk-openai-init-test")
        _save_to_encrypted_file("anthropic", "sk-ant-init-test")

        # Clear env vars again (simulate fresh start)
        for provider_info in PROVIDERS.values():
            for var_name in provider_info["env_vars"]:
                monkeypatch.delenv(var_name, raising=False)

        # Initialize should load all keys into cache (not os.environ)
        initialize_keys()

        assert get_api_key("openai") == "sk-openai-init-test"
        assert get_api_key("anthropic") == "sk-ant-init-test"
