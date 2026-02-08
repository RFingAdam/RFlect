r"""
API Key Management for RFlect — v4.1.0

Secure multi-provider API key storage and retrieval:
- OpenAI, Anthropic keys (Ollama is local, no key needed)
- OS Keyring (Windows Credential Manager / macOS Keychain / Linux keyring)
- Fernet-encrypted fallback file with restrictive permissions
- Environment variables and .env files for development

Priority order for loading:
1. OS Keyring (most secure)
2. Fernet-encrypted file in user data directory
3. Environment variable
4. .env file (development only)

Keys are NEVER stored in plaintext. The encrypted fallback uses AES-128-CBC
with HMAC-SHA256 (Fernet) keyed to the machine's unique identifier.
"""

import os
import sys
import uuid
from typing import Optional

# Try to import keyring for secure storage
try:
    import keyring  # type: ignore[import-unresolved]

    KEYRING_AVAILABLE = True
except ImportError:
    keyring = None  # type: ignore[assignment]
    KEYRING_AVAILABLE = False

# Try to import cryptography for encrypted file storage
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    import base64 as _b64

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Try to import dotenv for .env file loading
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# In-memory key cache — avoids leaking keys into os.environ
_key_cache: dict = {}

# ────────────────────────────────────────────────────────────────────────────
# PROVIDER REGISTRY
# ────────────────────────────────────────────────────────────────────────────

SERVICE_NAME = "RFlect"

PROVIDERS = {
    "openai": {
        "key_name": "openai_api_key",
        "env_vars": ["OPENAI_API_KEY"],
        "prefix": "sk-",
        "display_name": "OpenAI",
        "url": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "key_name": "anthropic_api_key",
        "env_vars": ["ANTHROPIC_API_KEY"],
        "prefix": "sk-ant-",
        "display_name": "Anthropic",
        "url": "https://console.anthropic.com/settings/keys",
    },
}

ENV_FILE_NAMES = [".env", "openai.env", "openapi.env"]


# ────────────────────────────────────────────────────────────────────────────
# USER DATA DIRECTORY
# ────────────────────────────────────────────────────────────────────────────


def get_user_data_dir() -> str:
    """Get user-specific data directory for storing settings."""
    if sys.platform == "win32":
        app_data = os.getenv("LOCALAPPDATA", os.path.expanduser("~"))
        user_dir = os.path.join(app_data, "RFlect")
    elif sys.platform == "darwin":
        user_dir = os.path.expanduser("~/Library/Application Support/RFlect")
    else:
        user_dir = os.path.expanduser("~/.config/RFlect")

    os.makedirs(user_dir, exist_ok=True)
    return user_dir


# ────────────────────────────────────────────────────────────────────────────
# FERNET ENCRYPTION (replaces base64 obfuscation from v4.0.0)
# ────────────────────────────────────────────────────────────────────────────


def _get_machine_id() -> str:
    """Get a machine-unique identifier, preferring OS-specific IDs over MAC address."""
    import sys

    if sys.platform == "linux":
        try:
            with open("/etc/machine-id", "r") as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            pass
    elif sys.platform == "darwin":
        import subprocess

        try:
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]
        except Exception:
            pass
    elif sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography"
            )
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(value)
        except Exception:
            pass
    # Fallback to MAC address
    return str(uuid.getnode())


def _get_encryption_key() -> bytes:
    """Derive a machine-unique Fernet key via PBKDF2.

    Uses an OS-specific machine identifier (falling back to MAC address)
    combined with a fixed application salt. The key is deterministic for
    a given machine, so encrypted files can only be decrypted on the
    same machine.
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package required for encrypted key storage")

    machine_id = _get_machine_id().encode("utf-8")
    salt = b"RFlect_v4.1_secure_key_storage"

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480_000,
    )
    key_material = kdf.derive(machine_id)
    return _b64.urlsafe_b64encode(key_material)


def _get_fallback_key_path(provider_name: str) -> str:
    """Get path to encrypted key file for a provider."""
    info = PROVIDERS.get(provider_name)
    if not info:
        raise ValueError(f"Unknown provider: {provider_name}")
    filename = f".{provider_name}_key"
    return os.path.join(get_user_data_dir(), filename)


def _set_restrictive_permissions(file_path: str) -> None:
    """Set owner-only permissions on a key file."""
    if sys.platform == "win32":
        try:
            import subprocess

            username = os.getenv("USERNAME", "")
            if username:
                subprocess.run(
                    [
                        "icacls",
                        file_path,
                        "/inheritance:r",
                        "/grant:r",
                        f"{username}:(R,W)",
                    ],
                    capture_output=True,
                    check=True,
                )
        except Exception as e:
            print(f"[WARNING] Could not set Windows ACL: {e}")
    else:
        try:
            os.chmod(file_path, 0o600)
        except OSError as e:
            print(f"[WARNING] Could not set file permissions: {e}")


def _save_to_encrypted_file(provider_name: str, api_key: str) -> bool:
    """Save API key to Fernet-encrypted file with restrictive permissions."""
    if not CRYPTO_AVAILABLE:
        print("[WARNING] cryptography package not installed — cannot save encrypted key")
        return False
    try:
        key_path = _get_fallback_key_path(provider_name)
        fernet = Fernet(_get_encryption_key())
        encrypted = fernet.encrypt(api_key.encode("utf-8"))

        with open(key_path, "wb") as f:
            f.write(encrypted)

        _set_restrictive_permissions(key_path)
        print(f"[OK] {PROVIDERS[provider_name]['display_name']} key saved to encrypted storage")
        return True
    except Exception as e:
        print(f"[WARNING] Could not save {provider_name} key to file: {e}")
        return False


def _load_from_encrypted_file(provider_name: str) -> Optional[str]:
    """Load API key from Fernet-encrypted file."""
    if not CRYPTO_AVAILABLE:
        return None
    try:
        key_path = _get_fallback_key_path(provider_name)
    except ValueError:
        return None
    if not os.path.exists(key_path):
        return None
    try:
        fernet = Fernet(_get_encryption_key())
        with open(key_path, "rb") as f:
            encrypted = f.read()
        decrypted = fernet.decrypt(encrypted).decode("utf-8")
        display = PROVIDERS[provider_name]["display_name"]
        print(f"[OK] {display} key loaded from encrypted storage")
        return decrypted
    except (InvalidToken, Exception) as e:
        print(f"[WARNING] Could not decrypt {provider_name} key file: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# KEYRING STORAGE
# ────────────────────────────────────────────────────────────────────────────


def _load_from_keyring(provider_name: str) -> Optional[str]:
    """Load API key from OS keyring."""
    if not KEYRING_AVAILABLE or keyring is None:
        return None
    info = PROVIDERS.get(provider_name)
    if not info:
        return None
    try:
        key = keyring.get_password(SERVICE_NAME, info["key_name"])
        if key:
            display = info["display_name"]
            print(f"[OK] {display} key loaded from OS keyring")
        return key
    except Exception as e:
        print(f"[WARNING] Could not access OS keyring for {provider_name}: {e}")
        return None


def _save_to_keyring(provider_name: str, api_key: str) -> bool:
    """Save API key to OS keyring."""
    if not KEYRING_AVAILABLE or keyring is None:
        return False
    info = PROVIDERS.get(provider_name)
    if not info:
        return False
    try:
        keyring.set_password(SERVICE_NAME, info["key_name"], api_key)
        print(f"[OK] {info['display_name']} key saved to OS keyring")
        return True
    except Exception as e:
        print(f"[WARNING] Could not save to OS keyring: {e}")
        return False


def _delete_from_keyring(provider_name: str) -> bool:
    """Delete API key from OS keyring."""
    if not KEYRING_AVAILABLE or keyring is None:
        return False
    info = PROVIDERS.get(provider_name)
    if not info:
        return False
    try:
        keyring.delete_password(SERVICE_NAME, info["key_name"])
        print(f"[OK] {info['display_name']} key removed from OS keyring")
        return True
    except Exception:
        return False


# ────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT & .ENV LOADING
# ────────────────────────────────────────────────────────────────────────────


def _load_from_env_var(provider_name: str) -> Optional[str]:
    """Load API key from environment variable."""
    info = PROVIDERS.get(provider_name)
    if not info:
        return None
    for var_name in info["env_vars"]:
        key = os.getenv(var_name)
        if key:
            print(f"[OK] {info['display_name']} key loaded from env: {var_name}")
            return key
    return None


def _load_from_env_file(provider_name: str) -> Optional[str]:
    """Load API key from .env file (development only)."""
    if not DOTENV_AVAILABLE:
        return None
    info = PROVIDERS.get(provider_name)
    if not info:
        return None

    search_paths = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]

    for search_path in search_paths:
        for env_file in ENV_FILE_NAMES:
            env_path = os.path.join(search_path, env_file)
            if os.path.exists(env_path):
                try:
                    load_dotenv(env_path)
                    for var_name in info["env_vars"]:
                        key = os.getenv(var_name)
                        if key:
                            print(f"[OK] {info['display_name']} key loaded from {env_file}")
                            return key
                except Exception as e:
                    print(f"[WARNING] Error loading {env_path}: {e}")

    return None


# ────────────────────────────────────────────────────────────────────────────
# LEGACY MIGRATION (v4.0.0 base64 → v4.1.0 Fernet)
# ────────────────────────────────────────────────────────────────────────────


def _migrate_legacy_keys() -> None:
    """Migrate v4.0.0 base64-encoded key files to v4.1.0 Fernet encryption."""
    import base64

    legacy_path = os.path.join(get_user_data_dir(), ".openai_key")
    if not os.path.exists(legacy_path):
        return

    try:
        with open(legacy_path, "r", encoding="utf-8") as f:
            encoded = f.read().strip()
        if not encoded:
            return

        # Try base64 decode — if it fails, file may already be Fernet-encrypted
        try:
            key = base64.b64decode(encoded.encode()).decode("utf-8")
        except Exception:
            return  # Not a legacy base64 file

        # Looks like a valid API key? Re-encrypt and remove legacy file
        if key and len(key) > 10:
            if save_api_key("openai", key):
                os.remove(legacy_path)
                print("[OK] Migrated legacy OpenAI key to encrypted storage")
    except Exception as e:
        print(f"[WARNING] Could not migrate legacy key: {e}")


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────────────────────


def load_api_key(provider_name: str = "openai") -> Optional[str]:
    """Load API key for a provider from the most secure source available.

    Priority: OS keyring → encrypted file → env var → .env file

    Args:
        provider_name: Provider identifier ("openai" or "anthropic")

    Returns:
        The API key string, or None if not found.
    """
    if provider_name not in PROVIDERS:
        return None

    info = PROVIDERS[provider_name]

    # 1. OS keyring
    key = _load_from_keyring(provider_name)
    if key:
        _key_cache[info["env_vars"][0]] = key
        return key

    # 2. Encrypted file
    key = _load_from_encrypted_file(provider_name)
    if key:
        _key_cache[info["env_vars"][0]] = key
        return key

    # 3. Environment variable
    key = _load_from_env_var(provider_name)
    if key:
        return key

    # 4. .env file
    key = _load_from_env_file(provider_name)
    if key:
        return key

    return None


def save_api_key(provider_name: str, api_key: str) -> bool:
    """Save API key for a provider to the most secure storage available.

    Tries OS keyring first, falls back to Fernet-encrypted file.

    Args:
        provider_name: Provider identifier ("openai" or "anthropic")
        api_key: The raw API key string

    Returns:
        True if saved successfully.
    """
    if not api_key or not api_key.strip():
        return False
    if provider_name not in PROVIDERS:
        return False

    api_key = api_key.strip()
    info = PROVIDERS[provider_name]

    # Try keyring first
    if _save_to_keyring(provider_name, api_key):
        _key_cache[info["env_vars"][0]] = api_key
        return True

    # Fall back to encrypted file
    if _save_to_encrypted_file(provider_name, api_key):
        _key_cache[info["env_vars"][0]] = api_key
        return True

    return False


def delete_api_key(provider_name: str) -> bool:
    """Delete stored API key for a provider from all storage locations.

    Args:
        provider_name: Provider identifier ("openai" or "anthropic")

    Returns:
        True if deleted from at least one location.
    """
    if provider_name not in PROVIDERS:
        return False

    info = PROVIDERS[provider_name]
    deleted = False

    # Remove from keyring
    if _delete_from_keyring(provider_name):
        deleted = True

    # Remove encrypted file
    try:
        key_path = _get_fallback_key_path(provider_name)
        if os.path.exists(key_path):
            os.remove(key_path)
            print(f"[OK] {info['display_name']} key file removed")
            deleted = True
    except Exception as e:
        print(f"[WARNING] Could not remove key file: {e}")

    # Remove from environment and cache
    for var_name in info["env_vars"]:
        if var_name in os.environ:
            del os.environ[var_name]
        _key_cache.pop(var_name, None)

    return deleted


def get_api_key(provider_name: str = "openai") -> Optional[str]:
    """Get the currently active API key for a provider.

    Checks in-memory cache first, then falls back to environment variables.

    Args:
        provider_name: Provider identifier ("openai" or "anthropic")

    Returns:
        The API key string, or None.
    """
    info = PROVIDERS.get(provider_name)
    if not info:
        return None
    for var_name in info["env_vars"]:
        # Check in-memory cache first
        key = _key_cache.get(var_name)
        if key:
            return key
        # Fall back to environment variable
        key = os.getenv(var_name)
        if key:
            return key
    return None


def is_api_key_configured(provider_name: str = "openai") -> bool:
    """Check if an API key is available for a provider.

    Args:
        provider_name: Provider identifier ("openai" or "anthropic")

    Returns:
        True if a key is available in environment.
    """
    return get_api_key(provider_name) is not None


def test_api_key(provider_name: str, api_key: str) -> tuple:
    """Test that an API key is valid by making a minimal API call.

    Args:
        provider_name: Provider identifier
        api_key: The key to test

    Returns:
        (success: bool, message: str)
    """
    if provider_name == "openai":
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=10)
            client.models.list()
            return (True, "OpenAI key is valid")
        except Exception as e:
            return (False, f"OpenAI key validation failed: {e}")

    elif provider_name == "anthropic":
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, timeout=10)
            client.models.list()
            return (True, "Anthropic key is valid")
        except ImportError:
            return (False, "anthropic package not installed")
        except Exception as e:
            return (False, f"Anthropic key validation failed: {e}")

    elif provider_name == "ollama":
        try:
            import requests

            url = api_key if api_key else "http://localhost:11434"
            resp = requests.get(f"{url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return (True, f"Ollama connected — {len(models)} model(s) available")
            return (False, f"Ollama returned status {resp.status_code}")
        except Exception as e:
            return (False, f"Ollama connection failed: {e}")

    return (False, f"Unknown provider: {provider_name}")


def initialize_keys() -> None:
    """Explicitly load all configured API keys into environment.

    Call once at application startup. Migrates legacy keys if found.
    """
    _migrate_legacy_keys()
    for provider_name in PROVIDERS:
        load_api_key(provider_name)


def clear_env_keys() -> None:
    """Clear API keys from environment variables and cache. Call on app shutdown."""
    _key_cache.clear()
    for info in PROVIDERS.values():
        for var_name in info["env_vars"]:
            if var_name in os.environ:
                del os.environ[var_name]
