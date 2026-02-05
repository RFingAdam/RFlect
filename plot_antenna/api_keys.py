r"""
API Key Management for RFlect

This module provides secure API key storage and retrieval:
- Local development: Loads from .env files (.env, openai.env, or openapi.env)
- Distributed app: Uses Windows Credential Manager / macOS Keychain / Linux keyring
- Fallback: Base64 obfuscated file in user data directory

Priority order for API key loading:
1. OS Keyring (most secure, recommended for end users)
2. User data file (~/.config/RFlect/.openai_key or %LOCALAPPDATA%\RFlect\.openai_key)
3. Environment variable OPENAI_API_KEY or OPENAI_API_KEY2
4. .env file (.env, openai.env, or openapi.env) for local development
"""

import os
import sys
import base64

# Try to import keyring for secure storage (optional dependency)
try:
    import keyring  # type: ignore[import-unresolved]

    KEYRING_AVAILABLE = True
except ImportError:
    keyring = None  # type: ignore[assignment]
    KEYRING_AVAILABLE = False
    print(
        "[INFO] keyring not installed - using fallback storage. Install with: pip install keyring"
    )

# Try to import dotenv for .env file loading
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# Constants
SERVICE_NAME = "RFlect"
KEY_NAME = "openai_api_key"
ENV_VAR_NAMES = ["OPENAI_API_KEY", "OPENAI_API_KEY2"]
ENV_FILE_NAMES = [".env", "openai.env", "openapi.env"]  # .env takes priority


def get_user_data_dir():
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


def get_fallback_key_path():
    """Get path to fallback key file."""
    return os.path.join(get_user_data_dir(), ".openai_key")


def _load_from_keyring():
    """Load API key from OS keyring (Windows Credential Manager, macOS Keychain, etc.)."""
    if not KEYRING_AVAILABLE or keyring is None:
        return None
    try:
        key = keyring.get_password(SERVICE_NAME, KEY_NAME)
        if key:
            print("[OK] API key loaded from secure storage (OS keyring)")
        return key
    except Exception as e:
        print(f"[WARNING] Could not access OS keyring: {e}")
        return None


def _load_from_fallback_file():
    """Load API key from base64 obfuscated file (fallback method)."""
    key_path = get_fallback_key_path()
    if not os.path.exists(key_path):
        return None
    try:
        with open(key_path, "r", encoding="utf-8") as f:
            encoded_key = f.read().strip()
            if encoded_key:
                key = base64.b64decode(encoded_key.encode()).decode()
                print(f"[OK] API key loaded from user storage: {key[:7]}...{key[-4:]}")
                return key
    except Exception as e:
        print(f"[WARNING] Could not load API key from file: {e}")
    return None


def _load_from_env_var():
    """Load API key from environment variable."""
    for var_name in ENV_VAR_NAMES:
        key = os.getenv(var_name)
        if key:
            print(f"[OK] API key loaded from environment variable: {var_name}")
            return key
    return None


def _load_from_env_file():
    """Load API key from .env file (for local development)."""
    if not DOTENV_AVAILABLE:
        return None

    # Try to find .env file in project root or plot_antenna directory
    search_paths = [
        os.getcwd(),  # Current working directory
        os.path.dirname(os.path.abspath(__file__)),  # plot_antenna/
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # project root
    ]

    for search_path in search_paths:
        for env_file in ENV_FILE_NAMES:
            env_path = os.path.join(search_path, env_file)
            if os.path.exists(env_path):
                try:
                    load_dotenv(env_path)
                    # Check if key is now available
                    for var_name in ENV_VAR_NAMES:
                        key = os.getenv(var_name)
                        if key:
                            print(f"[OK] API key loaded from {env_file} ({var_name})")
                            return key
                except Exception as e:
                    print(f"[WARNING] Error loading {env_path}: {e}")

    return None


def load_api_key():
    """
    Load OpenAI API key from the most appropriate source.

    Priority:
    1. OS Keyring (secure storage)
    2. User data file (fallback storage)
    3. Environment variable
    4. .env file (development)

    Returns:
        str or None: The API key if found, None otherwise
    """
    # 1. Try OS keyring first (most secure)
    key = _load_from_keyring()
    if key:
        os.environ["OPENAI_API_KEY"] = key
        return key

    # 2. Try user data file (fallback storage)
    key = _load_from_fallback_file()
    if key:
        os.environ["OPENAI_API_KEY"] = key
        return key

    # 3. Try environment variable (may be set externally)
    key = _load_from_env_var()
    if key:
        return key

    # 4. Try .env file (for development)
    key = _load_from_env_file()
    if key:
        return key

    print("[WARNING] No OpenAI API key found.")
    print("   Configure via: Help -> Manage OpenAI API Key")
    return None


def save_api_key(api_key):
    """
    Save API key to secure storage.

    Uses OS keyring if available, falls back to obfuscated file.

    Args:
        api_key: The OpenAI API key to save

    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not api_key or not api_key.strip():
        return False

    api_key = api_key.strip()

    # Try OS keyring first
    if KEYRING_AVAILABLE and keyring is not None:
        try:
            keyring.set_password(SERVICE_NAME, KEY_NAME, api_key)
            os.environ["OPENAI_API_KEY"] = api_key
            print("[OK] API key saved to secure storage (OS keyring)")
            return True
        except Exception as e:
            print(f"[WARNING] Could not save to OS keyring: {e}")
            # Fall through to file-based storage

    # Fallback to file-based storage
    try:
        key_path = get_fallback_key_path()
        encoded_key = base64.b64encode(api_key.encode()).decode()
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(encoded_key)
        os.environ["OPENAI_API_KEY"] = api_key
        print("[OK] API key saved to user storage")
        return True
    except Exception as e:
        print(f"[ERROR] Could not save API key: {e}")
        return False


def delete_api_key():
    """
    Delete stored API key from all storage locations.

    Returns:
        bool: True if deleted successfully, False otherwise
    """
    success = True

    # Remove from OS keyring
    if KEYRING_AVAILABLE and keyring is not None:
        try:
            keyring.delete_password(SERVICE_NAME, KEY_NAME)
            print("[OK] API key removed from secure storage")
        except Exception as e:
            # PasswordDeleteError means key wasn't stored there
            if "PasswordDeleteError" not in str(type(e).__name__):
                print(f"[WARNING] Could not remove from OS keyring: {e}")

    # Remove from file storage
    try:
        key_path = get_fallback_key_path()
        if os.path.exists(key_path):
            os.remove(key_path)
            print("[OK] API key file removed")
    except Exception as e:
        print(f"[WARNING] Could not remove key file: {e}")
        success = False

    # Remove from environment
    for var_name in ENV_VAR_NAMES:
        if var_name in os.environ:
            del os.environ[var_name]

    return success


def get_api_key():
    """
    Get the currently active API key.

    Returns:
        str or None: The API key if available
    """
    for var_name in ENV_VAR_NAMES:
        key = os.getenv(var_name)
        if key:
            return key
    return None


def is_api_key_configured():
    """
    Check if an API key is configured and available.

    Returns:
        bool: True if API key is available
    """
    return get_api_key() is not None


# Auto-load API key when module is imported
_loaded_key = load_api_key()
