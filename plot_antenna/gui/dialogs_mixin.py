"""
DialogsMixin - Dialog-related methods for RFlect GUI

This mixin provides all dialog-related functionality:
- About dialog
- API Key management
- AI Settings dialog
- Scan type settings dialogs (Active, Passive, VSWR)
"""

from __future__ import annotations

import os
import datetime
import webbrowser
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional

from ..config import (
    DARK_BG_COLOR,
    LIGHT_TEXT_COLOR,
    ACCENT_BLUE_COLOR,
    BUTTON_COLOR,
    SURFACE_COLOR,
    SECTION_HEADER_FONT,
    AI_MODEL,
    AI_RESPONSE_STYLE,
    AI_MAX_TOKENS,
    AI_REASONING_EFFORT,
)

# Import additional AI settings with fallbacks
try:
    from ..config import AI_TEXT_VERBOSITY
except ImportError:
    AI_TEXT_VERBOSITY = "auto"

try:
    from ..config import AI_GENERATE_REASONING_SUMMARY
except ImportError:
    AI_GENERATE_REASONING_SUMMARY = False

try:
    from ..config import AI_INCLUDE_RECOMMENDATIONS
except ImportError:
    AI_INCLUDE_RECOMMENDATIONS = False

# Import centralized API key management
from ..api_keys import (
    save_api_key as api_keys_save,
    delete_api_key as api_keys_delete,
    get_api_key,
    is_api_key_configured,
    test_api_key as api_keys_test,
    PROVIDERS,
)

# Import utility functions
from .utils import resource_path

if TYPE_CHECKING:
    from .base_protocol import AntennaPlotGUIProtocol


class DialogsMixin:
    """Mixin class providing dialog methods for AntennaPlotGUI.

    Type hints reference AntennaPlotGUIProtocol for IDE support.
    At runtime, this mixin is combined with the main GUI class.
    """

    # Type hints for IDE support (these are satisfied by the main class)
    root: tk.Tk
    scan_type: tk.StringVar
    passive_scan_type: tk.StringVar
    interpolate_3d_plots: bool
    axis_scale_mode: tk.StringVar
    axis_min: tk.DoubleVar
    axis_max: tk.DoubleVar
    datasheet_plots_var: tk.BooleanVar
    min_max_eff_gain_var: tk.BooleanVar
    min_max_vswr_var: tk.BooleanVar
    cb_groupdelay_sff_var: tk.BooleanVar
    ecc_analysis_enabled: bool
    shadowing_enabled: bool
    shadow_direction: str
    saved_limit1_freq1: float
    saved_limit1_freq2: float
    saved_limit1_start: float
    saved_limit1_stop: float
    saved_limit2_freq1: float
    saved_limit2_freq2: float
    saved_limit2_start: float
    saved_limit2_stop: float
    saved_min_max_vswr: bool
    cb_groupdelay_sff: bool
    CURRENT_VERSION: str

    hpol_file_path: Optional[str]
    vpol_file_path: Optional[str]

    # Method declarations for type checking only (not defined at runtime to avoid MRO conflicts)
    if TYPE_CHECKING:

        def resource_path(self, relative_path: str) -> str: ...
        def get_user_data_dir(self) -> str: ...
        def update_visibility(self) -> None: ...
        def _run_extrapolation(self, target_frequency: float) -> None: ...

    # ────────────────────────────────────────────────────────────────────────
    # ABOUT DIALOG
    # ────────────────────────────────────────────────────────────────────────

    def show_about_dialog(self):
        """Show the About RFlect dialog with version and credits."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About RFlect")
        about_window.geometry("500x400")
        about_window.resizable(False, False)
        about_window.configure(bg=DARK_BG_COLOR)

        # Center the window
        about_window.transient(self.root)
        about_window.grab_set()

        # Header Frame with Logo and Name
        header_frame = tk.Frame(about_window, bg=DARK_BG_COLOR)
        header_frame.pack(pady=(20, 10))

        # Logo (smith_logo.png)
        try:
            logo_path = resource_path(os.path.join("assets", "smith_logo.png"))
            if os.path.exists(logo_path):
                from PIL import Image, ImageTk

                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(header_frame, image=logo_photo, bg=DARK_BG_COLOR)
                logo_label.image = logo_photo  # type: ignore # Keep a reference
                logo_label.pack(side=tk.LEFT, padx=(0, 15))
        except (FileNotFoundError, ImportError, OSError) as e:
            print(f"[INFO] Could not load logo: {e}")
            pass  # No logo, that's okay

        # App Name (red color to match logo)
        name_label = tk.Label(
            header_frame,
            text="RFlect",
            font=("Arial", 28, "bold"),
            bg=DARK_BG_COLOR,
            fg="#E63946",  # Red color similar to smith_logo.png
        )
        name_label.pack(side=tk.LEFT)

        # Version
        version_label = tk.Label(
            about_window,
            text=f"Version {self.CURRENT_VERSION}",
            font=("Arial", 12),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        version_label.pack()

        # Description
        desc_label = tk.Label(
            about_window,
            text="Antenna Measurement & Analysis Tool",
            font=("Arial", 10, "italic"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        desc_label.pack(pady=(5, 20))

        # Credits
        credits_frame = tk.Frame(about_window, bg=DARK_BG_COLOR)
        credits_frame.pack(pady=10)

        credits_text = """
Developed by: Adam Engelbrecht

Features:
- Active & Passive Antenna Measurements
- 2D & 3D Radiation Pattern Visualization
- Polarization Analysis (AR, Tilt, Sense, XPD)
- AI-Powered Report Generation
- Group Delay & Fidelity Analysis
- VSWR & S-Parameter Analysis
"""

        credits_label = tk.Label(
            credits_frame,
            text=credits_text,
            font=("Arial", 9),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            justify=tk.LEFT,
        )
        credits_label.pack()

        # License info
        license_label = tk.Label(
            about_window,
            text="Licensed under GNU General Public License v3.0",
            font=("Arial", 8),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        license_label.pack(pady=(5, 0))

        # Links
        links_frame = tk.Frame(about_window, bg=DARK_BG_COLOR)
        links_frame.pack(pady=10)

        github_btn = tk.Button(
            links_frame,
            text="View on GitHub",
            command=lambda: webbrowser.open("https://github.com/RFingAdam/RFlect"),
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        github_btn.pack(side=tk.LEFT, padx=5)

        license_btn = tk.Button(
            links_frame,
            text="View License",
            command=lambda: webbrowser.open(
                "https://github.com/RFingAdam/RFlect/blob/main/LICENSE"
            ),
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        license_btn.pack(side=tk.LEFT, padx=5)

        # Close button
        close_btn = tk.Button(
            about_window,
            text="Close",
            command=about_window.destroy,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=10,
        )
        close_btn.pack(pady=20)

    # ────────────────────────────────────────────────────────────────────────
    # API KEY MANAGEMENT
    # ────────────────────────────────────────────────────────────────────────

    def load_api_key(self, provider_name="openai"):
        """Load API key using centralized api_keys module."""
        return get_api_key(provider_name)

    def save_api_key(self, api_key, provider_name="openai"):
        """Save API key using centralized api_keys module."""
        return api_keys_save(provider_name, api_key)

    def delete_api_key(self, provider_name="openai"):
        """Delete stored API key using centralized api_keys module."""
        return api_keys_delete(provider_name)

    def manage_api_key(self):
        """Show multi-provider API key management dialog with tabbed interface."""
        api_window = tk.Toplevel(self.root)
        api_window.title("Manage API Keys")
        api_window.geometry("600x480")
        api_window.resizable(False, False)
        api_window.configure(bg=DARK_BG_COLOR)
        api_window.transient(self.root)
        api_window.grab_set()

        # Title
        tk.Label(
            api_window,
            text="API Key Management",
            font=("Arial", 14, "bold"),
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
        ).pack(pady=(15, 5))

        tk.Label(
            api_window,
            text="Keys are encrypted (AES-128) and stored locally. Never uploaded.",
            font=("Arial", 9),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).pack(pady=(0, 10))

        # Configure ttk style for dark tabs
        style = ttk.Style()
        style.configure("Keys.TNotebook", background=DARK_BG_COLOR)
        style.configure(
            "Keys.TNotebook.Tab",
            background=BUTTON_COLOR,
            foreground=LIGHT_TEXT_COLOR,
            padding=[12, 4],
        )
        style.map(
            "Keys.TNotebook.Tab",
            background=[("selected", ACCENT_BLUE_COLOR)],
            foreground=[("selected", LIGHT_TEXT_COLOR)],
        )

        notebook = ttk.Notebook(api_window, style="Keys.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 5))

        def _build_provider_tab(parent, provider_name):
            """Build a single provider tab with status, entry, and action buttons."""
            info = PROVIDERS[provider_name]
            frame = tk.Frame(parent, bg=DARK_BG_COLOR)
            frame.pack(fill=tk.BOTH, expand=True)

            # --- Status ---
            current_key = get_api_key(provider_name)
            configured = current_key is not None

            status_frame = tk.Frame(frame, bg=DARK_BG_COLOR)
            status_frame.pack(fill=tk.X, padx=20, pady=(15, 5))

            status_color = "#4CAF50" if configured else "#FFC107"
            status_text = "Configured" if configured else "Not configured"
            status_lbl = tk.Label(
                status_frame,
                text=status_text,
                font=("Arial", 10, "bold"),
                bg=DARK_BG_COLOR,
                fg=status_color,
            )
            status_lbl.pack(side=tk.LEFT)

            if configured and current_key:
                masked = (
                    f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "***"
                )
                tk.Label(
                    status_frame,
                    text=f"  ({masked})",
                    font=("Courier", 9),
                    bg=DARK_BG_COLOR,
                    fg=LIGHT_TEXT_COLOR,
                ).pack(side=tk.LEFT)

            # --- Key entry ---
            entry_frame = tk.Frame(frame, bg=DARK_BG_COLOR)
            entry_frame.pack(fill=tk.X, padx=20, pady=10)

            tk.Label(
                entry_frame,
                text="API Key:",
                font=("Arial", 10),
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).pack(anchor=tk.W)

            key_var = tk.StringVar()
            key_entry = tk.Entry(
                entry_frame,
                textvariable=key_var,
                width=65,
                show="*",
                font=("Courier", 9),
            )
            key_entry.pack(fill=tk.X, pady=(3, 5))

            # Show/Hide toggle
            show_var = tk.BooleanVar(value=False)

            def _toggle_show():
                if show_var.get():
                    key_entry.config(show="")
                    show_btn.config(text="Hide")
                else:
                    key_entry.config(show="*")
                    show_btn.config(text="Show")

            show_btn = tk.Button(
                entry_frame,
                text="Show",
                command=_toggle_show,
                bg=BUTTON_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=6,
            )
            show_btn.pack(anchor=tk.W)

            # --- Link to provider key page ---
            link_text = f"Get your key: {info['url']}"
            link_lbl = tk.Label(
                frame,
                text=link_text,
                font=("Arial", 8, "italic"),
                bg=DARK_BG_COLOR,
                fg=ACCENT_BLUE_COLOR,
                cursor="hand2",
            )
            link_lbl.pack(padx=20, anchor=tk.W, pady=(0, 10))
            link_lbl.bind("<Button-1>", lambda e, u=info["url"]: webbrowser.open(u))

            # --- Action buttons ---
            btn_frame = tk.Frame(frame, bg=DARK_BG_COLOR)
            btn_frame.pack(padx=20, pady=(5, 10))

            def _save():
                key = key_var.get().strip()
                if not key:
                    messagebox.showwarning("Empty Key", "Please enter an API key.")
                    return
                prefix = info.get("prefix", "")
                if prefix and not key.startswith(prefix):
                    if not messagebox.askyesno(
                        "Format Warning",
                        f"Key doesn't start with '{prefix}'. Save anyway?",
                    ):
                        return
                if api_keys_save(provider_name, key):
                    status_lbl.config(text="Configured", fg="#4CAF50")
                    messagebox.showinfo("Saved", f"{info['display_name']} key saved.")
                    key_var.set("")

            def _delete():
                if not is_api_key_configured(provider_name):
                    messagebox.showinfo("Nothing to Delete", "No key is stored for this provider.")
                    return
                if messagebox.askyesno(
                    "Confirm Delete",
                    f"Delete the stored {info['display_name']} API key?",
                ):
                    if api_keys_delete(provider_name):
                        status_lbl.config(text="Not configured", fg="#FFC107")
                        messagebox.showinfo("Deleted", "Key removed.")

            def _test():
                key = key_var.get().strip()
                if not key:
                    # Test stored key
                    key = get_api_key(provider_name)
                if not key:
                    messagebox.showwarning(
                        "No Key",
                        "Enter a key or save one first to test.",
                    )
                    return
                status_lbl.config(text="Testing...", fg="#FFC107")

                def _test_worker():
                    ok, msg = api_keys_test(provider_name, key)
                    self.root.after(0, lambda: _test_done(ok, msg))

                def _test_done(ok, msg):
                    if ok:
                        status_lbl.config(text="Configured", fg="#4CAF50")
                        messagebox.showinfo("Connection OK", msg)
                    else:
                        status_lbl.config(
                            text=(
                                "Configured"
                                if is_api_key_configured(provider_name)
                                else "Not configured"
                            ),
                            fg="#4CAF50" if is_api_key_configured(provider_name) else "#FFC107",
                        )
                        messagebox.showerror("Connection Failed", msg)

                import threading

                threading.Thread(target=_test_worker, daemon=True).start()

            tk.Button(
                btn_frame,
                text="Save Key",
                command=_save,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=12,
                pady=4,
            ).pack(side=tk.LEFT, padx=4)

            tk.Button(
                btn_frame,
                text="Test",
                command=_test,
                bg=BUTTON_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=10,
                pady=4,
            ).pack(side=tk.LEFT, padx=4)

            tk.Button(
                btn_frame,
                text="Delete Key",
                command=_delete,
                bg="#F44336",
                fg=LIGHT_TEXT_COLOR,
                width=12,
                pady=4,
            ).pack(side=tk.LEFT, padx=4)

            return frame

        def _build_ollama_tab(parent):
            """Build the Ollama tab (local server, no API key)."""
            frame = tk.Frame(parent, bg=DARK_BG_COLOR)
            frame.pack(fill=tk.BOTH, expand=True)

            tk.Label(
                frame,
                text="Ollama runs locally — no API key required.",
                font=("Arial", 10),
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).pack(padx=20, pady=(20, 10), anchor=tk.W)

            url_frame = tk.Frame(frame, bg=DARK_BG_COLOR)
            url_frame.pack(fill=tk.X, padx=20, pady=5)

            tk.Label(
                url_frame,
                text="Server URL:",
                font=("Arial", 10),
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).pack(anchor=tk.W)

            try:
                from ..config import AI_OLLAMA_URL
            except ImportError:
                AI_OLLAMA_URL = "http://localhost:11434"

            url_var = tk.StringVar(value=AI_OLLAMA_URL)
            tk.Entry(
                url_frame,
                textvariable=url_var,
                width=50,
                font=("Courier", 9),
            ).pack(fill=tk.X, pady=5)

            ollama_status = tk.Label(
                frame,
                text="",
                font=("Arial", 10, "bold"),
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            ollama_status.pack(padx=20, pady=5, anchor=tk.W)

            def _test_ollama():
                import requests

                url = url_var.get().strip().rstrip("/")
                ollama_status.config(text="Testing...", fg="#FFC107")
                api_window.update()
                try:
                    resp = requests.get(f"{url}/api/tags", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        model_list = ", ".join(models[:5]) if models else "none found"
                        ollama_status.config(text="Connected", fg="#4CAF50")
                        messagebox.showinfo(
                            "Ollama Connected",
                            f"Server responding at {url}\nModels: {model_list}",
                        )
                    else:
                        ollama_status.config(text="Error", fg="#F44336")
                        messagebox.showerror("Error", f"HTTP {resp.status_code} from {url}")
                except Exception as e:
                    ollama_status.config(text="Not running", fg="#F44336")
                    messagebox.showerror(
                        "Connection Failed",
                        f"Could not reach Ollama at {url}\n\n{e}",
                    )

            tk.Button(
                frame,
                text="Test Connection",
                command=_test_ollama,
                bg=BUTTON_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=16,
                pady=4,
            ).pack(padx=20, pady=10, anchor=tk.W)

            tk.Label(
                frame,
                text="Install Ollama: https://ollama.com",
                font=("Arial", 8, "italic"),
                bg=DARK_BG_COLOR,
                fg=ACCENT_BLUE_COLOR,
                cursor="hand2",
            ).pack(padx=20, anchor=tk.W)

            return frame

        # Build tabs for each provider
        for pname in ("openai", "anthropic"):
            tab_frame = tk.Frame(notebook, bg=DARK_BG_COLOR)
            notebook.add(tab_frame, text=f"  {PROVIDERS[pname]['display_name']}  ")
            _build_provider_tab(tab_frame, pname)

        ollama_frame = tk.Frame(notebook, bg=DARK_BG_COLOR)
        notebook.add(ollama_frame, text="  Ollama  ")
        _build_ollama_tab(ollama_frame)

        # Close button at bottom
        tk.Button(
            api_window,
            text="Close",
            command=api_window.destroy,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=12,
            pady=4,
            relief=tk.RIDGE,
        ).pack(pady=(5, 15))

    # ────────────────────────────────────────────────────────────────────────
    # AI SETTINGS DIALOG
    # ────────────────────────────────────────────────────────────────────────

    def manage_ai_settings(self):
        """Show AI configuration settings dialog with multi-provider support."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("AI Settings")
        settings_window.geometry("650x750")
        settings_window.resizable(False, False)
        settings_window.configure(bg=DARK_BG_COLOR)

        # Center the window
        settings_window.transient(self.root)
        settings_window.grab_set()

        # Create scrollable canvas for content
        canvas = tk.Canvas(settings_window, bg=DARK_BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=DARK_BG_COLOR)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Title
        title_label = tk.Label(
            scrollable_frame,
            text="AI Model & Configuration",
            font=("Arial", 14, "bold"),
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
        )
        title_label.pack(pady=(20, 10))

        # Description
        desc_label = tk.Label(
            scrollable_frame,
            text="Configure AI provider, model, and behavior for report generation and chat.",
            font=("Arial", 9),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            wraplength=600,
        )
        desc_label.pack(pady=5, padx=20)

        # Manage API Keys button (opens the multi-provider key dialog)
        keys_btn_frame = tk.Frame(scrollable_frame, bg=DARK_BG_COLOR)
        keys_btn_frame.pack(pady=(5, 10), padx=30, anchor=tk.W)
        tk.Button(
            keys_btn_frame,
            text="Manage API Keys...",
            command=self.manage_api_key,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            font=("Arial", 10),
            padx=10,
            pady=3,
        ).pack(side=tk.LEFT)

        # Main frame
        main_frame = tk.Frame(scrollable_frame, bg=DARK_BG_COLOR)
        main_frame.pack(pady=15, padx=30, fill=tk.BOTH, expand=True)

        row_num = 0

        # ─────────────────────────────────────────────────────────────────
        # AI PROVIDER SELECTION
        # ─────────────────────────────────────────────────────────────────
        tk.Label(
            main_frame,
            text="AI Provider:",
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=row_num, column=0, sticky=tk.W, pady=(0, 5))
        row_num += 1

        try:
            from ..config import AI_PROVIDER
        except ImportError:
            AI_PROVIDER = "openai"

        provider_var = tk.StringVar(value=AI_PROVIDER)
        provider_dropdown = ttk.Combobox(
            main_frame,
            textvariable=provider_var,
            values=["openai", "anthropic", "ollama"],
            state="readonly",
            width=45,
            font=("Arial", 9),
        )
        provider_dropdown.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row_num += 1

        provider_desc_texts = {
            "openai": "OpenAI GPT models (cloud API, requires API key)",
            "anthropic": "Anthropic Claude models (cloud API, requires API key)",
            "ollama": "Ollama local models (runs locally, no API key needed)",
        }
        provider_desc_var = tk.StringVar(value=provider_desc_texts.get(AI_PROVIDER, ""))
        tk.Label(
            main_frame,
            textvariable=provider_desc_var,
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg="#A0A0A0",
            wraplength=550,
        ).grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        row_num += 1

        # Ollama URL field (only visible when Ollama selected)
        ollama_url_frame = tk.Frame(main_frame, bg=DARK_BG_COLOR)
        ollama_url_label = tk.Label(
            ollama_url_frame,
            text="Ollama URL:",
            font=("Arial", 9),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        )
        ollama_url_label.pack(side=tk.LEFT, padx=(0, 5))

        try:
            from ..config import AI_OLLAMA_URL
        except ImportError:
            AI_OLLAMA_URL = "http://localhost:11434"

        ollama_url_var = tk.StringVar(value=AI_OLLAMA_URL)
        ollama_url_entry = tk.Entry(
            ollama_url_frame,
            textvariable=ollama_url_var,
            width=35,
            font=("Arial", 9),
        )
        ollama_url_entry.pack(side=tk.LEFT)
        # Place in grid but only show for Ollama
        ollama_url_frame.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        if AI_PROVIDER != "ollama":
            ollama_url_frame.grid_remove()
        row_num += 1

        # ─────────────────────────────────────────────────────────────────
        # AI MODEL SELECTION
        # ─────────────────────────────────────────────────────────────────
        tk.Label(
            main_frame,
            text="AI Model:",
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=row_num, column=0, sticky=tk.W, pady=(0, 5))
        row_num += 1

        # Models per provider
        openai_models = [
            ("gpt-4o-mini", "GPT-4o Mini - Fast, affordable, functions + vision (RECOMMENDED)"),
            ("gpt-4o", "GPT-4o - Premium quality, functions + vision"),
            ("gpt-5-nano", "GPT-5 Nano - Fastest GPT-5, low cost"),
            ("gpt-5-mini", "GPT-5 Mini - Balanced speed/quality"),
            ("gpt-5.2", "GPT-5.2 - Flagship model, best quality"),
        ]
        anthropic_models = [
            ("claude-sonnet-4-20250514", "Claude Sonnet 4 - Fast, balanced (RECOMMENDED)"),
            ("claude-opus-4-20250514", "Claude Opus 4 - Highest quality"),
            ("claude-haiku-3-5-20241022", "Claude Haiku 3.5 - Fastest, lowest cost"),
        ]
        ollama_models = [
            ("llama3.1", "Llama 3.1 - General purpose, good tool support (RECOMMENDED)"),
            ("qwen2.5", "Qwen 2.5 - Strong multilingual, good reasoning"),
            ("llava", "LLaVA - Vision-capable for plot analysis"),
            ("gemma3", "Gemma 3 - Vision + text, efficient"),
        ]

        provider_models = {
            "openai": openai_models,
            "anthropic": anthropic_models,
            "ollama": ollama_models,
        }

        model_var = tk.StringVar(value=AI_MODEL)

        model_dropdown = ttk.Combobox(
            main_frame,
            textvariable=model_var,
            values=[m[0] for m in openai_models],
            state="readonly",
            width=45,
            font=("Arial", 9),
        )
        model_dropdown.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row_num += 1

        # Model description label
        all_models = openai_models + anthropic_models + ollama_models
        model_desc_var = tk.StringVar(
            value=next((m[1] for m in all_models if m[0] == AI_MODEL), "")
        )
        model_desc_label = tk.Label(
            main_frame,
            textvariable=model_desc_var,
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg="#A0A0A0",
            wraplength=550,
        )
        model_desc_label.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        row_num += 1

        def _update_for_provider(*args):
            prov = provider_var.get()
            provider_desc_var.set(provider_desc_texts.get(prov, ""))
            models = provider_models.get(prov, openai_models)
            model_dropdown["values"] = [m[0] for m in models]
            # Set default model for the selected provider
            if models:
                model_var.set(models[0][0])
            # Show/hide Ollama URL
            if prov == "ollama":
                ollama_url_frame.grid()
            else:
                ollama_url_frame.grid_remove()

        provider_var.trace("w", _update_for_provider)

        def update_model_desc(*args):
            selected = model_var.get()
            desc = next((m[1] for m in all_models if m[0] == selected), "")
            model_desc_var.set(desc)
            # Enable/disable GPT-5 options based on model selection
            is_gpt5 = selected.startswith("gpt-5")
            state = "readonly" if is_gpt5 else "disabled"
            reasoning_dropdown.config(state=state)
            verbosity_dropdown.config(state=state)
            reasoning_summary_cb.config(state=tk.NORMAL if is_gpt5 else tk.DISABLED)

        model_var.trace("w", update_model_desc)

        # Initialize model list for current provider
        cur_models = provider_models.get(AI_PROVIDER, openai_models)
        model_dropdown["values"] = [m[0] for m in cur_models]

        # ─────────────────────────────────────────────────────────────────
        # RESPONSE STYLE (GPT-4 & GPT-5)
        # ─────────────────────────────────────────────────────────────────
        tk.Label(
            main_frame,
            text="Response Style:",
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=row_num, column=0, sticky=tk.W, pady=(10, 5))
        row_num += 1

        style_var = tk.StringVar(value=AI_RESPONSE_STYLE)

        style_dropdown = ttk.Combobox(
            main_frame,
            textvariable=style_var,
            values=["concise", "detailed"],
            state="readonly",
            width=45,
            font=("Arial", 9),
        )
        style_dropdown.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row_num += 1

        tk.Label(
            main_frame,
            text="concise: ~80 words | detailed: ~200+ words with recommendations",
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg="#A0A0A0",
        ).grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        row_num += 1

        # ─────────────────────────────────────────────────────────────────
        # MAX TOKENS
        # ─────────────────────────────────────────────────────────────────
        tk.Label(
            main_frame,
            text="Max Response Length (tokens):",
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=row_num, column=0, sticky=tk.W, pady=(10, 5))
        row_num += 1

        tokens_var = tk.IntVar(value=AI_MAX_TOKENS)
        tokens_spinbox = tk.Spinbox(
            main_frame,
            from_=50,
            to=1000,
            increment=50,
            textvariable=tokens_var,
            width=10,
            font=("Arial", 9),
        )
        tokens_spinbox.grid(row=row_num, column=0, sticky=tk.W, pady=(0, 5))
        row_num += 1

        tk.Label(
            main_frame,
            text="50-150: concise | 200-300: detailed | 400+: comprehensive",
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg="#A0A0A0",
        ).grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        row_num += 1

        # ─────────────────────────────────────────────────────────────────
        # GPT-5.2 ADVANCED SETTINGS SECTION
        # ─────────────────────────────────────────────────────────────────
        gpt5_section_label = tk.Label(
            main_frame,
            text="━━━ GPT-5.2 Advanced Settings ━━━",
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
        )
        gpt5_section_label.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(15, 10))
        row_num += 1

        tk.Label(
            main_frame,
            text="(Only applies when using GPT-5 models)",
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg="#808080",
        ).grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row_num += 1

        # Reasoning Effort
        tk.Label(
            main_frame,
            text="Reasoning Effort:",
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=row_num, column=0, sticky=tk.W, pady=(5, 5))
        row_num += 1

        reasoning_var = tk.StringVar(value=AI_REASONING_EFFORT)
        reasoning_levels = ["none", "low", "medium", "high", "xhigh"]

        reasoning_dropdown = ttk.Combobox(
            main_frame,
            textvariable=reasoning_var,
            values=reasoning_levels,
            state="readonly" if AI_MODEL.startswith("gpt-5") else "disabled",
            width=45,
            font=("Arial", 9),
        )
        reasoning_dropdown.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row_num += 1

        tk.Label(
            main_frame,
            text="none: fastest | low: light reasoning (recommended) | high/xhigh: deep analysis",
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg="#A0A0A0",
        ).grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row_num += 1

        # Text Verbosity
        tk.Label(
            main_frame,
            text="Text Verbosity:",
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
        ).grid(row=row_num, column=0, sticky=tk.W, pady=(5, 5))
        row_num += 1

        verbosity_var = tk.StringVar(value=AI_TEXT_VERBOSITY)
        verbosity_levels = ["auto", "low", "medium", "high"]

        verbosity_dropdown = ttk.Combobox(
            main_frame,
            textvariable=verbosity_var,
            values=verbosity_levels,
            state="readonly" if AI_MODEL.startswith("gpt-5") else "disabled",
            width=45,
            font=("Arial", 9),
        )
        verbosity_dropdown.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row_num += 1

        tk.Label(
            main_frame,
            text="auto: based on max tokens | low: terse | medium: balanced | high: comprehensive",
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg="#A0A0A0",
        ).grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row_num += 1

        # Reasoning Summary Checkbox
        reasoning_summary_var = tk.BooleanVar(value=AI_GENERATE_REASONING_SUMMARY)
        reasoning_summary_cb = tk.Checkbutton(
            main_frame,
            text="Include Reasoning Summary (shows model's thought process)",
            variable=reasoning_summary_var,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            selectcolor=DARK_BG_COLOR,
            activebackground=DARK_BG_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
            state=tk.NORMAL if AI_MODEL.startswith("gpt-5") else tk.DISABLED,
        )
        reasoning_summary_cb.grid(row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(5, 5))
        row_num += 1

        # Include Recommendations Checkbox
        include_recommendations_var = tk.BooleanVar(value=AI_INCLUDE_RECOMMENDATIONS)
        include_recommendations_cb = tk.Checkbutton(
            main_frame,
            text="Include Design Recommendations in Analysis",
            variable=include_recommendations_var,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            selectcolor=DARK_BG_COLOR,
            activebackground=DARK_BG_COLOR,
            activeforeground=LIGHT_TEXT_COLOR,
        )
        include_recommendations_cb.grid(
            row=row_num, column=0, columnspan=2, sticky=tk.W, pady=(5, 15)
        )
        row_num += 1

        # ─────────────────────────────────────────────────────────────────
        # INFO BOX
        # ─────────────────────────────────────────────────────────────────
        info_text = """These settings apply to:
• AI-powered report generation (Word documents)
• Real-time AI chat assistant
• Image caption generation

Settings are saved to config_local.py and take effect immediately."""

        info_label = tk.Label(
            scrollable_frame,
            text=info_text,
            font=("Arial", 8),
            bg="#3A3A3A",
            fg=LIGHT_TEXT_COLOR,
            justify=tk.LEFT,
            relief=tk.RIDGE,
            padx=10,
            pady=10,
        )
        info_label.pack(pady=15, padx=30, fill=tk.X)

        # ─────────────────────────────────────────────────────────────────
        # BUTTONS
        # ─────────────────────────────────────────────────────────────────
        button_frame = tk.Frame(scrollable_frame, bg=DARK_BG_COLOR)
        button_frame.pack(pady=15)

        def save_settings():
            # Update config_local.py
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config_local.py"
            )

            config_content = f"""# Auto-generated AI Settings
# Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Provider Selection
AI_PROVIDER = "{provider_var.get()}"
AI_MODEL = "{model_var.get()}"
AI_OLLAMA_URL = "{ollama_url_var.get()}"

# Response Configuration
AI_RESPONSE_STYLE = "{style_var.get()}"
AI_MAX_TOKENS = {tokens_var.get()}
AI_INCLUDE_RECOMMENDATIONS = {include_recommendations_var.get()}

# GPT-5.2 Responses API Settings
AI_REASONING_EFFORT = "{reasoning_var.get()}"
AI_TEXT_VERBOSITY = "{verbosity_var.get()}"
AI_GENERATE_REASONING_SUMMARY = {reasoning_summary_var.get()}
"""

            try:
                # Read existing config_local.py if it exists
                existing_content = ""
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        existing_content = f.read()

                # Remove old AI settings section if it exists
                import re

                existing_content = re.sub(
                    r"# Auto-generated AI Settings.*?(?=\n# [A-Z]|\Z)",
                    "",
                    existing_content,
                    flags=re.DOTALL,
                )

                # Also remove any standalone AI_ settings that might be floating
                ai_settings = [
                    "AI_PROVIDER",
                    "AI_MODEL",
                    "AI_OLLAMA_URL",
                    "AI_RESPONSE_STYLE",
                    "AI_MAX_TOKENS",
                    "AI_REASONING_EFFORT",
                    "AI_TEXT_VERBOSITY",
                    "AI_GENERATE_REASONING_SUMMARY",
                    "AI_INCLUDE_RECOMMENDATIONS",
                ]
                for setting in ai_settings:
                    existing_content = re.sub(
                        rf"^{setting}\s*=.*\n?", "", existing_content, flags=re.MULTILINE
                    )

                # Write new settings
                with open(config_path, "w", encoding="utf-8") as f:
                    final_content = existing_content.strip()
                    if final_content:
                        f.write(final_content + "\n\n")
                    f.write(config_content)

                messagebox.showinfo(
                    "Settings Saved",
                    f"AI settings saved successfully!\n\n"
                    f"Provider: {provider_var.get()}\n"
                    f"Model: {model_var.get()}\n"
                    f"Style: {style_var.get()}\n"
                    f"Max Tokens: {tokens_var.get()}\n\n"
                    "Changes will apply to new AI operations.\n"
                    "Restart the application for full effect.",
                )
                settings_window.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Could not save settings:\n{str(e)}")

        save_btn = tk.Button(
            button_frame,
            text="Save Settings",
            command=save_settings,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=15,
            pady=5,
        )
        save_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=settings_window.destroy,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=15,
            pady=5,
            relief=tk.RIDGE,
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                # Canvas was destroyed; clean up the global binding
                canvas.unbind_all("<MouseWheel>")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Cleanup mouse wheel binding when window is destroyed (any close path)
        def _on_destroy(event):
            if event.widget is settings_window:
                canvas.unbind_all("<MouseWheel>")

        settings_window.bind("<Destroy>", _on_destroy)

    # ────────────────────────────────────────────────────────────────────────
    # SCAN TYPE SETTINGS DIALOG
    # ────────────────────────────────────────────────────────────────────────

    def show_settings(self):
        """Show settings dialog based on current scan type."""
        scan_type_value = self.scan_type.get()
        settings_window = tk.Toplevel(self.root)
        settings_window.geometry("600x350")
        settings_window.title(f"{scan_type_value.capitalize()} Settings")
        settings_window.configure(bg=DARK_BG_COLOR)

        # ────────────────────────────────────
        #  ACTIVE  (TRP) SETTINGS
        # ────────────────────────────────────
        if scan_type_value == "active":
            tk.Label(
                settings_window, text="Active Plot Settings", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, columnspan=4, pady=20)

            # 3-D interpolation
            self.interpolate_var = tk.BooleanVar(value=self.interpolate_3d_plots)
            tk.Checkbutton(
                settings_window,
                text="Interpolate 3-D Plots",
                variable=self.interpolate_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=0, sticky=tk.W, padx=20)

            # Manual / auto Z-axis
            tk.Label(
                settings_window, text="3-D Z-Axis Scale:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=2, column=0, sticky=tk.W, padx=20)
            tk.Radiobutton(
                settings_window,
                text="Auto",
                variable=self.axis_scale_mode,
                value="auto",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=1, sticky=tk.W)
            tk.Radiobutton(
                settings_window,
                text="Manual",
                variable=self.axis_scale_mode,
                value="manual",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=2, sticky=tk.W)
            tk.Label(settings_window, text="Min dBm:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=3, column=0, sticky=tk.W, padx=20
            )
            tk.Entry(
                settings_window,
                textvariable=self.axis_min,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=3, column=1)
            tk.Label(settings_window, text="Max dBm:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=3, column=2, sticky=tk.W
            )
            tk.Entry(
                settings_window,
                textvariable=self.axis_max,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=3, column=3)

            # Maritime / Horizon plot settings
            maritime_frame = tk.LabelFrame(
                settings_window,
                text="Maritime / Horizon Plots",
                bg=DARK_BG_COLOR,
                fg=ACCENT_BLUE_COLOR,
                font=SECTION_HEADER_FONT,
            )
            maritime_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=15, pady=5)

            self.cb_maritime_var = tk.BooleanVar(value=getattr(self, "maritime_plots_enabled", False))
            tk.Checkbutton(
                maritime_frame,
                text="Enable Maritime Plots",
                variable=self.cb_maritime_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

            tk.Label(maritime_frame, text="Theta Min (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=0, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame, textvariable=self.horizon_theta_min, width=6,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5)
            tk.Label(maritime_frame, text="Theta Max (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=2, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame, textvariable=self.horizon_theta_max, width=6,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5)

            tk.Label(maritime_frame, text="Coverage Threshold (dB):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=2, column=0, columnspan=2, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame, textvariable=self.horizon_gain_threshold, width=6,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=2, padx=5)

            tk.Label(maritime_frame, text="Theta Cuts (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=3, column=0, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame, textvariable=self.horizon_theta_cuts_var, width=25,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

            def save_active_settings():
                self.interpolate_3d_plots = self.interpolate_var.get()
                self.maritime_plots_enabled = self.cb_maritime_var.get()
                self.update_visibility()
                settings_window.destroy()

            tk.Button(
                settings_window,
                text="Save Settings",
                command=save_active_settings,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=5, column=0, columnspan=4, pady=20)

        # ────────────────────────────────────
        #  PASSIVE  (HPOL/VPOL  or  G&D) SETTINGS
        # ────────────────────────────────────
        elif scan_type_value == "passive":
            tk.Label(
                settings_window, text="Passive Plot Settings", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, columnspan=4, pady=10)

            # VPOL/HPOL  vs  G&D
            self.plot_type_var = tk.StringVar(value=self.passive_scan_type.get())
            r_hv = tk.Radiobutton(
                settings_window,
                text="VPOL / HPOL",
                variable=self.plot_type_var,
                value="VPOL/HPOL",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )
            r_gd = tk.Radiobutton(
                settings_window,
                text="G&D",
                variable=self.plot_type_var,
                value="G&D",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )
            r_hv.grid(row=1, column=0, sticky=tk.W, padx=20)
            r_gd.grid(row=1, column=1, sticky=tk.W, padx=20)

            # Datasheet-style plots  (only for VPOL/HPOL)
            self.cb_datasheet_plots = tk.Checkbutton(
                settings_window,
                text="Datasheet Plots",
                variable=self.datasheet_plots_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )

            # ECC calculation  (only for VPOL/HPOL)
            self.cb_ecc_analysis_var = tk.BooleanVar(
                value=getattr(self, "ecc_analysis_enabled", False)
            )
            self.cb_ecc_analysis = tk.Checkbutton(
                settings_window,
                text="Calculate Envelope Correlation Coefficient (ECC)",
                variable=self.cb_ecc_analysis_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )

            # Min/Max Eff & Gain  (only for G&D)
            self.cb_min_max_eff_gain = tk.Checkbutton(
                settings_window,
                text="Min/Max Eff & Gain",
                variable=self.min_max_eff_gain_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )

            # Human-torso shadowing model
            self.cb_shadowing_var = tk.BooleanVar(value=getattr(self, "shadowing_enabled", False))
            tk.Checkbutton(
                settings_window,
                text="Apply Human Torso Shadow",
                variable=self.cb_shadowing_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=6, column=0, sticky=tk.W, padx=20)
            tk.Label(
                settings_window, text="Shadow Direction:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=6, column=1, sticky=tk.E)
            self.shadow_direction_var = tk.StringVar(value=getattr(self, "shadow_direction", "-X"))
            ttk.Combobox(
                settings_window,
                textvariable=self.shadow_direction_var,
                values=["+X", "-X"],
                width=4,
                state="readonly",
            ).grid(row=6, column=2)

            # 3-D axis controls (shared with Active logic)
            self.lbl_axis = tk.Label(
                settings_window, text="3-D Z-Axis Scale:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            )
            self.rb_axis_auto = tk.Radiobutton(
                settings_window,
                text="Auto",
                variable=self.axis_scale_mode,
                value="auto",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )
            self.rb_axis_man = tk.Radiobutton(
                settings_window,
                text="Manual",
                variable=self.axis_scale_mode,
                value="manual",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            )
            self.lbl_min_dbi = tk.Label(
                settings_window, text="Min dBi:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            )
            self.ent_min_dbi = tk.Entry(
                settings_window,
                textvariable=self.axis_min,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            )
            self.lbl_max_dbi = tk.Label(
                settings_window, text="Max dBi:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            )
            self.ent_max_dbi = tk.Entry(
                settings_window,
                textvariable=self.axis_max,
                width=6,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            )

            # put them in the grid now (we'll hide some later)
            self.lbl_axis.grid(row=3, column=0, sticky=tk.W, padx=20)
            self.rb_axis_auto.grid(row=3, column=1, sticky=tk.W)
            self.rb_axis_man.grid(row=3, column=2, sticky=tk.W)
            self.lbl_min_dbi.grid(row=4, column=0, sticky=tk.W, padx=20)
            self.ent_min_dbi.grid(row=4, column=1)
            self.lbl_max_dbi.grid(row=4, column=2, sticky=tk.W)
            self.ent_max_dbi.grid(row=4, column=3)

            # Extrapolation controls (VPOL/HPOL only)
            extrap_frame = tk.Frame(settings_window, bg=DARK_BG_COLOR)
            self.extrap_freq_var = tk.StringVar(value="")
            tk.Label(
                extrap_frame,
                text="Extrapolate to Freq (MHz):",
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=(0, 5))
            tk.Entry(
                extrap_frame,
                textvariable=self.extrap_freq_var,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).pack(side=tk.LEFT, padx=(0, 5))

            has_files = bool(self.hpol_file_path and self.vpol_file_path)

            def _do_extrapolate():
                val = self.extrap_freq_var.get().strip()
                if not val:
                    from tkinter import messagebox as mb

                    mb.showwarning("Input Required", "Enter a target frequency in MHz.")
                    return
                try:
                    freq = float(val)
                except ValueError:
                    from tkinter import messagebox as mb

                    mb.showerror("Invalid Input", "Frequency must be a number.")
                    return
                self._run_extrapolation(freq)

            extrap_btn = tk.Button(
                extrap_frame,
                text="Extrapolate",
                command=_do_extrapolate,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                state=tk.NORMAL if has_files else tk.DISABLED,
            )
            extrap_btn.pack(side=tk.LEFT)
            extrap_frame.grid(row=7, column=0, columnspan=4, sticky=tk.W, padx=20, pady=5)

            # Helper to show / hide controls depending on radio-selection
            def refresh_passive_ui():
                if self.plot_type_var.get() == "G&D":
                    # hide VPOL/HPOL-only controls
                    self.cb_datasheet_plots.grid_remove()
                    self.cb_ecc_analysis.grid_remove()
                    extrap_frame.grid_remove()
                    # show G&D-specific
                    self.cb_min_max_eff_gain.grid(row=2, column=1, sticky=tk.W, padx=20)
                else:  # VPOL/HPOL
                    self.cb_min_max_eff_gain.grid_remove()
                    self.cb_datasheet_plots.grid(row=2, column=0, sticky=tk.W, padx=20)
                    self.cb_ecc_analysis.grid(row=5, column=0, sticky=tk.W, padx=20)
                    extrap_frame.grid(row=7, column=0, columnspan=4, sticky=tk.W, padx=20, pady=5)

            # first run + connect
            refresh_passive_ui()
            r_hv.config(command=refresh_passive_ui)
            r_gd.config(command=refresh_passive_ui)

            # Maritime / Horizon plot settings
            maritime_frame_p = tk.LabelFrame(
                settings_window,
                text="Maritime / Horizon Plots",
                bg=DARK_BG_COLOR,
                fg=ACCENT_BLUE_COLOR,
                font=SECTION_HEADER_FONT,
            )
            maritime_frame_p.grid(row=8, column=0, columnspan=4, sticky="ew", padx=15, pady=5)

            self.cb_maritime_var = tk.BooleanVar(value=getattr(self, "maritime_plots_enabled", False))
            tk.Checkbutton(
                maritime_frame_p,
                text="Enable Maritime Plots",
                variable=self.cb_maritime_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

            tk.Label(maritime_frame_p, text="Theta Min (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=0, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame_p, textvariable=self.horizon_theta_min, width=6,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5)
            tk.Label(maritime_frame_p, text="Theta Max (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=2, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame_p, textvariable=self.horizon_theta_max, width=6,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5)

            tk.Label(maritime_frame_p, text="Coverage Threshold (dB):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=2, column=0, columnspan=2, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame_p, textvariable=self.horizon_gain_threshold, width=6,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=2, column=2, padx=5)

            tk.Label(maritime_frame_p, text="Theta Cuts (°):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=3, column=0, sticky=tk.W, padx=5
            )
            tk.Entry(
                maritime_frame_p, textvariable=self.horizon_theta_cuts_var, width=25,
                bg=SURFACE_COLOR, fg=LIGHT_TEXT_COLOR, insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

            # Save button
            def save_passive_settings():
                self.passive_scan_type.set(self.plot_type_var.get())
                self.ecc_analysis_enabled = self.cb_ecc_analysis_var.get()
                self.shadowing_enabled = self.cb_shadowing_var.get()
                self.shadow_direction = self.shadow_direction_var.get()
                self.maritime_plots_enabled = self.cb_maritime_var.get()
                self.update_visibility()
                settings_window.destroy()

            tk.Button(
                settings_window,
                text="Save Settings",
                command=save_passive_settings,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=9, column=0, columnspan=4, pady=20)

        elif scan_type_value == "vswr":
            # Show settings specific to VNA with organized LabelFrame sections
            title = tk.Label(
                settings_window,
                text="VSWR/Return Loss Settings",
                font=("Arial", 12, "bold"),
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            title.grid(row=0, column=0, columnspan=2, pady=10)

            def save_vswr_settings():
                try:
                    f1 = self.limit1_freq1.get()
                    f2 = self.limit1_freq2.get()
                    if f1 != 0 and f2 != 0 and f1 >= f2:
                        messagebox.showwarning(
                            "Invalid", "Limit 1: Freq Start must be less than Freq End"
                        )
                        return
                    f1_2 = self.limit2_freq1.get()
                    f2_2 = self.limit2_freq2.get()
                    if f1_2 != 0 and f2_2 != 0 and f1_2 >= f2_2:
                        messagebox.showwarning(
                            "Invalid", "Limit 2: Freq Start must be less than Freq End"
                        )
                        return
                    self.saved_limit1_freq1 = self.limit1_freq1.get()
                    self.saved_limit1_freq2 = self.limit1_freq2.get()
                    self.saved_limit1_start = self.limit1_val1.get()
                    self.saved_limit1_stop = self.limit1_val2.get()
                    self.saved_limit2_freq1 = self.limit2_freq1.get()
                    self.saved_limit2_freq2 = self.limit2_freq2.get()
                    self.saved_limit2_start = self.limit2_val1.get()
                    self.saved_limit2_stop = self.limit2_val2.get()
                    self.cb_groupdelay_sff = self.cb_groupdelay_sff_var.get()
                    self.saved_min_max_vswr = self.min_max_vswr_var.get()
                    settings_window.destroy()
                except tk.TclError:
                    messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

            def default_vswr_settings():
                for var in (
                    self.limit1_freq1,
                    self.limit1_freq2,
                    self.limit1_val1,
                    self.limit1_val2,
                    self.limit2_freq1,
                    self.limit2_freq2,
                    self.limit2_val1,
                    self.limit2_val2,
                ):
                    var.set(0.0)
                self.saved_limit1_freq1 = 0.0
                self.saved_limit1_freq2 = 0.0
                self.saved_limit1_start = 0.0
                self.saved_limit1_stop = 0.0
                self.saved_limit2_freq1 = 0.0
                self.saved_limit2_freq2 = 0.0
                self.saved_limit2_start = 0.0
                self.saved_limit2_stop = 0.0
                self.cb_groupdelay_sff_var.set(False)
                self.saved_min_max_vswr = False
                self.min_max_vswr_var.set(False)

            # Options section
            opts_frame = tk.LabelFrame(
                settings_window,
                text="Options",
                padx=10,
                pady=5,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            opts_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
            tk.Checkbutton(
                opts_frame,
                text="Group Delay & SFF",
                variable=self.cb_groupdelay_sff_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).pack(anchor=tk.W)
            tk.Checkbutton(
                opts_frame,
                text="Tabled Min/Max VSWR",
                variable=self.min_max_vswr_var,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
                selectcolor=SURFACE_COLOR,
                activebackground=DARK_BG_COLOR,
                activeforeground=LIGHT_TEXT_COLOR,
            ).pack(anchor=tk.W)
            if hasattr(self, "saved_min_max_vswr"):
                self.min_max_vswr_var.set(self.saved_min_max_vswr)

            # Limit Line 1 section
            limit1_frame = tk.LabelFrame(
                settings_window,
                text="Limit Line 1",
                padx=10,
                pady=5,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            limit1_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

            self.limit1_freq1 = tk.DoubleVar()
            self.limit1_freq2 = tk.DoubleVar()
            self.limit1_val1 = tk.DoubleVar()
            self.limit1_val2 = tk.DoubleVar()

            tk.Label(
                limit1_frame, text="Freq Start (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_freq1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=1, padx=5, pady=2)
            tk.Label(limit1_frame, text="Value Start:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=0, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_val1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=3, padx=5, pady=2)

            tk.Label(
                limit1_frame, text="Freq End (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_freq2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5, pady=2)
            tk.Label(limit1_frame, text="Value End:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit1_frame,
                textvariable=self.limit1_val2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5, pady=2)

            # Limit Line 2 section
            limit2_frame = tk.LabelFrame(
                settings_window,
                text="Limit Line 2",
                padx=10,
                pady=5,
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            limit2_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

            self.limit2_freq1 = tk.DoubleVar()
            self.limit2_freq2 = tk.DoubleVar()
            self.limit2_val1 = tk.DoubleVar()
            self.limit2_val2 = tk.DoubleVar()

            tk.Label(
                limit2_frame, text="Freq Start (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=0, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_freq1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=1, padx=5, pady=2)
            tk.Label(limit2_frame, text="Value Start:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=0, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_val1,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=0, column=3, padx=5, pady=2)

            tk.Label(
                limit2_frame, text="Freq End (GHz):", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
            ).grid(row=1, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_freq2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=1, padx=5, pady=2)
            tk.Label(limit2_frame, text="Value End:", bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR).grid(
                row=1, column=2, sticky="e", padx=5, pady=2
            )
            tk.Entry(
                limit2_frame,
                textvariable=self.limit2_val2,
                width=10,
                bg=SURFACE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                insertbackground=LIGHT_TEXT_COLOR,
            ).grid(row=1, column=3, padx=5, pady=2)

            # Restore saved values
            if hasattr(self, "saved_limit1_freq1"):
                self.limit1_freq1.set(self.saved_limit1_freq1)
                self.limit1_freq2.set(self.saved_limit1_freq2)
                self.limit1_val1.set(self.saved_limit1_start)
                self.limit1_val2.set(self.saved_limit1_stop)
                self.limit2_freq1.set(self.saved_limit2_freq1)
                self.limit2_freq2.set(self.saved_limit2_freq2)
                self.limit2_val1.set(self.saved_limit2_start)
                self.limit2_val2.set(self.saved_limit2_stop)

            # Buttons
            btn_frame = tk.Frame(settings_window, bg=DARK_BG_COLOR)
            btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
            tk.Button(
                btn_frame,
                text="Save",
                command=save_vswr_settings,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=12,
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                btn_frame,
                text="Defaults",
                command=default_vswr_settings,
                bg=BUTTON_COLOR,
                fg=LIGHT_TEXT_COLOR,
                width=12,
            ).pack(side=tk.LEFT, padx=5)
