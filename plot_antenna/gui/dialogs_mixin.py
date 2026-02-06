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
from ..api_keys import save_api_key as api_keys_save, delete_api_key as api_keys_delete, get_api_key

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

    # Method declarations for type checking only (not defined at runtime to avoid MRO conflicts)
    if TYPE_CHECKING:

        def resource_path(self, relative_path: str) -> str: ...
        def get_user_data_dir(self) -> str: ...
        def update_visibility(self) -> None: ...

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

    def load_api_key(self):
        """Load API key using centralized api_keys module.

        Uses secure storage (OS keyring) when available, falls back to:
        - User data file (~/.config/RFlect/.openai_key)
        - Environment variables (OPENAI_API_KEY, OPENAI_API_KEY2)
        - .env files (openai.env, openapi.env, .env)
        """
        return get_api_key()

    def save_api_key(self, api_key):
        """Save API key using centralized api_keys module.

        Uses secure storage (OS keyring/Windows Credential Manager) when available,
        falls back to base64-obfuscated file in user data directory.
        """
        return api_keys_save(api_key)

    def delete_api_key(self):
        """Delete stored API key using centralized api_keys module."""
        return api_keys_delete()

    def manage_api_key(self):
        """Show API key management dialog."""
        api_window = tk.Toplevel(self.root)
        api_window.title("Manage OpenAI API Key")
        api_window.geometry("550x350")
        api_window.resizable(False, False)
        api_window.configure(bg=DARK_BG_COLOR)

        # Center the window
        api_window.transient(self.root)
        api_window.grab_set()

        # Title
        title_label = tk.Label(
            api_window,
            text="OpenAI API Key Management",
            font=("Arial", 14, "bold"),
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
        )
        title_label.pack(pady=(20, 10))

        # Description
        desc_text = """The OpenAI API key enables AI-powered report generation features.
Your key is stored securely in your user data folder and never shared."""

        desc_label = tk.Label(
            api_window,
            text=desc_text,
            font=("Arial", 9),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            justify=tk.LEFT,
            wraplength=500,
        )
        desc_label.pack(pady=10, padx=20)

        # Current status
        current_key = self.load_api_key()
        status_text = "[OK] API Key is configured" if current_key else "[!] No API Key configured"
        status_color = "#4CAF50" if current_key else "#FFC107"

        status_label = tk.Label(
            api_window,
            text=status_text,
            font=("Arial", 10, "bold"),
            bg=DARK_BG_COLOR,
            fg=status_color,
        )
        status_label.pack(pady=10)

        # If key exists, show masked version
        if current_key:
            masked_key = (
                f"{current_key[:7]}...{current_key[-4:]}" if len(current_key) > 11 else "***"
            )
            masked_label = tk.Label(
                api_window,
                text=f"Current: {masked_key}",
                font=("Courier", 9),
                bg=DARK_BG_COLOR,
                fg=LIGHT_TEXT_COLOR,
            )
            masked_label.pack()

        # Input frame
        input_frame = tk.Frame(api_window, bg=DARK_BG_COLOR)
        input_frame.pack(pady=20, padx=20, fill=tk.X)

        tk.Label(
            input_frame, text="API Key:", font=("Arial", 10), bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR
        ).pack(anchor=tk.W)

        api_key_var = tk.StringVar(value=current_key if current_key else "")
        api_key_entry = tk.Entry(
            input_frame, textvariable=api_key_var, width=60, show="*", font=("Courier", 9)
        )
        api_key_entry.pack(fill=tk.X, pady=5)

        # Show/Hide button
        show_var = tk.BooleanVar(value=False)

        def toggle_show():
            if show_var.get():
                api_key_entry.config(show="")
                show_btn.config(text="Hide")
            else:
                api_key_entry.config(show="*")
                show_btn.config(text="Show")

        show_btn = tk.Button(
            input_frame,
            text="Show",
            command=toggle_show,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=8,
        )
        show_btn.pack(anchor=tk.W, pady=5)

        # Help text
        help_text = "Get your API key from: https://platform.openai.com/api-keys"
        help_label = tk.Label(
            api_window,
            text=help_text,
            font=("Arial", 8, "italic"),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            cursor="hand2",
        )
        help_label.pack(pady=5)
        help_label.bind(
            "<Button-1>", lambda e: webbrowser.open("https://platform.openai.com/api-keys")
        )

        # Buttons frame
        button_frame = tk.Frame(api_window, bg=DARK_BG_COLOR)
        button_frame.pack(pady=20)

        def save_key():
            key = api_key_var.get().strip()
            if not key:
                messagebox.showwarning("Empty Key", "Please enter an API key.")
                return

            if not key.startswith("sk-"):
                response = messagebox.askyesno(
                    "Invalid Format", "API key doesn't start with 'sk-'. Save anyway?"
                )
                if not response:
                    return

            if self.save_api_key(key):
                messagebox.showinfo(
                    "Success", "API key saved successfully!\n\nAI features are now enabled."
                )
                api_window.destroy()

        def delete_key():
            if messagebox.askyesno(
                "Confirm Delete", "Are you sure you want to delete the stored API key?"
            ):
                if self.delete_api_key():
                    messagebox.showinfo("Deleted", "API key has been removed.")
                    api_window.destroy()

        save_btn = tk.Button(
            button_frame,
            text="Save Key",
            command=save_key,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=12,
            pady=5,
        )
        save_btn.pack(side=tk.LEFT, padx=5)

        if current_key:
            delete_btn = tk.Button(
                button_frame,
                text="Delete Key",
                command=delete_key,
                bg="#F44336",
                fg=LIGHT_TEXT_COLOR,
                width=12,
                pady=5,
            )
            delete_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=api_window.destroy,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=12,
            pady=5,
            relief=tk.RIDGE,
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)

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
            ollama_url_frame, text="Ollama URL:", font=("Arial", 9),
            bg=DARK_BG_COLOR, fg=LIGHT_TEXT_COLOR,
        )
        ollama_url_label.pack(side=tk.LEFT, padx=(0, 5))

        try:
            from ..config import AI_OLLAMA_URL
        except ImportError:
            AI_OLLAMA_URL = "http://localhost:11434"

        ollama_url_var = tk.StringVar(value=AI_OLLAMA_URL)
        ollama_url_entry = tk.Entry(
            ollama_url_frame, textvariable=ollama_url_var, width=35, font=("Arial", 9),
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
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Cleanup mouse wheel binding when window closes
        def on_close():
            canvas.unbind_all("<MouseWheel>")
            settings_window.destroy()

        settings_window.protocol("WM_DELETE_WINDOW", on_close)

    # ────────────────────────────────────────────────────────────────────────
    # SCAN TYPE SETTINGS DIALOG
    # ────────────────────────────────────────────────────────────────────────

    def show_settings(self):
        """Show settings dialog based on current scan type."""
        scan_type_value = self.scan_type.get()
        settings_window = tk.Toplevel(self.root)
        settings_window.geometry("600x350")
        settings_window.title(f"{scan_type_value.capitalize()} Settings")

        # ────────────────────────────────────
        #  ACTIVE  (TRP) SETTINGS
        # ────────────────────────────────────
        if scan_type_value == "active":
            tk.Label(settings_window, text="Active Plot Settings").grid(
                row=0, column=0, columnspan=4, pady=20
            )

            # 3-D interpolation
            self.interpolate_var = tk.BooleanVar(value=self.interpolate_3d_plots)
            tk.Checkbutton(
                settings_window, text="Interpolate 3-D Plots", variable=self.interpolate_var
            ).grid(row=1, column=0, sticky=tk.W, padx=20)

            # Manual / auto Z-axis
            tk.Label(settings_window, text="3-D Z-Axis Scale:").grid(
                row=2, column=0, sticky=tk.W, padx=20
            )
            tk.Radiobutton(
                settings_window, text="Auto", variable=self.axis_scale_mode, value="auto"
            ).grid(row=2, column=1, sticky=tk.W)
            tk.Radiobutton(
                settings_window, text="Manual", variable=self.axis_scale_mode, value="manual"
            ).grid(row=2, column=2, sticky=tk.W)
            tk.Label(settings_window, text="Min dBm:").grid(row=3, column=0, sticky=tk.W, padx=20)
            tk.Entry(settings_window, textvariable=self.axis_min, width=6).grid(row=3, column=1)
            tk.Label(settings_window, text="Max dBm:").grid(row=3, column=2, sticky=tk.W)
            tk.Entry(settings_window, textvariable=self.axis_max, width=6).grid(row=3, column=3)

            def save_active_settings():
                self.interpolate_3d_plots = self.interpolate_var.get()
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
            tk.Label(settings_window, text="Passive Plot Settings").grid(
                row=0, column=0, columnspan=4, pady=10
            )

            # VPOL/HPOL  vs  G&D
            self.plot_type_var = tk.StringVar(value=self.passive_scan_type.get())
            r_hv = tk.Radiobutton(
                settings_window, text="VPOL / HPOL", variable=self.plot_type_var, value="VPOL/HPOL"
            )
            r_gd = tk.Radiobutton(
                settings_window, text="G&D", variable=self.plot_type_var, value="G&D"
            )
            r_hv.grid(row=1, column=0, sticky=tk.W, padx=20)
            r_gd.grid(row=1, column=1, sticky=tk.W, padx=20)

            # Datasheet-style plots  (only for VPOL/HPOL)
            self.cb_datasheet_plots = tk.Checkbutton(
                settings_window, text="Datasheet Plots", variable=self.datasheet_plots_var
            )

            # ECC calculation  (only for VPOL/HPOL)
            self.cb_ecc_analysis_var = tk.BooleanVar(
                value=getattr(self, "ecc_analysis_enabled", False)
            )
            self.cb_ecc_analysis = tk.Checkbutton(
                settings_window,
                text="Calculate Envelope Correlation Coefficient (ECC)",
                variable=self.cb_ecc_analysis_var,
            )

            # Min/Max Eff & Gain  (only for G&D)
            self.cb_min_max_eff_gain = tk.Checkbutton(
                settings_window, text="Min/Max Eff & Gain", variable=self.min_max_eff_gain_var
            )

            # Human-torso shadowing model
            self.cb_shadowing_var = tk.BooleanVar(value=getattr(self, "shadowing_enabled", False))
            tk.Checkbutton(
                settings_window, text="Apply Human Torso Shadow", variable=self.cb_shadowing_var
            ).grid(row=6, column=0, sticky=tk.W, padx=20)
            tk.Label(settings_window, text="Shadow Direction:").grid(row=6, column=1, sticky=tk.E)
            self.shadow_direction_var = tk.StringVar(value=getattr(self, "shadow_direction", "-X"))
            ttk.Combobox(
                settings_window,
                textvariable=self.shadow_direction_var,
                values=["+X", "-X"],
                width=4,
                state="readonly",
            ).grid(row=6, column=2)

            # 3-D axis controls (shared with Active logic)
            self.lbl_axis = tk.Label(settings_window, text="3-D Z-Axis Scale:")
            self.rb_axis_auto = tk.Radiobutton(
                settings_window, text="Auto", variable=self.axis_scale_mode, value="auto"
            )
            self.rb_axis_man = tk.Radiobutton(
                settings_window, text="Manual", variable=self.axis_scale_mode, value="manual"
            )
            self.lbl_min_dbi = tk.Label(settings_window, text="Min dBi:")
            self.ent_min_dbi = tk.Entry(settings_window, textvariable=self.axis_min, width=6)
            self.lbl_max_dbi = tk.Label(settings_window, text="Max dBi:")
            self.ent_max_dbi = tk.Entry(settings_window, textvariable=self.axis_max, width=6)

            # put them in the grid now (we'll hide some later)
            self.lbl_axis.grid(row=3, column=0, sticky=tk.W, padx=20)
            self.rb_axis_auto.grid(row=3, column=1, sticky=tk.W)
            self.rb_axis_man.grid(row=3, column=2, sticky=tk.W)
            self.lbl_min_dbi.grid(row=4, column=0, sticky=tk.W, padx=20)
            self.ent_min_dbi.grid(row=4, column=1)
            self.lbl_max_dbi.grid(row=4, column=2, sticky=tk.W)
            self.ent_max_dbi.grid(row=4, column=3)

            # Helper to show / hide controls depending on radio-selection
            def refresh_passive_ui():
                if self.plot_type_var.get() == "G&D":
                    # hide VPOL/HPOL-only controls
                    self.cb_datasheet_plots.grid_remove()
                    self.cb_ecc_analysis.grid_remove()
                    # show G&D-specific
                    self.cb_min_max_eff_gain.grid(row=2, column=1, sticky=tk.W, padx=20)
                else:  # VPOL/HPOL
                    self.cb_min_max_eff_gain.grid_remove()
                    self.cb_datasheet_plots.grid(row=2, column=0, sticky=tk.W, padx=20)
                    self.cb_ecc_analysis.grid(row=5, column=0, sticky=tk.W, padx=20)

            # first run + connect
            refresh_passive_ui()
            r_hv.config(command=refresh_passive_ui)
            r_gd.config(command=refresh_passive_ui)

            # Save button
            def save_passive_settings():
                self.passive_scan_type.set(self.plot_type_var.get())
                self.ecc_analysis_enabled = self.cb_ecc_analysis_var.get()
                self.shadowing_enabled = self.cb_shadowing_var.get()
                self.shadow_direction = self.shadow_direction_var.get()
                self.update_visibility()
                settings_window.destroy()

            tk.Button(
                settings_window,
                text="Save Settings",
                command=save_passive_settings,
                bg=ACCENT_BLUE_COLOR,
                fg=LIGHT_TEXT_COLOR,
            ).grid(row=8, column=0, columnspan=4, pady=20)

        elif scan_type_value == "vswr":
            # Show settings specific to VNA with organized LabelFrame sections
            title = tk.Label(
                settings_window, text="VSWR/Return Loss Settings",
                font=("Arial", 12, "bold"),
            )
            title.grid(row=0, column=0, columnspan=2, pady=10)

            def save_vswr_settings():
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

            def default_vswr_settings():
                for var in (self.limit1_freq1, self.limit1_freq2, self.limit1_val1,
                            self.limit1_val2, self.limit2_freq1, self.limit2_freq2,
                            self.limit2_val1, self.limit2_val2):
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
            opts_frame = tk.LabelFrame(settings_window, text="Options", padx=10, pady=5)
            opts_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
            tk.Checkbutton(
                opts_frame, text="Group Delay & SFF",
                variable=self.cb_groupdelay_sff_var,
            ).pack(anchor=tk.W)
            tk.Checkbutton(
                opts_frame, text="Tabled Min/Max VSWR",
                variable=self.min_max_vswr_var,
            ).pack(anchor=tk.W)
            if hasattr(self, "saved_min_max_vswr"):
                self.min_max_vswr_var.set(self.saved_min_max_vswr)

            # Limit Line 1 section
            limit1_frame = tk.LabelFrame(settings_window, text="Limit Line 1", padx=10, pady=5)
            limit1_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

            self.limit1_freq1 = tk.DoubleVar()
            self.limit1_freq2 = tk.DoubleVar()
            self.limit1_val1 = tk.DoubleVar()
            self.limit1_val2 = tk.DoubleVar()

            tk.Label(limit1_frame, text="Freq Start (GHz):").grid(row=0, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(limit1_frame, textvariable=self.limit1_freq1, width=10).grid(row=0, column=1, padx=5, pady=2)
            tk.Label(limit1_frame, text="Value Start:").grid(row=0, column=2, sticky="e", padx=5, pady=2)
            tk.Entry(limit1_frame, textvariable=self.limit1_val1, width=10).grid(row=0, column=3, padx=5, pady=2)

            tk.Label(limit1_frame, text="Freq End (GHz):").grid(row=1, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(limit1_frame, textvariable=self.limit1_freq2, width=10).grid(row=1, column=1, padx=5, pady=2)
            tk.Label(limit1_frame, text="Value End:").grid(row=1, column=2, sticky="e", padx=5, pady=2)
            tk.Entry(limit1_frame, textvariable=self.limit1_val2, width=10).grid(row=1, column=3, padx=5, pady=2)

            # Limit Line 2 section
            limit2_frame = tk.LabelFrame(settings_window, text="Limit Line 2", padx=10, pady=5)
            limit2_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

            self.limit2_freq1 = tk.DoubleVar()
            self.limit2_freq2 = tk.DoubleVar()
            self.limit2_val1 = tk.DoubleVar()
            self.limit2_val2 = tk.DoubleVar()

            tk.Label(limit2_frame, text="Freq Start (GHz):").grid(row=0, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(limit2_frame, textvariable=self.limit2_freq1, width=10).grid(row=0, column=1, padx=5, pady=2)
            tk.Label(limit2_frame, text="Value Start:").grid(row=0, column=2, sticky="e", padx=5, pady=2)
            tk.Entry(limit2_frame, textvariable=self.limit2_val1, width=10).grid(row=0, column=3, padx=5, pady=2)

            tk.Label(limit2_frame, text="Freq End (GHz):").grid(row=1, column=0, sticky="e", padx=5, pady=2)
            tk.Entry(limit2_frame, textvariable=self.limit2_freq2, width=10).grid(row=1, column=1, padx=5, pady=2)
            tk.Label(limit2_frame, text="Value End:").grid(row=1, column=2, sticky="e", padx=5, pady=2)
            tk.Entry(limit2_frame, textvariable=self.limit2_val2, width=10).grid(row=1, column=3, padx=5, pady=2)

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
            btn_frame = tk.Frame(settings_window)
            btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
            tk.Button(
                btn_frame, text="Save", command=save_vswr_settings,
                bg=ACCENT_BLUE_COLOR, fg=LIGHT_TEXT_COLOR, width=12,
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                btn_frame, text="Defaults", command=default_vswr_settings,
                bg=BUTTON_COLOR, fg=LIGHT_TEXT_COLOR, width=12,
            ).pack(side=tk.LEFT, padx=5)
