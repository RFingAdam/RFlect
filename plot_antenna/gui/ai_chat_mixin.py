"""
AIChatMixin - AI Chat functionality for RFlect GUI

This mixin provides AI-powered analysis capabilities:
- AI Chat Assistant window
- OpenAI API integration with function calling
- AI analysis functions (2D plot, 3D plot, gain statistics, pattern analysis, polarization comparison)
- Base64 image encoding for inline plot display
"""

from __future__ import annotations

import io
import json
import base64
import threading
import tkinter as tk
from typing import TYPE_CHECKING, Optional, List, Any

import numpy as np

from ..config import (
    DARK_BG_COLOR,
    LIGHT_TEXT_COLOR,
    ACCENT_BLUE_COLOR,
    BUTTON_COLOR,
    AI_MODEL,
    AI_MAX_TOKENS,
)

# Import AI settings with fallbacks
try:
    from ..config import AI_TEMPERATURE
except ImportError:
    AI_TEMPERATURE = 0.7

try:
    from ..config import AI_CHAT_MAX_TOKENS
except ImportError:
    AI_CHAT_MAX_TOKENS = 800

from ..api_keys import is_api_key_configured, get_api_key

from ..plotting import (
    plot_2d_passive_data,
    plot_passive_3d_component,
    plot_active_2d_data,
    plot_active_3d_data,
)
from ..ai_analysis import AntennaAnalyzer

if TYPE_CHECKING:
    from .base_protocol import AntennaPlotGUIProtocol


class AIChatMixin:
    """Mixin class providing AI Chat functionality for AntennaPlotGUI.

    Provides:
    - AI Chat window with OpenAI integration
    - Function calling for interactive data analysis
    - Base64 image encoding for inline plot display
    """

    # Type hints for IDE support (satisfied by main class)
    root: tk.Tk
    scan_type: tk.StringVar
    selected_frequency: tk.StringVar
    freq_list: List[float]
    theta_list: Any
    phi_list: Any
    h_gain_dB: Any
    v_gain_dB: Any
    total_gain_dB: Any
    hpol_far_field: Any
    vpol_far_field: Any
    hpol_file_path: Optional[str]
    vpol_file_path: Optional[str]
    TRP_file_path: Optional[str]
    datasheet_plots_var: tk.BooleanVar
    axis_scale_mode: tk.StringVar
    axis_min: tk.DoubleVar
    axis_max: tk.DoubleVar
    data_points: Any
    theta_angles_rad: Any
    phi_angles_rad: Any
    theta_angles_deg: Any
    phi_angles_deg: Any
    total_power_dBm_2d: Any
    phi_angles_deg_plot: Any
    total_power_dBm_2d_plot: Any
    _measurement_context: dict

    # Method declarations for type checking only (not defined at runtime to avoid MRO conflicts)
    if TYPE_CHECKING:

        def process_data(self) -> None: ...
        def _process_data_without_plotting(self) -> bool: ...

    # ────────────────────────────────────────────────────────────────────────
    # IMAGE ENCODING UTILITIES
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _figure_to_base64(fig, format: str = "png", dpi: int = 100) -> str:
        """Convert a matplotlib figure to a base64-encoded string.

        Args:
            fig: matplotlib Figure object
            format: Image format ('png', 'jpg', etc.)
            dpi: Resolution in dots per inch

        Returns:
            Base64-encoded string of the image
        """
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format=format,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return img_base64

    @staticmethod
    def _base64_to_photoimage(base64_str: str) -> Any:
        """Convert a base64-encoded string to a Tkinter PhotoImage.

        Args:
            base64_str: Base64-encoded image string

        Returns:
            PIL ImageTk.PhotoImage suitable for Tkinter
        """
        try:
            from PIL import Image, ImageTk

            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            return ImageTk.PhotoImage(img)
        except ImportError:
            print("[WARNING] PIL not available for inline image display")
            return None

    def _insert_image_in_chat(
        self, chat_text: tk.Text, base64_str: str, max_width: int = 400
    ) -> None:
        """Insert a base64-encoded image into a Tkinter Text widget.

        Args:
            chat_text: The Text widget to insert into
            base64_str: Base64-encoded image string
            max_width: Maximum width for the displayed image
        """
        try:
            from PIL import Image, ImageTk

            # Decode base64 to image
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))

            # Resize if too large
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)  # type: ignore[assignment]

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)

            # Store reference to prevent garbage collection
            if not hasattr(self, "_chat_images"):
                self._chat_images: List[Any] = []
            self._chat_images.append(photo)

            # Insert into text widget
            chat_text.config(state=tk.NORMAL)
            chat_text.image_create(tk.END, image=photo)
            chat_text.insert(tk.END, "\n\n")
            chat_text.config(state=tk.DISABLED)
            chat_text.see(tk.END)

        except ImportError:
            chat_text.config(state=tk.NORMAL)
            chat_text.insert(tk.END, "[Image display requires PIL/Pillow]\n\n")
            chat_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"[WARNING] Could not display inline image: {e}")

    def open_ai_chat(self):
        """Open AI chat assistant for real-time data analysis."""
        from tkinter import messagebox

        # Check if API key is configured for the selected provider
        try:
            from ..config import AI_PROVIDER
        except ImportError:
            AI_PROVIDER = "openai"

        if AI_PROVIDER in ("openai", "anthropic"):
            if not is_api_key_configured(AI_PROVIDER):
                provider_label = "OpenAI" if AI_PROVIDER == "openai" else "Anthropic"
                messagebox.showwarning(
                    "API Key Required",
                    f"Please configure your {provider_label} API key first.\n\n"
                    "Go to Tools -> Manage API Keys",
                )
                return
        elif AI_PROVIDER == "ollama":
            pass  # Ollama is local, no API key needed

        chat_window = tk.Toplevel(self.root)
        chat_window.title("AI Chat Assistant")
        chat_window.geometry("700x600")
        chat_window.configure(bg=DARK_BG_COLOR)

        # Title
        title_label = tk.Label(
            chat_window,
            text="AI Chat Assistant - Analyze Your Data",
            font=("Arial", 14, "bold"),
            bg=DARK_BG_COLOR,
            fg=ACCENT_BLUE_COLOR,
        )
        title_label.pack(pady=10)

        # Chat history display
        chat_frame = tk.Frame(chat_window, bg=DARK_BG_COLOR)
        chat_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        chat_text = tk.Text(
            chat_frame,
            wrap=tk.WORD,
            bg="#1E1E1E",
            fg=LIGHT_TEXT_COLOR,
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(chat_frame, command=chat_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        chat_text.config(yscrollcommand=scrollbar.set)

        # Initial context message built from rich measurement context
        data_summary = self._build_chat_context()
        context_msg = f"""Welcome to AI Chat Assistant!

I can help you analyze your antenna measurement data.

{data_summary}

Ask me anything about your measurements, patterns, or RF analysis!
"""

        chat_text.insert(tk.END, context_msg + "\n" + "=" * 80 + "\n\n")
        chat_text.config(state=tk.DISABLED)

        # Chat history for context
        chat_history = []

        def send_message():
            user_msg = user_input.get("1.0", tk.END).strip()
            if not user_msg:
                return

            # Display user message
            chat_text.config(state=tk.NORMAL)
            chat_text.insert(tk.END, f"You: {user_msg}\n\n", "user")
            chat_text.tag_config("user", foreground="#4A90E2", font=("Arial", 10, "bold"))
            chat_text.see(tk.END)
            chat_text.config(state=tk.DISABLED)

            # Clear input
            user_input.delete("1.0", tk.END)

            # Show typing indicator
            chat_text.config(state=tk.NORMAL)
            chat_text.insert(tk.END, "AI: Thinking...\n\n", "ai_thinking")
            chat_text.tag_config("ai_thinking", foreground="#FFC107", font=("Arial", 10, "italic"))
            chat_text.see(tk.END)
            chat_text.config(state=tk.DISABLED)

            # Disable input while AI is processing
            user_input.config(state=tk.DISABLED)
            send_btn.config(state=tk.DISABLED)

            # Capture context on main thread before spawning worker
            data_context = self._build_chat_context()
            chat_history.append({"role": "user", "content": user_msg})

            def _ai_worker():
                try:
                    response = self._get_ai_response(chat_history, data_context)
                    chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}\n\nPlease check your API key configuration and internet connection."
                    print(f"AI Chat Error: {e}")
                self.root.after(0, lambda r=response: _display_response(r))

            def _display_response(ai_response):
                # Remove thinking indicator
                chat_text.config(state=tk.NORMAL)
                chat_text.delete("end-3l", "end-1c")

                # Display AI response
                chat_text.insert(tk.END, f"AI: {ai_response}\n\n", "ai")
                chat_text.tag_config("ai", foreground="#4CAF50")
                chat_text.insert(tk.END, "=" * 80 + "\n\n")
                chat_text.see(tk.END)
                chat_text.config(state=tk.DISABLED)

                # Re-enable input
                user_input.config(state=tk.NORMAL)
                send_btn.config(state=tk.NORMAL)
                user_input.focus_set()

            thread = threading.Thread(target=_ai_worker, daemon=True)
            thread.start()

        # Quick action buttons
        action_frame = tk.Frame(chat_window, bg=DARK_BG_COLOR)
        action_frame.pack(padx=20, fill=tk.X)

        quick_actions = [
            ("Gain Stats", "Show me the gain statistics for the current frequency"),
            ("Pattern", "Analyze the radiation pattern"),
            ("Polarization", "Compare H-pol and V-pol"),
            ("All Freqs", "Analyze performance across all frequencies"),
        ]

        def _quick_send(prompt_text):
            user_input.delete("1.0", tk.END)
            user_input.insert("1.0", prompt_text)
            send_message()

        for label, prompt in quick_actions:
            btn = tk.Button(
                action_frame,
                text=label,
                font=("Arial", 8),
                bg=BUTTON_COLOR,
                fg=LIGHT_TEXT_COLOR,
                command=lambda p=prompt: _quick_send(p),
            )
            btn.pack(side=tk.LEFT, padx=3, pady=3)

        def _clear_chat():
            chat_text.config(state=tk.NORMAL)
            chat_text.delete("1.0", tk.END)
            fresh_ctx = self._build_chat_context()
            fresh_msg = (
                f"Welcome to AI Chat Assistant!\n\n{fresh_ctx}\n\nChat cleared. Ask me anything!\n"
            )
            chat_text.insert(tk.END, fresh_msg + "\n" + "=" * 80 + "\n\n")
            chat_text.config(state=tk.DISABLED)
            chat_history.clear()

        clear_btn = tk.Button(
            action_frame,
            text="Clear",
            font=("Arial", 8),
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            command=_clear_chat,
        )
        clear_btn.pack(side=tk.RIGHT, padx=3, pady=3)

        # Input frame with text box and send button
        input_frame = tk.Frame(chat_window, bg=DARK_BG_COLOR)
        input_frame.pack(pady=10, padx=20, fill=tk.X, side=tk.BOTTOM)

        # Send button (pack first so it stays on right)
        send_btn = tk.Button(
            input_frame,
            text="Send",
            command=send_message,
            bg=ACCENT_BLUE_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=10,
            height=2,
        )
        send_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # User input text box
        user_input = tk.Text(
            input_frame,
            height=3,
            wrap=tk.WORD,
            bg="#2E2E2E",
            fg=LIGHT_TEXT_COLOR,
            font=("Arial", 10),
            relief=tk.SOLID,
            borderwidth=1,
            padx=5,
            pady=5,
        )
        user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind Enter key to send
        user_input.bind("<Shift-Return>", lambda e: "break")  # Shift+Enter for newline
        user_input.bind("<Return>", lambda e: (send_message(), "break"))  # Enter to send

    def _build_chat_context(self):
        """Build rich context string from measurement data for AI awareness."""
        ctx = self._measurement_context
        parts = []

        # Scan type
        scan = ctx.get("scan_type") or self.scan_type.get()
        parts.append(f"Scan Type: {scan}")

        # Loaded files
        if ctx.get("files_loaded"):
            parts.append("Loaded Files:")
            for f in ctx["files_loaded"]:
                parts.append(f"  - {f['path']} ({f['type']})")

        # Frequencies
        freqs = ctx.get("frequencies") or []
        if not freqs and hasattr(self, "freq_list") and self.freq_list:
            freqs = list(self.freq_list)
        if freqs:
            parts.append(
                f"Frequencies: {len(freqs)} points, " f"{min(freqs):.1f} - {max(freqs):.1f} MHz"
            )

        # Selected frequency
        if hasattr(self, "selected_frequency") and self.selected_frequency.get():
            try:
                freq = float(self.selected_frequency.get())
                parts.append(f"Selected Frequency: {freq:.1f} MHz")
            except (ValueError, AttributeError, TypeError):
                pass

        # Data shape
        if ctx.get("data_shape"):
            parts.append(f"Data Shape: {ctx['data_shape']}")

        # Cable loss
        if ctx.get("cable_loss_applied"):
            parts.append(f"Cable Loss: {ctx['cable_loss_applied']} dB applied")

        # NF2FF
        if ctx.get("nf2ff_applied"):
            parts.append("NF2FF Transformation: Applied")

        # Processing status
        if ctx.get("processing_complete"):
            parts.append("Processing: Complete")

        # Key metrics
        if ctx.get("key_metrics"):
            parts.append("Key Metrics:")
            for k, v in ctx["key_metrics"].items():
                parts.append(f"  - {k}: {v}")

        return "\n".join(parts) if parts else "No data currently loaded"

    def _get_ai_response(self, chat_history, data_context):
        """Get AI response using the configured LLM provider with tool calling.

        Uses the unified provider abstraction layer to support OpenAI, Anthropic, and Ollama.
        """
        from ..llm_provider import LLMMessage, ToolDefinition

        # Get provider configuration
        try:
            from ..config import AI_PROVIDER
        except ImportError:
            AI_PROVIDER = "openai"

        # Create provider instance
        try:
            provider = self._create_chat_provider(AI_PROVIDER)
        except (ValueError, ImportError) as e:
            return f"Error initializing AI provider '{AI_PROVIDER}': {str(e)}"

        # Define available functions that AI can call
        available_functions = {
            "generate_2d_plot": self._ai_generate_2d_plot,
            "generate_3d_plot": self._ai_generate_3d_plot,
            "get_gain_statistics": self._ai_get_gain_statistics,
            "analyze_pattern": self._ai_analyze_pattern,
            "compare_polarizations": self._ai_compare_polarizations,
            "analyze_all_frequencies": self._ai_analyze_all_frequencies,
        }

        # Build tool definitions (unified format for all providers)
        tool_defs = [
            ToolDefinition(
                name="generate_2d_plot",
                description="Generate and DISPLAY 2D radiation pattern plots (polar plots showing azimuth and elevation cuts with H-pol, V-pol, and total gain).",
                parameters={
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "number", "description": "Frequency in MHz to plot."},
                        "plot_type": {
                            "type": "string",
                            "enum": ["polar", "rectangular", "azimuth_cuts"],
                            "description": "Type of 2D plot to generate",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="generate_3d_plot",
                description="Generate and DISPLAY a 3D radiation pattern plot showing the full spatial distribution.",
                parameters={
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "number", "description": "Frequency in MHz"},
                        "component": {
                            "type": "string",
                            "enum": ["total", "hpol", "vpol"],
                            "description": "Which polarization component to plot",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="get_gain_statistics",
                description="Calculate and return gain statistics (max, min, average, variance) for the current data at a specified frequency.",
                parameters={
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "number", "description": "Frequency in MHz."},
                    },
                },
            ),
            ToolDefinition(
                name="analyze_pattern",
                description="Analyze the radiation pattern characteristics including pattern type, HPBW, front-to-back ratio, null detection, and beam direction.",
                parameters={
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz to analyze",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="compare_polarizations",
                description="Compare HPOL and VPOL performance including XPD, polarization balance, and dominant polarization.",
                parameters={
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz to compare",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="analyze_all_frequencies",
                description="Analyze gain/power trends across ALL measured frequencies. Returns resonance frequency, 3dB bandwidth, gain variation, and per-frequency peak gains.",
                parameters={"type": "object", "properties": {}},
            ),
        ]

        # Build enhanced system message
        system_message = f"""You are a Principal RF Engineer with PhD-level expertise in antenna theory, electromagnetics, and wireless communications. You provide detailed, technically accurate analysis using proper terminology and physical insights.

Current Measurement Data:
{data_context}

**Available Analysis Functions:**
You can call these functions to help answer user questions:
1. generate_2d_plot(frequency) - Display 2D polar plots showing azimuth/elevation cuts with H-pol, V-pol, and total gain
2. generate_3d_plot(frequency, component) - Display interactive 3D radiation pattern
3. get_gain_statistics(frequency) - Calculate min/max/average gain values at a frequency
4. analyze_pattern(frequency) - Analyze HPBW, front-to-back ratio, null detection, pattern type, beam direction
5. compare_polarizations(frequency) - Compare HPOL vs VPOL: XPD, polarization balance
6. analyze_all_frequencies() - Analyze gain/power trends across ALL frequencies

**Analysis Guidelines:**
- Use proper units and terminology (dBi, HPBW, F/B ratio, XPD)
- Interpret function results with PhD-level technical depth
- Compare results against theoretical expectations (dipole ~2.15 dBi, isotropic = 0 dBi)
- After calling plot functions, interpret the returned analysis JSON professionally
"""

        # Build messages in unified format
        messages = [LLMMessage(role="system", content=system_message)]
        for msg in chat_history:
            messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

        max_tokens_value = AI_CHAT_MAX_TOKENS
        temperature_value = AI_TEMPERATURE if "AI_TEMPERATURE" in globals() else 0.2

        try:
            # Determine tools to pass (only if provider supports them)
            tools = tool_defs if provider.supports_tools() else None

            # Initial API call
            response = provider.chat(
                messages,
                tools=tools,
                max_tokens=max_tokens_value,
                temperature=temperature_value,
            )

            print(f"[AI Debug] Provider: {provider.provider_name()}, Stop: {response.stop_reason}")

            # Handle tool call loop (provider-agnostic)
            max_iterations = 5
            iteration = 0
            while response.tool_calls and iteration < max_iterations:
                iteration += 1
                for tc in response.tool_calls:
                    print(f"[AI] Calling function: {tc.name}({tc.arguments})")

                    if tc.name in available_functions:
                        try:
                            func_result = available_functions[tc.name](**tc.arguments)
                        except Exception as func_error:
                            func_result = json.dumps({"error": str(func_error)})
                    else:
                        func_result = json.dumps({"error": f"Unknown function: {tc.name}"})

                    # Append assistant tool call + tool result
                    messages.append(LLMMessage(role="assistant", tool_calls=[tc]))
                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=str(func_result),
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                        )
                    )

                # Follow-up call with tool results
                response = provider.chat(
                    messages,
                    tools=tools,
                    max_tokens=max_tokens_value,
                    temperature=temperature_value,
                )

            if response.content and response.content.strip():
                return response.content.strip()

            return "Analysis complete."

        except Exception as e:
            raise Exception(f"AI provider error ({AI_PROVIDER}): {str(e)}")

    def _create_chat_provider(self, provider_name):
        """Create a provider instance for the AI chat based on current config."""
        from ..llm_provider import create_provider

        if provider_name == "openai":
            api_key = get_api_key("openai")
            if not api_key:
                raise ValueError("OpenAI API key not found. Configure in Tools -> Manage API Keys.")
            model = AI_MODEL if "AI_MODEL" in globals() else "gpt-4o-mini"
            return create_provider("openai", api_key=api_key, model=model)

        elif provider_name == "anthropic":
            api_key = get_api_key("anthropic")
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found. Configure in Tools -> Manage API Keys."
                )
            try:
                from ..config import AI_ANTHROPIC_MODEL
            except ImportError:
                AI_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            return create_provider("anthropic", api_key=api_key, model=AI_ANTHROPIC_MODEL)

        elif provider_name == "ollama":
            try:
                from ..config import AI_OLLAMA_MODEL, AI_OLLAMA_URL
            except ImportError:
                AI_OLLAMA_MODEL = "llama3.1"
                AI_OLLAMA_URL = "http://localhost:11434"
            return create_provider("ollama", model=AI_OLLAMA_MODEL, base_url=AI_OLLAMA_URL)

        else:
            raise ValueError(f"Unknown AI provider: {provider_name}")

    # ────────────────────────────────────────────────────────────────────────
    # AI FUNCTION IMPLEMENTATIONS
    # ────────────────────────────────────────────────────────────────────────

    def _build_analyzer(self):
        """Build an AntennaAnalyzer from current GUI state."""
        scan_type = self.scan_type.get()
        data = {}

        if scan_type == "passive":
            theta = getattr(self, "theta_list", None)
            phi = getattr(self, "phi_list", None)
            # GUI stores 2D arrays (points x freqs); AntennaAnalyzer expects 1D angles
            if theta is not None and hasattr(theta, "ndim") and theta.ndim == 2:
                theta = theta[:, 0]
            if phi is not None and hasattr(phi, "ndim") and phi.ndim == 2:
                phi = phi[:, 0]
            data["phi"] = phi
            data["theta"] = theta
            data["total_gain"] = getattr(self, "total_gain_dB", None)
            data["h_gain"] = getattr(self, "h_gain_dB", None)
            data["v_gain"] = getattr(self, "v_gain_dB", None)
        elif scan_type == "active":
            data["phi"] = getattr(self, "phi_angles_deg", None)
            data["theta"] = getattr(self, "theta_angles_deg", None)
            data["total_power"] = getattr(self, "total_power_dBm_2d", None)
            if hasattr(self, "TRP_dBm"):
                data["TRP_dBm"] = getattr(self, "TRP_dBm", 0)
            if hasattr(self, "h_TRP_dBm"):
                data["h_TRP_dBm"] = getattr(self, "h_TRP_dBm", 0)
            if hasattr(self, "v_TRP_dBm"):
                data["v_TRP_dBm"] = getattr(self, "v_TRP_dBm", 0)

        frequencies = list(self.freq_list) if hasattr(self, "freq_list") and self.freq_list else []
        return AntennaAnalyzer(data, scan_type=scan_type, frequencies=frequencies)

    def _ai_generate_2d_plot(self, frequency=None, plot_type="polar"):
        """Generate and display 2D radiation pattern plot."""
        try:
            # Use selected frequency if not specified
            if frequency is None and hasattr(self, "selected_frequency"):
                try:
                    frequency = float(self.selected_frequency.get())
                except (ValueError, AttributeError, TypeError):
                    return json.dumps({"error": "No frequency specified and no valid selection"})

            # Check if files have been imported based on scan type
            scan_type = str(self.scan_type.get()) if hasattr(self, "scan_type") else None
            has_passive_data = hasattr(self, "hpol_file_path") and self.hpol_file_path
            has_active_data = hasattr(self, "TRP_file_path") and self.TRP_file_path

            if scan_type == "passive" and not has_passive_data:
                return json.dumps(
                    {
                        "error": "No passive measurement data loaded. Please import HPOL and VPOL files first."
                    }
                )
            elif scan_type == "active" and not has_active_data:
                return json.dumps(
                    {"error": "No active measurement data loaded. Please import a TRP file first."}
                )
            elif not has_passive_data and not has_active_data:
                return json.dumps(
                    {"error": "No measurement data loaded. Please import antenna data files first."}
                )

            # Update selected frequency and process data
            if hasattr(self, "selected_frequency"):
                self.selected_frequency.set(str(frequency))

            # Process data at the requested frequency
            try:
                self.process_data()
            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                print(f"[AI Error] 2D plot data processing failed:\n{error_details}")
                return json.dumps({"error": f"Failed to process data: {str(e)}"})

            # Check if we have loaded data after processing
            if not hasattr(self, "freq_list") or not self.freq_list:
                return json.dumps(
                    {
                        "error": "No frequency data available. The data files may not contain frequency information."
                    }
                )

            # Find closest frequency
            freq_array = np.array(self.freq_list)
            freq_idx = np.argmin(np.abs(freq_array - frequency))
            actual_freq = float(self.freq_list[freq_idx])

            # Update the selected frequency in the GUI
            if hasattr(self, "selected_frequency"):
                self.selected_frequency.set(str(actual_freq))

            # Actually generate the plot based on scan type
            if str(self.scan_type.get()) == "passive":
                # Get required data
                theta_angles_deg = getattr(self, "theta_list", None)
                phi_angles_deg = getattr(self, "phi_list", None)
                h_gain_dB = getattr(self, "h_gain_dB", None)
                v_gain_dB = getattr(self, "v_gain_dB", None)
                total_gain_dB = getattr(self, "total_gain_dB", None)
                hpol_far_field = getattr(self, "hpol_far_field", None)
                vpol_far_field = getattr(self, "vpol_far_field", None)
                datasheet_plots = getattr(self, "datasheet_plots_var", None)

                print(
                    f"[AI Debug 2D] Data check: theta={theta_angles_deg is not None}, phi={phi_angles_deg is not None}, h_gain={h_gain_dB is not None}, v_gain={v_gain_dB is not None}, total={total_gain_dB is not None}, hpol_ff={hpol_far_field is not None}, vpol_ff={vpol_far_field is not None}"
                )

                if all(
                    x is not None
                    for x in [
                        theta_angles_deg,
                        phi_angles_deg,
                        h_gain_dB,
                        v_gain_dB,
                        total_gain_dB,
                        hpol_far_field,
                        vpol_far_field,
                    ]
                ):
                    # Call the actual plotting function (NOW it will display plots)
                    plot_2d_passive_data(
                        theta_angles_deg,
                        phi_angles_deg,
                        hpol_far_field,
                        vpol_far_field,
                        total_gain_dB,
                        self.freq_list,
                        actual_freq,
                        datasheet_plots.get() if datasheet_plots else False,
                    )

                    # Try to get detailed analysis, but fall back to simple response if it fails
                    try:
                        # Type assertions for numpy arrays (validated by if-check above)
                        assert total_gain_dB is not None
                        assert h_gain_dB is not None
                        assert v_gain_dB is not None
                        assert theta_angles_deg is not None
                        assert phi_angles_deg is not None

                        # Get gain stats for response
                        gain_at_freq = total_gain_dB[:, freq_idx]
                        max_gain = float(np.max(gain_at_freq))
                        min_gain = float(np.min(gain_at_freq))
                        avg_gain = float(np.mean(gain_at_freq))

                        # Find peak location
                        max_idx = np.argmax(gain_at_freq)
                        peak_theta = float(theta_angles_deg[max_idx, freq_idx])
                        peak_phi = float(phi_angles_deg[max_idx, freq_idx])

                        # Analyze H-pol vs V-pol
                        h_gain_at_freq = h_gain_dB[:, freq_idx]
                        v_gain_at_freq = v_gain_dB[:, freq_idx]
                        h_max = float(np.max(h_gain_at_freq))
                        v_max = float(np.max(v_gain_at_freq))
                        cross_pol_ratio = h_max - v_max

                        # Pattern analysis
                        gain_variation = max_gain - min_gain
                        if gain_variation < 5:
                            pattern_type = "omnidirectional"
                        elif gain_variation < 10:
                            pattern_type = "quasi-omnidirectional"
                        else:
                            pattern_type = "directional"

                        return json.dumps(
                            {
                                "success": True,
                                "action": "2D polar plots displayed (azimuth and elevation cuts)",
                                "frequency_MHz": actual_freq,
                                "analysis": {
                                    "pattern_type": pattern_type,
                                    "peak_gain_dBi": round(max_gain, 2),
                                    "peak_location": {
                                        "theta_deg": round(peak_theta, 1),
                                        "phi_deg": round(peak_phi, 1),
                                    },
                                    "min_gain_dBi": round(min_gain, 2),
                                    "average_gain_dBi": round(avg_gain, 2),
                                    "gain_variation_dB": round(gain_variation, 2),
                                    "polarization": {
                                        "h_pol_peak_dBi": round(h_max, 2),
                                        "v_pol_peak_dBi": round(v_max, 2),
                                        "cross_pol_ratio_dB": round(cross_pol_ratio, 2),
                                        "dominant": "H-pol" if h_max > v_max else "V-pol",
                                    },
                                    "assessment": f"This {'omnidirectional' if gain_variation < 5 else 'directional'} antenna exhibits a peak gain of {max_gain:.1f} dBi at theta={peak_theta:.0f} deg, phi={peak_phi:.0f} deg. The {gain_variation:.1f} dB variation indicates {'excellent azimuthal uniformity' if gain_variation < 3 else 'moderate directivity' if gain_variation < 10 else 'strong directivity'}. {'H-pol dominates with ' + str(abs(round(cross_pol_ratio, 1))) + ' dB higher peak gain' if cross_pol_ratio > 2 else 'V-pol dominates with ' + str(abs(round(cross_pol_ratio, 1))) + ' dB higher peak gain' if cross_pol_ratio < -2 else 'Balanced polarization characteristics observed'}.",
                                },
                            }
                        )
                    except Exception as analysis_error:
                        print(
                            f"[AI Warning] Could not generate detailed analysis: {analysis_error}"
                        )
                        # Return simple success message if analysis fails
                        return json.dumps(
                            {
                                "success": True,
                                "action": "2D polar plots displayed successfully",
                                "frequency_MHz": actual_freq,
                                "message": "Plots displayed showing azimuth and elevation cuts with H-pol, V-pol, and total gain patterns. The plots are now visible in separate windows for detailed inspection.",
                            }
                        )
                else:
                    return json.dumps({"error": "Required passive scan data not available"})

            elif str(self.scan_type.get()) == "active":
                # Get active scan data
                data_points = getattr(self, "data_points", None)
                theta_angles_rad = getattr(self, "theta_angles_rad", None)
                phi_angles_rad = getattr(self, "phi_angles_rad", None)
                total_power_dBm_2d = getattr(self, "total_power_dBm_2d", None)

                if all(
                    x is not None
                    for x in [data_points, theta_angles_rad, phi_angles_rad, total_power_dBm_2d]
                ):
                    plot_active_2d_data(
                        data_points,
                        theta_angles_rad,
                        phi_angles_rad,
                        total_power_dBm_2d,
                        actual_freq,
                    )
                    return json.dumps(
                        {
                            "success": True,
                            "action": "2D azimuth power cuts displayed",
                            "frequency": actual_freq,
                        }
                    )
                else:
                    return json.dumps({"error": "Required active scan data not available"})
            else:
                return json.dumps({"error": f"Unknown scan type: {self.scan_type.get()}"})

        except Exception as e:
            return json.dumps({"error": f"Failed to generate 2D plot: {str(e)}"})

    def _ai_generate_3d_plot(self, frequency=None, component="total"):
        """Generate and display 3D radiation pattern plot."""
        print(
            f"[AI Debug] _ai_generate_3d_plot called with frequency={frequency}, component={component}"
        )
        try:
            if frequency is None and hasattr(self, "selected_frequency"):
                try:
                    frequency = float(self.selected_frequency.get())
                except (ValueError, AttributeError, TypeError):
                    return json.dumps({"error": "No frequency specified"})

            print(f"[AI Debug] Target frequency: {frequency} MHz")

            # Check if files have been imported based on scan type
            scan_type = str(self.scan_type.get()) if hasattr(self, "scan_type") else None
            has_passive_data = hasattr(self, "hpol_file_path") and self.hpol_file_path
            has_active_data = hasattr(self, "TRP_file_path") and self.TRP_file_path

            print(
                f"[AI Debug] Scan type: {scan_type}, has_passive: {has_passive_data}, has_active: {has_active_data}"
            )
            if has_active_data:
                print(f"[AI Debug] TRP file: {self.TRP_file_path}")

            if scan_type == "passive" and not has_passive_data:
                return json.dumps(
                    {
                        "error": "No passive measurement data loaded. Please import HPOL and VPOL files first."
                    }
                )
            elif scan_type == "active" and not has_active_data:
                return json.dumps(
                    {"error": "No active measurement data loaded. Please import a TRP file first."}
                )
            elif scan_type is None:
                return json.dumps(
                    {"error": "Scan type not set. Please select a scan type and import data."}
                )
            elif not has_passive_data and not has_active_data:
                return json.dumps(
                    {"error": "No measurement data loaded. Please import antenna data files first."}
                )

            print(
                f"[AI Debug] Files loaded - Scan type: {scan_type}, Passive: {has_passive_data}, Active: {has_active_data}"
            )

            # Update selected frequency and process data
            if hasattr(self, "selected_frequency"):
                self.selected_frequency.set(str(frequency))

            # Process data WITHOUT automatically showing plots
            try:
                print(f"[AI Debug] Calling _process_data_without_plotting...")
                success = self._process_data_without_plotting()
                print(f"[AI Debug] Processing result: {success}")
                if not success:
                    return json.dumps({"error": "Failed to process measurement data"})
            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                print(f"[AI Error] 3D plot data processing failed:\n{error_details}")
                return json.dumps({"error": f"Failed to process data: {str(e)}"})

            # Check if we have loaded data after processing
            if not hasattr(self, "freq_list") or not self.freq_list:
                return json.dumps(
                    {
                        "error": "No frequency data available. The data files may not contain frequency information."
                    }
                )

            print(f"[AI Debug] Freq list has {len(self.freq_list)} frequencies")
            print(
                f"[AI Debug] Freq range: {min(self.freq_list):.1f} - {max(self.freq_list):.1f} MHz"
            )

            # Find closest frequency
            freq_array = np.array(self.freq_list)
            freq_idx = np.argmin(np.abs(freq_array - frequency))
            actual_freq = float(self.freq_list[freq_idx])

            print(
                f"[AI Debug] Requested: {frequency} MHz, Actual: {actual_freq} MHz, Index: {freq_idx}"
            )

            # Update the selected frequency in the GUI
            if hasattr(self, "selected_frequency"):
                self.selected_frequency.set(str(actual_freq))

            # Generate 3D plot based on scan type
            if str(self.scan_type.get()) == "passive":
                print(f"[AI Debug] Scan type: passive")
                theta_angles_deg = getattr(self, "theta_list", None)
                phi_angles_deg = getattr(self, "phi_list", None)
                h_gain_dB = getattr(self, "h_gain_dB", None)
                v_gain_dB = getattr(self, "v_gain_dB", None)
                total_gain_dB = getattr(self, "total_gain_dB", None)
                hpol_far_field = getattr(self, "hpol_far_field", None)
                vpol_far_field = getattr(self, "vpol_far_field", None)

                print(
                    f"[AI Debug] Data arrays: theta={theta_angles_deg is not None}, phi={phi_angles_deg is not None}, h={h_gain_dB is not None}, v={v_gain_dB is not None}, total={total_gain_dB is not None}, hpol_ff={hpol_far_field is not None}, vpol_ff={vpol_far_field is not None}"
                )

                # Get axis mode settings
                axis_mode = getattr(self, "axis_scale_mode", None)
                zmin_var = getattr(self, "axis_min", None)
                zmax_var = getattr(self, "axis_max", None)

                if all(
                    x is not None
                    for x in [
                        theta_angles_deg,
                        phi_angles_deg,
                        h_gain_dB,
                        v_gain_dB,
                        total_gain_dB,
                        hpol_far_field,
                        vpol_far_field,
                    ]
                ):
                    # Perform analysis BEFORE plotting
                    try:
                        # Type assertions for numpy arrays (validated by if-check above)
                        assert total_gain_dB is not None
                        assert h_gain_dB is not None
                        assert v_gain_dB is not None
                        assert theta_angles_deg is not None
                        assert phi_angles_deg is not None

                        print(
                            f"[AI Debug] Starting pre-plot analysis for component={component}, freq_idx={freq_idx}"
                        )
                        # Get peak location and detailed analysis
                        if component == "total":
                            gain_data = total_gain_dB[:, freq_idx]
                        elif component == "hpol":
                            gain_data = h_gain_dB[:, freq_idx]
                        else:  # vpol
                            gain_data = v_gain_dB[:, freq_idx]
                        print(f"[AI Debug] Extracted gain_data, shape={gain_data.shape}")

                        max_gain = float(np.max(gain_data))
                        min_gain = float(np.min(gain_data))
                        avg_gain = float(np.mean(gain_data))
                        max_idx = np.argmax(gain_data)
                        peak_theta = float(theta_angles_deg[max_idx, freq_idx])
                        peak_phi = float(phi_angles_deg[max_idx, freq_idx])

                        # Calculate beamwidth (approximate)
                        gain_3dB_threshold = max_gain - 3.0
                        beamwidth_points = np.sum(gain_data >= gain_3dB_threshold)
                        total_points = len(gain_data)
                        solid_angle_coverage = (beamwidth_points / total_points) * 100

                        # Front-to-back ratio (simplified)
                        front_hemisphere = gain_data[theta_angles_deg[:, freq_idx] <= 90]
                        back_hemisphere = gain_data[theta_angles_deg[:, freq_idx] > 90]
                        if len(back_hemisphere) > 0:
                            fb_ratio = float(np.max(front_hemisphere) - np.max(back_hemisphere))
                        else:
                            fb_ratio = None

                        # Spherical coverage assessment
                        gain_variation = max_gain - min_gain
                        if gain_variation < 5:
                            coverage = "excellent spherical coverage"
                        elif gain_variation < 10:
                            coverage = "good hemispherical coverage"
                        elif gain_variation < 15:
                            coverage = "moderate directional pattern"
                        else:
                            coverage = "highly directive beam"

                        analysis_json = json.dumps(
                            {
                                "success": True,
                                "action": f"3D {component} gain pattern displayed (interactive, rotatable)",
                                "frequency_MHz": actual_freq,
                                "component": component,
                                "analysis": {
                                    "peak_gain_dBi": round(max_gain, 2),
                                    "peak_direction": {
                                        "theta_deg": round(peak_theta, 1),
                                        "phi_deg": round(peak_phi, 1),
                                    },
                                    "minimum_gain_dBi": round(min_gain, 2),
                                    "average_gain_dBi": round(avg_gain, 2),
                                    "gain_variation_dB": round(gain_variation, 2),
                                    "3dB_beamwidth_coverage_pct": round(solid_angle_coverage, 1),
                                    "front_to_back_ratio_dB": (
                                        round(fb_ratio, 1) if fb_ratio else "N/A"
                                    ),
                                    "pattern_characteristics": coverage,
                                    "expert_assessment": f"The 3D radiation pattern at {actual_freq} MHz reveals a {coverage} with {max_gain:.1f} dBi peak gain directed toward theta={peak_theta:.0f} deg, phi={peak_phi:.0f} deg. The {gain_variation:.1f} dB front-to-back variation and {solid_angle_coverage:.0f}% solid angle coverage at -3dB suggest {'isotropic behavior suitable for mobile/IoT applications' if gain_variation < 5 else 'moderate directivity with acceptable omnidirectional properties' if gain_variation < 10 else 'directive characteristics ideal for point-to-point links' if gain_variation > 15 else 'balanced directivity for sectoral coverage'}.{' Front-to-back ratio of ' + str(round(fb_ratio, 1)) + ' dB indicates ' + ('poor' if fb_ratio and fb_ratio < 10 else 'adequate' if fb_ratio and fb_ratio < 20 else 'excellent') + ' isolation.' if fb_ratio else ''}",
                                },
                            }
                        )
                        print(f"[AI Debug] Analysis JSON prepared; proceeding to plotting")
                    except Exception as analysis_error:
                        print(f"[AI Warning] Pre-plot analysis failed: {analysis_error}")
                        import traceback

                        print(f"[AI Traceback] {traceback.format_exc()}")
                        analysis_json = json.dumps(
                            {
                                "success": True,
                                "action": f"3D {component} gain pattern displayed successfully",
                                "frequency_MHz": actual_freq,
                                "component": component,
                                "message": f"Interactive 3D {component} radiation pattern displayed at {actual_freq} MHz. (Analysis unavailable)",
                            }
                        )

                    # Plot AFTER analysis JSON is ready
                    print(f"[AI Debug] All data available, calling plot_passive_3d_component...")
                    try:
                        plot_passive_3d_component(
                            theta_angles_deg,
                            phi_angles_deg,
                            hpol_far_field,
                            vpol_far_field,
                            total_gain_dB,
                            self.freq_list,
                            actual_freq,
                            component,
                            axis_mode=axis_mode.get() if axis_mode else "auto",
                            zmin=float(zmin_var.get()) if zmin_var else -15.0,
                            zmax=float(zmax_var.get()) if zmax_var else 15.0,
                        )
                        print(f"[AI Debug] plot_passive_3d_component completed successfully")
                    except Exception as plot_error:
                        print(f"[AI Error] Plotting failed: {plot_error}")
                        import traceback

                        print(f"[AI Traceback] {traceback.format_exc()}")
                        # Append plotting error info to analysis_json
                        try:
                            analysis_obj = json.loads(analysis_json)
                            analysis_obj["plotting_error"] = str(plot_error)
                            analysis_json = json.dumps(analysis_obj)
                        except (json.JSONDecodeError, TypeError, KeyError) as e:
                            print(f"[WARNING] Could not append plot error to JSON: {e}")
                            pass
                    return analysis_json
                else:
                    return json.dumps({"error": "Required passive scan data not available"})

            elif str(self.scan_type.get()) == "active":
                theta_angles_deg = getattr(self, "theta_angles_deg", None)
                phi_angles_deg = getattr(self, "phi_angles_deg", None)
                total_power_dBm_2d = getattr(self, "total_power_dBm_2d", None)
                phi_angles_deg_plot = getattr(self, "phi_angles_deg_plot", None)
                total_power_dBm_2d_plot = getattr(self, "total_power_dBm_2d_plot", None)

                # Get axis settings
                axis_mode = getattr(self, "axis_mode", None)
                zmin_var = getattr(self, "zmin_var", None)
                zmax_var = getattr(self, "zmax_var", None)

                if all(
                    x is not None for x in [theta_angles_deg, phi_angles_deg, total_power_dBm_2d]
                ):
                    plot_active_3d_data(
                        theta_angles_deg,
                        phi_angles_deg,
                        total_power_dBm_2d,
                        phi_angles_deg_plot,
                        total_power_dBm_2d_plot,
                        actual_freq,
                        power_type=component,
                        axis_mode=axis_mode.get() if axis_mode else "auto",
                        zmin=float(zmin_var.get()) if zmin_var else -15.0,
                        zmax=float(zmax_var.get()) if zmax_var else 15.0,
                    )
                    return json.dumps(
                        {
                            "success": True,
                            "action": f"3D {component} power pattern displayed",
                            "frequency": actual_freq,
                        }
                    )
                else:
                    return json.dumps({"error": "Required active scan data not available"})
            else:
                return json.dumps({"error": f"Unknown scan type: {self.scan_type.get()}"})

        except Exception as e:
            return json.dumps({"error": f"Failed to generate 3D plot: {str(e)}"})

    def _ai_get_gain_statistics(self, frequency=None):
        """Calculate gain statistics using AntennaAnalyzer."""
        try:
            if frequency is None and hasattr(self, "selected_frequency"):
                try:
                    frequency = float(self.selected_frequency.get())
                except (ValueError, AttributeError, TypeError):
                    return json.dumps({"error": "No frequency specified"})

            if not hasattr(self, "freq_list") or not self.freq_list:
                return json.dumps({"error": "No measurement data loaded"})

            analyzer = self._build_analyzer()
            stats = analyzer.get_gain_statistics(frequency=frequency)
            return json.dumps(stats)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _ai_analyze_pattern(self, frequency=None):
        """Analyze radiation pattern characteristics using AntennaAnalyzer.

        For passive scans, delegates to AntennaAnalyzer which provides:
        - HPBW (E-plane and H-plane)
        - Front-to-back ratio
        - Pattern classification (omnidirectional/sectoral/directional)
        - Null detection
        - Main beam direction
        """
        try:
            if frequency is None and hasattr(self, "selected_frequency"):
                try:
                    frequency = float(self.selected_frequency.get())
                except (ValueError, AttributeError, TypeError):
                    return json.dumps({"error": "No frequency specified"})

            if not hasattr(self, "freq_list") or not self.freq_list:
                return json.dumps({"error": "No measurement data loaded"})

            scan_type = self.scan_type.get()

            if scan_type == "passive":
                analyzer = self._build_analyzer()
                analysis = analyzer.analyze_pattern(frequency=frequency)

                # Add peak location from GUI 2D angle arrays
                freq_array = np.array(self.freq_list)
                freq_idx = np.argmin(np.abs(freq_array - frequency))
                theta_list = getattr(self, "theta_list", None)
                phi_list = getattr(self, "phi_list", None)
                total_gain = getattr(self, "total_gain_dB", None)

                if total_gain is not None and theta_list is not None and phi_list is not None:
                    gain_at_freq = total_gain[:, freq_idx]
                    max_idx = np.argmax(gain_at_freq)
                    try:
                        analysis["max_gain_theta_deg"] = float(
                            theta_list[max_idx, freq_idx]
                            if theta_list.ndim == 2
                            else theta_list[max_idx]
                        )
                        analysis["max_gain_phi_deg"] = float(
                            phi_list[max_idx, freq_idx] if phi_list.ndim == 2 else phi_list[max_idx]
                        )
                    except (IndexError, TypeError):
                        pass

                return json.dumps(analysis)

            elif scan_type == "active":
                total_power = getattr(self, "total_power_dBm_2d")
                theta_angles_deg = getattr(self, "theta_angles_deg")
                phi_angles_deg = getattr(self, "phi_angles_deg")

                freq_array = np.array(self.freq_list)
                freq_idx = np.argmin(np.abs(freq_array - frequency))
                actual_freq = float(self.freq_list[freq_idx])

                max_power = float(np.max(total_power))
                min_power = float(np.min(total_power))
                avg_power = float(np.mean(total_power))

                max_idx = np.unravel_index(np.argmax(total_power), total_power.shape)
                max_theta = float(theta_angles_deg[max_idx[0]])
                max_phi = float(phi_angles_deg[max_idx[1]])

                analysis = {
                    "frequency": actual_freq,
                    "scan_type": "active",
                    "pattern_type": "active radiated power distribution",
                    "max_power_dBm": max_power,
                    "max_power_theta_deg": max_theta,
                    "max_power_phi_deg": max_phi,
                    "min_power_dBm": min_power,
                    "avg_power_dBm": avg_power,
                    "power_range_dB": float(max_power - min_power),
                }
                return json.dumps(analysis)

            else:
                return json.dumps({"error": f"Unknown scan type: {scan_type}"})

        except Exception as e:
            return json.dumps({"error": f"Pattern analysis failed: {str(e)}"})

    def _ai_compare_polarizations(self, frequency=None):
        """Compare HPOL and VPOL performance using AntennaAnalyzer for passive scans."""
        try:
            if frequency is None and hasattr(self, "selected_frequency"):
                try:
                    frequency = float(self.selected_frequency.get())
                except (ValueError, AttributeError, TypeError):
                    return json.dumps({"error": "No frequency specified"})

            scan_type = self.scan_type.get()

            # Active scan polarization comparison (not handled by AntennaAnalyzer)
            if scan_type == "active":
                if not (hasattr(self, "h_power_dBm_2d") and hasattr(self, "v_power_dBm_2d")):
                    return json.dumps(
                        {"error": "HPOL and VPOL power data not loaded for active scan"}
                    )

                h_power = getattr(self, "h_power_dBm_2d")
                v_power = getattr(self, "v_power_dBm_2d")
                actual_freq = getattr(self, "active_frequency", frequency)

                h_max = float(np.max(h_power))
                v_max = float(np.max(v_power))
                xpd = np.abs(h_power - v_power)

                if h_max > v_max:
                    dominant_pol = "HPOL (Horizontal)"
                    pol_advantage = h_max - v_max
                elif v_max > h_max:
                    dominant_pol = "VPOL (Vertical)"
                    pol_advantage = v_max - h_max
                else:
                    dominant_pol = "Balanced"
                    pol_advantage = 0.0

                return json.dumps(
                    {
                        "frequency": actual_freq,
                        "scan_type": "active",
                        "hpol_max_power_dBm": h_max,
                        "hpol_avg_power_dBm": float(np.mean(h_power)),
                        "vpol_max_power_dBm": v_max,
                        "vpol_avg_power_dBm": float(np.mean(v_power)),
                        "hpol_TRP_dBm": float(getattr(self, "h_TRP_dBm", 0)),
                        "vpol_TRP_dBm": float(getattr(self, "v_TRP_dBm", 0)),
                        "cross_pol_avg_dB": float(np.mean(xpd)),
                        "dominant_polarization": dominant_pol,
                        "polarization_advantage_dB": float(pol_advantage),
                    }
                )

            # Passive scan: delegate to AntennaAnalyzer
            if scan_type != "passive":
                return json.dumps(
                    {"error": "Polarization comparison only available for passive or active scans"}
                )

            if not (hasattr(self, "h_gain_dB") and hasattr(self, "v_gain_dB")):
                return json.dumps({"error": "HPOL and VPOL gain data not loaded"})

            if not hasattr(self, "freq_list") or not self.freq_list:
                return json.dumps({"error": "No measurement data loaded"})

            analyzer = self._build_analyzer()
            comparison = analyzer.compare_polarizations(frequency=frequency)

            # Add peak location details from GUI 2D angle arrays
            freq_array = np.array(self.freq_list)
            freq_idx = np.argmin(np.abs(freq_array - frequency))
            h_gain = getattr(self, "h_gain_dB")[:, freq_idx]
            v_gain = getattr(self, "v_gain_dB")[:, freq_idx]
            theta_list = getattr(self, "theta_list", None)
            phi_list = getattr(self, "phi_list", None)

            if theta_list is not None and phi_list is not None:
                h_max_idx = np.argmax(h_gain)
                v_max_idx = np.argmax(v_gain)
                try:
                    t = theta_list
                    p = phi_list
                    comparison["hpol_peak_theta_deg"] = float(
                        t[h_max_idx, freq_idx] if t.ndim == 2 else t[h_max_idx]
                    )
                    comparison["hpol_peak_phi_deg"] = float(
                        p[h_max_idx, freq_idx] if p.ndim == 2 else p[h_max_idx]
                    )
                    comparison["vpol_peak_theta_deg"] = float(
                        t[v_max_idx, freq_idx] if t.ndim == 2 else t[v_max_idx]
                    )
                    comparison["vpol_peak_phi_deg"] = float(
                        p[v_max_idx, freq_idx] if p.ndim == 2 else p[v_max_idx]
                    )
                except (IndexError, TypeError):
                    pass

            # Add dominant polarization label
            h_max_val = comparison.get("max_hpol_gain_dBi", 0)
            v_max_val = comparison.get("max_vpol_gain_dBi", 0)
            if h_max_val > v_max_val:
                comparison["dominant_polarization"] = "HPOL (Horizontal/Phi)"
            elif v_max_val > h_max_val:
                comparison["dominant_polarization"] = "VPOL (Vertical/Theta)"
            else:
                comparison["dominant_polarization"] = "Balanced"

            return json.dumps(comparison)

        except Exception as e:
            return json.dumps({"error": f"Polarization comparison failed: {str(e)}"})

    def _ai_analyze_all_frequencies(self):
        """Analyze gain/power trends across all measured frequencies using AntennaAnalyzer.

        Returns resonance frequency, 3dB bandwidth, gain variation,
        and per-frequency peak gains.
        """
        try:
            if not hasattr(self, "freq_list") or not self.freq_list:
                return json.dumps({"error": "No measurement data loaded"})

            analyzer = self._build_analyzer()
            result = analyzer.analyze_all_frequencies()
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"Frequency analysis failed: {str(e)}"})
