"""
AIChatMixin - AI Chat functionality for RFlect GUI

This mixin provides AI-powered analysis capabilities:
- AI Chat Assistant window
- OpenAI API integration with function calling
- AI analysis functions (2D plot, 3D plot, gain statistics, pattern analysis, polarization comparison)
- Base64 image encoding for inline plot display
"""

from __future__ import annotations

import os
import io
import json
import base64
import tkinter as tk
from typing import TYPE_CHECKING, Optional, List, Any

import numpy as np

from ..config import DARK_BG_COLOR, LIGHT_TEXT_COLOR, ACCENT_BLUE_COLOR, AI_MODEL, AI_MAX_TOKENS

# Import AI settings with fallbacks
try:
    from ..config import AI_TEMPERATURE
except ImportError:
    AI_TEMPERATURE = 0.7

try:
    from ..config import AI_REASONING_EFFORT
except ImportError:
    AI_REASONING_EFFORT = "low"

try:
    from ..config import AI_TEXT_VERBOSITY
except ImportError:
    AI_TEXT_VERBOSITY = "auto"

try:
    from ..config import AI_GENERATE_REASONING_SUMMARY
except ImportError:
    AI_GENERATE_REASONING_SUMMARY = False

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
        # Check if API key is configured
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY2")):
            from tkinter import messagebox

            messagebox.showwarning(
                "API Key Required",
                "Please configure your OpenAI API key first.\n\nGo to Help -> Manage OpenAI API Key",
            )
            return

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

        # Initial context message
        context_msg = f"""Welcome to AI Chat Assistant!

I can help you analyze your antenna measurement data. Current loaded data:
- Scan Type: {self.scan_type.get()}
- TRP File: {os.path.basename(self.TRP_file_path) if hasattr(self, 'TRP_file_path') and self.TRP_file_path else 'None'}
- HPOL File: {os.path.basename(self.hpol_file_path) if hasattr(self, 'hpol_file_path') and self.hpol_file_path else 'None'}
- VPOL File: {os.path.basename(self.vpol_file_path) if hasattr(self, 'vpol_file_path') and self.vpol_file_path else 'None'}

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
            chat_window.update()

            try:
                # Build context from loaded data
                data_context = self._build_chat_context()

                # Add user message to history
                chat_history.append({"role": "user", "content": user_msg})

                # Call OpenAI API (handles function calling internally)
                ai_response = self._get_ai_response(chat_history, data_context)

                # Add AI response to history
                chat_history.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                ai_response = f"Sorry, I encountered an error: {str(e)}\n\nPlease check your API key configuration and internet connection."
                print(f"AI Chat Error: {e}")

            # Remove thinking indicator
            chat_text.config(state=tk.NORMAL)
            chat_text.delete("end-3l", "end-1c")

            # Display AI response
            chat_text.insert(tk.END, f"AI: {ai_response}\n\n", "ai")
            chat_text.tag_config("ai", foreground="#4CAF50")
            chat_text.insert(tk.END, "=" * 80 + "\n\n")
            chat_text.see(tk.END)
            chat_text.config(state=tk.DISABLED)

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

        # Close button at bottom
        button_frame = tk.Frame(chat_window, bg=DARK_BG_COLOR)
        button_frame.pack(pady=(0, 10), padx=20, side=tk.BOTTOM)

        close_btn = tk.Button(
            button_frame,
            text="Close",
            command=chat_window.destroy,
            bg=DARK_BG_COLOR,
            fg=LIGHT_TEXT_COLOR,
            width=10,
        )
        close_btn.pack(pady=(0, 10))

    def _build_chat_context(self):
        """Build context string from currently loaded measurement data."""
        context_parts = []

        # Scan type
        context_parts.append(f"Scan Type: {self.scan_type.get()}")

        # Active scan data
        if (
            self.scan_type.get() == "active"
            and hasattr(self, "TRP_file_path")
            and self.TRP_file_path
        ):
            context_parts.append(f"Active TRP File: {os.path.basename(self.TRP_file_path)}")
            # Try to extract frequency if available
            if hasattr(self, "freq_list") and self.freq_list:
                context_parts.append(
                    f"Frequencies: {min(self.freq_list):.1f} - {max(self.freq_list):.1f} MHz"
                )

        # Passive scan data
        elif self.scan_type.get() == "passive":
            if hasattr(self, "hpol_file_path") and self.hpol_file_path:
                context_parts.append(f"HPOL File: {os.path.basename(self.hpol_file_path)}")
            if hasattr(self, "vpol_file_path") and self.vpol_file_path:
                context_parts.append(f"VPOL File: {os.path.basename(self.vpol_file_path)}")
            if hasattr(self, "freq_list") and self.freq_list:
                context_parts.append(
                    f"Frequencies: {min(self.freq_list):.1f} - {max(self.freq_list):.1f} MHz"
                )

        # Selected frequency
        if hasattr(self, "selected_frequency") and self.selected_frequency.get():
            try:
                freq = float(self.selected_frequency.get())
                context_parts.append(f"Selected Frequency: {freq:.1f} MHz")
            except (ValueError, AttributeError, TypeError) as e:
                print(f"[WARNING] Could not parse frequency: {e}")
                pass

        return "\n".join(context_parts) if context_parts else "No data currently loaded"

    def _get_ai_response(self, chat_history, data_context):
        """Get AI response from OpenAI API with function/tool calling for interactive analysis.

        Supports both:
        - GPT-4 family: Chat Completions API with function calling
        - GPT-5.2 family: Responses API with tool calling
        """
        from openai import OpenAI

        # Get API key
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY2")
        if not api_key:
            return "API key not found. Please configure it in Help -> Manage OpenAI API Key."

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Get AI model from config
        ai_model = AI_MODEL if "AI_MODEL" in globals() else "gpt-4o-mini"

        # Validate model supports function/tool calling (required for AI Chat)
        incompatible_models = ["o1-preview", "o1-mini", "o3-mini", "o3", "o1"]
        if any(ai_model.startswith(prefix) for prefix in incompatible_models):
            return (
                f"Error: Model '{ai_model}' does not support function calling, which is required for AI Chat Assistant.\n\n"
                "Please change your AI model in Help -> AI Settings to one of these compatible models:\n"
                "- gpt-4o-mini (recommended - best value)\n"
                "- gpt-4o (premium quality)\n"
                "- gpt-5-mini or gpt-5.2 (advanced reasoning with Responses API)\n\n"
                "O-series models (o1, o3) are designed for deep reasoning but cannot call Python functions interactively."
            )

        # Determine if using GPT-5 (Responses API) or GPT-4 (Chat Completions API)
        is_gpt5_model = ai_model.startswith("gpt-5")

        # Define available functions that AI can call
        available_functions = {
            "generate_2d_plot": self._ai_generate_2d_plot,
            "generate_3d_plot": self._ai_generate_3d_plot,
            "get_gain_statistics": self._ai_get_gain_statistics,
            "analyze_pattern": self._ai_analyze_pattern,
            "compare_polarizations": self._ai_compare_polarizations,
            "analyze_all_frequencies": self._ai_analyze_all_frequencies,
        }

        # Build enhanced system message
        system_message = f"""You are a Principal RF Engineer with PhD-level expertise in antenna theory, electromagnetics, and wireless communications. You provide detailed, technically accurate analysis using proper terminology and physical insights.

Current Measurement Data:
{data_context}

**Available Analysis Functions:**
You can call these functions to help answer user questions:
1. generate_2d_plot(frequency) - Display 2D polar plots showing azimuth/elevation cuts with H-pol, V-pol, and total gain. Returns expert analysis.
2. generate_3d_plot(frequency, component) - Display interactive 3D radiation pattern in a window the user can rotate. Returns expert analysis.
3. get_gain_statistics(frequency) - Calculate min/max/average gain values at a frequency, including per-polarization breakdown
4. analyze_pattern(frequency) - Analyze pattern characteristics: HPBW (E-plane/H-plane), front-to-back ratio, null detection, pattern type classification, beam direction
5. compare_polarizations(frequency) - Compare HPOL vs VPOL: cross-polarization discrimination (XPD), polarization balance, dominant polarization
6. analyze_all_frequencies() - Analyze gain/power trends across ALL frequencies: resonance frequency, 3dB bandwidth, gain variation, per-frequency peak gains

**Analysis Guidelines:**
- When plot functions return analysis data, interpret and explain the results with PhD-level technical depth
- Discuss physical mechanisms (e.g., "The elevated cross-pol ratio suggests symmetric current distribution")
- Reference relevant antenna theory (directivity, radiation efficiency, polarization purity, pattern multiplication)
- Provide actionable insights for antenna optimization
- Use proper units and terminology (dBi for gain, solid angle in steradians, HPBW, F/B ratio, XPD)
- When analyzing patterns, discuss implications for link budget, coverage, and interference
- Compare results against theoretical expectations (dipole ~2.15 dBi, isotropic = 0 dBi)

**Response Style:**
- After calling plot functions, interpret the returned analysis JSON professionally
- Provide context: "The displayed plots show..." then explain what the engineer should observe
- Highlight key findings: peak gain location, nulls, polarization balance, pattern symmetry

**Important**: Plot functions now return detailed expert analysis. Parse this JSON and present it professionally to the user with additional context and insights.
"""

        if is_gpt5_model:
            # Use GPT-5.2 Responses API with tool calling
            return self._get_gpt5_response(
                client, ai_model, system_message, chat_history, available_functions
            )
        else:
            # Use GPT-4 Chat Completions API with function calling
            return self._get_gpt4_response(
                client, ai_model, system_message, chat_history, available_functions
            )

    def _get_gpt4_response(
        self, client, ai_model, system_message, chat_history, available_functions
    ):
        """Handle GPT-4 family models using Chat Completions API with function calling."""
        # Function definitions for OpenAI Chat Completions API
        function_definitions = [
            {
                "name": "generate_2d_plot",
                "description": "Generate and DISPLAY 2D radiation pattern plots (polar plots showing azimuth and elevation cuts with H-pol, V-pol, and total gain). The plots will appear in separate windows that the user can see.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz to plot. If not specified, uses currently selected frequency.",
                        },
                        "plot_type": {
                            "type": "string",
                            "enum": ["polar", "rectangular", "azimuth_cuts"],
                            "description": "Type of 2D plot to generate",
                        },
                    },
                },
            },
            {
                "name": "generate_3d_plot",
                "description": "Generate and DISPLAY a 3D radiation pattern plot showing the full spatial distribution.",
                "parameters": {
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
            },
            {
                "name": "get_gain_statistics",
                "description": "Calculate and return gain statistics (max, min, average, variance) for the current data at a specified frequency.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz. If not specified, uses currently selected frequency.",
                        }
                    },
                },
            },
            {
                "name": "analyze_pattern",
                "description": "Analyze the radiation pattern characteristics including pattern type, HPBW (half-power beamwidth) for E-plane and H-plane, front-to-back ratio, null detection, and beam direction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz to analyze",
                        }
                    },
                },
            },
            {
                "name": "compare_polarizations",
                "description": "Compare HPOL and VPOL performance including cross-polarization discrimination (XPD), polarization balance, and dominant polarization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz to compare",
                        }
                    },
                },
            },
            {
                "name": "analyze_all_frequencies",
                "description": "Analyze gain/power trends across ALL measured frequencies. Returns resonance frequency, 3dB bandwidth, gain variation, and per-frequency peak gains.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

        # Prepare messages for API
        messages = [{"role": "system", "content": system_message}]
        messages.extend(chat_history)

        try:
            max_tokens_value = AI_MAX_TOKENS if "AI_MAX_TOKENS" in globals() else 500
            temperature_value = AI_TEMPERATURE if "AI_TEMPERATURE" in globals() else 0.7

            # Build API call parameters
            api_params = {
                "model": ai_model,
                "messages": messages,
                "functions": function_definitions,
                "function_call": "auto",
                "max_tokens": max_tokens_value,
                "temperature": temperature_value,
            }

            # Initial API call
            response = client.chat.completions.create(**api_params)
            response_message = response.choices[0].message

            print(f"[AI Debug] Model: {ai_model} (Chat Completions API)")
            print(
                f"[AI Debug] Has function_call: {hasattr(response_message, 'function_call') and response_message.function_call is not None}"
            )

            # Handle function calling
            if response_message.function_call:
                function_name = response_message.function_call.name
                function_args = json.loads(response_message.function_call.arguments)

                print(f"[AI] Calling function: {function_name}({function_args})")

                if function_name in available_functions:
                    function_response = available_functions[function_name](**function_args)

                    # Add function call and result to messages
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": function_name,
                                "arguments": response_message.function_call.arguments,
                            },
                        }
                    )
                    messages.append(
                        {"role": "function", "name": function_name, "content": function_response}
                    )

                    # Get final response
                    second_response = client.chat.completions.create(
                        model=ai_model,
                        messages=messages,
                        max_tokens=max_tokens_value,
                        temperature=temperature_value,
                    )

                    content = second_response.choices[0].message.content
                    return content.strip() if content else "Analysis complete."

            # No function call, return direct response
            content = response_message.content
            if content and content.strip():
                return content.strip()
            else:
                return "I analyzed your request but didn't generate a text response. Please try rephrasing your question."

        except Exception as e:
            raise Exception(f"OpenAI Chat Completions API error: {str(e)}")

    def _get_gpt5_response(
        self, client, ai_model, system_message, chat_history, available_functions
    ):
        """Handle GPT-5.2 family models using Responses API with tool calling."""
        # Tool definitions for GPT-5.2 Responses API (internally-tagged format)
        # Note: With strict=True, 'required' must include ALL properties
        tool_definitions = [
            {
                "type": "function",
                "name": "generate_2d_plot",
                "description": "Generate and DISPLAY 2D radiation pattern plots (polar plots showing azimuth and elevation cuts with H-pol, V-pol, and total gain).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "number", "description": "Frequency in MHz to plot."},
                        "plot_type": {
                            "type": "string",
                            "enum": ["polar", "rectangular", "azimuth_cuts"],
                            "description": "Type of 2D plot to generate",
                        },
                    },
                    "required": ["frequency", "plot_type"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "generate_3d_plot",
                "description": "Generate and DISPLAY a 3D radiation pattern plot showing the full spatial distribution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "number", "description": "Frequency in MHz"},
                        "component": {
                            "type": "string",
                            "enum": ["total", "hpol", "vpol"],
                            "description": "Which polarization component to plot",
                        },
                    },
                    "required": ["frequency", "component"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_gain_statistics",
                "description": "Calculate and return gain statistics (max, min, average, variance) for the current data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {"type": "number", "description": "Frequency in MHz."}
                    },
                    "required": ["frequency"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "analyze_pattern",
                "description": "Analyze the radiation pattern characteristics including pattern type, HPBW (half-power beamwidth) for E-plane and H-plane, front-to-back ratio, null detection, and beam direction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz to analyze",
                        }
                    },
                    "required": ["frequency"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "compare_polarizations",
                "description": "Compare HPOL and VPOL performance including cross-polarization discrimination (XPD), polarization balance, and dominant polarization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "Frequency in MHz to compare",
                        }
                    },
                    "required": ["frequency"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "analyze_all_frequencies",
                "description": "Analyze gain/power trends across ALL measured frequencies. Returns resonance frequency, 3dB bandwidth, gain variation, and per-frequency peak gains.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

        # Build input for Responses API
        input_messages = [{"role": "system", "content": system_message}]
        for msg in chat_history:
            input_messages.append({"role": msg["role"], "content": msg["content"]})

        # Get GPT-5.2 parameters from config
        reasoning_effort = AI_REASONING_EFFORT if "AI_REASONING_EFFORT" in globals() else "low"
        text_verbosity = AI_TEXT_VERBOSITY if "AI_TEXT_VERBOSITY" in globals() else "auto"
        generate_summary = (
            AI_GENERATE_REASONING_SUMMARY if "AI_GENERATE_REASONING_SUMMARY" in globals() else False
        )
        max_tokens_value = AI_MAX_TOKENS if "AI_MAX_TOKENS" in globals() else 500

        # Auto-map verbosity based on max_tokens if set to "auto"
        if text_verbosity == "auto":
            if max_tokens_value <= 100:
                text_verbosity = "low"
            elif max_tokens_value <= 250:
                text_verbosity = "medium"
            else:
                text_verbosity = "high"

        try:
            # Build reasoning config
            reasoning_config = {"effort": reasoning_effort}
            if generate_summary:
                reasoning_config["summary"] = "auto"

            print(f"[AI Debug] Model: {ai_model} (Responses API)")
            print(f"[AI Debug] Reasoning: {reasoning_effort}, Verbosity: {text_verbosity}")

            # Initial API call with tools
            response = client.responses.create(
                model=ai_model,
                input=input_messages,
                tools=tool_definitions,
                reasoning=reasoning_config,
                text={"verbosity": text_verbosity},
            )

            # Check for tool calls in the response output
            tool_calls = []
            text_output = None

            if hasattr(response, "output") and response.output:
                for item in response.output:
                    item_type = getattr(item, "type", None)
                    if item_type == "function_call":
                        tool_calls.append(item)
                    elif item_type == "message":
                        # Extract text from message content
                        for content_item in getattr(item, "content", []):
                            if getattr(content_item, "type", None) == "output_text":
                                text_output = getattr(content_item, "text", "")

            # Handle tool calls
            if tool_calls:
                # Process each tool call
                for tool_call in tool_calls:
                    function_name = getattr(tool_call, "name", None)
                    call_id = getattr(tool_call, "call_id", None)
                    arguments_str = getattr(tool_call, "arguments", "{}")

                    print(f"[AI] GPT-5.2 calling tool: {function_name} (call_id: {call_id})")

                    if function_name in available_functions:
                        try:
                            function_args = json.loads(arguments_str) if arguments_str else {}
                            function_response = available_functions[function_name](**function_args)
                        except Exception as func_error:
                            function_response = json.dumps({"error": str(func_error)})

                        # Add tool result to input and make follow-up call
                        input_messages.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": function_name,
                                "arguments": arguments_str,
                            }
                        )
                        input_messages.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": function_response,
                            }
                        )

                # Make follow-up call with tool results
                follow_up_response = client.responses.create(
                    model=ai_model,
                    input=input_messages,
                    reasoning=reasoning_config,
                    text={"verbosity": text_verbosity},
                )

                # Extract text from follow-up response
                if hasattr(follow_up_response, "output_text") and follow_up_response.output_text:
                    return follow_up_response.output_text.strip()

                if hasattr(follow_up_response, "output") and follow_up_response.output:
                    for item in follow_up_response.output:
                        if getattr(item, "type", None) == "message":
                            for content_item in getattr(item, "content", []):
                                if getattr(content_item, "type", None) == "output_text":
                                    return getattr(content_item, "text", "").strip()

                return "Analysis complete."

            # No tool calls - return direct text response
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text.strip()

            if text_output:
                return text_output.strip()

            return "I analyzed your request but didn't generate a text response. Please try rephrasing your question."

        except Exception as e:
            raise Exception(f"OpenAI Responses API error: {str(e)}")

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
                            theta_list[max_idx, freq_idx] if theta_list.ndim == 2 else theta_list[max_idx]
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

                return json.dumps({
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
                })

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
