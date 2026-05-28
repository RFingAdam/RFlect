"""
Instrument & positioner automation for RFlect (v6.0).

Defines hardware-agnostic driver protocols plus mock backends so the
acquisition workflow is fully testable without hardware. Real backends
(pyVISA SCPI, serial/GPIB positioners) implement the same protocols.

- VNA (SCPI): connect, set sweep, trigger, read S-parameter trace.        (#44)
- Positioner: home, move theta/phi, read position.                        (#45)

A SCPI/pyVISA backend and a serial-positioner backend are documented in the
class docstrings; they are intentionally optional (no pyvisa/pyserial import at
module load) so the deterministic core and CI never depend on hardware libs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# #44 — SCPI VNA control
# ---------------------------------------------------------------------------


class VnaBackend(Protocol):
    """Minimal VNA control surface a backend must provide."""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def write(self, scpi: str) -> None: ...
    def query(self, scpi: str) -> str: ...


class MockVna:
    """In-memory VNA backend that records SCPI writes and synthesizes a trace.

    Lets the acquisition workflow (and its tests) run with no hardware. It
    returns a deterministic resonant S11 sweep so downstream analysis has
    something physical to chew on.
    """

    def __init__(self, resonance_hz: float = 2.45e9):
        self.connected = False
        self.writes: List[str] = []
        self._start = 2.0e9
        self._stop = 3.0e9
        self._points = 201
        self._resonance = resonance_hz

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def write(self, scpi: str) -> None:
        self.writes.append(scpi)
        s = scpi.upper()
        # Parse the handful of commands the VnaController emits.
        if "FREQ:STAR" in s:
            self._start = float(scpi.split()[-1])
        elif "FREQ:STOP" in s:
            self._stop = float(scpi.split()[-1])
        elif "SWE:POIN" in s:
            self._points = int(float(scpi.split()[-1]))

    def query(self, scpi: str) -> str:
        s = scpi.upper()
        if "*IDN" in s:
            return "RFlect,MockVNA,0,1.0"
        if "FREQ:DATA" in s or "FORM:DATA" in s:
            f = np.linspace(self._start, self._stop, self._points)
            return ",".join(f"{v:.6e}" for v in f)
        if "CALC:DATA" in s or "TRAC:DATA" in s:
            f = np.linspace(self._start, self._stop, self._points)
            s11 = -2.0 - 18.0 * np.exp(-(((f - self._resonance) / 60e6) ** 2))
            # Return interleaved real,imag (zero phase for the mock).
            mag = 10 ** (s11 / 20.0)
            vals = []
            for m in mag:
                vals += [f"{m:.6e}", "0"]
            return ",".join(vals)
        return ""


class VnaController:
    """Hardware-agnostic VNA driver. Inject any VnaBackend (MockVna for tests,
    a pyVISA backend for real hardware).

    Real backend example::

        import pyvisa
        class PyvisaVna:
            def __init__(self, resource): self.r = resource; self.rm = pyvisa.ResourceManager()
            def connect(self): self.inst = self.rm.open_resource(self.r)
            def disconnect(self): self.inst.close()
            def write(self, s): self.inst.write(s)
            def query(self, s): return self.inst.query(s)

        vna = VnaController(PyvisaVna("TCPIP0::192.168.0.10::inst0::INSTR"))
    """

    def __init__(self, backend: VnaBackend):
        self.backend = backend

    def connect(self) -> str:
        self.backend.connect()
        return self.backend.query("*IDN?")

    def disconnect(self) -> None:
        self.backend.disconnect()

    def configure_sweep(self, start_hz: float, stop_hz: float, points: int) -> None:
        self.backend.write(f"SENS:FREQ:STAR {start_hz}")
        self.backend.write(f"SENS:FREQ:STOP {stop_hz}")
        self.backend.write(f"SENS:SWE:POIN {points}")

    def read_s_parameter(self, param: str = "S11") -> Dict[str, object]:
        """Trigger and read one S-parameter trace.

        Returns {freq_hz, s_real, s_imag, param}.
        """
        self.backend.write(f"CALC:PAR:DEF {param}")
        self.backend.write("INIT:IMM")
        freq_raw = self.backend.query("SENS:FREQ:DATA?")
        data_raw = self.backend.query("CALC:DATA? SDATA")
        freq = [float(x) for x in freq_raw.split(",") if x.strip()]
        flat = [float(x) for x in data_raw.split(",") if x.strip()]
        s_real = flat[0::2]
        s_imag = flat[1::2]
        return {"freq_hz": freq, "s_real": s_real, "s_imag": s_imag, "param": param}


# ---------------------------------------------------------------------------
# #45 — chamber positioner control
# ---------------------------------------------------------------------------


class PositionerBackend(Protocol):
    def home(self) -> None: ...
    def move_to(self, theta_deg: float, phi_deg: float) -> None: ...
    def position(self) -> Tuple[float, float]: ...
    def close(self) -> None: ...


class MockPositioner:
    """In-memory positioner: tracks commanded (theta, phi) and a move log."""

    def __init__(self):
        self._theta = 0.0
        self._phi = 0.0
        self.moves: List[Tuple[float, float]] = []
        self.homed = False

    def home(self) -> None:
        self._theta = 0.0
        self._phi = 0.0
        self.homed = True
        self.moves.append((0.0, 0.0))

    def move_to(self, theta_deg: float, phi_deg: float) -> None:
        self._theta = float(theta_deg) % 360.0
        self._phi = float(phi_deg) % 360.0
        self.moves.append((self._theta, self._phi))

    def position(self) -> Tuple[float, float]:
        return (self._theta, self._phi)

    def close(self) -> None:
        pass


class PositionerController:
    """Hardware-agnostic theta/phi positioner driver.

    A serial backend (e.g. a turntable controller) implements ``home``,
    ``move_to``, ``position``, ``close`` over pyserial.
    """

    def __init__(self, backend: PositionerBackend):
        self.backend = backend

    def home(self) -> Tuple[float, float]:
        self.backend.home()
        return self.backend.position()

    def goto(self, theta_deg: float, phi_deg: float) -> Tuple[float, float]:
        self.backend.move_to(theta_deg, phi_deg)
        return self.backend.position()

    def scan_grid(
        self, theta_steps: List[float], phi_steps: List[float]
    ) -> List[Tuple[float, float]]:
        """Visit a (theta x phi) grid, returning the realized positions in order.

        This is the unattended full-sphere sweep primitive — pair it with
        VnaController.read_s_parameter (or a chamber receiver read) at each point.
        """
        visited: List[Tuple[float, float]] = []
        for th in theta_steps:
            for ph in phi_steps:
                self.backend.move_to(th, ph)
                visited.append(self.backend.position())
        return visited
