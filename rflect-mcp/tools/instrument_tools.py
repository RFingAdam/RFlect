"""
Instrument & positioner automation MCP tools for RFlect (v6.0, #44/#45).

Drive a VNA over SCPI and a chamber positioner. With no hardware resource the
tools run against an in-memory mock (so the workflow is usable + testable
everywhere); given a VISA resource / serial port they use a real backend via
pyVISA / pyserial if those optional libraries are installed. Never raise;
failures populate `warnings`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from plot_antenna.instruments import (
    MockVna,
    MockPositioner,
    VnaController,
    PositionerController,
)


def _pyvisa_backend(resource: str):
    """Build a pyVISA-backed VNA backend, or return None if pyvisa is absent."""
    try:
        import pyvisa  # type: ignore
    except Exception:
        return None

    class _PyvisaVna:
        def __init__(self, res):
            self._res = res
            self._rm = pyvisa.ResourceManager()
            self._inst = None

        def connect(self):
            self._inst = self._rm.open_resource(self._res)

        def disconnect(self):
            if self._inst:
                self._inst.close()

        def write(self, s):
            self._inst.write(s)

        def query(self, s):
            return self._inst.query(s)

    return _PyvisaVna(resource)


def register_instrument_tools(mcp):
    """Register instrument / positioner automation tools."""

    @mcp.tool()
    def vna_read_trace(
        start_hz: float = 2.0e9,
        stop_hz: float = 3.0e9,
        points: int = 201,
        param: str = "S11",
        visa_resource: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure a VNA sweep and read one S-parameter trace over SCPI.

        With no `visa_resource`, runs against an in-memory mock VNA (a resonant
        S11 demo) so the workflow is usable without hardware. With a VISA
        resource string (e.g. "TCPIP0::192.168.0.10::inst0::INSTR") it uses a
        pyVISA backend if pyvisa is installed.

        Returns: idn, freq_hz, s_real, s_imag, param, backend ("mock"|"pyvisa"),
        warnings. Never raises.
        """
        result: Dict[str, Any] = {"backend": None, "warnings": []}
        if visa_resource:
            backend = _pyvisa_backend(visa_resource)
            if backend is None:
                result["warnings"].append(
                    "pyvisa not installed; install the optional 'instruments' extra "
                    "to drive real hardware. Falling back to the mock VNA."
                )
                backend = MockVna()
                result["backend"] = "mock"
            else:
                result["backend"] = "pyvisa"
        else:
            backend = MockVna()
            result["backend"] = "mock"

        try:
            vna = VnaController(backend)
            result["idn"] = vna.connect()
            vna.configure_sweep(start_hz, stop_hz, int(points))
            trace = vna.read_s_parameter(param)
            vna.disconnect()
            result.update(trace)
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"vna_read_trace failed: {exc}")
        return result

    @mcp.tool()
    def positioner_scan_grid(
        theta_steps: List[float],
        phi_steps: List[float],
        serial_port: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Drive a chamber positioner over a theta x phi grid (unattended sweep).

        With no `serial_port`, runs against an in-memory mock positioner (returns
        the visited positions) so the sweep logic is testable without hardware.
        A real serial backend can be wired in for production. Pair each visited
        point with vna_read_trace (or a chamber receiver read) for a full scan.

        Returns: n_points, visited [[theta,phi],...], backend, warnings.
        """
        result: Dict[str, Any] = {"backend": None, "n_points": 0, "visited": [], "warnings": []}
        if serial_port:
            result["warnings"].append(
                "no built-in serial positioner backend bundled; using the mock. "
                "Implement PositionerBackend over pyserial for your controller."
            )
        backend = MockPositioner()
        result["backend"] = "mock"
        try:
            ctrl = PositionerController(backend)
            ctrl.home()
            visited = ctrl.scan_grid([float(t) for t in theta_steps], [float(p) for p in phi_steps])
            result["visited"] = [[round(t, 3), round(p, 3)] for t, p in visited]
            result["n_points"] = len(visited)
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"positioner_scan_grid failed: {exc}")
        return result
