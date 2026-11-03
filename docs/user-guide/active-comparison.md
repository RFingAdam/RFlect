# Compare active measurements

After importing active measurements, call `compare_active_overlay` with two to six loaded measurement names and a new `.png` output path.

```python
compare_active_overlay(
    measurement_names=["capture_a", "capture_b", "capture_c"],
    output_path="comparison.png",
    polarization="total",
    plane="azimuth",
    cut_angle_deg=90.0,
    max_envelope=True,
)
```

The azimuth cut varies phi at fixed theta. The elevation cut varies theta at fixed phi. All traces retain their absolute dBm levels on the same radial scale. Select `total`, `hpol`, or `vpol`. The requested cut must be sampled in every measurement, and frequencies must match. Maximum envelopes additionally require matching angular grids. The tool rejects missing or nonfinite data and never overwrites an existing file.

Export labels default to Measurement 1, Measurement 2, and so on. Source paths and filenames are not included in the image. Supply an explicit `labels` list when approved labels are needed. Available TRP values appear in the legend for the selected polarization.

Run `python examples/active_overlay_demo.py comparison-demo.png` from a source checkout to generate a three-trace synthetic example. This illustrates the export; it is not chamber validation.

Active report summaries group finite TRP samples by frequency and explicit `dut_id` metadata. Report callers can instead supply `options={"measurement_groups": {"capture_a": "group_a", "capture_b": "group_a", "capture_c": "group_b"}}`. Group identifiers are displayed as DUT 1, DUT 2, and so on. Groups without metadata remain unassigned. Summary ranges and spreads describe the samples and do not claim compliance or measurement uncertainty.

Optional real-data integration tests read `RFLECT_TEST_DATA_DIR`. Private measurement data is not bundled. Tests skip when those files are unavailable.
