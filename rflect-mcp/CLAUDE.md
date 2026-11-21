# Agent notes

## MCP first (do not bypass with bash)

If this MCP covers the job, use its tools. If a call fails, root-cause and fix this MCP (branch, test, PR), then retry through the MCP. Do not silently fall back to bash, vendor CLIs, or ad-hoc scripts.

A temporary bypass is allowed only when you state it in your report and log it in `rflect-mcp/KNOWN_ISSUES.md` (or the Known issues file linked below). Record the command family only — no secrets, hostnames, LAN IPs, or private paths.

Plain bash is fine for local files, git, builds, and work this MCP does not cover.

### Use the `rflect` MCP for

- Antenna visualization and chamber-data jobs the RFlect MCP exposes
- Driving RFlect through ad-hoc scripts when this MCP can do the job

