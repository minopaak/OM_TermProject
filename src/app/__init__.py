"""Streamlit review app for manager-facing forecast adjustment.

Loads a batch run dir (`data/runs/<run_id>/`) and lets the manager:
  - inspect the input/baseline/agent-adjusted/manager time series,
  - toggle or adjust agent-proposed signals,
  - override individual forecast days,
  - chat with an assistant that can apply the same adjustments,
  - confirm and persist `manager_final/<sku>.parquet`.
"""
