# ⚠️ RETRACTED — the dashboard displayed the ground-truth label as a prediction

`scripts/generate_dashboard.py` (now under `quarantine/scripts/`) computed its
displayed "prediction" as the ground-truth label plus a fixed offset:

```python
if is_same_cell: p_rel = clip(p_rel + 0.40, 0.70, 0.98)
else:            p_rel = clip(p_rel - 0.35, 0.02, 0.45)
```

Its headline KPIs (e.g. "3,595.4 μm", "95.44%") were hardcoded HTML strings, not
computed from any run. Nothing this dashboard displayed was a measurement.

The viewer code in this directory is retained; it needs a real results source
before it can be used again.

See [`../docs/synthetic_data_audit_and_dataset_plan.md`](../docs/synthetic_data_audit_and_dataset_plan.md).
