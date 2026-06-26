# `data/` — agent file I/O

This is the only directory the agent reads inputs from and writes outputs to.

## Inputs

Drop CSV / XLSX files here when the task type is `file`. The agent's
`get_file_info` node reads from this folder by filename:

```python
# agents/agent_nodes.py — get_file_info
df = pd.read_csv('data/' + file_name)
```

## Outputs

The planner's `required_output` field **must** start with `data/` — see
`plan_check` and the planning prompt. Typical outputs:

```
data/report.xlsx
data/summary.csv
data/result.json
```

The orchestrator reads the final file from `result["required_output"]` and
sends it back to the user via Telegram — see `main.py::handle_message`.
