# M1 synthetic/dev fixture

`dev.jsonl` is a three-row Natural Questions-style adapter fixture for CI and
local protocol tests. It is not the Natural Questions benchmark and does not
contain a Wikipedia corpus. Real M1 runs must record the upstream dataset
revision, selected IDs, preprocessing version, and data hash in the manifest.
## C2 smoke fixture

`c2_smoke.jsonl` is an integration/stability fixture mechanically converted
from `data/m0/tasks.jsonl`, using only `task-001` through `task-010`. It is not
benchmark data and is not a research train/dev/test split. The two calibration
tasks (`task-011` and `task-012`) are intentionally excluded. Questions,
answer aliases, and gold evidence IDs are unchanged; gold fields remain on the
evaluator-side task and are not exposed through `Task.policy_view()`.
