# M1 synthetic/dev fixture

`dev.jsonl` is a three-row Natural Questions-style adapter fixture for CI and
local protocol tests. It is not the Natural Questions benchmark and does not
contain a Wikipedia corpus. Real M1 runs must record the upstream dataset
revision, selected IDs, preprocessing version, and data hash in the manifest.
