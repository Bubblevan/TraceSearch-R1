# M0 deterministic fixture

This is a compact synthetic/local corpus and task set for testing the
TraceSearch-R1 research substrate. It is intentionally not an evaluation
benchmark and contains no copied benchmark text. The corpus is aligned with
the task evidence IDs, includes single-hop and two-hop questions, and can be
run without network access or API keys.

`gold_evidence_ids` are evaluation metadata. The only component permitted to
use them directly is the explicitly named `OracleFixturePolicy` used by the
offline smoke runner; a normal policy receives task question and trajectory
observations instead.
