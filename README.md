# TraceSearch-R1

**A reproducible research scaffold for multi-turn Web Search Agents with contribution-aware credit assignment.**

TraceSearch-R1 studies an agent loop in which a language model alternates between reasoning, web search, page visits, and a cited final answer:

```text
question → think → search(query) → evidence → visit(url) → evidence → answer
```

The project is intentionally text-only. It begins with a Search-R1-style interleaved search baseline, then evaluates two targeted extensions:

1. **Fatal-aware trajectory handling**: stop learning from cascading tool failures while preserving usable prefixes.
2. **Contribution-weighted credit assignment**: reweight trajectory advantages with per-step evidence contribution scores.

## Scope

- Reproducible research implementation; not a copy of a training framework.
- Real search/visit integrations remain behind small tool interfaces.
- Metrics include answer correctness, citation coverage, search turns, tool failures, and cost.

## Repository layout

```text
src/tracesearch/
  agent/          Multi-turn control loop and trajectory types
  environment/    Search / visit tool protocols and deterministic test tools
  rewards/        Format, answer, fatal-aware, and contribution weighting helpers
  evaluation/     Trajectory-level metrics
tests/            Offline unit tests
docs/             Design and experimental plan
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Research plan

| Stage | Question | Deliverable |
| --- | --- | --- |
| 0 | Does an interleaved agent use evidence correctly? | Deterministic offline baseline and trajectory logging |
| 1 | Can SFT/RL improve search policy? | Search-R1-style training adapter and fixed evaluation split |
| 2 | Do failure-aware masks prevent harmful updates? | Fatal-aware ablation |
| 3 | Which search rounds matter? | Contribution-weighted GRPO ablation |

## Attribution

The research direction is informed by Search-R1, DeepResearcher, OpenSearch-VL, and CW-GRPO. This repository will document any reused code, datasets, models, and results explicitly in experiment reports.

## Status

Initial scaffold. The default tools are deterministic fixtures so tests never require credentials or network access.
