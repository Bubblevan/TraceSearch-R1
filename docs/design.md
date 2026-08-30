# Design notes

## Experimental contract

Every experiment must record the model, prompt/template version, search backend, data split, random seed, tool budget, token budget, reward formula, and API/model cost. Claims in the README or resume must link to a committed experiment report.

## Baseline and ablations

1. Prompted search agent: fixed policy, no RL.
2. Search-R1-style agentic RL: outcome and format rewards.
3. + fatal-aware handling: mask from the first cascading failure; preserve valid prefix data.
4. + contribution weighting: judge each search/visit step for retrieval utility and reasoning contribution, then scale the trajectory advantage per step.

The fourth setting is the project research hypothesis, not a pre-claimed improvement. It must be compared against the same model, data, search backend, and budget.

## Deferred work

Experience memory / self-evolving rollouts is deliberately deferred until the baseline and credit-assignment ablations are reproducible.
