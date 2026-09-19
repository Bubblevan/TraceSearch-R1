import json
from pathlib import Path

from tracesearch.data.io import load_trajectories
from tracesearch.data.hashing import hash_corpus
from tracesearch.environment import Corpus
from tracesearch.experiment.runner import run_from_files


def test_fixture_run_writes_recomputable_artifacts(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "run"
    manifest, trajectories = run_from_files(root / "data/m0/tasks.jsonl", root / "data/m0/corpus.jsonl", output, seed=42, run_id="test-run")
    assert len(trajectories) == 12
    assert (output / "manifest.json").exists()
    assert (output / "metrics.json").exists()
    assert (output / "summary.md").exists()
    saved = load_trajectories(output / "trajectories.jsonl")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert saved == trajectories
    assert metrics["normalized_exact_match"] == 1.0
    assert manifest.dataset.corpus_hash == hash_corpus(Corpus.from_jsonl(root / "data/m0/corpus.jsonl").documents)
