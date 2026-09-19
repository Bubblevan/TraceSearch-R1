from tracesearch.experiment.manifest import ExperimentManifest


def test_manifest_has_nested_required_fields_and_round_trips():
    manifest = ExperimentManifest(
        run_id="run-1",
        seed=42,
        dataset={"name": "fixture", "version": "1", "corpus_hash": "abc"},
        environment={"backend": "local_bm25", "top_k": 5},
        agent={"policy": "scripted", "max_turns": 4},
        model=None,
        training=None,
    )
    assert manifest.dataset.name == "fixture"
    assert ExperimentManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
