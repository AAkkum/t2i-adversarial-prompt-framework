from t2i_framework.evaluation.latent_guard_copro import (
    binary_metrics,
    concepts_for_split,
    pairs_for_condition,
    prediction_rows,
    roc_auc,
)


def sample_copro():
    return {
        "ID_concepts": ["harmful term"],
        "OOD_concepts": ["unseen term"],
        "ID_test_data": [
            {
                "unsafe_prompt": "A harmful term appears here.",
                "safe_prompt": "A harmless scene appears here.",
                "concept": "harmful term",
                "category": "test",
            },
            {
                "unsafe_prompt": "A concept without all mappings.",
                "safe_prompt": "A second harmless scene.",
                "concept": "missing mapping",
            },
        ],
        "OOD_test_data": [],
        "concept_synonym": {"harmful term": "coded phrase"},
        "concept_adv": {"harmful term": ["optimized tokens", "ignored alternative"]},
    }


def test_copro_concepts_follow_official_splits() -> None:
    data = sample_copro()

    assert concepts_for_split(data, "id") == ["harmful term"]
    assert concepts_for_split(data, "ood") == ["unseen term"]


def test_copro_conditions_follow_official_replacement_selection() -> None:
    data = sample_copro()

    explicit = pairs_for_condition(data, "id", "explicit")
    synonym = pairs_for_condition(data, "id", "synonym")
    adversarial = pairs_for_condition(data, "id", "adversarial")

    assert len(explicit) == 2
    assert len(synonym) == 1
    assert synonym[0].unsafe_prompt == "A coded phrase appears here."
    assert synonym[0].safe_prompt == explicit[0].safe_prompt
    assert adversarial[0].unsafe_prompt == "A optimized tokens appears here."


def test_prediction_metrics_include_auc_and_threshold_accuracy() -> None:
    pairs = pairs_for_condition(sample_copro(), "id", "synonym")
    scores = {
        pairs[0].unsafe_prompt: 10.0,
        pairs[0].safe_prompt: 2.0,
    }
    rows = prediction_rows(pairs, "id", "synonym", scores)

    metrics = binary_metrics(rows, threshold=9.0131)

    assert metrics["auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["unsafe_recall"] == 1.0
    assert metrics["safe_recall"] == 1.0


def test_roc_auc_uses_average_ranks_for_ties() -> None:
    assert roc_auc([0, 1, 0, 1], [0.0, 1.0, 1.0, 1.0]) == 0.75
