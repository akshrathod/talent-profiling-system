from evaluation.deterministic import score_profile
from evaluation.graph_validation import validate_queries


def test_score_profile_matches_equivalent_capabilities():
    predicted = {
        "researchers": ["Jane Doe"],
        "institution": "MIT",
        "research_capabilities": ["NLP", "Bi-LSTM"],
    }
    ground_truth = {
        "researchers": ["Jane Doe"],
        "institution": "Massachusetts Institute of Technology",
        "research_capabilities": ["Natural Language Processing", "BiLSTM"],
    }

    score = score_profile(predicted, ground_truth)

    assert score["researchers"]["f1"] == 1.0
    assert score["research_capabilities"]["f1"] == 1.0


def test_verbose_capability_matches_canonical_ground_truth():
    predicted = {
        "researchers": [],
        "institution": "",
        "research_capabilities": [
            "Development of research agendas for AI safety and openness"
        ],
    }
    ground_truth = {
        "researchers": [],
        "institution": "",
        "research_capabilities": ["AI Safety Research Strategy"],
    }

    score = score_profile(predicted, ground_truth)

    assert score["research_capabilities"]["f1"] == 1.0


def test_graph_validation_rejects_contract_violations():
    queries = [
        "MERGE (r:Researcher {name: 'Jane Doe'})",
        "CREATE (d:Domain {name: 'unknown'});",
    ]

    result = validate_queries(queries)

    assert result["valid_queries"] == 1
    assert result["invalid_queries"] == 1
    assert result["violation_counts"]["contains_semicolon"] == 1
    assert result["violation_counts"]["uses_create"] == 1
    assert result["violation_counts"]["invalid_labels:Domain"] == 1
    assert result["violation_counts"]["placeholder_node_name"] == 1
