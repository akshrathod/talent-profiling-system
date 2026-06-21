"""
Static validation for generated Cypher.

These checks are intentionally conservative: they validate the contract we ask
the graph agent to follow before queries touch Neo4j.
"""

import re


ALLOWED_LABELS = {"Researcher", "Institution", "ResearchCapability"}
ALLOWED_RELATIONSHIPS = {"RESEARCHES_IN", "AFFILIATED_WITH"}
PLACEHOLDERS = {"unknown", "n/a", "none", "not specified", "not mentioned", "your domain name", ""}


def _extract_labels(query: str) -> set[str]:
    return set(re.findall(r":([A-Za-z_][A-Za-z0-9_]*)\s*\{", query))


def _extract_relationships(query: str) -> set[str]:
    return set(re.findall(r"\[:([A-Za-z_][A-Za-z0-9_]*)\]", query))


def _extract_names(query: str) -> list[str]:
    return re.findall(r"name\s*:\s*'([^']*)'", query)


def validate_query(query: str) -> dict:
    """Validate one generated Cypher query against the project graph contract."""
    violations = []
    labels = _extract_labels(query)
    relationships = _extract_relationships(query)
    names = _extract_names(query)

    if ";" in query:
        violations.append("contains_semicolon")
    if re.search(r"\bCREATE\b", query, flags=re.IGNORECASE):
        violations.append("uses_create")
    if not re.search(r"\bMERGE\b", query):
        violations.append("missing_merge")

    invalid_labels = sorted(labels - ALLOWED_LABELS)
    if invalid_labels:
        violations.append(f"invalid_labels:{','.join(invalid_labels)}")

    invalid_relationships = sorted(relationships - ALLOWED_RELATIONSHIPS)
    if invalid_relationships:
        violations.append(f"invalid_relationships:{','.join(invalid_relationships)}")

    placeholder_names = [
        value for value in names
        if value.strip().lower() in PLACEHOLDERS
    ]
    if placeholder_names:
        violations.append("placeholder_node_name")

    return {
        "query": query,
        "valid": not violations,
        "violations": violations,
        "labels": sorted(labels),
        "relationships": sorted(relationships),
    }


def validate_queries(queries: list[str]) -> dict:
    """Validate a list of Cypher queries and return aggregate graph metrics."""
    query_results = [validate_query(query) for query in queries or []]
    invalid = [result for result in query_results if not result["valid"]]
    violation_counts = {}

    for result in invalid:
        for violation in result["violations"]:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1

    total = len(query_results)
    valid_count = total - len(invalid)

    return {
        "total_queries": total,
        "valid_queries": valid_count,
        "invalid_queries": len(invalid),
        "valid_query_rate": round(valid_count / total, 3) if total else 0.0,
        "violation_counts": violation_counts,
        "invalid_query_details": invalid,
    }
