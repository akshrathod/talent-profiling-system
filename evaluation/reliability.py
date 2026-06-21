"""
End-to-end reliability and performance metrics for pipeline runs.
"""


def token_total(state: dict) -> int:
    """Sum tracked input/output tokens from a final pipeline state."""
    total = 0
    for value in state.get("token_log", {}).values():
        if isinstance(value, dict):
            total += value.get("input", 0) + value.get("output", 0)
    return total


def summarize_reliability(states: list[dict]) -> dict:
    """Aggregate reliability metrics across final pipeline states."""
    total_docs = len(states)
    pipeline_errors = [s for s in states if s.get("status") == "pipeline_error"]
    completed = [
        s for s in states
        if s.get("status") != "pipeline_error" and not s.get("neo4j_error")
    ]
    graph_failures = [
        s for s in states
        if s.get("neo4j_error") or s.get("failed_queries")
    ]
    extraction_failures = [
        s for s in states
        if s.get("status") == "pipeline_error" and "extract" in s.get("error", "").lower()
    ]

    latencies = [
        s.get("latency_log", {}).get("total", 0)
        for s in states
        if isinstance(s.get("latency_log"), dict)
    ]
    token_counts = [token_total(s) for s in states]
    retries = [s.get("retry_count", 0) for s in states]
    pii_redactions = [
        s.get("pii_audit", {}).get("items_redacted", 0)
        for s in states
        if isinstance(s.get("pii_audit"), dict)
    ]

    return {
        "documents": total_docs,
        "completed": len(completed),
        "pipeline_errors": len(pipeline_errors),
        "extraction_failures": len(extraction_failures),
        "graph_failures": len(graph_failures),
        "pipeline_success_rate": round(len(completed) / total_docs, 3) if total_docs else 0.0,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0.0,
        "total_tokens": sum(token_counts),
        "avg_retries": round(sum(retries) / len(retries), 3) if retries else 0.0,
        "total_retries": sum(retries),
        "avg_pii_redactions": round(sum(pii_redactions) / len(pii_redactions), 3) if pii_redactions else 0.0,
        "total_pii_redactions": sum(pii_redactions),
    }
