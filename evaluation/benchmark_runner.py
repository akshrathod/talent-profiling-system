"""
Benchmark runner for the talent profiling pipeline.

Loads benchmark documents, runs the multi-agent pipeline, evaluates extraction,
validates generated graph queries, and summarizes reliability.
"""

import argparse
import json
from pathlib import Path

from evaluation.deterministic import score_profile
from evaluation.graph_validation import validate_queries
from evaluation.reliability import summarize_reliability


BENCHMARK_DOCS_DIR = Path("evaluation/benchmark_docs")
GROUND_TRUTH_DIR = Path("evaluation/ground_truth")
PREDICTIONS_DIR = Path("evaluation/predictions")
RESULTS_DIR = Path("evaluation/results")


def load_ground_truth(doc_id: str) -> dict:
    """Load ground truth JSON for one benchmark document."""
    path = GROUND_TRUTH_DIR / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing ground truth: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_benchmark_docs() -> list[dict]:
    """Load benchmark .txt files without importing the broader ingestion stack."""
    files = sorted(BENCHMARK_DOCS_DIR.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt benchmark docs found in {BENCHMARK_DOCS_DIR}")

    docs = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        docs.append({
            "doc_id": path.stem,
            "filename": path.name,
            "source": "benchmark_docs",
            "text": text,
            "char_count": len(text),
        })
    return docs


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def aggregate_deterministic(document_results: list[dict]) -> dict:
    """Aggregate deterministic extraction metrics across documents."""
    scores = [r["deterministic"]["overall_extraction_score"] for r in document_results]
    researcher_f1 = [r["deterministic"]["researchers"]["f1"] for r in document_results]
    capability_f1 = [r["deterministic"]["research_capabilities"]["f1"] for r in document_results]
    institution_scores = [r["deterministic"]["institution"]["score"] for r in document_results]

    return {
        "avg_overall_extraction_score": average(scores),
        "avg_researcher_f1": average(researcher_f1),
        "avg_institution_score": average(institution_scores),
        "avg_research_capability_f1": average(capability_f1),
    }


def aggregate_llm_judge(document_results: list[dict]) -> dict:
    """Aggregate semantic and source-grounding LLM judge scores."""
    judged = [r["llm_judge"] for r in document_results if "llm_judge" in r]
    semantic = [result["semantic"] for result in judged]
    grounding = [result["source_grounding"] for result in judged if "source_grounding" in result]

    semantic_fields = [
        "researcher_correctness_score",
        "institution_correctness_score",
        "capability_semantic_alignment_score",
        "capability_coverage_score",
        "overall_profile_quality_score",
    ]
    summary = {
        "documents_judged": len(judged),
        "semantic": {
            f"avg_{field}": average([float(item.get(field, 0)) for item in semantic])
            for field in semantic_fields
        },
        "source_grounding": {
            "avg_source_grounding_score": average([
                float(item.get("source_grounding_score", 0)) for item in grounding
            ]),
            "unsupported_researchers": sum(
                len(item.get("unsupported_researchers", [])) for item in grounding
            ),
            "unsupported_institutions": sum(
                1 for item in grounding if item.get("unsupported_institution")
            ),
            "unsupported_research_capabilities": sum(
                len(item.get("unsupported_research_capabilities", []))
                for item in grounding
            ),
        },
        "tokens": {
            "input": sum(
                item.get("input_tokens", 0)
                for result in judged
                for item in result.values()
                if isinstance(item, dict)
            ),
            "output": sum(
                item.get("output_tokens", 0)
                for result in judged
                for item in result.values()
                if isinstance(item, dict)
            ),
        },
    }
    summary["tokens"]["total"] = summary["tokens"]["input"] + summary["tokens"]["output"]
    return summary


def aggregate_graph(document_results: list[dict]) -> dict:
    """Aggregate static graph validation metrics across documents."""
    total_queries = sum(r["graph"]["total_queries"] for r in document_results)
    valid_queries = sum(r["graph"]["valid_queries"] for r in document_results)
    violation_counts = {}

    for result in document_results:
        for violation, count in result["graph"]["violation_counts"].items():
            violation_counts[violation] = violation_counts.get(violation, 0) + count

    return {
        "total_queries": total_queries,
        "valid_queries": valid_queries,
        "invalid_queries": total_queries - valid_queries,
        "valid_query_rate": round(valid_queries / total_queries, 3) if total_queries else 0.0,
        "violation_counts": violation_counts,
    }


def write_markdown_report(report: dict, path: Path) -> None:
    """Write a readable benchmark summary."""
    lines = [
        "# Talent Profiling Benchmark Report",
        "",
        f"Documents evaluated: {report['reliability']['documents']}",
        "",
        "## Graph",
        "",
        f"- Valid query rate: {report['graph']['valid_query_rate']}",
        f"- Invalid queries: {report['graph']['invalid_queries']}",
        f"- Violations: {json.dumps(report['graph']['violation_counts'])}",
        "",
        "## Reliability",
        "",
        f"- Pipeline success rate: {report['reliability']['pipeline_success_rate']}",
        f"- Pipeline errors: {report['reliability']['pipeline_errors']}",
        f"- Graph failures: {report['reliability']['graph_failures']}",
        f"- Average latency seconds: {report['reliability']['avg_latency_sec']}",
        f"- Average tokens: {report['reliability']['avg_tokens']}",
        f"- Average retries: {report['reliability']['avg_retries']}",
        "",
        "## Per Document",
        "",
    ]

    if "llm_judge" in report:
        lines[4:4] = [
            "## LLM Judge",
            "",
            f"- Overall profile quality: {report['llm_judge']['semantic']['avg_overall_profile_quality_score']}/10",
            f"- Researcher correctness: {report['llm_judge']['semantic']['avg_researcher_correctness_score']}/10",
            f"- Institution correctness: {report['llm_judge']['semantic']['avg_institution_correctness_score']}/10",
            f"- Capability semantic alignment: {report['llm_judge']['semantic']['avg_capability_semantic_alignment_score']}/10",
            f"- Capability coverage: {report['llm_judge']['semantic']['avg_capability_coverage_score']}/10",
            f"- Source grounding: {report['llm_judge']['source_grounding']['avg_source_grounding_score']}/10",
            f"- Unsupported capabilities: {report['llm_judge']['source_grounding']['unsupported_research_capabilities']}",
            "",
        ]

    if "deterministic" in report:
        lines.extend([
            "## Optional Deterministic Metrics",
            "",
            f"- Overall extraction score: {report['deterministic']['avg_overall_extraction_score']}",
            f"- Researcher F1: {report['deterministic']['avg_researcher_f1']}",
            f"- Institution score: {report['deterministic']['avg_institution_score']}",
            f"- Research capability F1: {report['deterministic']['avg_research_capability_f1']}",
            "",
        ])

    for result in report["documents"]:
        document_lines = [
            f"### {result['doc_id']}",
            "",
            f"- Graph valid query rate: {result['graph']['valid_query_rate']}",
            "",
        ]
        if "llm_judge" in result:
            semantic = result["llm_judge"]["semantic"]
            grounding = result["llm_judge"]["source_grounding"]
            document_lines[2:2] = [
                f"- Overall profile quality: {semantic.get('overall_profile_quality_score', 0)}/10",
                f"- Capability semantic alignment: {semantic.get('capability_semantic_alignment_score', 0)}/10",
                f"- Capability coverage: {semantic.get('capability_coverage_score', 0)}/10",
                f"- Source grounding: {grounding.get('source_grounding_score', 0)}/10",
                f"- Unsupported capabilities: {json.dumps(grounding.get('unsupported_research_capabilities', []))}",
            ]
        if "deterministic" in result:
            document_lines.insert(-1, f"- Deterministic capability F1: {result['deterministic']['research_capabilities']['f1']}")
        lines.extend(document_lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_benchmark(
    limit: int | None = None,
    use_llm_judge: bool = True,
    use_deterministic: bool = False,
) -> dict:
    """Run the full benchmark and save predictions/results."""
    from pipeline.multi_agent import run_pipeline

    docs = load_benchmark_docs()
    if limit:
        docs = docs[:limit]

    states = run_pipeline(docs)
    document_results = []

    for state in states:
        doc_id = state["doc_id"]
        prediction = state.get("profile", {})
        ground_truth = load_ground_truth(doc_id)
        source_doc = next((doc for doc in docs if doc["doc_id"] == doc_id), {})

        save_json(PREDICTIONS_DIR / f"{doc_id}.json", {
            "doc_id": doc_id,
            "profile": prediction,
            "cypher_queries": state.get("cypher_queries", []),
            "status": state.get("status"),
            "latency_log": state.get("latency_log", {}),
            "token_log": state.get("token_log", {}),
            "retry_count": state.get("retry_count", 0),
            "neo4j_error": state.get("neo4j_error", ""),
        })

        graph = validate_queries(state.get("cypher_queries", []))

        result = {
            "doc_id": doc_id,
            "graph": graph,
        }

        if use_deterministic:
            result["deterministic"] = score_profile(prediction, ground_truth)

        if use_llm_judge:
            from evaluation.llm_judge import judge_profile
            result["llm_judge"] = judge_profile(
                prediction,
                ground_truth,
                source_text=source_doc.get("text", ""),
            )

        document_results.append(result)

    report = {
        "documents": document_results,
        "graph": aggregate_graph(document_results),
        "reliability": summarize_reliability(states),
    }
    if use_llm_judge:
        report["llm_judge"] = aggregate_llm_judge(document_results)
    if use_deterministic:
        report["deterministic"] = aggregate_deterministic(document_results)

    save_json(RESULTS_DIR / "benchmark_report.json", report)
    write_markdown_report(report, RESULTS_DIR / "benchmark_report.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the talent profiling benchmark.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-llm-judge",
        action="store_false",
        dest="use_llm_judge",
        help="Skip semantic and hallucination LLM judging.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        dest="use_llm_judge",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_llm_judge=True)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Also calculate optional deterministic agreement metrics.",
    )
    args = parser.parse_args()

    report = run_benchmark(
        limit=args.limit,
        use_llm_judge=args.use_llm_judge,
        use_deterministic=args.deterministic,
    )
    summary = {
        "llm_judge": report.get("llm_judge"),
        "deterministic": report.get("deterministic"),
        "graph": report["graph"],
        "reliability": report["reliability"],
    }
    print(json.dumps({key: value for key, value in summary.items() if value is not None}, indent=2))


if __name__ == "__main__":
    main()
