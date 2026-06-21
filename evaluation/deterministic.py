"""
Deterministic extraction metrics for benchmark profiles.

Compares pipeline predictions against manually authored ground-truth JSON files
with normalized string matching and lightweight fuzzy matching.
"""

from difflib import SequenceMatcher
import re
from typing import Any


DEFAULT_THRESHOLD = 0.84


ALIASES = {
    "nlp": "natural language processing",
    "gnn": "graph neural network",
    "gnns": "graph neural network",
    "llm": "large language model",
    "llms": "large language model",
    "bert based models": "bert",
    "bi lstm": "bilstm",
    "bi-lstm": "bilstm",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "research agendas for ai safety and openness": "ai safety research strategy",
    "ai safety agenda development": "ai safety research strategy",
    "evaluation of machine learning models across datasets": "machine learning benchmarking",
    "ethical concerns in ai": "ai ethics analysis",
}

VERBOSE_PREFIX_PATTERN = re.compile(
    r"^(?:development|application|use|analysis|study) of (?:the )?"
)


def normalize_text(value: Any) -> str:
    """Normalize labels for matching while preserving semantic content."""
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9+#.\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = VERBOSE_PREFIX_PATTERN.sub("", text)
    return ALIASES.get(text, text)


def similarity(left: str, right: str) -> float:
    """Return a forgiving string similarity score."""
    a = normalize_text(left)
    b = normalize_text(right)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return SequenceMatcher(None, a, b).ratio()


def dedupe(values: list[str]) -> list[str]:
    """Deduplicate values after normalization, keeping the first spelling."""
    seen = set()
    result = []
    for value in values or []:
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def match_items(predicted: list[str], expected: list[str], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Greedily match predicted values to expected values.
    Returns precision, recall, F1, and match details.
    """
    predicted = dedupe(predicted)
    expected = dedupe(expected)
    matches = []
    used_expected = set()

    for pred in predicted:
        best_index = None
        best_score = 0.0
        for i, exp in enumerate(expected):
            if i in used_expected:
                continue
            score = similarity(pred, exp)
            if score > best_score:
                best_index = i
                best_score = score
        if best_index is not None and best_score >= threshold:
            used_expected.add(best_index)
            matches.append({
                "predicted": pred,
                "expected": expected[best_index],
                "score": round(best_score, 3),
            })

    matched_predicted = {m["predicted"] for m in matches}
    matched_expected = {m["expected"] for m in matches}
    precision = len(matches) / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = len(matches) / len(expected) if expected else (1.0 if not predicted else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matches": matches,
        "missing": [item for item in expected if item not in matched_expected],
        "extra": [item for item in predicted if item not in matched_predicted],
    }


def institution_score(predicted: str, expected: str) -> dict:
    """Score the single profile-level institution field."""
    score = similarity(predicted, expected)
    return {
        "score": round(score, 3),
        "match": score >= DEFAULT_THRESHOLD,
        "predicted": predicted or "",
        "expected": expected or "",
    }


def score_profile(predicted: dict, ground_truth: dict) -> dict:
    """Score researchers, institution, and research capabilities for one document."""
    researchers = match_items(
        predicted.get("researchers", []),
        ground_truth.get("researchers", []),
        threshold=0.9,
    )
    capabilities = match_items(
        predicted.get("research_capabilities", []),
        ground_truth.get("research_capabilities", []),
        threshold=DEFAULT_THRESHOLD,
    )
    institution = institution_score(
        predicted.get("institution", ""),
        ground_truth.get("institution", ""),
    )

    overall = (
        0.35 * researchers["f1"]
        + 0.20 * institution["score"]
        + 0.45 * capabilities["f1"]
    )

    return {
        "researchers": researchers,
        "institution": institution,
        "research_capabilities": capabilities,
        "overall_extraction_score": round(overall, 3),
    }
