"""
LLM-based judges for extracted profiles.

Semantic judging compares predictions to ground truth. Source-grounding judging
checks whether predictions are supported by the same source window used during extraction.
"""

import json
import os

from pipeline.config import MAX_SOURCE_CHARS


def _get_client():
    """Create the OpenAI client lazily so non-LLM evaluation stays dependency-light."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass

    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SEMANTIC_SYSTEM_PROMPT = """
You are a semantic evaluation judge for a research talent profiling system.
Compare the predicted profile to the human-written ground-truth profile by meaning,
not by exact wording. Treat synonymous and conceptually equivalent concise phrases
as matches. The ground truth is curated but may not be exhaustive, so do not assume
that every prediction absent from ground truth is incorrect.

Return ONLY valid JSON with this exact structure:
{
  "researcher_correctness_score": 0,
  "institution_correctness_score": 0,
  "capability_semantic_alignment_score": 0,
  "capability_coverage_score": 0,
  "overall_profile_quality_score": 0,
  "notes": "brief explanation"
}

All scores use 0-10, where higher is always better:
- researcher_correctness_score: accuracy and completeness of researcher names
- institution_correctness_score: semantic correctness of the primary institution
- capability_semantic_alignment_score: conceptual agreement between predicted and expected capabilities, without requiring exact phrase matches
- capability_coverage_score: coverage of important ground-truth capabilities; 10 means no important expected capability is missing
- overall_profile_quality_score: holistic semantic quality of the profile

Do not perform source-grounding or hallucination assessment here. That is handled
separately using the source document.
"""


SOURCE_GROUNDING_SYSTEM_PROMPT = """
You are a source-grounding evaluator for a research talent profiling system.
Determine whether each predicted researcher, institution, and research capability
is supported by the source document. Do not use ground truth for this assessment.

Support does NOT require an exact word or phrase match. Judge support from the
context and overall parsed text. A concise capability is supported when it is
explicitly stated or reasonably inferable from the authors' own methods,
experiments, analysis, implementation, or contributions.

Do not count a capability as supported when it appears only in related work,
citations, background discussion, or a list of compared methods that the authors
did not meaningfully use or contribute to. Do not infer career-wide expertise from
this single paper.

Return ONLY valid JSON with this exact structure:
{
  "unsupported_researchers": [],
  "unsupported_institution": false,
  "supported_research_capabilities": [],
  "unsupported_research_capabilities": [],
  "capability_assessments": [
    {
      "capability": "concise predicted capability",
      "supported": true,
      "evidence": "short paraphrase of supporting source context or reason it is unsupported"
    }
  ],
  "source_grounding_score": 0,
  "notes": "brief explanation"
}

The source_grounding_score uses 0-10, where higher is always better:
- 10 means every predicted item is clearly supported by explicit or contextual evidence
- 5 means the profile mixes supported and unsupported claims
- 0 means the profile is almost entirely unsupported

Assess every predicted research capability exactly once in capability_assessments.
Evidence must be a short paraphrase, not a long quotation.
"""


def _parse_json_response(raw_output: str) -> dict:
    """Parse a JSON-only judge response, tolerating accidental code fences."""
    raw_output = raw_output.strip()
    if "```" in raw_output:
        parts = raw_output.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw_output = part
                break
    return json.loads(raw_output)


def judge_semantic_profile(predicted: dict, ground_truth: dict, model: str = "gpt-4o-mini") -> dict:
    """Ask an LLM to semantically grade one predicted profile against ground truth."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Ground truth profile:\n"
                    f"{json.dumps(ground_truth, indent=2)}\n\n"
                    "Predicted profile:\n"
                    f"{json.dumps(predicted, indent=2)}"
                ),
            },
        ],
    )

    result = _parse_json_response(response.choices[0].message.content)
    result["input_tokens"] = response.usage.prompt_tokens
    result["output_tokens"] = response.usage.completion_tokens
    return result


def judge_source_grounding(
    predicted: dict,
    source_text: str,
    model: str = "gpt-4o-mini",
    max_source_chars: int = MAX_SOURCE_CHARS,
) -> dict:
    """Ask an LLM to check whether predictions are supported by the source document."""
    client = _get_client()
    truncated_source = source_text[:max_source_chars]
    response = client.chat.completions.create(
        model=model,
        max_tokens=900,
        messages=[
            {"role": "system", "content": SOURCE_GROUNDING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Source document text:\n"
                    f"{truncated_source}\n\n"
                    "Predicted profile:\n"
                    f"{json.dumps(predicted, indent=2)}"
                ),
            },
        ],
    )

    result = _parse_json_response(response.choices[0].message.content)
    result["input_tokens"] = response.usage.prompt_tokens
    result["output_tokens"] = response.usage.completion_tokens
    return result


def judge_profile(
    predicted: dict,
    ground_truth: dict,
    source_text: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Run LLM judging.

    Includes source-grounding evaluation only when source_text is provided.
    """
    result = {
        "semantic": judge_semantic_profile(predicted, ground_truth, model=model),
    }
    if source_text is not None:
        result["source_grounding"] = judge_source_grounding(predicted, source_text, model=model)
    return result
