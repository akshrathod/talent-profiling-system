"""
Extraction agent. Single responsibility: take sanitized document
text and return a structured researcher profile as a validated
Pydantic model.

This agent only knows about extraction. It knows nothing about
Cypher, Neo4j, or routing. That separation is the entire point.
"""

import os
import json
from dotenv import load_dotenv
# import anthropic
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    # Replcae it with Anthropic client if needed
# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Output Schema ─────────────────────────────────────────────────────────────

class ResearcherProfile(BaseModel):
    """
    Validated schema for extracted researcher profile.
    Pydantic enforces types and catches missing fields
    before anything reaches the graph agent.
    """
    researchers          : list[str]   = Field(default_factory=list)
    institution          : str         = Field(default="unknown")
    research_domains     : list[str]   = Field(default_factory=list)
    technical_skills     : list[str]   = Field(default_factory=list)
    ml_frameworks        : list[str]   = Field(default_factory=list)
    programming_languages: list[str]   = Field(default_factory=list)


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a specialized researcher profile extraction agent.
Your only job is to extract structured information about researchers from academic documents.

Rules:
- Return ONLY a valid JSON object matching the schema exactly
- No markdown, no code blocks, no explanation
- Every field must be present
- Use empty list for list fields you cannot find
- Use "unknown" for string fields you cannot find
- For technical_skills extract specific technologies, methods, algorithms
- For ml_frameworks extract only software frameworks like PyTorch, TensorFlow, JAX
- For researchers extract ALL author names exactly as they appear

JSON Schema:
{
    "researchers"          : ["list of ALL author names"],
    "institution"          : "primary institution or university",
    "research_domains"     : ["list of research areas"],
    "technical_skills"     : ["list of specific skills and algorithms"],
    "ml_frameworks"        : ["list of frameworks like PyTorch, TensorFlow"],
    "programming_languages": ["list of programming languages"]
}
"""

# ── Extraction Logic ──────────────────────────────────────────────────────────

def extract(sanitized_text: str, doc_id: str = "unknown") -> dict:
    """
    Core extraction function.
    Takes sanitized text, returns validated profile dict.
    Raises ValueError if LLM returns unparseable output.
    """
    truncated_text = sanitized_text[:15000]

    # response = client.messages.create(
    #     model     = "claude-haiku-4-5",
    #     max_tokens= 2048,
    #     system    = SYSTEM_PROMPT,
    #     messages  = [
    #         {
    #             "role"   : "user",
    #             "content": f"Extract the researcher profile from this document:\n\n{truncated_text}"
    #         }
    #     ]
    # )

    # raw_output = response.content[0].text.strip()
    # input_tokens  = response.usage.input_tokens
    # output_tokens = response.usage.output_tokens

    response = client.chat.completions.create(
        model      = "gpt-4o-mini",
        max_tokens = 2048,
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Extract the researcher profile from this document:\n\n{truncated_text}"}
        ]
    )
    raw_output    = response.choices[0].message.content.strip()
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # Strip markdown code fences if LLM added them
    if "```" in raw_output:
        parts = raw_output.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.strip().startswith("{"):
                raw_output = part.strip()
                break

    try:
        raw_dict = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Extractor returned invalid JSON for {doc_id}: {e}\nRaw: {raw_output[:300]}")

    try:
        profile = ResearcherProfile(**raw_dict)
    except Exception as e:
        raise ValueError(f"Profile validation failed for {doc_id}: {e}")

    profile_dict = profile.model_dump()
    profile_dict["researchers"] = profile_dict["researchers"][:3]

    # Remove placeholder values the LLM sometimes returns for empty fields
    PLACEHOLDER_VALUES = {"unknown", "n/a", "none", "not specified", "not mentioned", ""}

    for field in ["research_domains", "technical_skills", "ml_frameworks", "programming_languages", "researchers"]:
        profile_dict[field] = [
            item for item in profile_dict[field]
            if item.strip().lower() not in PLACEHOLDER_VALUES
        ]

    return {
        "profile"      : profile_dict,
        "input_tokens" : input_tokens,
        "output_tokens": output_tokens,
    }


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion.base_ingester import ingest
    from security.pii_gateway import redact

    docs = ingest("parsed", limit=1)
    doc  = docs[0]

    print(f"Testing extractor on: {doc['doc_id']}\n")

    result_pii  = redact(doc["text"], doc_id=doc["doc_id"])
    sanitized   = result_pii["sanitized_text"]

    result = extract(sanitized, doc_id=doc["doc_id"])

    print("Extracted Profile:")
    print(json.dumps(result["profile"], indent=2))
    print(f"\nTokens: {result['input_tokens']} in / {result['output_tokens']} out")