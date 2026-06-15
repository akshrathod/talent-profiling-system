"""
Graph construction agent. Single responsibility: take a validated
researcher profile dict and generate Cypher queries to insert it
into Neo4j.

This agent knows nothing about document text or extraction.
It only knows about graph schema and Cypher syntax.

Self-healing: if Neo4j returns a syntax error, the pipeline
routes back here with the error and this agent fixes its own query.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
# import anthropic

load_dotenv()

# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    # Replcae it with Anthropic client if needed

# ── System Prompt ─────────────────────────────────────────────────────────────
INCLUDE_COLLABORATIONS = False
SYSTEM_PROMPT = f"""
You are a specialized Neo4j Cypher generation agent.
Your only job is to convert a structured researcher profile JSON into Cypher queries.

Graph schema rules you must follow:
- Researcher node: (:Researcher {{name: string}})
- Skill node: (:Skill {{name: string}})
- Framework node: (:Framework {{name: string}})
- Language node: (:Language {{name: string}})
- Domain node: (:Domain {{name: string}})
- Institution node: (:Institution {{name: string}})

Relationships to generate:
- (:Researcher)-[:EXPERT_IN]->(:Skill)
- (:Researcher)-[:USES_FRAMEWORK]->(:Framework)
- (:Researcher)-[:PROGRAMS_IN]->(:Language)
- (:Researcher)-[:RESEARCHES]->(:Domain)
- (:Researcher)-[:AFFILIATED_WITH]->(:Institution)
{"- (:Researcher)-[:COLLABORATED_WITH]->(:Researcher)" if INCLUDE_COLLABORATIONS else "- Do NOT generate any COLLABORATED_WITH edges"}

Critical rules:
- The profile contains a researchers list with ALL authors
- Create a full Researcher node for EVERY author in the list
- Every author gets ALL skills, frameworks, languages, domains from the paper
- Always use MERGE not CREATE to avoid duplicates
- If a previous query used CREATE and failed, replace CREATE with MERGE, do not retry with CREATE again
- MERGE researchers on name only: MERGE (r:Researcher {{name: 'John Smith'}})
- Do not generate any id field on Researcher nodes
- Each query must end with a semicolon
- Return ONLY a JSON array of Cypher query strings
- No markdown, no explanation, no code blocks
"""


# ── Cypher Generation ─────────────────────────────────────────────────────────

def generate_cypher(profile: dict, doc_id: str = "unknown", error: str = None, failed_queries: list = None) -> dict:
    """
    Takes a validated profile dict and returns a list of Cypher queries.
    If error is provided, the agent attempts to fix the broken query.
    """

    if error and failed_queries:
        user_message = f"""
    The following Cypher queries failed with this error:
    ERROR: {error}

    Failed queries:
    {json.dumps(failed_queries, indent=2)}

    Original profile:
    {json.dumps(profile, indent=2)}

    Fix only the failed queries and return the corrected JSON array of Cypher strings.
    """
    elif error:
        user_message = f"""
    Cypher generation previously failed with this error:
    ERROR: {error}

    Original profile:
    {json.dumps(profile, indent=2)}

    Regenerate the Cypher queries correctly.
    """
    else:
        user_message = f"""
    Generate Cypher queries for this researcher profile:
    {json.dumps(profile, indent=2)}
    """

    # response = client.messages.create(
    #     model     = "claude-haiku-4-5",
    #     max_tokens= 4096,
    #     system    = SYSTEM_PROMPT,
    #     messages  = [
    #         {
    #             "role"   : "user",
    #             "content": user_message
    #         }
    #     ]
    # )

    # raw_output = response.content[0].text.strip()
        
    response = client.chat.completions.create(
        model      = "gpt-4o-mini",
        max_tokens = 16384,
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]
    )
    raw_output    = response.choices[0].message.content.strip()
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    if "```" in raw_output:
        parts = raw_output.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.strip().startswith("["):
                raw_output = part.strip()
                break

    try:
        queries = json.loads(raw_output)
        if not isinstance(queries, list):
            raise ValueError("Expected a JSON array of strings")
        return {
            "queries_list"      : queries,
            "input_tokens" : input_tokens,
            "output_tokens": output_tokens,
        }
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Graph agent returned invalid output for {doc_id}: {e}\nRaw: {raw_output[:300]}")


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    test_profile = {
        "researchers"          : ["Junaed Younus Khan", "Md. Al-Amin", "Tasnim Ahmed"],
        "institution"          : "Bangladesh University of Engineering and Technology",
        "research_domains"     : ["Natural Language Processing", "Fake News Detection"],
        "technical_skills"     : ["Feature Engineering", "Text Classification", "LSTM"],
        "ml_frameworks"        : ["TensorFlow", "Keras", "scikit-learn"],
        "programming_languages": ["Python", "R"],
    }

    # print("Testing graph agent with sample profile\n")
    # queries = generate_cypher(test_profile, doc_id="test")

    # print(f"Generated {len(queries['queries_list'])} Cypher queries:\n")
    # for i, q in enumerate(queries['queries_list']):
    #     print(f"  [{i+1}] {q}")

    print("\nTesting self-healing with a fake error\n")
    error_msg = "SyntaxError: Invalid input 'SET': expected 'WHERE' or 'RETURN'"
    fixed_queries = generate_cypher(test_profile, doc_id="test", error=error_msg)

    print(f"Fixed queries ({len(fixed_queries['queries_list'])}):\n")
    for i, q in enumerate(fixed_queries['queries_list']):
        print(f"  [{i+1}] {q}")
    print(f"\nTokens: {fixed_queries['input_tokens']} in / {fixed_queries['output_tokens']} out")