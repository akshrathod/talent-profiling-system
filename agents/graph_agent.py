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
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
# import anthropic

load_dotenv()

# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    # Replcae it with Anthropic client if needed

# System Prompt
INCLUDE_COLLABORATIONS = False
SYSTEM_PROMPT = f"""
You are a Neo4j Cypher generation agent.
Your only job: convert a researcher profile JSON into valid Cypher queries.

GRAPH SCHEMA:
Nodes:
  (:Researcher {{name: string}})
  (:ResearchCapability {{name: string}})
  (:Institution {{name: string}})

Relationships:
  (:Researcher)-[:RESEARCHES_IN]->(:ResearchCapability)
  (:Researcher)-[:AFFILIATED_WITH]->(:Institution)
  {"(:Researcher)-[:COLLABORATED_WITH]->(:Researcher)" if INCLUDE_COLLABORATIONS else "Do NOT generate COLLABORATED_WITH edges"}

GENERATION RULES:
1. Create a Researcher node for EVERY author in the researchers list
2. Every author receives ALL research capabilities from the profile
3. If institution is present, every author receives that institution from the profile
4. Always MERGE on name only: MERGE (r:Researcher {{name: 'John Smith'}})
5. Never add an id field to any node
6. If institution is empty, missing, or unknown, do not generate Institution nodes or AFFILIATED_WITH relationships

CYPHER SYNTAX RULES - FOLLOW EXACTLY:
7. Always use MERGE, never CREATE under any circumstances
8. Every relationship statement must be fully self-contained with explicit node properties.
   WRONG - bare variables have no context across statements:
     MERGE (r)-[:RESEARCHES_IN]->(c)
   CORRECT - both nodes explicitly defined in the same statement:
     MERGE (r:Researcher {{name: 'X'}}) MERGE (c:ResearchCapability {{name: 'Y'}}) MERGE (r)-[:RESEARCHES_IN]->(c)
9. Each string in the output array must contain exactly ONE Cypher statement
10. Do not include semicolons
11. Do not chain multiple statements inside a single string

OUTPUT RULES:
12. Return ONLY a valid JSON array of Cypher query strings
13. No markdown, no code blocks, no explanation, no preamble
14. Generate in this order: researcher nodes, institution node if present, research capability nodes, then all relationship statements
15. Example output format:
    [
      "MERGE (r:Researcher {{name: 'John Smith'}})",
      "MERGE (i:Institution {{name: 'MIT'}})",
      "MERGE (c:ResearchCapability {{name: 'Machine Learning Benchmarking'}})",
      "MERGE (r:Researcher {{name: 'John Smith'}}) MERGE (i:Institution {{name: 'MIT'}}) MERGE (r)-[:AFFILIATED_WITH]->(i)",
      "MERGE (r:Researcher {{name: 'John Smith'}}) MERGE (c:ResearchCapability {{name: 'Machine Learning Benchmarking'}}) MERGE (r)-[:RESEARCHES_IN]->(c)"
    ]
"""


# Cypher Generation

def generate_cypher(profile: dict, doc_id: str = "unknown", error: str = None, failed_queries: list = None) -> dict:
    """
    Takes a validated profile dict and returns a list of Cypher queries.
    If error is provided, the agent attempts to fix the broken query.
    """

    if error and failed_queries:
        user_message = f"""
    These Cypher queries failed with this error:
    ERROR: {error}

    Failed queries:
    {json.dumps(failed_queries, indent=2)}

    Use this profile data to fix the queries with correct values:
    Institution: {profile.get('institution', '')}
    Research capabilities: {json.dumps(profile.get('research_capabilities', []))}

    Rules:
    - Generate ONLY the fixed versions of the failed queries
    - Do not regenerate queries that already succeeded
    - Never use placeholder values like 'Your Domain Name'
    - Always use actual values from the profile data above
    - Never use inline path patterns
    - Do not include semicolons
    - Return a JSON array of corrected Cypher strings
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
    Generate Cypher queries for this researcher profile.
    Create all nodes first, then all relationships.
    Every relationship statement must be fully self-contained. Never use bare variables.

    Profile:
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

    # Remove invalid backslash escapes that break json.loads
    #raw_output = raw_output.replace("\\'", "'")
    raw_output = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_output)
    
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


# Test

if __name__ == "__main__":
    import json

    test_profile = {
        "researchers"          : ["Junaed Younus Khan", "Md. Al-Amin", "Tasnim Ahmed"],
        "institution"          : "Bangladesh University of Engineering and Technology",
        "research_capabilities": ["Natural Language Processing", "Fake News Detection", "Machine Learning Benchmarking"],
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
