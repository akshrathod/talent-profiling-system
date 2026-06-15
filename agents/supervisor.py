"""
Supervisor agent. Single responsibility: examine the current
pipeline state and decide what happens next.

Possible routing decisions:
- "extract"  : document ingested, send to extraction agent
- "graph"    : extraction done, send to graph agent for Cypher
- "execute"  : Cypher ready, send to Neo4j execution node
- "retry"    : Neo4j failed, send failed Cypher back to graph agent with error
- "end"      : pipeline complete or retry limit reached
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 3

SYSTEM_PROMPT = """
You are a pipeline supervisor agent for a talent profiling system.
You receive the current state of the pipeline and decide what to do next.

You must return ONLY one of these exact strings, nothing else:
- "extract"  : if sanitized_text exists but profile is empty
- "graph"    : if profile is populated but cypher_queries is empty and no error
- "execute"  : if cypher_queries is populated and no error and not yet executed
- "retry"    : if neo4j_error exists and retry_count is below limit
- "end"      : if pipeline is complete or retry limit reached

Rules:
- If sanitized_text exists but profile is empty -> return "extract"
- If profile is populated and cypher_queries is empty and no neo4j_error -> return "graph"
- If cypher_queries is populated and executed is False and no neo4j_error -> return "execute"
- If neo4j_error exists and retry_count is below limit -> return "retry"
- If neo4j_error exists and retry_count is at or above limit -> return "end"
- If executed is True and no neo4j_error -> return "end"
- Return ONLY the decision string, no explanation, no punctuation
"""


def decide(state: dict) -> str:
    """
    Takes the current pipeline state and returns a routing decision.
    Output is always validated against allowed decisions.
    """
    allowed_decisions = {"extract", "graph", "execute", "retry", "end"}

    state_summary = {
        "has_sanitized_text" : bool(state.get("sanitized_text")),
        "has_profile"        : bool(state.get("profile")),
        "has_cypher_queries" : bool(state.get("cypher_queries")),
        "executed"           : state.get("executed", False),
        "has_neo4j_error"    : bool(state.get("neo4j_error")),
        "retry_count"        : state.get("retry_count", 0),
        "max_retries"        : MAX_RETRIES,
        "status"             : state.get("status", "unknown")
    }

    response = client.chat.completions.create(
        model      = "gpt-4o-mini",
        max_tokens = 10,
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role"   : "user",
                "content": f"Current pipeline state:\n{json.dumps(state_summary, indent=2)}\n\nWhat is the next action?"
            }
        ]
    )

    decision = response.choices[0].message.content.strip().lower().strip('"').strip("'")

    if decision not in allowed_decisions:
        print(f"  Supervisor returned invalid decision: {decision}. Defaulting to end.")
        return "end"

    return decision


if __name__ == "__main__":
    print("Testing supervisor routing decisions\n")

    test_cases = [
        {
            "name"          : "Fresh document",
            "sanitized_text": "some text here",
            "profile"       : {},
            "cypher_queries": [],
            "executed"      : False,
            "neo4j_error"   : None,
            "retry_count"   : 0,
            "status"        : "ingested"
        },
        {
            "name"          : "Extraction done, needs Cyphers",
            "sanitized_text": "some text here",
            "profile"       : {"researchers": ["John Smith"], "institution": "MIT"},
            "cypher_queries": [],
            "executed"      : False,
            "neo4j_error"   : None,
            "retry_count"   : 0,
            "status"        : "extracted"
        },
        {
            "name"          : "Cypher ready, needs execution",
            "sanitized_text": "some text here",
            "profile"       : {"researchers": ["John Smith"], "institution": "MIT"},
            "cypher_queries": ["MERGE (r:Researcher {name: 'John Smith'});"],
            "executed"      : False,
            "neo4j_error"   : None,
            "retry_count"   : 0,
            "status"        : "cypher_ready"
        },
        {
            "name"          : "Neo4j failed, should retry",
            "sanitized_text": "some text here",
            "profile"       : {"researchers": ["John Smith"], "institution": "MIT"},
            "cypher_queries": ["MERGE (r:Researcher {name: 'John Smith'});"],
            "executed"      : False,
            "neo4j_error"   : "SyntaxError: Invalid Cypher",
            "retry_count"   : 1,
            "status"        : "error"
        },
        {
            "name"          : "Retry limit reached",
            "sanitized_text": "some text here",
            "profile"       : {"researchers": ["John Smith"], "institution": "MIT"},
            "cypher_queries": [],
            "executed"      : False,
            "neo4j_error"   : "SyntaxError: Invalid Cypher",
            "retry_count"   : 3,
            "status"        : "error"
        },
        {
            "name"          : "Pipeline complete",
            "sanitized_text": "some text here",
            "profile"       : {"researchers": ["John Smith"], "institution": "MIT"},
            "cypher_queries": ["MERGE (r:Researcher {name: 'John Smith'});"],
            "executed"      : True,
            "neo4j_error"   : None,
            "retry_count"   : 0,
            "status"        : "complete"
        },
    ]

    for case in test_cases:
        name = case.pop("name")
        decision = decide(case)
        print(f"  {name}")
        print(f"  Decision: {decision}\n")