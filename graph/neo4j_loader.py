"""
Neo4j driver and query execution engine.
Handles connection, constraint setup, and Cypher execution.
Includes the self-healing loop for failed queries.
"""

import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Driver

def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )

# Constraints

CONSTRAINTS = [
    "CREATE CONSTRAINT unique_researcher IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT unique_research_capability IF NOT EXISTS FOR (c:ResearchCapability) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT unique_institution IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
]


def setup_constraints(driver):
    """
    Run all constraint creation queries.
    IF NOT EXISTS means this is safe to call every time
    the pipeline starts. No error if they already exist.
    """
    with driver.session() as session:
        for constraint in CONSTRAINTS:
            session.run(constraint)
    print("Constraints verified")


# Query Execution

def execute_query(session, query: str) -> dict:
    """
    Execute a single Cypher query.
    Returns success status and any error message.
    Strips trailing semicolons since neo4j python
    driver does not need them unlike the browser.
    """
    clean_query = query.strip().rstrip(";").strip()

    if not clean_query:
        return {"success": True, "error": None}

    try:
        session.run(clean_query)
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_batch(queries: list[str], doc_id: str = "unknown") -> dict:
    """
    Execute a batch of Cypher queries for one document.
    Splits any multi-statement strings on semicolons since
    the Neo4j driver requires one statement per execution.
    """
    driver = get_driver()
    failed_queries = []
    errors         = []
    success_count  = 0

    # Flatten: split every query string into individual statements
    individual_statements = []
    for query in queries:
        parts = query.split(";")
        for part in parts:
            part = part.strip()
            if part:
                part = re.sub(r'\bCREATE\s+\(', 'MERGE (', part, flags=re.IGNORECASE)
                part = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", part)
                individual_statements.append(part)
    
    print(f"\n  --- {len(individual_statements)} normalized statements for {doc_id} ---")
    for i, stmt in enumerate(individual_statements):
        print(f"    [{i+1}] {stmt}")
    print(f"  --- end statements ---\n")

    with driver.session() as session:
        for statement in individual_statements:
            result = execute_query(session, statement)
            if result["success"]:
                success_count += 1
            else:
                failed_queries.append(statement)
                errors.append(result["error"])
                print(f"  Failed query for {doc_id}: {result['error']}")

    driver.close()

    return {
        "success"       : len(failed_queries) == 0,
        "success_count" : success_count,
        "failed_queries": failed_queries,
        "errors"        : errors,
        "executed"      : True,
    }

# Test

def test_connection() -> bool:
    """Verify Neo4j is reachable before running the pipeline."""
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run("RETURN 'connected' AS status")
            record = result.single()
            status = record["status"]
        driver.close()
        print(f"Neo4j connection: {status}")
        return True
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        return False


# Entry point

if __name__ == "__main__":
    print("Testing Neo4j connection and setup\n")

    connected = test_connection()
    if not connected:
        print("Make sure Neo4j Desktop instance is running")
        exit(1)

    driver = get_driver()
    setup_constraints(driver)
    driver.close()

    print("\nRunning test queries\n")

    test_queries = [
        "MERGE (r:Researcher {name: 'Test Researcher'})",
        "MERGE (c:ResearchCapability {name: 'Model Evaluation'})",
        "MERGE (r:Researcher {name: 'Test Researcher'}) MERGE (c:ResearchCapability {name: 'Model Evaluation'}) MERGE (r)-[:RESEARCHES_IN]->(c)",
    ]

    result = execute_batch(test_queries, doc_id="test")

    print(f"Success count : {result['success_count']}/{len(test_queries)}")
    print(f"Failed queries: {len(result['failed_queries'])}")

    if result["success"]:
        print("\nAll queries executed successfully")
        print("Check Neo4j Browser to see the test nodes")
    else:
        print(f"\nSome queries failed: {result['errors'][0]}")
