# """
# Naive single-agent pipeline. One LLM call, one massive prompt.
# Deliberately monolithic to benchmark its failures against
# the multi-agent architecture.

# Metrics captured per document:
# - latency
# - JSON parse success/failure
# - fields extracted vs fields missed
# - token usage
# """

# import os
# import json
# import time
# from pathlib import Path
# from dotenv import load_dotenv
# import anthropic
# from openai import OpenAI
# from ingestion.base_ingester import ingest

# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    # Replcae it with Anthropic client if needed

# #client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# EXTRACTION_PROMPT = """
# You are a talent profiling system. Given the research document below, extract a complete researcher profile.

# You must return ONLY a valid JSON object with exactly this structure. No explanation, no markdown, no extra text:

# {
#     "researcher_name": "full name of the primary researcher or first author",
#     "institution": "university or research lab affiliation",
#     "research_domains": ["list", "of", "research", "areas"],
#     "technical_skills": ["list", "of", "specific", "technical", "skills"],
#     "ml_frameworks": ["pytorch", "tensorflow", "jax", "etc"],
#     "programming_languages": ["python", "c++", "etc"],
#     "key_contributions": ["brief", "descriptions", "of", "contributions"],
#     "collaborators": ["names", "of", "co-authors"],
#     "publication_year": "year as string"
# }

# Rules:
# - Every field must be present even if empty list or unknown
# - Return ONLY the JSON object, nothing else
# - Do not wrap in markdown code blocks

# DOCUMENT:
# {text}
# """


# def extract_profile(doc: dict) -> dict:
#     """Run a single LLM call to extract everything from one document."""
#     start_time = time.time()

#     text = doc["text"]
#     if len(text) > 12000:
#         text = text[:12000]

#     try:
#         response = client.chat.completions.create(
#             model      = "gpt-4o-mini",
#             max_tokens = 1024,
#             messages   = [
#                 {"role": "system", "content": "You are a talent profiling system. Return only valid JSON."},
#                 {"role": "user",   "content": EXTRACTION_PROMPT.format(text=text)}
#             ]
#         )

#         latency       = round(time.time() - start_time, 2)
#         raw_output    = response.choices[0].message.content.strip()
#         input_tokens  = response.usage.prompt_tokens
#         output_tokens = response.usage.completion_tokens

#         # Anthropic client code
#         # response = client.messages.create(
#         #     model      = "claude-haiku-4-5",
#         #     max_tokens = 1024,
#         #     messages   = [
#         #         {
#         #             "role"   : "user",
#         #             "content": EXTRACTION_PROMPT.format(text=text)
#         #         }
#         #     ]
#         # )

#         # latency     = round(time.time() - start_time, 2)
#         # raw_output  = response.content[0].text.strip()
#         # input_tokens  = response.usage.input_tokens
#         # output_tokens = response.usage.output_tokens

#         try:
#             profile = json.loads(raw_output)
#             parse_success = True
#         except json.JSONDecodeError as e:
#             profile = {"raw_output": raw_output, "parse_error": str(e)}
#             parse_success = False

#         result = {
#             "doc_id"       : doc["doc_id"],
#             "parse_success": parse_success,
#             "latency_sec"  : latency,
#             "input_tokens" : input_tokens,
#             "output_tokens": output_tokens,
#             "profile"      : profile
#         }

#         return result

#     except Exception as e:
#         latency = round(time.time() - start_time, 2)
#         return {
#             "doc_id"       : doc["doc_id"],
#             "parse_success": False,
#             "latency_sec"  : latency,
#             "input_tokens" : 0,
#             "output_tokens": 0,
#             "profile"      : {"error": str(e)}
#         }


# def score_completeness(profile: dict) -> dict:
#     """
#     Score how complete the extracted profile is.
#     This is how we quantify single agent failures.
#     """
#     expected_fields = [
#         "researcher_name",
#         "institution",
#         "research_domains",
#         "technical_skills",
#         "ml_frameworks",
#         "programming_languages",
#         "key_contributions",
#         "collaborators",
#         "publication_year"
#     ]

#     if not isinstance(profile, dict) or "error" in profile or "parse_error" in profile:
#         return {"score": 0.0, "filled": 0, "empty": len(expected_fields), "missing": expected_fields}

#     filled  = 0
#     empty   = 0
#     missing = []

#     for field in expected_fields:
#         value = profile.get(field, None)
#         if value and value != "unknown" and value != [] and value != "":
#             filled += 1
#         else:
#             empty += 1
#             missing.append(field)

#     score = round(filled / len(expected_fields), 2)
#     return {"score": score, "filled": filled, "empty": empty, "missing": missing, "total": len(expected_fields)}


# def run_benchmark(limit: int = 10):
#     """
#     Run single agent on N documents and collect failure metrics.
#     This is the evidence that justifies multi-agent architecture.
#     """
#     print("SINGLE AGENT BENCHMARK")
#     print("=" * 50)
#     print(f"Running on {limit} documents\n")

#     docs    = ingest("parsed", limit=limit)
#     results = []

#     for i, doc in enumerate(docs):
#         print(f"[{i+1}/{limit}] Processing: {doc['doc_id'][:50]}")

#         result       = extract_profile(doc)
#         completeness = score_completeness(result["profile"])
#         result["completeness"] = completeness

#         status = "OK" if result["parse_success"] else "FAIL"
#         print(f"  Status      : {status}")
#         print(f"  Latency     : {result['latency_sec']}s")
#         print(f"  Completeness: {completeness['score']} ({completeness['filled']}/{completeness['total']} fields)")
#         print(f"  Tokens      : {result['input_tokens']} in / {result['output_tokens']} out")
#         if completeness["missing"]:
#             print(f"  Missing     : {completeness['missing']}")
#         print()

#         results.append(result)

#     print("=" * 50)
#     print("BENCHMARK SUMMARY")
#     print("=" * 50)

#     total           = len(results)
#     parse_failures  = sum(1 for r in results if not r["parse_success"])
#     avg_latency     = round(sum(r["latency_sec"] for r in results) / total, 2)
#     avg_completeness = round(sum(r["completeness"]["score"] for r in results) / total, 2)
#     total_tokens    = sum(r["input_tokens"] + r["output_tokens"] for r in results)

#     print(f"Documents processed : {total}")
#     print(f"JSON parse failures : {parse_failures}/{total}")
#     print(f"Avg latency         : {avg_latency}s per document")
#     print(f"Avg completeness    : {avg_completeness} ({int(avg_completeness*100)}%)")
#     print(f"Total tokens used   : {total_tokens}")
#     print()

#     all_missing = []
#     for r in results:
#         all_missing.extend(r["completeness"].get("missing", []))

#     if all_missing:
#         from collections import Counter
#         field_failures = Counter(all_missing)
#         print("Most commonly missed fields:")
#         for field, count in field_failures.most_common():
#             print(f"  {field}: missed {count}/{total} times")

#     output_path = Path("data/single_agent_benchmark.json")
#     with open(output_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nFull results saved to {output_path}")

#     return results


# if __name__ == "__main__":
#     run_benchmark(limit=10)