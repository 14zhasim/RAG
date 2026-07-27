import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import time
import json


load_dotenv()
model = "openrouter/free"


def get_client():
    # Build the API client lazily so commands that do not use query enhancement
    # can run without requiring OPENROUTER_API_KEY.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def get_rerank_score(prompt: str) -> str:
    message = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=message,
    )

    if response.usage is None:
        raise RuntimeError("API response has no usage data")
    
    # Keep the raw model output small and predictable for the caller to parse.
    enhanced_query = (response.choices[0].message.content or "").strip().strip('"')
    return enhanced_query if enhanced_query else 0

def parse_rerank_score(llm_rank: str) -> float:
    # Individual reranking asks for a number, but LLMs can still return extra text.
    # Extract the first valid 0-10 score so one bad response does not crash the CLI.
    match = re.search(r"\b(?:10(?:\.0+)?|[0-9](?:\.\d+)?)\b", llm_rank)
    if match is None:
        return 0.0

    score = float(match.group())
    return max(0.0, min(10.0, score))

def individual_rerank(query: str, results: list[dict]) -> list[dict]:
    for result in results:
        document = result["document"]
        prompt = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {document.get("title", "")} - {document.get("description", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:"""
        llm_rank = get_rerank_score(prompt)
        result["Re-rank Score"] = parse_rerank_score(llm_rank)
        time.sleep(2)
    
    sorted_results = sorted(results, key=lambda item: item["Re-rank Score"], reverse=True)

    return sorted_results

def batch_rerank(query: str, results: list[dict]) -> list[dict]:
    # Send a compact, readable list instead of raw nested result dictionaries.
    # This gives the LLM exactly the IDs it must return and avoids prompt noise.
    doc_list = []
    for result in results:
        doc = result["document"]
        doc_list.append(
            f'{doc["id"]}. {doc.get("title", "")} - {doc.get("description", "")[:300]}'
        )
    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return the movie IDs in order of relevance, best match first.

            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:"""
    
    llm_rank_list = json.loads(get_rerank_score(prompt))

    # Convert the LLM's ordered ID list into O(1) rank lookups.
    rank_by_doc_id = {
        doc_id: rank
        for rank, doc_id in enumerate(llm_rank_list, start=1)
    }

    fallback_rank = len(results) + 1

    for original_rank, result in enumerate(results, start=1):
        doc_id = result["document"]["id"]
        # The LLM should include every ID, but if it omits one, sort that result
        # after all ranked docs while preserving its original RRF order.
        result["_rerank_sort_key"] = rank_by_doc_id.get(
            doc_id,
            fallback_rank + original_rank,
        )

    sorted_results = sorted(
        results,
        key=lambda item: item["_rerank_sort_key"],
    )

    for display_rank, result in enumerate(sorted_results, start=1):
        # Keep the printed rank clean and gap-free; the fallback math above is
        # only an internal sort key.
        result["Re-rank Rank"] = display_rank
        del result["_rerank_sort_key"]

    return sorted_results
    

def rerank_results(method, query: str, results: list[dict]) -> list[dict]:
    match method:
        case "individual":
            return individual_rerank(query, results)
        case "batch":
            return batch_rerank(query, results)
        case _:
            return results
