import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import time


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
    
    #defensive programming here - in case llm does not return valid keyword, return original
    #and also stripping LLM response of things like whitespace, speech marks etc.!
    enhanced_query = (response.choices[0].message.content or "").strip().strip('"')
    return enhanced_query if enhanced_query else 0

def parse_rerank_score(llm_rank: str) -> float:
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

def rerank_results(method, query: str, results: list[dict]) -> list[dict]:
    match method:
        case "individual":
            return individual_rerank(query, results)
        case _:
            return results
