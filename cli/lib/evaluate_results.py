from .rerank_results import get_client
from .search_utils import MODEL
import json

def evaluate_results(query: str, results: list[dict]) -> list[int]:
    formatted_results = []
    for result in results:
        document = result["document"]
        formatted_results.append(
            f'{document.get("title", "")} - {document.get("description", "")[:300]}'
        )

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

            Query: "{query}"

            Results:
            {chr(10).join(formatted_results)}

            Scale:
            - 3: Highly relevant
            - 2: Relevant
            - 1: Marginally relevant
            - 0: Not relevant

            Do NOT give any numbers other than 0, 1, 2, or 3.

            Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

            [2, 0, 3, 2, 0, 1]"""

    message = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=message,
    )

    if response.usage is None:
        raise RuntimeError("API response has no usage data")
    
    # Keep the raw model output small and predictable for the caller to parse.
    llm_response = (response.choices[0].message.content or "").strip().strip('"')

    parsed_response = json.loads(llm_response)

    return parsed_response
