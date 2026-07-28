from .rerank_results import get_client
from .search_utils import MODEL
from .hybrid_search import HybridSearch
from .search_utils import RRF_SEARCH_LIMIT, RRF_SEARCH_PARAMETER, load_movies

def rag(query: str, results: list[dict]) -> str:
    formatted_results = []
    for result in results:
        document = result["document"]
        formatted_results.append(
            f'{document.get("title", "")} - {document.get("description", "")[:300]}'
        )

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {formatted_results}

    Answer:"""

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

    return llm_response



def rag_command(query: str) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, RRF_SEARCH_PARAMETER, RRF_SEARCH_LIMIT)
    answer = rag(query, results)

    return {
        "search_results": [result["document"] for result in results],
        "answer": answer,
    }
