from .rerank_results import get_client
from .search_utils import MODEL
from .hybrid_search import HybridSearch
from .search_utils import RRF_SEARCH_PARAMETER, load_movies

def summarize(query: str, results: list[dict]) -> str:
    formatted_results = []
    for result in results:
        document = result["document"]
        formatted_results.append(
            f'{document.get("title", "")} - {document.get("description", "")[:300]}'
        )

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {results}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

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



def summarize_command(query: str, limit: int) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, RRF_SEARCH_PARAMETER, limit)
    answer = summarize(query, results)

    return {
        "search_results": [result["document"] for result in results],
        "answer": answer,
    }
