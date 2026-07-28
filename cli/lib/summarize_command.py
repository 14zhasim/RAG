from .hybrid_search import HybridSearch
from .rerank_results import get_client
from .search_utils import MODEL, RRF_SEARCH_PARAMETER, load_movies


def format_documents(results: list[dict]) -> list[str]:
    formatted_results = []
    for result in results:
        document = result["document"]
        formatted_results.append(
            f'{document.get("title", "")} - {document.get("description", "")[:300]}'
        )
    return formatted_results


def format_citation_documents(results: list[dict]) -> list[str]:
    formatted_results = []
    for i, result in enumerate(results, start=1):
        document = result["document"]
        formatted_results.append(
            f'[{i}] {document.get("title", "")} - {document.get("description", "")[:300]}'
        )
    return formatted_results


def call_llm(prompt: str) -> str:
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


def summarize(query: str, results: list[dict]) -> str:
    formatted_results = "\n".join(format_documents(results))
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {formatted_results}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    return call_llm(prompt)


def answer_with_citations(query: str, results: list[dict]) -> str:
    formatted_results = "\n".join(format_citation_documents(results))
    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {formatted_results}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    return call_llm(prompt)


def summarize_command(query: str, limit: int) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, RRF_SEARCH_PARAMETER, limit)
    answer = summarize(query, results)

    return {
        "search_results": [result["document"] for result in results],
        "answer": answer,
    }


def citations_command(query: str, limit: int) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, RRF_SEARCH_PARAMETER, limit)
    answer = answer_with_citations(query, results)

    return {
        "search_results": [result["document"] for result in results],
        "answer": answer,
    }
