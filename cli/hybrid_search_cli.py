import argparse
from lib.hybrid_search import normalize_score, HybridSearch
from lib.search_utils import DEFAULT_ALPHA, HYBRID_SEARCH_LIMIT, load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Return normalised search result scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of search result scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Return weighted hybrid search results")
    weighted_search_parser.add_argument("query", type=str, help="User's search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weighting parameter between keyword and semantic search")
    weighted_search_parser.add_argument("--limit", type=int, default=HYBRID_SEARCH_LIMIT, help="Hybrid search result limit")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_score(*args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                document = result["document"]
                print(f"{i}. {document['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"   {document['description'][:100]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
