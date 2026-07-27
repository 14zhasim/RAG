import argparse
from lib.hybrid_search import normalize_score, HybridSearch
from lib.search_utils import DEFAULT_ALPHA, HYBRID_SEARCH_LIMIT, load_movies, RRF_SEARCH_LIMIT, RRF_SEARCH_PARAMETER, SEARCH_ENHANCEMENT_METHODS, RERANK_ENHANCEMENT_METHODS
from lib.enhance_query import enhance_query
from lib.rerank_results import rerank_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Return normalised search result scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of search result scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Return weighted hybrid search results")
    weighted_search_parser.add_argument("query", type=str, help="User's search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weighting parameter between keyword and semantic search")
    weighted_search_parser.add_argument("--limit", type=int, default=HYBRID_SEARCH_LIMIT, help="Hybrid search result limit")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Return hybrid search using rrf-scoring")
    rrf_search_parser.add_argument("query", type=str, help="User's search query")
    rrf_search_parser.add_argument("-k", type=int, default=RRF_SEARCH_PARAMETER, help="adjust parameter for how much weight placed on top-ranking results")
    rrf_search_parser.add_argument("--limit", type=int, default=RRF_SEARCH_LIMIT, help="Hybrid search result limit")
    rrf_search_parser.add_argument("--enhance", type=str, choices=SEARCH_ENHANCEMENT_METHODS, help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=RERANK_ENHANCEMENT_METHODS, help="Query reranking enhancement method")


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
        case "rrf-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            limit = args.limit * (5 if args.rerank_method else 1)
            # argparse choices already validates the enhancement method, so any
            # provided value can be passed through to the enhancement helper.
            if args.enhance:
                enhanced_query = enhance_query(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                results = hybrid_search.rrf_search(enhanced_query, args.k, limit)
            else:
                results = hybrid_search.rrf_search(args.query, args.k, limit)
            
            if args.rerank_method:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
                unfilitered_results = rerank_results(args.rerank_method, args.query, results)
                results = unfilitered_results[:args.limit]
            print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")

            for i, result in enumerate(results, start=1):
                document = result["document"]
                print(f"{i}. {document['title']}")
                if "Re-rank Score" in result:
                    print(f"   Re-rank Score: {result.get('Re-rank Score', 0):.3f}/10")    
                if "Re-rank Rank" in result:
                    print(f"   Re-rank Rank: {result.get('Re-rank Rank', 0)}")  
                if "Cross Encoder Score" in result:
                    print(f"   Cross Encoder Score: {result.get('Cross Encoder Score', 0)}:.3f")  
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                print(f"   {document['description'][:100]}...")            

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
