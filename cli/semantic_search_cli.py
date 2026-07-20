import argparse
from lib.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text, search
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_model_parser = subparsers.add_parser("verify", help="Verify embedding model is created")
    embed_parser = subparsers.add_parser("embed_text", help="Verify embedding model is created")
    embed_parser.add_argument("text", type=str, help="text you want to embed")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings made for movie list")
    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query")
    embed_query_parser.add_argument("query", type=str, help="search query you want to embed")
    search_parser = subparsers.add_parser("search", help="enter seach query and get search results")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="number of search results")


    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model()  
        case "embed_text":
            embed_text(args.text) 
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
