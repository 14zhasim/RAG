import argparse
from lib.rag_prompt import rag_command
from lib.search_utils import RRF_SEARCH_LIMIT
from lib.summarize_command import citations_command, summarize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Perform AI-generated summary")
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, default=RRF_SEARCH_LIMIT, help="search result limit")

    citations_parser = subparsers.add_parser("citations", help="Perform RAG with citation-aware answer")
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, default=RRF_SEARCH_LIMIT, help="search result limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Summary:")
            print(result["answer"])
        case "citations":
            result = citations_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"- {document['title']}")
            print()
            print("LLM Answer:")
            print(result["answer"])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
