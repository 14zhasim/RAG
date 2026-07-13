import argparse
import json
from pathlib import Path
import string
from lib.keyword_search import build_command, search_command
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case "build":
            first_doc_id = build_command()
            print(f"First document ID for 'merida': {first_doc_id}")
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
