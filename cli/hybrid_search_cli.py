import argparse
from lib.hybrid_search import normalize_score

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Return normalised search result scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of search result scores to normalize")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_score(*args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
