"""Parse an order-form PDF using its authoritative SKU catalog."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent import run_agent
from models import public_result
from tools import load_catalog, save_result

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(PROJECT_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", help="Dataset ID (of-0002) or PDF path")
    parser.add_argument(
        "--catalog", type=Path, default=PROJECT_DIR / "data/sku_catalog.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model", default=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    )
    args = parser.parse_args()
    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("Set ANTHROPIC_API_KEY in .env or the environment.")
    pdf = Path(args.document)
    if pdf.suffix.lower() != ".pdf":
        pdf = PROJECT_DIR / "data/pdfs" / f"{args.document}.pdf"
    output = args.output or PROJECT_DIR / "results" / f"{pdf.stem}.json"
    try:
        print(f"Parsing {pdf.name} with {args.model}...", flush=True)
        extraction, validation = run_agent(
            pdf, load_catalog(args.catalog, model=args.model), model=args.model
        )
        save_result(output, public_result(extraction.model_dump(mode="json")))
        print(
            f"{len(extraction.contract_items)} items; {validation.status}; saved {output}"
        )
        for message in dict.fromkeys(issue.message for issue in validation.issues):
            print(f"Review: {message}")
    except Exception as exc:
        print(f"Parse failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
