# Order form parser

This project turns an order-form PDF into structured JSON using Claude and a supplied SKU catalog. It extracts purchased items, pricing, dates and contract terms. Unknown facts stay null, and Python rejects items outside the catalog.

## Setup

Use Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project folder:

```text
ANTHROPIC_API_KEY=your_key_here
```

Place the supplied dataset in `data/`, with `pdfs/`, `ground_truth/` and `sku_catalog.json`. The catalog has 21 SKUs. The API key, environment, dataset and generated extraction files are ignored by Git.

## Run

```bash
# Parse one supplied PDF.
python src/main.py of-0002

# Parse your own PDF and catalog. The catalog can be PDF or JSON.
python src/main.py /path/order.pdf --catalog /path/catalog.pdf

# Score saved outputs without API calls.
python src/evaluate.py

# Extract and evaluate all supplied PDFs. This makes paid API calls.
python src/evaluate.py --live
```

Sonnet 5.5 is the default. Use `--model claude-haiku-4-5-20251001` to try Haiku. Parsing PDFs and reading a PDF catalog make paid API calls.

Outputs go in `results/` and always use the supplied ground-truth fields and key order: three contract dates and thirteen fields per item. Extra metadata, evidence, API traces and run information are not saved. Review warnings appear in the terminal. Use `--output /path/result.json` to choose the filename; otherwise a new parse replaces that PDF's saved result.

## How it works

All code is in `src/`:

- `main.py`: command-line inputs and JSON output.
- `agent.py`: Claude instructions and a bounded tool loop.
- `tools.py`: PDF reading, text search, page images, catalog loading and Decimal arithmetic.
- `models.py`: Pydantic schemas, source checks and business validation.
- `evaluate.py`: comparison against the supplied answers, including repeated SKUs.
wh
Claude receives the original PDF, indexed text and catalog. It can read/search pages, inspect images, calculate totals and submit an extraction. Python resolves source references, fills canonical SKU names/IDs and validates the submission. Errors return feedback. The loop allows at most six model turns and one warning-repair attempt. Invalid results are not saved; unresolved warnings remain flagged for review. Evidence is used internally, without a separate diagnostic file.

## Results

Scoring the saved Sonnet extractions across all 50 supplied PDFs:

| Measure | Result |
|---|---:|
| PDFs completed | 50/50 |
| Items matched; no missing or extra | 132/132 |
| Core fields correct | 1,462/1,470 (99.46%) |
| All labeled fields correct | 1,462/1,866 (78.35%) |
| Documents matching every label | 0/50 |

Core scoring excludes billing type, frequency and notes. Full scoring includes them. Amounts, dates and strings require exact matches; quantity tolerance is 0.000001. Repeated products are matched one-to-one. Failures and coverage are reported separately in the single `results/benchmark.json` file, which is included in Git.

All 50 forms had review warnings. Some expected billing facts are absent from the PDF wording, note wording differs, and eight quantity labels conflict with the catalog's support-hours instructions. These disagreements remain in the scores. The dataset was used during development; this is not held-out production accuracy.

The simplified version passed 92 regression checks run outside this folder, three authored edge-case extractions, and a live PDF-catalog check. Notes extraction now checks for omitted contract conditions; explicit billing omissions receive repair feedback. Spot checks of of-0001 and of-0002 confirmed populated source-supported notes. Source-text matching does not prove correct interpretation, and scanned quotations need manual review.
