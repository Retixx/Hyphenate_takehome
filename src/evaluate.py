"""Evaluate purchase instances one-to-one; expected answers never enter the agent."""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv

from agent import run_agent
from models import OUTPUT_ITEM_FIELDS, public_result
from tools import load_catalog, save_result

DATE_FIELDS = (
    "effective_date",
    "contract_signed_date_buying_company",
    "contract_signed_date_selling_company",
)
ITEM_FIELDS = OUTPUT_ITEM_FIELDS
CORE_FIELDS = set(ITEM_FIELDS) - {
    "invoicing_schedule_type",
    "invoicing_frequency",
    "special_notes",
}
MONEY_FIELDS = {"total_listed_value", "unit_price"}


def equal(field, left, right):
    if left is None or right is None:
        return left is right
    if field in MONEY_FIELDS or field == "quantity":
        try:
            tolerance = Decimal(0) if field in MONEY_FIELDS else Decimal("0.000001")
            return abs(Decimal(str(left)) - Decimal(str(right))) <= tolerance
        except InvalidOperation:
            return False
    return left == right


def hungarian(cost):
    """Minimum-cost assignment for a square matrix, O(n^3)."""
    n = len(cost)
    if not n:
        return []
    u, v, p, way = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


# Pair expected and extracted purchases one-to-one within each SKU.
# Repeated SKUs stay separate; dates and totals help pair the correct instances.
def match_items(expected, actual):
    groups_e, groups_a = defaultdict(list), defaultdict(list)
    for i, item in enumerate(expected):
        groups_e[item.get("sku_code")].append(i)
    for i, item in enumerate(actual):
        groups_a[item.get("sku_code")].append(i)
    pairs, missing, extra = [], [], []
    for code in sorted(groups_e.keys() | groups_a.keys(), key=str):
        ei, ai = groups_e[code], groups_a[code]
        size = max(len(ei), len(ai))
        cost = []
        for row in range(size):
            costs = []
            for col in range(size):
                if row >= len(ei):
                    value = 0
                elif col >= len(ai):
                    value = 10000
                else:
                    left, right = expected[ei[row]], actual[ai[col]]
                    value = sum(
                        (
                            4
                            if field
                            in (
                                "service_start_date",
                                "service_end_date",
                                "total_listed_value",
                            )
                            else 1
                        )
                        for field in CORE_FIELDS
                        if not equal(field, left.get(field), right.get(field))
                    )
                costs.append(value)
            cost.append(costs)
        used = set()
        assignment = hungarian(cost)
        for row in range(len(ei)):
            col = assignment[row]
            if col < len(ai):
                pairs.append((ei[row], ai[col]))
                used.add(ai[col])
            else:
                missing.append(ei[row])
        extra.extend(index for index in ai if index not in used)
    return pairs, missing, extra


def compare(expected, actual, document_id="unknown"):
    pairs, missing, extra = match_items(
        expected["contract_items"], actual.get("contract_items", [])
    )
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    differences = []

    def record(field, left, right, present):
        ok = present and equal(field, left, right)
        counts[field]["total"] += 1
        counts[field]["correct"] += int(ok)
        if not ok:
            differences.append({"field": field, "expected": left, "actual": right})

    for field in DATE_FIELDS:
        if field in expected:
            record(field, expected[field], actual.get(field), field in actual)
    paired = dict(pairs)
    for index, item in enumerate(expected["contract_items"]):
        extracted = actual["contract_items"][paired[index]] if index in paired else {}
        for field in ITEM_FIELDS:
            if field in item:
                record(field, item[field], extracted.get(field), field in extracted)
    core = [
        value
        for field, value in counts.items()
        if field in CORE_FIELDS or field in DATE_FIELDS
    ]
    return {
        "document_id": document_id,
        "true_positive_items": len(pairs),
        "false_negative_items": len(missing),
        "false_positive_items": len(extra),
        "field_counts": dict(counts),
        "core_correct": sum(c["correct"] for c in core),
        "core_total": sum(c["total"] for c in core),
        "strict_document_match": not differences and not extra and not missing,
        "differences": differences,
    }


def aggregate(reports, *, expected_documents=None, failed=None):
    fields = defaultdict(lambda: {"correct": 0, "total": 0})
    for report in reports:
        for field, counts in report["field_counts"].items():
            for key in counts:
                fields[field][key] += counts[key]
    correct, total = (
        sum(r["core_correct"] for r in reports),
        sum(r["core_total"] for r in reports),
    )
    all_correct, all_total = (
        sum(c["correct"] for c in fields.values()),
        sum(c["total"] for c in fields.values()),
    )
    expected_documents = expected_documents or len(reports)
    return {
        "documents_expected": expected_documents,
        "documents_evaluated": len(reports),
        "failed_documents": failed or [],
        "coverage": len(reports) / expected_documents if expected_documents else 0,
        "true_positive_items": sum(r["true_positive_items"] for r in reports),
        "false_positive_items": sum(r["false_positive_items"] for r in reports),
        "false_negative_items": sum(r["false_negative_items"] for r in reports),
        "core_field_correct": correct,
        "core_field_total": total,
        "core_field_accuracy": correct / total if total else 0,
        "all_labeled_field_correct": all_correct,
        "all_labeled_field_total": all_total,
        "all_labeled_field_accuracy": all_correct / all_total if all_total else 0,
        "strict_document_accuracy": sum(r["strict_document_match"] for r in reports)
        / len(reports)
        if reports
        else 0,
        "per_field": dict(fields),
    }


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(PROJECT_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--results", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument(
        "--model", default=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Extract all PDFs before scoring; makes paid API calls",
    )
    args = parser.parse_args()
    paths = sorted((args.dataset / "ground_truth").glob("*.json"))
    if not paths:
        parser.error("No expected answers found in dataset/ground_truth.")
    if args.live and not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("Set ANTHROPIC_API_KEY for live extraction.")
    catalog = load_catalog(args.dataset / "sku_catalog.json") if args.live else None

    def process(expected_path):
        output = args.results / expected_path.name
        try:
            if args.live:
                pdf = args.dataset / "pdfs" / f"{expected_path.stem}.pdf"
                extraction, validation = run_agent(pdf, catalog, model=args.model)
                actual = public_result(extraction.model_dump(mode="json"))
                save_result(output, actual)
                status = validation.status
            else:
                actual = json.loads(output.read_text())
                status = "saved"
            expected = json.loads(expected_path.read_text())
            report = compare(expected, actual, expected_path.stem)
            print(
                f"{expected_path.stem}: core {report['core_correct']}/{report['core_total']}; {status}",
                flush=True,
            )
            return report, None
        except Exception as exc:
            return None, {
                "document_id": expected_path.stem,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # Each worker writes a distinct result file; there is no shared diagnostic file.
    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(process, paths))
    reports = [report for report, _ in outcomes if report is not None]
    failures = [error for _, error in outcomes if error is not None]
    summary = aggregate(reports, expected_documents=len(paths), failed=failures)
    if args.live:
        summary["model"] = args.model
    save_result(args.results / "benchmark.json", summary)
    print(json.dumps(summary, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
