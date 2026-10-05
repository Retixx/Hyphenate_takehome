"""Extraction models. Unknown facts are null; Decimal handles monetary arithmetic."""

import re
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from tools import date_in_quote, normalize, number_in_quote

ScheduleType = Literal[
    "upfront", "recurring", "milestone-based", "usage-based", "hybrid"
]
Frequency = Literal["monthly", "quarterly", "yearly", "biannually", "one-time"]
Origin = Literal["stated", "calculated", "defaulted", "catalog"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FieldEvidence(Record):
    field_name: str
    page_number: int = Field(ge=1)
    quote: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    method: Literal["text", "visual"] = "text"


class Discount(Record):
    kind: Literal["percentage", "amount"]
    value: Decimal
    currency: str | None = None

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return float(value)


class InvoiceEvent(Record):
    invoice_date: date | None = None
    amount: Decimal | None = None
    currency: str | None = None
    milestone: str | None = None

    @field_serializer("amount", when_used="json")
    def serialize_amount(self, value):
        return float(value) if value is not None else None


class ContractItem(Record):
    sku_code: str
    # Python populates canonical identity from the catalog.
    sku_id: str | None = None
    sku_name: str | None = None
    currency: str | None = None
    total_listed_value: Decimal | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    unit_price_period: str | None = None
    discount: Discount | None = None
    service_start_date: date | None = None
    service_end_date: date | None = None
    invoicing_schedule_type: ScheduleType | None = None
    invoicing_frequency: Frequency | None = None
    invoicing_schedule: list[InvoiceEvent] = Field(default_factory=list)
    payment_terms: str | None = None
    payment_terms_details: str | None = None
    special_notes: str | None = None
    evidence: list[FieldEvidence] = Field(default_factory=list)
    value_origins: dict[str, Origin] = Field(default_factory=dict)

    @field_serializer("total_listed_value", "quantity", "unit_price", when_used="json")
    def serialize_decimal(self, value):
        # JSON numbers at the boundary; never calculate using these floats.
        return float(value) if value is not None else None


class OrderFormExtraction(Record):
    effective_date: date | None = None
    contract_signed_date_buying_company: date | None = None
    contract_signed_date_selling_company: date | None = None
    buying_company: str | None = None
    selling_company: str | None = None
    buyer_signatory: str | None = None
    seller_signatory: str | None = None
    document_total: Decimal | None = None
    currency: str | None = None
    contract_items: list[ContractItem]
    evidence: list[FieldEvidence] = Field(default_factory=list)
    unmatched_items: list[str] = Field(default_factory=list)

    @field_serializer("document_total", when_used="json")
    def serialize_total(self, value):
        return float(value) if value is not None else None


class ValidationIssue(Record):
    severity: Literal["error", "warning"]
    code: str
    path: str
    message: str


class ValidationReport(Record):
    status: Literal["validated", "needs_review", "invalid"]
    issues: list[ValidationIssue]


OUTPUT_ITEM_FIELDS = (
    "sku_id",
    "sku_code",
    "sku_name",
    "currency",
    "total_listed_value",
    "service_start_date",
    "service_end_date",
    "quantity",
    "unit_price",
    "invoicing_schedule_type",
    "invoicing_frequency",
    "payment_terms",
    "special_notes",
)


def public_result(payload):
    """Save only the fixed ground-truth fields, in the same key order."""
    return {
        **{
            field: payload.get(field)
            for field in (
                "effective_date",
                "contract_signed_date_buying_company",
                "contract_signed_date_selling_company",
            )
        },
        "contract_items": [
            {field: item.get(field) for field in OUTPUT_ITEM_FIELDS}
            for item in payload["contract_items"]
        ],
    }


class EvidenceReference(Record):
    """Compact model response; Python owns the quotation text."""

    fields: list[str] = Field(min_length=1, max_length=25)
    source_ids: list[str] = Field(default_factory=list, max_length=12)
    # Visual-only fallback for scanned pages, always marked unverified.
    page_number: int | None = Field(default=None, ge=1)
    quote: str | None = None


class ItemSubmission(ContractItem):
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ExtractionSubmission(OrderFormExtraction):
    contract_items: list[ItemSubmission]
    evidence: list[EvidenceReference] = Field(default_factory=list)


def resolve_evidence(references, document, inspected_pages=()):
    """Reject fabricated IDs; expand shared citations for internal validation."""
    output = []
    sources = document.source_index()
    for reference in references:
        data = EvidenceReference.model_validate(reference)
        if data.source_ids:
            if data.quote is not None or data.page_number is not None:
                raise ValueError("Source IDs and visual quote/page cannot be mixed.")
            if len(set(data.source_ids)) != len(data.source_ids):
                raise ValueError("Duplicate source IDs in one citation.")
            if any(identifier not in sources for identifier in data.source_ids):
                raise ValueError(
                    "Unknown source ID; cite a line returned by the document tools."
                )
            lines = [sources[identifier] for identifier in data.source_ids]
            pages = {line["page_number"] for line in lines}
            if len(pages) != 1:
                raise ValueError(
                    "Split cross-page citations into separate evidence entries."
                )
            page = lines[0]["page_number"]
            quote = "\n".join(line["text"] for line in lines)
        else:
            page, quote = data.page_number, data.quote
            if page is None or not quote or page > len(document.pages):
                raise ValueError(
                    "A visual citation needs an existing page and a quote."
                )
            if document.pages[page - 1].strip() and page not in inspected_pages:
                raise ValueError(
                    "Use source IDs for text pages, or inspect the page before quoting a visual OCR correction."
                )
        for field in dict.fromkeys(data.fields):
            output.append(
                FieldEvidence(
                    field_name=field,
                    page_number=page,
                    quote=quote,
                    source_ids=data.source_ids,
                    method="text" if data.source_ids else "visual",
                ).model_dump(mode="json")
            )
    return output


def expand_submission(payload, document, inspected_pages=()):
    """Separate the token-efficient wire format from the validated extraction model."""
    draft = ExtractionSubmission.model_validate(payload).model_dump()
    draft["evidence"] = resolve_evidence(draft["evidence"], document, inspected_pages)
    for item in draft["contract_items"]:
        item["evidence"] = resolve_evidence(item["evidence"], document, inspected_pages)
    return draft


def source_reference_matches(entry, document):
    """Check that evidence text and page match the indexed source lines."""
    sources = document.source_index()
    lines = [sources.get(identifier) for identifier in entry.source_ids]
    return (
        bool(lines)
        and not any(
            line is None or line["page_number"] != entry.page_number for line in lines
        )
        and entry.quote == "\n".join(line["text"] for line in lines if line)
    )


class CatalogEntry(Record):
    id: str | None = None
    sku_code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    parsing_instructions: str
    evidence: list[FieldEvidence]


class CatalogExtraction(Record):
    entries: list[CatalogEntry]


# Independent source and accounting checks.


CRITICAL_ITEM_FIELDS = (
    "sku_code",
    "currency",
    "total_listed_value",
    "quantity",
    "unit_price",
    "service_start_date",
    "service_end_date",
    "payment_terms",
)
BILLING_WORDS = {
    "invoicing_frequency": {
        "monthly": r"\b(monthly|each month|every month)\b",
        "quarterly": r"\b(quarterly|each quarter|every quarter)\b",
        "yearly": r"\b(annually|yearly|each year|every year)\b",
        "biannually": r"\b(biannually|semi.?annually|every six months|twice a year)\b",
        "one-time": r"\b(one.?time|single invoice|one invoice|(?:invoiced|billed) once)\b",
    },
    "invoicing_schedule_type": {
        "upfront": r"\b(up.?front|single invoice|one.?time|entire.*(?:due|invoice))\b",
        "recurring": r"\b(recurring|monthly|quarterly|annually|yearly|every month|every year)\b",
        "milestone-based": r"\b(milestone|upon completion)\b",
        "usage-based": r"\b(usage|consumption|actual hours)\b",
        "hybrid": r"\b(hybrid)\b",
    },
}


# Check catalog membership, source evidence, dates, currencies, and totals.
# Errors block acceptance; warnings identify facts that still need review.
def validate_extraction(result: OrderFormExtraction, document, catalog: dict):
    """Check source support, catalog membership, dates and compatible totals."""
    issues = []

    def add(code, path, message, *, error=False):
        issues.append(
            ValidationIssue(
                severity="error" if error else "warning",
                code=code,
                path=path,
                message=message,
            )
        )

    def check_record(record, path, required):
        evidence = {}
        for entry in record.evidence:
            location = f"{path}.{entry.field_name}"
            if entry.field_name not in type(
                record
            ).model_fields or entry.field_name in {"evidence", "value_origins"}:
                add(
                    "unknown_evidence_field",
                    location,
                    "Evidence references an unknown field.",
                    error=True,
                )
                continue
            if entry.page_number > len(document.pages):
                add(
                    "invalid_evidence_page",
                    location,
                    "Evidence page is outside the PDF.",
                    error=True,
                )
                continue
            page, quote = (
                normalize(document.pages[entry.page_number - 1]),
                normalize(entry.quote),
            )
            if entry.source_ids and not source_reference_matches(entry, document):
                add(
                    "invalid_source_reference",
                    location,
                    "Source text or page differs from the document index.",
                    error=True,
                )
                continue
            if not quote:
                add("empty_evidence", location, "Evidence quote is empty.", error=True)
                continue
            quote_exists = (
                all(
                    normalize(document.sources[key]["text"]) in page
                    for key in entry.source_ids
                )
                if entry.source_ids
                else quote in page
            )
            if not page or entry.method == "visual":
                add(
                    "visual_evidence_unverified",
                    location,
                    "Visual evidence needs manual verification.",
                )
            elif not quote_exists:
                add(
                    "quote_not_found",
                    location,
                    "Quote not found in page text; inspect layout or OCR.",
                )
                continue
            evidence.setdefault(entry.field_name, []).append(entry)
            if len(quote) < 8:
                add(
                    "weak_evidence",
                    location,
                    "Quote is too short to identify the pricing row.",
                )
            value = getattr(record, entry.field_name)
            stated = (
                getattr(record, "value_origins", {}).get(entry.field_name, "stated")
                == "stated"
            )
            if (
                isinstance(value, Decimal)
                and stated
                and not number_in_quote(value, entry.quote)
            ):
                add(
                    "value_not_in_quote",
                    location,
                    "Numeric value is absent from its supporting quote.",
                )
            if isinstance(value, date) and not date_in_quote(value, entry.quote):
                add(
                    "value_not_in_quote",
                    location,
                    "Date value is absent from its quote; confirm format and locale.",
                )
        for field in required:
            stated = (
                getattr(record, "value_origins", {}).get(field, "stated") == "stated"
            )
            if getattr(record, field) is not None and stated and field not in evidence:
                add(
                    "missing_evidence",
                    f"{path}.{field}",
                    "Stated value has no supporting evidence.",
                )
        return evidence

    check_record(
        result,
        "order_form",
        (
            "effective_date",
            "contract_signed_date_buying_company",
            "contract_signed_date_selling_company",
            "document_total",
        ),
    )
    explicit_payment_terms = re.search(
        r"payment\s+terms\s*(?:include|are|:)?\s*(Net\s+\d+|Due\s+(?:on|upon)\s+Receipt)",
        normalize(" ".join(document.pages)),
        re.IGNORECASE,
    )
    # Detect possible contract conditions when every item's notes were omitted.
    # These are review candidates; the model must still determine their scope.
    note_sources = [
        {"source_id": key, "text": source["text"]}
        for key, source in document.source_index().items()
        if re.search(
            r"automatically renew|renewal|(?:annual|yearly).*(?:uplift|increase)"
            r"|(?:usage|employee).*(?:cap|limit)|(?:includes?|provide).*(?:onboarding|training)"
            r"|(?:pricing|discount).*(?:approved|approval)|master subscription agreement",
            source["text"],
            re.IGNORECASE,
        )
    ][:8]
    if (
        result.contract_items
        and note_sources
        and all(
            item.special_notes is None or not item.special_notes.strip()
            for item in result.contract_items
        )
    ):
        add(
            "stated_notes_omitted",
            "contract_items.special_notes",
            "Notes were omitted despite possible contract conditions. Review their scope, "
            "summarize applicable conditions in special_notes, and cite supporting lines: "
            + str(note_sources),
        )
    # Explicit billing labels must be reviewed even if the model omitted citations.
    billing_sources = {field: [] for field in BILLING_WORDS}
    previous = None
    for key, source in document.source_index().items():
        context = source["text"]
        if previous and previous["page_number"] == source["page_number"]:
            context = previous["text"] + " " + context
        previous = source
        if not re.search(r"\b(?:invoic\w*|bill\w*)\b", context, re.IGNORECASE):
            continue
        for field, patterns in BILLING_WORDS.items():
            if any(
                re.search(pattern, source["text"], re.IGNORECASE)
                for pattern in patterns.values()
            ):
                billing_sources[field].append(
                    {"source_id": key, "text": source["text"]}
                )
    for index, item in enumerate(result.contract_items):
        path = f"contract_items[{index}]"
        if item.sku_code not in catalog:
            add(
                "unknown_sku",
                path + ".sku_code",
                "SKU is not in the supplied catalog.",
                error=True,
            )
        if (
            item.service_start_date
            and item.service_end_date
            and item.service_end_date < item.service_start_date
        ):
            add(
                "reversed_dates",
                path,
                "Service end precedes service start.",
                error=True,
            )
        if item.currency and not re.fullmatch(r"[A-Z]{3}", item.currency):
            add(
                "invalid_currency",
                path + ".currency",
                "Use a three-letter uppercase currency code.",
                error=True,
            )
        if item.currency is None:
            add(
                "missing_currency",
                path + ".currency",
                "Currency is unknown; do not combine different currencies.",
            )
        for field in item.value_origins:
            if field not in ContractItem.model_fields:
                add(
                    "unknown_origin_field",
                    path + "." + field,
                    "Origin refers to an unknown field.",
                    error=True,
                )
        evidence = check_record(item, path, CRITICAL_ITEM_FIELDS)
        if item.payment_terms is None and explicit_payment_terms:
            add(
                "stated_payment_terms_omitted",
                path + ".payment_terms",
                "The PDF states payment terms. Check whether they cover this item: "
                + explicit_payment_terms.group(0),
            )
        for field, patterns in BILLING_WORDS.items():
            value = getattr(item, field)
            if value is None:
                add(
                    "missing_billing",
                    path + "." + field,
                    "Billing is unspecified; another agreement may be needed.",
                )
                if billing_sources[field]:
                    add(
                        "stated_billing_omitted",
                        path + "." + field,
                        "Explicit billing wording was omitted. Check item scope and cite its source: "
                        + str(billing_sources[field][:4]),
                    )
                if field == "invoicing_frequency":
                    quotes = " ".join(
                        entry.quote
                        for name in ("invoicing_schedule_type", "invoicing_schedule")
                        for entry in evidence.get(name, [])
                    )
                    if any(
                        re.search(pattern, quotes, re.IGNORECASE)
                        for pattern in patterns.values()
                    ):
                        add(
                            "billing_requires_review",
                            path + "." + field,
                            "Cited billing terms name a frequency. Check its scope: "
                            + quotes,
                        )
            else:
                quotes = " ".join(entry.quote for entry in evidence.get(field, []))
                if not re.search(patterns[value], quotes, re.IGNORECASE):
                    add(
                        "billing_requires_review",
                        path + "." + field,
                        "Quoted wording does not directly establish this billing label.",
                    )
        if item.special_notes and not evidence.get("special_notes"):
            add(
                "missing_evidence",
                path + ".special_notes",
                "Contractual notes need evidence.",
            )
        if item.total_listed_value is None:
            add(
                "missing_amount", path + ".total_listed_value", "Line total is unknown."
            )
        if item.quantity is None:
            add(
                "quantity_requires_review",
                path + ".quantity",
                "Quantity is unknown or catalog-specific.",
            )
        if (
            item.quantity is not None
            and item.unit_price is not None
            and item.total_listed_value is not None
        ):
            if (
                abs(item.quantity * item.unit_price - item.total_listed_value)
                > Decimal("0.01")
                and not item.unit_price_period
                and not item.discount
            ):
                add(
                    "pricing_basis_unspecified",
                    path + ".unit_price_period",
                    "Quantity × unit price differs from total; price period or discount basis is unstated.",
                )
        if (
            item.discount
            and item.discount.kind == "percentage"
            and not 0 <= item.discount.value <= 100
        ):
            add(
                "invalid_discount",
                path + ".discount",
                "Percentage discount must be between 0 and 100.",
                error=True,
            )
        if item.invoicing_schedule:
            amounts = [invoice.amount for invoice in item.invoicing_schedule]
            currencies = {
                invoice.currency or item.currency for invoice in item.invoicing_schedule
            }
            if (
                item.total_listed_value is not None
                and all(amount is not None for amount in amounts)
                and item.currency is not None
                and currencies == {item.currency}
            ):
                if abs(sum(amounts, Decimal(0)) - item.total_listed_value) > Decimal(
                    "0.000001"
                ):
                    add(
                        "invoice_total_mismatch",
                        path + ".invoicing_schedule",
                        "Invoice amounts differ from the line total; check partial schedules, credits, tax or usage.",
                    )
            else:
                add(
                    "invoice_total_not_comparable",
                    path + ".invoicing_schedule",
                    "Unknown invoice amounts or currencies prevent reconciliation.",
                )

    if result.document_total is not None and result.contract_items:
        amounts = [item.total_listed_value for item in result.contract_items]
        currencies = {item.currency for item in result.contract_items}
        if (
            all(value is not None for value in amounts)
            and result.currency is not None
            and currencies == {result.currency}
            and not result.unmatched_items
        ):
            if abs(sum(amounts, Decimal(0)) - result.document_total) > Decimal(
                "0.000001"
            ):
                add(
                    "total_mismatch",
                    "document_total",
                    "Item totals differ; check taxes, discounts, credits and scope.",
                )
        else:
            add(
                "total_not_comparable",
                "document_total",
                "Currencies, amounts or unmatched products prevent reconciliation.",
            )
    if result.unmatched_items:
        add(
            "unmatched_products",
            "unmatched_items",
            "Some sold products could not be mapped to the catalog.",
        )
    if not result.contract_items:
        add(
            "no_items",
            "contract_items",
            "No catalog items identified; check for missed purchases.",
        )
    status = (
        "invalid"
        if any(issue.severity == "error" for issue in issues)
        else "needs_review"
        if issues
        else "validated"
    )
    return ValidationReport(status=status, issues=issues)
