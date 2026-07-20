from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET

import mobile_paper_library as library
from paper_analysis import build_daily_markdown, process_top_papers, render_html
from report_contract import build_report, write_report


BEIJING = dt.timezone(dt.timedelta(hours=8))
ARXIV_ID_RE = re.compile(r"/abs/([^?#]+)")


def history_by_report_date() -> dict[str, list[dict]]:
    payload = json.loads(library.HISTORY_PATH.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for entry in payload.get("sent", []):
        try:
            sent_at = dt.datetime.fromisoformat(str(entry["sent_at"]).replace("Z", "+00:00"))
            report_date = sent_at.astimezone(BEIJING).date().isoformat()
        except Exception:
            continue
        grouped.setdefault(report_date, []).append(entry)
    return grouped


def fetch_arxiv_ids(entries: list[dict], source_errors: list[str]) -> list[library.Item]:
    ids = []
    for entry in entries:
        match = ARXIV_ID_RE.search(str(entry.get("url") or ""))
        if match:
            ids.append(match.group(1))
    if not ids:
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    items: list[library.Item] = []
    for start in range(0, len(ids), 20):
        query = urllib.parse.urlencode({"id_list": ",".join(ids[start : start + 20]), "max_results": 20})
        try:
            root = ET.fromstring(library.request_text(f"https://export.arxiv.org/api/query?{query}"))
        except Exception as exc:
            source_errors.append(f"historical arXiv batch: {type(exc).__name__}")
            continue
        for node in root.findall("atom:entry", ns):
            title = library.normalize(node.findtext("atom:title", default="", namespaces=ns))
            url = node.findtext("atom:id", default="", namespaces=ns)
            summary = library.normalize(node.findtext("atom:summary", default="", namespaces=ns))
            authors = ", ".join(
                library.normalize(author.findtext("atom:name", default="", namespaces=ns))
                for author in node.findall("atom:author", ns)
            )
            primary = node.find("arxiv:primary_category", ns)
            category = primary.attrib.get("term", "arXiv") if primary is not None else "arXiv"
            item = library.Item(
                title=title,
                url=url,
                pdf_url=library.arxiv_pdf_url(url),
                source=f"arXiv {category}",
                published=node.findtext("atom:published", default="", namespaces=ns),
                authors=authors,
                summary=summary,
                score=0,
            )
            item.score = library.relevance_score(item)
            if item.score > 0:
                items.append(item)
    return library.dedupe(items)


def backfill(report_date: str, skip_llm: bool) -> int:
    entries = history_by_report_date().get(report_date, [])
    if not entries:
        print(f"No sent_history entries for {report_date}")
        return 1
    errors: list[str] = []
    items = sorted(fetch_arxiv_ids(entries, errors), key=lambda item: item.score, reverse=True)[: library.MAX_ITEMS]
    configs = library.resolve_llm_configs()
    invoke_json = None if skip_llm else library.make_json_invoker(configs)
    if not skip_llm:
        items, translation_errors = library.enrich_items_with_llm(items, invoke_json)
        errors.extend(translation_errors)
    for item in items:
        item.title_en = item.title_en or item.title
        item.abstract_en = item.abstract_en or item.summary
        item.abstract_zh = item.abstract_zh or item.summary_zh
    items, pdf_errors = process_top_papers(
        items,
        library.RESEARCH_PROFILE,
        invoke_json,
    )
    errors.extend(pdf_errors)
    base_url = library.page_base_url()
    body = build_daily_markdown(items, library.REPORT_TITLE, report_date, f"{base_url}/{report_date}/")
    report = build_report(
        stream=library.STREAM,
        title=f"{library.REPORT_TITLE} - {report_date}",
        body=body,
        items=items,
        generation_status="partial" if errors else "complete",
        email_status="historical-not-sent",
        source_errors=errors,
        report_date=report_date,
        default_kind="paper",
    )
    report["reconstructed_from_sources"] = True
    write_report(report)
    daily_dir = library.DOCS_DIR / report_date
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / "index.html").write_text(
        render_html(items, library.REPORT_TITLE, report_date, base_url), encoding="utf-8"
    )
    print(f"Backfilled {report_date}: {len(items)} candidates")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild a historical daily report from sent_history + arXiv.")
    parser.add_argument("--date", action="append", required=True, help="YYYY-MM-DD; repeat for multiple dates")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    return max(backfill(value, args.skip_llm) for value in args.date)


if __name__ == "__main__":
    raise SystemExit(main())
