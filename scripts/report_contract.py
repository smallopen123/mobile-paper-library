from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from paper_analysis import validate_schema_v2_item


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
BEIJING = dt.timezone(dt.timedelta(hours=8))


def beijing_now() -> dt.datetime:
    return dt.datetime.now(BEIJING)


def generator_commit() -> str:
    value = os.getenv("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNPINNED"


def source_run_url() -> str:
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return ""


def _item_value(item: Any, name: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def serialize_item(item: Any, default_kind: str = "other") -> dict[str, Any]:
    item_id = _item_value(item, "key") or _item_value(item, "id")
    title_en = str(_item_value(item, "title_en", "") or _item_value(item, "title", ""))
    abstract_en = str(_item_value(item, "abstract_en", "") or _item_value(item, "summary", ""))
    serialized = {
        "id": str(item_id),
        "type": str(_item_value(item, "kind", _item_value(item, "type", default_kind))),
        "title": title_en,
        "title_en": title_en,
        "title_zh": str(_item_value(item, "title_zh", "")),
        "url": str(_item_value(item, "url", "")),
        "pdf_url": str(_item_value(item, "pdf_url", "")),
        "published_at": str(_item_value(item, "published", _item_value(item, "date", ""))),
        "source": str(_item_value(item, "source", "")),
        "authors": str(_item_value(item, "authors", "")),
        "summary": abstract_en,
        "abstract_en": abstract_en,
        "abstract_zh": str(_item_value(item, "abstract_zh", _item_value(item, "summary_zh", ""))),
        "evidence_level": "secondary",
        "analysis_rank": _item_value(item, "analysis_rank", None),
        "evidence_scope": str(_item_value(item, "evidence_scope", "abstract")),
        "source_pages": list(_item_value(item, "source_pages", []) or []),
        "datasets": list(_item_value(item, "datasets", []) or []),
        "baselines": list(_item_value(item, "baselines", []) or []),
        "metrics": list(_item_value(item, "metrics", []) or []),
        "key_results_zh": list(_item_value(item, "key_results_zh", []) or []),
        "limitations_zh": list(_item_value(item, "limitations_zh", []) or []),
        "research_question_zh": str(_item_value(item, "research_question_zh", "")),
        "hypothesis_zh": str(_item_value(item, "hypothesis_zh", "")),
        "method_chain_zh": str(_item_value(item, "method_chain_zh", "")),
        "frontier_zh": str(_item_value(item, "frontier_zh", "")),
        "relevance_zh": str(_item_value(item, "relevance_zh", "")),
        "reproducibility_zh": str(_item_value(item, "reproducibility_zh", "")),
        "research_idea_zh": str(_item_value(item, "research_idea_zh", "")),
        "core_figure": dict(_item_value(item, "core_figure", {}) or {}),
        "summary_diagram_mermaid": str(_item_value(item, "summary_diagram_mermaid", "")),
        "diagram_source_pages": list(_item_value(item, "diagram_source_pages", []) or []),
        "fulltext_status": str(_item_value(item, "fulltext_status", "not_attempted")),
        "figure_status": str(_item_value(item, "figure_status", "not_attempted")),
        "pdf_page_count": int(_item_value(item, "pdf_page_count", 0) or 0),
        "parsed_page_count": int(_item_value(item, "parsed_page_count", 0) or 0),
    }
    errors = validate_schema_v2_item(serialized)
    if errors:
        raise ValueError(f"schema v2 item validation failed for {item_id}: {', '.join(errors)}")
    return serialized


def build_report(
    *,
    stream: str,
    title: str,
    body: str,
    items: Iterable[Any],
    generation_status: str,
    email_status: str,
    source_errors: Iterable[str] = (),
    report_date: str | None = None,
    default_kind: str = "other",
) -> dict[str, Any]:
    now = beijing_now()
    serialized = [serialize_item(item, default_kind=default_kind) for item in items]
    return {
        "schema_version": 2,
        "stream": stream,
        "report_date": report_date or now.date().isoformat(),
        "generated_at": now.isoformat(timespec="seconds"),
        "generator_commit": generator_commit(),
        "source_run_url": source_run_url(),
        "generation_status": generation_status,
        "email_status": email_status,
        "source_errors": list(source_errors),
        "item_count": len(serialized),
        "items": serialized,
        "title": title,
        "body": body.strip(),
    }


def _frontmatter(report: dict[str, Any]) -> str:
    scalar_keys = [
        "schema_version",
        "stream",
        "report_date",
        "generated_at",
        "generator_commit",
        "source_run_url",
        "generation_status",
        "email_status",
        "item_count",
    ]
    lines = ["---"]
    for key in scalar_keys:
        lines.append(f"{key}: {json.dumps(report.get(key), ensure_ascii=False)}")
    for key in ("reconstructed_from_sources", "original_email_body_available"):
        if key in report:
            lines.append(f"{key}: {json.dumps(report.get(key), ensure_ascii=False)}")
    lines.append(f"source_errors: {json.dumps(report.get('source_errors', []), ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def write_report(report: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = str(report["report_date"])
    json_path = OUTPUTS_DIR / f"{stem}.json"
    md_path = OUTPUTS_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    warnings = ""
    if report.get("source_errors"):
        warnings = "\n## 数据源状态\n\n" + "\n".join(
            f"- {error}" for error in report["source_errors"]
        ) + "\n"
    markdown = (
        f"{_frontmatter(report)}\n\n# {report['title']}\n\n"
        "> 此文件由自动化生成。它是待复核的外部情报，不是已经验证的科研结论。\n"
        f"{warnings}\n{report['body'].strip()}\n"
    )
    md_path.write_text(markdown, encoding="utf-8")
    return md_path, json_path
