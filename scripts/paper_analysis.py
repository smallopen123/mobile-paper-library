from __future__ import annotations

import html
import json
import math
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24 compatibility
    import fitz  # type: ignore[no-redef]
import requests


MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 40
TOP_FULLTEXT = 10
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
FIGURE_KEYWORDS = ("framework", "architecture", "overview", "pipeline", "method", "system")
FIGURE_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*\d+[a-z]?)\s*[:.\-]?\s*(.*)", re.I | re.S)
CORE_FIGURE_TOKEN = "<!-- CORE_FIGURE:{item_id} -->"


@dataclass
class PdfInspection:
    page_count: int
    parsed_pages: int
    page_texts: list[str]
    core_figure: dict[str, Any]

    @property
    def marked_text(self) -> str:
        return "\n\n".join(
            f"[[PAGE {page_number}]]\n{text}"
            for page_number, text in enumerate(self.page_texts, 1)
        )


class PdfLimitError(RuntimeError):
    pass


class PdfFormatError(RuntimeError):
    pass


class PdfOcrRequired(RuntimeError):
    pass


def _safe_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return value[:80] or "paper"


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            pass
    return min(4.0 * (2**attempt), 45.0)


def download_pdf(url: str, destination: Path, attempts: int = 3) -> int:
    """Download one PDF with a hard byte limit and no persistent cache."""
    if not url.lower().startswith(("http://", "https://")):
        raise PdfFormatError("missing HTTP PDF URL")
    headers = {
        "User-Agent": "Codex-Obsidian-research-digest/2.0",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with requests.get(url, headers=headers, timeout=(15, 90), stream=True) as response:
                response.raise_for_status()
                declared = int(response.headers.get("Content-Length") or 0)
                if declared > MAX_PDF_BYTES:
                    raise PdfLimitError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
                total = 0
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=128 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_PDF_BYTES:
                            raise PdfLimitError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
                        handle.write(chunk)
            if total < 5 or destination.read_bytes()[:5] != b"%PDF-":
                raise PdfFormatError("downloaded content is not a PDF")
            return total
        except PdfLimitError:
            destination.unlink(missing_ok=True)
            raise
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            destination.unlink(missing_ok=True)
            last_error = exc
            status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response else 0
            if status and status not in RETRYABLE_HTTP_CODES:
                break
            if attempt < attempts - 1:
                retry_after = exc.response.headers.get("Retry-After") if isinstance(exc, requests.HTTPError) and exc.response else None
                time.sleep(_retry_delay(attempt, retry_after))
        except OSError as exc:
            destination.unlink(missing_ok=True)
            last_error = exc
            break
    raise RuntimeError(f"PDF download failed: {type(last_error).__name__ if last_error else 'unknown'}")


def _figure_candidate(page: fitz.Page, page_number: int, block: tuple[Any, ...], all_text: str) -> dict[str, Any] | None:
    x0, y0, x1, y1, text = block[:5]
    caption = _safe_text(text)
    match = FIGURE_RE.match(caption)
    if not match:
        return None
    label = _safe_text(match.group(1)).replace(".", "")
    caption_tail = _safe_text(match.group(2))
    lowered = caption_tail.lower()
    matched = [keyword for keyword in FIGURE_KEYWORDS if keyword in lowered]
    if not matched:
        return None

    page_rect = page.rect
    caption_rect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
    image_rects: list[fitz.Rect] = []
    for info in page.get_image_info(xrefs=True):
        try:
            rect = fitz.Rect(info["bbox"])
        except Exception:
            continue
        if rect.y1 <= caption_rect.y0 + 8 and rect.get_area() > 2_500:
            image_rects.append(rect)
    if image_rects:
        crop = max(image_rects, key=lambda rect: rect.get_area())
        crop = fitz.Rect(
            max(page_rect.x0, crop.x0 - 8),
            max(page_rect.y0, crop.y0 - 8),
            min(page_rect.x1, crop.x1 + 8),
            min(page_rect.y1, crop.y1 + 8),
        )
        visual_area = crop.get_area()
        crop_basis = "embedded-image-nearest-caption"
    else:
        drawing_rects: list[fitz.Rect] = []
        for drawing in page.get_drawings():
            try:
                rect = fitz.Rect(drawing["rect"])
            except Exception:
                continue
            if rect.y1 <= caption_rect.y0 + 8 and rect.get_area() > 2_500:
                drawing_rects.append(rect)
        if drawing_rects:
            crop = max(drawing_rects, key=lambda rect: rect.get_area())
            crop = fitz.Rect(
                max(page_rect.x0, crop.x0 - 10),
                max(page_rect.y0, crop.y0 - 10),
                min(page_rect.x1, crop.x1 + 10),
                min(caption_rect.y0 - 4, crop.y1 + 10),
            )
            visual_area = crop.get_area()
            crop_basis = "vector-drawing-region"
        else:
            # Last resort for vector figures whose drawing primitives are unavailable.
            # Keep the crop above the caption and inside page margins; never substitute a results chart.
            upper = max(page_rect.y0 + 28, caption_rect.y0 - page_rect.height * 0.48)
            crop = fitz.Rect(page_rect.x0 + 28, upper, page_rect.x1 - 28, max(upper + 80, caption_rect.y0 - 6))
            visual_area = crop.get_area()
            crop_basis = "vector-region-above-caption"

    ref_pattern = re.compile(rf"\b{re.escape(label)}\b", re.I)
    reference_count = max(0, len(ref_pattern.findall(all_text)) - 1)
    keyword_score = sum(4 if keyword in ("framework", "architecture", "pipeline") else 2 for keyword in matched)
    score = keyword_score + min(reference_count, 6) + min(visual_area / max(page_rect.get_area(), 1) * 4, 4)
    return {
        "status": "found",
        "figure_label": label,
        "page": page_number,
        "caption_en": caption,
        "caption_zh": "",
        "selection_reason_zh": (
            f"Caption 命中 {', '.join(matched)}；正文引用约 {reference_count} 次；"
            f"按标题匹配、引用次数和图形面积综合排序。"
        ),
        "caption_bbox": [round(v, 2) for v in caption_rect],
        "crop_bbox": [round(v, 2) for v in crop],
        "page_size": [round(page_rect.width, 2), round(page_rect.height, 2)],
        "crop_basis": crop_basis,
        "score": round(score, 3),
    }


def inspect_pdf(path: Path) -> PdfInspection:
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise PdfFormatError(f"cannot open PDF: {type(exc).__name__}") from exc
    try:
        if not document.is_pdf or document.page_count < 1:
            raise PdfFormatError("empty or invalid PDF")
        parsed_pages = min(document.page_count, MAX_PDF_PAGES)
        page_texts = [_safe_text(document.load_page(i).get_text("text")) for i in range(parsed_pages)]
        character_count = sum(len(text) for text in page_texts)
        if character_count < max(300, parsed_pages * 80):
            raise PdfOcrRequired("too little extractable text")
        all_text = "\n".join(page_texts)
        candidates: list[dict[str, Any]] = []
        for index in range(parsed_pages):
            page = document.load_page(index)
            for block in page.get_text("blocks"):
                candidate = _figure_candidate(page, index + 1, block, all_text)
                if candidate:
                    candidates.append(candidate)
        core_figure = max(candidates, key=lambda item: item["score"]) if candidates else {
            "status": "not_found",
            "figure_label": "",
            "page": None,
            "caption_en": "",
            "caption_zh": "",
            "selection_reason_zh": "未自动识别到包含 Framework、Architecture、Overview、Pipeline、Method 或 System 的核心框图 Caption。",
            "caption_bbox": [],
            "crop_bbox": [],
            "page_size": [],
            "crop_basis": "none",
            "score": 0,
        }
        return PdfInspection(document.page_count, parsed_pages, page_texts, core_figure)
    finally:
        document.close()


def _prompt_excerpt(inspection: PdfInspection, max_chars: int = 62_000) -> str:
    chunks: list[str] = []
    remaining = max_chars
    per_page = max(1_200, min(3_500, max_chars // max(len(inspection.page_texts), 1)))
    for page_number, page_text in enumerate(inspection.page_texts, 1):
        if remaining <= 0:
            break
        text = page_text[: min(len(page_text), per_page, remaining)]
        chunk = f"[[PAGE {page_number}]]\n{text}"
        chunks.append(chunk)
        remaining -= len(chunk)
    return "\n\n".join(chunks)


def fulltext_prompt(item: Any, inspection: PdfInspection, research_profile: str) -> str:
    figure = inspection.core_figure
    return f"""
你是严谨的博士科研文献分析助手。只能依据下面带页码标记的论文文本回答，不得补写文本中不存在的指标、数值、数据集、Baseline、结论或页码。

研究范围：
{research_profile}

论文标题：{getattr(item, 'title', '')}
论文摘要：{getattr(item, 'summary', '')}
PDF 总页数：{inspection.page_count}；本次解析前 {inspection.parsed_pages} 页。
候选核心图：{json.dumps(figure, ensure_ascii=False)}

返回一个 JSON 对象，不要 Markdown。字段必须是：
- title_zh: 中文标题
- abstract_zh: 忠实的中文摘要翻译
- research_question_zh: 研究问题
- hypothesis_zh: 核心假设；没有明确假设则说明未明确陈述
- method_chain_zh: 方法与理论链路
- frontier_zh: 为什么前沿
- relevance_zh: 与研究范围的联系
- reproducibility_zh: 可复现条件和最小复现实验
- research_idea_zh: 可证伪的博士研究构想
- datasets, baselines, metrics, key_results_zh, limitations_zh: 数组；每项必须含 name、page、evidence，其中 evidence 是论文英文原文中的短证据片段
- source_pages: 实际使用的页码整数数组
- summary_flow: 3-7 个顺序节点，用于“输入/场景→核心模块→机制→输出→指标”中文总结图；保留模型和指标英文名
- diagram_source_pages: 总结图依据的页码整数数组
- core_figure_caption_zh: 候选核心图英文 Caption 的中文翻译；没有候选图则为空字符串

页面文本：
{_prompt_excerpt(inspection)}
""".strip()


def _normalized_evidence(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _valid_page(value: Any, page_count: int) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if 1 <= page <= page_count else None


def _validated_evidence_list(payload: Any, inspection: PdfInspection) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    validated: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        page = _valid_page(entry.get("page"), inspection.parsed_pages)
        evidence = _safe_text(entry.get("evidence"))
        name = _safe_text(entry.get("name"))
        if not page or len(evidence) < 12 or not name:
            continue
        needle = _normalized_evidence(evidence)
        haystack = _normalized_evidence(inspection.page_texts[page - 1])
        if len(needle) < 10 or needle not in haystack:
            continue
        validated.append({"name": name, "page": page, "evidence": evidence})
    return validated


def _mermaid_label(value: str) -> str:
    value = _safe_text(value).replace('"', "'").replace("`", "'")
    value = re.sub(r"[{}\[\]<>]", "", value)
    return value[:100]


def build_mermaid(flow: Any) -> str:
    if not isinstance(flow, list):
        return ""
    labels = [_mermaid_label(item) for item in flow if _mermaid_label(item)]
    labels = labels[:7]
    if len(labels) < 3:
        return ""
    lines = ["flowchart LR"]
    for index, label in enumerate(labels):
        lines.append(f'  N{index}["{label}"]')
    for index in range(len(labels) - 1):
        lines.append(f"  N{index} --> N{index + 1}")
    return "\n".join(lines)


def _safe_mermaid(value: Any) -> str:
    """Keep generated Mermaid line breaks while rejecting fenced or malformed content."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in text.split("\n") if line.strip()]
    if len(lines) < 4 or lines[0].strip() not in {"flowchart LR", "flowchart TD"}:
        return ""
    if "```" in text:
        return ""
    return "\n".join(lines)


def apply_fulltext_payload(item: Any, inspection: PdfInspection, payload: dict[str, Any]) -> None:
    string_fields = (
        "title_zh",
        "abstract_zh",
        "research_question_zh",
        "hypothesis_zh",
        "method_chain_zh",
        "frontier_zh",
        "relevance_zh",
        "reproducibility_zh",
        "research_idea_zh",
    )
    for field_name in string_fields:
        value = _safe_text(payload.get(field_name))
        if value:
            setattr(item, field_name, value)
    for field_name in ("datasets", "baselines", "metrics", "key_results_zh", "limitations_zh"):
        setattr(item, field_name, _validated_evidence_list(payload.get(field_name), inspection))

    pages = {
        page
        for page in (_valid_page(value, inspection.parsed_pages) for value in payload.get("source_pages", []))
        if page
    }
    evidence_pages = {
        entry["page"]
        for field_name in ("datasets", "baselines", "metrics", "key_results_zh", "limitations_zh")
        for entry in getattr(item, field_name, [])
    }
    item.source_pages = sorted(pages | evidence_pages)
    item.summary_diagram_mermaid = build_mermaid(payload.get("summary_flow"))
    item.diagram_source_pages = sorted(
        page
        for page in (_valid_page(value, inspection.parsed_pages) for value in payload.get("diagram_source_pages", []))
        if page
    )
    if item.core_figure.get("status") == "found":
        item.core_figure["caption_zh"] = _safe_text(payload.get("core_figure_caption_zh"))


def process_top_papers(
    items: list[Any],
    research_profile: str,
    invoke_json: Callable[[str], dict[str, Any]] | None,
    top_n: int = TOP_FULLTEXT,
    downloader: Callable[[str, Path], int] = download_pdf,
    inspector: Callable[[Path], PdfInspection] = inspect_pdf,
) -> tuple[list[Any], list[str]]:
    """Mutate ranked items with verified PDF analysis, replacing failed candidates."""
    errors: list[str] = []
    rank = 0
    with tempfile.TemporaryDirectory(prefix="daily-paper-pdf-") as temp_dir:
        temp_root = Path(temp_dir)
        for item in items:
            item.title_en = _safe_text(getattr(item, "title_en", "") or getattr(item, "title", ""))
            item.abstract_en = _safe_text(getattr(item, "abstract_en", "") or getattr(item, "summary", ""))
            if rank >= top_n:
                break
            pdf_url = _safe_text(getattr(item, "pdf_url", ""))
            if not pdf_url:
                item.fulltext_status = "not_available"
                continue
            path = temp_root / f"{_slug(getattr(item, 'key', str(rank + 1)))}.pdf"
            try:
                downloader(pdf_url, path)
                inspection = inspector(path)
            except PdfLimitError as exc:
                item.fulltext_status = "too_large"
                errors.append(f"{getattr(item, 'title', 'paper')}: {exc}")
                continue
            except PdfOcrRequired:
                item.fulltext_status = "needs_ocr"
                errors.append(f"{getattr(item, 'title', 'paper')}: needs_ocr")
                continue
            except PdfFormatError as exc:
                item.fulltext_status = "invalid_pdf"
                errors.append(f"{getattr(item, 'title', 'paper')}: {exc}")
                continue
            except Exception as exc:
                item.fulltext_status = "download_or_parse_failed"
                errors.append(f"{getattr(item, 'title', 'paper')}: {type(exc).__name__}")
                continue
            finally:
                # The original PDF is temporary even when parsing or model analysis fails.
                pass

            rank += 1
            item.analysis_rank = rank
            item.fulltext_status = "verified" if inspection.page_count <= MAX_PDF_PAGES else "verified_first_40_pages"
            item.evidence_scope = "fulltext" if inspection.page_count <= MAX_PDF_PAGES else "fulltext_first_40_pages"
            item.pdf_page_count = inspection.page_count
            item.parsed_page_count = inspection.parsed_pages
            item.core_figure = inspection.core_figure
            item.figure_status = str(inspection.core_figure.get("status") or "not_found")
            if invoke_json:
                try:
                    payload = invoke_json(fulltext_prompt(item, inspection, research_profile))
                    if not isinstance(payload, dict):
                        raise ValueError("LLM response is not a JSON object")
                    apply_fulltext_payload(item, inspection, payload)
                except Exception as exc:
                    errors.append(f"{getattr(item, 'title', 'paper')}: fulltext synthesis {type(exc).__name__}")
            path.unlink(missing_ok=True)
    if rank < top_n:
        errors.append(f"Only {rank} of {top_n} papers had a downloadable, extractable PDF")
    return items, errors


def _topic(item: Any) -> str:
    text = f"{getattr(item, 'title', '')} {getattr(item, 'summary', '')}".lower()
    topics = (
        ("Foundation/LLM", ("foundation model", "large language model", "llm", "vlm", "vla")),
        ("World Model", ("world model", "spatiotemporal foundation")),
        ("Trajectory", ("trajectory", "motion forecasting", "intent prediction")),
        ("Safety/Risk", ("safety", "risk", "collision", "conflict")),
        ("Multi-Agent", ("multi-agent", "multi uav", "multi-uav", "swarm")),
        ("Edge", ("edge", "deployment", "efficient", "quantization")),
    )
    for label, terms in topics:
        if any(term in text for term in terms):
            return label
    return "Other"


def _bar_svg(counts: Counter[str], title: str, width: int = 760) -> str:
    entries = counts.most_common(7) or [("No data", 0)]
    row_height = 34
    height = 54 + row_height * len(entries)
    max_value = max((value for _, value in entries), default=1) or 1
    rows: list[str] = []
    for index, (label, value) in enumerate(entries):
        y = 42 + index * row_height
        bar_width = int((width - 260) * value / max_value)
        rows.append(f'<text x="16" y="{y + 16}" fill="#cbd5e1" font-size="14">{html.escape(label)}</text>')
        rows.append(f'<rect x="180" y="{y}" width="{bar_width}" height="20" rx="5" fill="#2dd4bf"/>')
        rows.append(f'<text x="{190 + bar_width}" y="{y + 16}" fill="#e2e8f0" font-size="13">{value}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(title)}" style="max-width:100%;background:#0f172a;border-radius:12px">'
        f'<text x="16" y="26" fill="#f8fafc" font-size="17" font-weight="700">{html.escape(title)}</text>'
        + "".join(rows)
        + "</svg>"
    )


def _evidence_line(entries: Any) -> str:
    if not entries:
        return "- 未从可核验原文片段中确认。"
    return "\n".join(
        f"- {entry['name']}（PDF 第 {entry['page']} 页；证据：\"{entry['evidence']}\"）"
        for entry in entries
    )


def _callout(title: str, content: str, folded: bool = True) -> str:
    marker = "-" if folded else "+"
    lines = [f"> [!info]{marker} {title}"]
    lines.extend(f"> {line}" if line else ">" for line in content.splitlines())
    return "\n".join(lines)


def build_daily_markdown(items: list[Any], title: str, report_date: str, web_url: str = "") -> str:
    top = sorted((item for item in items if getattr(item, "analysis_rank", None)), key=lambda item: item.analysis_rank)
    other = [item for item in items if not getattr(item, "analysis_rank", None)]
    topic_counts = Counter(_topic(item) for item in items)
    evidence_counts = Counter("全文核验" if item in top else "摘要级" for item in items)
    lines = [
        "## 今日核心发现",
        "",
        f"- 候选论文 {len(items)} 篇；成功完成 PDF 全文提取与页码校验 {len(top)} 篇。",
        f"- 主题分布以 {', '.join(label for label, _ in topic_counts.most_common(3)) or '暂无'} 为主。",
        "- 所有数值、数据集、Baseline 与局限仅在存在可回查英文证据片段时展示。",
        "",
        "## Top 10 阅读优先级",
        "",
        "| 排名 | 中文标题 | English Title | 全文状态 | 核心图 |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for item in top:
        lines.append(
            f"| {item.analysis_rank} | {_safe_text(getattr(item, 'title_zh', '')) or '待翻译'} | "
            f"{_safe_text(getattr(item, 'title_en', ''))} | {item.fulltext_status} | {item.figure_status} |"
        )
    lines.extend([
        "",
        "## 主题与方法分布",
        "",
        _bar_svg(topic_counts, "Topic and Method Distribution / 主题与方法分布"),
        "",
        "## 证据深度统计",
        "",
        _bar_svg(evidence_counts, "Evidence Depth / 证据深度"),
        "",
        "## 当日整体技术路线图",
        "",
        "```mermaid",
        "flowchart LR",
        '  A["Low-altitude scenario / 低空场景"] --> B["Perception and context / 感知与上下文"]',
        '  B --> C["Foundation model or predictor / 基础模型或预测器"]',
        '  C --> D["Risk, trajectory or action / 风险、航迹或动作"]',
        '  D --> E["Safety and performance metrics / 安全与性能指标"]',
        "```",
        "",
        "## 研究空白与实验建议",
        "",
        "1. 统一比较跨场景泛化：固定数据划分，比较 in-domain、cross-city 与极端天气性能。",
        "2. 补齐不确定性与安全闭环：同时报告预测误差、校准误差、碰撞/冲突风险和推理延迟。",
        "3. 检验通用模型的真实增益：以轻量专用模型为 Baseline，做参数量、数据规模、工具调用与消融实验。",
        "",
    ])
    if web_url:
        lines.extend([f"网页版本：{web_url}", ""])

    for item in top:
        item_id = _safe_text(getattr(item, "key", "paper"))
        figure = getattr(item, "core_figure", {}) or {}
        pages = "、".join(str(page) for page in getattr(item, "source_pages", [])) or "未形成可引用分析页码"
        lines.extend([
            f"## Top {item.analysis_rank}. {_safe_text(getattr(item, 'title_zh', '')) or _safe_text(getattr(item, 'title', ''))}",
            "",
            f"**English Title:** {_safe_text(getattr(item, 'title_en', ''))}",
            "",
            f"- Authors: {_safe_text(getattr(item, 'authors', '')) or 'Unknown'}",
            f"- Source: {_safe_text(getattr(item, 'source', ''))}",
            f"- Published: {_safe_text(getattr(item, 'published', ''))}",
            f"- [Original Page]({_safe_text(getattr(item, 'url', ''))}) | [Available PDF]({_safe_text(getattr(item, 'pdf_url', ''))})",
            f"- Evidence scope: `{getattr(item, 'evidence_scope', 'abstract')}`；分析引用页：{pages}",
            "",
            "### 中文摘要",
            "",
            _safe_text(getattr(item, "abstract_zh", "")) or "尚未获得可靠中文翻译。",
            "",
            _callout("English Abstract", _safe_text(getattr(item, "abstract_en", "")) or "Unavailable", True),
            "",
            "### 论文原始核心框图",
            "",
            CORE_FIGURE_TOKEN.format(item_id=item_id),
        ])
        if figure.get("status") == "found":
            lines.extend([
                f"- Figure: {figure.get('figure_label')}；PDF 第 {figure.get('page')} 页",
                f"- English Caption: {figure.get('caption_en')}",
                f"- 中文 Caption: {figure.get('caption_zh') or '待翻译'}",
                f"- 选择理由：{figure.get('selection_reason_zh')}",
            ])
        else:
            lines.append("未自动识别到论文核心框图；没有使用结果曲线或无关图片代替。")
        diagram = _safe_mermaid(getattr(item, "summary_diagram_mermaid", ""))
        diagram_pages = "、".join(str(page) for page in getattr(item, "diagram_source_pages", [])) or "未记录"
        lines.extend([
            "",
            "### AI 中文总结框图",
            "",
            f"> AI 总结框图，不是论文原图；依据论文第 {diagram_pages} 页生成。",
            "",
        ])
        if diagram:
            lines.extend(["```mermaid", diagram, "```"])
        else:
            lines.extend([
                "| 环节 | 当前可核验内容 |",
                "| --- | --- |",
                "| 输入/场景 | 未生成可验证总结图 |",
                "| 核心模块 | 请根据 PDF 页码证据人工复核 |",
                "| 输出/指标 | 不根据摘要推断 |",
            ])
        lines.extend([
            "",
            "### 核心内容",
            "",
            f"- **研究问题：** {_safe_text(getattr(item, 'research_question_zh', '')) or '论文未在当前解析内容中明确陈述。'}",
            f"- **核心假设：** {_safe_text(getattr(item, 'hypothesis_zh', '')) or '未确认。'}",
            f"- **方法与理论链路：** {_safe_text(getattr(item, 'method_chain_zh', '')) or '待基于全文进一步核验。'}",
            f"- **为什么前沿：** {_safe_text(getattr(item, 'frontier_zh', '')) or '待核验。'}",
            "",
            "### 数据集、Baselines 与 Metrics",
            "",
            "**Datasets**",
            _evidence_line(getattr(item, "datasets", [])),
            "",
            "**Baselines**",
            _evidence_line(getattr(item, "baselines", [])),
            "",
            "**Metrics**",
            _evidence_line(getattr(item, "metrics", [])),
            "",
            "### 主要结果与页码",
            "",
            _evidence_line(getattr(item, "key_results_zh", [])),
            "",
            "### 局限与证据边界",
            "",
            _evidence_line(getattr(item, "limitations_zh", [])),
            "",
            f"- **与研究方向的联系：** {_safe_text(getattr(item, 'relevance_zh', '')) or '待复核。'}",
            f"- **可复现方案：** {_safe_text(getattr(item, 'reproducibility_zh', '')) or '待复核。'}",
            f"- **博士研究构想：** {_safe_text(getattr(item, 'research_idea_zh', '')) or '待复核。'}",
            "",
        ])

    if other:
        lines.extend(["## 其余候选（摘要级，默认折叠）", ""])
        for index, item in enumerate(other, len(top) + 1):
            content = "\n".join([
                f"- English Title: {_safe_text(getattr(item, 'title_en', '')) or _safe_text(getattr(item, 'title', ''))}",
                f"- 中文摘要：{_safe_text(getattr(item, 'abstract_zh', '')) or '待翻译'}",
                f"- English Abstract: {_safe_text(getattr(item, 'abstract_en', '')) or _safe_text(getattr(item, 'summary', ''))}",
                f"- 核心贡献/相关性：{_safe_text(getattr(item, 'relevance_zh', '')) or '仅依据摘要，待全文核验。'}",
                f"- 证据等级：secondary / abstract-only",
                f"- [Original Page]({_safe_text(getattr(item, 'url', ''))}) | [PDF]({_safe_text(getattr(item, 'pdf_url', ''))})",
            ])
            lines.extend([_callout(f"{index}. {_safe_text(getattr(item, 'title_zh', '')) or _safe_text(getattr(item, 'title', ''))}", content, True), ""])
    return "\n".join(lines).strip()


def build_email_summary(items: list[Any], title: str, report_date: str, web_url: str = "") -> str:
    top = sorted((item for item in items if getattr(item, "analysis_rank", None)), key=lambda item: item.analysis_rank)
    other = [item for item in items if not getattr(item, "analysis_rank", None)]
    lines = [
        f"{title} - {report_date}",
        f"今日候选 {len(items)} 篇，PDF 全文核验 {len(top)} 篇。完整双语报告与双框图请在 Obsidian 查看。",
        "",
        "Top 10 核心内容",
    ]
    for item in top:
        result = getattr(item, "key_results_zh", [])
        result_text = result[0]["name"] + f"（第 {result[0]['page']} 页）" if result else "未确认可引用数值结果"
        lines.extend([
            "",
            f"{item.analysis_rank}. {_safe_text(getattr(item, 'title_zh', '')) or _safe_text(getattr(item, 'title', ''))}",
            f"   English: {_safe_text(getattr(item, 'title_en', ''))}",
            f"   问题：{_safe_text(getattr(item, 'research_question_zh', '')) or '待复核'}",
            f"   方法：{_safe_text(getattr(item, 'method_chain_zh', '')) or '待复核'}",
            f"   结果：{result_text}",
            f"   链接：{_safe_text(getattr(item, 'url', ''))}",
        ])
    if other:
        lines.extend(["", "其余论文简表"])
        for item in other:
            lines.append(f"- {_safe_text(getattr(item, 'title_zh', '')) or _safe_text(getattr(item, 'title', ''))} | {_safe_text(getattr(item, 'url', ''))}")
    if web_url:
        lines.extend(["", f"网页完整版本：{web_url}"])
    return "\n".join(lines).strip()


def render_html(items: list[Any], title: str, report_date: str, base_url: str) -> str:
    cards: list[str] = []
    for item in sorted(items, key=lambda value: getattr(value, "analysis_rank", None) or 999):
        rank = getattr(item, "analysis_rank", None)
        badge = f"Top {rank} · Full-text verified" if rank else "Abstract-level"
        mermaid = _safe_mermaid(getattr(item, "summary_diagram_mermaid", ""))
        diagram = f'<pre class="mermaid">{html.escape(mermaid)}</pre>' if mermaid else '<p class="muted">未生成可验证总结框图。</p>'
        results = "".join(f"<li>{html.escape(e['name'])}（PDF p.{e['page']}）</li>" for e in getattr(item, "key_results_zh", [])) or "<li>未确认可引用结果。</li>"
        cards.append(f"""
        <article class="paper">
          <span class="badge">{html.escape(badge)}</span>
          <h2>{html.escape(_safe_text(getattr(item, 'title_zh', '')) or _safe_text(getattr(item, 'title', '')))}</h2>
          <p class="english">{html.escape(_safe_text(getattr(item, 'title_en', '')))}</p>
          <p class="meta">{html.escape(_safe_text(getattr(item, 'authors', '')))} · {html.escape(_safe_text(getattr(item, 'source', '')))}</p>
          <p><a href="{html.escape(_safe_text(getattr(item, 'url', '')))}">Original Page</a> · <a href="{html.escape(_safe_text(getattr(item, 'pdf_url', '')))}">PDF</a></p>
          <h3>中文摘要</h3><p>{html.escape(_safe_text(getattr(item, 'abstract_zh', '')) or '待翻译')}</p>
          <details><summary>English Abstract</summary><p>{html.escape(_safe_text(getattr(item, 'abstract_en', '')))}</p></details>
          <h3>论文原始核心框图</h3><p class="muted">原始图不在公开网页再分发；Figure 元数据保留，并在个人 Obsidian 同步时本地裁剪。</p>
          <h3>AI 中文总结框图</h3>{diagram}
          <h3>方法与理论链路</h3><p>{html.escape(_safe_text(getattr(item, 'method_chain_zh', '')) or '待核验')}</p>
          <h3>主要结果</h3><ul>{results}</ul>
          <h3>博士研究构想</h3><p>{html.escape(_safe_text(getattr(item, 'research_idea_zh', '')) or '待复核')}</p>
        </article>
        """)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} - {report_date}</title>
<style>
:root{{--bg:#f1f5f9;--paper:#fff;--ink:#0f172a;--muted:#64748b;--brand:#0f766e}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.7}}
header{{background:#083c3a;color:white;padding:28px 18px}}header div,main{{max-width:980px;margin:auto}}main{{padding:18px 14px 60px}}
.paper{{background:var(--paper);border:1px solid #cbd5e1;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 8px 24px #0f172a12}}
.badge{{display:inline-block;background:#ccfbf1;color:#115e59;border-radius:999px;padding:3px 10px;font-weight:700}}
h1{{margin:0}}h2{{line-height:1.35}}h3{{color:#115e59}}.english,.meta,.muted{{color:var(--muted)}}a{{color:var(--brand)}}
.mermaid{{background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;padding:12px;overflow:auto}}
</style></head><body><header><div><h1>{html.escape(title)}</h1><p>{report_date} · Top 10 full-text · Bilingual</p><a style="color:#99f6e4" href="{html.escape(base_url)}/">查看历史归档</a></div></header>
<main>{''.join(cards)}</main><script type="module">import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';mermaid.initialize({{startOnLoad:true,theme:'neutral'}});</script></body></html>"""


def validate_schema_v2_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("title_en", "title_zh", "abstract_en", "abstract_zh", "fulltext_status", "figure_status"):
        if key not in item:
            errors.append(f"missing {key}")
    rank = item.get("analysis_rank")
    if rank is not None and (not isinstance(rank, int) or not 1 <= rank <= TOP_FULLTEXT):
        errors.append("invalid analysis_rank")
    figure = item.get("core_figure") or {}
    if figure.get("status") == "found":
        if not figure.get("figure_label") or not isinstance(figure.get("page"), int):
            errors.append("invalid core_figure metadata")
        bbox = figure.get("crop_bbox") or []
        if len(bbox) != 4 or any(not isinstance(value, (int, float)) for value in bbox):
            errors.append("invalid core_figure crop_bbox")
    return errors
