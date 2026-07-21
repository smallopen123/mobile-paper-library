from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
import html
import json
import os
import random
import re
import smtplib
import ssl
import sys
import time
import textwrap
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

import requests
from openai import OpenAI

from paper_analysis import (
    build_daily_markdown,
    build_email_summary,
    load_recent_review_records,
    process_top_papers,
    render_html,
)
import report_contract
from report_contract import build_report, write_report


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
HISTORY_PATH = ROOT / "data" / "sent_history.json"
REPORTS_DIR = ROOT / "outputs"
MAX_ITEMS = 20
RECENT_DAYS = 7
PRIMARY_DAYS = 3
STREAM = "low-altitude-paper-library"
REPORT_TITLE = "低空经济通用大模型前沿论文库"
USER_AGENT = "mobile-paper-library/2.0 (+https://github.com/smallopen123/mobile-paper-library)"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

RESEARCH_PROFILE = """
博士研究重点：低空场景中的 LLM、VLM、VLA、World Model、时空基础模型、Agent、
安全评估与边缘部署。论文必须同时具有低空/无人机/城市空中交通语境和通用大模型、
基础模型或大规模预训练方法语境；机器人、临床、牙科和通用商业智能体不单独纳入。
"""

LOW_ALTITUDE_TERMS = (
    "low altitude", "urban air mobility", "advanced air mobility", "uav", "drone",
    "unmanned aerial", "air traffic", "aerial vehicle", "flight trajectory",
)
FOUNDATION_MODEL_TERMS = (
    "foundation model", "large language model", "llm", "vision-language", "vlm",
    "vision-language-action", "vla", "world model", "pretrained model", "pre-trained model",
    "multimodal model", "spatiotemporal foundation", "spatio-temporal foundation", "agentic",
)

KEYWORDS = [
    "low altitude economy",
    "urban air mobility",
    "advanced air mobility",
    "unmanned aerial vehicle",
    "uav",
    "drone",
    "trajectory prediction",
    "motion forecasting",
    "spatio-temporal",
    "risk assessment",
    "airspace safety",
    "large language model",
    "foundation model",
    "vision-language model",
    "vision-language-action",
    "multimodal model",
    "agentic",
    "world model",
    "pretrained model",
    "edge deployment",
]


@dataclass
class Item:
    title: str
    url: str
    pdf_url: str
    source: str
    published: str
    authors: str
    summary: str
    score: float
    title_zh: str = ""
    summary_zh: str = ""
    reading_hint_zh: str = ""
    relevance_zh: str = ""
    practice_zh: str = ""
    title_en: str = ""
    abstract_en: str = ""
    abstract_zh: str = ""
    analysis_rank: int | None = None
    evidence_scope: str = "abstract"
    source_pages: list[int] = field(default_factory=list)
    datasets: list[dict] = field(default_factory=list)
    baselines: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    key_results_zh: list[dict] = field(default_factory=list)
    limitations_zh: list[dict] = field(default_factory=list)
    research_question_zh: str = ""
    hypothesis_zh: str = ""
    method_chain_zh: str = ""
    frontier_zh: str = ""
    reproducibility_zh: str = ""
    research_idea_zh: str = ""
    core_figure: dict = field(default_factory=dict)
    summary_diagram_mermaid: str = ""
    diagram_source_pages: list[int] = field(default_factory=list)
    fulltext_status: str = "not_attempted"
    figure_status: str = "not_attempted"
    pdf_page_count: int = 0
    parsed_page_count: int = 0
    selection_mode: str = "new"
    source_report_date: str = ""

    @property
    def key(self) -> str:
        raw = f"{self.title}|{self.url}".lower()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def item_from_review_record(record: dict) -> Item:
    item = Item(
        title=str(record.get("title_en") or record.get("title") or ""),
        url=str(record.get("url") or ""),
        pdf_url=str(record.get("pdf_url") or ""),
        source=str(record.get("source") or ""),
        published=str(record.get("published_at") or ""),
        authors=str(record.get("authors") or ""),
        summary=str(record.get("abstract_en") or record.get("summary") or ""),
        score=0.0,
    )
    for name in (
        "title_zh", "title_en", "abstract_en", "abstract_zh", "analysis_rank", "evidence_scope",
        "source_pages", "datasets", "baselines", "metrics", "key_results_zh", "limitations_zh",
        "research_question_zh", "hypothesis_zh", "method_chain_zh", "frontier_zh", "relevance_zh",
        "reproducibility_zh", "research_idea_zh", "core_figure", "summary_diagram_mermaid",
        "diagram_source_pages", "fulltext_status", "figure_status", "pdf_page_count", "parsed_page_count",
        "selection_mode", "source_report_date",
    ):
        if name in record:
            setattr(item, name, record[name])
    item.summary_zh = item.abstract_zh
    return item


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: str | None = None


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            pass
    return min(5.0 * (2**attempt) + random.uniform(0.0, 2.0), 60.0)


def request_text(url: str, attempts: int = 4) -> str:
    last_error: Exception | None = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/atom+xml,text/xml,*/*"}
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=(15, 60))
            response.raise_for_status()
            return response.text
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status not in RETRYABLE_HTTP_CODES or attempt == attempts - 1:
                raise
            retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
            time.sleep(_retry_delay(attempt, retry_after))
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(_retry_delay(attempt))
    raise last_error if last_error else RuntimeError("request_text failed")


def normalize(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_date(value: str) -> dt.datetime:
    if not value:
        return dt.datetime.now(dt.timezone.utc)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
        except Exception:
            return dt.datetime.now(dt.timezone.utc)


def arxiv_pdf_url(entry_url: str) -> str:
    if "/abs/" in entry_url:
        return entry_url.replace("/abs/", "/pdf/")
    return entry_url


def matched_keywords(item: Item) -> list[str]:
    haystack = f"{item.title} {item.summary} {item.source}".lower()
    return [keyword for keyword in KEYWORDS if keyword in haystack]


def relevance_score(item: Item) -> float:
    haystack = f"{item.title} {item.summary} {item.source}".lower()
    low_altitude_hits = sum(term in haystack for term in LOW_ALTITUDE_TERMS)
    foundation_hits = sum(term in haystack for term in FOUNDATION_MODEL_TERMS)
    if not low_altitude_hits or not foundation_hits:
        return -100.0
    score = len(matched_keywords(item)) * 2.0
    score += low_altitude_hits * 4.0 + foundation_hits * 5.0
    if "safety" in haystack or "risk" in haystack:
        score += 2.0
    if "edge" in haystack or "deployment" in haystack:
        score += 2.0
    age_days = max((dt.datetime.now(dt.timezone.utc) - parse_date(item.published)).days, 0)
    score += max(0, 4 - age_days * 0.5)
    return score


def fetch_arxiv(source_errors: list[str] | None = None) -> list[Item]:
    queries = [
        '((ti:UAV OR abs:UAV OR ti:drone OR abs:drone) AND (abs:"foundation model" OR abs:"large language model" OR abs:LLM))',
        '((abs:"urban air mobility" OR abs:"advanced air mobility" OR abs:"air traffic") AND (abs:"foundation model" OR abs:"world model" OR abs:agentic))',
        '((ti:UAV OR abs:UAV OR ti:drone OR abs:drone) AND (abs:"vision-language" OR abs:VLM OR abs:"vision-language-action" OR abs:VLA))',
        '((ti:UAV OR abs:UAV OR ti:drone OR abs:drone) AND (abs:"multimodal model" OR abs:"pretrained model" OR abs:"pre-trained model"))',
        '((abs:"flight trajectory" OR abs:"aerial vehicle") AND (abs:"spatiotemporal foundation" OR abs:"spatio-temporal foundation" OR abs:"world model"))',
    ]
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    items: list[Item] = []
    for query_index, query in enumerate(queries, 1):
        params = urllib.parse.urlencode(
            {
                "search_query": f"({query}) AND (cat:cs.RO OR cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:eess.SY OR cat:stat.ML)",
                "start": 0,
                "max_results": 45,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        try:
            root = ET.fromstring(request_text(f"https://export.arxiv.org/api/query?{params}"))
        except Exception as exc:
            if source_errors is not None:
                source_errors.append(f"arXiv query {query_index}: {type(exc).__name__}")
            continue
        for entry in root.findall("atom:entry", ns):
            title = normalize(entry.findtext("atom:title", default="", namespaces=ns))
            url = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            summary = normalize(entry.findtext("atom:summary", default="", namespaces=ns))
            authors = ", ".join(
                normalize(author.findtext("atom:name", default="", namespaces=ns))
                for author in entry.findall("atom:author", ns)
            )
            category_node = entry.find("arxiv:primary_category", ns)
            category = category_node.attrib.get("term", "arXiv") if category_node is not None else "arXiv"
            item = Item(
                title=title,
                url=url,
                pdf_url=arxiv_pdf_url(url),
                source=f"arXiv {category}",
                published=published,
                authors=authors,
                summary=summary,
                score=0.0,
            )
            item.score = relevance_score(item)
            items.append(item)
        if query_index < len(queries):
            time.sleep(3.0)
    return items


def dedupe(items: Iterable[Item]) -> list[Item]:
    seen: set[str] = set()
    result: list[Item] = []
    for item in items:
        if not item.title or not item.url or item.key in seen:
            continue
        seen.add(item.key)
        result.append(item)
    return result


def load_history() -> set[str]:
    if not HISTORY_PATH.exists():
        return set()
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return {entry["key"] for entry in data.get("sent", []) if "key" in entry}
    except Exception:
        return set()


def save_history(items: list[Item]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if HISTORY_PATH.exists():
        try:
            existing = json.loads(HISTORY_PATH.read_text(encoding="utf-8")).get("sent", [])
        except Exception:
            existing = []
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    current = [{"key": item.key, "title": item.title, "url": item.url, "sent_at": now} for item in items]
    merged = {entry.get("key", ""): entry for entry in existing + current if entry.get("key")}
    keep = sorted(merged.values(), key=lambda entry: entry.get("sent_at", ""), reverse=True)[:800]
    HISTORY_PATH.write_text(json.dumps({"sent": keep}, ensure_ascii=False, indent=2), encoding="utf-8")


def select_items(items: list[Item], history: set[str]) -> list[Item]:
    now = dt.datetime.now(dt.timezone.utc)
    recent_cutoff = now - dt.timedelta(days=RECENT_DAYS)
    primary_cutoff = now - dt.timedelta(days=PRIMARY_DAYS)
    relevant = [item for item in items if item.score > 0]
    filtered = [item for item in relevant if item.key not in history and parse_date(item.published) >= recent_cutoff]
    primary = [item for item in filtered if parse_date(item.published) >= primary_cutoff]
    candidates = primary if len(primary) >= MAX_ITEMS else filtered
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:MAX_ITEMS]


def rule_based_notes(item: Item) -> dict[str, str]:
    keywords = matched_keywords(item)
    keyword_text = ", ".join(keywords[:8]) if keywords else "未命中特定关键词，但因发布时间和类别被纳入候选。"
    reading_hint = (
        "中文翻译在模型重试后仍未生成，本条已按 partial 状态保留英文原文，等待下一次自动补全；"
        "这不是免费模式提示，也不代表英文摘要已经完成中文核验。"
    )
    relevance = (
        f"规则相关性：命中关键词 {keyword_text}。可优先检查论文的问题定义、数据来源、模型输入输出、"
        "评价指标，以及是否能迁移到低空安全评估、航迹预测或多智能体协同场景。"
    )
    practice = (
        "实践建议：先阅读摘要和实验设置，记录任务、数据集、模型结构、损失函数和评价指标；"
        "再判断是否能替换为无人机轨迹、低空空域风险或多智能体交互数据，并设计一个最小复现实验。"
    )
    return {
        "reading_hint": reading_hint,
        "relevance": relevance,
        "practice": practice,
    }


def resolve_llm_configs() -> list[LLMConfig]:
    configs: list[LLMConfig] = []
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        configs.append(
            LLMConfig(
                provider="DeepSeek",
                api_key=deepseek_api_key,
                model=os.getenv("DEEPSEEK_MODEL") or "deepseek-chat",
                base_url=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            )
        )
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        configs.append(
            LLMConfig(
                provider="OpenAI",
                api_key=openai_api_key,
                model=os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        )
    return configs


def resolve_llm_config() -> LLMConfig | None:
    """Backward-compatible primary-provider accessor."""
    configs = resolve_llm_configs()
    return configs[0] if configs else None


def _parse_json_object(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I | re.S)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", cleaned):
        try:
            payload, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
    raise ValueError("LLM response did not contain a valid JSON object")


def make_json_invoker(config: LLMConfig | list[LLMConfig] | None):
    configs = config if isinstance(config, list) else ([config] if config else [])
    if not configs:
        return None
    clients: list[tuple[LLMConfig, OpenAI]] = []
    for candidate in configs:
        client_kwargs = {"api_key": candidate.api_key}
        if candidate.base_url:
            client_kwargs["base_url"] = candidate.base_url
        clients.append((candidate, OpenAI(**client_kwargs)))

    def invoke(prompt: str) -> dict:
        failure_types: list[str] = []
        for candidate, client in clients:
            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=candidate.model,
                        messages=[
                            {"role": "system", "content": "你是严谨的科研全文分析与中英翻译助手，只输出 JSON。"},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                    )
                    content = (response.choices[0].message.content or "").strip()
                    return _parse_json_object(content)
                except Exception as exc:
                    failure_types.append(f"{candidate.provider}:{type(exc).__name__}")
                    if attempt == 0:
                        time.sleep(1.0 + random.uniform(0.0, 0.5))
        raise RuntimeError("All LLM providers failed: " + ", ".join(failure_types))

    return invoke


def _contains_chinese(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value or ""))


def _translation_prompt(payload: list[dict]) -> str:
    return (
        "你是科研论文中文整理助手。请基于给定论文列表返回严格 JSON 对象，顶层字段 items 为数组。"
        "每项字段必须包括：idx, title_zh, abstract_zh, reading_hint_zh, relevance_zh, practice_zh。"
        "abstract_zh 必须忠实翻译英文摘要；reading_hint_zh 用1到2句提示如何阅读；"
        "relevance_zh 用2到3句说明与低空场景通用大模型、基础模型、安全评估或边缘部署的相关性；"
        "practice_zh 用2到3句给出可落地实验或复现建议。不得添加原摘要中不存在的指标或结论。\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _translation_payload(items: list[Item], start: int) -> list[dict]:
    return [
        {
            "idx": start + offset,
            "title": item.title,
            "authors": item.authors,
            "source": item.source,
            "published": item.published,
            "abstract": item.summary[:2200],
        }
        for offset, item in enumerate(items, 1)
    ]


def _apply_translation_entries(items: list[Item], result: dict) -> set[int]:
    completed: set[int] = set()
    entries = result.get("items", []) if isinstance(result, dict) else []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry["idx"])
            item = items[idx - 1]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        title_zh = normalize(str(entry.get("title_zh") or ""))
        abstract_zh = normalize(str(entry.get("abstract_zh") or entry.get("summary_zh") or ""))
        if not (_contains_chinese(title_zh) and _contains_chinese(abstract_zh)):
            continue
        item.title_zh = title_zh
        item.summary_zh = abstract_zh
        item.abstract_zh = abstract_zh
        item.reading_hint_zh = normalize(str(entry.get("reading_hint_zh") or ""))
        item.relevance_zh = normalize(str(entry.get("relevance_zh") or ""))
        item.practice_zh = normalize(str(entry.get("practice_zh") or ""))
        completed.add(idx)
    return completed


def enrich_items_with_llm(items: list[Item], invoke_json=None) -> tuple[list[Item], list[str]]:
    for item in items:
        item.title_en = item.title_en or item.title
        item.abstract_en = item.abstract_en or item.summary
    if invoke_json is None:
        invoke_json = make_json_invoker(resolve_llm_configs())
    if not invoke_json:
        return items, ["Chinese translation: no configured LLM provider"]

    completed: set[int] = set()
    batch_size = 5
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        try:
            result = invoke_json(_translation_prompt(_translation_payload(batch, start)))
            completed.update(_apply_translation_entries(items, result))
        except Exception:
            pass
        missing = [start + offset for offset in range(1, len(batch) + 1) if start + offset not in completed]
        for idx in missing:
            try:
                result = invoke_json(_translation_prompt(_translation_payload([items[idx - 1]], idx - 1)))
                completed.update(_apply_translation_entries(items, result))
            except Exception:
                continue

    missing = [idx for idx in range(1, len(items) + 1) if idx not in completed]
    errors = [f"Chinese translation unavailable after retry: item {idx} ({items[idx - 1].title})" for idx in missing]
    return items, errors


def page_base_url() -> str:
    explicit = os.getenv("PAGES_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    repo = os.getenv("GITHUB_REPOSITORY", "smallopen123/mobile-paper-library")
    owner, name = repo.split("/", 1)
    return f"https://{owner}.github.io/{name}"


def render_daily_page(items: list[Item], today: str, base_url: str) -> str:
    cards = []
    for idx, item in enumerate(items, 1):
        notes = rule_based_notes(item)
        cards.append(
            f"""
            <article class="paper" id="paper-{idx}">
              <div class="paper-num">{idx:02d}</div>
              <h2>{html.escape(item.title)}</h2>
              <p class="meta">{html.escape(item.authors or "未知作者")} · {html.escape(item.source)} · {html.escape(item.published[:10])}</p>
              <div class="links">
                <a href="{html.escape(item.url)}" target="_blank" rel="noopener">原文页面</a>
                <a href="{html.escape(item.pdf_url)}" target="_blank" rel="noopener">PDF链接</a>
              </div>
              <section><h3>English Abstract</h3><p>{html.escape(item.summary)}</p></section>
              <section><h3>中文阅读提示</h3><p>{html.escape(notes["reading_hint"])}</p></section>
              <section><h3>规则相关性说明</h3><p>{html.escape(notes["relevance"])}</p></section>
              <section><h3>实践阅读思路</h3><p>{html.escape(notes["practice"])}</p></section>
            </article>
            """
        )
    nav = "\n".join(f'<a href="#paper-{idx}">{idx:02d}</a>' for idx in range(1, len(items) + 1))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>低空经济前沿论文库 - {today}</title>
  <style>
    :root{{color-scheme:light;--bg:#f4f6f8;--ink:#172033;--muted:#667085;--line:#d8dee9;--brand:#0f766e;--paper:#fff}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.72}}
    header{{background:#0b3b39;color:#fff;padding:26px 18px 22px}}
    header .wrap,main{{max-width:920px;margin:0 auto}}
    h1{{font-size:27px;line-height:1.22;margin:0 0 8px}}
    .sub{{margin:0;color:#d6f2ee}}
    main{{padding:18px 14px 64px}}
    .toc{{position:sticky;top:0;z-index:2;display:flex;gap:8px;overflow:auto;padding:10px 0;background:var(--bg)}}
    .toc a{{flex:0 0 auto;text-decoration:none;color:#0f766e;background:#fff;border:1px solid var(--line);border-radius:999px;padding:6px 10px;font-weight:700}}
    .paper{{position:relative;background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 8px 24px rgba(16,24,40,.05)}}
    .paper-num{{position:absolute;right:16px;top:14px;color:#94a3b8;font-weight:800}}
    h2{{font-size:20px;line-height:1.35;margin:0 38px 8px 0}}
    h3{{font-size:16px;margin:18px 0 6px;color:#0b3b39}}
    p{{margin:0 0 8px}}
    .meta{{color:var(--muted);font-size:14px}}
    .links{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 4px}}
    .links a{{text-decoration:none;color:#fff;background:var(--brand);border-radius:8px;padding:8px 11px;font-weight:700}}
    .notice{{background:#ecfdf5;border:1px solid #99f6e4;border-radius:12px;padding:14px;margin:12px 0;color:#064e3b}}
    .back{{display:inline-block;margin:14px 0 0;color:#d6f2ee}}
    @media (max-width:520px){{h1{{font-size:23px}}h2{{font-size:18px}}.paper{{padding:16px 14px}}}}
  </style>
</head>
<body>
  <header><div class="wrap">
    <h1>低空经济前沿论文库</h1>
    <p class="sub">{today} · {len(items)} 篇论文 · 免费模式 · 手机阅读版</p>
    <a class="back" href="{base_url}/">查看历史归档</a>
  </div></header>
  <main>
    <div class="notice">当前为无 API 免费模式：页面提供英文摘要、PDF 链接、规则相关性和阅读建议。中文翻译可使用安卓浏览器自带网页翻译完成。</div>
    <nav class="toc" aria-label="论文目录">{nav}</nav>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def render_index(today: str, base_url: str) -> str:
    entries = []
    for path in sorted(DOCS_DIR.iterdir(), reverse=True):
        if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name):
            label = path.name
            entries.append(f'<li><a href="{base_url}/{label}/">{label} 低空经济通用大模型前沿论文库</a></li>')
    items = "\n".join(entries) or "<li>暂无归档</li>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>低空经济通用大模型前沿论文库</title>
  <style>
    body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#f4f6f8;color:#172033;line-height:1.7}}
    main{{max-width:860px;margin:0 auto;padding:30px 18px 64px}}
    h1{{font-size:28px;margin:0 0 8px}}
    .muted{{color:#667085}}
    ul{{list-style:none;padding:0;margin:22px 0}}
    li{{margin:10px 0}}
    a{{display:block;background:#fff;border:1px solid #d8dee9;border-radius:10px;padding:14px 16px;color:#0f766e;text-decoration:none;font-weight:800}}
  </style>
</head>
<body>
  <main>
    <h1>低空经济前沿论文库</h1>
    <p class="muted">最后更新：{today}。每天北京时间 09:00 自动生成。当前为无 API 免费模式。</p>
    <ul>{items}</ul>
  </main>
</body>
</html>
"""


def render_daily_page_deepseek(items: list[Item], today: str, base_url: str) -> str:
    cards = []
    for idx, item in enumerate(items, 1):
        notes = rule_based_notes(item)
        title_zh = item.title_zh or item.title
        summary_zh = item.summary_zh or "当前未生成中文摘要翻译。"
        reading_hint = item.reading_hint_zh or notes["reading_hint"]
        relevance = item.relevance_zh or notes["relevance"]
        practice = item.practice_zh or notes["practice"]
        cards.append(
            f"""
            <article class="paper" id="paper-{idx}">
              <div class="paper-num">{idx:02d}</div>
              <h2>{html.escape(title_zh)}</h2>
              <p>{html.escape(item.title)}</p>
              <p class="meta">{html.escape(item.authors or "Unknown")} | {html.escape(item.source)} | {html.escape(item.published[:10])}</p>
              <div class="links">
                <a href="{html.escape(item.url)}" target="_blank" rel="noopener">Original Page</a>
                <a href="{html.escape(item.pdf_url)}" target="_blank" rel="noopener">PDF</a>
              </div>
              <section><h3>中文摘要翻译</h3><p>{html.escape(summary_zh)}</p></section>
              <section><h3>English Abstract</h3><p>{html.escape(item.summary)}</p></section>
              <section><h3>中文阅读提示</h3><p>{html.escape(reading_hint)}</p></section>
              <section><h3>相关性说明</h3><p>{html.escape(relevance)}</p></section>
              <section><h3>实践思路</h3><p>{html.escape(practice)}</p></section>
            </article>
            """
        )
    nav = "\n".join(f'<a href="#paper-{idx}">{idx:02d}</a>' for idx in range(1, len(items) + 1))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>低空经济前沿论文库 - {today}</title>
  <style>
    :root{{color-scheme:light;--bg:#f4f6f8;--ink:#172033;--muted:#667085;--line:#d8dee9;--brand:#0f766e;--paper:#fff}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.72}}
    header{{background:#0b3b39;color:#fff;padding:26px 18px 22px}}
    header .wrap,main{{max-width:920px;margin:0 auto}}
    h1{{font-size:27px;line-height:1.22;margin:0 0 8px}}
    .sub{{margin:0;color:#d6f2ee}}
    main{{padding:18px 14px 64px}}
    .toc{{position:sticky;top:0;z-index:2;display:flex;gap:8px;overflow:auto;padding:10px 0;background:var(--bg)}}
    .toc a{{flex:0 0 auto;text-decoration:none;color:#0f766e;background:#fff;border:1px solid var(--line);border-radius:999px;padding:6px 10px;font-weight:700}}
    .paper{{position:relative;background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 8px 24px rgba(16,24,40,.05)}}
    .paper-num{{position:absolute;right:16px;top:14px;color:#94a3b8;font-weight:800}}
    h2{{font-size:20px;line-height:1.35;margin:0 38px 8px 0}}
    h3{{font-size:16px;margin:18px 0 6px;color:#0b3b39}}
    p{{margin:0 0 8px}}
    .meta{{color:var(--muted);font-size:14px}}
    .links{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 4px}}
    .links a{{text-decoration:none;color:#fff;background:var(--brand);border-radius:8px;padding:8px 11px;font-weight:700}}
    .notice{{background:#ecfdf5;border:1px solid #99f6e4;border-radius:12px;padding:14px;margin:12px 0;color:#064e3b}}
    .back{{display:inline-block;margin:14px 0 0;color:#d6f2ee}}
    @media (max-width:520px){{h1{{font-size:23px}}h2{{font-size:18px}}.paper{{padding:16px 14px}}}}
  </style>
</head>
<body>
  <header><div class="wrap">
    <h1>低空经济前沿论文库</h1>
    <p class="sub">{today} | {len(items)} 篇论文 | DeepSeek 中文版 | 手机阅读友好</p>
    <a class="back" href="{base_url}/">查看历史归档</a>
  </div></header>
  <main>
    <div class="notice">当前页面优先展示 DeepSeek 生成的中文标题、中文摘要和中文阅读提示；若模型不可用，则自动回退到规则版英文摘要页面。</div>
    <nav class="toc" aria-label="论文目录">{nav}</nav>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def render_index_deepseek(today: str, base_url: str) -> str:
    entries = []
    for path in sorted(DOCS_DIR.iterdir(), reverse=True):
        if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name):
            label = path.name
            entries.append(f'<li><a href="{base_url}/{label}/">{label} 前沿论文库</a></li>')
    items = "\n".join(entries) or "<li>暂无归档。</li>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>低空经济前沿论文库</title>
  <style>
    body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#f4f6f8;color:#172033;line-height:1.7}}
    main{{max-width:860px;margin:0 auto;padding:30px 18px 64px}}
    h1{{font-size:28px;margin:0 0 8px}}
    .muted{{color:#667085}}
    ul{{list-style:none;padding:0;margin:22px 0}}
    li{{margin:10px 0}}
    a{{display:block;background:#fff;border:1px solid #d8dee9;border-radius:10px;padding:14px 16px;color:#0f766e;text-decoration:none;font-weight:800}}
  </style>
</head>
<body>
  <main>
    <h1>低空经济通用大模型前沿论文库</h1>
    <p class="muted">最后更新：{today}。每天北京时间 05:00 自动生成；Top 10 进行 PDF 全文与页码核验。</p>
    <ul>{items}</ul>
  </main>
</body>
</html>
"""


def render_markdown_digest(items: list[Item], today: str, base_url: str) -> str:
    lines = [
        f"网页版本：{base_url}/{today}/",
        "",
        "以下内容来自 arXiv 元数据与摘要；未读取论文全文的条目不能视为精读结论。",
        "",
    ]
    for idx, item in enumerate(items, 1):
        notes = rule_based_notes(item)
        lines.extend(
            [
                f"## {idx}. {item.title_zh or item.title}",
                "",
                f"- English title: {item.title}",
                f"- Authors: {item.authors or 'Unknown'}",
                f"- Source: {item.source}",
                f"- Published: {item.published}",
                f"- Original: {item.url}",
                f"- PDF: {item.pdf_url}",
                "",
                "### 摘要级内容",
                "",
                item.summary_zh or item.summary,
                "",
                "### 相关性与阅读建议",
                "",
                item.relevance_zh or notes["relevance"],
                "",
                item.practice_zh or notes["practice"],
                "",
            ]
        )
    return "\n".join(lines).strip()


def _send_email_once(subject: str, body: str) -> None:
    host = require_env("QQ_SMTP_HOST")
    port = int(require_env("QQ_SMTP_PORT"))
    user = require_env("QQ_SMTP_USER")
    password = require_env("QQ_SMTP_AUTH_CODE")
    from_addr = require_env("QQ_SMTP_FROM")
    to_addr = require_env("QQ_SMTP_TO")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body, charset="utf-8")
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context, timeout=45) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)


def send_email(subject: str, body: str, attempts: int = 3) -> None:
    for attempt in range(attempts):
        try:
            _send_email_once(subject, body)
            return
        except smtplib.SMTPAuthenticationError:
            raise
        except (smtplib.SMTPException, OSError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(5.0 * (2**attempt), 30.0))


def completed_report_already_sent(report_date: str) -> bool:
    path = report_contract.OUTPUTS_DIR / f"{report_date}.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        report.get("stream") == STREAM
        and report.get("report_date") == report_date
        and report.get("generation_status") in {"complete", "partial"}
        and report.get("email_status") == "sent"
        and int(report.get("item_count") or 0) > 0
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Generate files without sending email.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip optional DeepSeek enrichment.")
    parser.add_argument("--force-send", action="store_true", help="Regenerate and resend even if today succeeded.")
    args = parser.parse_args()
    load_env_file()
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    if not args.dry_run and not args.force_send and completed_report_already_sent(today):
        print(f"Today's paper-library report was already sent; backup run skipped ({today}).")
        return
    base_url = page_base_url()
    source_errors: list[str] = []
    all_items = dedupe(fetch_arxiv(source_errors))
    new_items = select_items(all_items, load_history())

    all_arxiv_queries_failed = sum(error.startswith("arXiv query ") for error in source_errors) >= 5
    if not new_items and not all_items and all_arxiv_queries_failed:
        subject = f"{REPORT_TITLE}生成失败 - {today}"
        body = "今日未取得可验证的 arXiv 候选条目。已保存失败记录，请稍后重试。"
        report = build_report(
            stream=STREAM,
            title=subject,
            body=body,
            items=[],
            generation_status="failed",
            email_status="skipped" if args.dry_run else "pending",
            source_errors=source_errors or ["arXiv: no verified candidates"],
            report_date=today,
            default_kind="paper",
        )
        write_report(report)
        if args.dry_run:
            print(body)
            return
        try:
            send_email(subject, body)
            report["email_status"] = "sent"
            write_report(report)
        except Exception:
            report["email_status"] = "failed"
            write_report(report)
            raise
        raise RuntimeError("No candidate papers were found; a failure notice was saved and emailed.")

    if new_items and len(new_items) < MAX_ITEMS:
        source_errors.append(f"Only {len(new_items)} of {MAX_ITEMS} requested new papers were available")

    processed_new: list[Item] = []
    if new_items:
        llm_configs = resolve_llm_configs()
        invoke_json = None if args.skip_llm else make_json_invoker(llm_configs)
        if not args.skip_llm:
            new_items, translation_errors = enrich_items_with_llm(new_items, invoke_json)
            source_errors.extend(translation_errors)
        for item in new_items:
            item.title_en = item.title_en or item.title
            item.abstract_en = item.abstract_en or item.summary
            item.abstract_zh = item.abstract_zh or item.summary_zh
        processed_new, pdf_errors = process_top_papers(new_items, RESEARCH_PROFILE, invoke_json)
        source_errors.extend(pdf_errors)

    verified_new = [item for item in processed_new if item.analysis_rank]
    needed_reviews = max(0, 10 - len(verified_new))
    review_records = load_recent_review_records(
        REPORTS_DIR,
        STREAM,
        today,
        limit=needed_reviews,
        exclude_keys=(item.key for item in processed_new),
    )
    review_items = [item_from_review_record(record) for record in review_records]
    next_rank = max((item.analysis_rank or 0 for item in processed_new), default=0)
    for offset, item in enumerate(review_items, 1):
        item.analysis_rank = next_rank + offset
    selected = processed_new + review_items
    title_suffix = "（今日无新增·近30天回看）" if not processed_new else ""

    daily_dir = DOCS_DIR / today
    daily_dir.mkdir(parents=True, exist_ok=True)
    web_url = f"{base_url}/{today}/"
    (daily_dir / "index.html").write_text(
        render_html(selected, REPORT_TITLE + title_suffix, today, base_url), encoding="utf-8"
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(render_index_deepseek(today, base_url), encoding="utf-8")
    body = build_daily_markdown(selected, REPORT_TITLE + title_suffix, today, web_url)
    email_body = build_email_summary(selected, REPORT_TITLE + title_suffix, today, web_url)
    generation_status = "partial" if source_errors else "complete"
    report = build_report(
        stream=STREAM,
        title=f"{REPORT_TITLE}{title_suffix} - {today}",
        body=body,
        items=selected,
        generation_status=generation_status,
        email_status="skipped" if args.dry_run else "pending",
        source_errors=source_errors,
        report_date=today,
        default_kind="paper",
    )
    write_report(report)
    if args.dry_run:
        print(f"Generated {len(selected)} papers without sending email: {web_url}")
        return
    try:
        send_email(f"{REPORT_TITLE}{title_suffix} - {today}", email_body)
        report["email_status"] = "sent"
        write_report(report)
        save_history(processed_new)
    except Exception:
        report["email_status"] = "failed"
        write_report(report)
        raise
    print(f"Generated and emailed {len(selected)} papers ({generation_status}): {web_url}")


if __name__ == "__main__":
    main()
