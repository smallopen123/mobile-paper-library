from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import html
import json
import os
import re
import smtplib
import ssl
import textwrap
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

import requests
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
HISTORY_PATH = ROOT / "data" / "sent_history.json"
MAX_ITEMS = 20
RECENT_DAYS = 7
PRIMARY_DAYS = 3

RESEARCH_PROFILE = """
博士研究方向：低空经济安全评估、低空空域/无人机/城市空中交通安全风险建模、
航迹预测、时空序列建模、多智能体/具身智能/机器人学习、智能体协同与决策。
重点会议：IROS、ICRA、RSS、CoRL、NeurIPS、ICML、ICLR、KDD、AAAI、IJCAI、ICDE。
"""

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
    "multi-agent",
    "robot learning",
    "embodied ai",
    "autonomous navigation",
    "collision avoidance",
    "intent prediction",
    "world model",
    "diffusion policy",
    "reinforcement learning",
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

    @property
    def key(self) -> str:
        raw = f"{self.title}|{self.url}".lower()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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


def request_text(url: str) -> str:
    response = requests.get(url, timeout=35)
    response.raise_for_status()
    return response.text


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


def relevance_score(item: Item) -> float:
    haystack = f"{item.title} {item.summary} {item.source}".lower()
    score = 0.0
    for keyword in KEYWORDS:
        if keyword in haystack:
            score += 2.0
    for term in ["iros", "icra", "rss", "corl", "neurips", "icml", "iclr", "kdd", "aaai", "ijcai", "icde"]:
        if term in haystack:
            score += 1.2
    if "uav" in haystack or "drone" in haystack:
        score += 3.0
    if "trajectory" in haystack or "motion forecasting" in haystack:
        score += 3.0
    if "risk" in haystack or "safety" in haystack:
        score += 2.5
    if "multi-agent" in haystack:
        score += 2.0
    age_days = max((dt.datetime.now(dt.timezone.utc) - parse_date(item.published)).days, 0)
    score += max(0, 4 - age_days * 0.5)
    return score


def fetch_arxiv() -> list[Item]:
    queries = [
        '(ti:"trajectory prediction" OR abs:"trajectory prediction" OR ti:"motion forecasting" OR abs:"motion forecasting")',
        '(ti:UAV OR abs:UAV OR ti:drone OR abs:drone OR abs:"urban air mobility" OR abs:"advanced air mobility")',
        '(abs:"multi-agent" OR ti:"multi-agent" OR abs:"robot learning" OR abs:"embodied AI")',
        '(abs:"risk assessment" OR abs:"airspace safety" OR abs:"collision avoidance")',
        '(abs:"world model" OR abs:"diffusion policy" OR abs:"autonomous navigation")',
    ]
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    items: list[Item] = []
    for query in queries:
        params = urllib.parse.urlencode(
            {
                "search_query": f"({query}) AND (cat:cs.RO OR cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:eess.SY OR cat:stat.ML)",
                "start": 0,
                "max_results": 45,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        root = ET.fromstring(request_text(f"https://export.arxiv.org/api/query?{params}"))
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
    return items


def dedupe(items: Iterable[Item]) -> list[Item]:
    seen: set[str] = set()
    result: list[Item] = []
    for item in items:
        if not item.title or not item.url:
            continue
        if item.key in seen:
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
    filtered = [item for item in items if item.key not in history and parse_date(item.published) >= recent_cutoff]
    primary = [item for item in filtered if parse_date(item.published) >= primary_cutoff]
    candidates = primary if len(primary) >= MAX_ITEMS else filtered
    if len(candidates) < MAX_ITEMS:
        candidates = items
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:MAX_ITEMS]


def extract_json(text: str) -> list[dict]:
    match = re.search(r"```json\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        text = match.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"sk-[A-Za-z0-9_\-]+", "[redacted-openai-key]", message)
    message = re.sub(r"sk-proj-[A-Za-z0-9_\-]+", "[redacted-openai-key]", message)
    return message[:1200]


def fallback_analysis(items: list[Item]) -> list[dict]:
    rows = []
    for item in items:
        rows.append(
            {
                "key": item.key,
                "english_core": item.summary,
                "chinese_translation": "未生成模型翻译。请检查 OPENAI_API_KEY 后重新运行 workflow。",
                "method_theory": "候选论文已收集；方法理论分析需要模型生成。",
                "why_frontier": "该条目因关键词、发布时间和研究方向相关性被选入今日论文库。",
                "innovation_practice": "可先阅读原文 PDF，提取数据、任务、模型和评价指标，再设计低空安全或航迹预测迁移实验。",
            }
        )
    return rows


def generate_analysis(items: list[Item]) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI analysis skipped: OPENAI_API_KEY is missing.")
        return fallback_analysis(items)
    model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    print(f"OpenAI analysis enabled: key_present=yes, model={model}")
    client = OpenAI(api_key=api_key)
    source = json.dumps([asdict(item) | {"key": item.key} for item in items], ensure_ascii=False, indent=2)
    prompt = f"""
你是面向博士研究的低空经济与智能体论文分析助手。请基于候选论文生成 JSON 数组，必须只输出 JSON。

{RESEARCH_PROFILE}

每个数组元素必须包含：
- key：沿用输入中的 key
- english_core：英文核心内容，优先概括 abstract 中的核心问题、方法、实验，不超过 120 words
- chinese_translation：对应中文翻译与解释，不是全文翻译，强调博士研究者读论文需要抓住什么
- method_theory：方法理论分析，包含建模假设、关键算法/理论工具、相较传统方法的改进、适用于低空安全评估/航迹预测/智能体系统的方式、潜在局限
- why_frontier：为什么前沿，从新问题、新场景、新模型、新训练范式、新指标、安全性/实时性/可解释性、顶会趋势或产业价值解释
- innovation_practice：拓展创新方向与具体实践思路，包含可验证假设、数据/仿真环境、模型改造、实验设计、评价指标、消融实验、落地路径

要求：
- 不编造输入中没有的作者、链接或录用状态。
- 语言专业、紧凑、可直接用于手机阅读。
- 必须覆盖输入的全部 {len(items)} 篇论文。

候选论文：
{source}
"""
    try:
        response = client.responses.create(model=model, input=prompt, temperature=0.25)
        data = extract_json(response.output_text)
        by_key = {row.get("key"): row for row in data if row.get("key")}
        fallback = {row["key"]: row for row in fallback_analysis(items)}
        print(f"OpenAI analysis generated rows: {len(by_key)}")
        return [by_key.get(item.key, fallback[item.key]) for item in items]
    except Exception:
        raise


def generate_analysis_with_fallback(items: list[Item]) -> list[dict]:
    try:
        return generate_analysis(items)
    except Exception as exc:
        print(f"Using fallback analysis because OpenAI failed: {safe_error(exc)}")
        return fallback_analysis(items)


def page_base_url() -> str:
    explicit = os.getenv("PAGES_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    repo = os.getenv("GITHUB_REPOSITORY", "smallopen123/mobile-paper-library")
    owner, name = repo.split("/", 1)
    return f"https://{owner}.github.io/{name}"


def render_daily_page(items: list[Item], analysis: list[dict], today: str, base_url: str) -> str:
    cards = []
    analysis_by_key = {row["key"]: row for row in analysis}
    for idx, item in enumerate(items, 1):
        row = analysis_by_key.get(item.key, {})
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
              <section><h3>English Core</h3><p>{html.escape(row.get("english_core", item.summary))}</p></section>
              <section><h3>中文对照</h3><p>{html.escape(row.get("chinese_translation", ""))}</p></section>
              <section><h3>方法理论分析</h3><p>{html.escape(row.get("method_theory", ""))}</p></section>
              <section><h3>为什么前沿</h3><p>{html.escape(row.get("why_frontier", ""))}</p></section>
              <section><h3>拓展创新方向与实践思路</h3><p>{html.escape(row.get("innovation_practice", ""))}</p></section>
            </article>
            """
        )
    nav = "\n".join(f'<a href="#paper-{idx}">{idx:02d}</a>' for idx in range(1, len(items) + 1))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>低空经济前沿双语论文库 - {today}</title>
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
    .back{{display:inline-block;margin:14px 0 0;color:#d6f2ee}}
    @media (max-width:520px){{h1{{font-size:23px}}h2{{font-size:18px}}.paper{{padding:16px 14px}}}}
  </style>
</head>
<body>
  <header><div class="wrap">
    <h1>低空经济前沿双语论文库</h1>
    <p class="sub">{today} · 20 篇论文 · 手机阅读版</p>
    <a class="back" href="{base_url}/">查看历史归档</a>
  </div></header>
  <main>
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
            entries.append(f'<li><a href="{base_url}/{label}/">{label} 双语论文库</a></li>')
    items = "\n".join(entries) or "<li>暂无归档</li>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>低空经济前沿双语论文库</title>
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
    <h1>低空经济前沿双语论文库</h1>
    <p class="muted">最后更新：{today}。每天北京时间 09:00 自动生成。</p>
    <ul>{items}</ul>
  </main>
</body>
</html>
"""


def send_email(subject: str, body: str) -> None:
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


def main() -> None:
    load_env_file()
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    base_url = page_base_url()
    all_items = dedupe(fetch_arxiv())
    selected = select_items(all_items, load_history())
    if len(selected) < MAX_ITEMS:
        raise RuntimeError(f"Only found {len(selected)} candidate papers.")
    analysis = generate_analysis_with_fallback(selected)

    daily_dir = DOCS_DIR / today
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_page = render_daily_page(selected, analysis, today, base_url)
    (daily_dir / "index.html").write_text(daily_page, encoding="utf-8")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(render_index(today, base_url), encoding="utf-8")
    save_history(selected)

    top_titles = "\n".join(f"{idx}. {item.title}" for idx, item in enumerate(selected[:8], 1))
    body = textwrap.dedent(
        f"""
        今日低空经济前沿双语论文库已生成：
        {base_url}/{today}/

        手机浏览器打开即可查看 20 篇论文的原文 PDF 链接、英文核心内容、中文对照解释、方法理论分析和创新实践思路。

        今日部分条目：
        {top_titles}

        历史归档：
        {base_url}/
        """
    ).strip()
    send_email(f"低空经济前沿双语论文库 - {today}", body)
    print(f"Generated mobile paper library for {today}: {base_url}/{today}/")


if __name__ == "__main__":
    main()
