from __future__ import annotations

import json
import re
import smtplib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import requests


if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object
    sys.modules["openai"] = fake_openai

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mobile_paper_library as library  # noqa: E402
import report_contract  # noqa: E402


class _Response:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status
        self.headers: dict[str, str] = {"Retry-After": "1"} if status == 429 else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class MobilePaperLibraryTests(unittest.TestCase):
    def test_failed_translation_batch_retries_each_paper(self) -> None:
        items = [
            library.Item(
                title=f"Paper {idx}",
                url=f"https://arxiv.org/abs/2607.{idx:05d}",
                pdf_url=f"https://arxiv.org/pdf/2607.{idx:05d}",
                source="arXiv cs.RO",
                published="2026-07-20T00:00:00Z",
                authors="Researcher",
                summary=f"English abstract {idx}",
                score=10,
            )
            for idx in range(1, 6)
        ]

        def invoke(prompt: str) -> dict:
            indices = [int(value) for value in re.findall(r'"idx":\s*(\d+)', prompt)]
            if len(indices) > 1:
                raise ValueError("malformed batch response")
            idx = indices[0]
            return {
                "items": [
                    {
                        "idx": idx,
                        "title_zh": f"论文{idx}",
                        "abstract_zh": f"这是第{idx}篇论文的中文摘要。",
                        "reading_hint_zh": "先读问题与实验。",
                        "relevance_zh": "与低空通用大模型研究相关。",
                        "practice_zh": "建立最小复现实验。",
                    }
                ]
            }

        enriched, errors = library.enrich_items_with_llm(items, invoke)
        self.assertEqual(errors, [])
        self.assertTrue(all(item.title_zh.startswith("论文") for item in enriched))
        self.assertTrue(all("中文摘要" in item.abstract_zh for item in enriched))

    def test_json_parser_accepts_fenced_array(self) -> None:
        parsed = library._parse_json_object('```json\n[{"idx": 1}]\n```')
        self.assertEqual(parsed["items"][0]["idx"], 1)

    def test_retry_after_429(self) -> None:
        with mock.patch.object(
            library.requests, "get", side_effect=[_Response("", 429), _Response("<feed />")]
        ), mock.patch.object(library.time, "sleep") as sleep:
            self.assertEqual(library.request_text("https://example.test", attempts=2), "<feed />")
        sleep.assert_called_once_with(1.0)

    def test_retry_after_503(self) -> None:
        with mock.patch.object(
            library.requests, "get", side_effect=[_Response("", 503), _Response("<feed />")]
        ), mock.patch.object(library.time, "sleep") as sleep, mock.patch.object(
            library.random, "uniform", return_value=0.0
        ):
            self.assertEqual(library.request_text("https://example.test", attempts=2), "<feed />")
        sleep.assert_called_once_with(5.0)

    def test_smtp_transient_failure_is_retried(self) -> None:
        with mock.patch.object(
            library,
            "_send_email_once",
            side_effect=[smtplib.SMTPServerDisconnected("temporary"), None],
        ) as send, mock.patch.object(library.time, "sleep") as sleep:
            library.send_email("subject", "body")
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(5.0)

    def test_successful_report_skips_scheduled_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            report_contract, "OUTPUTS_DIR", Path(temp_dir)
        ), mock.patch.object(library, "fetch_arxiv") as fetch, mock.patch.object(
            library, "send_email"
        ) as send, mock.patch.object(sys, "argv", ["mobile_paper_library.py"]):
            today = library.dt.datetime.now(
                library.dt.timezone(library.dt.timedelta(hours=8))
            ).date().isoformat()
            (Path(temp_dir) / f"{today}.json").write_text(
                json.dumps(
                    {
                        "stream": library.STREAM,
                        "report_date": today,
                        "generation_status": "partial",
                        "email_status": "sent",
                        "item_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            library.main()
        fetch.assert_not_called()
        send.assert_not_called()

    def test_dry_run_with_fewer_items_writes_partial_report(self) -> None:
        item = library.Item(
            title="Safe UAV forecasting",
            url="https://arxiv.org/abs/2607.00001",
            pdf_url="https://arxiv.org/pdf/2607.00001",
            source="arXiv cs.RO",
            published="2026-07-19T00:00:00Z",
            authors="Researcher",
            summary="Abstract metadata",
            score=10,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with mock.patch.object(report_contract, "OUTPUTS_DIR", temp / "outputs"), mock.patch.object(
                library, "DOCS_DIR", temp / "docs"
            ), mock.patch.object(library, "HISTORY_PATH", temp / "history.json"), mock.patch.object(
                library, "REPORTS_DIR", temp / "reports"
            ), mock.patch.object(
                library, "fetch_arxiv", return_value=[item]
            ), mock.patch.object(
                library, "process_top_papers", return_value=([item], ["Only 0 of 10 papers had a downloadable, extractable PDF"])
            ), mock.patch.object(sys, "argv", ["mobile_paper_library.py", "--dry-run", "--skip-llm"]):
                library.main()
            report_path = next((temp / "outputs").glob("*.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["generation_status"], "partial")
        self.assertEqual(report["email_status"], "skipped")
        self.assertEqual(report["item_count"], 1)
        self.assertEqual(report["schema_version"], 2)
        self.assertIn("title_en", report["items"][0])

    def test_zero_new_candidates_uses_verified_review_and_is_not_failed(self) -> None:
        previous = library.Item(
            title="Verified UAV foundation model",
            url="https://arxiv.org/abs/2607.00009",
            pdf_url="https://arxiv.org/pdf/2607.00009",
            source="arXiv cs.RO",
            published="2026-07-20T00:00:00Z",
            authors="Researcher",
            summary="A verified English abstract.",
            score=10,
            title_en="Verified UAV foundation model",
            title_zh="已核验的无人机基础模型",
            abstract_en="A verified English abstract.",
            abstract_zh="这是一段已核验的中文摘要。",
            analysis_rank=1,
            fulltext_status="verified",
            figure_status="not_found",
            core_figure={"status": "not_found"},
        )
        current_date = library.dt.datetime.now(library.dt.timezone(library.dt.timedelta(hours=8))).date()
        previous_date = (current_date - library.dt.timedelta(days=1)).isoformat()
        dynamic = library.Item(
            title="Already sent paper",
            url="https://arxiv.org/abs/2607.00010",
            pdf_url="https://arxiv.org/pdf/2607.00010",
            source="arXiv cs.RO",
            published=current_date.isoformat(),
            authors="Researcher",
            summary="Already handled.",
            score=10,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            reports = temp / "reports"
            reports.mkdir()
            (reports / f"{previous_date}.json").write_text(
                json.dumps(
                    {
                        "stream": library.STREAM,
                        "report_date": previous_date,
                        "items": [report_contract.serialize_item(previous)],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with mock.patch.object(report_contract, "OUTPUTS_DIR", temp / "outputs"), mock.patch.object(
                library, "REPORTS_DIR", reports
            ), mock.patch.object(library, "DOCS_DIR", temp / "docs"), mock.patch.object(
                library, "HISTORY_PATH", temp / "history.json"
            ), mock.patch.object(library, "fetch_arxiv", return_value=[dynamic]), mock.patch.object(
                library, "select_items", return_value=[]
            ), mock.patch.object(sys, "argv", ["mobile_paper_library.py", "--dry-run", "--skip-llm"]):
                library.main()
            report = json.loads(next((temp / "outputs").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["generation_status"], "complete")
        self.assertEqual(report["email_status"], "skipped")
        self.assertEqual(report["item_count"], 1)
        self.assertEqual(report["items"][0]["selection_mode"], "review")
        self.assertEqual(report["items"][0]["source_report_date"], previous_date)
        self.assertIn("今日新增 0 篇", report["body"])
        self.assertIn("回看条目不是今日新增", report["body"])

    def test_all_arxiv_queries_failed_still_creates_failure_report(self) -> None:
        def failed_source(errors: list[str]) -> list[library.Item]:
            errors.extend(f"arXiv query {index}: ReadTimeout" for index in range(1, 6))
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with mock.patch.object(report_contract, "OUTPUTS_DIR", temp / "outputs"), mock.patch.object(
                library, "REPORTS_DIR", temp / "reports"
            ), mock.patch.object(library, "DOCS_DIR", temp / "docs"), mock.patch.object(
                library, "HISTORY_PATH", temp / "history.json"
            ), mock.patch.object(library, "fetch_arxiv", side_effect=failed_source), mock.patch.object(
                sys, "argv", ["mobile_paper_library.py", "--dry-run", "--skip-llm"]
            ):
                library.main()
            report = json.loads(next((temp / "outputs").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["generation_status"], "failed")
        self.assertEqual(report["item_count"], 0)


if __name__ == "__main__":
    unittest.main()
