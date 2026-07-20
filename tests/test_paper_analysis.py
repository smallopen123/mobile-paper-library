from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore[no-redef]


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import paper_analysis as analysis  # noqa: E402


@dataclass
class FakeItem:
    title: str
    url: str
    pdf_url: str
    summary: str
    authors: str = "Researcher"
    source: str = "arXiv cs.RO"
    published: str = "2026-07-20"
    title_en: str = ""
    title_zh: str = ""
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
    relevance_zh: str = ""
    reproducibility_zh: str = ""
    research_idea_zh: str = ""
    core_figure: dict = field(default_factory=dict)
    summary_diagram_mermaid: str = ""
    diagram_source_pages: list[int] = field(default_factory=list)
    fulltext_status: str = "not_attempted"
    figure_status: str = "not_attempted"
    pdf_page_count: int = 0
    parsed_page_count: int = 0

    @property
    def key(self) -> str:
        return self.title.lower().replace(" ", "-")


def make_pdf(path: Path, caption: str = "Figure 1. Overview of the proposed framework") -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((60, 65), "Abstract This paper studies safe UAV trajectory forecasting with uncertainty calibration.")
    page.draw_rect(fitz.Rect(90, 170, 500, 410), color=(0.1, 0.5, 0.5), width=2)
    page.insert_text((115, 235), "UAV observations -> encoder -> world model -> safe trajectory")
    page.insert_text((60, 440), caption)
    page.insert_text((60, 500), "Method. We train the model on AeroSet and compare against CVAE.")
    page.insert_text((60, 535), "The minimum ADE is 1.20 m on AeroSet.")
    page.insert_text((60, 570), "A limitation is evaluation in one city only.")
    document.save(path)
    document.close()


class PaperAnalysisTests(unittest.TestCase):
    def test_inspect_pdf_finds_framework_and_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            make_pdf(path)
            inspected = analysis.inspect_pdf(path)
        self.assertEqual(inspected.page_count, 1)
        self.assertEqual(inspected.core_figure["status"], "found")
        self.assertEqual(inspected.core_figure["page"], 1)
        self.assertEqual(len(inspected.core_figure["crop_bbox"]), 4)

    def test_unrelated_result_figure_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.pdf"
            make_pdf(path, "Figure 1. Accuracy curves on the test split")
            inspected = analysis.inspect_pdf(path)
        self.assertEqual(inspected.core_figure["status"], "not_found")

    def test_evidence_and_pages_are_validated(self) -> None:
        inspection = analysis.PdfInspection(
            page_count=2,
            parsed_pages=2,
            page_texts=["We evaluate on AeroSet. The minimum ADE is 1.20 m.", "Limitations include one city only."],
            core_figure={"status": "not_found"},
        )
        item = FakeItem("Paper", "https://example.test", "https://example.test/p.pdf", "Abstract")
        analysis.apply_fulltext_payload(
            item,
            inspection,
            {
                "datasets": [{"name": "AeroSet", "page": 1, "evidence": "evaluate on AeroSet"}],
                "metrics": [
                    {"name": "ADE 1.20 m", "page": 1, "evidence": "The minimum ADE is 1.20 m"},
                    {"name": "fabricated", "page": 9, "evidence": "not present"},
                ],
                "source_pages": [1, 2, 99],
                "summary_flow": ["UAV observations", "World Model", "Safe trajectory", "ADE"],
                "diagram_source_pages": [1, 99],
            },
        )
        self.assertEqual(len(item.metrics), 1)
        self.assertEqual(item.source_pages, [1, 2])
        self.assertIn("flowchart LR", item.summary_diagram_mermaid)
        self.assertEqual(item.diagram_source_pages, [1])

    def test_failed_pdf_is_replaced_by_next_candidate(self) -> None:
        items = [
            FakeItem("Broken", "https://example.test/1", "https://example.test/1.pdf", "Abstract"),
            FakeItem("Good", "https://example.test/2", "https://example.test/2.pdf", "Abstract"),
        ]

        def downloader(url: str, path: Path) -> int:
            if url.endswith("1.pdf"):
                raise analysis.PdfFormatError("broken")
            path.write_bytes(b"%PDF-placeholder")
            return path.stat().st_size

        inspected = analysis.PdfInspection(
            1,
            1,
            ["Enough extractable paper text describing a complete method and evaluation." * 8],
            {"status": "not_found"},
        )
        result, errors = analysis.process_top_papers(
            items,
            "test profile",
            None,
            top_n=1,
            downloader=downloader,
            inspector=lambda _path: inspected,
        )
        self.assertIsNone(result[0].analysis_rank)
        self.assertEqual(result[0].fulltext_status, "invalid_pdf")
        self.assertEqual(result[1].analysis_rank, 1)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
