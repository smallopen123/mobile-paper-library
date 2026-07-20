from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object
    sys.modules["openai"] = fake_openai

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import backfill_recent as backfill  # noqa: E402


class BackfillRecentTests(unittest.TestCase):
    def test_utc_send_time_maps_to_beijing_report_date(self) -> None:
        payload = {
            "sent": [
                {
                    "key": "x",
                    "title": "Paper",
                    "url": "https://arxiv.org/abs/2607.00001",
                    "sent_at": "2026-07-17T21:55:00+00:00",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            history = Path(temp_dir) / "history.json"
            history.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(backfill.library, "HISTORY_PATH", history):
                grouped = backfill.history_by_report_date()
        self.assertIn("2026-07-18", grouped)


if __name__ == "__main__":
    unittest.main()
