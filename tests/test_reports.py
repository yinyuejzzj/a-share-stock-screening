import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tail_screen"))
from wei_pan_core import config, orchestrator, report_formatter


class ReportTests(unittest.TestCase):
    def test_package_exports_are_importable(self):
        namespace = {}
        exec("from wei_pan_core import *", namespace)
        self.assertIn("sector_ranking", namespace)

    def test_effective_thresholds_are_visible_in_both_reports(self):
        result = {"mode": "B", "filter_report": {"thresholds": {"pct_lo": .025, "pct_hi": .06}}}
        for renderer in (report_formatter.format_markdown, report_formatter.format_html):
            rendered = renderer(result)
            self.assertIn("pct_lo", rendered)
            self.assertIn("0.025", rendered)

    def test_exit_rule_text_uses_configured_percentages_and_time(self):
        top = pd.DataFrame({"close": [10., 10., 10., 10.]})
        with patch.dict(config.SELL_STRATEGIES["B"], {"stop_loss_pct": -.07, "take_profit_pct": .09}), \
             patch.dict(config.SELL_STRATEGIES["C"], {"hard_stop_loss_pct": -.06, "take_profit_fast_pct": .08,
                 "trailing_trigger_pct": .03, "trailing_drawdown_pct": .01, "final_close_time": "09:55"}):
            result = orchestrator._assign_sell_strategies(top)
        self.assertIn("-7.0%", result.iloc[1]["sell_detail"])
        self.assertIn("+9.0%", result.iloc[1]["sell_detail"])
        self.assertIn("-6.0%", result.iloc[3]["sell_detail"])
        self.assertIn("+8.0%", result.iloc[3]["sell_detail"])
        self.assertIn("09:55", result.iloc[3]["sell_detail"])


if __name__ == "__main__":
    unittest.main()
