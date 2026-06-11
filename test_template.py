"""Unit tests for the deterministic core of the Munder Difflin multi-agent system.

These cover the parts that must be correct regardless of the LLM: item-name
normalization, unit conversion, the bulk-discount ladder, free-text line parsing,
the retrieval-augmented discount benchmark, structured tool results, and a few
DB-backed helpers/tools. No live model calls are made.

Run from the project/ directory:

    .venv/bin/python -m unittest test_template -v
"""
import unittest

import template as t


class TestNormalization(unittest.TestCase):
    def test_fuzzy_aliases_resolve(self):
        self.assertEqual(t.normalize_item_name("A4 glossy paper"), "Glossy paper")
        self.assertEqual(t.normalize_item_name("heavy cardstock (white)"), "Cardstock")
        self.assertEqual(t.normalize_item_name("decorative washi tape"),
                         "Decorative adhesive tape (washi tape)")
        self.assertEqual(t.normalize_item_name("table napkins (white)"), "Paper napkins")
        self.assertEqual(t.normalize_item_name("white printer paper"), "Standard copy paper")

    def test_not_carried_returns_none(self):
        for raw in ("A3 paper", "A5 colored paper", "balloons", "cardboard for signage"):
            self.assertIsNone(t.normalize_item_name(raw), raw)


class TestUnitConversion(unittest.TestCase):
    def test_reams_multiply_for_paper(self):
        qty, note = t.convert_to_catalog_units(500, "reams", "A4 paper")
        self.assertEqual(qty, 250_000)
        self.assertIn("sheets", note)

    def test_bulk_unit_not_applied_to_products(self):
        qty, _ = t.convert_to_catalog_units(3, "boxes", "Paper cups")
        self.assertEqual(qty, 3)

    def test_sheets_are_one_to_one(self):
        qty, _ = t.convert_to_catalog_units(200, "sheets", "Glossy paper")
        self.assertEqual(qty, 200)


class TestBulkDiscount(unittest.TestCase):
    def test_ladder_boundaries(self):
        self.assertEqual(t.bulk_discount(50), 0.0)
        self.assertEqual(t.bulk_discount(99), 0.0)
        self.assertEqual(t.bulk_discount(100), 0.02)
        self.assertEqual(t.bulk_discount(499), 0.02)
        self.assertEqual(t.bulk_discount(500), 0.05)
        self.assertEqual(t.bulk_discount(999), 0.05)
        self.assertEqual(t.bulk_discount(1000), 0.08)


class TestHistoricalDiscountExtraction(unittest.TestCase):
    def test_extracts_pct_only_when_discount_mentioned(self):
        quotes = [{"quote_explanation": "we offer a 10% discount", "original_request": ""}]
        self.assertEqual(t._extract_discount_fractions(quotes), [0.10])

    def test_ignores_pct_without_discount_word(self):
        quotes = [{"quote_explanation": "made from 99% recycled fiber", "original_request": ""}]
        self.assertEqual(t._extract_discount_fractions(quotes), [])

    def test_ignores_implausible_pct(self):
        quotes = [{"quote_explanation": "a 99% discount (typo)", "original_request": ""}]
        self.assertEqual(t._extract_discount_fractions(quotes), [])  # > 50% rejected


class TestLineParsing(unittest.TestCase):
    def test_bulleted_multiline(self):
        text = ("- 200 sheets of A4 glossy paper\n"
                "- 100 sheets of heavy cardstock (white)")
        items = t.parse_line_items(text)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["catalog_name"], "Glossy paper")
        self.assertEqual(items[0]["qty_units"], 200)
        self.assertEqual(items[1]["catalog_name"], "Cardstock")

    def test_inline_prose_with_connector(self):
        text = ("I need 500 sheets of recycled cardstock, along with "
                "250 sheets of A4 printer paper.")
        items = t.parse_line_items(text)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["qty_raw"], 500)
        self.assertEqual(items[1]["qty_raw"], 250)

    def test_declined_names_have_no_trailing_connector(self):
        text = "200 sheets of A3 paper, and 500 sheets of A4 glossy paper"
        for it in t.parse_line_items(text):
            self.assertNotRegex(it["raw_name"].lower(), r"\b(and|plus|along)\s*$")

    def test_date_and_deadline_not_parsed_as_items(self):
        text = ("100 sheets of A4 paper delivered by April 15, 2025 "
                "(Date of request: 2025-04-01)")
        items = t.parse_line_items(text)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["catalog_name"], "A4 paper")

    def test_request_date_and_deadline_extraction(self):
        text = "order (Date of request: 2025-04-07) delivered by April 15, 2025"
        self.assertEqual(t.parse_request_date(text), "2025-04-07")
        self.assertEqual(t.parse_deadline(text), "2025-04-15")


class TestToolResult(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(t.ok(a=1), {"status": "ok", "data": {"a": 1},
                                     "error_type": "", "message": ""})
        self.assertEqual(t.err("db_error", "boom")["status"], "error")
        self.assertEqual(t.err("db_error", "boom")["error_type"], "db_error")
        self.assertEqual(t.declined("nope")["status"], "declined")


class TestDatabaseBacked(unittest.TestCase):
    """Helpers/tools that read the seeded SQLite DB (deterministic, no LLM)."""

    @classmethod
    def setUpClass(cls):
        t.init_database(t.db_engine)          # fixed seed 137 -> reproducible inventory
        cls.date = "2025-04-01"

    def test_catalog_price(self):
        self.assertEqual(t.catalog_price("A4 paper"), 0.05)
        self.assertIsNone(t.catalog_price("Nonexistent item"))

    def test_stock_level_is_nonnegative_int(self):
        stock = t._stock_level("A4 paper", self.date)
        self.assertIsInstance(stock, int)
        self.assertGreaterEqual(stock, 0)
        self.assertEqual(t._stock_level("Definitely Not A Real Item", self.date), 0)

    def test_historical_discount_from_quotes(self):
        frac, n = t.historical_discount(["party"])
        self.assertAlmostEqual(frac, 0.10, places=2)
        self.assertGreaterEqual(n, 1)
        # No comparable history -> falls back to no benchmark.
        self.assertEqual(t.historical_discount(["zzz-not-a-real-term"]), (0.0, 0))

    def test_check_item_stock_tool_returns_structured_ok(self):
        result = t.check_item_stock("A4 paper", self.date)
        self.assertEqual(result["status"], "ok")
        self.assertIn("current_stock", result["data"])

    def test_find_similar_quotes_reports_avg_discount(self):
        result = t.find_similar_quotes("party")
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["data"]["avg_discount_pct"], 10.0, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
