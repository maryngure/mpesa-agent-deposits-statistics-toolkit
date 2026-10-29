"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import unittest
import tempfile
from pathlib import Path

from stats_toolkit import (
    mean, median, mode, variance, std_dev, quartiles, iqr_outliers,
    StatisticsAnalyzer, load_column,
)


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.assertEqual(mean([2, 4, 6]), 4)
        self.assertEqual(mean([10]), 10)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([3, 1, 4, 1, 5]), 3)    # odd: middle of [1, 1, 3, 4, 5]
        self.assertEqual(median([3, 1, 4, 2]), 2.5)     # even: mean of 2 and 3

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])
        self.assertEqual(mode([5, 6, 7]), [])           # every value appears once

    def test_empty_input_raises(self):
        for func in (mean, median, mode, variance, std_dev, quartiles):
            with self.subTest(func=func.__name__):
                with self.assertRaises(ValueError):
                    func([])
  


class TestSpread(unittest.TestCase):
    def test_sample_vs_population_variance(self):
        data = [2, 4, 6]  # mean 4; squared distances 4 + 0 + 4 = 8
        self.assertAlmostEqual(variance(data, sample=True), 4.0)       # 8 / (3 - 1)
        self.assertAlmostEqual(variance(data, sample=False), 8 / 3)    # 8 / 3

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([7], sample=False), 0.0)   # one value has no spread
        with self.assertRaises(ValueError):
            std_dev([7], sample=True)                       # n - 1 would be zero
  

class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        text = "agent,amount\nA,100\nA,\nA,n/a\nA,-\nA,abc\nA, 250.5 \nA,nan\nA,-5\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            path.write_text(text, encoding="utf-8")
            valid, rejected = load_column(path, "amount")
        self.assertEqual(valid, [100.0, 250.5])
        self.assertEqual(
            rejected,
            [(3, ""), (4, "n/a"), (5, "-"), (6, "abc"), (8, "nan"), (9, "-5")],
        )


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([5, 5, 5, 5, 5])
        self.assertEqual(result["outliers"], [])
        self.assertEqual(result["iqr"], 0)

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 100])
        self.assertEqual(result["outliers"], [100])


if __name__ == "__main__":
    unittest.main()
