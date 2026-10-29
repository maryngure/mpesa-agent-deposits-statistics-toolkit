"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
   
    valid = []
    rejected = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Column {column!r} not found in {path}")

        for row_number, row in enumerate(reader, start=2):
            raw = row[column]
            if raw is None:              # row was shorter than the header
                raw = ""

            try:
                number = float(raw.strip())
            except ValueError:           # empty string or text that isn't a number
                rejected.append((row_number, raw))
                continue

            if not math.isfinite(number) or number < 0:
                rejected.append((row_number, raw))
                continue

            valid.append(number)

    return valid, rejected
    


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean() requires at least one value")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median() requires at least one value")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    if not values:
        raise ValueError("mode() requires at least one value")
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    highest = max(counts.values())
    if highest == 1:
        return []
    return sorted(v for v, c in counts.items() if c == highest)


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    if not values:
        raise ValueError("variance() requires at least one value")
    if sample and len(values) < 2:
        raise ValueError("sample variance requires at least two values")
    m = mean(values)
    squared_diffs = [(v - m) ** 2 for v in values]
    divisor = len(values) - 1 if sample else len(values)
    return sum(squared_diffs) / divisor


def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample))


def _percentile_sorted(ordered: list[float], p: float) -> float:
    """Percentile p (0 to 1) of an already-sorted list, using linear interpolation."""
    position = (len(ordered) - 1) * p
    lower = int(position)
    fraction = position - lower
    if lower + 1 < len(ordered):
        return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])
    return ordered[lower]


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """ Return (Q1, Q2, Q3). Method: linear interpolation (Hyndman-Fan type 7), the same as NumPy's default."""
    if not values:
        raise ValueError("quartiles() requires at least one value")
    ordered = sorted(values)
    return (
        _percentile_sorted(ordered, 0.25),
        _percentile_sorted(ordered, 0.50),
        _percentile_sorted(ordered, 0.75),
    )


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    q1, _, q3 = quartiles(values)
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    outliers = [v for v in values if v < lower_fence or v > upper_fence]
    return {
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers": sorted(outliers),
    }


def _reject_reason(raw: str) -> str:
    """Explain why a raw cell was rejected."""
    text = raw.strip()
    if text == "":
        return "missing value"
    try:
        number = float(text)
    except ValueError:
        return "not a number"
    if not math.isfinite(number):
        return "not a finite number"
    return "negative value"


def _read_labels(path: Path, column: str) -> dict[float, tuple[str, str]]:
    """Map each valid value to its (date, agent_id) so outliers can be traced back."""
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                number = float((row[column] or "").strip())
            except ValueError:
                continue
            if math.isfinite(number) and number >= 0:
                labels[number] = (row.get("date", "?"), row.get("agent_id", "?"))
    return labels



class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values") -> None:
        if not values:
            raise ValueError("StatisticsAnalyzer requires at least one value")
        self.name = name
        self.values = list(values)
        self._sorted = sorted(values)
        self.rejected = []
        self.labels = {}

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        values, rejected = load_column(path, column)
        analyzer = cls(values, column)
        analyzer.rejected = rejected
        analyzer.labels = _read_labels(path, column)
        return analyzer

    def summary(self) -> dict:
        n = len(self.values)
        q1 = _percentile_sorted(self._sorted, 0.25)
        median_value = _percentile_sorted(self._sorted, 0.50)
        q3 = _percentile_sorted(self._sorted, 0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        return {
            "count": n,
            "mean": mean(self.values),
            "median": median_value,
            "mode": mode(self.values),
            "variance": variance(self.values) if n >= 2 else None,
            "std_dev": std_dev(self.values) if n >= 2 else None,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outliers": [v for v in self._sorted if v < lower_fence or v > upper_fence],
        }
    

    def without_outliers(self) -> "StatisticsAnalyzer":
        info = self.summary()
        kept = [v for v in self.values if info["lower_fence"] <= v <= info["upper_fence"]]
        cleaned = StatisticsAnalyzer(kept, self.name + " (outliers removed)")
        cleaned.rejected = self.rejected
        cleaned.labels = self.labels
        return cleaned

    def report(self) -> str:
        n = len(self.values)
        lines = [
            f"=== Statistics report: {self.name} ===",
            f"Rows read         : {n + len(self.rejected)}",
            f"Valid values      : {n}",
            f"Rejected values   : {len(self.rejected)}",
        ]
        for row_number, raw in self.rejected:
            lines.append(f"  row {row_number}: {raw!r} ({_reject_reason(raw)})")
        cleaned_analyzer = self.without_outliers()
        a = self.summary()
        b = cleaned_analyzer.summary()

        def money(x):
            return "n/a" if x is None else f"{x:,.2f}"

        def mode_text(m):
            return "none" if not m else ", ".join(f"{v:,.2f}" for v in m)

        def pair(x, y):
            return f"{x:,.2f} / {y:,.2f}"

        rows = [
            ("count", str(a["count"]), str(b["count"])),
            ("mean", money(a["mean"]), money(b["mean"])),
            ("median", money(a["median"]), money(b["median"])),
            ("mode", mode_text(a["mode"]), mode_text(b["mode"])),
            ("variance (sample)", money(a["variance"]), money(b["variance"])),
            ("std dev (sample)", money(a["std_dev"]), money(b["std_dev"])),
            ("min / max", pair(self._sorted[0], self._sorted[-1]),
             pair(cleaned_analyzer._sorted[0], cleaned_analyzer._sorted[-1])),
            ("Q1 / Q3", pair(a["q1"], a["q3"]), pair(b["q1"], b["q3"])),
            ("IQR", money(a["iqr"]), money(b["iqr"])),
        ]

        lines.append("")
        lines.append(f"{'':<20}{'all data':<22}outliers removed")
        for label, left, right in rows:
            lines.append(f"{label:<20}{left:<22}{right}")
        lines.append("")
        lines.append(f"Outlier fences      : lower {a['lower_fence']:,.2f}, upper {a['upper_fence']:,.2f}")
        found = []
        for v in a["outliers"]:
            date, agent = self.labels.get(v, ("unknown date", "unknown agent"))
            found.append(f"{v:,.2f} ({date}, {agent})")
        lines.append("Outliers            : " + (", ".join(found) if found else "none"))
        return "\n".join(lines)


def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}.

    Matching uses a tolerance, not ==. NumPy is told to use the same choices as our code:
    ddof=1 (sample variance / std dev) and linear-interpolation percentiles (type 7).
    """
    import numpy as np

    s = analyzer.summary()
    data = np.array(analyzer.values, dtype=float)

    def close(mine, theirs):
        return math.isclose(mine, float(theirs), rel_tol=1e-9, abs_tol=1e-9)

    q1, q3 = np.percentile(data, [25, 75])  # default method is linear (type 7)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    numpy_outliers = sorted(float(v) for v in data if v < lower or v > upper)

    results = {
        "mean": close(s["mean"], np.mean(data)),
        "median": close(s["median"], np.median(data)),
        "q1": close(s["q1"], q1),
        "q3": close(s["q3"], q3),
        "iqr": close(s["iqr"], iqr),
        "lower_fence": close(s["lower_fence"], lower),
        "upper_fence": close(s["upper_fence"], upper),
        "outliers": numpy_outliers == s["outliers"],
    }
    if s["variance"] is not None:
        results["variance"] = close(s["variance"], np.var(data, ddof=1))
        results["std_dev"] = close(s["std_dev"], np.std(data, ddof=1))
    return results


RISK_NOTE = (
    "=== Note for the risk team ===\n"
    "Three of the four flagged days (96,500 on 20 Sep, 1,200 on 2 Oct and 88,400 on 6 Oct) belong to "
    "agent AG-104 in Nakuru, which also has a missing entry and an unreadable one, so that agent's "
    "records should be reviewed first.\n"
    "The two very high days were not matched by high transaction counts (86 and 61), so they could be "
    "an extra digit typed in error, an unusual float top-up or something more serious that needs checking "
    "against the M-Pesa statements.\n"
    "The 1,200 day had 103 transactions, which does not fit such a low total and points to a data-entry "
    "error, while the borderline 29,060.99 day from AG-101 looks like a genuinely busy day rather than a problem."
)


def numpy_section(results: dict[str, bool]) -> str:
    def status(*names):
        return "MATCH" if all(results.get(n, False) for n in names) else "MISMATCH"

    return "\n".join([
        "=== NumPy verification ===",
        f"mean      {status('mean')}",
        f"median    {status('median')}",
        f"variance  {status('variance')}  (ddof=1)",
        f"std dev   {status('std_dev')}",
        f"Q1 / Q3   {status('q1', 'q3')}  (method=linear)",
    ])  


def unit_test_section() -> str:
    """Run test_stats_toolkit.py and summarise the run in the expected format."""
    import io
    import time
    import unittest

    start_dir = str(Path(__file__).resolve().parent)
    suite = unittest.defaultTestLoader.discover(start_dir, pattern="test_stats_toolkit.py")
    began = time.perf_counter()
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    elapsed = time.perf_counter() - began
    verdict = "OK" if result.wasSuccessful() else "FAILED"
    return "\n".join(["=== Unit tests ===", f"Ran {result.testsRun} tests in {elapsed:.3f}s", verdict])


if __name__ == "__main__":
    from_file = StatisticsAnalyzer.from_csv(DATA_PATH, "daily_deposits_kes")
    full_report = "\n\n".join([
        from_file.report(),
        numpy_section(compare_with_numpy(from_file)),
        unit_test_section(),
        RISK_NOTE,
    ])
    print(full_report)

    output_path = Path("output") / "stats_report.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(full_report + "\n", encoding="utf-8")
    print(f"\nSaved to {output_path}")

    




    
