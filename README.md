# M-Pesa Agent Deposits: Statistical Analysis Toolkit

A dependency-light statistics toolkit that profiles daily deposit totals for a network of M-Pesa agents, cleans unreliable entries without guessing, and flags unusual days for the risk team. Every statistic is implemented from first principles in plain Python and then independently verified against NumPy.

**Context.** Wakala Plus operates agents across Nairobi, Thika, Machakos and Nakuru. Its risk team watches daily deposit totals per agent, where unusually high or low days can signal fraud, a float shortage or a data-entry error. The data is typed into a phone app, so it contains gaps, placeholders and typos alongside genuinely extreme days.

## Key findings

The dataset has 40 daily records from four agents (1 Sep to 10 Oct 2026). 35 are usable and 5 were rejected.

| Statistic | All data (n = 35) | Outliers removed (n = 31) |
|---|---|---|
| Mean | 22,400.00 | 18,349.65 |
| Median | 18,805.47 | 18,506.72 |
| Standard deviation (sample) | 18,111.32 | 3,180.63 |
| Min / Max | 1,200.00 / 96,500.00 | 12,772.70 / 26,725.69 |
| Q1 / Q3 | 15,998.40 / 20,977.21 | 15,998.40 / 20,730.60 |

All amounts are in KES.

- **Four outliers were flagged** using Tukey's IQR rule (fences at 8,530.19 and 28,445.41).
- **They distort the mean far more than the median.** Removing them moves the mean by 18% and the median by under 2%, and it cuts the standard deviation by 82%. The median is the more dependable measure of a typical day here.
- **Three of the four outliers belong to one agent.** AG-104 (Nakuru) accounts for the 96,500 and 88,400 spikes and the 1,200 dip. The same agent also has a missing entry and an unreadable one, so its records are the first to review.

| Value (KES) | Date | Agent | Transactions | Note |
|---|---|---|---|---|
| 96,500.00 | 2026-09-20 | AG-104 (Nakuru) | 86 | About five times the median day |
| 88,400.00 | 2026-10-06 | AG-104 (Nakuru) | 61 | About 4.7 times the median day |
| 1,200.00 | 2026-10-02 | AG-104 (Nakuru) | 103 | Total does not fit the transaction count |
| 29,060.99 | 2026-10-07 | AG-101 (Nairobi) | 112 | Borderline; likely a genuinely busy day |

## Data quality

Entries that cannot be trusted are rejected and reported with a reason, never repaired or guessed.

| Row | Raw value | Agent / date | Reason |
|---|---|---|---|
| 7 | *(empty)* | AG-102, 2026-09-06 | Missing value |
| 10 | `-` | AG-101, 2026-09-09 | Not a number |
| 14 | `n/a` | AG-101, 2026-09-13 | Not a number |
| 25 | `12,40O` | AG-104, 2026-09-24 | Not a number (letter O typed for zero) |
| 29 | *(empty)* | AG-104, 2026-09-28 | Missing value |

Row numbers count the header as row 1, so they match what a spreadsheet shows. `12,40O` looks like 12,400, but the intent cannot be confirmed, so it is rejected. The validation rules also reject `nan`, `inf` and negative amounts; none of those occurred in this dataset.

## Design decisions

- **Sample statistics by default.** Variance and standard deviation divide by n - 1. The 35 valid days are a slice of ongoing agent activity rather than every day that could occur, and five further days were lost to bad entries. The population versions are available with `sample=False`; with 35 values the two standard deviations differ by about 1.5% (18,111.32 vs 17,850.71).
- **Quartile method.** Linear interpolation (Hyndman-Fan type 7), the same as NumPy's default, so the verification compares like with like.
- **Outlier rule.** A value is an outlier if it lies more than 1.5 x IQR beyond Q1 or Q3. Quartiles are used because, unlike the mean and standard deviation, they are not distorted by the extreme days being searched for.
- **Mode.** Returned as a list so ties can be reported. If every value appears exactly once, the result is an empty list. That is the case here: amounts carry cents, so no value repeats, and the mode is not informative for this kind of data.
- **Efficiency.** `StatisticsAnalyzer` sorts its data once on creation and reuses it, rather than re-sorting on every call.
- **Rounding.** Full precision is kept in all calculations. Values are rounded to two decimal places only for display, so outlier decisions and the NumPy comparison are never affected by rounding.

## Verification

`compare_with_numpy()` is the only place NumPy is imported. It recomputes each statistic and reports MATCH or MISMATCH using a tolerance (`math.isclose`, relative 1e-9) rather than `==`, which avoids false mismatches from floating-point rounding. NumPy is run with `ddof=1` and linear-interpolation percentiles to match the toolkit's choices. Using NumPy's default `ddof=0` would produce population figures and an expected MISMATCH on variance and standard deviation. In this dataset, all statistics match.

## Testing

Nine unit tests (standard-library `unittest`) use small hand-checkable inputs rather than the dataset. They cover empty input, odd and even lengths, ties in the mode, sample vs population variance, a single-value standard deviation, outlier detection with uniform and skewed data, and rejection of missing and invalid values. The test run is also reported at the end of `output/stats_report.txt`.

## Project structure

```
day3-stats-toolkit/
├── stats_toolkit.py          # functions and the StatisticsAnalyzer class
├── test_stats_toolkit.py     # unit tests
├── data/agent_deposits.csv   # input data
├── output/stats_report.txt   # generated report
├── requirements.txt          # numpy (verification only), pytest (optional)
└── README.md
```

## How to run

Run these from the project folder so the data file is found:

```
pip install -r requirements.txt
python stats_toolkit.py        # prints the report and saves output/stats_report.txt
python -m unittest -v          # runs the unit tests
```

The `output` folder is created automatically if it does not exist.

## Limitations and possible extensions

- The IQR rule flags unusual values but cannot distinguish a genuine busy day from an error. The 29,060.99 day sits only just beyond the fence, which is why human review of the flagged days matters.
- Outliers are traced back to a date and agent by matching amounts. This is reliable here because no amount repeats, but a repeated amount would need a row-based lookup.
- The analysis treats all agents as one population. A natural extension would compare agents individually, since agents in different towns may have different normal ranges.
- The `k` multiplier in the IQR rule is fixed at 1.5 in the analyzer; making it configurable would allow stricter or looser screening.