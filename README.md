# Active Share Minimizer — S&P 500 Portfolio Builder

This project builds a **concentrated 60-stock portfolio** out of the S&P 500 that
stays as close to the index as possible — it has the mathematically **lowest
possible active share** while still matching the index's sector and
industry-group weights within tight limits.

You feed it a FactSet export of the S&P 500 (with each stock's index weight), and
it hands you back a spreadsheet of 60 stocks and how much to put in each one.

This README is written for someone who understands **portfolios and indexes** but
has **never touched code**. You do not need to understand any of the Python to run
this. Just follow the steps.

---

## Table of contents
1. [What this tool does (in plain English)](#1-what-this-tool-does-in-plain-english)
2. [One-time setup](#2-one-time-setup)
3. [The key ideas](#3-the-key-ideas)
4. [The quarterly workflow — step by step](#4-the-quarterly-workflow--step-by-step)
5. [Understanding the output](#5-understanding-the-output)
6. [The manual classification file](#6-the-manual-classification-file)
7. [Assumptions the tool makes (please read)](#7-assumptions-the-tool-makes-please-read)
8. [The historical backtest (optional / advanced)](#8-the-historical-backtest-optional--advanced)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary & file map](#10-glossary--file-map)

---

## 1. What this tool does (in plain English)

An index fund that holds all ~500 stocks tracks the S&P 500 almost perfectly, but
it holds a lot of names. Sometimes you want **far fewer holdings** (say 60) while
still behaving as much like the index as possible.

"Behaving like the index" is measured by **active share** — how different your
holdings are from the index (see [The key ideas](#3-the-key-ideas)). This tool
finds the 60 stocks and their weights that make active share **as low as it can
possibly be**, subject to a handful of rules — most importantly, your portfolio's
**sector** mix and **industry-group** mix must each stay within **2 percentage
points** of the index (the full list is in [The key ideas](#3-the-key-ideas)).

You run this **every 3–6 months** with a fresh S&P 500 snapshot. Each run produces
an updated 60-stock portfolio.

---

## 2. One-time setup

You only do this **once** on a given computer.

1. **Install Python** (version 3.10 or newer) from <https://www.python.org/downloads/>.
   On a Mac it may already be installed.
2. **Get the project folder** onto your computer (the folder containing this README).
3. **Add the data key.** There is a file named `.env` in the project folder that holds
   the FactSet/FMP data key. It is needed only for the optional historical backtest,
   not for the normal quarterly run. If it is missing, ask whoever set this up.
4. **First run creates the environment automatically.** The launcher builds a small,
   self-contained Python environment (a "virtual environment", or *venv*) the first
   time you use it and installs what it needs. You don't have to do anything — just
   run it once and wait a minute.

> **What is a "venv"?** Think of it as a private toolbox for this project so it
> doesn't interfere with anything else on your computer. It lives in the `venv`
> folder. You never edit it.

---

## 3. The key ideas

### Active share
Active share is **how different your portfolio is from the index**, on a 0–100% scale.

- **0%** = you hold exactly the index (an index fund).
- **100%** = you hold nothing the index holds.

It is calculated as **half the sum of the differences** between each stock's weight
in your portfolio and its weight in the index. A 60-stock portfolio can't be 0%
(it's missing ~440 names), so this tool finds the **lowest achievable** number.
For the S&P 500 today that's typically in the **30s–40s percent**.

### The rules the optimizer follows
"Lowest active share" on its own could still end up lopsided (too heavy in tech) or
silly (one giant position). So the optimizer minimizes active share subject to **five
rules**:

1. **Exactly 60 holdings.**
2. **Fully invested** — the 60 weights add up to 100%.
3. **Per-stock weight cap** — no single position can run away (the three-part cap
   described just below).
4. **Sector band** — your weight in each of the **11 GICS sectors** stays within
   **±2 percentage points** of the index's weight in that sector.
5. **Industry-group band** — your weight in each of the **20 industry groups** stays
   within **±2 percentage points** of the index.

Rules 4 and 5 keep the portfolio broadly shaped like the index; rule 3 keeps
individual positions sane. (Under the hood there are also two purely mechanical
constraints that let the solver choose *which* 60 stocks to hold and compute active
share — those aren't portfolio choices, just math plumbing.)

### Sector vs. industry group vs. industry
The data has three levels of classification. Two of them are enforced; one is just
for your reading:

| Level | How many | Enforced by the ±2% rule? |
|---|---|---|
| **Sector** (e.g. *Information Technology*) | 11 | **Yes** |
| **Industry group** (e.g. *Electronic Technology*) | 20 | **Yes** |
| **Industry** (e.g. *Semiconductors*) | ~120 | No — shown for reference only |

> **Important:** the "industry group" here is **not** the GICS industry group. It's a
> separate FactSet classification (the 20 "Economic Sectors" like *Finance*,
> *Electronic Technology*, *Retail Trade*). This matters when you export from FactSet
> — see [Assumptions](#7-assumptions-the-tool-makes-please-read).

### The weight cap
The math that minimizes active share doesn't care *how* extra weight is spread among
your 60 stocks. Without a limit, it might pile a huge weight onto one stock. To keep
positions sensible, each stock is capped at:

> **cap = the smaller of (3 × its index weight) or (the larger of 5% and its index weight)**

In practice: small index names can go up to 3× their index weight, mid names are
capped at 5%, and the few giant names (≥5% of the index) are held at exactly their
index weight. This barely changes active share but prevents silly concentration.

---

## 4. The quarterly workflow — step by step

Every 3–6 months:

### Step 1 — Export the S&P 500 from FactSet
Pull the **"Active Share" report** (Vanguard 500 Index Fund vs. S&P 500) as an Excel
file. It should have one row per stock with a **ticker**, the stock's **weight**, and
its **GICS sector**. This is the same report format the sample file uses. Save it as
`.xlsx`.

### Step 2 — Drop the file in the inbox
Put that one Excel file into the **`data/incoming/`** folder. Make sure it's the
**only** file in there.

### Step 3 — Run it
**Easiest:** double-click **`Update Portfolio.command`** in the project folder. A
window opens, does the work, and shows the result.

**Or**, from a terminal in the project folder:
```
python3 update_portfolio.py
```

### Step 4 — Read the result
You'll see a plain-English summary, for example:
```
Found 503 S&P 500 stocks as of 2026-06.
Classified: 496 matched from history, 7 from your manual list, 0 from the file itself.
...
DONE — portfolio for 2026-06
Holdings:        60 stocks
Active share:    33.40%
Sector check:    PASS (max deviation 2.00% vs 2.00% limit)
Ind-group check: PASS (max deviation 2.00% vs 2.00% limit)
Portfolio saved: results-excel/portfolio_2026-06.xlsx
```
Open that `portfolio_2026-06.xlsx` file to see your 60 stocks.

### Step 5 — (Sometimes) classify a few new stocks
Occasionally the index adds a **brand-new company** the tool has never seen. When
that happens it **stops safely** and lists them, for example:
```
STOPPED — 7 stock(s) are new and need a category
Ticker   Company
BNY      Bank of New York Mellon Corp
CASY     Casey's General Stores, Inc.
...
```
This is normal — it never guesses. Open **`data/manual_classifications.csv`**, add one
line per listed ticker (see [section 6](#6-the-manual-classification-file)), save,
and run again. You only ever classify each company once.

---

## 5. Understanding the output

### Your portfolio: `results-excel/portfolio_<date>.xlsx`
This is the deliverable. One sheet with three stacked tables:

**SECTOR BREAKDOWN** and **INDUSTRY GROUP BREAKDOWN** — how your portfolio compares to
the index, bucket by bucket:

| Column | Meaning |
|---|---|
| Sector / Industry_Group | the bucket name |
| # Held | how many of your 60 stocks fall in this bucket |
| # in Bench | how many index names are in this bucket |
| Port. Weight | your % weight in this bucket |
| Bench. Weight | the index's % weight in this bucket |
| Difference | Port − Bench (this is what must stay within ±2%) |

**STOCK DETAIL** — every stock, your 60 holdings listed first (rest of the index below
with 0 weight):

| Column | Meaning |
|---|---|
| Name / Ticker | the company |
| Port. Weight | how much you hold (0 if not held) |
| Bench. Weight | the index weight |
| Difference | Port − Bench |
| AS Contribution | how much this stock adds to active share (half the absolute difference) |
| Industry_Group / Industry / Sector | its classifications |

**PASS / FAIL and "max deviation":** PASS means every bucket's `Difference` is within
the ±2% guardrail. "Max deviation" is the single worst bucket — if that's ≤ 2%, all of
them are.

### The cleaned snapshots: `data/current-snapshots.xlsx`
Every run also saves the **cleaned, house-format version of the index** it used, as a
dated sheet (e.g. `2026-06`). Your original FactSet export is never modified — this is
a tidy, converted copy. Over time this file becomes your archive of index snapshots.

### The processed inbox: `data/processed/`
After a successful run, the tool moves the FactSet file you dropped in `data/incoming/`
into `data/processed/` so the inbox is clean for next time.

---

## 6. The manual classification file

**File:** `data/manual_classifications.csv` — open it in Excel or any text editor.

**Why it exists:** the tool figures out each stock's industry group by looking up its
**history** (the same company always keeps the same group). A company that's brand new
to the index has no history, so you tell it the group **once**, here.

**Format:** three columns. `industry` is optional (it's display-only).
```
ticker,industry_group,industry
VRT,Producer Manufacturing,Electrical Products
```

**The `industry_group` must be exactly one of these 20 names:**

```
Commercial Services      Finance                  Process Industries
Communications           Health Services          Producer Manufacturing
Consumer Durables        Health Technology        Retail Trade
Consumer Non-Durables    Industrial Services      Technology Services
Consumer Services        Miscellaneous            Transportation
Distribution Services    Non-Energy Minerals      Utilities
Electronic Technology    Energy Minerals
```

If you type a group that isn't on this list, the tool ignores that line and will stop
again asking for a valid one. Pick the closest fit to the company's business (e.g. a
software company → `Technology Services`; a bank → `Finance`; a chip/hardware maker →
`Electronic Technology`).

---

## 7. Assumptions the tool makes (please read)

These are the rules baked in. If a future FactSet export changes shape, these are the
things to check.

1. **The weight used is the fund's "Port. Ending Weight" column**, rescaled so all
   holdings sum to exactly 100%. (This matches how the historical data was built.)
2. **A "real" stock is any row that has a GICS sector.** Section headers, subtotals,
   cash lines (`[Cash]`), and the export footer have no sector and are dropped
   automatically.
3. **Sector names must be the 11 standard GICS sectors.** If the export has an
   unexpected sector name, the tool stops (it means the wrong file or a changed format).
4. **Industry group is recovered from history by ticker — not from the export.** The
   FactSet "Active Share" report gives *GICS* industry groups, which are a different
   system from the 20 groups this project uses. So the tool looks each stock up in the
   historical data instead. New names go through the manual file (section 6).
5. **Tickers are converted to house format:** the `-US` country suffix is removed and
   share-class dots are kept (`GOOGL-US` → `GOOGL`, `BRK.B-US` → `BRK.B`).
6. **The snapshot date is read from inside the file** (the `30-JUN-2026`-style cell) and
   becomes the label, like `2026-06`. If it can't be found, you can pass `--as-of 2026-06`.
7. **Your historical workbook is never modified.** `data/1999-2025-S&P500-cleaned.xlsx`
   stays frozen; new snapshots go into the separate `data/current-snapshots.xlsx`.
8. **The tool never guesses a classification.** Anything it can't place stops the run.

---

## 8. The historical backtest (optional / advanced)

Separately from the quarterly portfolio, this project can **backtest** the strategy
over **2000–2025** — building the portfolio from each year-end S&P 500 snapshot and
measuring how it would have performed the following year versus the index.

Run it with:
```
python3 run.py                    # full backtest, holding years 2000–2025
python3 run.py --start-year 2024 --end-year 2025    # quick 2-year test
```

It pulls each stock's actual annual return from the FMP data service (this needs the
`.env` key and an internet connection; the first run is slow, later runs use a cache)
and compares to the official S&P 500 total-return numbers. It writes:

- `results/annual_performance_milp.csv` — year-by-year returns, active share, checks.
- `results/summary_milp.csv` — overall statistics.
- `results-excel/backtest_results.xlsx`, `portfolio_holdings.xlsx`,
  `return_attribution.xlsx`, `Portfolio_individual_stock_returns.xlsx`.

**Note (future add-on):** the backtest currently works in **whole calendar years**. It
does **not** yet measure returns for the new 3–6-month snapshots going forward — that
needs a sub-annual S&P total-return benchmark and is not built yet. For now, each
quarterly run gives you the *portfolio*; return tracking of those snapshots can be
added later.

---

## 9. Troubleshooting

| What you see | What to do |
|---|---|
| **"STOPPED — N stock(s) are new"** | Normal. Add the listed tickers to `data/manual_classifications.csv` (section 6) and run again. |
| **"no .xlsx file found in data/incoming"** | Put your FactSet export into the `data/incoming/` folder. |
| **"found N files … expected exactly one"** | Leave only the newest export in `data/incoming/`; remove the others. |
| **"missing expected column(s)"** | The file isn't the "Active Share" report, or the sheet name differs. Check the export, or pass `--sheet "<name>"`. |
| **"Unexpected sector name(s)"** | The export's sectors aren't the standard 11 GICS — likely the wrong report or a format change. |
| **"Found N constituents — far outside ~500"** | Wrong sheet or file. Check what you exported. |
| **The double-click window closed instantly** | Open the Terminal app, drag the project folder in, and run `python3 update_portfolio.py` to see the message. |
| **Couldn't find the snapshot date** | Add `--as-of 2026-06` (use your snapshot's year-month). |

To re-run any time, it's safe: re-running overwrites that date's snapshot and portfolio
rather than duplicating them.

---

## 10. Glossary & file map

**Glossary**
- **Active share** — how different your holdings are from the index (0% = identical).
- **Benchmark** — the S&P 500 index you're compared to.
- **Constituent** — a stock in the index.
- **Industry group** — the 20-bucket FactSet "Economic Sector" classification (enforced).
- **MILP** — the optimization method that finds the provably best answer.
- **Snapshot** — one point-in-time picture of the index (one FactSet export).

**What you touch**
| Path | What it is |
|---|---|
| `data/incoming/` | **Drop your FactSet export here.** |
| `Update Portfolio.command` | **Double-click to run.** |
| `data/manual_classifications.csv` | Where you classify brand-new stocks. |
| `results-excel/portfolio_<date>.xlsx` | **Your 60-stock portfolio (the output).** |

**What the tool manages for you (don't edit)**
| Path | What it is |
|---|---|
| `data/current-snapshots.xlsx` | Cleaned, converted copies of each snapshot. |
| `data/processed/` | Archive of FactSet files already processed. |
| `data/1999-2025-S&P500-cleaned.xlsx` | Frozen historical data (never changes). |
| `src/` | The program code. |
| `venv/`, `cache/` | The Python toolbox and the data cache. |

**Commands at a glance**
```
python3 update_portfolio.py            # build the current portfolio from data/incoming/
python3 update_portfolio.py --file "path/to/export.xlsx"   # use a specific file
python3 run.py                         # optional: full 2000–2025 historical backtest
```
