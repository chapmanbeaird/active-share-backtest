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
   - **Windows:** on the very first screen of the installer, tick the box
     **"Add python.exe to PATH"** before clicking Install. This one checkbox is what
     lets the launcher find Python — don't skip it.
   - **Mac:** Python is often already installed. If the first run tells you it's missing,
     install it from the link above.
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

> **Windows vs. Mac:** the only difference is which launcher you double-click and, if you
> ever fall back to a terminal, one word in the command. This guide writes terminal
> commands the Mac way (`python3 …`); **on Windows type `python` instead of `python3`**.
> Everything else — the folders, the output, the prompts — is identical.

### Step 1 — Export the S&P 500 from FactSet
Pull the **"Active Share" report** (Vanguard 500 Index Fund vs. S&P 500) as an Excel
file. It should have one row per stock with a **ticker**, the stock's **weight**, and
its **GICS sector**. This is the same report format the sample file uses. Save it as
`.xlsx`.

### Step 2 — Drop the file in the inbox
Put that one Excel file into the **`data/incoming/`** folder. Make sure it's the
**only** file in there.

### Step 3 — Run it

**Easiest — double-click the launcher for your computer:**

| Your computer | Double-click this file in the project folder |
|---|---|
| **Windows** | **`Update Portfolio.bat`** |
| **Mac** | **`Update Portfolio.command`** |

A window opens, does the work, and shows the result. The **first** run takes an extra
minute while it quietly builds its Python toolbox (the *venv*); every run after that is
quick.

> **Windows — if you see a security warning the first time:** because the file came from
> another computer, Windows may show a **"Windows protected your PC"** box or an **"Open
> File – Security Warning"**. Click **More info → Run anyway** (or **Run**). This is
> expected for a script you were handed; you only confirm it once.

**Or**, from a terminal / command prompt opened in the project folder:

- **Windows:** `python update_portfolio.py`
- **Mac:** `python3 update_portfolio.py`

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
Portfolio saved: results/portfolios/portfolio_2026-06.xlsx
```
Open that `portfolio_2026-06.xlsx` file to see your 60 stocks.

### Step 5 — (Sometimes) classify a few new stocks
Occasionally the index adds a **brand-new company** the tool has never seen. The
**sector** always comes straight from the file, but the tool won't guess the
**industry group** or the **industry** — so it **asks you, right there in the window**,
one company at a time. First it asks for the **industry group**:
```
[1/7]  BNY — Bank of New York Mellon Corp
        Sector (already set): Financials
        Choose its industry group (one of these 20):
           1. Commercial Services
           2. Communications
           ...
           9. Finance
           ...
          20. Utilities
        Enter a number 1-20, the exact group name, or 'q' to quit: 9
```
then, once the group is set, for the **industry** — it lists the industries already
seen in that group so you can pick one by number, or type a new name:
```
        Now choose BNY's industry (shown for reference; not a constraint):
           1. Finance/Rental/Leasing
           2. Financial Conglomerates
           3. Insurance Brokers/Services
           4. Investment Banks/Brokers
           ...
        Enter a number 1-14, or type a new industry name (or 'q' to quit): 4
        -> BNY = Finance / Investment Banks/Brokers
```
Type the **number** of the best fit (or the exact/new name). The **group** must be one
of the 20 (anything else re-asks); the **industry** accepts any name but **can't be
left blank** — so your saved records always carry both. Type **`q`** to quit without
saving. Your answers are **saved automatically** to `data/manual_classifications.csv`,
so you're only ever asked about each company **once** — future runs remember it.

> Why it asks for the industry: your FactSet export doesn't contain it (the export's
> only classification columns are the GICS sector and a *GICS* industry group, which is
> a different system — see [Assumptions](#7-assumptions-the-tool-makes-please-read)).
> For every stock already in your history the industry is filled in automatically, so
> you're only ever prompted for a genuine first-time index entrant.

> Tip: pick the group that best matches the company's business — a bank → `Finance`, a
> software company → `Technology Services`, a chip/hardware maker → `Electronic
> Technology`. (Full list and guidance in [section 6](#6-the-manual-classification-file).)

---

## 5. Understanding the output

### Your portfolio: `results/portfolios/portfolio_<date>.xlsx`
This is the deliverable. One sheet, starting with a summary and then three stacked
tables:

**PORTFOLIO SUMMARY** — the headline numbers at the very top, so you don't have to
scan the tables to see how the portfolio came out:

| Metric | Example | Meaning |
|---|---|---|
| Holdings | 60 of 503 benchmark names | how many stocks you hold vs. how many are in the index |
| Portfolio Active Share | 33.40% | the number the tool minimized (see [key ideas](#3-the-key-ideas)) |
| Portfolio Weight Sum | 100.00% | a sanity check — the 60 weights should add to 100% |
| Max Sector Deviation | 2.00% (PASS vs 2.00% limit) | worst sector's gap from the index, and whether it's inside ±2% |
| Max Industry-Group Deviation | 2.00% (PASS vs 2.00% limit) | same, for industry groups |
| Largest Position | NVDA (7.61%) | your biggest single holding |
| Largest Overweight | LLY (+2.97%) | where you're most *above* the index |
| Largest Underweight | TMO (−0.29%) | where you're most *below* the index |

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

**File:** `data/manual_classifications.csv` — the tool's memory of the classifications
you've assigned.

**Why it exists:** the tool figures out each stock's industry group **and** industry by
looking up its **history** (the same company always keeps the same classification). A
company that's brand new to the index has no history — and the FactSet export doesn't
carry these either — so the tool asks you once (Step 5) and **records your answers
here** so it never has to ask again.

**You normally don't touch this file** — the interactive prompt writes to it for you.
But you can open it in Excel or any text editor to review or change a past answer.

**Format:** a header row followed by three columns per stock. The prompt always fills
in all three:
```
ticker,industry_group,industry
VRT,Producer Manufacturing,Electrical Products
```

> **Keep the first line — `ticker,industry_group,industry` — intact.** It's the header
> the tool reads the columns from. If you hand-edit the file and delete or blank that
> line, the tool will stop with a clear message asking you to restore it. (`industry`
> is display-only, so a row with a blank industry still loads, but the prompt never
> creates one.)

**The `industry_group` must be exactly one of these 20 names** (the same list the prompt
shows you):

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
4. **Industry group *and* industry are recovered from history by ticker — not from the
   export.** The FactSet "Active Share" report carries only a GICS sector and a *GICS*
   industry group, which is a different system from the 20 groups this project uses, and
   it has no `industry` column at all. So the tool looks each stock's group and industry
   up in the historical data instead. Genuinely new names (no history) are asked about
   interactively — both group and industry (Step 5) — and remembered in the manual file
   (section 6).
5. **Tickers are converted to house format:** the `-US` country suffix is removed and
   share-class dots are kept (`GOOGL-US` → `GOOGL`, `BRK.B-US` → `BRK.B`).
6. **The snapshot date is read from inside the file** (the `30-JUN-2026`-style cell) and
   becomes the label, like `2026-06`. If it can't be found, you can pass `--as-of 2026-06`.
7. **Your historical workbook is never modified.** `data/1999-2025-S&P500-cleaned.xlsx`
   stays frozen; new snapshots go into the separate `data/current-snapshots.xlsx`.
8. **The tool never guesses a classification.** For a new stock it asks you in the
   terminal (or, in a non-interactive/automated run, stops safely).

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
and compares to the official S&P 500 total-return numbers. Everything it writes lands
under `results/backtest/`:

- `results/backtest/csv/` — `annual_performance.csv` (year-by-year returns,
  active share, checks), `summary.csv` (overall statistics), `missing_tickers.csv`,
  and the tax/turnover CSVs.
- `results/backtest/excel/` — `backtest_results.xlsx`, `holdings.xlsx`,
  `stock_returns.xlsx`, `return_attribution.xlsx`, `turnover_tax.xlsx`.

**Note (future add-on):** the backtest currently works in **whole calendar years**. It
does **not** yet measure returns for the new 3–6-month snapshots going forward — that
needs a sub-annual S&P total-return benchmark and is not built yet. For now, each
quarterly run gives you the *portfolio*; return tracking of those snapshots can be
added later.

---

## 9. Troubleshooting

| What you see | What to do |
|---|---|
| **It asks "N new stock(s) need a classification"** | Normal — for each new company type the number of the best-fitting industry group, then pick or type its industry (or `q` to quit). Your answers are saved automatically (section 6). |
| **"manual_classifications.csv is missing its header row"** | You (or an edit) deleted the top `ticker,industry_group,industry` line. Put it back as the first line of the file and re-run (section 6). |
| **"no .xlsx file found in data/incoming"** | Put your FactSet export into the `data/incoming/` folder. |
| **"found N files … expected exactly one"** | Leave only the newest export in `data/incoming/`; remove the others. |
| **"missing expected column(s)"** | The file isn't the "Active Share" report, or the sheet name differs. Check the export, or pass `--sheet "<name>"`. |
| **"Unexpected sector name(s)"** | The export's sectors aren't the standard 11 GICS — likely the wrong report or a format change. |
| **"Found N constituents — far outside ~500"** | Wrong sheet or file. Check what you exported. |
| **The double-click window closed instantly (Mac)** | Open the Terminal app, drag the project folder in, and run `python3 update_portfolio.py` to read the message. |
| **The window closed too fast to read it (Windows)** | Open **Command Prompt**, type `cd ` (with a trailing space), drag the project folder onto the window, press Enter, then run `python update_portfolio.py`. |
| **"Python was not found" / the launcher can't find Python (Windows)** | Python isn't installed or wasn't added to PATH. Reinstall from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"** (section 2), then run the launcher again. |
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

**Where everything lives**
```
data/
  incoming/                     <- DROP your FactSet export here
  processed/                    <- raw exports after processing (archive)
  current-snapshots.xlsx        <- cleaned, house-format copy of each snapshot
  manual_classifications.csv    <- saved industry-group + industry answers for new stocks
  1999-2025-S&P500-cleaned.xlsx <- frozen historical benchmark (never changes)
  missing-tickers/              <- reference data for the backtest's ticker checks

results/
  portfolios/                   <- YOUR quarterly deliverable
    portfolio_<date>.xlsx
  backtest/                     <- the 2000-2025 study (advanced/optional)
    excel/                      <- backtest_results, holdings, stock_returns, ...
    csv/                        <- annual_performance, summary, tax/turnover, ...

docs/                           <- background notes & research (not needed to run)

Update Portfolio.bat            <- Windows: DOUBLE-CLICK to run
Update Portfolio.command        <- Mac: DOUBLE-CLICK to run
update_portfolio.py             <- the quarterly build (what the launcher runs)
run.py                          <- the optional historical backtest
src/                            <- the rest of the program code (incl. generate_portfolio.py)
venv/, cache/                   <- Python toolbox + data cache (don't edit)
```

**What you touch:** drop a file in `data/incoming/`, double-click the launcher for your
computer (**`Update Portfolio.bat`** on Windows, **`Update Portfolio.command`** on Mac),
open your portfolio in `results/portfolios/`. Everything else the tool manages for you.
Your quarterly portfolio files are generated fresh each run (they're not stored in git).

**Commands at a glance** (Mac shown; on **Windows** type `python` instead of `python3`)
```
python3 update_portfolio.py            # build the current portfolio from data/incoming/
python3 update_portfolio.py --file "path/to/export.xlsx"   # use a specific file
python3 run.py                         # optional: full 2000–2025 historical backtest
```
