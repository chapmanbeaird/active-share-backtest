# Active Share Minimizer Backtest

## Project Goal
Build a concentrated 60-stock portfolio from the S&P 500 that achieves the provably minimum active share, subject to Sector (±2%) and Industry_Group (±2%) constraints. Backtest across 1999–2024 snapshots (returns 2000–2025) using FMP API for price/dividend data.

## Architecture
- `src/milp_optimizer.py` — `MILPActiveShareOptimizer` class: formulates and solves a Mixed-Integer Linear Program via `scipy.optimize.milp` (HiGHS solver). No extra dependencies.
- `src/data_loader.py` — `load_benchmark_from_xlsx()`: reads cleaned FactSet xlsx, filters artifacts, normalizes weights
- `src/backtest_engine.py` — Runs MILP optimizer across years, fetches returns via FMP, prints/saves results
- `src/fmp_client.py` — FMP API wrapper with disk caching
- `run.py` — Entry point (thin wrapper around backtest_engine)
- `data/1999-2025-S&P500-cleaned.xlsx` — 27 sheets (1999–2025), year-end Vanguard 500 Fund holdings used as S&P 500 benchmark weights

## MILP Formulation
For N benchmark stocks (~503), the optimizer uses 3N variables:
- **x_i ∈ {0,1}** — binary: is stock i selected?
- **w_i ≥ 0** — continuous: portfolio weight (%)
- **d_i ≥ 0** — continuous: |w_i - b_i| (absolute deviation from benchmark)

**Objective**: minimize 0.5 × Σd_i (= active share)

**Constraints**:
- Σx_i = 60 (exactly 60 stocks)
- Σw_i = 100 (weights sum to 100%)
- w_i ≤ 100·x_i (Big-M: unselected stocks get 0 weight)
- w_i ≤ min(3·b_i, max(5, b_i)) (proportional weight cap — see below)
- d_i ≥ w_i - b_i, d_i ≥ b_i - w_i (absolute value linearization)
- |Σw_i(sector s) - B_s| ≤ 2% for each of 11 GICS Sectors
- |Σw_i(group g) - B_g| ≤ 2% for each of ~20 Industry_Groups

~1,540 constraints, 1,509 variables. Solves in <1 second per year.

**Key insight**: Industry_Groups do NOT nest within Sectors (15/20 span multiple GICS Sectors), so both constraint sets are genuinely independent. Simple per-group reweighting cannot guarantee compliance on both dimensions — a proper MILP is required.

## Proportional Weight Cap
The MILP objective (minimize active share) is indifferent to how excess weight is distributed among held stocks — active share is the same whether 4.7% of excess goes to one stock or is spread across ten. Without a cap, the solver arbitrarily concentrates weight (e.g., Morgan Stanley at 16x its benchmark weight).

**Formula**: `cap_i = min(3 × b_i, max(5%, b_i))`

Three regimes:
- **b_i < 1.667%**: cap at 3× benchmark weight (e.g., 0.31% → 0.93% max)
- **1.667% ≤ b_i < 5%**: cap at 5% ceiling (e.g., 2.0% → 5.0% max)
- **b_i ≥ 5%**: cap at benchmark weight (held at exactly benchmark, 0% AS contribution)

**Impact**: Active share increases by only +0.11pp on average (negligible — cap constrains weight distribution, not stock selection). Max position ratio capped at 3.0x across all years. Backtest returns decrease (annualized +10.0% → +8.8%) because the old uncapped portfolio had inflated alpha from arbitrary weight concentration that happened to backtest well.

## Observed Active Share (MILP, 60 stocks, 3x cap)
- 2000: 38.8%
- 2005: 47.2%
- 2010: 46.4%
- 2015: 48.0%
- 2020: 41.8%
- 2024: 36.5% (extreme mega-cap concentration)

## Holding Period
Snapshots are year-end December. Returns are measured Jan 1 → Dec 31 of the following year.

## FMP API Notes
- API key is in .env file
- `ticker_to_fmp()` in backtest_engine.py handles two ticker formats:
  - **Share classes** (dot → hyphen): BRK.B → BRK-B, BF.B → BF-B, LEN.B → LEN-B, etc.
  - **Disambiguation suffixes** (strip): DD.2 → DD, T.2 → T, CB.1 → CB, etc.
- `src/fmp_client.py` caches 402 failures in `_failed_symbols` set to avoid repeated API calls
- FMP lacks data for many pre-2010 acquired/delisted companies — coverage is 74–90% for early years, 95–100% from ~2003 onward
- Uses disk caching by default (`cache/fmp/`)

## Running
```bash
# Full backtest (1999-2024 snapshots, 2000-2025 returns)
python3 run.py

# Quick test (2 years)
python3 run.py --start-year 2023 --end-year 2025

# Test optimizer on single year
python3 src/milp_optimizer.py

# Skip saving results
python3 run.py --no-save

# Force fresh FMP data (no cache)
python3 run.py --no-cache
```

## Output Files
- `results/annual_performance_milp.csv` — year-by-year returns, active share, constraint checks
- `results/summary_milp.csv` — aggregate stats
- `results/missing_tickers.csv` — tickers FMP couldn't find
- `results-excel/backtest_results.xlsx` — all of the above in one Excel workbook (3 sheets)

## FactSet Excel Data File
`data/1999-2025-S&P500-cleaned.xlsx` — 27 sheets (1999–2025), one per year. Each sheet contains Vanguard 500 Index Fund holdings (used as S&P 500 benchmark proxy).

### Column Layout (header at row 7)
Name | Port. Weight | Bench. Weight | Difference | Industry_Group | Industry | Sector | Symbol | ISIN | SEDOL | CUSIP | Ticker

### Data Cleaning Performed
All cleaning was done in a single session. The data is now production-ready.

#### 1. Renamed Columns
- `Sector_Test` → `Industry_Group` (20 unique values — FactSet economic sectors)
- `Industry_Test` → `Industry` (99–115 unique values per year, 124 total across all years)

#### 2. Filled Missing Sectors (3,602 rows across all years)
~27% of rows had Sector = `--` (benchmark-only holdings FactSet didn't map to GICS). Filled using a deterministic (Industry_Group, Industry) → Sector mapping:
- **Tier 1 (84 combos, 2,401 rows):** Clean 1:1 mapping from existing data
- **Tier 2 (20 combos, 975 rows):** Majority-vote where combo mapped to multiple sectors
- **Tier 3 (15 combos, 226 rows):** Manual hardcode for combos that only appeared in missing-Sector rows (e.g., Finance/Financial Conglomerates → Financials, Energy Minerals/Coal → Energy)

Key validation: no ticker ever changes its (Industry_Group, Industry) combo or Sector across years, so every ticker gets the same Sector in every year.

#### 3. Deleted Non-Equity Rows & Rescaled Weights (87 rows)
Removed non-S&P 500 holdings that were fund artifacts:
- CASH_USD (cash position, ~0.3% weight, 25 years)
- SPUS_PR:00000117 (benchmark index line, 0% weight, all 27 years)
- 16 US Treasury bills (0% weight, 2016–2020)
- Fund artifacts: private placements, derivatives, futures, fixed income, mutual funds

After deletion, Port. Weight was proportionally rescaled so each sheet sums to exactly 100%. Scale factors ranged from 0.999944 (2009) to 1.006816 (2002). Sector sub-header rows were recalculated to match.

#### 4. Ticker Cleanup (907 cells updated)
FactSet uses internal disambiguation suffixes and non-standard ticker formats. Cleaned:

**Stripped non-overlapping dot suffixes (118 tickers):**
e.g., `BUD.2` → `BUD`, `EDS.` → `EDS`, `DNB.I` → `DNB`, `DELL.1` → `DELL`

**Mapped numeric CUSIPs to proper tickers (9 tickers):**
`202980` → `ASD`, `209399` → `BR`, `218333` → `CBSS`, `221781` → `CNG`, `231565` → `EC`, `237585` → `MIR`, `247236` → `JP`, `282406` → `SOTR`, `297519` → `TAP.B`

**Fixed Canadian prefix tickers (3):**
`*AL.XX1` → `AL`, `*N.XX1` → `N`, `*PDG.XX1` → `PDG`

**Left as-is by design:**
- 8 overlapping suffixed tickers (DD.2, T.2, CB.1, AGN.2, FCX.A, JPM.Z, TWX.1, USB.1) — different companies sharing same base ticker in same years
- 5 share classes (BRK.A, BRK.B, BF.B, LEN.B, FSL.B)
- 4 collision cases (WB.1, WB.3, ACV.A, ACV.I, S.XX1) — would create duplicates if stripped
- 3 numeric CUSIPs that couldn't be mapped without collisions: `247680` (Johnson Controls, JCI taken by Tyco), `278731` (Sears, S collides with Sprint), `2614528` (Seagram preferred)
- 54 Q-suffix tickers (LEHMQ, WAMUQ, etc.) — legitimate bankruptcy/OTC tickers

### Current Data Quality
- 11 GICS Sectors, fully populated on all equity rows
- 20 Industry_Group values, 99–115 Industry values per year
- Port. Weight sums to exactly 100% on every sheet
- 0 duplicate tickers in any sheet
- 502–510 stocks per year

## Running
```bash
# Quick single-year test
python3 run.py --start-year 2020 --end-year 2021

# Full backtest
python3 run.py

# Self-test optimizer
python3 src/portfolio_optimizer.py
```
