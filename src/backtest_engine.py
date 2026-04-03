"""
Backtest Engine for Minimum Active Share Portfolio Strategy
Uses MILP optimizer for provably optimal portfolio construction.
Uses FMP (Financial Modeling Prep) API for historical price data and dividends.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import argparse

from milp_optimizer import MILPActiveShareOptimizer
from fmp_client import FMPClient
from data_loader import load_benchmark_from_xlsx

warnings.filterwarnings('ignore')

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_TICKER = "SPY"

# S&P 500 Total Return Index (SPXTR) official calendar-year returns
# Source: S&P Dow Jones Indices via Motley Fool, US500.com, ycharts (cross-referenced)
SP500_TOTAL_RETURNS = {
    2000: -9.10,
    2001: -11.89,
    2002: -22.10,
    2003: 28.68,
    2004: 10.88,
    2005: 4.91,
    2006: 15.79,
    2007: 5.49,
    2008: -37.00,
    2009: 26.46,
    2010: 15.06,
    2011: 2.11,
    2012: 16.00,
    2013: 32.39,
    2014: 13.69,
    2015: 1.38,
    2016: 11.96,
    2017: 21.83,
    2018: -4.38,
    2019: 31.49,
    2020: 18.40,
    2021: 28.71,
    2022: -18.11,
    2023: 26.29,
    2024: 25.02,
    2025: 17.88,
}

# Tickers with dot suffixes that are share classes (dot → hyphen for FMP)
_SHARE_CLASS_TICKERS = {'BRK.A', 'BRK.B', 'BF.B', 'LEN.B', 'FSL.B', 'TAP.B'}

# Hardcoded fallback returns for tickers FMP cannot find (acquired/merged/delisted).
# Verified against 1stock1.com, Bloomberg, SEC filings, Wells Fargo IR, and other sources.
# Key: (ticker, return_year) → (total_return%, price_return%, dividend_return%)
FALLBACK_RETURNS = {
    # WYE — American Home Products / Wyeth (acquired by Pfizer 2009)
    # Sources: 1stock1.com prices, Pfizer IR for dividends (~$0.92/yr through 2002, rising after)
    ('WYE', 2000): (64.3, 61.9, 2.3),
    ('WYE', 2001): (-2.0, -3.5, 1.4),
    ('WYE', 2002): (-37.6, -39.1, 1.5),
    ('WYE', 2003): (9.44, 7.0, 2.5),
    ('WYE', 2004): (-0.88, -3.1, 2.2),
    ('WYE', 2005): (8.27, 6.1, 2.2),
    ('WYE', 2006): (9.51, 7.3, 2.2),
    ('WYE', 2007): (-14.04, -16.1, 2.1),
    ('WYE', 2008): (-14.54, -17.2, 2.6),
    ('WYE', 2009): (31.26, 28.2, 3.1),

    # BUD — Anheuser-Busch Companies (acquired by InBev 2008)
    # Sources: 1stock1.com prices, div ~$0.72-0.80/yr
    ('BUD', 2002): (8.7, 7.1, 1.6),
    ('BUD', 2003): (10.5, 8.8, 1.6),
    ('BUD', 2004): (-1.95, -3.5, 1.5),

    # ONE — Bank One Corporation (acquired by JPMorgan 2004)
    # Sources: SEC filings, estimated ~3% dividend yield
    ('ONE', 2002): (-4.4, -7.4, 3.0),
    ('ONE', 2003): (27.71, 24.7, 3.0),
    ('ONE', 2004): (12.78, 9.8, 3.0),

    # BLS — BellSouth Corporation (acquired by AT&T Dec 2006)
    # Sources: 1stock1.com, historicalstockinfo.com, div ~$1.16-1.23/yr
    ('BLS', 2000): (-11.05, -13.7, 2.6),
    ('BLS', 2001): (-5.09, -8.1, 3.0),
    ('BLS', 2002): (-30.36, -33.6, 3.2),
    ('BLS', 2003): (13.16, 8.4, 4.8),
    ('BLS', 2004): (1.94, -2.4, 4.3),
    ('BLS', 2005): (1.77, -2.8, 4.5),
    ('BLS', 2006): (79.8, 75.5, 4.3),
    ('BLS', 2007): (24.30, 18.92, 5.38),  # BLS acquired by AT&T Dec 2006; use AT&T (T) 2007 return

    # CBS — CBS Corporation (merged into Viacom May 2000)
    # Sources: merger math (1.085 VIA.B shares per CBS share), digrin.com for Viacom prices
    ('CBS', 2000): (-22.7, -22.7, 0.0),

    # CMCSK — Comcast Class A Special
    # Sources: verified spreadsheet, consistent with CMCSA performance
    ('CMCSK', 2007): (-35.1, -35.1, 0.0),

    # DELL — Dell Computer (no dividends in this era)
    # Sources: 1stock1.com (exact matches for most years)
    ('DELL', 2000): (-65.81, -65.81, 0.0),
    ('DELL', 2001): (55.87, 55.87, 0.0),
    ('DELL', 2002): (-1.62, -1.62, 0.0),
    ('DELL', 2003): (27.08, 27.08, 0.0),
    ('DELL', 2004): (24.01, 24.01, 0.0),
    ('DELL', 2005): (-28.93, -28.93, 0.0),
    ('DELL', 2006): (-16.23, -16.23, 0.0),
    ('DELL', 2007): (-2.31, -2.31, 0.0),
    ('DELL', 2008): (-58.22, -58.22, 0.0),

    # EMC — EMC Corporation (acquired by Dell 2016, no dividends until 2013)
    # Sources: 1stock1.com (exact matches for 2010-2012), dividendmax.com for 2013 div
    ('EMC', 2000): (21.74, 21.74, 0.0),
    ('EMC', 2001): (-79.79, -79.79, 0.0),
    ('EMC', 2010): (31.08, 31.08, 0.0),
    ('EMC', 2011): (-5.94, -5.94, 0.0),
    ('EMC', 2012): (17.46, 17.46, 0.0),
    ('EMC', 2013): (0.21, -0.59, 0.8),

    # ENRNQ — Enron Corp (bankruptcy Dec 2001)
    ('ENRNQ', 2001): (-99.27, -99.27, 0.0),

    # FBF — FleetBoston Financial (acquired by Bank of America 2004)
    # Sources: verified spreadsheet, div ~$1.40/yr est
    ('FBF', 2002): (-30.19, -33.2, 3.0),
    ('FBF', 2004): (3.66, 0.7, 3.0),

    # G — Gillette Company (acquired by P&G 2005)
    # Sources: verified spreadsheet, div ~$0.65/yr
    ('G', 2003): (23.55, 21.6, 2.0),

    # GTE — GTE Corporation (merged with Bell Atlantic → Verizon Jun 2000)
    # Sources: companiesmarketcap.com, FMP VZ prices, Verizon IR
    # 1.22 VZ shares per GTE; VZ $46.16 YE2000 + ~$1.94 total div vs $62.25 start
    ('GTE', 2000): (-6.4, -9.5, 3.1),

    # LU — Lucent Technologies (tiny/no dividends)
    # Sources: 1stock1.com
    ('LU', 2000): (-81.0, -82.0, 1.0),
    ('LU', 2001): (-53.2, -53.4, 0.2),

    # WCOEQ — MCI WorldCom / WorldCom (bankruptcy Jul 2002)
    ('WCOEQ', 2000): (-73.5, -73.5, 0.0),
    ('WCOEQ', 2002): (-99.02, -99.02, 0.0),

    # MER — Merrill Lynch (acquired by Bank of America 2009)
    # Sources: verified spreadsheet, div ~$0.64-1.40/yr
    ('MER', 2001): (-22.89, -23.9, 1.0),
    ('MER', 2002): (-26.04, -27.4, 1.3),
    ('MER', 2003): (56.75, 55.4, 1.4),
    ('MER', 2004): (3.12, 1.7, 1.4),
    ('MER', 2005): (14.77, 13.0, 1.8),
    ('MER', 2006): (38.82, 37.0, 1.8),
    ('MER', 2007): (-41.32, -43.7, 2.4),

    # MON — Monsanto Company (acquired by Bayer 2018)
    # Sources: 1stock1.com prices, div ~$0.96-1.00/yr
    ('MON', 2008): (-36.54, -37.01, 0.5),
    ('MON', 2009): (17.74, 16.2, 1.5),
    ('MON', 2010): (-13.24, -14.8, 1.0),

    # TFCFA — News Corp Class A / 21st Century Fox
    # Sources: verified spreadsheet
    ('TFCFA', 2005): (-16.07, -16.07, 0.0),
    ('TFCFA', 2013): (57.18, 57.18, 0.0),
    ('TFCFA', 2014): (9.99, 9.99, 0.0),

    # NRTLQ — Nortel Networks (no dividends, bankruptcy)
    # Sources: verified spreadsheet
    ('NRTLQ', 2000): (-35.75, -35.75, 0.0),
    ('NRTLQ', 2001): (-76.68, -76.68, 0.0),

    # PHA — Pharmacia Corp (acquired by Pfizer Apr 2003)
    # Sources: verified spreadsheet, small dividend
    ('PHA', 2001): (-29.32, -30.3, 1.0),
    ('PHA', 2002): (5.71, 4.7, 1.0),
    ('PHA', 2003): (8.15, 7.2, 1.0),

    # RDPL — Royal Dutch Petroleum ADR (merged into Royal Dutch Shell 2005)
    # Sources: verified spreadsheet, div ~3-4%
    ('RDPL', 2000): (1.52, -2.0, 3.5),
    ('RDPL', 2001): (-17.11, -20.6, 3.5),
    ('RDPL', 2002): (-7.27, -10.3, 3.0),

    # SGP — Schering-Plough Corp (acquired by Merck 2009)
    # Sources: verified spreadsheet, div ~$0.62/yr
    ('SGP', 2000): (36.96, 35.0, 2.0),
    ('SGP', 2001): (-35.92, -37.4, 1.5),
    ('SGP', 2002): (-36.41, -37.9, 1.5),
    ('SGP', 2003): (-18.71, -20.2, 1.5),

    # S.XX1 — Sprint / Sprint Nextel
    # Sources: verified spreadsheet, minimal/no dividend in 2000, small div later
    ('S.XX1', 2000): (-69.87, -69.87, 0.0),
    ('S.XX1', 2006): (-11.45, -13.5, 2.0),
    ('S.XX1', 2007): (-30.12, -32.1, 2.0),

    # JAVAD — Sun Microsystems (no dividends)
    # Sources: 1stock1.com
    ('JAVAD', 2000): (-28.01, -28.01, 0.0),
    ('JAVAD', 2001): (-55.87, -55.87, 0.0),
    ('JAVAD', 2002): (-74.72, -74.72, 0.0),

    # PARAA — Viacom Inc Class A (2001 only)
    ('PARAA', 2001): (-5.85, -5.85, 0.0),

    # PSKY — Viacom Inc Class B (no dividends until 2005)
    # Sources: digrin.com monthly prices cross-validated against SEC 10-K filings
    ('PSKY', 2002): (-7.7, -7.7, 0.0),
    ('PSKY', 2003): (8.9, 8.9, 0.0),
    ('PSKY', 2004): (-18.0, -18.0, 0.0),

    # WB.3 — Wachovia Corp (acquired by Wells Fargo 2008)
    # Sources: 1stock1.com prices, Wells Fargo IR for dividends
    # NOTE: 2003 total return corrected from erroneous +1.16% to ~+31.3%
    ('WB.3', 2002): (19.52, 16.2, 3.3),
    ('WB.3', 2003): (31.3, 27.85, 3.4),
    ('WB.3', 2004): (16.8, 12.9, 3.6),
    ('WB.3', 2005): (4.29, 0.49, 3.7),
    ('WB.3', 2006): (11.79, 7.74, 4.1),
    ('WB.3', 2007): (-29.91, -33.22, 4.2),
    ('WB.3', 2008): (-84.73, -84.56, 3.0),

    # WLA — Warner-Lambert Co (acquired by Pfizer Jun 2000)
    ('WLA', 2000): (0.58, -1.4, 2.0),

    # WAMUQ — Washington Mutual (failed 2008)
    # Sources: verified spreadsheet
    ('WAMUQ', 2003): (25.26, 22.3, 3.0),

    # LBTA — Yahoo! Inc (no dividends)
    # Sources: verified spreadsheet
    ('LBTA', 2000): (-87.34, -87.34, 0.0),
    ('LBTA', 2005): (2.62, 2.62, 0.0),
    ('LBTA', 2006): (-37.57, -37.57, 0.0),
}


def ticker_to_fmp(ticker: str) -> str:
    """Convert xlsx ticker to FMP API ticker format."""
    if ticker in _SHARE_CLASS_TICKERS:
        return ticker.replace('.', '-')
    elif '.' in ticker:
        return ticker.split('.')[0]
    return ticker


def fetch_returns_fmp(fmp_client: FMPClient, tickers: List[str], start_date: str,
                      end_date: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch historical returns using FMP API, with fallback to hardcoded verified returns
    for tickers FMP cannot find (acquired/merged/delisted companies).

    Returns:
        Tuple of (returns DataFrame with columns [ticker, return, price_return, dividend_return],
                  list of missing tickers)
    """
    results = []
    missing = []
    return_year = int(start_date[:4])

    for ticker in tickers:
        fmp_ticker = ticker_to_fmp(ticker)
        total_ret, price_ret, div_ret = fmp_client.get_total_return(fmp_ticker, start_date, end_date)

        if total_ret is not None:
            results.append({
                'ticker': ticker,
                'return': total_ret,
                'price_return': price_ret,
                'dividend_return': div_ret
            })
        elif (ticker, return_year) in FALLBACK_RETURNS:
            fb_total, fb_price, fb_div = FALLBACK_RETURNS[(ticker, return_year)]
            results.append({
                'ticker': ticker,
                'return': fb_total,
                'price_return': fb_price,
                'dividend_return': fb_div
            })
        else:
            missing.append(ticker)

    return pd.DataFrame(results), missing


def get_benchmark_return(holding_year: int) -> float:
    """Get official S&P 500 Total Return (SPXTR) for a calendar year."""
    ret = SP500_TOTAL_RETURNS.get(holding_year)
    if ret is None:
        return np.nan
    return ret


def calculate_portfolio_return(portfolio: pd.DataFrame, returns: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Calculate weight-adjusted portfolio return with dividend breakdown.

    Returns:
        Tuple of (total_return, price_return, dividend_return, coverage_pct)
    """
    if returns.empty or 'ticker' not in returns.columns:
        return np.nan, np.nan, np.nan, 0.0

    merged = pd.merge(portfolio, returns, on='ticker', how='left')

    has_return = merged['return'].notna()
    covered_weight = merged.loc[has_return, 'weight'].sum()
    total_weight = merged['weight'].sum()
    coverage_pct = (covered_weight / total_weight) * 100

    merged = merged.dropna(subset=['return'])
    if len(merged) == 0:
        return np.nan, np.nan, np.nan, 0.0

    merged['norm_weight'] = merged['weight'] / merged['weight'].sum()
    total_return = (merged['norm_weight'] * merged['return']).sum()
    price_return = (merged['norm_weight'] * merged['price_return']).sum()
    dividend_return = (merged['norm_weight'] * merged['dividend_return']).sum()

    return total_return, price_return, dividend_return, coverage_pct


def calculate_active_share(portfolio: pd.DataFrame, benchmark: pd.DataFrame) -> float:
    """Calculate active share between portfolio and benchmark."""
    merged = pd.merge(
        portfolio[['ticker', 'weight']].rename(columns={'weight': 'port_weight'}),
        benchmark[['ticker', 'weight']].rename(columns={'weight': 'bench_weight'}),
        on='ticker',
        how='outer'
    ).fillna(0)
    return 0.5 * np.abs(merged['port_weight'] - merged['bench_weight']).sum()


def run_single_year(snapshot_year: int, fmp_client: FMPClient, target_stocks: int = 60,
                    verbose: bool = False) -> Dict:
    """
    Run backtest for a single snapshot year using MILP optimizer.

    Args:
        snapshot_year: Year-end snapshot to construct portfolio from (e.g., 1999)
        fmp_client: FMPClient instance
        target_stocks: Number of stocks in portfolio
        verbose: Print detailed output

    Returns:
        Dictionary with results keyed by holding year (snapshot_year + 1)
    """
    holding_year = snapshot_year + 1

    try:
        benchmark = load_benchmark_from_xlsx(snapshot_year)
    except Exception as e:
        return {'year': holding_year, 'error': str(e)}

    # Suppress output during batch runs
    import io, sys
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        optimizer = MILPActiveShareOptimizer(benchmark, target_stocks=target_stocks,
                                              sector_tolerance=2.0, ig_tolerance=2.0)
        portfolio = optimizer.optimize()
    finally:
        if not verbose:
            sys.stdout = old_stdout

    if portfolio is None:
        return {'year': holding_year, 'error': 'MILP infeasible'}

    # Calculate active share
    active_share = calculate_active_share(portfolio, benchmark)

    # Constraint checks
    bench_sectors = benchmark.groupby('sector')['weight'].sum()
    port_sectors = portfolio.groupby('sector')['weight'].sum()
    max_sector_dev = max(abs(port_sectors.get(s, 0) - bench_sectors[s]) for s in bench_sectors.index)
    passed_sector = max_sector_dev <= 2.01

    bench_ig = benchmark.groupby('industry_group')['weight'].sum()
    port_ig = portfolio.groupby('industry_group')['weight'].sum()
    max_ig_dev = max(abs(port_ig.get(g, 0) - bench_ig[g]) for g in bench_ig.index)
    passed_ig = max_ig_dev <= 2.01

    # Holding period: Jan 1 to Dec 31 of the holding year
    start_date = f"{holding_year}-01-01"
    end_date = f"{holding_year}-12-31"

    # Fetch returns
    all_tickers = portfolio['ticker'].tolist()
    returns_df, missing_tickers = fetch_returns_fmp(fmp_client, all_tickers, start_date, end_date)

    portfolio_return, price_return, dividend_return, coverage = calculate_portfolio_return(portfolio, returns_df)
    benchmark_return = get_benchmark_return(holding_year)
    active_return = portfolio_return - benchmark_return if not (pd.isna(portfolio_return) or pd.isna(benchmark_return)) else np.nan

    return {
        'year': holding_year,
        'portfolio_return': portfolio_return,
        'price_return': price_return,
        'dividend_return': dividend_return,
        'benchmark_return': benchmark_return,
        'benchmark_price_return': np.nan,
        'benchmark_dividend_return': np.nan,
        'active_return': active_return,
        'active_share': active_share,
        'max_sector_deviation': max_sector_dev,
        'max_ig_deviation': max_ig_dev,
        'passed_sector_constraint': passed_sector,
        'passed_ig_constraint': passed_ig,
        'coverage_pct': coverage,
        'n_stocks': len(portfolio),
        'missing_tickers': missing_tickers,
        'portfolio': portfolio,
        'stock_returns': returns_df
    }


def run_full_backtest(start_year: int = 2000, end_year: int = 2025, target_stocks: int = 60,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Run full backtest across all years.

    Args:
        start_year: First holding year (e.g., 2000 = portfolio constructed from 1999 snapshot)
        end_year: Last holding year inclusive (e.g., 2025)
        target_stocks: Number of stocks in portfolio
        use_cache: Use FMP disk cache

    Returns:
        DataFrame with annual results
    """
    fmp_client = FMPClient(use_cache=use_cache)
    all_results = []
    all_missing_tickers = []

    print(f"\n{'='*80}")
    print(f"MILP MINIMUM ACTIVE SHARE BACKTEST")
    print(f"Holding years: {start_year}-{end_year}")
    print(f"Target: {target_stocks} stocks | Constraints: Sector +-2%, Industry_Group +-2%")
    print(f"Data Source: FMP API ({'cached' if use_cache else 'live'})")
    print(f"{'='*80}\n")

    # Iterate over snapshot years; holding year = snapshot_year + 1
    for snapshot_year in range(start_year - 1, end_year):
        holding_year = snapshot_year + 1
        print(f"Processing {holding_year}...", end=" ", flush=True)
        result = run_single_year(snapshot_year, fmp_client, target_stocks, verbose=False)
        all_results.append(result)

        if 'missing_tickers' in result and result['missing_tickers']:
            for ticker in result['missing_tickers']:
                all_missing_tickers.append({'year': holding_year, 'ticker': ticker})

        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"AS={result['active_share']:.1f}% cov={result['coverage_pct']:.0f}%")

    # Extract portfolios and per-stock returns keyed by holding year
    portfolios = {}
    stock_returns = {}
    for result in all_results:
        if 'portfolio' in result and result['portfolio'] is not None:
            portfolios[result['year']] = result['portfolio']
        if 'stock_returns' in result and result['stock_returns'] is not None:
            stock_returns[result['year']] = result['stock_returns']

    results_df = pd.DataFrame(all_results)
    for col in ['portfolio', 'missing_tickers', 'stock_returns']:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])

    missing_df = pd.DataFrame(all_missing_tickers)

    return results_df, missing_df, portfolios, stock_returns


def calculate_summary_stats(results_df: pd.DataFrame) -> Dict:
    """Calculate summary statistics."""
    valid = results_df.dropna(subset=['portfolio_return', 'benchmark_return'])
    if len(valid) == 0:
        return {}

    cumulative_return = np.prod(1 + valid['portfolio_return'] / 100) - 1
    benchmark_cumulative = np.prod(1 + valid['benchmark_return'] / 100) - 1

    n_years = len(valid)
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    benchmark_annualized = (1 + benchmark_cumulative) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        'total_return': cumulative_return * 100,
        'annualized_return': annualized_return * 100,
        'vs_benchmark': (cumulative_return - benchmark_cumulative) * 100,
        'benchmark_total': benchmark_cumulative * 100,
        'benchmark_annualized': benchmark_annualized * 100,
        'avg_active_share': valid['active_share'].mean(),
        'avg_coverage': valid['coverage_pct'].mean(),
        'max_sector_deviation': valid['max_sector_deviation'].max(),
        'max_ig_deviation': valid['max_ig_deviation'].max(),
        'total_years': len(valid),
    }


def print_results(results_df: pd.DataFrame, missing_df: pd.DataFrame):
    """Print formatted backtest results."""
    stats = calculate_summary_stats(results_df)
    if not stats:
        print("No valid results to display.")
        return

    print("\n" + "=" * 100)
    print("MILP MINIMUM ACTIVE SHARE BACKTEST RESULTS")
    print("=" * 100)

    print(f"\nSUMMARY")
    print("-" * 100)
    print(f"  Portfolio Total Return:     {stats['total_return']:>+10.1f}%")
    print(f"  Portfolio Annualized:       {stats['annualized_return']:>+10.1f}%")
    print(f"  S&P 500 Total Return:      {stats['benchmark_total']:>+10.1f}%")
    print(f"  S&P 500 Annualized:        {stats['benchmark_annualized']:>+10.1f}%")
    print(f"  vs Benchmark:              {stats['vs_benchmark']:>+10.1f}%")
    print(f"  Avg Active Share:          {stats['avg_active_share']:>10.1f}%")
    print(f"  Avg Data Coverage:         {stats['avg_coverage']:>10.0f}%")
    print(f"  Max Sector Deviation:      {stats['max_sector_deviation']:>10.2f}%")
    print(f"  Max Industry_Group Dev:    {stats['max_ig_deviation']:>10.2f}%")
    print("-" * 100)

    # Year-by-year detail
    print(f"\nYEAR-BY-YEAR DETAIL")
    print("-" * 110)
    print(f"{'Year':<6} {'Portfolio':>12} {'S&P 500':>12} {'Active':>12} {'Active':>10} {'Sector':>10} {'IG':>10} {'Coverage':>10}")
    print(f"{'':6} {'Return':>12} {'Return':>12} {'Return':>12} {'Share':>10} {'MaxDev':>10} {'MaxDev':>10} {'':>10}")
    print("-" * 110)

    valid = results_df.dropna(subset=['active_share']).sort_values('year')
    for _, row in valid.iterrows():
        port_ret = f"{row['portfolio_return']:+.2f}%" if pd.notna(row['portfolio_return']) else "N/A"
        bench_ret = f"{row['benchmark_return']:+.2f}%" if pd.notna(row['benchmark_return']) else "N/A"
        active_ret = f"{row['active_return']:+.2f}%" if pd.notna(row['active_return']) else "N/A"

        print(f"{int(row['year']):<6} {port_ret:>12} {bench_ret:>12} {active_ret:>12} "
              f"{row['active_share']:>9.1f}% {row['max_sector_deviation']:>9.2f}% "
              f"{row['max_ig_deviation']:>9.2f}% {row['coverage_pct']:>9.0f}%")

    print("-" * 110)

    # Dividend analysis
    div_valid = valid.dropna(subset=['dividend_return'])
    if len(div_valid) > 0 and not div_valid['dividend_return'].isna().all():
        print(f"\nDIVIDEND ANALYSIS")
        print("-" * 80)
        for _, row in div_valid.iterrows():
            if pd.notna(row.get('price_return')) and pd.notna(row.get('dividend_return')):
                print(f"  {int(row['year'])}: price={row['price_return']:+.1f}%  "
                      f"div={row['dividend_return']:+.2f}%  total={row['portfolio_return']:+.1f}%")
        avg_div = div_valid['dividend_return'].mean()
        print(f"\n  Avg Dividend Contribution: {avg_div:.2f}% annually")
        print("-" * 80)

    # Missing tickers
    if len(missing_df) > 0:
        print(f"\nMISSING TICKERS SUMMARY")
        print("-" * 80)
        unique_missing = missing_df.groupby('ticker').size().sort_values(ascending=False)
        print(f"  Total unique missing tickers: {len(unique_missing)}")
        print(f"  Top 10 most frequently missing:")
        for ticker, count in unique_missing.head(10).items():
            print(f"    {ticker}: missing in {count} years")
        print("-" * 80)


def save_results(results_df: pd.DataFrame, missing_df: pd.DataFrame, portfolios: Dict = None,
                  stock_returns: Dict = None):
    """Save results to CSV and Excel files."""
    # CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "annual_performance_milp.csv", index=False)

    stats = calculate_summary_stats(results_df)
    if stats:
        pd.DataFrame([stats]).to_csv(RESULTS_DIR / "summary_milp.csv", index=False)

    if len(missing_df) > 0:
        missing_df.to_csv(RESULTS_DIR / "missing_tickers.csv", index=False)

    print(f"\nCSV results saved to {RESULTS_DIR}/")

    # Excel — backtest results
    excel_dir = PROJECT_ROOT / "results-excel"
    excel_dir.mkdir(exist_ok=True)

    with pd.ExcelWriter(excel_dir / "backtest_results.xlsx", engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Annual Performance', index=False)
        if stats:
            pd.DataFrame([stats]).to_excel(writer, sheet_name='Summary', index=False)
        if len(missing_df) > 0:
            missing_df.to_excel(writer, sheet_name='Missing Tickers', index=False)

    print(f"Excel results saved to {excel_dir}/backtest_results.xlsx")

    # Excel — portfolio holdings per year (one sheet per year)
    if portfolios:
        save_portfolio_holdings(portfolios, excel_dir)

    # Excel — portfolio snapshots (60 held stocks per year with returns)
    if portfolios and stock_returns:
        save_portfolio_snapshots(portfolios, stock_returns, excel_dir)

    # Excel — return attribution by stock, sector, and industry group
    if portfolios and stock_returns:
        save_attribution_report(portfolios, stock_returns, excel_dir)


def save_portfolio_holdings(portfolios: Dict[int, pd.DataFrame], excel_dir: Path):
    """
    Save portfolio holdings to an Excel workbook with one sheet per year.
    Each sheet has three sections:
    1. Sector breakdown (portfolio vs benchmark weight + difference)
    2. Industry_Group breakdown (portfolio vs benchmark weight + difference)
    3. Stock-level detail with active share contribution
    """
    with pd.ExcelWriter(excel_dir / "portfolio_holdings.xlsx", engine='openpyxl') as writer:
        for year in sorted(portfolios.keys()):
            portfolio = portfolios[year]
            benchmark = load_benchmark_from_xlsx(year - 1)  # snapshot year = holding year - 1

            # Merge portfolio weights with full benchmark
            merged = pd.merge(
                benchmark[['ticker', 'weight', 'sector', 'industry_group', 'industry', 'company_name']],
                portfolio[['ticker', 'weight']].rename(columns={'weight': 'port_weight'}),
                on='ticker',
                how='left'
            )
            merged['port_weight'] = merged['port_weight'].fillna(0.0)
            merged = merged.rename(columns={'weight': 'bench_weight'})
            merged['difference'] = merged['port_weight'] - merged['bench_weight']
            merged['active_share_contribution'] = 0.5 * np.abs(merged['difference'])

            # --- Sector summary ---
            sector_bench = benchmark.groupby('sector')['weight'].sum()
            sector_port = merged.groupby('sector')['port_weight'].sum()
            held_mask = merged['port_weight'] > 0
            sector_held_counts = merged[held_mask].groupby('sector').size()
            sector_bench_counts = benchmark.groupby('sector').size()
            sector_df = pd.DataFrame({
                'Sector': sector_bench.index,
                '# Held': [int(sector_held_counts.get(s, 0)) for s in sector_bench.index],
                '# in Bench': [int(sector_bench_counts.get(s, 0)) for s in sector_bench.index],
                'Port. Weight': [sector_port.get(s, 0.0) for s in sector_bench.index],
                'Bench. Weight': sector_bench.values,
                'Difference': [sector_port.get(s, 0.0) - sector_bench[s] for s in sector_bench.index],
            }).sort_values('Bench. Weight', ascending=False).reset_index(drop=True)
            # Add total row
            sector_total = pd.DataFrame([{
                'Sector': 'Total',
                '# Held': int(sector_df['# Held'].sum()),
                '# in Bench': int(sector_df['# in Bench'].sum()),
                'Port. Weight': sector_df['Port. Weight'].sum(),
                'Bench. Weight': sector_df['Bench. Weight'].sum(),
                'Difference': sector_df['Difference'].sum(),
            }])
            sector_df = pd.concat([sector_df, sector_total], ignore_index=True)

            # --- Industry_Group summary ---
            ig_bench = benchmark.groupby('industry_group')['weight'].sum()
            ig_port = merged.groupby('industry_group')['port_weight'].sum()
            ig_held_counts = merged[held_mask].groupby('industry_group').size()
            ig_bench_counts = benchmark.groupby('industry_group').size()
            ig_df = pd.DataFrame({
                'Industry_Group': ig_bench.index,
                '# Held': [int(ig_held_counts.get(g, 0)) for g in ig_bench.index],
                '# in Bench': [int(ig_bench_counts.get(g, 0)) for g in ig_bench.index],
                'Port. Weight': [ig_port.get(g, 0.0) for g in ig_bench.index],
                'Bench. Weight': ig_bench.values,
                'Difference': [ig_port.get(g, 0.0) - ig_bench[g] for g in ig_bench.index],
            }).sort_values('Bench. Weight', ascending=False).reset_index(drop=True)
            # Add total row
            ig_total = pd.DataFrame([{
                'Industry_Group': 'Total',
                '# Held': int(ig_df['# Held'].sum()),
                '# in Bench': int(ig_df['# in Bench'].sum()),
                'Port. Weight': ig_df['Port. Weight'].sum(),
                'Bench. Weight': ig_df['Bench. Weight'].sum(),
                'Difference': ig_df['Difference'].sum(),
            }])
            ig_df = pd.concat([ig_df, ig_total], ignore_index=True)

            # --- Stock detail ---
            held = merged[merged['port_weight'] > 0].sort_values('port_weight', ascending=False)
            not_held = merged[merged['port_weight'] == 0].sort_values('bench_weight', ascending=False)
            stock_df = pd.concat([held, not_held], ignore_index=True)
            stock_df = stock_df[[
                'company_name', 'port_weight', 'bench_weight', 'difference',
                'active_share_contribution', 'industry_group', 'industry', 'sector', 'ticker'
            ]]
            stock_df.columns = [
                'Name', 'Port. Weight', 'Bench. Weight', 'Difference',
                'AS Contribution', 'Industry_Group', 'Industry', 'Sector', 'Ticker'
            ]

            # Write all three sections to the sheet with spacing
            row = 0
            # Header label
            header_df = pd.DataFrame([{'': f'MILP Portfolio — {year} (S&P {year-1} snapshot, Active Share: {merged["active_share_contribution"].sum():.2f}%)'}])
            header_df.to_excel(writer, sheet_name=str(year), startrow=row, index=False, header=False)
            row += 2

            # Sector breakdown
            label_df = pd.DataFrame([{'': 'SECTOR BREAKDOWN'}])
            label_df.to_excel(writer, sheet_name=str(year), startrow=row, index=False, header=False)
            row += 1
            sector_df.to_excel(writer, sheet_name=str(year), startrow=row, index=False)
            row += len(sector_df) + 2

            # Industry_Group breakdown
            label_df = pd.DataFrame([{'': 'INDUSTRY GROUP BREAKDOWN'}])
            label_df.to_excel(writer, sheet_name=str(year), startrow=row, index=False, header=False)
            row += 1
            ig_df.to_excel(writer, sheet_name=str(year), startrow=row, index=False)
            row += len(ig_df) + 2

            # Stock detail
            label_df = pd.DataFrame([{'': f'STOCK DETAIL ({(stock_df["Port. Weight"] > 0).sum()} held / {len(stock_df)} total)'}])
            label_df.to_excel(writer, sheet_name=str(year), startrow=row, index=False, header=False)
            row += 1
            stock_df.to_excel(writer, sheet_name=str(year), startrow=row, index=False)

    print(f"Portfolio holdings saved to {excel_dir}/portfolio_holdings.xlsx")


def save_portfolio_snapshots(portfolios: Dict[int, pd.DataFrame],
                              stock_returns: Dict[int, pd.DataFrame],
                              excel_dir: Path):
    """
    Save a workbook with one sheet per year showing the 60 held stocks,
    their portfolio weight, benchmark weight, and annual return from FMP.
    """
    with pd.ExcelWriter(excel_dir / "Portfolio_individual_stock_returns.xlsx", engine='openpyxl') as writer:
        for year in sorted(portfolios.keys()):
            portfolio = portfolios[year]
            benchmark = load_benchmark_from_xlsx(year - 1)  # snapshot year = holding year - 1
            returns_df = stock_returns.get(year, pd.DataFrame())

            # Start from held stocks only
            held = portfolio[portfolio['weight'] > 0].copy()

            # Merge benchmark weight
            bench_map = benchmark.set_index('ticker')[['weight', 'sector', 'industry_group', 'company_name']]
            held = held.merge(bench_map, left_on='ticker', right_index=True, how='left', suffixes=('', '_bench'))
            held = held.rename(columns={'weight': 'Port. Weight', 'weight_bench': 'Bench. Weight'})

            # Merge returns
            if not returns_df.empty:
                held = held.merge(
                    returns_df[['ticker', 'return', 'price_return', 'dividend_return']],
                    on='ticker', how='left'
                )
                held = held.rename(columns={
                    'return': 'Total Return (%)',
                    'price_return': 'Price Return (%)',
                    'dividend_return': 'Dividend Return (%)',
                })
            else:
                held['Total Return (%)'] = np.nan
                held['Price Return (%)'] = np.nan
                held['Dividend Return (%)'] = np.nan

            held['Difference'] = held['Port. Weight'] - held['Bench. Weight']
            held = held.sort_values('Port. Weight', ascending=False)

            # Select and order columns
            snapshot = held[[
                'ticker', 'company_name', 'sector', 'industry_group',
                'Port. Weight', 'Bench. Weight', 'Difference',
                'Total Return (%)', 'Price Return (%)', 'Dividend Return (%)',
            ]].copy()
            snapshot.columns = [
                'Ticker', 'Name', 'Sector', 'Industry Group',
                'Port. Weight', 'Bench. Weight', 'Difference',
                'Total Return (%)', 'Price Return (%)', 'Dividend Return (%)',
            ]

            sheet_name = str(year)
            header_df = pd.DataFrame([{'': f'Portfolio Snapshot — {year} (S&P {year-1} snapshot)'}])
            header_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False, header=False)
            snapshot.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)

    print(f"Portfolio snapshots saved to {excel_dir}/Portfolio_individual_stock_returns.xlsx")


def save_attribution_report(portfolios: Dict[int, pd.DataFrame],
                             stock_returns: Dict[int, pd.DataFrame],
                             excel_dir: Path):
    """
    Save a return attribution workbook with one sheet per year.
    Each sheet has: Sector Attribution, Industry Group Attribution, Stock Attribution.
    """
    with pd.ExcelWriter(excel_dir / "return_attribution.xlsx", engine='openpyxl') as writer:
        for year in sorted(portfolios.keys()):
            portfolio = portfolios[year]
            returns_df = stock_returns.get(year, pd.DataFrame())
            benchmark = load_benchmark_from_xlsx(year - 1)  # snapshot year = holding year - 1

            # Merge portfolio with benchmark company names and returns
            held = portfolio[portfolio['weight'] > 0].copy()
            bench_names = benchmark.set_index('ticker')[['company_name']]
            held = held.merge(bench_names, left_on='ticker', right_index=True, how='left')

            if not returns_df.empty:
                held = held.merge(
                    returns_df[['ticker', 'return']],
                    on='ticker', how='left'
                )
            else:
                held['return'] = np.nan

            # Normalize weights (only among stocks with return data) to match
            # calculate_portfolio_return logic
            has_return = held['return'].notna()
            covered_weight = held.loc[has_return, 'weight'].sum()
            held['norm_weight'] = 0.0
            if covered_weight > 0:
                held.loc[has_return, 'norm_weight'] = held.loc[has_return, 'weight'] / covered_weight
            held['contribution'] = held['norm_weight'] * held['return']

            sheet_name = str(year)
            row = 0

            # Header
            total_return = held['contribution'].sum()
            header_df = pd.DataFrame([{'': f'Return Attribution — {year} (Portfolio Return: {total_return:.2f}%)'}])
            header_df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
            row += 2

            # --- Sector Attribution ---
            def _group_attribution(held_df, group_col, label):
                grouped = held_df[held_df['return'].notna()].groupby(group_col).agg(
                    n_stocks=('ticker', 'size'),
                    port_weight=('norm_weight', lambda x: x.sum() * 100),
                    contribution=('contribution', 'sum'),
                ).reset_index()
                # Weighted average return per group
                def wavg_return(sub):
                    w = sub['norm_weight']
                    if w.sum() == 0:
                        return np.nan
                    return (sub['return'] * w).sum() / w.sum()
                avg_returns = held_df[held_df['return'].notna()].groupby(group_col).apply(wavg_return).reset_index()
                avg_returns.columns = [group_col, 'avg_return']
                grouped = grouped.merge(avg_returns, on=group_col, how='left')

                grouped = grouped.sort_values('contribution', ascending=False).reset_index(drop=True)
                result = pd.DataFrame({
                    label: grouped[group_col],
                    '# Stocks': grouped['n_stocks'].astype(int),
                    'Port. Weight (%)': grouped['port_weight'],
                    'Return (%)': grouped['avg_return'],
                    'Contribution (%)': grouped['contribution'],
                })
                # Total row
                total = pd.DataFrame([{
                    label: 'Total',
                    '# Stocks': int(result['# Stocks'].sum()),
                    'Port. Weight (%)': result['Port. Weight (%)'].sum(),
                    'Return (%)': np.nan,
                    'Contribution (%)': result['Contribution (%)'].sum(),
                }])
                return pd.concat([result, total], ignore_index=True)

            # Sector
            label_df = pd.DataFrame([{'': 'SECTOR ATTRIBUTION'}])
            label_df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
            row += 1
            sector_attr = _group_attribution(held, 'sector', 'Sector')
            sector_attr.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
            row += len(sector_attr) + 2

            # Industry Group
            label_df = pd.DataFrame([{'': 'INDUSTRY GROUP ATTRIBUTION'}])
            label_df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
            row += 1
            ig_attr = _group_attribution(held, 'industry_group', 'Industry_Group')
            ig_attr.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
            row += len(ig_attr) + 2

            # --- Stock Attribution ---
            label_df = pd.DataFrame([{'': 'STOCK ATTRIBUTION'}])
            label_df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
            row += 1
            stock_attr = held.sort_values('contribution', ascending=False)
            stock_out = pd.DataFrame({
                'Ticker': stock_attr['ticker'],
                'Name': stock_attr['company_name'],
                'Sector': stock_attr['sector'],
                'Industry Group': stock_attr['industry_group'],
                'Port. Weight (%)': stock_attr['norm_weight'] * 100,
                'Return (%)': stock_attr['return'],
                'Contribution (%)': stock_attr['contribution'],
            }).reset_index(drop=True)
            stock_out.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)

    print(f"Return attribution saved to {excel_dir}/return_attribution.xlsx")


def main():
    parser = argparse.ArgumentParser(description='Run MILP Minimum Active Share Backtest')
    parser.add_argument('--start-year', type=int, default=2000, help='First holding year (default: 2000)')
    parser.add_argument('--end-year', type=int, default=2025, help='Last holding year inclusive (default: 2025)')
    parser.add_argument('--target-stocks', type=int, default=60, help='Number of stocks (default: 60)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to CSV')
    parser.add_argument('--no-cache', action='store_true', help='Disable FMP API caching')

    args = parser.parse_args()

    results_df, missing_df, portfolios, stock_returns = run_full_backtest(
        start_year=args.start_year,
        end_year=args.end_year,
        target_stocks=args.target_stocks,
        use_cache=not args.no_cache,
    )

    print_results(results_df, missing_df)

    # Turnover and tax analysis
    from turnover_analyzer import compute_turnover, TaxSimulator, print_turnover_results, print_tax_results, save_turnover_and_tax_results

    turnover_df = compute_turnover(portfolios, stock_returns)
    print_turnover_results(turnover_df)

    tax_sim = TaxSimulator(portfolios, stock_returns)
    no_harvest_df, with_harvest_df = tax_sim.run()
    print_tax_results(no_harvest_df, with_harvest_df)

    if not args.no_save:
        save_results(results_df, missing_df, portfolios, stock_returns)
        save_turnover_and_tax_results(turnover_df, no_harvest_df, with_harvest_df)


if __name__ == '__main__':
    main()
