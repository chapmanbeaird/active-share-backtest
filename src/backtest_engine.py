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

# Tickers with dot suffixes that are share classes (dot → hyphen for FMP)
_SHARE_CLASS_TICKERS = {'BRK.A', 'BRK.B', 'BF.B', 'LEN.B', 'FSL.B', 'TAP.B'}


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
    Fetch historical returns using FMP API.

    Returns:
        Tuple of (returns DataFrame with columns [ticker, return, price_return, dividend_return],
                  list of missing tickers)
    """
    results = []
    missing = []

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
        else:
            missing.append(ticker)

    return pd.DataFrame(results), missing


def fetch_benchmark_return_fmp(fmp_client: FMPClient, start_date: str,
                               end_date: str) -> Tuple[float, float, float]:
    """Fetch S&P 500 (SPY) return for the period using FMP."""
    total_ret, price_ret, div_ret = fmp_client.get_total_return(BENCHMARK_TICKER, start_date, end_date)
    if total_ret is None:
        return np.nan, np.nan, np.nan
    return total_ret, price_ret, div_ret


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


def run_single_year(year: int, fmp_client: FMPClient, target_stocks: int = 60,
                    verbose: bool = False) -> Dict:
    """
    Run backtest for a single year using MILP optimizer.

    Args:
        year: Year to construct portfolio (uses year-end snapshot)
        fmp_client: FMPClient instance
        target_stocks: Number of stocks in portfolio
        verbose: Print detailed output

    Returns:
        Dictionary with year results
    """
    try:
        benchmark = load_benchmark_from_xlsx(year)
    except Exception as e:
        return {'year': year, 'error': str(e)}

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
        return {'year': year, 'error': 'MILP infeasible'}

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

    # Define holding period (Jan 1 to Dec 31 of the next year)
    # Snapshot is year-end December, so returns are for the full following year
    start_date = f"{year + 1}-01-01"
    end_date = f"{year + 1}-12-31"

    # Fetch returns
    all_tickers = portfolio['ticker'].tolist()
    returns_df, missing_tickers = fetch_returns_fmp(fmp_client, all_tickers, start_date, end_date)

    portfolio_return, price_return, dividend_return, coverage = calculate_portfolio_return(portfolio, returns_df)
    benchmark_return, bench_price, bench_div = fetch_benchmark_return_fmp(fmp_client, start_date, end_date)
    active_return = portfolio_return - benchmark_return if not (pd.isna(portfolio_return) or pd.isna(benchmark_return)) else np.nan

    return {
        'year': year,
        'portfolio_return': portfolio_return,
        'price_return': price_return,
        'dividend_return': dividend_return,
        'benchmark_return': benchmark_return,
        'benchmark_price_return': bench_price,
        'benchmark_dividend_return': bench_div,
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


def run_full_backtest(start_year: int = 1999, end_year: int = 2025, target_stocks: int = 60,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Run full backtest across all years.

    Args:
        start_year: First year of backtest
        end_year: Last year of backtest (exclusive)
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
    print(f"Snapshots: {start_year}-{end_year - 1} (year-end) | Returns: {start_year + 1}-{end_year}")
    print(f"Target: {target_stocks} stocks | Constraints: Sector +-2%, Industry_Group +-2%")
    print(f"Data Source: FMP API ({'cached' if use_cache else 'live'})")
    print(f"{'='*80}\n")

    for year in range(start_year, end_year):
        print(f"Processing {year}...", end=" ", flush=True)
        result = run_single_year(year, fmp_client, target_stocks, verbose=False)
        all_results.append(result)

        if 'missing_tickers' in result and result['missing_tickers']:
            for ticker in result['missing_tickers']:
                all_missing_tickers.append({'year': year, 'ticker': ticker})

        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"AS={result['active_share']:.1f}% cov={result['coverage_pct']:.0f}%")

    # Extract portfolios and per-stock returns before dropping from results
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
            benchmark = load_benchmark_from_xlsx(year)

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
            sector_df = pd.DataFrame({
                'Sector': sector_bench.index,
                'Port. Weight': [sector_port.get(s, 0.0) for s in sector_bench.index],
                'Bench. Weight': sector_bench.values,
                'Difference': [sector_port.get(s, 0.0) - sector_bench[s] for s in sector_bench.index],
            }).sort_values('Bench. Weight', ascending=False).reset_index(drop=True)
            # Add total row
            sector_total = pd.DataFrame([{
                'Sector': 'Total',
                'Port. Weight': sector_df['Port. Weight'].sum(),
                'Bench. Weight': sector_df['Bench. Weight'].sum(),
                'Difference': sector_df['Difference'].sum(),
            }])
            sector_df = pd.concat([sector_df, sector_total], ignore_index=True)

            # --- Industry_Group summary ---
            ig_bench = benchmark.groupby('industry_group')['weight'].sum()
            ig_port = merged.groupby('industry_group')['port_weight'].sum()
            ig_df = pd.DataFrame({
                'Industry_Group': ig_bench.index,
                'Port. Weight': [ig_port.get(g, 0.0) for g in ig_bench.index],
                'Bench. Weight': ig_bench.values,
                'Difference': [ig_port.get(g, 0.0) - ig_bench[g] for g in ig_bench.index],
            }).sort_values('Bench. Weight', ascending=False).reset_index(drop=True)
            # Add total row
            ig_total = pd.DataFrame([{
                'Industry_Group': 'Total',
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
            header_df = pd.DataFrame([{'': f'MILP Portfolio — {year} (Active Share: {merged["active_share_contribution"].sum():.2f}%)'}])
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
            benchmark = load_benchmark_from_xlsx(year)
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
            # Header with return period
            header_df = pd.DataFrame([{'': f'Portfolio Snapshot — {year} (Returns: {year+1})'}])
            header_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False, header=False)
            snapshot.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)

    print(f"Portfolio snapshots saved to {excel_dir}/Portfolio_individual_stock_returns.xlsx")


def main():
    parser = argparse.ArgumentParser(description='Run MILP Minimum Active Share Backtest')
    parser.add_argument('--start-year', type=int, default=1999, help='Start snapshot year (default: 1999)')
    parser.add_argument('--end-year', type=int, default=2025, help='End snapshot year exclusive (default: 2025, skips 2025 since 2026 not finished)')
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
