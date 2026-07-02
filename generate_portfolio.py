#!/usr/bin/env python3
"""Generate the minimum-active-share portfolio for a SINGLE benchmark snapshot.

Unlike run.py (which backtests historical Dec->Dec holding periods and fetches
forward returns via FMP), this solves the MILP for ONE snapshot sheet and
outputs the recommended holdings + active share + sector/industry-group
constraint checks. No returns are fetched, so it works for a current snapshot
that has no forward returns yet (e.g. a 6/30 mid-year S&P snapshot).

Usage:
    # New 6/30 snapshot added to the workbook as a sheet named "2026-06"
    python3 generate_portfolio.py --sheet 2026-06

    # Sanity-check against an existing year-end sheet
    python3 generate_portfolio.py --sheet 2025
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import load_benchmark_from_xlsx
from milp_optimizer import MILPActiveShareOptimizer


def build_sections(portfolio: pd.DataFrame, benchmark: pd.DataFrame):
    """Build the sector / industry-group / stock-detail tables.

    Mirrors save_portfolio_holdings() in backtest_engine.py so the output
    workbook matches the historical portfolio_holdings.xlsx layout.
    """
    merged = pd.merge(
        benchmark[['ticker', 'weight', 'sector', 'industry_group', 'industry', 'company_name']],
        portfolio[['ticker', 'weight']].rename(columns={'weight': 'port_weight'}),
        on='ticker',
        how='left',
    )
    merged['port_weight'] = merged['port_weight'].fillna(0.0)
    merged = merged.rename(columns={'weight': 'bench_weight'})
    merged['difference'] = merged['port_weight'] - merged['bench_weight']
    merged['active_share_contribution'] = 0.5 * np.abs(merged['difference'])

    held_mask = merged['port_weight'] > 0

    # --- Sector summary ---
    sector_bench = benchmark.groupby('sector')['weight'].sum()
    sector_port = merged.groupby('sector')['port_weight'].sum()
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
    sector_df = pd.concat([sector_df, pd.DataFrame([{
        'Sector': 'Total',
        '# Held': int(sector_df['# Held'].sum()),
        '# in Bench': int(sector_df['# in Bench'].sum()),
        'Port. Weight': sector_df['Port. Weight'].sum(),
        'Bench. Weight': sector_df['Bench. Weight'].sum(),
        'Difference': sector_df['Difference'].sum(),
    }])], ignore_index=True)

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
    ig_df = pd.concat([ig_df, pd.DataFrame([{
        'Industry_Group': 'Total',
        '# Held': int(ig_df['# Held'].sum()),
        '# in Bench': int(ig_df['# in Bench'].sum()),
        'Port. Weight': ig_df['Port. Weight'].sum(),
        'Bench. Weight': ig_df['Bench. Weight'].sum(),
        'Difference': ig_df['Difference'].sum(),
    }])], ignore_index=True)

    # --- Stock detail (held first, then remaining benchmark names) ---
    held = merged[held_mask].sort_values('port_weight', ascending=False)
    not_held = merged[~held_mask].sort_values('bench_weight', ascending=False)
    stock_df = pd.concat([held, not_held], ignore_index=True)[[
        'company_name', 'port_weight', 'bench_weight', 'difference',
        'active_share_contribution', 'industry_group', 'industry', 'sector', 'ticker'
    ]]
    stock_df.columns = [
        'Name', 'Port. Weight', 'Bench. Weight', 'Difference',
        'AS Contribution', 'Industry_Group', 'Industry', 'Sector', 'Ticker'
    ]

    active_share = float(merged['active_share_contribution'].sum())
    max_sector_dev = float(sector_df.iloc[:-1]['Difference'].abs().max())
    max_ig_dev = float(ig_df.iloc[:-1]['Difference'].abs().max())
    return sector_df, ig_df, stock_df, active_share, max_sector_dev, max_ig_dev


def generate_portfolio_excel(benchmark, label, out_path, target_stocks=60, verbose=True):
    """Solve the MILP for one benchmark snapshot and write the holdings workbook.

    Args:
        benchmark: canonical benchmark DataFrame (ticker, weight, sector,
            industry_group, industry, company_name).
        label: snapshot label used in the sheet name / titles (e.g. "2026-06").
        out_path: path to write the .xlsx portfolio to.
        target_stocks: portfolio size (default 60).
        verbose: print a plain-text summary of the result.

    Returns:
        dict: active_share, max_sector_dev, max_ig_dev, n_holdings, out_path.

    Raises:
        RuntimeError: if the MILP is infeasible for this snapshot.
    """
    optimizer = MILPActiveShareOptimizer(benchmark, target_stocks=target_stocks,
                                         sector_tolerance=2.0, ig_tolerance=2.0)
    portfolio = optimizer.optimize()
    if portfolio is None:
        raise RuntimeError(f"MILP infeasible for snapshot '{label}'")

    sector_df, ig_df, stock_df, active_share, max_sector_dev, max_ig_dev = build_sections(portfolio, benchmark)
    n_held = int((stock_df['Port. Weight'] > 0).sum())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        sheet = str(label)
        row = 0
        pd.DataFrame([{'': f'MILP Portfolio — snapshot {label} (Active Share: {active_share:.2f}%)'}]) \
            .to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
        row += 2
        pd.DataFrame([{'': 'SECTOR BREAKDOWN'}]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
        row += 1
        sector_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
        row += len(sector_df) + 2
        pd.DataFrame([{'': 'INDUSTRY GROUP BREAKDOWN'}]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
        row += 1
        ig_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
        row += len(ig_df) + 2
        pd.DataFrame([{'': f'STOCK DETAIL ({n_held} held / {len(stock_df)} total)'}]) \
            .to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
        row += 1
        stock_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Snapshot '{label}'  |  {n_held} holdings")
        print(f"Active share:            {active_share:.2f}%")
        print(f"Max sector deviation:    {max_sector_dev:.2f}%  ({'PASS' if max_sector_dev <= 2.01 else 'FAIL'})")
        print(f"Max ind-group deviation: {max_ig_dev:.2f}%  ({'PASS' if max_ig_dev <= 2.01 else 'FAIL'})")
        print(f"{'='*70}")
        print("\nTop 20 holdings by weight:")
        top = stock_df[stock_df['Port. Weight'] > 0].head(20)[['Ticker', 'Name', 'Port. Weight', 'Bench. Weight', 'Difference']]
        print(top.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        print(f"\nSaved portfolio to {out_path}")

    return {
        'active_share': active_share,
        'max_sector_dev': max_sector_dev,
        'max_ig_dev': max_ig_dev,
        'n_holdings': n_held,
        'out_path': str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate min-active-share portfolio for one snapshot')
    parser.add_argument('--sheet', required=True,
                        help='Sheet name to load (e.g. 2026-06 or a historical year like 2025)')
    parser.add_argument('--workbook', default=None,
                        help='Workbook path to read the sheet from '
                             '(default: the historical 1999-2025 workbook)')
    parser.add_argument('--target-stocks', type=int, default=60, help='Number of stocks (default: 60)')
    parser.add_argument('--out', default=None,
                        help='Output xlsx path (default: results-excel/portfolio_<sheet>.xlsx)')
    args = parser.parse_args()

    print(f"Loading benchmark snapshot from sheet '{args.sheet}'...")
    if args.workbook:
        benchmark = load_benchmark_from_xlsx(args.sheet, path=Path(args.workbook))
    else:
        benchmark = load_benchmark_from_xlsx(args.sheet)
    print(f"  {len(benchmark)} stocks | {benchmark['sector'].nunique()} sectors | "
          f"{benchmark['industry_group'].nunique()} industry groups | "
          f"weight sum {benchmark['weight'].sum():.4f}%")

    print(f"\nSolving MILP ({args.target_stocks} stocks, ±2% sector, ±2% industry_group)...")
    out_path = Path(args.out) if args.out else Path('results-excel') / f'portfolio_{args.sheet}.xlsx'
    try:
        generate_portfolio_excel(benchmark, args.sheet, out_path, target_stocks=args.target_stocks)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
