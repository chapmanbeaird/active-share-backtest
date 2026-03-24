"""
Data loader for S&P 500 benchmark snapshots from cleaned FactSet Excel file.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
XLSX_PATH = PROJECT_ROOT / "data" / "1999-2025-S&P500-cleaned.xlsx"


def load_benchmark_from_xlsx(year: int) -> pd.DataFrame:
    """
    Load S&P 500 benchmark snapshot for a given year from the cleaned xlsx file.

    Args:
        year: Year to load (1999-2025)

    Returns:
        DataFrame with columns ['ticker', 'weight', 'sector', 'industry_group', 'industry', 'company_name']
        Weights normalized to sum to exactly 100%.
    """
    df = pd.read_excel(XLSX_PATH, sheet_name=str(year), header=6)

    # Rename columns
    col_map = {
        df.columns[0]: 'company_name',
        'Port. Weight': 'weight',
        'Ticker': 'ticker',
        'Sector': 'sector',
        'Industry_Group': 'industry_group',
        'Industry': 'industry',
    }
    df = df.rename(columns=col_map)

    # Keep only valid stock rows (have a ticker and aren't artifacts)
    df = df[df['ticker'].notna() & (df['sector'] != '--')]

    # Select and clean columns
    df = df[['ticker', 'weight', 'sector', 'industry_group', 'industry', 'company_name']].copy()
    df['ticker'] = df['ticker'].astype(str).str.strip()
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df = df.dropna(subset=['weight'])

    # Normalize weights to exactly 100%
    weight_sum = df['weight'].sum()
    df['weight'] = (df['weight'] / weight_sum) * 100

    df = df.reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = load_benchmark_from_xlsx(2024)
    print(f"Stocks: {len(df)}")
    print(f"Weight sum: {df['weight'].sum():.4f}%")
    print(f"Sectors: {df['sector'].nunique()}")
    print(f"Industry_Groups: {df['industry_group'].nunique()}")
    print(f"\nTop 10 by weight:")
    print(df.nlargest(10, 'weight')[['ticker', 'weight', 'sector', 'industry_group']].to_string(index=False))
