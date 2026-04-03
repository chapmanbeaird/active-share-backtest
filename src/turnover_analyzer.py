"""
Turnover analysis and tax-aware rebalancing simulation.

Computes year-over-year portfolio turnover (stocks swapped, weight changes)
and simulates the tax implications of annual rebalancing with FIFO cost basis.
Shows results with and without loss harvesting at 15% LTCG.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------------
# Data structures for FIFO cost basis tracking
# ---------------------------------------------------------------------------

@dataclass
class TaxLot:
    """A single FIFO tax lot."""
    purchase_year: int
    cost_basis: float       # dollars paid
    current_value: float    # current market value

    def apply_return(self, return_pct: float):
        """Grow lot value by a return percentage."""
        self.current_value *= (1 + return_pct / 100)

    @property
    def unrealized_gain(self) -> float:
        return self.current_value - self.cost_basis


@dataclass
class Position:
    """A position in one stock, consisting of one or more FIFO tax lots."""
    ticker: str
    lots: List[TaxLot] = field(default_factory=list)

    @property
    def current_value(self) -> float:
        return sum(lot.current_value for lot in self.lots)

    @property
    def total_cost_basis(self) -> float:
        return sum(lot.cost_basis for lot in self.lots)

    @property
    def unrealized_gain(self) -> float:
        return sum(lot.unrealized_gain for lot in self.lots)

    def sell_fifo(self, amount_to_sell: float) -> Tuple[float, float]:
        """
        Sell $amount_to_sell of market value using FIFO.

        Returns:
            (proceeds, cost_basis_of_sold)
        """
        remaining = amount_to_sell
        proceeds = 0.0
        cost_basis_sold = 0.0

        while remaining > 0.01 and self.lots:
            lot = self.lots[0]
            if lot.current_value <= remaining + 0.01:
                # Sell entire lot
                proceeds += lot.current_value
                cost_basis_sold += lot.cost_basis
                remaining -= lot.current_value
                self.lots.pop(0)
            else:
                # Sell partial lot
                fraction = remaining / lot.current_value
                proceeds += remaining
                cost_basis_sold += lot.cost_basis * fraction
                lot.cost_basis *= (1 - fraction)
                lot.current_value -= remaining
                remaining = 0.0

        return proceeds, cost_basis_sold

    def sell_all(self) -> Tuple[float, float]:
        """Sell entire position. Returns (proceeds, cost_basis_sold)."""
        proceeds = self.current_value
        cost_basis = self.total_cost_basis
        self.lots.clear()
        return proceeds, cost_basis


# ---------------------------------------------------------------------------
# Turnover computation
# ---------------------------------------------------------------------------

def compute_turnover(
    portfolios: Dict[int, pd.DataFrame],
    stock_returns: Dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute year-over-year turnover statistics.

    For each holding year transition Y -> Y+1:
    - Compare stock lists (names turnover)
    - Drift Y's weights by Y's returns, compare to Y+1 target weights

    Args:
        portfolios: {holding_year: DataFrame with ticker, weight columns}
        stock_returns: {holding_year: DataFrame with ticker, return columns}

    Returns:
        DataFrame with turnover stats per year transition
    """
    years = sorted(portfolios.keys())
    rows = []

    for i in range(len(years) - 1):
        y_prev = years[i]
        y_next = years[i + 1]

        prev_port = portfolios[y_prev]
        next_port = portfolios[y_next]

        prev_tickers = set(prev_port['ticker'])
        next_tickers = set(next_port['ticker'])

        added = next_tickers - prev_tickers
        dropped = prev_tickers - next_tickers
        held = prev_tickers & next_tickers

        # Build return lookup from stock_returns[y_prev]
        # These are returns earned during holding year y_prev
        ret_lookup = {}
        if y_prev in stock_returns and stock_returns[y_prev] is not None:
            ret_df = stock_returns[y_prev]
            # Column is 'return' (total return %)
            ret_col = 'return' if 'return' in ret_df.columns else 'total_return'
            for _, r in ret_df.iterrows():
                ret_lookup[r['ticker']] = r[ret_col]

        # Drift previous portfolio weights by returns
        prev_weights = {}
        for _, row in prev_port.iterrows():
            t = row['ticker']
            w = row['weight']
            ret = ret_lookup.get(t, 0.0)  # 0% if missing
            prev_weights[t] = w * (1 + ret / 100)

        # Normalize drifted weights to sum to 100
        total_drifted = sum(prev_weights.values())
        if total_drifted > 0:
            for t in prev_weights:
                prev_weights[t] = prev_weights[t] / total_drifted * 100

        # Next portfolio target weights
        next_weights = {}
        for _, row in next_port.iterrows():
            next_weights[row['ticker']] = row['weight']

        # Weight turnover: sum of |target - drifted| over all tickers / 2
        all_tickers = set(prev_weights.keys()) | set(next_weights.keys())
        total_abs_diff = sum(
            abs(next_weights.get(t, 0) - prev_weights.get(t, 0))
            for t in all_tickers
        )
        weight_turnover = total_abs_diff / 2  # one-way turnover

        rows.append({
            'year': y_next,
            'stocks_added': len(added),
            'stocks_dropped': len(dropped),
            'stocks_held': len(held),
            'weight_turnover_pct': round(weight_turnover, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tax-aware rebalancing simulation
# ---------------------------------------------------------------------------

class TaxSimulator:
    """
    Simulate holding a portfolio with annual rebalancing,
    tracking FIFO cost basis and tax liabilities.

    Taxes are tracked separately (not deducted from portfolio).
    """

    TAX_RATE = 0.15  # 15% long-term capital gains

    def __init__(
        self,
        portfolios: Dict[int, pd.DataFrame],
        stock_returns: Dict[int, pd.DataFrame],
        initial_value: float = 1_000_000.0,
    ):
        self.portfolios = portfolios
        self.stock_returns = stock_returns
        self.initial_value = initial_value

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run simulation with and without loss harvesting.

        Returns:
            (no_harvesting_df, with_harvesting_df)
        """
        no_harvest = self._simulate(use_loss_harvesting=False)
        with_harvest = self._simulate(use_loss_harvesting=True)
        return no_harvest, with_harvest

    def _get_return_lookup(self, year: int) -> Dict[str, float]:
        """Build ticker -> total return % lookup for a holding year."""
        lookup = {}
        if year in self.stock_returns and self.stock_returns[year] is not None:
            ret_df = self.stock_returns[year]
            ret_col = 'return' if 'return' in ret_df.columns else 'total_return'
            for _, r in ret_df.iterrows():
                lookup[r['ticker']] = r[ret_col]
        return lookup

    def _simulate(self, use_loss_harvesting: bool) -> pd.DataFrame:
        """Core simulation loop."""
        years = sorted(self.portfolios.keys())
        positions: Dict[str, Position] = {}
        loss_carryforward = 0.0
        cumulative_tax = 0.0
        rows = []

        for idx, year in enumerate(years):
            portfolio = self.portfolios[year]
            target_weights = {
                row['ticker']: row['weight']
                for _, row in portfolio.iterrows()
            }

            if idx == 0:
                # Initial purchase — no rebalancing, no gains
                for ticker, weight in target_weights.items():
                    allocation = self.initial_value * weight / 100
                    positions[ticker] = Position(
                        ticker=ticker,
                        lots=[TaxLot(
                            purchase_year=year,
                            cost_basis=allocation,
                            current_value=allocation,
                        )]
                    )

                rows.append({
                    'year': year,
                    'portfolio_value': self.initial_value,
                    'realized_gains': 0.0,
                    'realized_losses': 0.0,
                    'net_realized': 0.0,
                    'tax_liability': 0.0,
                    'cumulative_tax': 0.0,
                    'unrealized_gains': 0.0,
                    'loss_carryforward': loss_carryforward,
                    'total_trades': 0.0,
                })
                continue

            # Apply returns from previous year's holding period
            prev_year = years[idx - 1]
            ret_lookup = self._get_return_lookup(prev_year)

            for ticker, pos in positions.items():
                ret = ret_lookup.get(ticker, 0.0)
                for lot in pos.lots:
                    lot.apply_return(ret)

            # Current portfolio value after returns
            portfolio_value = sum(pos.current_value for pos in positions.values())

            # Target dollar amounts
            target_dollars = {
                t: portfolio_value * w / 100
                for t, w in target_weights.items()
            }

            # Rebalance: determine sells and buys
            realized_gains = 0.0
            realized_losses = 0.0
            total_trades = 0.0

            current_tickers = set(positions.keys())
            target_tickers = set(target_weights.keys())

            # 1. Sell positions not in new portfolio
            for ticker in current_tickers - target_tickers:
                pos = positions[ticker]
                proceeds, cost_basis = pos.sell_all()
                gain = proceeds - cost_basis
                if gain > 0:
                    realized_gains += gain
                else:
                    realized_losses += abs(gain)
                total_trades += proceeds
                del positions[ticker]

            # 2. Trim positions that are overweight
            for ticker in current_tickers & target_tickers:
                pos = positions[ticker]
                target = target_dollars[ticker]
                if pos.current_value > target + 0.01:
                    excess = pos.current_value - target
                    proceeds, cost_basis = pos.sell_fifo(excess)
                    gain = proceeds - cost_basis
                    if gain > 0:
                        realized_gains += gain
                    else:
                        realized_losses += abs(gain)
                    total_trades += proceeds

            # 3. Buy new positions and add to underweight positions
            for ticker in target_tickers:
                target = target_dollars[ticker]
                if ticker in positions:
                    current = positions[ticker].current_value
                    if target > current + 0.01:
                        buy_amount = target - current
                        positions[ticker].lots.append(TaxLot(
                            purchase_year=year,
                            cost_basis=buy_amount,
                            current_value=buy_amount,
                        ))
                        total_trades += buy_amount
                else:
                    # New position
                    positions[ticker] = Position(
                        ticker=ticker,
                        lots=[TaxLot(
                            purchase_year=year,
                            cost_basis=target,
                            current_value=target,
                        )]
                    )
                    total_trades += target

            # Tax calculation
            net_realized = realized_gains - realized_losses

            if use_loss_harvesting:
                taxable = max(0, net_realized - loss_carryforward)
                tax = taxable * self.TAX_RATE
                if net_realized >= 0:
                    loss_carryforward = max(0, loss_carryforward - net_realized)
                else:
                    loss_carryforward += abs(net_realized)
            else:
                # Without harvesting: only gains are taxed, losses don't offset
                tax = realized_gains * self.TAX_RATE

            cumulative_tax += tax

            # Unrealized gains across all positions
            unrealized = sum(pos.unrealized_gain for pos in positions.values())

            rows.append({
                'year': year,
                'portfolio_value': round(portfolio_value, 2),
                'realized_gains': round(realized_gains, 2),
                'realized_losses': round(realized_losses, 2),
                'net_realized': round(net_realized, 2),
                'tax_liability': round(tax, 2),
                'cumulative_tax': round(cumulative_tax, 2),
                'unrealized_gains': round(unrealized, 2),
                'loss_carryforward': round(loss_carryforward, 2),
                'total_trades': round(total_trades, 2),
            })

        # After the last year, apply final returns (mark to market, no rebalance)
        if len(years) >= 1:
            last_year = years[-1]
            ret_lookup = self._get_return_lookup(last_year)

            for ticker, pos in positions.items():
                ret = ret_lookup.get(ticker, 0.0)
                for lot in pos.lots:
                    lot.apply_return(ret)

            final_value = sum(pos.current_value for pos in positions.values())
            final_unrealized = sum(pos.unrealized_gain for pos in positions.values())

            rows.append({
                'year': last_year + 1,  # End-of-period value (e.g., 2026 = after 2025 returns)
                'portfolio_value': round(final_value, 2),
                'realized_gains': 0.0,
                'realized_losses': 0.0,
                'net_realized': 0.0,
                'tax_liability': 0.0,
                'cumulative_tax': round(cumulative_tax, 2),
                'unrealized_gains': round(final_unrealized, 2),
                'loss_carryforward': round(loss_carryforward, 2),
                'total_trades': 0.0,
            })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Print functions
# ---------------------------------------------------------------------------

def print_turnover_results(turnover_df: pd.DataFrame):
    """Print formatted turnover analysis."""
    if turnover_df.empty:
        return

    print(f"\n{'='*80}")
    print("TURNOVER ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Year':>6}  {'Added':>6}  {'Dropped':>8}  {'Held':>5}  {'Wt Turnover':>12}")
    print(f"{'-'*6}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*12}")

    for _, row in turnover_df.iterrows():
        print(f"{int(row['year']):>6}  {int(row['stocks_added']):>6}  "
              f"{int(row['stocks_dropped']):>8}  {int(row['stocks_held']):>5}  "
              f"{row['weight_turnover_pct']:>11.1f}%")

    print(f"{'-'*6}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*12}")
    print(f"{'Avg':>6}  {turnover_df['stocks_added'].mean():>6.1f}  "
          f"{turnover_df['stocks_dropped'].mean():>8.1f}  "
          f"{turnover_df['stocks_held'].mean():>5.1f}  "
          f"{turnover_df['weight_turnover_pct'].mean():>11.1f}%")
    print()


def print_tax_results(no_harvest_df: pd.DataFrame, with_harvest_df: pd.DataFrame):
    """Print formatted tax simulation results."""
    if no_harvest_df.empty:
        return

    initial = 1_000_000.0

    def _print_table(df: pd.DataFrame, title: str, show_carryforward: bool = False):
        print(f"\n{title}")
        print(f"{'-'*100}")

        if show_carryforward:
            print(f"{'Year':>6}  {'Port Value':>14}  {'Realized':>12}  {'Realized':>12}  "
                  f"{'Net':>12}  {'Carry Fwd':>12}  {'Tax':>12}  {'Cum Tax':>12}")
            print(f"{'':>6}  {'':>14}  {'Gains':>12}  {'Losses':>12}  "
                  f"{'Realized':>12}  {'':>12}  {'Liability':>12}  {'':>12}")
        else:
            print(f"{'Year':>6}  {'Port Value':>14}  {'Realized':>12}  {'Realized':>12}  "
                  f"{'Tax':>12}  {'Cum Tax':>12}  {'Unrealized':>12}")
            print(f"{'':>6}  {'':>14}  {'Gains':>12}  {'Losses':>12}  "
                  f"{'Liability':>12}  {'':>12}  {'Gains':>12}")

        print(f"{'-'*100}")

        for _, row in df.iterrows():
            yr = int(row['year'])
            pv = row['portfolio_value']
            rg = row['realized_gains']
            rl = row['realized_losses']
            tax = row['tax_liability']
            ctax = row['cumulative_tax']

            if show_carryforward:
                nr = row['net_realized']
                cf = row['loss_carryforward']
                print(f"{yr:>6}  ${pv:>13,.0f}  ${rg:>11,.0f}  "
                      f"(${rl:>10,.0f}) ${nr:>11,.0f}  ${cf:>11,.0f}  "
                      f"${tax:>11,.0f}  ${ctax:>11,.0f}")
            else:
                ug = row['unrealized_gains']
                print(f"{yr:>6}  ${pv:>13,.0f}  ${rg:>11,.0f}  "
                      f"(${rl:>10,.0f}) ${tax:>11,.0f}  ${ctax:>11,.0f}  "
                      f"${ug:>11,.0f}")

        print(f"{'-'*100}")

        # Totals (exclude first year which is just purchase, and last year which is mark-to-market)
        rebalance_rows = df[(df['realized_gains'] > 0) | (df['realized_losses'] > 0)]
        total_gains = rebalance_rows['realized_gains'].sum()
        total_losses = rebalance_rows['realized_losses'].sum()
        total_tax = df['cumulative_tax'].iloc[-1] if len(df) > 0 else 0

        print(f"{'Total':>6}  {'':>14}  ${total_gains:>11,.0f}  "
              f"(${total_losses:>10,.0f}) {'':>12}  ${total_tax:>11,.0f}")

    print(f"\n{'='*100}")
    print(f"TAX-AWARE REBALANCING SIMULATION (${initial:,.0f} initial investment, 15% LTCG, FIFO)")
    print(f"{'='*100}")

    _print_table(no_harvest_df, "WITHOUT LOSS HARVESTING (gains taxed independently)")
    _print_table(with_harvest_df, "WITH LOSS HARVESTING (losses offset gains, carryforward)",
                 show_carryforward=True)

    # Summary comparison
    no_harvest_total_tax = no_harvest_df['cumulative_tax'].iloc[-1]
    with_harvest_total_tax = with_harvest_df['cumulative_tax'].iloc[-1]
    savings = no_harvest_total_tax - with_harvest_total_tax
    final_value = no_harvest_df['portfolio_value'].iloc[-1]
    final_unrealized = no_harvest_df['unrealized_gains'].iloc[-1]

    print(f"\n{'='*100}")
    print("TAX SUMMARY")
    print(f"{'='*100}")
    print(f"  Final portfolio value (pre-tax):    ${final_value:>14,.0f}")
    print(f"  Embedded unrealized gains:          ${final_unrealized:>14,.0f}")
    print(f"  Total return:                       {(final_value / initial - 1) * 100:>13.1f}%")
    n_years = len(no_harvest_df) - 1  # exclude initial purchase row
    if n_years > 0:
        annualized = ((final_value / initial) ** (1 / n_years) - 1) * 100
        print(f"  Annualized return (pre-tax):        {annualized:>13.1f}%")
    print()
    print(f"  Taxes WITHOUT loss harvesting:      ${no_harvest_total_tax:>14,.0f}")
    print(f"  Taxes WITH loss harvesting:         ${with_harvest_total_tax:>14,.0f}")
    print(f"  Tax savings from harvesting:        ${savings:>14,.0f}"
          f"  ({savings / no_harvest_total_tax * 100:.1f}%)" if no_harvest_total_tax > 0 else "")
    if with_harvest_df['loss_carryforward'].iloc[-1] > 0:
        print(f"  Remaining loss carryforward:        ${with_harvest_df['loss_carryforward'].iloc[-1]:>14,.0f}")
    print()


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_turnover_and_tax_results(
    turnover_df: pd.DataFrame,
    no_harvest_df: pd.DataFrame,
    with_harvest_df: pd.DataFrame,
    results_dir: str = 'results',
    excel_dir: str = 'results-excel',
):
    """Save turnover and tax analysis to CSV and Excel."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)

    # CSV files
    turnover_df.to_csv(f'{results_dir}/turnover_analysis.csv', index=False)
    no_harvest_df.to_csv(f'{results_dir}/tax_no_harvesting.csv', index=False)
    with_harvest_df.to_csv(f'{results_dir}/tax_with_harvesting.csv', index=False)

    print(f"Turnover analysis saved to {results_dir}/turnover_analysis.csv")
    print(f"Tax simulation saved to {results_dir}/tax_no_harvesting.csv, tax_with_harvesting.csv")

    # Excel workbook
    excel_path = f'{excel_dir}/turnover_tax_analysis.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        turnover_df.to_excel(writer, sheet_name='Turnover', index=False)
        no_harvest_df.to_excel(writer, sheet_name='Tax - No Harvesting', index=False)
        with_harvest_df.to_excel(writer, sheet_name='Tax - With Harvesting', index=False)

        # Summary sheet
        initial = 1_000_000.0
        no_tax = no_harvest_df['cumulative_tax'].iloc[-1]
        with_tax = with_harvest_df['cumulative_tax'].iloc[-1]
        final_val = no_harvest_df['portfolio_value'].iloc[-1]

        rebalance_no = no_harvest_df[(no_harvest_df['realized_gains'] > 0) | (no_harvest_df['realized_losses'] > 0)]
        rebalance_with = with_harvest_df[(with_harvest_df['realized_gains'] > 0) | (with_harvest_df['realized_losses'] > 0)]

        summary = pd.DataFrame([{
            'Metric': 'Initial Investment',
            'Value': f'${initial:,.0f}',
        }, {
            'Metric': 'Final Portfolio Value',
            'Value': f'${final_val:,.0f}',
        }, {
            'Metric': 'Total Return',
            'Value': f'{(final_val / initial - 1) * 100:.1f}%',
        }, {
            'Metric': 'Total Realized Gains',
            'Value': f'${rebalance_no["realized_gains"].sum():,.0f}',
        }, {
            'Metric': 'Total Realized Losses',
            'Value': f'${rebalance_no["realized_losses"].sum():,.0f}',
        }, {
            'Metric': 'Taxes (No Harvesting)',
            'Value': f'${no_tax:,.0f}',
        }, {
            'Metric': 'Taxes (With Harvesting)',
            'Value': f'${with_tax:,.0f}',
        }, {
            'Metric': 'Tax Savings from Harvesting',
            'Value': f'${no_tax - with_tax:,.0f}',
        }, {
            'Metric': 'Avg Weight Turnover',
            'Value': f'{turnover_df["weight_turnover_pct"].mean():.1f}%',
        }, {
            'Metric': 'Avg Stocks Added/Year',
            'Value': f'{turnover_df["stocks_added"].mean():.1f}',
        }])
        summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Excel workbook saved to {excel_path}")
