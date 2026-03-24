"""
MILP Active Share Minimizer

Finds the provably minimum active share portfolio of exactly K stocks
from a benchmark universe, subject to Sector and Industry_Group weight
constraints (±tolerance).

Uses scipy.optimize.milp with the HiGHS solver (no extra dependencies).
"""

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


class MILPActiveShareOptimizer:
    """
    Construct the minimum active share portfolio using Mixed-Integer Linear Programming.

    Decision variables (3N total, N = number of benchmark stocks):
        x_i ∈ {0,1}  — stock i selected?
        w_i ≥ 0       — portfolio weight of stock i (%)
        d_i ≥ 0       — |w_i - b_i| (absolute deviation from benchmark)

    Objective: minimize 0.5 × Σ d_i  (= active share)

    Constraints:
        Σ x_i = target_stocks
        Σ w_i = 100
        w_i ≤ M × x_i              (Big-M: unselected stocks have 0 weight)
        d_i ≥ w_i - b_i             (absolute value linearization)
        d_i ≥ b_i - w_i
        |Σ w_i(sector s) - B_s| ≤ sector_tolerance       for each sector s
        |Σ w_i(group g) - B_g| ≤ ig_tolerance             for each industry group g
    """

    def __init__(self, benchmark_df: pd.DataFrame, target_stocks: int = 60,
                 sector_tolerance: float = 2.0, ig_tolerance: float = 2.0):
        """
        Args:
            benchmark_df: DataFrame with columns ['ticker', 'weight', 'sector', 'industry_group']
                          Weights in percentage, summing to ~100.
            target_stocks: Exactly how many stocks to hold.
            sector_tolerance: Max absolute deviation per GICS Sector (percentage points).
            ig_tolerance: Max absolute deviation per Industry_Group (percentage points).
        """
        self.benchmark = benchmark_df.copy().reset_index(drop=True)
        self.target_stocks = target_stocks
        self.sector_tolerance = sector_tolerance
        self.ig_tolerance = ig_tolerance
        self.N = len(self.benchmark)
        self.b = self.benchmark['weight'].values.astype(float)

        # Normalize benchmark to exactly 100
        bsum = self.b.sum()
        if not (99.9 <= bsum <= 100.1):
            self.b = self.b / bsum * 100

    def optimize(self) -> pd.DataFrame:
        """
        Solve the MILP and return the optimal portfolio.

        Returns:
            DataFrame with columns ['ticker', 'weight', 'sector', 'industry_group', 'industry']
            or None if the problem is infeasible.
        """
        result = self._build_and_solve()

        if result is None or not result.success:
            status = result.message if result else "solver returned None"
            print(f"MILP optimization failed: {status}")
            return None

        portfolio = self._extract_solution(result)

        # Verify constraints
        self._verify(portfolio)

        return portfolio

    def _build_and_solve(self):
        """Build the MILP and solve it with HiGHS."""
        N = self.N
        b = self.b

        # Variable layout: [x_0..x_{N-1}, w_0..w_{N-1}, d_0..d_{N-1}]
        # Indices
        x_start, w_start, d_start = 0, N, 2 * N
        n_vars = 3 * N

        # --- Objective: minimize 0.5 * Σ d_i ---
        c = np.zeros(n_vars)
        c[d_start:d_start + N] = 0.5

        # --- Bounds ---
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)
        # x_i ∈ [0, 1]
        ub[x_start:x_start + N] = 1.0
        # w_i ∈ [0, min(3×bench, 5%)] for bench<5%, or [0, bench] for bench≥5%
        for i in range(N):
            ub[w_start + i] = min(3.0 * b[i], max(5.0, b[i]))
        # d_i ∈ [0, 100]
        ub[d_start:d_start + N] = 100.0
        bounds = Bounds(lb, ub)

        # --- Integrality: first N are binary ---
        integrality = np.zeros(n_vars, dtype=int)
        integrality[x_start:x_start + N] = 1

        # --- Build constraint matrix rows ---
        # We'll collect all constraints as (row_coeffs, lb, ub) and then stack
        A_rows = []
        con_lb = []
        con_ub = []

        def add_constraint(coeffs, lower, upper):
            """Add a single constraint row."""
            A_rows.append(coeffs)
            con_lb.append(lower)
            con_ub.append(upper)

        # Constraint 1: Σ x_i = target_stocks
        row = np.zeros(n_vars)
        row[x_start:x_start + N] = 1.0
        add_constraint(row, self.target_stocks, self.target_stocks)

        # Constraint 2: Σ w_i = 100
        row = np.zeros(n_vars)
        row[w_start:w_start + N] = 1.0
        add_constraint(row, 100.0, 100.0)

        # Constraint 3: w_i - M * x_i ≤ 0  (Big-M linking)
        M = 100.0
        for i in range(N):
            row = np.zeros(n_vars)
            row[w_start + i] = 1.0
            row[x_start + i] = -M
            add_constraint(row, -np.inf, 0.0)

        # Constraint 4: w_i - d_i ≤ b_i  (d_i ≥ w_i - b_i)
        for i in range(N):
            row = np.zeros(n_vars)
            row[w_start + i] = 1.0
            row[d_start + i] = -1.0
            add_constraint(row, -np.inf, b[i])

        # Constraint 5: -w_i - d_i ≤ -b_i  (d_i ≥ b_i - w_i)
        for i in range(N):
            row = np.zeros(n_vars)
            row[w_start + i] = -1.0
            row[d_start + i] = -1.0
            add_constraint(row, -np.inf, -b[i])

        # Constraint 6: Sector constraints — B_s - tol ≤ Σ w_i(sector s) ≤ B_s + tol
        sector_groups = self.benchmark.groupby('sector')
        for sector, group in sector_groups:
            B_s = group['weight'].sum()
            row = np.zeros(n_vars)
            for idx in group.index:
                row[w_start + idx] = 1.0
            add_constraint(row, max(0.0, B_s - self.sector_tolerance), B_s + self.sector_tolerance)

        # Constraint 7: Industry_Group constraints — B_g - tol ≤ Σ w_i(ig g) ≤ B_g + tol
        ig_groups = self.benchmark.groupby('industry_group')
        for ig, group in ig_groups:
            B_g = group['weight'].sum()
            row = np.zeros(n_vars)
            for idx in group.index:
                row[w_start + idx] = 1.0
            add_constraint(row, max(0.0, B_g - self.ig_tolerance), B_g + self.ig_tolerance)

        # Stack into matrix
        A = np.array(A_rows)
        con_lb = np.array(con_lb)
        con_ub = np.array(con_ub)

        constraints = LinearConstraint(A, con_lb, con_ub)

        # Solve
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 60, "presolve": True}
        )

        return result

    def _extract_solution(self, result) -> pd.DataFrame:
        """Convert scipy result to a portfolio DataFrame."""
        N = self.N
        x_vals = result.x[:N]
        w_vals = result.x[N:2 * N]

        # Round binary variables
        selected = x_vals > 0.5

        # Zero out unselected weights (handle floating-point noise)
        w_vals[~selected] = 0.0
        w_vals = np.maximum(w_vals, 0.0)

        # Renormalize to exactly 100%
        w_sum = w_vals.sum()
        if w_sum > 0:
            w_vals = w_vals / w_sum * 100.0

        # Build portfolio from selected stocks
        portfolio = self.benchmark.loc[selected].copy()
        portfolio['weight'] = w_vals[selected]

        cols = ['ticker', 'weight', 'sector', 'industry_group']
        if 'industry' in portfolio.columns:
            cols.append('industry')

        return portfolio[cols].reset_index(drop=True)

    def _verify(self, portfolio: pd.DataFrame):
        """Print verification of all constraints."""
        n_stocks = len(portfolio)
        w_sum = portfolio['weight'].sum()

        print(f"\nMILP Solution Verification:")
        print(f"  Stocks: {n_stocks} (target: {self.target_stocks})")
        print(f"  Weight sum: {w_sum:.4f}%")

        # Active share
        active_share = self._calculate_active_share(portfolio)
        print(f"  Active share: {active_share:.2f}%")

        # Sector deviations
        bench_sectors = self.benchmark.groupby('sector')['weight'].sum()
        port_sectors = portfolio.groupby('sector')['weight'].sum()
        max_sector_dev = 0.0
        for sector in bench_sectors.index:
            dev = abs(port_sectors.get(sector, 0.0) - bench_sectors[sector])
            max_sector_dev = max(max_sector_dev, dev)
        print(f"  Max sector deviation: {max_sector_dev:.3f}% (limit: {self.sector_tolerance}%)")

        # Industry_Group deviations
        bench_ig = self.benchmark.groupby('industry_group')['weight'].sum()
        port_ig = portfolio.groupby('industry_group')['weight'].sum()
        max_ig_dev = 0.0
        for ig in bench_ig.index:
            dev = abs(port_ig.get(ig, 0.0) - bench_ig[ig])
            max_ig_dev = max(max_ig_dev, dev)
        print(f"  Max industry_group deviation: {max_ig_dev:.3f}% (limit: {self.ig_tolerance}%)")

        # Check pass/fail
        passed = (n_stocks == self.target_stocks and
                  abs(w_sum - 100.0) < 0.01 and
                  max_sector_dev <= self.sector_tolerance + 0.01 and
                  max_ig_dev <= self.ig_tolerance + 0.01)
        print(f"  Result: {'ALL CONSTRAINTS SATISFIED' if passed else 'CONSTRAINT VIOLATION'}")

    def _calculate_active_share(self, portfolio: pd.DataFrame) -> float:
        """Calculate active share between portfolio and benchmark."""
        merged = pd.merge(
            portfolio[['ticker', 'weight']].rename(columns={'weight': 'port_weight'}),
            self.benchmark[['ticker', 'weight']].rename(columns={'weight': 'bench_weight'}),
            on='ticker',
            how='outer'
        ).fillna(0)
        return 0.5 * np.abs(merged['port_weight'] - merged['bench_weight']).sum()


if __name__ == '__main__':
    from data_loader import load_benchmark_from_xlsx

    year = 2024
    print(f"Loading benchmark for {year}...")
    benchmark = load_benchmark_from_xlsx(year)
    print(f"Benchmark: {len(benchmark)} stocks, {benchmark['sector'].nunique()} sectors, "
          f"{benchmark['industry_group'].nunique()} industry groups")

    print(f"\nRunning MILP optimizer (60 stocks, ±2% sector, ±2% industry_group)...")
    optimizer = MILPActiveShareOptimizer(benchmark, target_stocks=60,
                                          sector_tolerance=2.0, ig_tolerance=2.0)
    portfolio = optimizer.optimize()

    if portfolio is not None:
        print(f"\n--- Portfolio (top 20 by weight) ---")
        print(portfolio.nlargest(20, 'weight').to_string(index=False))

        print(f"\n--- Sector Breakdown ---")
        bench_s = benchmark.groupby('sector')['weight'].sum()
        port_s = portfolio.groupby('sector')['weight'].sum()
        for sector in bench_s.sort_values(ascending=False).index:
            bw = bench_s[sector]
            pw = port_s.get(sector, 0.0)
            print(f"  {sector:<30} bench={bw:6.2f}%  port={pw:6.2f}%  dev={pw - bw:+.2f}%")

        print(f"\n--- Industry_Group Breakdown ---")
        bench_ig = benchmark.groupby('industry_group')['weight'].sum()
        port_ig = portfolio.groupby('industry_group')['weight'].sum()
        for ig in bench_ig.sort_values(ascending=False).index:
            bw = bench_ig[ig]
            pw = port_ig.get(ig, 0.0)
            print(f"  {ig:<30} bench={bw:6.2f}%  port={pw:6.2f}%  dev={pw - bw:+.2f}%")
