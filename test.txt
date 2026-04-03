The Mechanical Source of Alpha

  Your intuition is largely correct, but let me sharpen it. The outperformance comes from three interacting mechanisms:

  1. Implicit Mega-Cap Momentum Bet (your core insight)

  The optimizer's objective function — minimize active share — creates a deterministic stock selection rule: always keep the largest-weight
  stocks, always drop the smallest. Dropping a 0.05% stock costs 0.05% of active share; dropping a 2% stock costs 2%. So the optimizer never
  drops a large stock voluntarily.

  This means you're systematically holding the winners that have already grown into the top of the index. Since benchmark weights reflect
  trailing price performance (cap-weighting), your portfolio is mechanically long recent winners. That's momentum.

  The annual rebalancing reinforces this: each December snapshot reflects which stocks have grown into top positions over the prior year. You
  rotate into rising names and out of fading ones — a momentum signal with a 12-month lookback, rebalanced annually. This is remarkably close to
   the academic momentum factor (Jegadeesh & Titman 12-1 momentum).

  2. Excess Weight Flows Upward

  This is the subtler and possibly more important mechanism. When you drop ~443 stocks, their combined weight (~40-50% of the portfolio) must be
   redistributed upward to the 60 held stocks. The cap structure determines where it goes:

  - b_i ≥ 5% (mega-caps like AAPL, MSFT): Capped at benchmark weight — absorbs zero excess
  - 1.67% ≤ b_i < 5% (large-caps): Can absorb up to 5% total — these are the primary beneficiaries
  - b_i < 1.67% (smaller held stocks): Capped at 3× — limited absorption capacity

  So the excess weight concentrates into the upper-mid tier of the S&P 500 — stocks in the ~50th-150th largest range. These are large enough to
  be held but not so large that the cap binds them. The optimizer is effectively saying: "I believe the 30th-largest stock in the S&P 500
  deserves 2-3× its benchmark weight."

  This creates a barbell: mega-caps at benchmark weight + second-tier large-caps overweighted. When large-cap leadership broadens even slightly
  (from top-10 to top-50), your portfolio captures that.

  3. Dropping the Long Tail Is a Positive-EV Bet

  The ~443 dropped stocks are predominantly the smallest S&P 500 members (0.01-0.20% each). These are the stocks that:
  - Were recently added (smaller companies entering the index at the worst time — the "index inclusion" effect works against them)
  - Are on their way out (shrinking companies about to be removed)
  - Have the highest turnover rate in the index

  Research shows that the bottom quintile of the S&P 500 by weight has historically underperformed the top quintile. Your portfolio mechanically
   avoids this drag.

  Your Momentum Hypothesis: Correct, With Nuance

  You're right that this is fundamentally a momentum strategy, but it's a specific flavor of momentum:

  - Not cross-sectional momentum (long winners, short losers across all stocks)
  - Not time-series momentum (trend-following)
  - It's cap-weight momentum — betting that the market's revealed preference for large stocks (expressed through cap weights) persists

  This is closely related to what academics call the "size premium in reverse" — you're systematically overweighting large-cap winners within an
   already large-cap universe.

  When This Strategy Fails

  Your 2009 observation is exactly right. The strategy underperforms in environments where:

  Mean-Reversion / Value Rallies

  - 2009 (+16.7% vs +22.2%): Classic snapback — beaten-down small/mid names within the S&P 500 rallied hardest
  - 2000-2002: If you had run this in the late 1990s, the portfolio would have been loaded with dot-com mega-caps (Cisco, Intel, MSFT, GE) right
   before they crashed and value/small stocks outperformed for years

  Market Broadening

  - When the rally rotates away from mega-caps to the rest of the index
  - 2022's equal-weight S&P 500 (RSP) outperformed the cap-weighted S&P 500 — your portfolio would have been on the wrong side of that
  initially, though your data shows you still outperformed (+30.8% vs +26.6%) likely because the sector constraints forced you into energy names
   that rallied

  Factor Regime Shifts

  - Deep value environments (2000-2006 value cycle): Small-value stocks dramatically outperformed large-growth
  - Anti-momentum crashes (March 2009, November 2020 vaccine announcement): Sharp, sudden rotation from momentum winners to laggards
  - Interest rate regime changes: Rising rates in 2022 punished mega-cap growth — but your sector constraints may have partially insulated you

  Is It Transferable to Future Years?

  The honest answer: it depends on whether mega-cap concentration continues.

  Bull case for the strategy:
  - The secular trend toward winner-take-all markets (network effects, scale advantages, AI/tech monopolies) favors continued mega-cap dominance
  - Index fund flows mechanically buy more of the largest stocks, creating a self-reinforcing cycle
  - As long as the S&P 500 remains top-heavy, your strategy's alpha source persists

  Bear case:
  - We're at historically extreme levels of mega-cap concentration (top 10 stocks = ~35% of S&P 500 in 2024). This is the best possible
  environment for your strategy. Mean-reversion from here would hurt
  - Antitrust, regulation, or competitive disruption of mega-caps would reverse the trend
  - A sustained value/small-cap cycle (like 2000-2006) would generate persistent underperformance
  - The +1.28% annualized alpha over 26 years is real but modest — a few bad years of mean-reversion could erase it

  The Beneficial Constraint You Identified

  Your last point is the most interesting. The forced concentration into 60 stocks creates a structural advantage that's distinct from momentum:

  By forcing the optimizer to choose, it creates a portfolio that implicitly bets on quality/stability within each sector. The sector/industry
  constraints ensure you hold stocks in every corner of the index, but you hold only the dominant firm in each industry group. These tend to be
  the most profitable, most stable, highest-quality companies — which itself is a factor (the "quality premium") that has generated alpha
  historically.

  So the full factor decomposition is likely: momentum + quality + anti-small-cap, all emerging naturally from the mathematical structure of the
   optimization, not from any explicit factor model. That's intellectually elegant but also means the strategy is exposed to factor reversals —
  it's not a free lunch, it's a factor bet disguised as an index optimization.