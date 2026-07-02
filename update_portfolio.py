#!/usr/bin/env python3
"""
Quarterly one-command workflow: turn a raw FactSet S&P 500 export into the
recommended 60-stock minimum-active-share portfolio.

Typical use (non-technical): drop the FactSet file into data/incoming/ and run

    python3 update_portfolio.py

or double-click "Update Portfolio.command".

What it does:
  1. Finds the FactSet export (in data/incoming/, or via --file).
  2. Normalizes it to the house format (converts tickers, recovers the
     industry_group classification by ticker from history).
  3. If any stock is new/unclassified, STOPS and tells you which ones to add to
     data/manual_classifications.csv (it never guesses).
  4. Saves the cleaned snapshot into data/current-snapshots.xlsx.
  5. Solves the optimizer and writes results-excel/portfolio_<date>.xlsx.
  6. Files the processed export away in data/processed/.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from snapshot_normalizer import (
    normalize_factset_snapshot, run_qc, SnapshotError, RAW_SHEET_DEFAULT,
    append_manual_classifications,
)
from snapshot_store import write_snapshot_sheet, SNAPSHOT_PATH
from generate_portfolio import generate_portfolio_excel

PROJECT_ROOT = Path(__file__).parent
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results-excel"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NEEDS_CLASSIFICATION = 2


def _bar():
    print("=" * 72)


def find_incoming_file(incoming_dir: Path) -> Path:
    """Return the single .xlsx in the incoming folder, or exit with guidance."""
    if not incoming_dir.exists():
        print(f"ERROR: the drop folder does not exist: {incoming_dir}")
        print("Create it and put your FactSet export inside, then run again.")
        sys.exit(EXIT_ERROR)
    # Ignore Excel lock/temp files like ~$whatever.xlsx
    files = [p for p in sorted(incoming_dir.glob("*.xlsx")) if not p.name.startswith("~$")]
    if not files:
        print(f"ERROR: no .xlsx file found in {incoming_dir}")
        print("Put your FactSet 'Active Share' export in that folder and run again.")
        sys.exit(EXIT_ERROR)
    if len(files) > 1:
        print(f"ERROR: found {len(files)} files in {incoming_dir}; expected exactly one:")
        for p in files:
            print(f"   - {p.name}")
        print("Leave only the newest export in the folder and run again.")
        sys.exit(EXIT_ERROR)
    return files[0]


def print_unclassified_stop(report, csv_path: Path) -> None:
    _bar()
    print(f"STOPPED — {len(report.unclassified)} stock(s) are new and need a category")
    _bar()
    print("These companies are in the index for the first time, so we don't yet")
    print("know which industry group they belong to. Add a line for each one to:")
    print(f"\n   {csv_path}\n")
    print("Ticker   Company")
    print("-------  ---------------------------------")
    for u in report.unclassified:
        print(f"{u['ticker']:<7}  {u['company_name']}")
    print("\nExample line to add (pick the best-fitting group from the list below):")
    print("   VRT,Producer Manufacturing,Electrical Products")
    print("\nValid industry groups (use one of these exactly):")
    for g in report.valid_igroups:
        print(f"   - {g}")
    print("\nThen run this again. Nothing was saved.")
    _bar()


_EXIT_WORDS = {"q", "quit", "exit"}


def _resolve_choice(raw, valid):
    """Map a typed answer to exactly one of the 20 groups.

    Returns the canonical group name, the string 'EXIT' to quit, or None if the
    input doesn't match a listed number or an exact group name.
    """
    raw = raw.strip()
    if raw.lower() in _EXIT_WORDS:
        return "EXIT"
    if raw.isdigit():
        i = int(raw)
        return valid[i - 1] if 1 <= i <= len(valid) else None
    for g in valid:  # exact name match (case-insensitive); return canonical spelling
        if raw.lower() == g.lower():
            return g
    return None


def classify_interactively(report, canonical):
    """Ask the user, in the terminal, to assign an industry group to each new
    stock. The GICS sector is already set from the file — only the group is
    asked. Input must match one of the 20 groups exactly; 'q' quits.

    Returns {ticker: industry_group} and fills the choices into `canonical`.
    """
    valid = report.valid_igroups
    n = len(report.unclassified)
    _bar()
    print(f"{n} new stock(s) need an industry group.")
    print("(The GICS sector is already set from the file — only the industry group is missing.)")
    _bar()
    chosen = {}
    for i, u in enumerate(report.unclassified, 1):
        ticker, name = u["ticker"], u["company_name"]
        sec = canonical.loc[canonical["ticker"] == ticker, "sector"]
        sector = sec.iloc[0] if len(sec) else "?"
        print(f"\n[{i}/{n}]  {ticker} — {name}")
        print(f"        Sector (already set): {sector}")
        print("        Choose its industry group (one of these 20):")
        for j, g in enumerate(valid, 1):
            print(f"          {j:>2}. {g}")
        while True:
            try:
                raw = input(f"        Enter a number 1-{len(valid)}, the exact group name, or 'q' to quit: ")
            except (EOFError, KeyboardInterrupt):
                raw = "q"
            choice = _resolve_choice(raw, valid)
            if choice == "EXIT":
                print("\nExited — no portfolio was built and nothing was saved.")
                sys.exit(EXIT_NEEDS_CLASSIFICATION)
            if choice:
                break
            print("        Not one of the 20 groups — enter a listed number or the exact name (or 'q' to quit).")
        canonical.loc[canonical["ticker"] == ticker, "industry_group"] = choice
        chosen[ticker] = choice
        print(f"        -> {ticker} = {choice}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Build the current portfolio from a FactSet export")
    parser.add_argument("--file", default=None, help="Path to the FactSet export (default: the one in data/incoming/)")
    parser.add_argument("--incoming-dir", default=str(INCOMING_DIR), help="Folder to look in for the export")
    parser.add_argument("--sheet", default=RAW_SHEET_DEFAULT, help=f"Sheet name in the export (default: '{RAW_SHEET_DEFAULT}')")
    parser.add_argument("--as-of", default=None, help="Override the snapshot label, e.g. 2026-06 (if auto-detect fails)")
    parser.add_argument("--target-stocks", type=int, default=60, help="Portfolio size (default: 60)")
    parser.add_argument("--allow-unclassified", action="store_true",
                        help="Advanced: proceed even with unclassified stocks (they are bucketed as 'Miscellaneous')")
    parser.add_argument("--out", default=None, help="Output portfolio xlsx path")
    args = parser.parse_args()

    incoming_dir = Path(args.incoming_dir)
    src_file = Path(args.file) if args.file else find_incoming_file(incoming_dir)
    if not src_file.exists():
        print(f"ERROR: file not found: {src_file}")
        sys.exit(EXIT_ERROR)

    print(f"Reading: {src_file.name}")
    try:
        canonical, report = normalize_factset_snapshot(
            src_file, sheet=args.sheet, as_of=args.as_of)
        run_qc(canonical, report)
    except SnapshotError as e:
        print(f"\nERROR: {e}")
        sys.exit(EXIT_ERROR)

    print(f"Found {report.n_constituents} S&P 500 stocks as of {report.as_of_label}.")
    print(f"Classified: {report.n_from_inheritance} matched from history, "
          f"{report.n_from_manual} from your manual list, "
          f"{report.n_from_native} from the file itself.")
    for w in report.warnings:
        print(f"  NOTE: {w}")

    # --- New/unclassified stocks: ask interactively, bucket, or halt ---
    if report.unclassified:
        if args.allow_unclassified:
            tickers = [u["ticker"] for u in report.unclassified]
            print(f"  WARNING: bucketing {len(tickers)} unclassified stock(s) as 'Miscellaneous': {tickers}")
            canonical["industry_group"] = canonical["industry_group"].fillna("Miscellaneous")
        elif sys.stdin.isatty():
            chosen = classify_interactively(report, canonical)
            append_manual_classifications([(t, g, "") for t, g in chosen.items()])
            print(f"\nSaved {len(chosen)} classification(s) to data/manual_classifications.csv "
                  "(so you won't be asked again).")
        else:
            # Not an interactive terminal (e.g. automation) — can't prompt; stop safely.
            print_unclassified_stop(report, Path("data/manual_classifications.csv"))
            sys.exit(EXIT_NEEDS_CLASSIFICATION)

    label = report.as_of_label

    # --- Save the cleaned snapshot ---
    snap_path = write_snapshot_sheet(canonical, label)
    print(f"Saved cleaned benchmark: {snap_path} (sheet {label})")

    # --- Build the portfolio ---
    print(f"\nBuilding your {args.target_stocks}-stock portfolio...")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"portfolio_{label}.xlsx"
    try:
        result = generate_portfolio_excel(canonical, label, out_path,
                                          target_stocks=args.target_stocks, verbose=False)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(EXIT_ERROR)

    sector_ok = result["max_sector_dev"] <= 2.01
    ig_ok = result["max_ig_dev"] <= 2.01
    print()
    _bar()
    print(f"DONE — portfolio for {label}")
    _bar()
    print(f"Holdings:        {result['n_holdings']} stocks")
    print(f"Active share:    {result['active_share']:.2f}%")
    print(f"Sector check:    {'PASS' if sector_ok else 'FAIL'} "
          f"(max deviation {result['max_sector_dev']:.2f}% vs 2.00% limit)")
    print(f"Ind-group check: {'PASS' if ig_ok else 'FAIL'} "
          f"(max deviation {result['max_ig_dev']:.2f}% vs 2.00% limit)")
    print(f"\nPortfolio saved: {result['out_path']}")
    _bar()

    # --- Archive the processed input (only if it came from the drop folder) ---
    try:
        if src_file.resolve().parent == incoming_dir.resolve():
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            dest = PROCESSED_DIR / src_file.name
            shutil.move(str(src_file), str(dest))
            print(f"Filed the export away in: {dest}")
    except Exception as e:
        print(f"  NOTE: could not archive the input file ({e}). It's still in data/incoming/.")

    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
