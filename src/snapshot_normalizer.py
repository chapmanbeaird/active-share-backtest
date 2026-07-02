"""
Normalize a raw FactSet "Active Share" S&P 500 export into the house benchmark
format used by the optimizer.

The quarterly FactSet export uses a DIFFERENT layout and classification system
than the frozen historical workbook:

  - Its GICS `sector` names match the historical `sector` exactly (11 GICS
    sectors) -> used as-is.
  - Its `JasonGICSIndustryGroup` (25 GICS groups) is a completely different
    taxonomy from the historical `industry_group` (20 FactSet "Economic
    Sectors"). It CANNOT be used. The optimizer constrains only `sector` and
    `industry_group` (`industry` is display-only), so `industry_group` is
    recovered per-ticker by inheritance from the historical workbook.
  - Tickers look like `GOOGL-US`, `BRK.B-US`; converted to house format
    (strip `-US`, keep share-class dots) so they match historical tickers and
    stay compatible with `ticker_to_fmp()`.

This module never guesses: a ticker with no known `industry_group` is reported
as `unclassified` for a human to fill in `data/manual_classifications.csv`.
"""

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_loader import load_benchmark_from_xlsx

PROJECT_ROOT = Path(__file__).parent.parent
HIST_XLSX_PATH = PROJECT_ROOT / "data" / "1999-2025-S&P500-cleaned.xlsx"
MANUAL_CSV_PATH = PROJECT_ROOT / "data" / "manual_classifications.csv"

RAW_SHEET_DEFAULT = "Active Share"

# The 11 GICS sectors, as spelled in both the historical data and the FactSet
# export. The new file's `sector` values must be a subset of this set.
KNOWN_SECTORS = {
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
}

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_DATE_RE = re.compile(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})")

CANONICAL_COLUMNS = ["ticker", "weight", "sector", "industry_group", "industry", "company_name"]


class SnapshotError(Exception):
    """Raised when the raw file is structurally wrong (bad sheet, missing
    column, unexpected sector, duplicate tickers, absurd constituent count)."""


@dataclass
class NormalizationReport:
    as_of_label: str = ""
    as_of_date: Optional[datetime] = None
    n_raw_rows: int = 0
    n_constituents: int = 0
    n_from_native: int = 0
    n_from_inheritance: int = 0
    n_from_manual: int = 0
    unclassified: List[dict] = field(default_factory=list)
    raw_weight_sum: float = 0.0
    sector_count: int = 0
    igroup_count: int = 0
    valid_igroups: List[str] = field(default_factory=list)
    # {industry_group -> sorted list of industries seen for it in history}, so the
    # interactive classifier can offer real industry choices for a chosen group.
    industries_by_group: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ticker conversion
# ---------------------------------------------------------------------------

def v1symbol_to_house(sym) -> str:
    """Convert a FactSet V1Symbol to the house ticker format.

    Strips the trailing `-US` country suffix and upper-cases, preserving
    share-class dots: 'GOOGL-US' -> 'GOOGL', 'BRK.B-US' -> 'BRK.B'.
    """
    s = str(sym).strip().upper()
    if s.endswith("-US"):
        s = s[:-3]
    return s


# ---------------------------------------------------------------------------
# Date detection
# ---------------------------------------------------------------------------

def detect_snapshot_date(path, sheet: str = RAW_SHEET_DEFAULT) -> Tuple[str, datetime]:
    """Read the `DD-MON-YYYY` as-of date from the export header block.

    In the sample the date sits at raw cell [row 5, col 0] ("30-JUN-2026").
    We scan the first ~10 rows of column 0 for a DD-MON-YYYY token so a small
    layout shift doesn't break detection.

    Returns (label, date) where label is like "2026-06".
    Raises SnapshotError if no date is found (caller may override with --as-of).
    """
    head = pd.read_excel(path, sheet_name=sheet, header=None, nrows=10)
    for val in head.iloc[:, 0].tolist():
        if val is None:
            continue
        m = _DATE_RE.search(str(val))
        if m:
            day, mon, year = int(m.group(1)), m.group(2).upper(), int(m.group(3))
            if mon in _MONTHS:
                month = _MONTHS[mon]
                return f"{year}-{month:02d}", datetime(year, month, day)
    raise SnapshotError(
        "Could not find a snapshot date (DD-MON-YYYY) in the file header. "
        "Pass one explicitly, e.g. --as-of 2026-06."
    )


# ---------------------------------------------------------------------------
# Historical taxonomy + inheritance
# ---------------------------------------------------------------------------

def build_inheritance_map(hist_path=HIST_XLSX_PATH) -> Dict[str, Tuple[str, str]]:
    """Build {house_ticker -> (industry_group, industry)} from every historical
    sheet. Later (more recent) years overwrite earlier ones, so a ticker gets
    its most recent known classification.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    xl = pd.ExcelFile(hist_path)
    for sheet in xl.sheet_names:  # 1999..2025, ascending
        df = load_benchmark_from_xlsx(sheet, path=hist_path)
        for _, r in df.iterrows():
            ig = r["industry_group"]
            if pd.isna(ig):
                continue
            tkr = str(r["ticker"]).strip().upper()
            ind = r["industry"] if not pd.isna(r["industry"]) else ""
            mapping[tkr] = (str(ig), str(ind))
    return mapping


def valid_industry_groups(inheritance_map: Dict[str, Tuple[str, str]]) -> set:
    """The canonical set of allowed `industry_group` values (derived from
    history so it can never drift from the real data)."""
    return {ig for ig, _ind in inheritance_map.values()}


def _valid_industries(inheritance_map: Dict[str, Tuple[str, str]]) -> set:
    return {ind for _ig, ind in inheritance_map.values() if ind}


def industries_for_groups(inheritance_map: Dict[str, Tuple[str, str]]
                          ) -> Dict[str, List[str]]:
    """{industry_group -> sorted list of industries seen for it in history}.

    Used to offer the interactive classifier a realistic set of industry choices
    for whichever group the user picks (industry is display-only, so a custom
    name is still allowed)."""
    out: Dict[str, set] = {}
    for ig, ind in inheritance_map.values():
        if not ig or not ind:
            continue
        out.setdefault(str(ig), set()).add(str(ind))
    return {g: sorted(inds) for g, inds in out.items()}


# ---------------------------------------------------------------------------
# Manual classifications
# ---------------------------------------------------------------------------

def load_manual_classifications(csv_path=MANUAL_CSV_PATH,
                                valid_igroups: Optional[set] = None,
                                warnings: Optional[List[str]] = None
                                ) -> Dict[str, Tuple[str, str]]:
    """Read data/manual_classifications.csv -> {ticker -> (industry_group, industry)}.

    Columns: ticker, industry_group, industry (industry optional). A missing
    file is fine (returns {}). Rows whose industry_group is not one of the valid
    20 are skipped with a warning (so a typo can't sneak a bad bucket through).
    """
    csv_path = Path(csv_path)
    result: Dict[str, Tuple[str, str]] = {}
    if not csv_path.exists():
        return result
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = [(f or "").strip().lower() for f in (reader.fieldnames or [])]
        if "ticker" not in header:
            raise SnapshotError(
                f"{csv_path.name} is missing its header row. The first line must be "
                f"'ticker,industry_group,industry' (found: {reader.fieldnames!r}). "
                f"Restore the header and re-run."
            )
        for raw in reader:
            # A row with more columns than the header puts the extras in a list
            # under the None key; coerce any non-string cell to a plain string so
            # a malformed row degrades gracefully instead of raising.
            row = {}
            for k, v in raw.items():
                if isinstance(v, list):
                    v = v[0] if v else ""
                row[(k or "").strip().lower()] = (v or "").strip()
            tkr = row.get("ticker", "").upper()
            ig = row.get("industry_group", "")
            ind = row.get("industry", "")
            if not tkr or not ig:
                continue
            if valid_igroups is not None and ig not in valid_igroups and warnings is not None:
                warnings.append(
                    f"manual_classifications.csv: '{tkr}' has industry_group "
                    f"'{ig}' which is not one of the valid 20 groups — ignored."
                )
                continue
            result[tkr] = (ig, ind)
    return result


def append_manual_classifications(entries, csv_path=MANUAL_CSV_PATH) -> None:
    """Append (ticker, industry_group, industry) rows to the manual CSV for any
    tickers not already present, so a choice made once is remembered next run.
    Creates the file (with a header) if it doesn't exist.
    """
    csv_path = Path(csv_path)
    existing = set()
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                t = (row.get("ticker") or "").strip().upper()
                if t:
                    existing.add(t)
    new_rows = [(str(t).strip().upper(), ig, ind) for (t, ig, ind) in entries
                if str(t).strip().upper() not in existing]
    if not new_rows:
        return
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["ticker", "industry_group", "industry"])
        for t, ig, ind in new_rows:
            writer.writerow([t, ig, ind])


# ---------------------------------------------------------------------------
# Native-column detection (future-proofing)
# ---------------------------------------------------------------------------

def detect_native_columns(raw_df: pd.DataFrame, valid_igroups: set,
                          valid_industries: set) -> Tuple[Optional[str], Optional[str]]:
    """If a future export already carries native FactSet Economic Sector /
    Industry columns, find them so we can prefer them over inheritance.

    Conservative: only accept a column whose non-null values are a SUBSET of the
    known historical taxonomy and cover the vast majority of rows. Returns
    (economic_sector_col, industry_col), either may be None.
    """
    skip = {"V1Symbol", "JasonGICSSector", "JasonGICSIndustryGroup",
            "Bench. Ending Weight", "Port. Ending Weight", "Active Share",
            raw_df.columns[0]}
    n = len(raw_df)
    ig_col = ind_col = None
    for col in raw_df.columns:
        if col in skip:
            continue
        vals = raw_df[col].dropna().astype(str)
        if len(vals) < 0.9 * n or vals.empty:
            continue
        uniq = set(vals.unique())
        if ig_col is None and uniq <= valid_igroups:
            ig_col = col
        elif ind_col is None and uniq <= valid_industries:
            ind_col = col
    return ig_col, ind_col


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def normalize_factset_snapshot(raw_path, sheet: str = RAW_SHEET_DEFAULT,
                               hist_path=HIST_XLSX_PATH,
                               manual_csv=MANUAL_CSV_PATH,
                               as_of: Optional[str] = None
                               ) -> Tuple[pd.DataFrame, NormalizationReport]:
    """Normalize a raw FactSet export into the canonical benchmark DataFrame.

    Returns (canonical_df, report). Does NOT raise on unclassified tickers —
    they are listed in report.unclassified for the caller to handle. Raises
    SnapshotError only on structural problems.
    """
    report = NormalizationReport()

    # --- Date ---
    if as_of:
        report.as_of_label = as_of
    else:
        report.as_of_label, report.as_of_date = detect_snapshot_date(raw_path, sheet)

    # --- Read raw table (header on row index 6) ---
    raw = pd.read_excel(raw_path, sheet_name=sheet, header=6)
    report.n_raw_rows = len(raw)

    required = {"V1Symbol", "Port. Ending Weight", "JasonGICSSector"}
    missing = required - set(raw.columns)
    if missing:
        raise SnapshotError(
            f"Export is missing expected column(s): {sorted(missing)}. "
            f"Found columns: {list(raw.columns)}. Is this the 'Active Share' report?"
        )

    name_col = raw.columns[0]

    # --- Taxonomy + lookups ---
    inheritance = build_inheritance_map(hist_path)
    valid_igroups = valid_industry_groups(inheritance)
    valid_inds = _valid_industries(inheritance)
    report.valid_igroups = sorted(valid_igroups)
    report.industries_by_group = industries_for_groups(inheritance)
    manual = load_manual_classifications(manual_csv, valid_igroups, report.warnings)
    native_ig_col, native_ind_col = detect_native_columns(raw, valid_igroups, valid_inds)

    # --- Filter to constituents: real stocks have a ticker + a GICS sector ---
    df = raw.copy()
    df["_weight"] = pd.to_numeric(df["Port. Ending Weight"], errors="coerce")
    is_stock = df["V1Symbol"].notna() & df["JasonGICSSector"].notna() & df["_weight"].notna()
    df = df[is_stock].copy()
    report.n_constituents = len(df)

    # --- Ticker conversion ---
    df["ticker"] = df["V1Symbol"].map(v1symbol_to_house)
    stray = df[df["ticker"].str.contains("-", regex=False)]
    if not stray.empty:
        report.warnings.append(
            f"{len(stray)} ticker(s) kept a non-US suffix (contain '-'): "
            f"{sorted(stray['ticker'].tolist())[:10]}"
        )

    # --- Sector passthrough + validation ---
    df["sector"] = df["JasonGICSSector"].astype(str).str.strip()
    bad_sectors = set(df["sector"].unique()) - KNOWN_SECTORS
    if bad_sectors:
        raise SnapshotError(
            f"Unexpected sector name(s) not in the 11 GICS sectors: {sorted(bad_sectors)}. "
            "Wrong file or the taxonomy changed."
        )
    report.sector_count = df["sector"].nunique()

    # --- Weight: record raw sum, then normalize to 100 ---
    report.raw_weight_sum = float(df["_weight"].sum())
    total = df["_weight"].sum()
    df["weight"] = df["_weight"] / total * 100.0

    # --- Recover industry_group (+ industry) per ticker ---
    igroups, industries = [], []
    for _, r in df.iterrows():
        tkr = r["ticker"]
        ig = ind = None
        # 1) native columns
        if native_ig_col is not None and pd.notna(r.get(native_ig_col)):
            ig = str(r[native_ig_col]).strip()
            ind = str(r[native_ind_col]).strip() if native_ind_col and pd.notna(r.get(native_ind_col)) else ""
            report.n_from_native += 1
        # 2) inheritance
        elif tkr in inheritance:
            ig, ind = inheritance[tkr]
            report.n_from_inheritance += 1
        # 3) manual CSV
        elif tkr in manual:
            ig, ind = manual[tkr]
            report.n_from_manual += 1
        # 4) unknown
        else:
            report.unclassified.append({"ticker": tkr, "company_name": str(r[name_col])})
        igroups.append(ig)
        industries.append(ind if ind is not None else "")

    df["industry_group"] = igroups
    df["industry"] = industries
    df["company_name"] = df[name_col].astype(str)
    report.igroup_count = df["industry_group"].notna().sum() and \
        len(set(g for g in igroups if g))

    canonical = df[CANONICAL_COLUMNS].reset_index(drop=True)
    return canonical, report


def run_qc(canonical: pd.DataFrame, report: NormalizationReport) -> None:
    """Apply sanity thresholds. Appends soft warnings to the report; raises
    SnapshotError on clearly-broken input."""
    n = report.n_constituents

    # Duplicate tickers would corrupt the merge / inheritance.
    dupes = canonical["ticker"][canonical["ticker"].duplicated()].unique().tolist()
    if dupes:
        raise SnapshotError(f"Duplicate tickers after conversion: {sorted(dupes)}")

    # Constituent count sanity.
    if n < 400 or n > 600:
        raise SnapshotError(
            f"Found {n} constituents — far outside the expected ~500. Wrong sheet or file?"
        )
    if n < 480 or n > 520:
        report.warnings.append(f"Constituent count is {n} (expected ~500).")

    # Raw weight sum sanity (before normalization).
    if not (95.0 <= report.raw_weight_sum <= 101.0):
        report.warnings.append(
            f"Raw weight column summed to {report.raw_weight_sum:.2f}% "
            "(expected ~100). Wrong weight column?"
        )

    # Post-normalization must be ~100.
    post = float(canonical["weight"].sum())
    if abs(post - 100.0) > 1e-6:
        report.warnings.append(f"Normalized weights sum to {post:.6f}% (expected 100).")

    if report.sector_count != 11:
        report.warnings.append(f"Saw {report.sector_count} sectors (expected 11).")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        PROJECT_ROOT / "data" / "June 30 2026 SP 500 Portfolio weights.xlsx")
    df, rep = normalize_factset_snapshot(path)
    run_qc(df, rep)
    print(f"As of: {rep.as_of_label}  |  raw rows: {rep.n_raw_rows}  |  constituents: {rep.n_constituents}")
    print(f"Classified: native={rep.n_from_native} inheritance={rep.n_from_inheritance} manual={rep.n_from_manual}")
    print(f"Unclassified ({len(rep.unclassified)}): {[u['ticker'] for u in rep.unclassified]}")
    print(f"Sectors: {rep.sector_count} | industry_groups: {rep.igroup_count} | raw weight sum: {rep.raw_weight_sum:.4f}")
    for w in rep.warnings:
        print("  WARN:", w)
