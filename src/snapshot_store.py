"""
Persist normalized quarterly snapshots into a SEPARATE workbook
(data/current-snapshots.xlsx), leaving the frozen historical workbook untouched.

Each snapshot is written in the SAME sheet layout as the historical workbook
(header on row index 6, columns Name / Port. Weight / Bench. Weight / Difference
/ Industry_Group / Industry / Sector / Ticker), so the single
`load_benchmark_from_xlsx(sheet, path=SNAPSHOT_PATH)` reads snapshots and
historical years identically.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "current-snapshots.xlsx"

# Row index the historical loader expects the header on (7th row).
HEADER_ROW = 6


def write_snapshot_sheet(canonical: pd.DataFrame, sheet_name: str,
                         path: Path = SNAPSHOT_PATH) -> Path:
    """Write a normalized snapshot as a dated sheet, replacing it if it exists
    and preserving every other sheet in the workbook.

    `canonical` must have columns: ticker, weight, sector, industry_group,
    industry, company_name.
    """
    out = pd.DataFrame({
        "Name": canonical["company_name"],
        "Port. Weight": canonical["weight"],
        "Bench. Weight": canonical["weight"],
        "Difference": 0.0,
        "Industry_Group": canonical["industry_group"],
        "Industry": canonical["industry"],
        "Sector": canonical["sector"],
        "Ticker": canonical["ticker"],
    })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        writer_kwargs = dict(mode="a", if_sheet_exists="replace")
    else:
        writer_kwargs = dict(mode="w")

    with pd.ExcelWriter(path, engine="openpyxl", **writer_kwargs) as writer:
        # startrow=HEADER_ROW puts the column headers on row index 6 and data
        # below, matching the historical workbook the loader reads.
        out.to_excel(writer, sheet_name=str(sheet_name), startrow=HEADER_ROW, index=False)

    return path


def list_snapshots(path: Path = SNAPSHOT_PATH) -> list:
    """Return the sheet (snapshot) names currently stored, newest last."""
    path = Path(path)
    if not path.exists():
        return []
    return pd.ExcelFile(path).sheet_names
