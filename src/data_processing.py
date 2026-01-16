from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


def _maybe_to_numeric(series: pd.Series) -> pd.Series:
    """
    If a column looks numeric but is stored as strings (e.g., '123 MPa', '45%'),
    try to clean and convert it to numeric. If conversion fails, return original.
    """
    if series.dtype != "object":
        return series  # already numeric or bool

    s = series.astype(str)

    # If too many are non-numeric strings, don't touch it (likely categorical)
    # We'll attempt conversion only if it seems mostly numeric after cleaning.
    cleaned = (
        s.str.strip()
         .str.replace(",", "", regex=False)
         .str.replace(r"(?i)\bmpa\b", "", regex=True)   # remove 'MPa'
         .str.replace("%", "", regex=False)             # remove '%'
         .str.replace(r"[^\d\.\-eE+]", "", regex=True)  # keep digits, dot, signs, exponent
    )

    numeric = pd.to_numeric(cleaned, errors="coerce")
    # If at least 70% become numeric (or NaN), we accept conversion
    ratio = numeric.notna().mean()
    if ratio >= 0.70:
        return numeric
    return series


def preprocess_dataset(
    input_csv: Path,
    output_csv: Path,
    drop_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    convert_numeric_like: bool = True,
) -> None:
    """
    Minimal + safe preprocessing for tabular materials datasets.

    Parameters
    ----------
    input_csv : Path
        Path to raw CSV (e.g., data/raw/dataset.csv)
    output_csv : Path
        Path to save cleaned CSV (e.g., data/processed/cleaned_dataset.csv)
    drop_cols : list[str]
        Columns to drop (IDs, names, serial numbers).
    target_cols : list[str]
        Optional: list of target columns you care about. If provided, script prints
        missing-value counts for targets and doesn't drop target rows here (training scripts can).
    convert_numeric_like : bool
        If True, attempts to convert numeric-looking object columns to numeric.
    """
    drop_cols = drop_cols or []
    target_cols = target_cols or []

    if not input_csv.exists():
        raise FileNotFoundError(f"Raw dataset not found: {input_csv}")

    df = pd.read_csv(input_csv)

    # 1) Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2) Drop user-specified columns (if present)
    present_drop = [c for c in drop_cols if c in df.columns]
    if present_drop:
        df = df.drop(columns=present_drop)

    # 3) Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # 4) Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    # 5) Try converting numeric-like text columns to numeric
    if convert_numeric_like:
        for col in df.columns:
            df[col] = _maybe_to_numeric(df[col])

    # 6) Save cleaned dataset
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # 7) Print summary (helpful during debugging)
    print("=== Preprocessing Summary ===")
    print(f"Input : {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Dropped columns: {present_drop if present_drop else 'None'}")
    print(f"Dropped duplicates: {before - after}")
    print("\nColumn types:")
    print(df.dtypes)

    if target_cols:
        print("\nMissing values in targets:")
        for t in target_cols:
            if t in df.columns:
                print(f"  {t}: {df[t].isna().sum()}")
            else:
                print(f"  {t}: (not found)")
