"""Prepare cleaned CSV inputs used by the DuckDB SQL pipeline."""

from pathlib import Path
import re
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SUPPLY_PATH = RAW_DIR / "global_supply_chain_risk_2026.csv"
COMMODITY_PATH = RAW_DIR / "CMO-Historical-Data-Monthly.xlsx"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_sql_friendly(text: str) -> str:
    """
    Convert a column name into a clean snake_case style name suitable
    for CSV export and later SQL usage.
    """
    text = str(text).strip().lower()

    replacements = {
        "$": "usd",
        "%": "pct",
        "/": "_per_",
        "&": "and",
        ",": "",
        ".": "",
        "-": "_",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        ":": "",
        ";": "",
        "'": "",
        '"': "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def flatten_commodity_columns(columns: pd.Index) -> list[str]:
    """
    Flatten the 2-row Excel header from the World Bank commodity sheet.
    Keeps period_code as raw source field for later month creation in SQL.
    """
    clean_cols = []

    for level_1, level_2 in columns:
        level_1 = str(level_1).strip()
        level_2 = str(level_2).strip()

        if "Unnamed" in level_1:
            clean_cols.append("period_code")
            continue

        base = make_sql_friendly(level_1)
        unit = make_sql_friendly(level_2)

        if unit:
            clean_cols.append(f"{base}__{unit}")
        else:
            clean_cols.append(base)

    return clean_cols


def clean_standard_columns(columns: pd.Index) -> list[str]:
    """
    Standard cleaner for normal one-row CSV headers.
    """
    return [make_sql_friendly(col) for col in columns]


# -------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------

def load_supply_data(path: Path) -> pd.DataFrame:
    """
    Load supply-chain dataset and standardize column names.
    """
    df_supply = pd.read_csv(path)
    df_supply.columns = clean_standard_columns(df_supply.columns)
    return df_supply


def load_commodity_data(path: Path) -> pd.DataFrame:
    """
    Load World Bank commodity data from the Monthly Prices sheet,
    repair the 2-row header, and keep period_code for SQL-based month creation.
    """
    df_com = pd.read_excel(
        path,
        sheet_name="Monthly Prices",
        header=[4, 5],
        engine="openpyxl",
        na_values=["...", "…", "..", ""],
    )

    df_com.columns = flatten_commodity_columns(df_com.columns)

    # Make sure weird text leftovers become proper missing values where possible
    for col in df_com.columns:
        if col != "period_code":
            df_com[col] = pd.to_numeric(df_com[col], errors="coerce")

    return df_com


# -------------------------------------------------------------------
# Save processed outputs
# -------------------------------------------------------------------

def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    df_supply = load_supply_data(SUPPLY_PATH)
    df_com_price = load_commodity_data(COMMODITY_PATH)

    save_processed_data(df_supply, PROCESSED_DIR / "supply_chain_risk_clean.csv")
    save_processed_data(df_com_price, PROCESSED_DIR / "commodity_prices_clean.csv")

    print("Supply data shape:", df_supply.shape)
    print("Commodity data shape:", df_com_price.shape)
    print("\nSupply columns:")
    print(df_supply.columns[:10].tolist())
    print("\nCommodity columns:")
    print(df_com_price.columns[:10].tolist())

if __name__ == "__main__":
    main()