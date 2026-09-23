"""Small dataframe I/O helpers with a Parquet -> CSV fallback.

Parquet is preferred for the large CompTox matrices, but it requires pyarrow or
fastparquet. The fallback keeps the prototype runnable in a minimal environment.
"""
from pathlib import Path
import pandas as pd


def _csv_alternative(path: Path) -> Path:
    return path.with_suffix(".csv")


def write_table(df: pd.DataFrame, preferred_path) -> Path:
    path = Path(preferred_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".parquet":
        df.to_csv(path, index=False)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ModuleNotFoundError):
        csv_path = _csv_alternative(path)
        df.to_csv(csv_path, index=False)
        return csv_path


def read_table(preferred_path) -> pd.DataFrame:
    path = Path(preferred_path)
    csv_path = _csv_alternative(path) if path.suffix.lower() == ".parquet" else path

    if path.exists() and path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError):
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise RuntimeError(
                f"{path.name} exists but no Parquet engine is installed. "
                "Install pyarrow or create the CSV fallback."
            )

    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"Could not find {path} or CSV fallback {csv_path}."
    )


def table_exists(preferred_path) -> bool:
    path = Path(preferred_path)
    return path.exists() or _csv_alternative(path).exists()
