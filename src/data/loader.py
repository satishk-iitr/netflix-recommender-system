"""
Netflix Prize Data Loader
=========================
Parses the idiosyncratic Netflix Prize ``combined_data_*.txt`` format into
tabular DataFrames suitable for modelling.

File format
-----------
Each file contains blocks like::

    MovieID:
    UserID,Rating,Date
    UserID,Rating,Date
    ...
    MovieID:
    ...

The loader streams through these blocks, keeping a running ``movie_id``
state variable, and emits rows of ``(user_id, movie_id, rating, date)``.

Uses **Polars** for blazing-fast I/O when available, falling back to
**Pandas** transparently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Polars-first, Pandas-fallback strategy
# ---------------------------------------------------------------------------
try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

import pandas as pd

logger = logging.getLogger(__name__)

# Type alias — DataFrame is either Polars or Pandas depending on runtime
DataFrame = Union["pl.DataFrame", "pd.DataFrame"]


class NetflixDataLoader:
    """Load and convert the raw Netflix Prize text files into DataFrames.

    Parameters
    ----------
    use_polars : bool, optional
        If *True* (default) and Polars is installed, return Polars
        DataFrames.  Otherwise fall back to Pandas.

    Examples
    --------
    >>> loader = NetflixDataLoader()
    >>> df = loader.load_all_data("data/raw")
    >>> df.shape
    (100480507, 4)
    """

    def __init__(self, use_polars: bool = True) -> None:
        self.use_polars: bool = use_polars and _HAS_POLARS
        if use_polars and not _HAS_POLARS:
            logger.warning(
                "Polars not installed — falling back to Pandas. "
                "Install with: pip install polars"
            )

    # ------------------------------------------------------------------
    # Core parser
    # ------------------------------------------------------------------
    def parse_combined_file(self, filepath: Union[str, Path]) -> DataFrame:
        """Parse a single ``combined_data_*.txt`` file.

        Parameters
        ----------
        filepath : str or Path
            Path to one of the four combined data files.

        Returns
        -------
        DataFrame
            Columns: ``[user_id, movie_id, rating, date]``.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info("Parsing %s ...", filepath.name)

        user_ids: list[int] = []
        movie_ids: list[int] = []
        ratings: list[int] = []
        dates: list[str] = []

        current_movie_id: int = -1
        total_lines = _count_lines(filepath)

        with open(filepath, "r", encoding="utf-8") as fh:
            for line in tqdm(
                fh,
                total=total_lines,
                desc=f"Parsing {filepath.name}",
                unit=" lines",
                mininterval=1.0,
            ):
                line = line.strip()
                if not line:
                    continue

                if line.endswith(":"):
                    # Movie ID header line, e.g. "12345:"
                    current_movie_id = int(line[:-1])
                else:
                    parts = line.split(",")
                    if len(parts) != 3:
                        logger.debug("Skipping malformed line: %s", line)
                        continue
                    user_ids.append(int(parts[0]))
                    movie_ids.append(current_movie_id)
                    ratings.append(int(parts[1]))
                    dates.append(parts[2])

        logger.info(
            "Parsed %s — %s ratings from %s",
            filepath.name,
            f"{len(user_ids):,}",
            filepath.name,
        )

        if self.use_polars:
            return pl.DataFrame(
                {
                    "user_id": user_ids,
                    "movie_id": movie_ids,
                    "rating": ratings,
                    "date": dates,
                }
            ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        else:
            df = pd.DataFrame(
                {
                    "user_id": user_ids,
                    "movie_id": movie_ids,
                    "rating": ratings,
                    "date": dates,
                }
            )
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
            return df

    # ------------------------------------------------------------------
    # Convenience loaders
    # ------------------------------------------------------------------
    def load_all_data(self, data_dir: Union[str, Path]) -> DataFrame:
        """Load and concatenate all four ``combined_data_*.txt`` files.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing ``combined_data_1.txt`` … ``combined_data_4.txt``.

        Returns
        -------
        DataFrame
            Concatenated DataFrame with all ~100 M ratings.
        """
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Data directory not found: {data_dir}")

        dfs: list[DataFrame] = []
        for i in range(1, 5):
            fpath = data_dir / f"combined_data_{i}.txt"
            if not fpath.exists():
                logger.warning("File not found, skipping: %s", fpath)
                continue
            dfs.append(self.parse_combined_file(fpath))

        if not dfs:
            raise FileNotFoundError(
                f"No combined_data_*.txt files found in {data_dir}"
            )

        logger.info("Concatenating %d parsed files ...", len(dfs))
        if self.use_polars:
            combined = pl.concat(dfs)
        else:
            combined = pd.concat(dfs, ignore_index=True)

        logger.info("Total ratings: %s", f"{len(combined):,}")
        return combined

    def load_movie_titles(self, filepath: Union[str, Path]) -> DataFrame:
        """Load the ``movie_titles.csv`` file.

        The file uses ISO-8859-1 encoding and has no header.  Columns are
        ``movie_id, year_of_release, title``.  Some titles contain commas,
        so we only split on the first two commas.

        Parameters
        ----------
        filepath : str or Path
            Path to ``movie_titles.csv``.

        Returns
        -------
        DataFrame
            Columns: ``[movie_id, year, title]``.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Movie titles file not found: {filepath}")

        logger.info("Loading movie titles from %s ...", filepath.name)

        movie_ids: list[int] = []
        years: list[int | None] = []
        titles: list[str] = []

        with open(filepath, "r", encoding="ISO-8859-1") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Split on first two commas only (titles may contain commas)
                parts = line.split(",", 2)
                if len(parts) < 3:
                    logger.debug("Skipping malformed title line: %s", line)
                    continue
                movie_ids.append(int(parts[0]))
                year_str = parts[1].strip()
                years.append(int(year_str) if year_str and year_str != "NULL" else None)
                titles.append(parts[2].strip())

        logger.info("Loaded %s movie titles.", f"{len(movie_ids):,}")

        if self.use_polars:
            return pl.DataFrame(
                {
                    "movie_id": movie_ids,
                    "year": years,
                    "title": titles,
                }
            )
        else:
            return pd.DataFrame(
                {
                    "movie_id": movie_ids,
                    "year": years,
                    "title": titles,
                }
            )

    # ------------------------------------------------------------------
    # Parquet persistence
    # ------------------------------------------------------------------
    def save_processed(
        self, df: DataFrame, output_path: Union[str, Path]
    ) -> Path:
        """Save a DataFrame to Parquet format.

        Parameters
        ----------
        df : DataFrame
            Polars or Pandas DataFrame to persist.
        output_path : str or Path
            Destination ``.parquet`` file path.

        Returns
        -------
        Path
            The resolved output path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_polars and isinstance(df, pl.DataFrame):
            df.write_parquet(output_path)
        else:
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
            df.to_parquet(output_path, index=False, engine="pyarrow")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Saved processed data to %s (%.1f MB)", output_path, size_mb)
        return output_path

    def load_processed(self, path: Union[str, Path]) -> DataFrame:
        """Load a previously saved Parquet file.

        Parameters
        ----------
        path : str or Path
            Path to a ``.parquet`` file.

        Returns
        -------
        DataFrame
            Polars or Pandas DataFrame.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        if self.use_polars:
            return pl.read_parquet(path)
        else:
            return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_lines(filepath: Path) -> int:
    """Fast line count for progress bar estimation."""
    count = 0
    with open(filepath, "rb") as fh:
        # Read in 64 KB chunks for speed
        buf_size = 65_536
        buf = fh.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = fh.read(buf_size)
    return count
