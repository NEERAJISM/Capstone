import os
import calendar
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Dict, Union

import polars as pl
from joblib import Parallel, delayed

from .utils import get_logger


class StockDataLoader:
    """Manages loading, preprocessing, and caching of stock market data.

    This class is designed to efficiently handle large datasets by:
    - Loading data in parallel using all available CPU cores.
    - Caching loaded data to avoid redundant disk I/O.
    - Imputing missing data points to ensure continuity.
    - Resampling data to different time frequencies.
    """

    def __init__(
        self,
        tickers: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        agg_func: Optional[Union[str, Callable]] = None,
        resample_freq: Optional[str] = None,
        select_columns: Optional[List[str]] = None,
        impute: bool = True,
        base_dir: Optional[Union[str, Path]] = "../downloaded_files",
    ):
        """Initializes the data loader with configuration for data retrieval and processing."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing StockDataLoader.")
        
        self._start = self._parse_datetime(start)
        self._end = self._parse_datetime(end)
        self._base_dir = Path(base_dir)
        
        # If no tickers are provided, find all common tickers across the date range.
        self._tickers = tickers or self._find_available_tickers()
        
        self._agg_func = agg_func
        self._resample_freq = resample_freq
        self._select_columns = select_columns  
        self._impute = impute

        # In-memory cache for loaded dataframes to speed up repeated access.
        self._data_cache: Dict[str, pl.DataFrame] = {}

        # Mappings for renaming columns from the raw CSV format.
        self._rev_cols_map = {
            "open_interest": "<o/i> ",
            "date": "<date>",
            "time": "<time>",
            "open": "<open>",
            "high": "<high>",
            "low": "<low>",
            "close": "<close>",
            "volume": "<volume>",
        }
        self._cols_map = {v: k for k, v in self._rev_cols_map.items()}
        
        self._new_columns, self._columns = zip(*self._rev_cols_map.items())
        if self._select_columns:
            self._new_columns = self._select_columns + ["date", "time"]
            self._columns = [self._rev_cols_map[c] for c in self._new_columns]

        # Define the schema for the final dataframe.
        self._schema = {
             "date": pl.String,
            "time":pl.String,
            "datetime": pl.Datetime("us"),
            "open_price": pl.Float64,
            "high_price": pl.Float64,
            "low_price": pl.Float64,
            "close_price": pl.Float64,
            "volume": pl.Float64,
            "open_interest": pl.Float64,
        }

    def _find_available_tickers(self) -> List[str]:
        """Identifies tickers that are present in all months within the specified date range."""
        self.logger.info("Finding available tickers...")
        months = self._generate_monthly_files()
        all_month_tickers = []

        for month in months:
            folder = self._get_month_dir(month)
            if not folder.exists():
                continue

            tickers_in_month = {f.stem for f in folder.glob("*.csv") if f.is_file()}
            if tickers_in_month:
                all_month_tickers.append(tickers_in_month)

        if not all_month_tickers:
            raise FileNotFoundError(f"No CSV files found in any month between {self._start} and {self._end}")

        # The intersection of tickers across all months ensures data continuity.
        common_tickers = set.intersection(*all_month_tickers)
        if not common_tickers:
            raise ValueError("No common tickers found across all months in the given date range.")

        self.logger.info(f"Found {len(common_tickers)} common tickers.")
        return sorted(common_tickers)

    def _parse_datetime(self, dt: Union[str, datetime]) -> datetime:
        """Parses a string into a datetime object."""
        if isinstance(dt, datetime):
            return dt
        return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

    def _generate_monthly_files(self) -> List[str]:
        """Generates a list of month strings (YYYY-MM) between the start and end dates."""
        start_month = self._start.replace(day=1)
        end_month = self._end.replace(day=1)

        months = []
        current_month = start_month
        while current_month <= end_month:
            months.append(current_month.strftime("%Y-%m"))
            # Move to the next month.
            current_month = (current_month + timedelta(days=32)).replace(day=1)
        return months

    def _get_month_dir(self, month: str) -> Path:
        """Constructs the directory path for a given month."""
        year, month_num = month.split("-")
        month_name = calendar.month_name[int(month_num)]
        return self._base_dir / year / f"Cash Data {month_name} {year}"

    def _empty_df(self) -> pl.DataFrame:
        """Creates an empty dataframe with the correct schema."""
        return pl.DataFrame(schema=self._schema)
        
    def _load_single_file(self, ticker: str, month: str) -> pl.DataFrame:
        """Loads and preprocesses a single monthly CSV file for a given ticker."""
        folder = self._get_month_dir(month)
        file_path = folder / f"{ticker}.csv"
    
        if not file_path.exists():
            return self._empty_df()
    
        df = pl.read_csv(file_path, columns=self._columns)
        df = df.rename({i: self._cols_map[i] for i in df.columns})
    
        # Create a single datetime column from the date and time columns.
        df = df.with_columns(
            pl.concat_str([df["date"], df["time"]], separator=" ")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S")
            .alias("datetime")
        ).drop(["date", "time"]).sort("datetime")
    
        df = df.set_sorted("datetime")
        return df
    
    def _merge_monthly_data(self, ticker: str) -> pl.DataFrame:
        """Merges data from multiple monthly files for a single ticker."""
        self.logger.info(f"Merging monthly data for {ticker}...")
        months = self._generate_monthly_files()
        dfs = [self._load_single_file(ticker, month) for month in months]
        dfs = [d for d in dfs if not d.is_empty()]
    
        if not dfs:
            return self._empty_df()
    
        df = pl.concat(dfs, how="vertical")
        df = df.sort("datetime")
        df = df.unique(subset=["datetime"], keep="last", maintain_order=True)
    
        # Filter the data to the specified date range.
        df = df.filter((pl.col("datetime") >= self._start) & (pl.col("datetime") <= self._end))
    
        # Impute missing data by forward-filling and then backward-filling.
        if self._impute:
            time_range = pl.DataFrame({
                "datetime": pl.datetime_range(
                    start=self._start,
                    end=self._end,
                    interval="1m",
                    eager=True
                )
            })
            df = time_range.join(df, on="datetime", how="left")
            df = df.fill_null(strategy="forward").fill_null(strategy="backward")
    
        # Resample the data to the specified frequency.
        if not df.is_empty() and self._resample_freq:
            agg_func = self._agg_func or "mean"

            if isinstance(agg_func, str):
                aggs = [getattr(pl.col(c), agg_func)() for c in df.columns if c != "datetime"]
            else: # callable
                aggs = [pl.col(c).apply(agg_func) for c in df.columns if c != "datetime"]

            df = df.group_by_dynamic(
                index_column="datetime",
                every=self._resample_freq,
                closed="left"
            ).agg(aggs)
            df = df.sort("datetime")
    
        keep_cols = ["datetime"] + [col for col in self._new_columns if col in df.columns]
        return df.select(keep_cols)

    def get_data_for_tickers(self, tickers: Optional[List[str]] = None) -> Dict[str, pl.DataFrame]:
        """Retrieves processed data for a list of tickers, using parallel processing and caching."""
        if not tickers:
            tickers = self._tickers
        else:
            tickers = [t for t in tickers if t in self._tickers]

        if not tickers:
            raise ValueError("No valid tickers provided or found.")

        self.logger.info(f"Getting data for {len(tickers)} tickers in parallel...")
        # Parallelize data loading across all available CPU cores.
        n_jobs = os.cpu_count() or 1
        results_list = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._load_or_get_from_cache)(ticker) for ticker in tickers
        )

        return dict(zip(tickers, results_list))

    def _load_or_get_from_cache(self, ticker: str) -> pl.DataFrame:
        """Loads data for a ticker from the cache or from disk if not already cached."""
        if ticker not in self._data_cache:
            self.logger.info(f"Loading data for {ticker} from disk.")
            self._data_cache[ticker] = self._merge_monthly_data(ticker)
        else:
            self.logger.info(f"Loading data for {ticker} from cache.")
        return self._data_cache[ticker]
