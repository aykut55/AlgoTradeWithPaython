import os
import time
import numpy as np
import pandas as pd
from typing import Tuple


class DataManager:
    """Data management class for reading and managing security data"""
    
    def __init__(self):
        # kolon isimleri
        self._timestamp_col = "timestamp"
        self._open_col = "open"
        self._high_col = "high"
        self._low_col = "low"
        self._close_col = "close"
        self._volume_col = "volume"
        self._lot_col = "lot"
        self._bar_count = 0

        # dataframe
        self._df: pd.DataFrame | None = None

        # dosya bilgisi
        self._last_filename = None
        self._last_filesize = None

        # timing bilgisi
        self._timing_report = {}

        # data reading modes
        self._read_mode = "all_data"  # all_data, last_n, first_n, range
        self._read_params = {}  # parameters for each mode

        self.set_columns(timestamp_col="timestamp", open_col="open", high_col="high", low_col="low",
                                close_col="close", volume_col="volume", lot_col="lot")
    
    def readSecurityData(self, data_dir: str, subdir: str, filename: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, int]:
        """
        Reads security data from CSV file and returns OHLCV data as pandas Series
        
        Args:
            data_dir: Base data directory (e.g., "algorithmic_trading\\data")
            subdir: Subdirectory (e.g., "01")
            filename: CSV filename (e.g., "BTCUSD.csv")
            
        Returns:
            Tuple of (Open, High, Low, Close, Volume, Lot, BarCount)
        """
        file_path = os.path.join(data_dir, subdir, filename)
        
        # Read CSV file (limit to first 10000 rows for testing)
        df = pd.read_csv(file_path, nrows=10000)
        
        # Extract series
        Open = pd.Series(df['open'].values)
        High = pd.Series(df['high'].values)
        Low = pd.Series(df['low'].values)
        Close = pd.Series(df['close'].values)
        Volume = pd.Series(df['volume'].values)
        Lot = pd.Series(df['lot'].values)  # Using lot as Lot equivalent
        BarCount = len(df)
        
        return Open, High, Low, Close, Volume, Lot, BarCount

    # --------------------------------------------------------
    # Genel zaman ölçer
    def _timeit(self, name, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self._timing_report[name] = elapsed
        return result

    # --------------------------------------------------------
    # Timer raporu
    def reportTimes(self):
        print("\n=== Timing Report ===")
        if not self._timing_report:
            print("No timing data collected.")
            return
        for k, v in self._timing_report.items():
            print(f"{k:25s}: {v:.6f} sec")

    # --------------------------------------------------------
    # kolon isimlerini ayarla
    def set_columns(self, timestamp_col="timestamp",
                    open_col="open", high_col="high",
                    low_col="low", close_col="close",
                    volume_col="volume", lot_col="lot"):
        self._timestamp_col = timestamp_col
        self._open_col = open_col
        self._high_col = high_col
        self._low_col = low_col
        self._close_col = close_col
        self._volume_col = volume_col
        self._lot_col = lot_col

    # --------------------------------------------------------
    # Data reading mode configuration
    def set_read_mode_all_data(self):
        """Read all data from CSV file"""
        self._read_mode = "all_data"
        self._read_params = {}
    
    def set_read_mode_last_n(self, n: int):
        """Read last n rows from CSV file"""
        self._read_mode = "last_n"
        self._read_params = {"n": n}
    
    def set_read_mode_first_n(self, n: int):
        """Read first n rows from CSV file"""
        self._read_mode = "first_n"
        self._read_params = {"n": n}
    
    def set_read_mode_range(self, start: int, end: int):
        """Read data from start to end row (inclusive)"""
        """start dahil, end  dahil değil"""
        self._read_mode = "range"
        self._read_params = {"start": start, "end": end}

    def _apply_read_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured read mode to the dataframe"""
        if self._read_mode == "all_data":
            return df
        elif self._read_mode == "last_n":
            n = self._read_params.get("n", 1000)
            return df.tail(n).copy()
        elif self._read_mode == "first_n":
            n = self._read_params.get("n", 1000)
            return df.head(n).copy()
        elif self._read_mode == "range":
            start = self._read_params.get("start", 0)
            end = self._read_params.get("end", len(df))
            # Ensure bounds are valid
            start = max(0, start)
            end = min(len(df), end)
            if start >= end:
                print(f"Warning: Invalid range start={start}, end={end}, returning empty dataframe")
                return df.iloc[0:0].copy()  # Empty dataframe with same columns
            return df.iloc[start:end].copy()
        else:
            print(f"Warning: Unknown read mode '{self._read_mode}', using all data")
            return df

    # --------------------------------------------------------
    # CSV oku
    def load_prices_from_csv(self, data_dir: str, subdir: str, file_name: str, auto_time=False):
        def _impl():
            self.clear_dataframe()

            file_path = os.path.join(data_dir, subdir, file_name)
            # Read full CSV file first
            df_full = pd.read_csv(file_path)
            
            # Apply read mode to filter data
            df = self._apply_read_mode(df_full)
            
            self._df = df
            self._last_filename = file_name
            self._last_filesize = os.path.getsize(file_path)
            self._bar_count = len(df)
            
            print(f"Read mode: {self._read_mode}")
            if self._read_params:
                print(f"Read params: {self._read_params}")
            print(f"Total rows in file: {len(df_full)}")
            print(f"Loaded rows: {len(df)}")
            
            if auto_time:
                self.add_time_columns()

        return self._timeit("load_prices_from_csv", _impl)

    # --------------------------------------------------------
    # zaman kolonlarını ekle
    def add_time_columns(self):
        def _impl():
            if self._df is None:
                print("No data loaded.")
                return

            ts = self._df[self._timestamp_col]

            # epoch mu datetime mı kontrol
            if pd.api.types.is_integer_dtype(ts) or (ts.dtype == "object" and ts.str.isdigit().all()):
                dt = pd.to_datetime(ts, unit="s", errors="coerce").dt.tz_localize(None)
                self._df["epoch_time_stamp"] = ts.astype("int64")
            else:
                dt = pd.to_datetime(ts, errors="coerce").dt.tz_localize(None)
                self._df["epoch_time_stamp"] = dt.astype("int64") // 1_000_000_000

            # ek kolonlar
            self._df["date_time"] = dt
            self._df["date"] = dt.dt.date
            self._df["time"] = dt.dt.time

        return self._timeit("add_time_columns", _impl)

    # --------------------------------------------------------
    # index işlemleri
    def set_datetime_index(self, drop=False):
        if self._df is None or "date_time" not in self._df:
            print("No data to set index.")
            return
        self._df.set_index("date_time", drop=drop, inplace=True)

    def reset_datetime_index(self):
        if self._df is None:
            print("No data to reset index.")
            return
        self._df.reset_index(drop=False, inplace=True)

    # --------------------------------------------------------
    # DataFrame erişim
    def get_dataframe(self):
        return self._df

    def clear_dataframe(self):
        self._df = None
        self._last_filename = None
        self._last_filesize = None

    def summary(self, n=5):
        def _impl():
            if self._df is None:
                print("No data loaded.")
                return
            print(self._df.head(n))
            print("...")
            print(self._df.tail(n))
            print(f"Rows: {len(self._df)}")
            print(f"Columns: {list(self._df.columns)}")
        return self._timeit("summary", _impl)

    # --------------------------------------------------------
    # getter'lar (NumPy array tabanlı)
    def get_open_array(self):
        if self._df is None or self._open_col not in self._df:
            return np.array([])
        return self._df[self._open_col].to_numpy()

    def get_high_array(self):
        if self._df is None or self._high_col not in self._df:
            return np.array([])
        return self._df[self._high_col].to_numpy()

    def get_low_array(self):
        if self._df is None or self._low_col not in self._df:
            return np.array([])
        return self._df[self._low_col].to_numpy()

    def get_close_array(self):
        if self._df is None or self._close_col not in self._df:
            return np.array([])
        return self._df[self._close_col].to_numpy()

    def get_volume_array(self):
        if self._df is None or self._volume_col not in self._df:
            return np.array([])
        return self._df[self._volume_col].to_numpy()

    def get_lot_array(self):
        if self._df is None or self._lot_col not in self._df:
            return np.array([])
        return self._df[self._lot_col].to_numpy()

    # --- time based ---
    def get_date_time_array(self):
        if self._df is None or "date_time" not in self._df:
            return np.array([])
        return self._df["date_time"].to_numpy()

    def get_date_time_array_as_str(self):
        if self._df is None or "date_time" not in self._df:
            return np.array([])
        if "date_time_str" not in self._df:
            self._df["date_time_str"] = self._df["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return self._df["date_time_str"].to_numpy()

    def get_date_array(self):
        if self._df is None or "date_time" not in self._df:
            return np.array([])
        return self._df["date_time"].values.astype("datetime64[D]")

    def get_date_array_as_str(self):
        if self._df is None or "date_time" not in self._df:
            return np.array([])
        if "date_str" not in self._df:
            self._df["date_str"] = self._df["date_time"].dt.strftime("%Y-%m-%d")
        return self._df["date_str"].to_numpy()

    def get_time_array(self):
        if self._df is None or "date_time" not in self._df:
            return np.array([])
        if "time_obj" not in self._df:
            self._df["time_obj"] = self._df["date_time"].dt.time
        return self._df["time_obj"].to_numpy()

    def get_time_array_as_str(self):
        if self._df is None or "date_time" not in self._df:
            return np.array([])
        if "time_str" not in self._df:
            self._df["time_str"] = self._df["date_time"].dt.strftime("%H:%M:%S")
        return self._df["time_str"].to_numpy()

    # --- timestamp / epoch ---
    def get_timestamp_array(self):
        if self._df is None or self._timestamp_col not in self._df:
            return np.array([])
        return self._df[self._timestamp_col].to_numpy()

    def get_timestamp_array_as_str(self):
        if self._df is None or self._timestamp_col not in self._df:
            return np.array([])
        if "timestamp_str" not in self._df:
            self._df["timestamp_str"] = self._df[self._timestamp_col].astype(str)
        return self._df["timestamp_str"].to_numpy()

    def get_epoch_time_array(self):
        if self._df is None or "epoch_time_stamp" not in self._df:
            return np.array([])
        return self._df["epoch_time_stamp"].to_numpy()

    def get_epoch_time_array_as_str(self):
        if self._df is None or "epoch_time_stamp" not in self._df:
            return np.array([])
        if "epoch_time_stamp_str" not in self._df:
            self._df["epoch_time_stamp_str"] = self._df["epoch_time_stamp"].astype(str)
        return self._df["epoch_time_stamp_str"].to_numpy()

    # --- misc ---
    def get_items_count(self):
        return len(self._df) if self._df is not None else 0

    # --------------------------------------------------------
    # Timer utilities
    def startTimer(self):
        """Başlangıç zamanını kaydeder."""
        self._t0 = time.time()

    def stopTimer(self, label="Elapsed"):
        """Başlangıçtan itibaren geçen süreyi ekrana basar ve saniye olarak döndürür."""
        if not hasattr(self, "_t0"):
            print("Timer not started.")
            return None
        elapsed = time.time() - self._t0
        print(f"{label}: {elapsed:.4f} seconds")
        return elapsed

    def print_head(self, n=5):
        """DataFrame'in ilk n satırını tüm kolonlarla birlikte yazdırır."""
        if self._df is None:
            print("No data loaded.")
            return
        with pd.option_context(
                "display.max_columns", None,
                "display.width", None,
                "display.max_colwidth", None
        ):
            print(self._df.head(n))

    def print_tail(self, n=5):
        """DataFrame'in son n satırını tüm kolonlarla birlikte yazdırır."""
        if self._df is None:
            print("No data loaded.")
            return
        with pd.option_context(
                "display.max_columns", None,
                "display.width", None,
                "display.max_colwidth", None
        ):
            print(self._df.tail(n))

    def get_bar_count(self):
        return self._bar_count