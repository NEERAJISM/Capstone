import zipfile
import rarfile
from pathlib import Path
import pandas as pd


class Utils:

    @staticmethod
    def extract_archive_at_source(self, archive_path: str) -> None:
        archive_path = Path(archive_path)

        if not archive_path.exists():
            raise FileNotFoundError(f"File not found: {archive_path}")

        dest_dir = archive_path.parent
        suffix = archive_path.suffix.lower()

        if suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            print(f"Extracted ZIP: {archive_path.name} to {dest_dir}")

        elif suffix == '.rar':
            try:
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    rar_ref.extractall(dest_dir)
                print(f"Extracted RAR: {archive_path.name} to {dest_dir}")
            except rarfile.NeedFirstVolume:
                raise RuntimeError("This is part of a split archive (e.g., .part1.rar). Please extract manually.")
            except rarfile.RarCannotExec as e:
                raise RuntimeError("Missing 'unrar' or 'bsdtar'. Please install it to handle RAR files.") from e

        else:
            raise ValueError(f"Unsupported archive format: {suffix}")

    @staticmethod
    def load_stock_csv(filepath):
        """Load CSV with columns <date>, <time>, <close> -> DataFrame indexed by datetime with column Close."""
        df = pd.read_csv(filepath)
        if '<date>' in df.columns and '<time>' in df.columns:
            df['datetime'] = pd.to_datetime(df['<date>'] + ' ' + df['<time>'],
                                            format='%m/%d/%Y %H:%M:%S', errors='coerce')
        else:
            # Fallback for other schemas
            for c in ['datetime', 'timestamp', 'DateTime', 'Date', 'Time']:
                if c in df.columns:
                    df['datetime'] = pd.to_datetime(df[c], errors='coerce')
                    break
        # find close col
        ccol = None
        for c in ['<close>', 'Close', 'close', 'LAST', 'Adj Close']:
            if c in df.columns:
                ccol = c
                break
        if ccol is None:
            raise ValueError(f"Close column not found in {filepath}. Columns={df.columns.tolist()}")

        df = df[['datetime', ccol]].copy()
        df.rename(columns={ccol: 'Close'}, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['datetime', 'Close'], inplace=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        return df
