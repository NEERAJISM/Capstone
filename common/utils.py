import zipfile
import rarfile
from pathlib import Path

def extract_archive_at_source(archive_path: str) -> None:
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
