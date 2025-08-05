import zipfile
from pathlib import Path

def unzip_csvs_at_source(zip_path: str) -> None:
    """
    Unzips all contents of the given ZIP file

    Args:
        zip_path (str): Path to the ZIP file.

    Returns:
        None
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if not zip_path.suffix == '.zip':
        raise ValueError(f"Not a .zip file: {zip_path}")

    dest_dir = zip_path.parent

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    print(f"Extracted {zip_path.name} to {dest_dir}")
