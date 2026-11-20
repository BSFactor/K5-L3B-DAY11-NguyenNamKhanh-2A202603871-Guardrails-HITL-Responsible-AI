"""Write complete JSON files so an interrupted write leaves the previous file intact."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def write_json(filepath: str | Path, data: object) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging_dir:
        staged = Path(staging_dir) / path.name
        staged.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged.replace(path)
