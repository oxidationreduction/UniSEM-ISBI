"""Build the lightweight local source archive; weights remain separate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile

from check_release import ROOT, source_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    site_models = ROOT / "site" / "models"
    site_models.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "models").glob("*"):
        if source.is_file() and source.suffix in {".md", ".json"}:
            copyfile(source, site_models / source.name)
    target = args.output or ROOT / "site" / "downloads" / "unisem-source.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in source_files() if "downloads" not in p.relative_to(ROOT).parts and p.suffix.lower() != ".pdf"]
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, "UniSEM-ISBI/" + path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(json.dumps({"archive": target.name, "files": len(files), "bytes": target.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
