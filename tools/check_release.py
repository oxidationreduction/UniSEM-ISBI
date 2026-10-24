"""Check source hygiene, local site links, and the prepared model manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "artifacts", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}
TEXT_SUFFIXES = {".py", ".json", ".ipynb", ".md", ".toml", ".yml", ".yaml", ".html", ".css", ".js", ".txt"}


def source_files(root: Path = ROOT):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and not (set(relative.parts) & SKIP_DIRS):
            if path.suffix != ".zip":
                yield path


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for key in ("src", "href", "poster"):
            if values.get(key):
                self.urls.append(values[key])


def check_links() -> list[str]:
    errors = []
    site = ROOT / "site"
    for page in site.rglob("*.html"):
        parsed = Links()
        parsed.feed(page.read_text(encoding="utf-8"))
        for value in parsed.urls:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            target = (page.parent / unquote(url.path)).resolve() if url.path else page
            if url.path and not target.is_file():
                errors.append(f"Broken local link in {page.relative_to(ROOT)}: {value}")
            elif url.fragment and target.suffix == ".html":
                destination = Links()
                destination.feed(target.read_text(encoding="utf-8"))
                if url.fragment not in destination.ids:
                    errors.append(f"Missing anchor in {page.relative_to(ROOT)}: {value}")
    return errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model_manifest(manifest: dict) -> list[str]:
    """Require a verifiable download identity for each published asset."""
    errors = []
    for asset in manifest.get("assets", []):
        name = asset.get("path", "<unnamed asset>")
        published = asset.get("release_status") == "published"
        url = asset.get("download_url")
        if published:
            revision = asset.get("revision", "")
            if not re.fullmatch(r"[a-f0-9]{40}", revision):
                errors.append(f"Published model needs an immutable revision: {name}")
            parsed = urlsplit(url or "")
            if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None or parsed.fragment:
                errors.append(f"Published model needs a public HTTPS download: {name}")
            provider = asset.get("provider", "huggingface")
            if provider == "huggingface":
                matches = (parsed.hostname == "huggingface.co" and not parsed.query
                           and f"/resolve/{revision}/" in parsed.path
                           and parsed.path.endswith("/" + Path(name).name))
            elif provider == "modelscope":
                matches = (parsed.hostname == "modelscope.cn"
                           and bool(asset.get("repo_id"))
                           and parsed.path == f"/api/v1/models/{asset.get('repo_id')}/repo"
                           and parse_qs(parsed.query) == {"Revision": [revision], "FilePath": [Path(name).name]})
            else:
                matches = False
            if not matches:
                errors.append(f"Published model URL does not match revision/file: {name}")
            if not asset.get("license"):
                errors.append(f"Published model is missing license metadata: {name}")
        elif url:
            errors.append(f"Unpublished model must not claim a download: {name}")
        if not re.fullmatch(r"[a-f0-9]{64}", asset.get("sha256", "")):
            errors.append(f"Invalid model SHA256: {name}")
    if manifest.get("public_download_url") and not any(
        asset.get("release_status") == "published" for asset in manifest.get("assets", [])
    ):
        errors.append("A public model page requires at least one published asset.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-artifacts", action="store_true", help="Also hash the large local model files")
    args = parser.parse_args()
    errors = []
    # Split literals keep the hygiene scanner itself free of the strings it checks.
    private = ["/" + "home/", "/" + "mnt/", "/opt/" + "data/", "agent_" + "workpacks", "192" + ".168."]
    windows_path = re.compile(r"[A-Za-z]:[\\/]" + "Users[\\/]", re.I)
    file_count = 0
    for path in source_files():
        file_count += 1
        if path.suffix.lower() == ".pdf":
            errors.append(f"PDF publication is prohibited: {path.relative_to(ROOT)}")
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"Source file exceeds 50 MiB: {path.relative_to(ROOT)}")
        if path.suffix in TEXT_SUFFIXES:
            content = path.read_text(encoding="utf-8-sig")
            if any(token in content for token in private) or windows_path.search(content):
                errors.append(f"Private development reference: {path.relative_to(ROOT)}")
            if path.suffix in {".json", ".ipynb"}:
                try:
                    parsed_json = json.loads(content)
                    if path.suffix == ".ipynb":
                        if "TO_BE_PINNED" in content:
                            errors.append(f"Notebook revision is not pinned: {path.relative_to(ROOT)}")
                        for cell in parsed_json.get("cells", []):
                            if cell.get("outputs") or cell.get("execution_count") is not None:
                                errors.append(f"Notebook contains saved execution data: {path.relative_to(ROOT)}")
                except ValueError as error:
                    errors.append(f"Invalid JSON {path.relative_to(ROOT)}: {error}")
    errors.extend(check_links())
    manifest_path = ROOT / "models" / "manifest.json"
    artifact_count = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors.extend(check_model_manifest(manifest))
        if args.check_artifacts:
            for asset in manifest["assets"]:
                file = ROOT / asset["path"]
                artifact_count += 1
                if not file.is_file():
                    errors.append(f"Missing local model: {asset['path']}")
                elif file.stat().st_size != asset["bytes"] or sha256(file) != asset["sha256"]:
                    errors.append(f"Model size/hash mismatch: {asset['path']}")
    else:
        errors.append("Model manifest is missing.")
    result = {"ok": not errors, "source_files": file_count, "hashed_artifacts": artifact_count, "errors": errors}
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
