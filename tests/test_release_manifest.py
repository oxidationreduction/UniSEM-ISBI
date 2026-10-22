"""Published and local candidate assets must not be confused."""
from copy import deepcopy
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).parents[1] / "tools" / "check_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def published_manifest():
    revision = "1" * 40
    return {
        "public_download_url": "https://huggingface.co/dreamBuilder/UniSEM",
        "assets": [{
            "path": "artifacts/model.pt", "sha256": "2" * 64,
            "release_status": "published", "license": "apache-2.0",
            "revision": revision,
            "download_url": f"https://huggingface.co/dreamBuilder/UniSEM/resolve/{revision}/model.pt",
        }],
    }


def test_mixed_public_and_local_assets():
    manifest = published_manifest()
    candidate = deepcopy(manifest["assets"][0])
    candidate.update(path="artifacts/target.pt", release_status="local_candidate_only")
    candidate.pop("download_url")
    manifest["assets"].append(candidate)
    assert release.check_model_manifest(manifest) == []


def test_unpublished_asset_cannot_advertise_download():
    manifest = published_manifest()
    manifest["assets"][0]["release_status"] = "local_candidate_only"
    assert any("Unpublished" in error for error in release.check_model_manifest(manifest))


def test_published_asset_must_pin_matching_file():
    manifest = published_manifest()
    manifest["assets"][0]["download_url"] = "https://huggingface.co/dreamBuilder/UniSEM/resolve/main/model.pt"
    assert any("revision/file" in error for error in release.check_model_manifest(manifest))


def test_modelscope_download_pins_revision_and_file():
    manifest = published_manifest()
    asset = manifest["assets"][0]
    asset.update(provider="modelscope", repo_id="dreamBuilder/UniSEM")
    base = "https://modelscope.cn/api/v1/models/dreamBuilder/UniSEM/repo"
    asset["download_url"] = f"{base}?Revision={asset['revision']}&FilePath=model.pt"
    assert release.check_model_manifest(manifest) == []
    asset["download_url"] = f"{base}?Revision=master&FilePath=model.pt"
    assert any("revision/file" in error for error in release.check_model_manifest(manifest))
    asset["download_url"] = f"{base}?Revision={asset['revision']}&FilePath=other.pt"
    assert any("revision/file" in error for error in release.check_model_manifest(manifest))
