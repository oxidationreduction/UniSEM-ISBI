"""Validate the native showcase and mixed-resolution viewer data contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "site" / "assets"


def test_native_showcase_matches_verified_asset_identity():
    case = json.loads((ASSETS / "highres" / "case.json").read_text(encoding="utf-8"))
    assert case["crop_shape"] == [1024, 1024]
    assert case["full_slice_shape"] == [1024, 1024]
    assert case["xy_nm"] is None
    for method, relative in case["images"].items():
        path = ASSETS / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["provenance"]["display_asset_sha256"][method]
        with Image.open(path) as image:
            assert image.format == "PNG" and image.size == (1024, 1024)


def test_case_specific_methods_defaults_and_metrics_remain_consistent():
    manifest = json.loads((ASSETS / "cases.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in manifest["cases"]}
    assert manifest["default_case_id"] in cases
    assert {"typical_improvement", "minimum_improvement"} <= cases.keys()
    for case in cases.values():
        methods = case.get("methods", list(case["images"]))
        assert set(methods) <= manifest["methods"].keys()
        assert set(case["whole_slice_metrics"]) <= set(methods)
        if "defaults" in case:
            assert set(case["defaults"].values()) <= set(methods)
        for relative in case["images"].values():
            with Image.open(ASSETS / relative) as image:
                assert list(reversed(image.size)) == case["crop_shape"]


def test_rebuilding_wafer_assets_retains_the_native_showcase():
    spec = importlib.util.spec_from_file_location("build_site_assets", ROOT / "tools" / "build_site_assets.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    manifest = json.loads((ASSETS / "cases.json").read_text(encoding="utf-8"))
    manifest["cases"] = [case for case in manifest["cases"] if case["id"] != manifest["default_case_id"]]
    showcase = builder.attach_highres_case(manifest, ASSETS)
    assert manifest["cases"][0] == showcase
    assert len(manifest["cases"]) == 3
    assert manifest["default_case_id"] == showcase["id"]
