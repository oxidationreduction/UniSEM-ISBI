"""Build the local UniSEM project-page assets from the frozen paper package.

No training, inference, metric recomputation, or source-data download is performed.
The twelve PNG crops are copied byte-for-byte; GIFs and PDF previews are display
derivatives. Requires Pillow and Poppler's pdftoppm only when rebuilding assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


RELEASE_ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "raw": "Raw EM",
    "gt": "Ground truth",
    "sam160": "Target-only · 160 slices",
    "sam625": "Target-only · 625 slices",
    "source160": "Two-stage · 160 slices",
    "source625": "Two-stage · 625 slices",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" in line and re.search(r"\d+\.\d+", line):
            rows.append([float(value) for value in re.findall(r"\d+\.\d+", line)])
    return rows


def evidence_data(paper_root: Path) -> dict:
    tables = paper_root / "tables"
    baseline = numeric_rows(tables / "performance.tex")
    structure = numeric_rows(tables / "structure_ablation.tex")
    stages = numeric_rows(tables / "stage_ablation.tex")
    efficiency = numeric_rows(tables / "efficiency.tex")
    if list(map(len, [baseline, structure, stages, efficiency])) != [13, 4, 3, 10]:
        raise ValueError("The frozen paper tables have changed shape; review the public mapping.")
    baseline_names = ["UniSEM", "µSAM", "HQ-SAM", "SegNeuron", "TokenUnify-500M", "nnUNet"]
    wafer_names = ["UniSEM FullFT", "UniSEM LightFT", *baseline_names[1:]]

    def row(name: str, values: list[float], n: int | None = None) -> dict:
        result = {"name": name, "voi": values[0]}
        if len(values) == 4:
            result.update(voi_sd=values[1], arand=values[2], arand_sd=values[3])
        else:
            result["arand"] = values[1]
        if n is not None:
            result["n"] = n
        return result

    em_rows = [dict(name=name, voi=v[0], voi_merge=v[1], arand=v[2])
               for name, v in zip(baseline_names, baseline[:6])]
    wafer_rows = [dict(name=name, voi=v[0], voi_merge=v[1], arand=v[2])
                  for name, v in zip(wafer_names, baseline[6:])]
    names = ["Dense decoder", "Final-feature residual", "Repeated-final fusion", "Cross-layer fusion"]
    structure_rows = [dict(row(name, values, 3), residual=index > 0, fusion=index > 1, early=index == 3)
                      for index, (name, values) in enumerate(zip(names, structure))]
    stage_rows = [
        dict(row("Source training only", stages[0], 1), stage1=True, stage2=False),
        dict(row("Target training only", stages[1], 3), stage1=False, stage2=True),
        dict(row("Two-stage training", stages[2], 3), stage1=True, stage2=True),
    ]
    budget_rows = []
    for source, budget, values in zip([False, False, True, True, True], [160, 625, 160, 315, 625], efficiency[:5]):
        budget_rows.append(dict(row("Two-stage" if source else "Target-only", values, 3),
                                source=source, slices=budget))
    update_names = ["Decoder", "Decoder + HQ", "Last 6 blocks + heads", "All 24 blocks + heads", "FullFT"]
    update_rows = []
    for index, (name, values) in enumerate(zip(update_names, efficiency[5:])):
        update_rows.append(dict(row(name, values[1:], 2 if index == 1 else 3),
                                parameters_m=values[0], lightft=index == 3))
    return {
        "schema": "unisem-public-evidence-v1",
        "metric_direction": "lower_is_better",
        "baseline": {"emneuron": em_rows, "waferem": wafer_rows},
        "structure": structure_rows,
        "stages": stage_rows,
        "annotation": budget_rows,
        "parameters": update_rows,
        "sources": {f"tables/{name}.tex": sha256(tables / f"{name}.tex")
                    for name in ["performance", "structure_ablation", "stage_ablation", "efficiency"]},
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def comparison_film(asset_root: Path, cases: list[dict]) -> None:
    """A 10-second wipe of two actual prediction pairs, with no morphing."""
    frames = []
    panel_size = 274
    for case in cases:
        images = {key: Image.open(asset_root / case["images"][key]).convert("RGB")
                  .resize((panel_size, panel_size), Image.Resampling.NEAREST)
                  for key in ["raw", "gt", "sam625", "source160"]}
        for index in range(20):
            frame = Image.new("RGB", (960, 420), "#ffffff")
            draw = ImageDraw.Draw(frame)
            draw.text((30, 17), "UniSEM  /  From generalist to specialist", fill="#211d30", font=font(23, True))
            draw.text((30, 51), case["title"] + f"  ·  slice {case['z']}", fill="#625c70", font=font(16))
            cut = round((0.5 + 0.42 * math.sin((index / 19 - 0.5) * math.pi)) * panel_size)
            comparison = images["source160"].copy()
            comparison.paste(images["sam625"].crop((0, 0, cut, panel_size)), (0, 0))
            ImageDraw.Draw(comparison).line((cut, 0, cut, panel_size), fill="white", width=3)
            for x, key in [(30, "raw"), (343, "prediction"), (656, "gt")]:
                frame.paste(comparison if key == "prediction" else images[key], (x, 92))
            draw.text((30, 372), "Raw EM", fill="#211d30", font=font(17, True))
            draw.text((343, 372), "Target-only 625  |  Two-stage 160", fill="#211d30", font=font(15, True))
            draw.text((656, 372), "Ground truth", fill="#211d30", font=font(17, True))
            draw.text((30, 400), "Precomputed predictions · same crop · seed 12345 · no live inference", fill="#625c70", font=font(13))
            frames.append(frame)
    frames[10].save(asset_root / "comparison-poster.png", optimize=True)
    palette = frames[10].quantize(colors=256)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(asset_root / "comparison.gif", save_all=True, append_images=indexed[1:],
                    duration=250, loop=0, optimize=True, disposal=2)


def render_pdf(source: Path, destination: Path, max_width: int = 2200) -> None:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise RuntimeError("pdftoppm is required to rebuild the PDF preview assets.")
    subprocess.run([executable, "-f", "1", "-singlefile", "-png", "-scale-to", str(max_width),
                    str(source), str(destination.with_suffix(""))], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True, help="Directory containing the author-approved manuscript and figure sources")
    parser.add_argument("--site-root", type=Path, default=RELEASE_ROOT / "site")
    args = parser.parse_args()
    paper_root = args.paper_root.resolve()
    assets = args.site_root.resolve() / "assets"
    (assets / "crops").mkdir(parents=True, exist_ok=True)
    manifest_path = paper_root / "data" / "qualitative_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_copy = []
    asset_sources = []
    for original_case in original["cases"]:
        case_id = original_case["case_id"]
        z = original_case["z"]
        images = {}
        for method in original["column_order"]:
            name = f"{case_id}_z{z:03d}_{method}.png"
            source = paper_root / "figures" / "qualitative_assets" / name
            destination = assets / "crops" / name
            shutil.copyfile(source, destination)
            expected = original_case["pixel_hashes_sha256"][method]
            actual = hashlib.sha256(Image.open(destination).tobytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Frozen crop pixel hash mismatch: {name}")
            images[method] = f"crops/{name}"
            asset_sources.append(dict(file=f"assets/crops/{name}", sha256=sha256(source),
                                      source=f"figures/qualitative_assets/{name}",
                                      role="frozen_display_crop", release_status="local_research_preview"))
        typical = case_id == "typical_improvement"
        case_copy.append({
            "id": case_id,
            "title": "Typical improvement" if typical else "Residual merge / regression",
            "z": z,
            "crop_yxyx": original_case["bbox_yxyx"],
            "crop_shape": [256, 256],
            "xy_nm": original["pixel_scale"]["xy_nm"],
            "images": images,
            "whole_slice_metrics": original_case["whole_slice_existing_metrics"],
            "interpretation": (
                "At one contact, target-only training with 625 slices joins two GT instances; "
                "the two-stage model with 160 slices separates them. Residual errors remain elsewhere."
                if typical else
                "Source initialization does not improve every location. Here the two-stage model "
                "with 160 slices still merges adjacent GT instances that target-only training separates."
            ),
        })
    cases = {
        "schema": "unisem-precomputed-viewer-v1",
        "dataset": "WaferEM, 8 nm XY",
        "volume": "wafer4_ds2",
        "seed": original["seed"],
        "full_slice_shape": [625, 625],
        "methods": METHODS,
        "selection": {
            "slices": "Typical: closest observed positive whole-slice CREMI gap to its median. Residual: smallest gap among all 125 slices.",
            "crops": "256 px windows, stride 128 with terminal coordinates; maximum boundary disagreement, ties by y then x.",
            "colors": "Prediction instance colors follow maximum full-slice GT overlap. Actual prediction boundaries are retained; merged instances remain a single region.",
        },
        "source_manifest_sha256": sha256(manifest_path),
        "prediction_checkpoints": {
            key: {
                "checkpoint_sha256": manifest["sources"][key]["checkpoint_sha256"],
                "segmentation_sha256": manifest["sources"][key]["file_sha256"],
                "target_label_slices": 160 if key.endswith("160") else 625,
            }
            for key in ("sam160", "sam625", "source160", "source625")
        },
        "cases": case_copy,
    }
    write_json(assets / "cases.json", cases)
    write_json(args.site_root.resolve() / "data" / "results.json", evidence_data(paper_root))
    comparison_film(assets, case_copy)
    paper = paper_root / "UniSEM-ISBI.pdf"
    supplement = paper_root / "output" / "UniSEM_ISBI_zh_supplement.pdf"
    for source, name in [(paper, "paper.pdf"), (supplement, "supplement.pdf"),
                         (paper_root / "figures" / "1-overview.pdf", "author-overview.pdf"),
                         (paper_root / "figures" / "14-results-show.pdf", "qualitative-comparison.pdf")]:
        shutil.copyfile(source, assets / name)
        asset_sources.append(dict(file=f"assets/{name}", sha256=sha256(source),
                                  source=str(source.relative_to(paper_root)).replace("\\", "/"),
                                  role="author_manuscript_asset", release_status="local_research_preview"))
    render_pdf(assets / "author-overview.pdf", assets / "author-overview.png")
    render_pdf(assets / "qualitative-comparison.pdf", assets / "qualitative-comparison.png", 2600)
    write_json(assets / "asset-provenance.json", {
        "schema": "unisem-site-assets-v1",
        "software_license_is_not_data_permission": True,
        "samples": "Frozen WaferEM crops, local research preview. Source dataset terms apply.",
        "sources": asset_sources,
        "generated": ["comparison.gif", "comparison-poster.png", "author-overview.png", "qualitative-comparison.png"],
    })
    print(json.dumps({"site": str(args.site_root.resolve()), "copied_crops": 12,
                      "cases": len(case_copy), "gif_duration_seconds": 10,
                      "asset_files": len(list(assets.rglob("*")))}, indent=2))


if __name__ == "__main__":
    main()
