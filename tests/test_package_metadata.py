from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_valid_and_anonymous() -> None:
    protocol = json.loads((ROOT / "configs" / "paper_protocol.json").read_text(encoding="utf-8"))
    assert protocol["waferem"]["final_test_volume"] == "wafer4_ds2"
    assert len(protocol["waferem"]["adaptation_volumes"]) == 5
    assert protocol["inference"]["rag_rf"]["feature_count"] == 55

    forbidden = ("/" + "home/", "/" + "mnt/", "agent_" + "workpacks", "liu" + "hongyu", "192" + ".168.")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(name in {".git", "artifacts", ".cache", "__pycache__"} for name in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".json", ".md", ".yml", ".yaml", ".css", ".js", ".html", ".toml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in text for token in forbidden), path
