#!/usr/bin/env python3
"""Apply human review decisions to annotation JSONL splits."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotations"
DECISIONS = ANN / "review_decisions.jsonl"
SPLITS = ("train", "val", "test")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    decisions = load_jsonl(DECISIONS)
    if not decisions:
        raise SystemExit(f"No decisions at {DECISIONS}. Use the /review UI first.")

    # last decision wins per annotation id
    by_id: dict[str, dict] = {}
    for d in decisions:
        by_id[d["id"]] = d

    stats = {"accepted": 0, "rejected": 0, "edited": 0, "untouched": 0}
    for split in SPLITS:
        path = ANN / f"{split}.jsonl"
        rows = load_jsonl(path)
        out = []
        for r in rows:
            d = by_id.get(r["id"])
            if not d:
                out.append(r)
                stats["untouched"] += 1
                continue
            action = d["action"]
            if action == "reject":
                stats["rejected"] += 1
                continue
            if action in {"accept", "edit"}:
                updated = dict(r)
                if action == "edit":
                    for key in (
                        "src_span",
                        "src_char_start",
                        "src_char_end",
                        "nuance_type",
                        "en_frame",
                        "target_frame",
                        "explanation",
                        "evidence",
                        "tgt_span",
                    ):
                        if key in d.get("fields", {}):
                            updated[key] = d["fields"][key]
                    # re-sync span text if offsets provided with source
                    s, e = updated["src_char_start"], updated["src_char_end"]
                    updated["src_span"] = updated["source_text"][s:e]
                    stats["edited"] += 1
                else:
                    stats["accepted"] += 1
                updated["confidence"] = "high"
                updated["annotator"] = d.get("reviewer", "human-review")
                updated["review_action"] = action
                out.append(updated)
        write_jsonl(path, out)
        print(f"{split}: {len(rows)} → {len(out)}")

    print("stats:", stats)
    print("Done. Re-run: python -m training.prepare_data && python -m training.train_span_classifier")


if __name__ == "__main__":
    main()
