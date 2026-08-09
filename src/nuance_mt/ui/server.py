"""Local HTML UI: translate + human review of medium imports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nuance_mt.pipeline import NuancePipeline, highlight_html

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
ANN = ROOT / "data" / "annotations"
DECISIONS = ANN / "review_decisions.jsonl"
SPLITS = ("train", "val", "test")

app = FastAPI(title="Nuance MT local UI")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def get_pipeline() -> NuancePipeline:
    return NuancePipeline(repo_root=ROOT)


class TranslateRequest(BaseModel):
    text: str
    target: Literal["de", "ja"] = "de"


class ReviewDecision(BaseModel):
    id: str
    action: Literal["accept", "edit", "reject"]
    reviewer: str = "human"
    fields: dict[str, Any] = Field(default_factory=dict)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _decided_ids() -> set[str]:
    return {d["id"] for d in _load_jsonl(DECISIONS)}


def _medium_queue(lang_pair: str = "all") -> list[dict]:
    decided = _decided_ids()
    queue: list[dict] = []
    for split in SPLITS:
        for row in _load_jsonl(ANN / f"{split}.jsonl"):
            if row.get("confidence") != "medium":
                continue
            if row["id"] in decided:
                continue
            if lang_pair != "all" and row.get("lang_pair") != lang_pair:
                continue
            item = dict(row)
            item["split"] = split
            queue.append(item)
    return queue


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/review")
def review_page():
    return FileResponse(STATIC / "review.html")


@app.post("/api/translate")
def api_translate(body: TranslateRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    result = get_pipeline().run(text, body.target)
    return {
        "source": result.source,
        "translation": result.translation,
        "target": result.target,
        "source_html": highlight_html(result.source, result.hits),
        "hits": [h.__dict__ for h in result.hits],
    }


@app.get("/api/review/next")
def api_review_next(lang_pair: str = Query("all")):
    queue = _medium_queue(lang_pair)
    return {
        "remaining": len(queue),
        "item": queue[0] if queue else None,
    }


@app.post("/api/review/decide")
def api_review_decide(body: ReviewDecision):
    if body.action not in {"accept", "edit", "reject"}:
        raise HTTPException(400, "bad action")
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": body.id,
        "action": body.action,
        "reviewer": body.reviewer,
        "fields": body.fields if body.action == "edit" else {},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "remaining": len(_medium_queue())}


def main(host: str = "127.0.0.1", port: int = 7860) -> None:
    import uvicorn

    print(f"Translate UI:  http://{host}:{port}/")
    print(f"Review UI:     http://{host}:{port}/review")
    uvicorn.run("nuance_mt.ui.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
