import json
import os
import re
from pathlib import Path

DATA_DIR = Path(os.environ.get("ONMAEUM_DATA_DIR", "data"))
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_escalators() -> dict:
    p = DATA_DIR / "escalators.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_escalators(escalators: dict):
    _write(DATA_DIR / "escalators.json", json.dumps(escalators, ensure_ascii=False, indent=2))


def save_inspection(insp: dict):
    iid = insp["inspectionId"]
    if not _SAFE.match(iid):
        return
    data = {k: v for k, v in insp.items() if k not in ("seen", "skipped", "pending", "log", "lastSpeech")}
    _write(DATA_DIR / "inspections" / f"{iid}.json", json.dumps(data, ensure_ascii=False, indent=2))


def load_inspections() -> list:
    d = DATA_DIR / "inspections"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print("점검 파일을 읽지 못함:", p, e)
    return sorted(out, key=lambda x: x.get("submittedAt") or "")


def save_report(iid: str, html: str):
    if _SAFE.match(iid):
        _write(DATA_DIR / "reports" / f"{iid}.html", html)


def report_path(iid: str):
    if not _SAFE.match(iid):
        return None
    p = DATA_DIR / "reports" / f"{iid}.html"
    return p if p.exists() else None
