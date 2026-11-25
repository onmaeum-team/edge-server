import itertools
import re
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse

import pipeline as P
import report as RPT
import storage as S


@asynccontextmanager
async def lifespan(app):
    P.load_models()
    restore()
    yield
    P.unload_models()


app = FastAPI(title="모시응 보드 API", lifespan=lifespan)
_proc_lock = threading.Lock()


ESCALATORS = {
    "esc-1": {"escalatorId": "esc-1", "name": "1호기 에스컬레이터", "location": "정문 1층",
              "spec": {"elevatorNo": "2104-381", "manufacturer": "가나승강기", "ratedSpeed": 0.5, "driveType": "인버터식"}},
    "esc-2": {"escalatorId": "esc-2", "name": "2호기 에스컬레이터", "location": "B동 1층",
              "spec": {"elevatorNo": "2104-382", "manufacturer": "가나승강기", "ratedSpeed": 0.5, "driveType": "인버터식"}},
    "esc-8": {"escalatorId": "esc-8", "name": "8호기 에스컬레이터", "location": "C동 1층",
              "spec": {"elevatorNo": "3307-116", "manufacturer": "다라엘리베이터", "ratedSpeed": 0.5, "driveType": "직결식"}},
    "esc-3": {"escalatorId": "esc-3", "name": "3호기 에스컬레이터", "location": "B동 2층",
              "spec": {"elevatorNo": "2104-383", "manufacturer": "가나승강기", "ratedSpeed": 0.5, "driveType": "와이델타식"}},
    "esc-7": {"escalatorId": "esc-7", "name": "7호기 에스컬레이터", "location": "C동 지하 1층",
              "spec": {"elevatorNo": "3307-115", "manufacturer": "다라엘리베이터", "ratedSpeed": 0.5, "driveType": "인버터식"}},
}
HISTORIES = {
    "esc-7": [{"inspectionId": "insp-0915-07", "inspectedAt": "2026-09-15T10:30:00", "inspector": "박철우",
               "durationMinutes": 14, "status": "NORMAL", "checkedItems": ["3.6.2", "3.10"], "cautionItems": [], "note": None}],
}
INSPECTIONS = {}
CUSTOM = {}
_seq = itertools.count(1)


def history_entry(insp):
    items = list(insp["items"].values())
    defects = [i for i in items if i["status"] == "부적합"]
    return {
        "inspectionId": insp["inspectionId"], "inspectedAt": insp.get("submittedAt"), "inspector": insp["inspector"],
        "inspectionType": insp.get("inspectionType", "regular"),
        "durationMinutes": insp.get("durationMinutes", 0), "status": "CAUTION" if defects else "NORMAL",
        "checkedItems": [i["itemCode"] for i in items if i["status"] != "미확인"],
        "cautionItems": [i["itemCode"] for i in defects],
        "note": "; ".join(f"{i['itemCode']} {i.get('note') or ''}" for i in defects) or None,
    }


def restore():
    CUSTOM.update(S.load_escalators())
    ESCALATORS.update(CUSTOM)
    n = 0
    for insp in S.load_inspections():
        iid = insp["inspectionId"]
        if iid in INSPECTIONS:
            continue
        insp.update(seen={}, skipped=set(), pending=None, log=[], lastSpeech=None, submitted=True)
        INSPECTIONS[iid] = insp
        HISTORIES.setdefault(insp["escalatorId"], []).insert(0, history_entry(insp))
        n += 1
    print(f"저장된 설비 {len(CUSTOM)}개, 지난 점검 {n}개를 불러왔습니다. (저장 위치: {S.DATA_DIR.resolve()})")


def ok(data):
    return {"success": True, "data": data, "error": None}


def fail(status, code, message):
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def get_insp(iid):
    insp = INSPECTIONS.get(iid)
    if not insp:
        fail(404, "INSPECTION_NOT_FOUND", "점검 정보를 찾을 수 없습니다.")
    return insp


def progress(insp):
    done = sum(1 for it in insp["items"].values() if it["status"] != "미확인")
    return {"recorded": done, "total": len(insp["items"])}


def speak(text):
    return {"text": text, "audioId": P.tts(text)}


@app.get("/api/health")
def health():
    return ok({"status": "ok", **P.STATUS, "serverTime": datetime.now().isoformat()})


@app.get("/api/audio/{audio_id}")
def get_audio(audio_id: str):
    p = P.audio_path(audio_id)
    if p is None:
        fail(404, "AUDIO_NOT_FOUND", "음성 파일이 없습니다.")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/escalators")
def list_escalators():
    return ok(list(ESCALATORS.values()))


@app.post("/api/escalators")
def register_escalator(body: dict):
    eid = str(body.get("escalatorId") or body.get("id") or "").strip().lower()
    name = str(body.get("name") or "").strip()
    if not eid or not name:
        fail(400, "INVALID_REQUEST", "escalatorId 와 name 이 필요합니다.")
    if eid in ESCALATORS:
        fail(409, "ESCALATOR_ALREADY_EXISTS", "이미 등록된 설비입니다.")
    e = {"escalatorId": eid, "name": name, "location": str(body.get("location") or "")}
    if isinstance(body.get("spec"), dict):
        e["spec"] = body["spec"]
    ESCALATORS[eid] = CUSTOM[eid] = e
    S.save_escalators(CUSTOM)
    return ok(e)


@app.get("/api/escalators/{escalator_id}")
def get_escalator(escalator_id: str):
    e = ESCALATORS.get(escalator_id.lower())
    if not e:
        fail(404, "ESCALATOR_NOT_FOUND", "등록되지 않은 설비입니다.")
    return ok(e)


@app.get("/api/escalators/{escalator_id}/histories")
def get_histories(escalator_id: str, limit: int = 20):
    if escalator_id.lower() not in ESCALATORS:
        fail(404, "ESCALATOR_NOT_FOUND", "등록되지 않은 설비입니다.")
    return ok(HISTORIES.get(escalator_id.lower(), [])[:limit])


TYPE_LABEL = {"self": "자체점검", "regular": "정기검사"}


def session_items(escalator_id, inspection_type):
    if inspection_type == "regular":
        return {c: {"itemCode": c, "itemName": v["name"], "status": "미확인"} for c, v in P.ITEMS.items()}
    due = P.due_items(HISTORIES.get(escalator_id, []))
    return {c: {"itemCode": c, "itemName": P.ALL_ITEMS[c]["name"], "status": "미확인",
                "due": d, "lastChecked": last, "cycleMonths": P.ALL_ITEMS[c]["cycleMonths"]} for c, d, last in due}


@app.get("/api/escalators/{escalator_id}/items")
def get_items(escalator_id: str, type: str = "all"):
    if type == "self":
        if escalator_id.lower() not in ESCALATORS:
            fail(404, "ESCALATOR_NOT_FOUND", "등록되지 않은 설비입니다.")
        return ok(list(session_items(escalator_id.lower(), "self").values()))
    return ok([{"itemCode": c, **v} for c, v in P.ALL_ITEMS.items()])


@app.post("/api/inspections")
def start_inspection(escalatorId: str = Form(...), inspector: str = Form("점검자"), inspectionType: str = Form("self")):
    if escalatorId.lower() not in ESCALATORS:
        fail(404, "ESCALATOR_NOT_FOUND", "등록되지 않은 설비입니다.")
    if inspectionType not in TYPE_LABEL:
        fail(400, "INVALID_REQUEST", "inspectionType 은 self 또는 regular 입니다.")
    iid = f"insp-{datetime.now():%Y%m%d-%H%M%S}-{next(_seq):02d}"
    while iid in INSPECTIONS:
        iid = f"insp-{datetime.now():%Y%m%d-%H%M%S}-{next(_seq):02d}"
    INSPECTIONS[iid] = {
        "inspectionId": iid, "escalatorId": escalatorId.lower(), "inspector": inspector,
        "startedAt": datetime.now().isoformat(), "submitted": False,
        "inspectionType": inspectionType,
        "items": session_items(escalatorId.lower(), inspectionType),
        "log": [],
        "pending": None,
        "seen": {},
        "lastSpeech": None,
        "skipped": set(),
    }
    return ok({"inspectionId": iid, "inspectionType": inspectionType, "items": list(INSPECTIONS[iid]["items"].values())})


@app.get("/api/inspections/{iid}")
def inspection_state(iid: str):
    insp = get_insp(iid)
    return ok({"progress": progress(insp), "pending": insp["pending"], "lastRecord": insp["log"][-1] if insp["log"] else None})


@app.post("/api/inspections/{iid}/utterances")
async def utterance(iid: str, clientSeq: int = Form(...), audio: UploadFile | None = File(None), text: str | None = Form(None)):
    insp = get_insp(iid)
    if text is None and audio is None:
        fail(400, "INVALID_REQUEST", "audio 또는 text 가 필요합니다.")
    wav = await audio.read() if text is None else None
    return ok(await run_in_threadpool(process_utterance, insp, clientSeq, wav, text))


def process_utterance(insp, seq, wav, text):
    with _proc_lock:
        if insp["submitted"]:
            fail(409, "INSPECTION_ALREADY_SUBMITTED", "이미 제출된 점검입니다.")
        if seq in insp["seen"]:
            return insp["seen"][seq]
        t0 = time.perf_counter()
        if text is None:
            try:
                text = P.stt(wav)
            except NotImplementedError as e:
                fail(503, "MODEL_NOT_READY", str(e))
            text = (text or "").strip()
            if len(re.sub(r"[^가-힣A-Za-z0-9]", "", text)) < 2:
                res = {"kind": "ignored", "reason": "말소리 아님", "transcript": text, "progress": progress(insp)}
                insp["seen"][seq] = res
                print(f"[무시] 말소리 아님: \"{text}\"")
                return res
        t1 = time.perf_counter()
        res = handle_text(insp, P.normalize(text))
        t2 = time.perf_counter()
        res["timing"] = {"sttMs": round((t1 - t0) * 1000), "llmMs": round((t2 - t1) * 1000)}
        print(f"[처리] STT {t1 - t0:.1f}초 · LLM {t2 - t1:.1f}초 · {res['kind']} · \"{text}\" → {res.get('llmRaw', '')}")
        res["transcript"] = text
        res["progress"] = progress(insp)
        insp["seen"][seq] = res
        if res.get("speech"):
            insp["lastSpeech"] = res["speech"]
        return res


def handle_text(insp, t):

    cmd = P.match_command(t)
    if cmd == "undo":
        return {"kind": "command", "command": "undo", **undo_last(insp)}
    if cmd == "progress":
        p = progress(insp)
        left = [it for it in insp["items"].values() if it["status"] == "미확인" and it["itemCode"] not in insp["skipped"]]
        if not left:
            msg = f"{p['total']}개 모두 기록했습니다. 점검 끝이라고 하세요."
        else:
            nxt = ", ".join(it["itemName"] for it in left[:3])
            more = f" 등 {len(left)}개" if len(left) > 3 else ""
            msg = f"{p['total']}개 중 {p['recorded']}개 기록, {len(left)}개 남았습니다. 다음은 {nxt}{more}."
        return {"kind": "command", "command": "progress", "speech": speak(msg)}
    if cmd == "repeat":
        return {"kind": "command", "command": "repeat", "speech": insp["lastSpeech"]}
    if cmd == "finish":
        return {"kind": "command", "command": "finish", **finish_check(insp)}
    if cmd == "skip_all":
        left = [it for it in insp["items"].values() if it["status"] == "미확인"]
        insp["skipped"].update(it["itemCode"] for it in left)
        insp["pending"] = None
        p = progress(insp)
        msg = f"남은 {len(left)}개 항목을 미점검으로 남깁니다. 화면에서 확인하고 제출해 주세요." if left else "모든 항목을 기록했습니다. 화면에서 확인하고 제출해 주세요."
        return {"kind": "command", "command": "finish", "missing": [], "speech": speak(msg)}
    if cmd == "skip" and not insp["pending"]:
        return {"kind": "command", "command": "skip", "speech": speak("지금 묻고 있는 항목이 없습니다. 남은 항목을 모두 넘기려면 나머지 건너뛰기라고 하세요.")}
    if cmd == "skip" and insp["pending"]:
        code = insp["pending"]["itemCode"]
        insp["pending"] = None
        insp["skipped"].add(code)
        return {"kind": "command", "command": "skip", "speech": speak(f"{P.ALL_ITEMS[code]['name']} 항목은 미점검으로 남깁니다.")}

    if insp["pending"]:
        q = insp["pending"]
        ans = P.match_answer(t, q["expect"])
        if ans:
            item = insp["items"][q["itemCode"]]
            item["location" if q["expect"] == "location" else "status"] = ans
            insp["pending"] = None
            return {"kind": "answered", "item": item, "speech": speak(read_back(item))}

    if len(P.compact(t)) <= P.SHORT_LEN:
        return {"kind": "retry", "reason": "짧은 말", "speech": speak("잘 못 들었습니다. 다시 말씀해 주세요.")}

    raw = P.llm(t)
    try:
        func, kw = P.parse_call(raw)
        if func is None:
            return {"kind": "ignored", "llmRaw": raw}
        P.validate(func, kw)
    except ValueError as e:
        return {"kind": "retry", "llmRaw": raw, "reason": str(e), "speech": speak("다시 말씀해 주세요.")}
    if not P.related(t, func):
        print(f"[차단] 말과 관련 없는 항목: \"{t}\" → {raw}")
        return {"kind": "retry", "llmRaw": raw, "reason": "말과 관련 없는 항목", "speech": speak("잘 못 들었습니다. 항목 이름을 넣어 다시 말씀해 주세요.")}

    code = P.FUNC_TO_CODE[func]
    verdict, criteria = P.judge(code, kw)
    added = code not in insp["items"]
    item = insp["items"].setdefault(code, {"itemCode": code, "itemName": P.ALL_ITEMS[code]["name"], "status": "미확인"})
    before = dict(item)
    item.update({"function": func, "status": verdict or "미확인", "judgedBy": "speech" if kw.get("판정") else ("selection" if P.ALL_ITEMS[code].get("selection") else "rule"),
                 "detail": kw.get("세부"), "location": kw.get("위치"), "note": kw.get("특기사항"),
                 "action": kw.get("조치"), "criteria": criteria,
                 "measure": {k: v for k, v in kw.items() if k.startswith("측정")} or None,
                 "updatedAt": datetime.now().isoformat()})
    insp["log"].append({"itemCode": code, "before": before, "added": added})

    if verdict is None:
        insp["pending"] = {"itemCode": code, "expect": "verdict"}
        return {"kind": "question", "item": item, "llmRaw": raw, "speech": speak("적합인가요, 부적합인가요?")}
    if P.ALL_ITEMS[code]["hasLocation"] and item["location"] is None and verdict != "해당없음":
        insp["pending"] = {"itemCode": code, "expect": "location"}
        return {"kind": "question", "item": item, "llmRaw": raw, "speech": speak("상부, 하부 중 어디인가요?")}
    return {"kind": "recorded", "item": item, "llmRaw": raw, "speech": speak(read_back(item))}


def read_back(item):
    if item.get("judgedBy") == "selection" and item.get("criteria"):
        return f"{item['itemCode']} {item['itemName']}, {item['criteria'].split('= ')[-1]}으로 기록했습니다."
    loc = f" {item['location']}" if item.get("location") else ""
    note = f", {item['note']}" if item.get("note") else ""
    return f"{item['itemCode']} {item['itemName']}{loc}, {item['status']}으로 기록했습니다{note}."


def undo_last(insp):
    insp["pending"] = None
    if not insp["log"]:
        return {"speech": speak("취소할 기록이 없습니다.")}
    last = insp["log"].pop()
    if last.get("added"):
        insp["items"].pop(last["itemCode"], None)
    else:
        insp["items"][last["itemCode"]] = last["before"]
    return {"item": last["before"], "speech": speak(f"{last['before']['itemName']} 기록을 취소했습니다.")}


@app.post("/api/inspections/{iid}/undo")
def undo(iid: str):
    return ok(undo_last(get_insp(iid)))


@app.patch("/api/inspections/{iid}/items/{item_code}")
def edit_item(iid: str, item_code: str, body: dict):
    insp = get_insp(iid)
    item = insp["items"].get(item_code)
    if not item:
        fail(404, "ITEM_NOT_FOUND", "이 점검에 없는 항목입니다.")
    if body.get("status") == "부적합" and not (body.get("note") or item.get("note")):
        fail(422, "NOTE_REQUIRED", "부적합 항목에 특기사항을 입력하세요.")
    item.update({k: v for k, v in body.items() if k in {"status", "location", "note", "action", "detail"}}, judgedBy="manual")
    return ok(item)


def finish_check(insp):
    missing = [it for it in insp["items"].values() if it["status"] == "미확인" and it["itemCode"] not in insp["skipped"]]
    p = progress(insp)
    if missing:
        insp["pending"] = {"itemCode": missing[0]["itemCode"], "expect": "verdict"}
        msg = f"{p['total']}개 중 {p['recorded']}개 기록했습니다. {missing[0]['itemCode']} {missing[0]['itemName']} 결과가 없습니다. 말씀하시거나 건너뛰기라고 해 주세요."
        if len(missing) > 3:
            msg += f" 남은 {len(missing)}개를 모두 미점검으로 두려면 나머지 건너뛰기라고 하세요."
    else:
        msg = "모든 항목을 기록했습니다. 화면에서 확인하고 제출해 주세요."
    return {"missing": missing, "speech": speak(msg)}


@app.post("/api/inspections/{iid}/finish")
def finish(iid: str):
    return ok(finish_check(get_insp(iid)))


@app.get("/api/inspections/{iid}/result")
def result(iid: str):
    insp = get_insp(iid)
    items = list(insp["items"].values())
    summary = {s: sum(1 for i in items if i["status"] == s) for s in ["적합", "부적합", "해당없음", "미확인"]}
    return ok({"inspectionId": iid, "escalatorId": insp["escalatorId"], "summary": summary, "items": items,
               "defects": [i for i in items if i["status"] == "부적합"], "submitted": insp["submitted"],
               "inspector": insp["inspector"], "startedAt": insp.get("startedAt"),
               "inspectionType": insp.get("inspectionType", "regular"),
               "submittedAt": insp.get("submittedAt"), "durationMinutes": insp.get("durationMinutes"),
               "reportUrl": f"/api/inspections/{iid}/report" if insp["submitted"] else None})


@app.get("/api/inspections/{iid}/report")
def get_report(iid: str):
    p = S.report_path(iid)
    if p is not None:
        return FileResponse(p, media_type="text/html; charset=utf-8")
    insp = get_insp(iid)
    return HTMLResponse(RPT.build(insp, ESCALATORS.get(insp["escalatorId"])))


@app.post("/api/inspections/{iid}/submit")
def submit(iid: str, acknowledgeMissing: bool = False):
    insp = get_insp(iid)
    if insp["submitted"]:
        fail(409, "INSPECTION_ALREADY_SUBMITTED", "이미 제출된 점검입니다.")
    items = list(insp["items"].values())
    missing = [i for i in items if i["status"] == "미확인"]
    if missing and not acknowledgeMissing:
        fail(409, "MISSING_REQUIRED_ITEMS", f"확인하지 않은 항목 {len(missing)}개가 있습니다.")
    now = datetime.now()
    insp["submitted"] = True
    insp["submittedAt"] = now.isoformat(timespec="seconds")
    started = datetime.fromisoformat(insp["startedAt"])
    insp["durationMinutes"] = max(1, round((now - started).total_seconds() / 60))
    defects = [i for i in items if i["status"] == "부적합"]
    HISTORIES.setdefault(insp["escalatorId"], []).insert(0, history_entry(insp))
    S.save_inspection(insp)
    S.save_report(iid, RPT.build(insp, ESCALATORS.get(insp["escalatorId"])))
    return ok({"inspectionId": iid, "submittedAt": insp["submittedAt"], "defects": defects,
               "reportUrl": f"/api/inspections/{iid}/report"})
