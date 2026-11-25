from datetime import datetime
from html import escape

STATUS_ORDER = ["부적합", "미확인", "적합", "해당없음"]
STATUS_LABEL = {"미확인": "미점검"}
STATUS_COLOR = {"적합": "#1f7a3a", "부적합": "#b42318", "해당없음": "#667085", "미확인": "#b54708"}


def _code_key(code: str):
    return [int(x) if x.isdigit() else x for x in code.split(".")]


def _fmt(ts):
    if not ts:
        return "-"
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts


def _e(v):
    if v is None or v == "" or v == []:
        return "-"
    if isinstance(v, list):
        v = ", ".join(map(str, v))
    return escape(str(v))


def build(insp, escalator):
    items = sorted(insp["items"].values(), key=lambda i: _code_key(i["itemCode"]))
    count = {s: sum(1 for i in items if i["status"] == s) for s in STATUS_ORDER}
    defects = [i for i in items if i["status"] == "부적합"]
    esc = escalator or {}
    regular = insp.get("inspectionType", "regular") == "regular"
    kind = "정기검사" if regular else "자체점검"
    basis = ("승강기 정기검사기준 [별표 2] 3. 에스컬레이터" if regular
             else "자체점검기준 [별표 3] 3. 에스컬레이터 (점검주기에 따라 이번 달 대상 항목)")
    scope = "" if regular else f" (이번 달 대상 {len(items)}개 항목)"
    spec = esc.get("spec") or {}

    rows = "".join(
        f"<tr class='{'todo' if i['status'] == '미확인' else ''}'><td>{_e(i['itemCode'])}</td><td>{_e(i.get('itemName'))}</td><td>{_e(i.get('location'))}</td>"
        f"<td class='st' style='color:{STATUS_COLOR.get(i['status'], '#000')}'>{_e(STATUS_LABEL.get(i['status'], i['status']))}</td>"
        f"<td>{_e(i.get('note'))}</td><td>{_e(i.get('action'))}</td>"
        f"<td class='small'>{_e(i.get('criteria'))}</td></tr>"
        for i in items
    )
    defect_list = "".join(
        f"<li><b>{_e(i['itemCode'])} {_e(i.get('itemName'))}</b>"
        f"{(' · ' + _e(i.get('location'))) if i.get('location') else ''} — {_e(i.get('note'))}"
        f"{(' (조치: ' + _e(i.get('action')) + ')') if i.get('action') else ''}</li>"
        for i in defects
    ) or "<li>없음</li>"

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(kind)} 보고서 {_e(insp['inspectionId'])}</title>
<style>
  body {{ font-family: -apple-system, 'Malgun Gothic', 'Noto Sans KR', sans-serif; margin: 24px; color: #101828; background: #fff; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: #475467; font-size: 13px; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  .wrap {{ overflow-x: auto; }}
  .items {{ min-width: 640px; table-layout: fixed; }}
  .items col.c-no {{ width: 56px; }} .items col.c-name {{ width: 30%; }} .items col.c-loc {{ width: 48px; }}
  .items col.c-st {{ width: 60px; }} .items col.c-act {{ width: 80px; }} .items col.c-cr {{ width: 18%; }}
  tr.todo td {{ color: #98a2b3; }}
  td {{ word-break: keep-all; overflow-wrap: anywhere; }}
  th, td {{ border: 1px solid #d0d5dd; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f2f4f7; }}
  .info td:first-child {{ width: 110px; background: #f9fafb; color: #475467; }}
  .sum {{ display: flex; gap: 8px; margin: 16px 0; flex-wrap: wrap; }}
  .sum div {{ border: 1px solid #d0d5dd; border-radius: 8px; padding: 8px 14px; font-size: 13px; }}
  .sum b {{ font-size: 18px; display: block; }}
  .st {{ font-weight: bold; white-space: nowrap; }}
  .small {{ font-size: 11px; color: #475467; }}
  h2 {{ font-size: 16px; margin: 24px 0 8px; }}
  .sign {{ margin-top: 32px; display: flex; gap: 40px; font-size: 13px; }}
  .print {{ margin: 16px 0; }}
  @media print {{ .print {{ display: none; }} body {{ margin: 0; }} .wrap {{ overflow: visible; }} .items {{ min-width: 0; font-size: 11px; }} table {{ width: 99.5%; }} th, td {{ padding: 4px 6px; }} }}
</style></head><body>
<h1>에스컬레이터 {_e(kind)} 보고서</h1>
<div class="sub">{_e(basis)} · 음성 기록 (모시응)</div>
<button class="print" onclick="window.print()">인쇄 / PDF로 저장</button>

<table class="info">
  <tr><td>설비</td><td>{_e(esc.get('name') or insp['escalatorId'])} ({_e(insp['escalatorId'])})</td></tr>
  <tr><td>위치</td><td>{_e(esc.get('location'))}</td></tr>
  <tr><td>승강기번호</td><td>{_e(spec.get('elevatorNo'))}</td></tr>
  <tr><td>점검 종류</td><td>{_e(kind)}{scope}</td></tr>
  <tr><td>점검 번호</td><td>{_e(insp['inspectionId'])}</td></tr>
  <tr><td>점검자</td><td>{_e(insp.get('inspector'))}</td></tr>
  <tr><td>점검 시작</td><td>{_fmt(insp.get('startedAt'))}</td></tr>
  <tr><td>제출</td><td>{_fmt(insp.get('submittedAt'))} (소요 {_e(insp.get('durationMinutes'))}분)</td></tr>
</table>

<div class="sum">
  <div><b style="color:{STATUS_COLOR['적합']}">{count['적합']}</b>적합</div>
  <div><b style="color:{STATUS_COLOR['부적합']}">{count['부적합']}</b>부적합</div>
  <div><b style="color:{STATUS_COLOR['해당없음']}">{count['해당없음']}</b>해당없음</div>
  <div><b style="color:{STATUS_COLOR['미확인']}">{count['미확인']}</b>미점검</div>
</div>

<h2>부적합 항목</h2>
<ul>{defect_list}</ul>

<h2>전체 항목</h2>
<div class="wrap"><table class="items">
  <colgroup><col class="c-no"><col class="c-name"><col class="c-loc"><col class="c-st"><col><col class="c-act"><col class="c-cr"></colgroup>
  <tr><th>번호</th><th>항목</th><th>위치</th><th>판정</th><th>특기사항</th><th>조치</th><th>판정 기준</th></tr>
  {rows}
</table></div>

<div class="sign"><div>점검자: {_e(insp.get('inspector'))} (서명) ____________</div><div>확인자: ____________</div></div>
</body></html>"""
