import ast
import difflib
import io
import json
import os
import re
import threading
import uuid
import wave
from pathlib import Path


STATUS = {"stt": "stub", "llm": "stub", "tts": "stub"}
_stt_fn = None
_llm_fn = None
_tts_fn = None
_model_lock = threading.Lock()
AUDIO_DIR = Path(os.environ.get("ONMAEUM_AUDIO_DIR", "audio_cache"))


def load_models():
    global _stt_fn, _llm_fn, _tts_fn
    stt_dir = os.environ.get("ONMAEUM_STT_DIR")
    whisper_dir = os.environ.get("ONMAEUM_STT_WHISPER")
    if whisper_dir:
        try:
            _stt_fn = _load_whisper(whisper_dir)
            STATUS["stt"] = "whisper-small"
        except Exception as e:
            STATUS["stt"] = f"error: {e}"
    elif stt_dir:
        try:
            _stt_fn = _load_sensevoice(stt_dir)
            STATUS["stt"] = "sensevoice"
        except Exception as e:
            STATUS["stt"] = f"error: {e}"
    ct2_dir = os.environ.get("ONMAEUM_LLM_CT2")
    llm_path = os.environ.get("ONMAEUM_LLM_PATH")
    if ct2_dir:
        try:
            _llm_fn = _load_ct2(ct2_dir)
            STATUS["llm"] = "ct2"
        except Exception as e:
            STATUS["llm"] = f"error: {e}"
    elif llm_path:
        try:
            import rkllm_runner
            _llm_fn = rkllm_runner.load(llm_path)
            STATUS["llm"] = "rkllm"
        except Exception as e:
            STATUS["llm"] = f"error: {e}"
    if os.environ.get("ONMAEUM_TTS") == "1":
        try:
            import tts_runner
            _tts_fn = tts_runner.load()
            STATUS["tts"] = "vits"
        except Exception as e:
            STATUS["tts"] = f"error: {e}"
    _warm_up()
    print("모델 상태:", STATUS)


def _silence_wav(sec: float = 1.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * sec))
    return buf.getvalue()


def _warm_up():
    try:
        if _stt_fn is not None:
            _stt_fn(_silence_wav())
        if _llm_fn is not None:
            _llm_fn("비상정지 스위치 정상입니다")
    except Exception as e:
        print("워밍업 실패 (무시):", e)


def unload_models():
    try:
        import rkllm_runner
        if _llm_fn is not None:
            rkllm_runner.release()
    except Exception:
        pass


def _load_whisper(model_dir: str):
    from faster_whisper import WhisperModel

    threads = int(os.environ.get("ONMAEUM_STT_THREADS", "4"))
    if os.environ.get("ONMAEUM_STT_FAST", "1") == "1":
        import stt_fast
        return stt_fast.load(model_dir, threads=threads, pad_s=float(os.environ.get("ONMAEUM_STT_PAD", "2.0")))
    model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)

    def run(audio_bytes: bytes) -> str:
        segs, _ = model.transcribe(io.BytesIO(audio_bytes), language="ko", beam_size=1, without_timestamps=True,
                                   suppress_blank=False, condition_on_previous_text=False, vad_filter=True)
        return "".join(s.text for s in segs).strip()

    return run


def _load_sensevoice(model_dir: str):
    import numpy as np
    import sherpa_onnx

    d = Path(model_dir)
    onnx = d / "model.int8.onnx"
    if not onnx.exists():
        onnx = d / "model.onnx"
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(onnx),
        tokens=str(d / "tokens.txt"),
        language="ko",
        use_itn=True,
        num_threads=int(os.environ.get("ONMAEUM_STT_THREADS", os.cpu_count() or 4)),
    )

    def run(audio_bytes: bytes) -> str:
        with wave.open(io.BytesIO(audio_bytes)) as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        samples = pcm.astype(np.float32) / 32768.0
        s = rec.create_stream()
        s.accept_waveform(sr, samples)
        rec.decode_stream(s)
        return s.result.text.strip()

    return run


SYSTEM = "에스컬레이터 정기검사 기록기. 발화를 검사기준 함수 호출로 바꾼다."


def build_prompt(text: str) -> str:
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def _load_ct2(model_dir: str):
    import ctranslate2
    from tokenizers import Tokenizer

    threads = int(os.environ.get("ONMAEUM_LLM_THREADS", os.cpu_count() or 2))
    gen = ctranslate2.Generator(model_dir, device="cpu", compute_type="int8", intra_threads=threads)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    special = tok.token_to_id("<esc_end>") is not None

    def run(text: str) -> str:
        r = gen.generate_batch([tok.encode(build_prompt(text), add_special_tokens=False).tokens],
                               max_length=160, sampling_topk=1,
                               end_token=["<esc_end>", "<|eot_id|>"] if special else "<|eot_id|>",
                               include_prompt_in_result=False, return_end_token=True)[0]
        out = tok.decode(r.sequences_ids[0], skip_special_tokens=False).replace("<|eot_id|>", "").strip()
        return out or "<esc_end>"

    return run


def stt(audio_bytes: bytes) -> str:
    if _stt_fn is None:
        raise NotImplementedError("STT 미연결: ONMAEUM_STT_DIR 을 정하거나, 요청에 text 필드를 넣어 테스트하세요")
    with _model_lock:
        return _stt_fn(audio_bytes)


try:
    import textnorm as _textnorm
except Exception:
    _textnorm = None


def normalize(text: str) -> str:
    if _textnorm is not None:
        text = _textnorm.stt_to_llm(text)
    t = text.replace("mm", "밀리").replace("미리", "밀리")
    t = re.sub(r"[.,!?]", "", t)
    for wrong, right in CORRECTIONS.items():
        t = t.replace(wrong, right)
    return t.strip()


CORRECTIONS = {"콩 ": "콤 ", "콧 ": "콤 ", "매가": "메가", "디딘판": "디딤판"}


def llm(text: str) -> str:
    if _llm_fn is None:
        return _mock_llm(text)
    with _model_lock:
        return _llm_fn(text).strip()


def _mock_llm(text: str) -> str:
    if "스텝 롤러" in text and ("벗겨" in text or "박리" in text):
        return "<esc_10>(판정='부적합', 세부=['스텝 롤러'], 위치=None, 특기사항='스텝 롤러 고무 박리', 조치=None)<esc_end>"
    if "비상정지" in text:
        loc = "'상부'" if "상부" in text else "'하부'" if "하부" in text else "None"
        verdict = "'부적합'" if "안 " in text else "'적합'"
        return f"<esc_6_2>(판정={verdict}, 위치={loc}, 특기사항=None, 조치=None)<esc_end>"
    return "<esc_end>"


def tts(text: str):
    if _tts_fn is None or not text:
        return None
    with _model_lock:
        wav_bytes = _tts_fn(text)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_id = uuid.uuid4().hex
    (AUDIO_DIR / f"{audio_id}.wav").write_bytes(wav_bytes)
    return audio_id


def audio_path(audio_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", audio_id):
        return None
    p = AUDIO_DIR / f"{audio_id}.wav"
    return p if p.exists() else None


_CHECKLIST = [
    ("3.1.1", "일반사항 (정보망 일치)"), ("3.1.2", "구동기 설치 위치"), ("3.1.3", "제어반 설치 위치"),
    ("3.1.4", "디딤판 형식"), ("3.1.5", "구동방식"), ("3.1.6", "구동 운전방식"),
    ("3.1.7", "디딤판 체인 윤활"), ("3.1.8", "손잡이 체인 윤활"), ("3.1.9", "구동체인 윤활"), ("3.1.10", "기타"),
    ("3.2.1", "안전표시"), ("3.2.2", "수동핸들 지침"), ("3.2.3", "기계류 접근 출입문 안내"),
    ("3.2.4", "비상정지장치 색상"), ("3.2.5", "유지보수 및 점검 중 접근방지 수단"), ("3.2.6", "자동 기동 운행방향 표시장치"),
    ("3.3.1", "접근금지장치"), ("3.3.2", "미끄럼방지장치"), ("3.3.3", "장애물로부터의 보호"),
    ("3.3.4", "방화셔터 인근의 에스컬레이터"), ("3.3.5", "연속되는 에스컬레이터 사이 공간"),
    ("3.3.6", "손잡이 바깥쪽 건물난간"), ("3.3.7", "조명 및 콘센트"),
    ("3.4.1", "조명 절연저항"), ("3.4.2", "접지, 전동기 절연저항 및 배선"),
    ("3.5.1", "디딤판 주행안내"), ("3.5.2", "스텝/팔레트 사이의 틈새"), ("3.5.3", "디딤판과 스커트 사이 틈새"),
    ("3.5.4", "트레드 홈에 맞물리는 콤의 깊이"), ("3.5.5", "손잡이 외형 및 가이드/덮개"), ("3.5.6", "손잡이 틈새"),
    ("3.6.1", "유지점검 및 보수용 정지스위치"), ("3.6.2", "비상정지장치"), ("3.6.3", "과부하(전류 증가)"),
    ("3.6.4", "과부하(온도 증가)"), ("3.6.5", "과속감지"), ("3.6.6", "의도되지 않은 운행 방향의 역전"),
    ("3.6.7", "보조브레이크의 미-작동 감지"), ("3.6.8", "인장장치의 과도한 움직임"),
    ("3.6.9", "디딤판체인의 파단과 과도한 늘어짐"), ("3.6.10", "콤 안전장치"),
    ("3.6.11", "연속 에스컬레이터 및 무빙워크의 추가 정지장치"), ("3.6.12", "손잡이 인입구 장치"),
    ("3.6.13", "스텝/팔레트 처짐 감지 장치"), ("3.6.14", "스텝/팔레트 누락 감지 장치"),
    ("3.6.15", "브레이크의 미-작동 감지장치"), ("3.6.16", "손잡이의 속도 편차 감지"), ("3.6.17", "점검용 덮개 열림"),
    ("3.6.18", "과도한 정지거리"), ("3.6.19", "접지 결함"), ("3.6.20", "수동핸들의 설치 감지"),
    ("3.6.21", "점검운전 제어장치의 정지장치 작동 감지"), ("3.6.22", "쇼핑 카트 및 수하물 카트 진입방지대 감지"),
    ("3.7.1", "점검운전 제어장치"), ("3.7.2", "수동 기동운전"), ("3.7.3", "자동 기동 운전"),
    ("3.8.1", "디딤판 주행"), ("3.8.2", "손잡이 주행"), ("3.8.3", "스커트 및 스커트 디플렉터"),
    ("3.8.4", "기어오름방지장치"), ("3.8.5", "접근금지장치"), ("3.8.6", "미끄럼방지장치"),
    ("3.8.7", "안전 보호판"), ("3.8.8", "쇼핑카트"), ("3.8.9", "옥외용 에스컬레이터의 추가요건"),
    ("3.9.1", "속도, 전류 및 정지거리"), ("3.9.2", "보조브레이크"),
    ("3.10", "에스컬레이터를 구성하는 승강기부품의 상태"), ("3.11", "그 밖의 표시 등"), ("3.12", "자체점검의 확인"),
]
_HAS_LOCATION = {
    "3.1.10", "3.2.1", "3.2.3", "3.2.4", "3.2.6", "3.3.1", "3.3.4", "3.3.6", "3.3.7", "3.4.1", "3.5.3", "3.5.4",
    "3.6.1", "3.6.2", "3.6.10", "3.6.12", "3.6.14", "3.6.17", "3.6.22", "3.7.1", "3.7.3", "3.8.8", "3.10",
}


SELF_CYCLE = {
    "3.2.1": 3, "3.2.2": 3, "3.2.3": 3, "3.2.4": 3, "3.2.5": 3, "3.2.6": 1,
    "3.3.1": 3, "3.3.2": 3, "3.3.3": 1, "3.3.4": 6, "3.3.5": 3, "3.3.6": 1, "3.3.7": 1,
    "3.4.1": 1, "3.4.2": 3,
    "3.5.1": 1, "3.5.2": 1, "3.5.3": 1, "3.5.4": 1, "3.5.5": 3, "3.5.6": 3,
    "3.6.1": 1, "3.6.2": 1, "3.6.3": 1, "3.6.4": 1, "3.6.5": 1, "3.6.6": 1, "3.6.7": 1, "3.6.8": 1,
    "3.6.9": 1, "3.6.10": 1, "3.6.11": 1, "3.6.12": 1, "3.6.13": 1, "3.6.14": 1, "3.6.15": 1,
    "3.6.16": 1, "3.6.17": 1, "3.6.18": 1, "3.6.20": 1, "3.6.21": 1, "3.6.22": 1,
    "3.7.1": 3, "3.7.2": 3, "3.7.3": 1,
    "3.8.1": 1, "3.8.2": 1, "3.8.3": 3, "3.8.4": 1, "3.8.5": 1, "3.8.6": 1, "3.8.7": 1, "3.8.8": 1, "3.8.9": 6,
    "3.9.1": 1, "3.9.2": 1,
}

SELECTION = {
    "esc_1_2": "설치위치", "esc_1_3": "설치위치", "esc_1_4": "디딤판형식", "esc_1_5": "구동방식", "esc_1_6": "운전방식",
}


def _func_of(code: str) -> str:
    return "esc_" + code[2:].replace(".", "_")


def _code_of(func: str) -> str:
    return "3." + func[len("esc_"):].replace("_", ".")


ALL_ITEMS = {
    code: {"function": _func_of(code), "name": name, "hasLocation": code in _HAS_LOCATION,
           "selection": _func_of(code) in SELECTION, "cycleMonths": SELF_CYCLE.get(code)}
    for code, name in _CHECKLIST
}
DEMO_CODES = ["3.5.3", "3.6.2", "3.6.12", "3.8.3", "3.10"]


ITEMS = ({c: ALL_ITEMS[c] for c in DEMO_CODES} if os.environ.get("ONMAEUM_SESSION_ITEMS") == "demo" else dict(ALL_ITEMS))


def _load_functions():
    path = Path(os.environ.get("ONMAEUM_FUNCTIONS", Path(__file__).with_name("functions_v3.json")))
    if not path.exists():
        return
    import json
    for f in json.loads(path.read_text(encoding="utf-8")):
        name = f.get("name", "")
        if not name.startswith("esc_") or _code_of(name) in ALL_ITEMS:
            continue
        params = f.get("parameters") or {}
        props = params.get("properties", params) if isinstance(params, dict) else {}
        ALL_ITEMS[_code_of(name)] = {"function": name, "name": f.get("desc") or f.get("description") or name,
                                     "hasLocation": "위치" in props, "selection": False, "cycleMonths": None}


_load_functions()


def due_items(histories, today=None):
    from datetime import date, datetime
    today = today or date.today()
    last = {}
    for h in histories:
        try:
            d = datetime.fromisoformat(str(h.get("inspectedAt"))).date()
        except ValueError:
            continue
        for c in h.get("checkedItems") or []:
            if c not in last or d > last[c]:
                last[c] = d
    out = []
    for code, it in ALL_ITEMS.items():
        cyc = it.get("cycleMonths")
        if not cyc:
            continue
        d = last.get(code)
        if d is None:
            out.append((code, "overdue", None))
            continue
        months = (today.year - d.year) * 12 + (today.month - d.month)
        if months > cyc:
            out.append((code, "overdue", d.isoformat()))
        elif months == cyc:
            out.append((code, "thisMonth", d.isoformat()))
    out.sort(key=lambda x: 0 if x[1] == "overdue" else 1)
    return out


FUNC_TO_CODE = {v["function"]: k for k, v in ALL_ITEMS.items()}
VERDICTS = {"적합", "부적합", "해당없음", None}
LOCATIONS = {"상부", "하부", "양쪽", None}


CALL_RE = re.compile(r"^\s*<(esc_\d+(?:_\d+)*)>\((.*)\)<esc_end>\s*$", re.S)


def parse_call(raw: str):
    if raw.strip() == "<esc_end>":
        return None, {}
    m = CALL_RE.match(raw)
    if not m:
        raise ValueError(f"형식 오류: {raw!r}")
    func, args = m.group(1), m.group(2)
    call = ast.parse(f"f({args})", mode="eval").body
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return func, kwargs


def validate(func: str, kw: dict):
    if func not in FUNC_TO_CODE:
        raise ValueError(f"모르는 함수: {func}")
    if kw.get("판정") not in VERDICTS:
        raise ValueError(f"허용값 밖 판정: {kw.get('판정')}")
    if kw.get("위치") not in LOCATIONS:
        raise ValueError(f"허용값 밖 위치: {kw.get('위치')}")


def judge(code: str, kw: dict):
    if kw.get("판정") is not None:
        return kw["판정"], None
    func = ALL_ITEMS.get(code, {}).get("function")
    if func in SELECTION and kw.get(SELECTION[func]):
        return "적합", f"선택값 기록: {SELECTION[func]} = {kw[SELECTION[func]]}"
    if code == "3.5.3" and kw.get("측정_좌") is not None and kw.get("측정_우") is not None:
        l, r = kw["측정_좌"], kw["측정_우"]
        ok = l <= 4 and r <= 4 and l + r <= 7
        return ("적합" if ok else "부적합"), "각 측 4mm 이하, 합 7mm 이하"
    return None, None


COMMANDS = {

    "undo": ["취소", "치소", "취서", "추소", "방금 거 지워", "되돌려", "돌려줘", "지워 줘", "삭제", "삭제해 줘", "잘못 말했"],
    "progress": ["몇 개 남았", "얼마나 남았"],
    "finish": ["점검 끝", "점검 종료", "다 했어"],
    "repeat": ["다시 읽어", "뭐라고"],

    "skip_all": ["머지 건", "나머지 넘어가", "나머지 미점검", "전부 건너", "모두 건너", "다 건너"],
    "skip": ["건너뛰", "건너띠", "건너 뛰", "넘어가", "패스"],
}


SHORT_LEN = 5


def compact(text: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", text)


def match_command(text: str):
    c = compact(text)
    for cmd, words in COMMANDS.items():
        if any(compact(w) in c for w in words):
            return cmd
    if len(c) <= SHORT_LEN + 2:
        best, score = None, 0.0
        for cmd, words in COMMANDS.items():
            for w in words:
                s = difflib.SequenceMatcher(None, c, compact(w)).ratio()
                if s > score:
                    best, score = cmd, s
        if score >= 0.6:
            return best
    return None


_KW_PATH = Path(__file__).with_name("item_keywords.json")
try:
    ITEM_KEYWORDS = {f: set(v) for f, v in json.loads(_KW_PATH.read_text(encoding="utf-8")).items()}
except Exception as e:
    print("item_keywords.json 을 읽지 못함 (관련성 검사 끔):", e)
    ITEM_KEYWORDS = {}


def related(text: str, func: str) -> bool:
    kws = ITEM_KEYWORDS.get(func)
    if not kws:
        return True
    s = re.sub(r"[^가-힣]", "", text)
    if any(ch in kws and ch in s for ch in ONE_CHAR):
        return True
    return any(s[i:i + 2] in kws for i in range(len(s) - 1))


ONE_CHAR = ["콤"]


def match_answer(text: str, expect: str):
    choices = {"location": ["상부", "하부", "양쪽"], "verdict": ["부적합", "적합", "해당없음"]}[expect]
    for c in choices:
        if c in text:
            return c
    return None
