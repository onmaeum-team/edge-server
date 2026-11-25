import glob
import io
import os
import re
import wave

DEFAULT_DIR = "/home/orangepi/models/tts/vits-mimic3-ko_KO-kss_low"

_DIGITS = "영일이삼사오육칠팔구"


def _num_to_ko(m: re.Match) -> str:
    s = m.group(0)
    if "." in s:
        a, b = s.split(".", 1)
        return _int_ko(a) + " 점 " + " ".join(_DIGITS[int(c)] for c in b)
    return _int_ko(s)


def _int_ko(s: str) -> str:
    n = int(s)
    if n == 0:
        return "영"
    out = ""
    for unit, name in ((10000, "만"), (1000, "천"), (100, "백"), (10, "십")):
        q, n = divmod(n, unit)
        if q:
            out += ("" if q == 1 else _int_ko(str(q))) + name
    if n:
        out += _DIGITS[n]
    return out


_NATIVE1 = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE10 = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]


def _count_ko(m: re.Match) -> str:
    n = int(m.group(1))
    if n == 0 or n >= 100:
        return _int_ko(m.group(1)) + " " + m.group(2)
    ten, one = divmod(n, 10)
    w = _NATIVE10[ten] + _NATIVE1[one]
    return (w if n != 20 else "스무") + " " + m.group(2)


def clean(text: str) -> str:
    t = text.replace("mm", " 밀리").replace("MΩ", " 메가옴").replace("%", " 퍼센트")
    code = lambda m: " 점 ".join(_int_ko(x) for x in m.group(0).split("."))
    t = re.sub(r"\d+(?:\.\d+){2,}", code, t)
    t = re.sub(r"\b\d\.1[0-2]\b", code, t)
    t = re.sub(r"(\d+)\s*(개|가지|군데)", _count_ko, t)
    t = re.sub(r"\d+(?:\.\d+)?", _num_to_ko, t)
    t = t.replace("?", ".")
    t = re.sub(r"[·/]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load():
    import numpy as np
    import sherpa_onnx

    d = os.environ.get("ONMAEUM_TTS_DIR", DEFAULT_DIR)
    onnx = [p for p in glob.glob(os.path.join(d, "*.onnx"))]
    if not onnx:
        raise FileNotFoundError(f"TTS 모델(.onnx)이 없습니다: {d}")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=onnx[0],
                tokens=os.path.join(d, "tokens.txt"),
                data_dir=os.path.join(d, "espeak-ng-data") if os.path.isdir(os.path.join(d, "espeak-ng-data")) else "",
            ),
            num_threads=int(os.environ.get("ONMAEUM_TTS_THREADS", "2")),
            provider="cpu",
        ),
        max_num_sentences=1,
    )
    tts = sherpa_onnx.OfflineTts(cfg)
    speed = float(os.environ.get("ONMAEUM_TTS_SPEED", "1.0"))

    def synth(text: str) -> bytes:
        audio = tts.generate(clean(text), sid=0, speed=speed)
        pcm = (np.clip(np.asarray(audio.samples, dtype=np.float32), -1, 1) * 32767).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(audio.sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()

    return synth
