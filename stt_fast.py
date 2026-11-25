import io
import re

import numpy as np

SR = 16000


def _looks_broken(text: str, sec: float) -> bool:
    if len(text.replace(" ", "")) > 12 * sec + 10:
        return True
    words = text.split()
    for n in (2, 3, 4):
        for i in range(len(words) - 3 * n):
            if words[i:i + n] == words[i + n:i + 2 * n] == words[i + 2 * n:i + 3 * n] == words[i + 3 * n:i + 4 * n]:
                return True
    return False


def load(model_dir: str, threads: int = 4, pad_s: float = 2.0, min_s: float = 3.0):
    import ctranslate2
    from faster_whisper import WhisperModel, decode_audio
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    m = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)
    tok = m.hf_tokenizer
    prompt = [[tok.token_to_id(t) for t in ("<|startoftranscript|>", "<|ko|>", "<|transcribe|>", "<|notimestamps|>")]]

    def full(a) -> str:
        segs, _ = m.transcribe(a, language="ko", beam_size=1, without_timestamps=True,
                               condition_on_previous_text=False, suppress_blank=False, vad_filter=True)
        return "".join(s.text for s in segs).strip()

    def run(audio) -> str:
        a = decode_audio(io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio)
        ts = get_speech_timestamps(a, VadOptions())
        if not ts:
            return ""
        c = collect_chunks(a, ts)
        c = c[0] if isinstance(c, tuple) else c
        a = np.concatenate(c) if isinstance(c, list) else c
        sec = len(a) / SR
        if sec > 29:
            return full(a)
        frames = int(min(30.0, max(min_s, np.ceil(sec + pad_s))) * 100)
        feat = m.feature_extractor(a)[:, :frames]
        x = np.full((feat.shape[0], frames), feat.min(), np.float32)
        x[:, :feat.shape[1]] = feat
        enc = m.model.encode(ctranslate2.StorageView.from_array(x[None]))
        out = m.model.generate(enc, prompt, beam_size=1, suppress_blank=False,
                               max_length=int(15 * sec + 30))
        text = tok.decode(out[0].sequences_ids[0]).strip()
        if _looks_broken(text, sec):
            return full(a)
        return text

    return run
