import re


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


DIG = "영일이삼사오육칠팔구"


def sino(n):
    n = int(n)
    if n == 0:
        return "영"
    out = ""
    for unit, v in (("천", 1000), ("백", 100), ("십", 10)):
        q, n = divmod(n, v)
        if q:
            out += ("" if q == 1 else DIG[q]) + unit
    return out + (DIG[n] if n else "")


def read_num(m):
    ip, _, fp = m.group(0).partition(".")
    return sino(ip) + (" 점 " + " ".join(DIG[int(c)] for c in fp) if fp else "")


LETTER = {"A": "에이", "B": "비", "C": "씨"}


def to_spoken(t):

    t = re.sub(r"\d+(?:\.\d+){2,}", lambda m: " " + " 점 ".join(sino(p) for p in m.group().split(".")) + " ", t)
    t = re.sub(r"\d+(\.\d+)?", lambda m: " " + read_num(m) + " ", t)
    t = re.sub(r"(?<![A-Za-z])([ABC])(?![A-Za-z])", lambda m: LETTER[m.group(1)], t)
    return clean(t)


NUMK = "영일이삼사오육칠팔구십백"


def stt_to_llm(t):
    t = re.sub(r"(?i)(\d)\s*mm\b", r"\1 밀리", t)
    t = re.sub(r"(?i)(\d)\s*cm\b", r"\1 센티", t)
    t = re.sub(r"밀리미터|미리미터", "밀리", t)
    t = re.sub(rf"(?<=[\d{NUMK}])(\s*)미리", r"\1밀리", t)
    t = re.sub(r"(\d)\s*(?:일리|리)(?![가-힣])", r"\1 밀리", t)
    t = re.sub(r"(?<!\d)[.,!?~]|[.,!?~](?!\d)", " ", t)
    return clean(to_spoken(t))


def norm_for_cer(t):
    t = re.sub(r"(?i)mm\b|밀리미터", "밀리", t)
    t = to_spoken(t)
    return re.sub(r"[^가-힣a-zA-Z]", "", t)


def cer(ref, hyp):
    r, h = norm_for_cer(ref), norm_for_cer(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


TERMS = ["콤", "스커트", "스텝", "요수리", "요주의", "인입구", "핸드레일", "비상정지", "기동", "간극", "틈새",
         "데마케이션", "디플렉터", "난간", "맞물림", "트레드"]


def term_hits(ref, hyp):
    hits = tot = 0
    for t in TERMS:
        if t in ref:
            tot += 1; hits += t in hyp
    for n in re.findall(r"\d+(?:\.\d+)?", ref):
        tot += 1; hits += n in hyp
    return hits, tot
