import os
import subprocess
import threading

_proc = None
_lock = threading.Lock()
_cmd = None
LAST = {}


def _start():
    global _proc
    _proc = subprocess.Popen(_cmd, cwd=os.path.dirname(_cmd[0]), text=True, encoding="utf-8",
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1)


def load(model_path: str):
    global _cmd
    model_path = os.path.abspath(os.path.expanduser(model_path))
    exe = os.path.expanduser(os.environ.get("ONMAEUM_ESC_DEMO", os.path.join(os.path.dirname(model_path), "esc_demo")))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f".rkllm 파일이 없습니다: {model_path}")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"esc_demo 가 없습니다: {exe}")
    _cmd = [exe, model_path, "96", "256", "1"]
    _start()

    def generate(text: str) -> str:
        return _generate(text)

    return generate


def _generate(text: str) -> str:
    with _lock:
        if _proc is None or _proc.poll() is not None:
            _start()
        _proc.stdin.write(text.replace("\n", " ").strip() + "\n")
        _proc.stdin.flush()
        while True:
            line = _proc.stdout.readline()
            if not line:
                raise RuntimeError("LLM 프로세스(esc_demo)가 종료됐습니다")
            if "\t" in line:
                f = line.rstrip("\n").split("\t")
                LAST.update(prefill_ms=float(f[1]), gen_ms=float(f[3]), gen_tok=int(f[4]), mem_mb=float(f[5]))
                out = f[0].strip()
                if not out.endswith("<esc_end>"):
                    out += "<esc_end>"
                return out


def release():
    global _proc
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.stdin.close()
            _proc.wait(timeout=5)
        except Exception:
            _proc.kill()
    _proc = None
