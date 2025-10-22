import time
import uuid
from contextlib import contextmanager
from astrbot.api import logger


def mk_req_id() -> str:
    return str(uuid.uuid4())[:8]


def _fmt_kv(**kv) -> str:
    items = []
    for k, v in kv.items():
        try:
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            items.append(f"{k}={s}")
        except Exception:
            items.append(f"{k}=<unrepr>")
    return " ".join(items)


def log_with(level: str, tag: str, msg: str, **kv):
    line = f"[{tag}] {msg}"
    if kv:
        line += " | " + _fmt_kv(**kv)
    lvl = (level or "INFO").upper()
    if lvl == "DEBUG":
        logger.debug(line)
    elif lvl in ("WARN", "WARNING"):
        logger.warning(line)
    elif lvl == "ERROR":
        logger.error(line)
    else:
        logger.info(line)


@contextmanager
def timing(level: str, tag: str, name: str, **kv):
    t0 = time.time()
    try:
        yield
    finally:
        cost_ms = int((time.time() - t0) * 1000)
        log_with(level, tag, f"timing:{name}", cost_ms=cost_ms, **kv)
