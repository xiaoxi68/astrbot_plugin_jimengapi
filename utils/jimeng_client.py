import aiohttp
import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from astrbot.api import logger


@dataclass
class JimengConfig:
    base_url: str
    session_token: str
    model: str = "jimeng-4.0"
    default_ratio: str = "1:1"
    default_resolution: str = "2k"
    negative_prompt: str = ""
    sample_strength: float = 0.7
    response_format: str = "url"  # "url" | "b64_json"
    max_retry_attempts: int = 3
    video_model: str = "jimeng-video-3.0"
    video_stream: bool = True


async def _request_json(
    method: str,
    url: str,
    json: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout_total: int = 60,
    max_retry: int = 3,
) -> Optional[dict]:
    """Do a JSON HTTP request with simple retry for transient network errors."""
    headers = headers or {}
    timeout = aiohttp.ClientTimeout(total=timeout_total)
    for attempt in range(max_retry):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, json=json, headers=headers) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status // 100 == 2:
                        return data
                    logger.warning(f"HTTP {resp.status} when calling {url}: {data}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Network error on attempt {attempt + 1}/{max_retry}: {e}")
        # small exponential backoff
        await asyncio.sleep(min(2 ** attempt, 5))
    return None


async def _request_sse(
    url: str,
    json: dict,
    headers: dict,
    timeout_total: int = 120,
) -> Optional[str]:
    """Very simple SSE reader: returns concatenated 'delta.content' text."""
    import json as pyjson
    timeout = aiohttp.ClientTimeout(total=timeout_total)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=json, headers=headers) as resp:
                if resp.status // 100 != 2:
                    logger.warning(f"SSE HTTP {resp.status}")
                    return None
                acc = []
                async for raw_line in resp.content:
                    try:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            obj = pyjson.loads(payload)
                            choices = (obj.get("choices") or [])
                            for ch in choices:
                                delta = ch.get("delta") or {}
                                txt = delta.get("content")
                                if isinstance(txt, str):
                                    acc.append(txt)
                        except Exception:
                            # ignore malformed chunk
                            pass
                return "".join(acc) if acc else None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"SSE request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"SSE unexpected error: {e}")
        return None


def _extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    import re
    m = re.search(r"https?://\S+", text)
    if not m:
        return None
    url = m.group(0)
    # strip trailing markdown ) or ] if present
    return url.rstrip(")]")


async def generate_video(
    cfg: JimengConfig,
    prompt: str,
    model: Optional[str] = None,
    stream: Optional[bool] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Video generation via /v1/chat/completions.
    Returns (video_url, raw_text). If URL未找到，raw_text保留SSE合成文本以便上层回退展示。
    """
    url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.session_token}",
    }
    use_model = (model or cfg.video_model)
    use_stream = cfg.video_stream if stream is None else stream

    payload = {
        "model": use_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": bool(use_stream),
    }

    if use_stream:
        text = await _request_sse(url, payload, headers)
        if not text:
            return None, None
        vurl = _extract_first_url(text)
        return (vurl, text)
    else:
        data = await _request_json("POST", url, json=payload, headers=headers, max_retry=cfg.max_retry_attempts)
        if not data:
            return None, None
        try:
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            content = ""
        vurl = _extract_first_url(content)
        return (vurl, content or None)


async def generate_image(
    cfg: JimengConfig,
    prompt: str,
    ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    response_format: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Text-to-image via /v1/images/generations.
    Returns: (image_url, b64_data)
    """
    url = cfg.base_url.rstrip("/") + "/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.session_token}",
    }
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "ratio": ratio or cfg.default_ratio,
        "resolution": resolution or cfg.default_resolution,
    }
    if (negative_prompt or cfg.negative_prompt):
        payload["negative_prompt"] = negative_prompt or cfg.negative_prompt

    rf = (response_format or cfg.response_format).lower().strip()
    if rf in ("url", "b64_json"):
        payload["response_format"] = rf

    data = await _request_json("POST", url, json=payload, headers=headers, max_retry=cfg.max_retry_attempts)
    if not data:
        return None, None

    # Expected response:
    # { "created": 1759058, "data": [{"url": "..."}] } or b64_json style
    try:
        first = (data.get("data") or [None])[0]
        if not first:
            return None, None
        if "url" in first:
            return first["url"], None
        if "b64_json" in first:
            return None, first["b64_json"]
    except Exception as e:
        logger.error(f"Unexpected response payload: {e}")
    return None, None


async def compose_image(
    cfg: JimengConfig,
    prompt: str,
    image_urls: List[str],
    ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    sample_strength: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    response_format: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Image-to-image via /v1/images/compositions using JSON with URL images.
    Returns: (image_url, b64_data)
    """
    url = cfg.base_url.rstrip("/") + "/v1/images/compositions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.session_token}",
    }
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "images": image_urls,
        "ratio": ratio or cfg.default_ratio,
        "resolution": resolution or cfg.default_resolution,
    }
    if sample_strength is not None:
        payload["sample_strength"] = float(sample_strength)
    if (negative_prompt or cfg.negative_prompt):
        payload["negative_prompt"] = negative_prompt or cfg.negative_prompt

    rf = (response_format or cfg.response_format).lower().strip()
    if rf in ("url", "b64_json"):
        payload["response_format"] = rf

    data = await _request_json("POST", url, json=payload, headers=headers, max_retry=cfg.max_retry_attempts)
    if not data:
        return None, None

    try:
        first = (data.get("data") or [None])[0]
        if not first:
            return None, None
        if "url" in first:
            return first["url"], None
        if "b64_json" in first:
            return None, first["b64_json"]
    except Exception as e:
        logger.error(f"Unexpected response payload: {e}")
    return None, None
