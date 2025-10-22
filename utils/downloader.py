import aiohttp
import asyncio
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import uuid
from astrbot.api import logger


def _ext_from_content_type(ct: str) -> str:
    if not ct:
        return ""
    ct = ct.lower()
    if "image/png" in ct:
        return "png"
    if "image/jpeg" in ct or "image/jpg" in ct:
        return "jpg"
    if "image/webp" in ct:
        return "webp"
    if "image/gif" in ct:
        return "gif"
    if "image/bmp" in ct:
        return "bmp"
    if ct.startswith("video/"):
        # video/mp4, video/webm, video/quicktime, video/x-matroska
        sub = ct.split("/", 1)[1]
        # quicktime 通常是 mov
        if sub.startswith("quicktime"):
            return "mov"
        if sub.startswith("x-matroska"):
            return "mkv"
        # 其余直接返回后缀
        return sub.split(";")[0]
    return ""


def _ext_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
        suffix = Path(path).suffix.lower().lstrip(".")
        return suffix
    except Exception:
        return ""


async def download_to_images_dir(url: str, images_dir: Path) -> str | None:
    """Download an image URL to images_dir and return the local file path.
    Picks extension from content-type or URL; falls back to .png.
    """
    timeout = aiohttp.ClientTimeout(total=60)
    ext_from_url = _ext_from_url(url)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status // 100 != 2:
                    logger.error(f"下载图片失败 HTTP {resp.status}: {url}")
                    return None
                ct = resp.headers.get("Content-Type", "")
                # 优先 Content-Type，再退 URL 后缀；默认：图片用 png，视频用 mp4
                guessed = _ext_from_content_type(ct)
                if not guessed:
                    guessed = ext_from_url
                if not guessed:
                    guessed = "mp4" if (ct and ct.lower().startswith("video/")) else "png"
                ext = guessed
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                uid = str(uuid.uuid4())[:8]
                images_dir.mkdir(exist_ok=True)
                file_path = images_dir / f"jm_image_{ts}_{uid}.{ext}"
                data = await resp.read()
                file_path.write_bytes(data)
                logger.info(f"已下载图片到本地: {file_path} (ct={ct or 'unknown'})")
                return str(file_path)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"下载图片网络失败: {e}")
        return None
    except Exception as e:
        logger.error(f"下载图片异常: {e}")
        return None
