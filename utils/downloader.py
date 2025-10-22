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


async def download_to_images_dir(url: str, images_dir: Path, prefer_video: bool = False, prefer_image: bool = False) -> str | None:
    """Download a media URL to images_dir and return the local file path.
    - prefer_video=True: 仅当 Content-Type 为 video/* 或 URL 后缀为常见视频后缀时保存；否则返回 None。
    - prefer_image=True: 仅当 Content-Type 为 image/* 或 URL 后缀为常见图片后缀时保存；否则返回 None。
    - 两者都 False: 尝试保存（不推荐）。
    """
    timeout = aiohttp.ClientTimeout(total=60)
    url_ext = _ext_from_url(url)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status // 100 != 2:
                    logger.error(f"下载媒体失败 HTTP {resp.status}: {url}")
                    return None
                ct = (resp.headers.get("Content-Type") or "").lower()
                # prefer_video: 仅接收视频
                if prefer_video:
                    is_video_ct = ct.startswith("video/")
                    is_video_ext = url_ext in {"mp4", "webm", "mov", "mkv"}
                    if not (is_video_ct or is_video_ext):
                        logger.warning(f"目标并非视频内容(ct={ct or 'unknown'}), 跳过下载: {url}")
                        return None
                # prefer_image: 仅接收图片
                if prefer_image and not prefer_video:
                    is_img_ct = ct.startswith("image/")
                    is_img_ext = url_ext in {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
                    if not (is_img_ct or is_img_ext):
                        logger.warning(f"目标并非图片内容(ct={ct or 'unknown'}), 跳过下载: {url}")
                        return None
                # 猜测后缀
                guessed = _ext_from_content_type(ct) or url_ext
                if not guessed:
                    guessed = "mp4" if prefer_video else ("png" if prefer_image else "bin")
                ext = guessed
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                uid = str(uuid.uuid4())[:8]
                images_dir.mkdir(exist_ok=True)
                prefix = "jm_media"
                file_path = images_dir / f"{prefix}_{ts}_{uid}.{ext}"
                data = await resp.read()
                file_path.write_bytes(data)
                logger.info(f"已下载媒体到本地: {file_path} (ct={ct or 'unknown'})")
                return str(file_path)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"下载媒体网络失败: {e}")
        return None
    except Exception as e:
        logger.error(f"下载媒体异常: {e}")
        return None
