from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply

from .utils.jimeng_client import JimengConfig, generate_image as jm_generate, compose_image as jm_compose
from .utils.jimeng_client import generate_video as jm_video
from .utils.file_send_server import send_file
from .utils.downloader import download_to_images_dir
from .utils.plugin_logger import mk_req_id, log_with, timing

import aiofiles
import base64
import asyncio
import time
import os
from pathlib import Path


@register("astrbot_plugin_jimengapi", "薄暝", "对接即梦2API，支持生图，图生图与文生视频", "0.2.0")
class JimengPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        # 配置项（可被全局配置覆盖）
        self.base_url = config.get("base_url", "http://localhost:5100").strip()
        self.session_token = config.get("session_token", "").strip()
        # 支持多个 Session（轮询），向后兼容旧字段
        tokens_cfg = config.get("session_tokens", [])
        if isinstance(tokens_cfg, list):
            self.session_tokens = [str(x).strip() for x in tokens_cfg if str(x).strip()]
        else:
            self.session_tokens = []
        if self.session_token and not self.session_tokens:
            self.session_tokens = [self.session_token]
        self.model = config.get("model", "jimeng-4.0").strip()
        self.default_ratio = config.get("default_ratio", "1:1").strip()
        self.default_resolution = config.get("default_resolution", "2k").strip()
        self.negative_prompt = config.get("negative_prompt", "")
        self.sample_strength = float(config.get("sample_strength", 0.7))
        self.response_format = config.get("response_format", "url").strip()
        self.max_retry_attempts = int(config.get("max_retry_attempts", 3))
        # 视频默认配置
        self.video_model = config.get("video_model", "jimeng-video-3.0").strip()
        self.video_stream = bool(config.get("video_stream", True))

        # 日志配置（不占用插件配置项，可用环境变量覆盖）
        self.log_level = os.getenv("JIMENG_LOG_LEVEL", "DEBUG").upper()
        self.log_verbose = os.getenv("JIMENG_LOG_VERBOSE", "true").lower() in ("1","true","yes","on")
        self.log_trace_http = os.getenv("JIMENG_LOG_TRACE_HTTP", "false").lower() in ("1","true","yes","on")
        self.log_media_headers = os.getenv("JIMENG_LOG_MEDIA_HEADERS", "false").lower() in ("1","true","yes","on")
        self.log_policy = os.getenv("JIMENG_LOG_POLICY", "true").lower() in ("1","true","yes","on")
        self.log_timing = os.getenv("JIMENG_LOG_TIMING", "true").lower() in ("1","true","yes","on")

        # 权限与限流配置
        self.group_list_mode = str(config.get("group_list_mode", "blacklist")).strip().lower()
        self.groups = [str(x).strip() for x in config.get("groups", []) if str(x).strip()]
        # 兼容旧字段（若新字段未配置且旧字段存在）
        if not self.groups:
            old_wl = config.get("group_whitelist")
            old_bl = config.get("group_blacklist")
            if isinstance(old_wl, list) and old_wl:
                self.group_list_mode = "whitelist"
                self.groups = [str(x).strip() for x in old_wl if str(x).strip()]
            elif isinstance(old_bl, list) and old_bl:
                self.group_list_mode = "blacklist"
                self.groups = [str(x).strip() for x in old_bl if str(x).strip()]
        self.rate_limit_enabled = bool(config.get("rate_limit_enabled", False))
        self.rate_limit_window_minutes = int(config.get("rate_limit_window_minutes", 10))
        self.rate_limit_max_calls = int(config.get("rate_limit_max_calls", 5))

        # 定时清理配置
        self.cleanup_enabled = bool(config.get("cleanup_enabled", True))
        self.cleanup_every_days = int(config.get("cleanup_every_days", 3))

        # 运行时状态
        self._usage = {}  # group_id -> list[timestamps]
        self._usage_lock = asyncio.Lock()
        self._cleanup_task = None
        self._cleanup_stop = asyncio.Event()

        self.callback_api_base = config.get("callback_api_base", "").strip()
        self.nap_server_address = config.get("nap_server_address", "localhost").strip()
        self.nap_server_port = int(config.get("nap_server_port", 3658))

        self._global_loaded = False

    async def initialize(self):
        """启动后台清理任务（若启用）。"""
        if self.cleanup_enabled and (self._cleanup_task is None or self._cleanup_task.done()):
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("即梦插件：已启动定时清理任务")

    async def terminate(self):
        """停止后台清理任务。"""
        try:
            self._cleanup_stop.set()
            if self._cleanup_task:
                self._cleanup_task.cancel()
        except Exception as e:
            logger.debug(f"停止清理任务异常: {e}")

    async def _cleanup_once(self):
        try:
            base = Path(__file__).parent
            total = 0
            for folder in ("images", "videos"):
                p = base / folder
                if not p.exists() or not p.is_dir():
                    continue
                for fp in p.iterdir():
                    try:
                        if not fp.is_file():
                            continue
                        fp.unlink(missing_ok=True)
                        total += 1
                    except Exception as e:
                        logger.debug(f"清理 {fp} 失败: {e}")
            if total:
                logger.info(f"即梦插件：清理完成，本次删除 {total} 个文件")
        except Exception as e:
            logger.error(f"执行清理任务失败: {e}")

    async def _cleanup_loop(self):
        try:
            # 初次延迟，避免冷启动拥挤
            await asyncio.sleep(10)
            while not self._cleanup_stop.is_set():
                await self._cleanup_once()
                days = max(1, int(self.cleanup_every_days))
                try:
                    await asyncio.wait_for(self._cleanup_stop.wait(), timeout=days * 86400)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"清理任务异常退出: {e}")

    async def _load_global_config(self):
        if self._global_loaded:
            return
        try:
            cfg = await sp.global_get("astrbot_plugin_jimengapi", {})
            if not isinstance(cfg, dict):
                self._global_loaded = True
                return
            self.base_url = cfg.get("base_url", self.base_url)
            # tokens: 新优先，旧字段兼容
            tokens = cfg.get("session_tokens")
            if isinstance(tokens, list):
                self.session_tokens = [str(x).strip() for x in tokens if str(x).strip()] or self.session_tokens
            self.session_token = cfg.get("session_token", self.session_token)
            if self.session_token and not self.session_tokens:
                self.session_tokens = [self.session_token]
            self.model = cfg.get("model", self.model)
            self.default_ratio = cfg.get("default_ratio", self.default_ratio)
            self.default_resolution = cfg.get("default_resolution", self.default_resolution)
            self.negative_prompt = cfg.get("negative_prompt", self.negative_prompt)
            self.sample_strength = float(cfg.get("sample_strength", self.sample_strength))
            self.response_format = cfg.get("response_format", self.response_format)
            self.max_retry_attempts = int(cfg.get("max_retry_attempts", self.max_retry_attempts))
            self.callback_api_base = cfg.get("callback_api_base", self.callback_api_base)
            # 视频配置（全局覆盖）
            self.video_model = cfg.get("video_model", self.video_model)
            self.video_stream = bool(cfg.get("video_stream", self.video_stream))
            # 权限与限流（全局覆盖）
            if "group_list_mode" in cfg:
                self.group_list_mode = str(cfg.get("group_list_mode", self.group_list_mode)).strip().lower()
            if "groups" in cfg and isinstance(cfg.get("groups"), list):
                self.groups = [str(x).strip() for x in cfg.get("groups", []) if str(x).strip()]
            # 旧字段回退
            if not self.groups:
                wl = cfg.get("group_whitelist")
                bl = cfg.get("group_blacklist")
                if isinstance(wl, list) and wl:
                    self.group_list_mode = "whitelist"
                    self.groups = [str(x).strip() for x in wl if str(x).strip()]
                elif isinstance(bl, list) and bl:
                    self.group_list_mode = "blacklist"
                    self.groups = [str(x).strip() for x in bl if str(x).strip()]
            if "rate_limit_enabled" in cfg:
                self.rate_limit_enabled = bool(cfg.get("rate_limit_enabled", self.rate_limit_enabled))
            if "rate_limit_window_minutes" in cfg:
                self.rate_limit_window_minutes = int(cfg.get("rate_limit_window_minutes", self.rate_limit_window_minutes))
            if "rate_limit_max_calls" in cfg:
                self.rate_limit_max_calls = int(cfg.get("rate_limit_max_calls", self.rate_limit_max_calls))
            self._global_loaded = True
        except Exception as e:
            logger.error(f"加载全局配置失败: {e}")
            self._global_loaded = True

    def _cfg(self) -> JimengConfig:
        return JimengConfig(
            base_url=self.base_url,
            session_token=self.session_token,
            session_tokens=self.session_tokens,
            model=self.model,
            default_ratio=self.default_ratio,
            default_resolution=self.default_resolution,
            negative_prompt=self.negative_prompt,
            sample_strength=self.sample_strength,
            response_format=self.response_format,
            max_retry_attempts=self.max_retry_attempts,
            video_model=self.video_model,
            video_stream=self.video_stream,
        )

    async def _save_b64_image(self, b64_data: str, fmt: str = "png") -> str:
        from datetime import datetime
        import uuid
        images_dir = Path(__file__).parent / "images"
        images_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        file_path = images_dir / f"jm_image_{ts}_{uid}.{fmt}"
        data = base64.b64decode(b64_data)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data)
        return str(file_path)

    async def _image_from_path_with_callback(self, image_path: str) -> Image:
        # 优先使用回调URL以减少跨机文件发送
        callback_api_base = self.context.get_config().get("callback_api_base") or self.callback_api_base
        if callback_api_base:
            try:
                comp = Image.fromFileSystem(image_path)
                url = await comp.convert_to_web_link()
                return Image.fromURL(url)
            except Exception as e:
                logger.warning(f"生成回调URL失败，使用本地文件发送: {e}")
        return Image.fromFileSystem(image_path)

    async def _check_group_policy(self, event: AstrMessageEvent) -> tuple[bool, str]:
        """群黑白名单与每群频率限制检查。私聊不受限。"""
        gid = None
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        # 仅群聊受限
        if not gid:
            return True, ""
        gid = str(gid)
        mode = (self.group_list_mode or "blacklist").lower()
        groups = self.groups or []
        if mode == "whitelist":
            if groups and gid not in groups:
                return False, "本群未在白名单，禁止使用此插件。"
        else:  # blacklist
            if gid in groups:
                return False, "本群在黑名单中，已禁止使用此插件。"
        # 频率限制
        if not self.rate_limit_enabled:
            return True, ""
        window_sec = max(1, int(self.rate_limit_window_minutes) * 60)
        max_calls = max(1, int(self.rate_limit_max_calls))
        now = time.time()
        async with self._usage_lock:
            lst = self._usage.get(gid, [])
            # 清理窗口外
            lst = [ts for ts in lst if now - ts < window_sec]
            if len(lst) >= max_calls:
                return False, f"频率限制：每{self.rate_limit_window_minutes}分钟最多{self.rate_limit_max_calls}次，本群已达上限，请稍后再试。"
            # 记录一次
            lst.append(now)
            self._usage[gid] = lst
        return True, ""

    async def _component_to_http_url(self, comp) -> str | None:
        """尽量把任意图片组件转换为可用于对接 API 的 http(s) 链接。
        优先使用 convert_to_web_link；若缺失，则回退到属性 url/file；
        如果仅有本地 path，可尝试转为回调直链。
        """
        # 1) 新版 Image 可能有 convert_to_web_link
        try:
            fn = getattr(comp, "convert_to_web_link", None)
            if callable(fn):
                return await fn()
        except Exception as e:
            logger.debug(f"convert_to_web_link 失败，继续回退: {e}")
        # 2) 旧组件字段回退: url / file
        for attr in ("url", "file"):
            try:
                val = getattr(comp, attr, None)
            except Exception:
                val = None
            if isinstance(val, str) and val.startswith("http"):
                return val
        # 3) 本地路径回退（需要转直链）
        try:
            path_val = getattr(comp, "path", None)
            if isinstance(path_val, str) and path_val:
                img_comp = Image.fromFileSystem(path_val)
                try:
                    return await img_comp.convert_to_web_link()
                except Exception:
                    # 回退最终返回 None，让上层决定是否忽略
                    return None
        except Exception:
            pass
        return None

    async def _collect_image_urls_from_event(self, event: AstrMessageEvent) -> list:
        urls: list[str] = []
        if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    url = await self._component_to_http_url(comp)
                    if url:
                        urls.append(url)
                elif isinstance(comp, Reply) and getattr(comp, 'chain', None):
                    for r in comp.chain:
                        # Reply 链中的 Image 可能是旧版结构，没有 convert_to_web_link
                        if isinstance(r, Image):
                            url = await self._component_to_http_url(r)
                            if url:
                                urls.append(url)
        return urls

    async def _video_from_path(self, video_path: str):
        try:
            return Video.fromFileSystem(video_path)
        except Exception as e:
            logger.warning(f"构造视频组件失败，回退为链接: {e}")
            return Plain(video_path)

    @filter.command("即梦帮助")
    async def jhelp(self, event: AstrMessageEvent):
        """列出所有指令"""
        text = (
            "即梦2 插件指令:\n"
            "- /即梦生图 <prompt> [ratio=1:1] [res=2k] [fmt=url|b64]\n"
            "  说明: 生成图片。\n"
            "- /即梦改图 <prompt> [ratio=1:1] [res=2k] [fmt=url|b64] [strength=0.7]\n"
            "  说明: 修改图片；需附带或引用至少一张图片。\n"
            "- /即梦视频 <prompt> [model=jimeng-video-3.0] [stream=true|false]\n"
            "  说明: 生成视频。\n"
        )
        yield event.plain_result(text)

    # 命令：/jvideo <prompt> [model=jimeng-video-3.0] [stream=true|false]
    @filter.command("即梦视频")
    async def jvideo(self, event: AstrMessageEvent):
        """生成视频"""
        await self._load_global_config()
        # 禁用默认 LLM 回复，避免与本指令重复发消息
        try:
            event.call_llm = False
        except Exception:
            pass
        req = mk_req_id()
        gid = None
        sid = None
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        try:
            sid = event.get_session_id()
        except Exception:
            sid = None
        tag = f"JIMENG|video|req={req}|gid={gid}|sid={sid}"
        if self.log_verbose:
            log_with("INFO", tag, "recv", text=event.message_str)
        ok, msg = await self._check_group_policy(event)
        if not ok:
            log_with("WARN", tag, "policy_block", reason=msg)
            yield event.plain_result(msg)
            return
        text = event.message_str or ""
        if not text:
            yield event.plain_result("请提供视频描述，如 /即梦视频 海浪拍打海岸")
            return

        model = None
        stream_opt = None
        parts = text.split()
        prompt = " ".join([p for p in parts if "=" not in p])
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip().lower(); v = v.strip()
                if k == "model":
                    model = v
                elif k == "stream":
                    stream_opt = (v.lower() in ("1","true","yes","on"))

        cfg = self._cfg()
        try:
            tokens = self.session_tokens or ([self.session_token] if self.session_token else [])
            if not tokens:
                log_with("ERROR", tag, "no_tokens")
                yield event.plain_result("未配置 session_tokens 或 session_token，无法生成视频。")
                return
            last_raw = None
            for i, tok in enumerate(tokens, start=1):
                log_with("INFO", tag, "try_token", idx=i)
                if self.log_timing:
                    with timing("INFO", tag, "api_video", model=self.video_model, stream=bool(stream_opt if stream_opt is not None else self.video_stream)):
                        vurl, raw = await jm_video(cfg, prompt=prompt, model=model, stream=stream_opt, session_tokens=[tok])
                else:
                    vurl, raw = await jm_video(cfg, prompt=prompt, model=model, stream=stream_opt, session_tokens=[tok])
                if not vurl:
                    last_raw = raw
                    continue
                videos_dir = Path(__file__).parent / "videos"
                if self.log_timing:
                    with timing("INFO", tag, "download_video", url=vurl):
                        video_path = await download_to_images_dir(vurl, videos_dir, prefer_video=True)
                else:
                    video_path = await download_to_images_dir(vurl, videos_dir, prefer_video=True)
                if not video_path:
                    log_with("WARN", tag, "video_download_pending", url=vurl)
                    # 下一个 token
                    continue
                if self.nap_server_address and self.nap_server_address != "localhost":
                    if self.log_timing:
                        with timing("INFO", tag, "nap_transfer"):
                            sent = await send_file(video_path, self.nap_server_address, self.nap_server_port)
                    else:
                        sent = await send_file(video_path, self.nap_server_address, self.nap_server_port)
                    video_path = sent or video_path
                comp = await self._video_from_path(video_path)
                yield event.chain_result([comp])
                return
            # 全部 token 尝试后仍失败
            log_with("ERROR", tag, "video_all_tokens_failed")
            yield event.plain_result("视频生成失败，请稍后再试。")
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            yield event.plain_result(f"视频生成失败：{e}")

    # removed llm_tool: jimeng-video-gen
    async def llm_video_gen(self, event: AstrMessageEvent, video_description: str, prefer_stream: bool = True):
        await self._load_global_config()
        cfg = self._cfg()
        try:
            vurl, raw = await jm_video(cfg, prompt=video_description, stream=prefer_stream, session_tokens=self.session_tokens)
            if vurl:
                videos_dir = Path(__file__).parent / "videos"
                video_path = await download_to_images_dir(vurl, videos_dir, prefer_video=True)
                if not video_path:
                    yield event.chain_result([Plain(vurl)])
                    return
                if self.nap_server_address and self.nap_server_address != "localhost":
                    sent = await send_file(video_path, self.nap_server_address, self.nap_server_port)
                    video_path = sent or video_path
                comp = await self._video_from_path(video_path)
                yield event.chain_result([comp])
                return
            yield event.chain_result([Plain(raw or "视频生成完成但未获取直链")])
        except Exception as e:
            logger.error(f"LLM视频工具失败: {e}")
            yield event.chain_result([Plain(f"视频生成失败：{e}")])

    # 命令：/jgen <prompt> [ratio=1:1] [res=2k] [fmt=url|b64]
    @filter.command("即梦生图")
    async def jgen(self, event: AstrMessageEvent):
        """生成图片"""
        await self._load_global_config()
        try:
            event.call_llm = False
        except Exception:
            pass
        req = mk_req_id()
        gid = None
        sid = None
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        try:
            sid = event.get_session_id()
        except Exception:
            sid = None
        tag = f"JIMENG|image_gen|req={req}|gid={gid}|sid={sid}"
        if self.log_verbose:
            log_with("INFO", tag, "recv", text=event.message_str)
        ok, msg = await self._check_group_policy(event)
        if not ok:
            log_with("WARN", tag, "policy_block", reason=msg)
            yield event.plain_result(msg)
            return
        text = event.message_str or ""
        if not text:
            yield event.plain_result("请提供生成描述，如 /即梦生图 一只可爱的小猫咪")
            return

        # 解析简单的 kv 选项
        ratio, res, fmt = None, None, None
        prompt = text
        # 简单解析：用空格分隔，形如 key=value
        parts = text.split()
        if len(parts) > 1:
            # 第一段视为prompt的起始；后续解析 kv
            prompt = " ".join([p for p in parts if "=" not in p])
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k in ("ratio", "r"):
                        ratio = v
                    elif k in ("res", "resolution"):
                        res = v
                    elif k in ("fmt", "format"):
                        fmt = v

        cfg = self._cfg()
        try:
            tokens = self.session_tokens or ([self.session_token] if self.session_token else [])
            if not tokens:
                yield event.plain_result("未配置 session_tokens 或 session_token，无法生成图片。")
                return
            for i, tok in enumerate(tokens, start=1):
                log_with("INFO", tag, "try_token", idx=i)
                if self.log_timing:
                    with timing("INFO", tag, "api_image_gen", model=self.model, ratio=ratio or self.default_ratio, res=res or self.default_resolution):
                        image_url, b64 = await jm_generate(
                            cfg,
                            prompt=prompt,
                            ratio=ratio,
                            resolution=res,
                            negative_prompt=None,
                            response_format=fmt,
                            session_tokens=[tok],
                        )
                else:
                    image_url, b64 = await jm_generate(
                        cfg,
                        prompt=prompt,
                        ratio=ratio,
                        resolution=res,
                        negative_prompt=None,
                        response_format=fmt,
                        session_tokens=[tok],
                    )
                if not image_url and not b64:
                    continue
                if image_url:
                    images_dir = Path(__file__).parent / "images"
                    if self.log_timing:
                        with timing("INFO", tag, "download_image", url=image_url):
                            image_path = await download_to_images_dir(image_url, images_dir, prefer_image=True)
                    else:
                        image_path = await download_to_images_dir(image_url, images_dir, prefer_image=True)
                    if not image_path:
                        log_with("WARN", tag, "image_download_failed", url=image_url)
                        continue
                    if self.nap_server_address and self.nap_server_address != "localhost":
                        if self.log_timing:
                            with timing("INFO", tag, "nap_transfer"):
                                sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                        else:
                            sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                        image_path = sent or image_path
                    comp = await self._image_from_path_with_callback(image_path)
                    yield event.chain_result([comp])
                    return
                # b64 成功
                image_path = await self._save_b64_image(b64, fmt="png")
                if self.nap_server_address and self.nap_server_address != "localhost":
                    sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    image_path = sent or image_path
                comp = await self._image_from_path_with_callback(image_path)
                yield event.chain_result([comp])
                return
            yield event.plain_result("生成失败，请检查 API 配置或稍后再试。")
        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result(f"生图失败：{e}")

    # 命令：/jedit <prompt> [ratio=1:1] [res=2k] [fmt=url|b64] [strength=0.7]
    @filter.command("即梦改图")
    async def jedit(self, event: AstrMessageEvent):
        """修改图片"""
        await self._load_global_config()
        try:
            event.call_llm = False
        except Exception:
            pass
        req = mk_req_id()
        gid = None
        sid = None
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        try:
            sid = event.get_session_id()
        except Exception:
            sid = None
        tag = f"JIMENG|image_edit|req={req}|gid={gid}|sid={sid}"
        if self.log_verbose:
            log_with("INFO", tag, "recv", text=event.message_str)
        ok, msg = await self._check_group_policy(event)
        if not ok:
            log_with("WARN", tag, "policy_block", reason=msg)
            yield event.plain_result(msg)
            return
        text = event.message_str or ""
        if not text:
            yield event.plain_result("请提供修改描述，并附带或引用图片，如 /即梦改图 转油画风格")
            return

        ratio, res, fmt, strength = None, None, None, None
        prompt = text
        parts = text.split()
        if len(parts) > 1:
            prompt = " ".join([p for p in parts if "=" not in p])
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k in ("ratio", "r"):
                        ratio = v
                    elif k in ("res", "resolution"):
                        res = v
                    elif k in ("fmt", "format"):
                        fmt = v
                    elif k in ("strength", "s"):
                        try:
                            strength = float(v)
                        except Exception:
                            pass

        image_urls = await self._collect_image_urls_from_event(event)
        if not image_urls:
            yield event.plain_result("未检测到参考图片，请附带或引用至少一张图片。")
            return

        cfg = self._cfg()
        try:
            tokens = self.session_tokens or ([self.session_token] if self.session_token else [])
            if not tokens:
                yield event.plain_result("未配置 session_tokens 或 session_token，无法修改图片。")
                return
            for i, tok in enumerate(tokens, start=1):
                log_with("INFO", tag, "try_token", idx=i)
                if self.log_timing:
                    with timing("INFO", tag, "api_image_edit", model=self.model, imgs=len(image_urls)):
                        image_url, b64 = await jm_compose(
                            cfg,
                            prompt=prompt,
                            image_urls=image_urls,
                            ratio=ratio,
                            resolution=res,
                            sample_strength=strength,
                            negative_prompt=None,
                            response_format=fmt,
                            session_tokens=[tok],
                        )
                else:
                    image_url, b64 = await jm_compose(
                        cfg,
                        prompt=prompt,
                        image_urls=image_urls,
                        ratio=ratio,
                        resolution=res,
                        sample_strength=strength,
                        negative_prompt=None,
                        response_format=fmt,
                        session_tokens=[tok],
                    )
                if not image_url and not b64:
                    continue
                if image_url:
                    images_dir = Path(__file__).parent / "images"
                    if self.log_timing:
                        with timing("INFO", tag, "download_image", url=image_url):
                            image_path = await download_to_images_dir(image_url, images_dir, prefer_image=True)
                    else:
                        image_path = await download_to_images_dir(image_url, images_dir, prefer_image=True)
                    if not image_path:
                        log_with("WARN", tag, "image_download_failed", url=image_url)
                        continue
                    if self.nap_server_address and self.nap_server_address != "localhost":
                        if self.log_timing:
                            with timing("INFO", tag, "nap_transfer"):
                                sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                        else:
                            sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                        image_path = sent or image_path
                    comp = await self._image_from_path_with_callback(image_path)
                    yield event.chain_result([comp])
                    return
                image_path = await self._save_b64_image(b64, fmt="png")
                if self.nap_server_address and self.nap_server_address != "localhost":
                    sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    image_path = sent or image_path
                comp = await self._image_from_path_with_callback(image_path)
                yield event.chain_result([comp])
                return
            yield event.plain_result("改图失败，请检查 API 配置或稍后再试。")
        except Exception as e:
            logger.error(f"改图失败: {e}")
            yield event.plain_result(f"改图失败：{e}")

    # LLM 工具：方便在会话代理中被调用
    # removed llm_tool: jimeng-image-gen
    async def llm_image_gen(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True):
        await self._load_global_config()
        cfg = self._cfg()
        image_urls = []
        if use_reference_images:
            image_urls = await self._collect_image_urls_from_event(event)

        try:
            if image_urls:
                image_url, b64 = await jm_compose(
                    cfg,
                    prompt=image_description,
                    image_urls=image_urls,
                    session_tokens=self.session_tokens,
                )
            else:
                image_url, b64 = await jm_generate(cfg, prompt=image_description, session_tokens=self.session_tokens)

            if not image_url and not b64:
                yield event.chain_result([Plain("图像生成失败，请稍后再试。")])
                return

            if image_url:
                images_dir = Path(__file__).parent / "images"
                image_path = await download_to_images_dir(image_url, images_dir, prefer_image=True)
                if not image_path:
                    yield event.chain_result([Plain("下载图片失败，请稍后再试。")])
                    return
                if self.nap_server_address and self.nap_server_address != "localhost":
                    sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    image_path = sent or image_path
                comp = await self._image_from_path_with_callback(image_path)
                yield event.chain_result([comp])
                return

            image_path = await self._save_b64_image(b64, fmt="png")
            if self.nap_server_address and self.nap_server_address != "localhost":
                sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                image_path = sent or image_path
            comp = await self._image_from_path_with_callback(image_path)
            yield event.chain_result([comp])
        except Exception as e:
            logger.error(f"LLM工具生成失败: {e}")
            yield event.chain_result([Plain(f"图像生成失败：{e}")])
