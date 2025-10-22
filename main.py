from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply

from .utils.jimeng_client import JimengConfig, generate_image as jm_generate, compose_image as jm_compose
from .utils.jimeng_client import generate_video as jm_video
from .utils.file_send_server import send_file
from .utils.downloader import download_to_images_dir

import aiofiles
import base64
from pathlib import Path


@register("jimengapi-image", "Codex", "对接‘即梦2’API，支持生图与图生图", "0.1.0")
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

        self.callback_api_base = config.get("callback_api_base", "").strip()
        self.nap_server_address = config.get("nap_server_address", "localhost").strip()
        self.nap_server_port = int(config.get("nap_server_port", 3658))

        self._global_loaded = False

    async def _load_global_config(self):
        if self._global_loaded:
            return
        try:
            cfg = await sp.global_get("jimengapi-image", {})
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

    # 命令：/jvideo <prompt> [model=jimeng-video-3.0] [stream=true|false]
    @filter.command("即梦视频")
    async def jvideo(self, event: AstrMessageEvent):
        """生成视频"""
        await self._load_global_config()
        text = event.message_str or ""
        if not text:
            yield event.plain_result("请提供视频描述，如 /jvideo 海浪拍打海岸")
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
            vurl, raw = await jm_video(cfg, prompt=prompt, model=model, stream=stream_opt)
            if not vurl and not raw:
                yield event.plain_result("视频生成失败，请稍后再试。")
                return

            if vurl:
                # 统一策略：下载到本地再发送，避免外链被 QQ 拒绝
                videos_dir = Path(__file__).parent / "videos"
                video_path = await download_to_images_dir(vurl, videos_dir)
                if not video_path:
                    yield event.chain_result([Plain(f"视频生成成功，但下载失败：{vurl}")])
                    return
                if self.nap_server_address and self.nap_server_address != "localhost":
                    sent = await send_file(video_path, self.nap_server_address, self.nap_server_port)
                    video_path = sent or video_path
                comp = await self._video_from_path(video_path)
                yield event.chain_result([comp])
                return

            # 无直链，回退为展示文本（SSE 合成文本中可能含说明或链接）
            yield event.chain_result([Plain(raw or "生成完成但未获取链接")])
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            yield event.plain_result(f"视频生成失败：{e}")

    # removed llm_tool: jimeng-video-gen
    async def llm_video_gen(self, event: AstrMessageEvent, video_description: str, prefer_stream: bool = True):
        await self._load_global_config()
        cfg = self._cfg()
        try:
            vurl, raw = await jm_video(cfg, prompt=video_description, stream=prefer_stream)
            if vurl:
                videos_dir = Path(__file__).parent / "videos"
                video_path = await download_to_images_dir(vurl, videos_dir)
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
        text = event.message_str or ""
        if not text:
            yield event.plain_result("请提供生成描述，如 /jgen 一只可爱的小猫咪")
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
            image_url, b64 = await jm_generate(
                cfg,
                prompt=prompt,
                ratio=ratio,
                resolution=res,
                negative_prompt=None,
                response_format=fmt,
            )
            if not image_url and not b64:
                yield event.plain_result("生成失败，请检查 API 配置或稍后再试。")
                return

            if image_url:
                # 改为：下载到本地 → 统一发送，避免外链被 QQ 拒绝
                images_dir = Path(__file__).parent / "images"
                image_path = await download_to_images_dir(image_url, images_dir)
                if not image_path:
                    yield event.plain_result("下载图片失败，请稍后再试。")
                    return
                if self.nap_server_address and self.nap_server_address != "localhost":
                    sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    image_path = sent or image_path
                comp = await self._image_from_path_with_callback(image_path)
                yield event.chain_result([comp])
                return

            # b64 路径
            image_path = await self._save_b64_image(b64, fmt="png")
            # 跨机文件（选用NAP）
            if self.nap_server_address and self.nap_server_address != "localhost":
                sent = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                image_path = sent or image_path
            comp = await self._image_from_path_with_callback(image_path)
            yield event.chain_result([comp])
        except Exception as e:
            logger.error(f"生图失败: {e}")
            yield event.plain_result(f"生图失败：{e}")

    # 命令：/jedit <prompt> [ratio=1:1] [res=2k] [fmt=url|b64] [strength=0.7]
    @filter.command("即梦改图")
    async def jedit(self, event: AstrMessageEvent):
        """修改图片"""
        await self._load_global_config()
        text = event.message_str or ""
        if not text:
            yield event.plain_result("请提供修改描述，并附带或引用图片，如 /jedit 转油画风格")
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
            image_url, b64 = await jm_compose(
                cfg,
                prompt=prompt,
                image_urls=image_urls,
                ratio=ratio,
                resolution=res,
                sample_strength=strength,
                negative_prompt=None,
                response_format=fmt,
            )
            if not image_url and not b64:
                yield event.plain_result("改图失败，请检查 API 配置或稍后再试。")
                return

            if image_url:
                images_dir = Path(__file__).parent / "images"
                image_path = await download_to_images_dir(image_url, images_dir)
                if not image_path:
                    yield event.plain_result("下载图片失败，请稍后再试。")
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
                )
            else:
                image_url, b64 = await jm_generate(cfg, prompt=image_description)

            if not image_url and not b64:
                yield event.chain_result([Plain("图像生成失败，请稍后再试。")])
                return

            if image_url:
                images_dir = Path(__file__).parent / "images"
                image_path = await download_to_images_dir(image_url, images_dir)
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
