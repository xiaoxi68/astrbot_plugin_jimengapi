# AstrBot 插件：即梦2 API（生图/改图/视频）

对接 “即梦2” 开放 API，提供文生图（/v1/images/generations）与图生图（/v1/images/compositions）能力。

- 命令：`/即梦生图 <prompt> [ratio=1:1] [res=2k] [fmt=url|b64]`
- 命令：`/即梦改图 <prompt> [ratio=1:1] [res=2k] [fmt=url|b64] [strength=0.7]`（需引用或附带图片）
- 命令：`/即梦视频 <prompt> [model=jimeng-video-3.0] [stream=true|false]`

## 安装与配置

1. 将本插件文件夹放入 AstrBot 插件目录。
2. 在 AstrBot Web 管理界面 → 插件管理 → 本插件 → 配置：
   - `base_url`: 例如 `http://localhost:5100`
   - `session_token`: `Authorization: Bearer <TOKEN>` 所需 Session
   - 其他参数参考 `_conf_schema.json`

## 关键特性

- 兼容 AstrBot 插件规范：`metadata.yaml`、`_conf_schema.json`、`@register`、命令与 LLM 工具
- 与 NapCat/QQ 兼容：支持 `callback_api_base` 与 `NAP 文件转发` 场景
- 高内聚低耦合：HTTP 逻辑集中在 `utils/jimeng_client.py`
- 可扩展：新增参数与模型无需改动调用方（开闭原则）

## 工作流程

- 生图：调用 `/v1/images/generations`，优先返回 URL；必要时支持 `b64_json` → 保存本地 → 转 URL 发送
- 改图：从消息或引用中提取图片 → `convert_to_web_link()` → 作为 `images` 传给 `/compositions`
- 视频：调用 `/v1/chat/completions`（视频模型），默认用 SSE 汇聚文本提取直链 → 统一下载到本地后以视频组件发送

## 注意

- 若启用 `fmt=b64`，将落盘到 `images/` 再发送；如跨机部署可结合 `NAP` 进行转发
- `session_token` 必填；网络异常有重试（`max_retry_attempts` 可配）
