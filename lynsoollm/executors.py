"""
executors.py
============
云端模型执行器集合：覆盖主流厂商的"原生 API"协议。

支持的协议：
    1. OpenAIChatExecutor        : POST /v1/chat/completions         (传统 Chat Completions)
    2. OpenAIResponsesExecutor   : POST /v1/responses                (OpenAI 新一代 Responses API)
    3. AnthropicMessagesExecutor : POST /v1/messages                 (Claude 原生 Messages API)
    4. GeminiGenerateContentExecutor : POST /v1beta/models/{m}:generateContent (Gemini 原生)
    5. AutoExecutor              : 根据 provider / endpoint 自动选上面 4 种之一

设计要点：
    - 所有 executor 都是 callable：(entry, ctx) -> str
    - 用标准库 urllib，无外部依赖
    - 用户可在 YAML 里写 `protocol: anthropic_messages` 强制指定，
      或写 `protocol: auto` 让 AutoExecutor 自动选
    - 已生成的本地上下文（ctx.generated_text）作为 assistant 预填
      （通过 RelayContext.chat_messages()），让云端模型无缝接续；
      不会拼到 user prompt 里，避免污染用户输入

协议字段对照（截至 2026-07）：

| 协议                  | 端点                                  | 鉴权 header            | 请求体关键字段              | 响应文本路径                                  |
|-----------------------|---------------------------------------|------------------------|---------------------------|-----------------------------------------------|
| openai_chat           | /v1/chat/completions                  | Authorization: Bearer  | messages: [{role,content}]| choices[0].message.content                    |
| openai_responses      | /v1/responses                         | Authorization: Bearer  | input, instructions       | output[].content[].text (type=output_text)    |
| anthropic_messages    | /v1/messages                          | x-api-key + version    | messages, max_tokens      | content[] (type=text).text                    |
| gemini_generate       | /v1beta/models/{model}:generateContent| x-goog-api-key         | contents[].parts[].text   | candidates[0].content.parts[].text            |
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

from .exit_signal import RelayContext
from .model_registry import ModelEntry


# --------------------------------------------------------------------- #
#  HTTP 工具
# --------------------------------------------------------------------- #
def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str],
               timeout: float = 30.0) -> Dict[str, Any]:
    """POST JSON 并返回 JSON 响应。"""
    data = json.dumps(payload).encode("utf-8")
    final_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "LynSooLLM/0.4",
    }
    final_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=final_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body)


def _require(entry: ModelEntry, *fields: str) -> None:
    """校验 entry 必填字段。"""
    for f in fields:
        v = getattr(entry, f, "")
        if not v:
            raise RuntimeError(f"模型 {entry.name} 缺少必填字段: {f}")


# --------------------------------------------------------------------- #
#  1) OpenAI Chat Completions（兼容协议，最广）
# --------------------------------------------------------------------- #
class OpenAIChatExecutor:
    """
    OpenAI Chat Completions 兼容执行器：POST /v1/chat/completions。

    支持 OpenAI / DeepSeek / OpenRouter / vLLM / 任何 OpenAI 兼容接口。
    """

    def __init__(self, timeout: float = 30.0,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    def __call__(self, entry: ModelEntry, ctx: RelayContext) -> str:
        _require(entry, "endpoint", "api_key")
        url = entry.endpoint.rstrip("/") + "/chat/completions"
        # 用 chat_messages 把本地已生成文本作为 assistant 预填
        msgs = ctx.chat_messages()
        # 有 prefill 时加 system 提示，强化接续意图（兼容不完全支持
        # prefill 的 OpenAI 兼容代理）
        sys_hint = ctx.system_hint_for_continuation()
        if sys_hint:
            msgs = [{"role": "system", "content": sys_hint}] + msgs
        payload = {
            "model": entry.model_id or entry.name,
            "messages": msgs,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {entry.api_key}",
            **self.extra_headers,
        }
        out = _post_json(url, payload, headers, self.timeout)
        return out["choices"][0]["message"]["content"]


# 向后兼容别名
HTTPCloudExecutor = OpenAIChatExecutor


# --------------------------------------------------------------------- #
#  2) OpenAI Responses API（新一代）
# --------------------------------------------------------------------- #
class OpenAIResponsesExecutor:
    """
    OpenAI Responses API 执行器：POST /v1/responses。

    Responses API 是 OpenAI 推荐的新一代接口，与 Chat Completions 的区别：
        - 请求体用 `input` + `instructions`（而非 `messages`）
        - 响应是 `output` 数组，包含 message / function_call / reasoning 等
        - 支持 `previous_response_id` 状态管理
        - 默认有 `store: true`，这里强制设为 false 以保持无状态

    响应解析：遍历 output 数组，提取 type=="message" 项里的
    content 数组中 type=="output_text" 的 text 字段。
    """

    def __init__(self, timeout: float = 60.0,
                 extra_headers: Optional[Dict[str, str]] = None,
                 max_output_tokens: Optional[int] = None) -> None:
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.max_output_tokens = max_output_tokens

    def __call__(self, entry: ModelEntry, ctx: RelayContext) -> str:
        _require(entry, "endpoint", "api_key")
        url = entry.endpoint.rstrip("/") + "/responses"
        # Responses API 的 input 支持字符串或消息数组；用消息数组
        # 把本地已生成文本作为 assistant 预填
        msgs = ctx.chat_messages()
        sys_hint = ctx.system_hint_for_continuation()
        if sys_hint:
            msgs = [{"role": "system", "content": sys_hint}] + msgs
        payload: Dict[str, Any] = {
            "model": entry.model_id or entry.name,
            "input": msgs,
            "store": False,                  # 无状态，不存服务端
            "stream": False,
        }
        if self.max_output_tokens:
            payload["max_output_tokens"] = self.max_output_tokens
        headers = {
            "Authorization": f"Bearer {entry.api_key}",
            **self.extra_headers,
        }
        out = _post_json(url, payload, headers, self.timeout)
        return self._extract_text(out)

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        """
        从 Responses API 响应里提取文本。

        响应结构：
            {
              "id": "...",
              "output": [
                {
                  "type": "message",
                  "content": [
                    {"type": "output_text", "text": "...", "annotations": []}
                  ]
                },
                ...
              ]
            }
        """
        output = response.get("output") or []
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            # 直接是 output_text 项（部分版本）
            if item.get("type") == "output_text" and "text" in item:
                parts.append(item["text"])
                continue
            # message 项里嵌套 content
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
                    elif isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
        # 兜底：output_text 顶层字段
        if not parts and response.get("output_text"):
            parts.append(response["output_text"])
        return "".join(parts)


# --------------------------------------------------------------------- #
#  3) Anthropic Messages API（Claude 原生）
# --------------------------------------------------------------------- #
class AnthropicMessagesExecutor:
    """
    Anthropic Messages API 执行器：POST /v1/messages。

    鉴权头：x-api-key + anthropic-version
    请求体：{model, max_tokens, messages: [{role, content}]}
    响应体：{content: [{type:"text", text:"..."}]}
    """

    DEFAULT_VERSION = "2023-06-01"

    def __init__(self, timeout: float = 60.0,
                 anthropic_version: str = DEFAULT_VERSION,
                 max_tokens: int = 1024,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.timeout = timeout
        self.anthropic_version = anthropic_version
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}

    def __call__(self, entry: ModelEntry, ctx: RelayContext) -> str:
        _require(entry, "endpoint", "api_key")
        url = entry.endpoint.rstrip("/") + "/messages"
        payload: Dict[str, Any] = {
            "model": entry.model_id or entry.name,
            "max_tokens": self.max_tokens,
            # Claude 原生支持 assistant 预填（prefill），
            # 把本地已生成文本作为 assistant 消息接续
            "messages": ctx.chat_messages(),
        }
        # Claude 用顶级 system 字段（不是 messages 里的 system role）
        sys_hint = ctx.system_hint_for_continuation()
        if sys_hint:
            payload["system"] = sys_hint
        headers = {
            "x-api-key": entry.api_key,
            "anthropic-version": self.anthropic_version,
            **self.extra_headers,
        }
        out = _post_json(url, payload, headers, self.timeout)
        return self._extract_text(out)

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        """content 数组里 type=="text" 的项拼接。"""
        parts = []
        for block in response.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)


# --------------------------------------------------------------------- #
#  4) Gemini generateContent API（原生）
# --------------------------------------------------------------------- #
class GeminiGenerateContentExecutor:
    """
    Gemini 原生 generateContent 执行器：
        POST /v1beta/models/{model}:generateContent

    鉴权头：x-goog-api-key 或 URL 查询参数 ?key=
    请求体：{contents: [{role:"user", parts:[{text:"..."}]}]}
    响应体：{candidates: [{content: {parts: [{text:"..."}]}}]}
    """

    def __init__(self, timeout: float = 60.0,
                 use_query_param: bool = False,
                 extra_headers: Optional[Dict[str, str]] = None,
                 generation_config: Optional[Dict[str, Any]] = None) -> None:
        self.timeout = timeout
        self.use_query_param = use_query_param   # 某些代理只支持 ?key=
        self.extra_headers = extra_headers or {}
        self.generation_config = generation_config or {}

    def __call__(self, entry: ModelEntry, ctx: RelayContext) -> str:
        _require(entry, "endpoint", "api_key")
        model = entry.model_id or entry.name
        base = entry.endpoint.rstrip("/")
        # endpoint 可能是 https://generativelanguage.googleapis.com/v1beta
        # 也可能是 https://.../v1beta （去掉 :generateContent 后缀）
        # 我们自动补全路径
        if base.endswith(model):
            url = f"{base}:generateContent"
        elif "/models/" in base and base.endswith(model.split("/")[-1]):
            url = f"{base}:generateContent"
        else:
            url = f"{base}/models/{model}:generateContent"

        if self.use_query_param:
            url = f"{url}?key={entry.api_key}"
            headers = {"Content-Type": "application/json"}
        else:
            headers = {"x-goog-api-key": entry.api_key}
        headers.update(self.extra_headers)

        payload: Dict[str, Any] = {
            # Gemini 用 "model" 而非 "assistant"；本地已生成文本作为
            # model 角色预填，让云端接续
            "contents": [
                {
                    "role": "model" if m["role"] != "user" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in ctx.chat_messages(assistant_role="model")
            ],
        }
        # Gemini 用 systemInstruction 字段
        sys_hint = ctx.system_hint_for_continuation()
        if sys_hint:
            payload["systemInstruction"] = {
                "parts": [{"text": sys_hint}]
            }
        if self.generation_config:
            payload["generationConfig"] = self.generation_config
        out = _post_json(url, payload, headers, self.timeout)
        return self._extract_text(out)

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        """candidates[0].content.parts[].text 拼接。"""
        parts = []
        for cand in response.get("candidates") or []:
            content = cand.get("content") or {}
            for p in content.get("parts") or []:
                if isinstance(p, dict) and "text" in p:
                    # 跳过 thought 中间产物
                    if p.get("thought"):
                        continue
                    parts.append(p["text"])
        return "".join(parts)


# --------------------------------------------------------------------- #
#  5) AutoExecutor：根据 provider / endpoint / protocol 字段自动选
# --------------------------------------------------------------------- #
PROTOCOLS = {
    "openai_chat", "openai_responses", "anthropic_messages",
    "gemini_generate", "auto",
}


class AutoExecutor:
    """
    根据 ModelEntry 的 provider / endpoint / 自定义 protocol 字段，
    自动选择最合适的执行器。

    选择优先级：
        1. entry.protocol 显式指定（openai_chat / openai_responses /
           anthropic_messages / gemini_generate）
        2. 按 provider 推断：
              anthropic / claude  -> anthropic_messages
              google / gemini     -> gemini_generate
              openai             -> openai_chat （最稳，向后兼容）
              其他               -> openai_chat
        3. 按 endpoint URL 启发式：
              含 /v1beta/ 且含 :generateContent -> gemini_generate
              含 /v1/messages                  -> anthropic_messages
              含 /v1/responses                 -> openai_responses
              其他                            -> openai_chat

    用法：
        executor = AutoExecutor(timeout=60)
        text = executor(entry, ctx)
    """

    def __init__(self, timeout: float = 60.0,
                 extra_headers: Optional[Dict[str, str]] = None,
                 anthropic_version: str = AnthropicMessagesExecutor.DEFAULT_VERSION,
                 anthropic_max_tokens: int = 1024) -> None:
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.anthropic_version = anthropic_version
        self.anthropic_max_tokens = anthropic_max_tokens
        self._cache: Dict[str, Callable] = {}

    def _select(self, entry: ModelEntry) -> Callable:
        """根据 entry 选 executor（带缓存）。"""
        key = f"{entry.provider}|{entry.endpoint}|{getattr(entry, 'protocol', '')}"
        if key in self._cache:
            return self._cache[key]

        proto = getattr(entry, "protocol", None) or "auto"
        if proto == "auto":
            proto = self._infer(entry)

        if proto == "openai_responses":
            ex = OpenAIResponsesExecutor(timeout=self.timeout,
                                          extra_headers=self.extra_headers)
        elif proto == "anthropic_messages":
            ex = AnthropicMessagesExecutor(
                timeout=self.timeout,
                anthropic_version=self.anthropic_version,
                max_tokens=self.anthropic_max_tokens,
                extra_headers=self.extra_headers,
            )
        elif proto == "gemini_generate":
            ex = GeminiGenerateContentExecutor(
                timeout=self.timeout, extra_headers=self.extra_headers,
            )
        else:  # openai_chat 或未知
            ex = OpenAIChatExecutor(timeout=self.timeout,
                                     extra_headers=self.extra_headers)
        self._cache[key] = ex
        return ex

    @staticmethod
    def _infer(entry: ModelEntry) -> str:
        """根据 provider / endpoint / model_id 推断协议。"""
        # 1) provider 优先
        prov = (entry.provider or "").lower()
        if prov in ("anthropic", "claude"):
            return "anthropic_messages"
        if prov in ("google", "gemini", "google_gemini"):
            return "gemini_generate"

        # 2) endpoint 启发式
        ep = (entry.endpoint or "").lower()
        if ":generatecontent" in ep or "/v1beta" in ep:
            return "gemini_generate"
        if "/v1/messages" in ep:
            return "anthropic_messages"
        if "/v1/responses" in ep:
            return "openai_responses"

        # 3) model_id 启发式（兜底）
        mid = (entry.model_id or "").lower()
        if "gemini" in mid:
            return "gemini_generate"
        if "claude" in mid:
            return "anthropic_messages"

        # 4) 默认 OpenAI Chat（兼容性最广）
        return "openai_chat"

    def __call__(self, entry: ModelEntry, ctx: RelayContext) -> str:
        executor = self._select(entry)
        return executor(entry, ctx)

    # ------------------------------------------------------------------ #
    #  诊断
    # ------------------------------------------------------------------ #
    def explain(self, entry: ModelEntry) -> Dict[str, str]:
        """返回对某个 entry 的协议选择解释。"""
        proto = getattr(entry, "protocol", None) or "auto"
        inferred = None
        if proto == "auto":
            inferred = self._infer(entry)
            proto = inferred
        return {
            "model_name": entry.name,
            "provider": entry.provider,
            "endpoint": entry.endpoint,
            "protocol": proto,
            "inferred_by": "explicit" if inferred is None else "auto_infer",
        }


# --------------------------------------------------------------------- #
#  Mock 执行器（演示与测试用）
# --------------------------------------------------------------------- #
class MockCloudExecutor:
    """默认执行器：返回占位文本。用于演示与测试。"""

    def __init__(self, prefix: str = "cloud") -> None:
        self.prefix = prefix

    def __call__(self, entry: ModelEntry, ctx: RelayContext) -> str:
        return f"{ctx.generated_text}<{self.prefix}:{entry.name}>"
