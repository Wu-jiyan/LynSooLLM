"""
server.py
=========
灵枢 LynSooLLM HTTP 服务端：对外暴露统一接口，让用户用任意标准 SDK 接入。

定位：一个"统一入口的智能路由中转站"。用户只需要：
    1. 在 YAML 里配置路由模型（Qwen3.5-0.8B / Gemma-3-270M）+ Multi-LoRA
    2. 在 YAML 里配置上游模型池（多模型/多渠道/多协议/多价格源）
    3. 启动本服务，对外只暴露一个虚拟模型名 "lynsoo-auto"
    4. 用户用任意标准 SDK（OpenAI / Anthropic / Gemini / curl）接入，
       就像调用任何一家云厂商一样，内部按 YAML 自动路由到合适的上游。

对外端点：
    GET  /health                                健康检查
    GET  /v1/models                             OpenAI 风格的列表（只返回 lynsoo-auto）
    GET  /v1beta/models                         Gemini 风格的列表
    POST /v1/chat/completions                   OpenAI Chat Completions（含 SSE 流式）
    POST /v1/responses                          OpenAI Responses API（含 SSE 流式）
    POST /v1/messages                           Anthropic Messages API（含 SSE 流式）
    POST /v1beta/models/{m}:generateContent     Gemini generateContent
    POST /v1beta/models/{m}:streamGenerateContent  Gemini 流式

用户侧使用示例：
    # OpenAI 客户端
    from openai import OpenAI
    c = OpenAI(base_url="http://localhost:8000/v1", api_key="any")
    c.chat.completions.create(model="lynsoo-auto",
                              messages=[{"role":"user","content":"讲个笑话"}])

    # Anthropic 客户端
    from anthropic import Anthropic
    c = Anthropic(base_url="http://localhost:8000")
    c.messages.create(model="lynsoo-auto", max_tokens=1024,
                      messages=[{"role":"user","content":"讲个笑话"}])

    # Gemini 客户端
    import google.generativeai as genai
    genai.configure(api_key="any",
                    transport="rest",
                    client_options={"api_endpoint":"http://localhost:8000"})
    m = genai.GenerativeModel("lynsoo-auto")
    m.generate_content("讲个笑话")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Tuple

# 离线模式（已下载本地权重）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .app import LynSooApp
from .executors import AutoExecutor
from .mock_local_model import MockLocalModel
from .real_router_engine import RealRouterEvent


DEFAULT_MODEL_NAME = "lynsoo-auto"
DEFAULT_PORT = 8000


# ===================================================================== #
#  请求解析：从 4 种协议请求体里提取 prompt + stream flag
# ===================================================================== #
def _flatten_content(content: Any) -> str:
    """把 OpenAI/Anthropic 的 content（可能是 str 或 list）拍平为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if "text" in c:
                    parts.append(str(c["text"]))
                elif c.get("type") == "text":
                    parts.append(str(c.get("text", "")))
            elif isinstance(c, str):
                parts.append(c)
        return "".join(parts)
    return str(content)


def extract_openai_chat(req: Dict) -> Tuple[str, bool]:
    """OpenAI Chat Completions: messages: [{role, content}]"""
    msgs = req.get("messages", [])
    parts: List[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = _flatten_content(m.get("content", ""))
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(content)
    return "\n".join(parts), bool(req.get("stream", False))


def extract_openai_responses(req: Dict) -> Tuple[str, bool]:
    """OpenAI Responses API: input + instructions"""
    inp = req.get("input", "")
    instructions = req.get("instructions", "")

    if isinstance(inp, list):
        parts = []
        for item in inp:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = _flatten_content(item.get("content", ""))
                role = item.get("role", "user")
                if role == "system":
                    parts.append(f"[System]\n{content}")
                elif role == "assistant":
                    parts.append(f"[Assistant]\n{content}")
                else:
                    parts.append(content)
        inp = "\n".join(parts)
    else:
        inp = str(inp)

    if instructions:
        return f"[Instructions]\n{instructions}\n\n{inp}", bool(req.get("stream", False))
    return inp, bool(req.get("stream", False))


def extract_anthropic(req: Dict) -> Tuple[str, bool]:
    """Anthropic Messages: system + messages"""
    system = req.get("system", "")
    if isinstance(system, list):
        system = " ".join(
            s.get("text", "") for s in system if isinstance(s, dict) and "text" in s
        )
    elif not isinstance(system, str):
        system = str(system)

    msgs = req.get("messages", [])
    parts: List[str] = []
    if system:
        parts.append(f"[System]\n{system}")
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = _flatten_content(m.get("content", ""))
        if role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
    return "\n".join(parts), bool(req.get("stream", False))


def extract_gemini(req: Dict) -> Tuple[str, bool]:
    """Gemini: contents + systemInstruction"""
    sys_inst = req.get("systemInstruction") or {}
    if isinstance(sys_inst, dict):
        sys_parts = sys_inst.get("parts", []) or []
        system = " ".join(
            p.get("text", "") for p in sys_parts if isinstance(p, dict) and "text" in p
        )
    else:
        system = ""

    contents = req.get("contents", [])
    parts: List[str] = []
    if system:
        parts.append(f"[System]\n{system}")
    for c in contents:
        if not isinstance(c, dict):
            continue
        role = c.get("role", "user")
        text_parts = []
        for p in c.get("parts", []) or []:
            if isinstance(p, dict) and "text" in p:
                text_parts.append(str(p["text"]))
        text = " ".join(text_parts)
        if role == "user":
            parts.append(text)
        elif role == "model":
            parts.append(f"[Assistant]\n{text}")
    return "\n".join(parts), False  # Gemini 流式靠 URL 后缀判断


# ===================================================================== #
#  内部输出聚合：把 app.stream() 的事件流聚合成 (text_chunks, meta)
# ===================================================================== #
def iter_app_events(app: LynSooApp, prompt: str,
                    prefer_model: Optional[str] = None
                    ) -> Iterator[Tuple[str, Dict]]:
    """
    迭代 LynSooApp.stream()，yield (text_chunk, meta_dict)。

    - token 事件：吐出 token 字符串
    - exit 事件：吐出 cloud_text（整段，作为单个 chunk）
    - decide / done：不吐出文本，meta 累积
    """
    meta: Dict[str, Any] = {}
    for ev in app.stream(prompt, prefer_model=prefer_model):
        if ev.kind == "decide":
            meta["adapter"] = ev.adapter
            meta["route_label"] = ev.route_label
            meta["route_ms"] = ev.payload.get("route_ms") if ev.payload else None
            meta["strategy"] = ev.payload.get("strategy") if ev.payload else None
            continue
        if ev.kind == "token":
            yield ev.token, meta
            continue
        if ev.kind == "exit":
            p = ev.payload or {}
            cloud_text = p.get("cloud_text", "") or ""
            meta["cloud_model"] = p.get("model")
            meta["cloud_latency_ms"] = p.get("latency_ms")
            meta["fallback_used"] = p.get("fallback_used", False)
            meta["hard_route"] = p.get("hard_route", False)
            if cloud_text:
                yield cloud_text, meta
            continue
        if ev.kind == "done":
            p = ev.payload or {}
            meta["ttft_ms"] = p.get("ttft_ms")
            meta["total_ms"] = p.get("total_ms")
            meta["exited"] = p.get("exited", False)
            # 最后再 yield 一次空文本，让上层有机会用最终 meta 收尾
            yield "", meta
            return


def collect_full_text(app: LynSooApp, prompt: str,
                      prefer_model: Optional[str] = None
                      ) -> Tuple[str, Dict]:
    """非流式：把所有 chunk 拼起来。"""
    chunks: List[str] = []
    final_meta: Dict = {}
    for chunk, meta in iter_app_events(app, prompt, prefer_model=prefer_model):
        if chunk:
            chunks.append(chunk)
        final_meta = meta or final_meta
    return "".join(chunks), final_meta


# ===================================================================== #
#  响应格式化器（非流式 + SSE 流式）：4 种协议
# ===================================================================== #
def _ts() -> int:
    return int(time.time())


def _uuid() -> str:
    return uuid.uuid4().hex


# ---------- OpenAI Chat Completions ----------
def fmt_openai_chat_response(text: str, model: str, meta: Dict) -> Dict:
    return {
        "id": f"chatcmpl-{_uuid()}",
        "object": "chat.completion",
        "created": _ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": max(1, len(text) // 2),
            "total_tokens": max(1, len(text) // 2),
        },
        "x_lynsoo_meta": meta,
    }


def sse_openai_chat_chunks(app: LynSooApp, prompt: str, model: str,
                           prefer_model: Optional[str] = None
                           ) -> Iterator[str]:
    """OpenAI Chat SSE 格式：每个 chunk 是 data: {choices:[{delta:{content}}]}"""
    chat_id = f"chatcmpl-{_uuid()}"
    created = _ts()

    def _delta(content: str = "") -> str:
        return json.dumps({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }, ensure_ascii=False)

    # 首个 chunk：role
    yield f"data: {_delta()}\n\n"  # 空 delta
    final_meta: Dict = {}
    for chunk, meta in iter_app_events(app, prompt, prefer_model=prefer_model):
        final_meta = meta or final_meta
        if not chunk:
            continue
        yield f"data: {_delta(chunk)}\n\n"
    # 结束 chunk
    end = json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "x_lynsoo_meta": final_meta,
    }, ensure_ascii=False)
    yield f"data: {end}\n\n"
    yield "data: [DONE]\n\n"


# ---------- OpenAI Responses API ----------
def fmt_openai_responses_response(text: str, model: str, meta: Dict) -> Dict:
    return {
        "id": f"resp-{_uuid()}",
        "object": "response",
        "created_at": _ts(),
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg-{_uuid()}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": text, "annotations": []}
                ],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": 0,
            "output_tokens": max(1, len(text) // 2),
            "total_tokens": max(1, len(text) // 2),
        },
        "x_lynsoo_meta": meta,
    }


def sse_openai_responses_chunks(app: LynSooApp, prompt: str, model: str,
                                prefer_model: Optional[str] = None
                                ) -> Iterator[str]:
    """OpenAI Responses SSE 格式：
        event: response.created
        event: response.output_text.delta  (含 delta 字段)
        event: response.completed
    """
    resp_id = f"resp-{_uuid()}"

    def _evt(name: str, data: Dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    # created
    yield _evt("response.created", {
        "type": "response.created",
        "response": {
            "id": resp_id, "object": "response", "created_at": _ts(),
            "model": model, "status": "in_progress",
            "output": [],
        },
    })
    # output_item.added
    msg_id = f"msg-{_uuid()}"
    yield _evt("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message", "id": msg_id, "status": "in_progress",
            "role": "assistant", "content": [],
        },
    })
    # content_part.added
    yield _evt("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": msg_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })

    final_meta: Dict = {}
    for chunk, meta in iter_app_events(app, prompt, prefer_model=prefer_model):
        final_meta = meta or final_meta
        if not chunk:
            continue
        yield _evt("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": msg_id, "output_index": 0, "content_index": 0,
            "delta": chunk,
        })

    # 收尾
    yield _evt("response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": msg_id, "output_index": 0, "content_index": 0,
        "text": "",
    })
    yield _evt("response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": msg_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })
    yield _evt("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message", "id": msg_id, "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        },
    })
    yield _evt("response.completed", {
        "type": "response.completed",
        "response": {
            "id": resp_id, "object": "response", "created_at": _ts(),
            "model": model, "status": "completed", "output": [],
            "x_lynsoo_meta": final_meta,
        },
    })


# ---------- Anthropic Messages ----------
def fmt_anthropic_response(text: str, model: str, meta: Dict) -> Dict:
    return {
        "id": f"msg_{_uuid()}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": max(1, len(text) // 2),
        },
        "x_lynsoo_meta": meta,
    }


def sse_anthropic_chunks(app: LynSooApp, prompt: str, model: str,
                         prefer_model: Optional[str] = None
                         ) -> Iterator[str]:
    """Anthropic SSE 格式：
        event: message_start
        event: content_block_start
        event: content_block_delta  (含 delta: {type:text, text})
        event: content_block_stop
        event: message_delta
        event: message_stop
    """
    msg_id = f"msg_{_uuid()}"

    def _evt(name: str, data: Dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    # message_start
    yield _evt("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model, "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    # content_block_start
    yield _evt("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    # ping（部分 SDK 期待）
    yield _evt("ping", {"type": "ping"})

    final_meta: Dict = {}
    for chunk, meta in iter_app_events(app, prompt, prefer_model=prefer_model):
        final_meta = meta or final_meta
        if not chunk:
            continue
        yield _evt("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        })

    # content_block_stop
    yield _evt("content_block_stop", {"type": "content_block_stop", "index": 0})
    # message_delta
    yield _evt("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    })
    # message_stop
    yield _evt("message_stop", {
        "type": "message_stop",
        "x_lynsoo_meta": final_meta,
    })


# ---------- Gemini generateContent ----------
def fmt_gemini_response(text: str, model: str, meta: Dict) -> Dict:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": text}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": max(1, len(text) // 2),
            "totalTokenCount": max(1, len(text) // 2),
        },
        "x_lynsoo_meta": meta,
    }


def sse_gemini_chunks(app: LynSooApp, prompt: str, model: str,
                      prefer_model: Optional[str] = None
                      ) -> Iterator[str]:
    """Gemini SSE 格式（streamGenerateContent）：每个 chunk data: {candidates:...}"""
    for chunk, meta in iter_app_events(app, prompt, prefer_model=prefer_model):
        if not chunk:
            continue
        data = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": chunk}]},
                    "finishReason": None,
                    "index": 0,
                }
            ],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    # 末尾 chunk 带 finishReason
    end = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": ""}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"totalTokenCount": 0},
    }
    yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"


# ===================================================================== #
#  HTTP Handler
# ===================================================================== #
class LynSooHandler(BaseHTTPRequestHandler):
    # LynSooApp 实例由 server 注入
    app: LynSooApp = None  # type: ignore[assignment]
    server_model_name: str = DEFAULT_MODEL_NAME
    # 全局共享（ThreadingHTTPServer 多线程访问）
    _lock = threading.Lock()

    # 通用工具 -------------------------------------------------------- #
    def _read_body(self) -> Dict:
        ln = int(self.headers.get("Content-Length", 0))
        if ln == 0:
            return {}
        try:
            return json.loads(self.rfile.read(ln).decode("utf-8"))
        except Exception:
            return {}

    def _send_json(self, obj: Dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str, err_type: str = "invalid_request") -> None:
        self._send_json({"error": {"message": message, "type": err_type, "code": status}}, status)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _sse_write(self, chunk: str) -> None:
        # chunk 末尾已经包含 \n\n
        self.wfile.write(chunk.encode("utf-8"))
        self.wfile.flush()

    def _extract_prefer_model(self, req: Dict) -> Optional[str]:
        """从请求里找用户指定的 prefer_model（自定义字段，可选）。"""
        # 兼容多种写法
        prefer = req.get("x_lynsoo_prefer_model") or req.get("prefer_model")
        if prefer:
            return prefer
        metadata = req.get("metadata")
        if isinstance(metadata, dict) and metadata.get("prefer_model"):
            return metadata["prefer_model"]
        return None

    def log_message(self, fmt: str, *args) -> None:
        # 简化日志：方法 + 路径 + 状态
        sys.stderr.write(f"[lynsoo] {self.command} {self.path}\n")

    # CORS 预检 ------------------------------------------------------- #
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # GET ------------------------------------------------------------- #
    def do_GET(self) -> None:
        path = self.path
        if path == "/health" or path == "/":
            self._send_json({
                "status": "ok",
                "service": "lynsoollm",
                "version": "0.4.0",
                "model": self.server_model_name,
                "models_configured": len(self.app.registry.list_enabled()),
                "strategy": self.app.calculator.strategy,
            })
            return

        if path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [
                    {
                        "id": self.server_model_name,
                        "object": "model",
                        "created": _ts(),
                        "owned_by": "lynsoollm",
                    }
                ],
            })
            return

        if path == "/v1beta/models":
            self._send_json({
                "models": [
                    {
                        "name": f"models/{self.server_model_name}",
                        "version": "001",
                        "displayName": self.server_model_name,
                        "description": "LynSooLLM auto-routing virtual model",
                        "supportedGenerationMethods": [
                            "generateContent", "streamGenerateContent",
                        ],
                    }
                ],
            })
            return

        self._send_error(404, f"unknown path: {path}", "not_found")

    # POST ------------------------------------------------------------ #
    def do_POST(self) -> None:
        path = self.path
        try:
            # OpenAI Chat Completions
            if path.endswith("/v1/chat/completions"):
                self._handle_openai_chat()
                return
            # OpenAI Responses
            if path.endswith("/v1/responses"):
                self._handle_openai_responses()
                return
            # Anthropic Messages
            if path.endswith("/v1/messages"):
                self._handle_anthropic_messages()
                return
            # Gemini generateContent / streamGenerateContent
            m = re.match(
                r"/v1beta/models/([^:/?]+)(?::(streamGenerateContent|generateContent))",
                path,
            )
            if m:
                model_in_path = m.group(1)
                is_stream = m.group(2) == "streamGenerateContent"
                self._handle_gemini(is_stream=is_stream)
                return
            # 其他兼容：根路径的 /v1beta/{model}:generateContent
            if ":generateContent" in path or ":streamGenerateContent" in path:
                is_stream = ":streamGenerateContent" in path
                self._handle_gemini(is_stream=is_stream)
                return

            self._send_error(404, f"unknown path: {path}", "not_found")
        except Exception as ex:
            import traceback
            sys.stderr.write(traceback.format_exc())
            self._send_error(500, f"internal error: {ex}", "internal_error")

    # ---- 4 个协议各自的 handler ---- #
    def _handle_openai_chat(self) -> None:
        req = self._read_body()
        prompt, stream = extract_openai_chat(req)
        prefer = self._extract_prefer_model(req)
        model = self.server_model_name

        if stream:
            self._start_sse()
            for chunk in sse_openai_chat_chunks(self.app, prompt, model, prefer):
                self._sse_write(chunk)
        else:
            text, meta = collect_full_text(self.app, prompt, prefer)
            self._send_json(fmt_openai_chat_response(text, model, meta))

    def _handle_openai_responses(self) -> None:
        req = self._read_body()
        prompt, stream = extract_openai_responses(req)
        prefer = self._extract_prefer_model(req)
        model = self.server_model_name

        if stream:
            self._start_sse()
            for chunk in sse_openai_responses_chunks(self.app, prompt, model, prefer):
                self._sse_write(chunk)
        else:
            text, meta = collect_full_text(self.app, prompt, prefer)
            self._send_json(fmt_openai_responses_response(text, model, meta))

    def _handle_anthropic_messages(self) -> None:
        req = self._read_body()
        prompt, stream = extract_anthropic(req)
        prefer = self._extract_prefer_model(req)
        model = self.server_model_name

        if stream:
            self._start_sse()
            for chunk in sse_anthropic_chunks(self.app, prompt, model, prefer):
                self._sse_write(chunk)
        else:
            text, meta = collect_full_text(self.app, prompt, prefer)
            self._send_json(fmt_anthropic_response(text, model, meta))

    def _handle_gemini(self, is_stream: bool) -> None:
        req = self._read_body()
        prompt, _ = extract_gemini(req)
        prefer = self._extract_prefer_model(req)
        model = self.server_model_name

        if is_stream:
            self._start_sse()
            for chunk in sse_gemini_chunks(self.app, prompt, model, prefer):
                self._sse_write(chunk)
        else:
            text, meta = collect_full_text(self.app, prompt, prefer)
            self._send_json(fmt_gemini_response(text, model, meta))


# ===================================================================== #
#  Server 启动器
# ===================================================================== #
def make_app(config_path: str, device: str = "cpu") -> LynSooApp:
    """从 YAML 配置构造 LynSooApp。"""
    # 一个轻量 mock 本地生成器（用户后续可注入真实 vLLM/llama.cpp）
    local_gen = MockLocalModel(
        default_tokens=["你好", "，", "我是", "灵枢", "路由", "。"],
        entropy_schedule=[0.2, 0.3, 0.4, 2.5, 3.0, 2.8],
    )
    return LynSooApp.from_config(
        config_path,
        local_generator=local_gen,
        executor=AutoExecutor(timeout=60.0),
        device=device,
        fetch_pricing=True,
        verbose=True,
    )


def run_server(config_path: str, port: int = DEFAULT_PORT,
               model_name: str = DEFAULT_MODEL_NAME,
               device: str = "cpu",
               host: str = "0.0.0.0") -> None:
    """启动 HTTP 服务。"""
    print("=" * 72)
    print(f"  灵枢 LynSooLLM 路由中转站")
    print(f"  配置: {config_path}")
    print(f"  监听: http://{host}:{port}")
    print(f"  对外虚拟模型名: {model_name}（用户用任意 SDK 接入即可）")
    print("=" * 72)

    app = make_app(config_path, device=device)
    LynSooHandler.app = app
    LynSooHandler.server_model_name = model_name

    server = ThreadingHTTPServer((host, port), LynSooHandler)
    print(f"\n  已配置上游模型 ({len(app.registry.list_enabled())} 个):")
    for e in sorted(app.registry.list_enabled(), key=lambda x: x.priority):
        p = e.pricing
        cost = f"${p.input_per_1k + p.output_per_1k:.5f}/1k" if p else "n/a"
        proto = e.protocol
        if proto == "auto":
            from .executors import AutoExecutor
            proto = AutoExecutor._infer(e)
        print(f"    P{e.priority:<2} {e.name:<22} {e.model_id:<22} "
              f"{proto:<22} {cost}")

    print(f"\n  路由模型: {app.engine.router_model.backbone_name}")
    print(f"  路由策略: {app.calculator.strategy}")
    print(f"\n  端点:")
    print(f"    GET  /health")
    print(f"    GET  /v1/models                              (OpenAI 风格)")
    print(f"    GET  /v1beta/models                          (Gemini 风格)")
    print(f"    POST /v1/chat/completions                    (OpenAI Chat)")
    print(f"    POST /v1/responses                           (OpenAI Responses)")
    print(f"    POST /v1/messages                            (Anthropic Messages)")
    print(f"    POST /v1beta/models/{{m}}:generateContent        (Gemini)")
    print(f"    POST /v1beta/models/{{m}}:streamGenerateContent  (Gemini SSE)")
    print(f"\n  试用: curl http://localhost:{port}/v1/models\n")
    print("  按 Ctrl+C 停止\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] 收到中断信号，正在停止...")
        server.shutdown()


# ===================================================================== #
#  CLI
# ===================================================================== #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="灵枢 LynSooLLM 路由中转站 HTTP 服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m lynsoollm.server --config config.yaml --port 8000
    python -m lynsoollm.server -c config.yaml -p 8000 --model lynsoo-auto

接入示例（用户侧）:
    # OpenAI
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"lynsoo-auto","messages":[{"role":"user","content":"你好"}]}'

    # Anthropic
    curl http://localhost:8000/v1/messages \\
      -H "Content-Type: application/json" \\
      -H "x-api-key: any" -H "anthropic-version: 2023-06-01" \\
      -d '{"model":"lynsoo-auto","max_tokens":1024,"messages":[{"role":"user","content":"你好"}]}'

    # Gemini
    curl "http://localhost:8000/v1beta/models/lynsoo-auto:generateContent" \\
      -H "Content-Type: application/json" \\
      -d '{"contents":[{"parts":[{"text":"你好"}]}]}'
        """,
    )
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="YAML 配置文件路径（默认: config.yaml）")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT,
                        help=f"监听端口（默认: {DEFAULT_PORT}）")
    parser.add_argument("--host", default="0.0.0.0",
                        help="监听地址（默认: 0.0.0.0）")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help=f"对外虚拟模型名（默认: {DEFAULT_MODEL_NAME}）")
    parser.add_argument("--device", default="cpu",
                        help="路由模型运行设备（cpu / cuda，默认: cpu）")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.stderr.write(f"[error] 配置文件不存在: {args.config}\n")
        sys.stderr.write("  可以先执行: python -c \"from lynsoollm import write_example_config; "
                          "write_example_config('config.yaml')\"\n")
        return 2

    run_server(
        config_path=args.config,
        port=args.port,
        model_name=args.model,
        device=args.device,
        host=args.host,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
