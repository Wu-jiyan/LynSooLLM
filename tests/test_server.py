"""
test_server.py
==============
端到端验证 lynsoollm HTTP 服务端：

启动 mock 上游服务（同时支持 4 种原生协议），
再启动 lynsoo server，分别用 4 种协议客户端访问虚拟模型 lynsoo-auto，
验证响应格式符合各协议规范。
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict

import pytest

# 复用 product_demo 里的多协议 mock 上游
from lynsoollm.product_demo import start_multi_protocol_mock_server


# --------------------------------------------------------------------- #
#  启动 lynsoo server（在子线程，端口 18770）
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def lynsoo_server():
    """启动一个 mock 上游 + lynsoo server，整个模块共享。"""
    # 1) 启动 mock 上游服务（端口 18769）
    upstream_port = 18769
    upstream_srv = start_multi_protocol_mock_server(upstream_port)
    time.sleep(0.3)

    # 2) 构造一个最小 YAML 配置（指向 mock 上游）
    import tempfile, os
    base = f"http://127.0.0.1:{upstream_port}"
    config_yaml = f"""
models:
  - name: mock-deepseek
    provider: deepseek
    model_id: deepseek-chat
    endpoint: {base}/v1
    api_key: demo-key-1
    protocol: auto
    pricing_source: manual
    manual_pricing: {{input_per_1k: 0.0001, output_per_1k: 0.0003}}
    quality_tier: 3

  - name: mock-claude
    provider: anthropic
    model_id: claude-3.5-sonnet
    endpoint: {base}/v1
    api_key: demo-key-2
    protocol: anthropic_messages
    pricing_source: manual
    manual_pricing: {{input_per_1k: 0.003, output_per_1k: 0.015}}
    quality_tier: 4

  - name: mock-gemini
    provider: gemini
    model_id: gemini-2.5-flash
    endpoint: {base}/v1beta
    api_key: demo-key-3
    protocol: gemini_generate
    pricing_source: manual
    manual_pricing: {{input_per_1k: 0.0005, output_per_1k: 0.0015}}
    quality_tier: 3

  - name: mock-gpt4o-responses
    provider: openai
    model_id: gpt-4o
    endpoint: {base}/v1
    api_key: demo-key-4
    protocol: openai_responses
    pricing_source: manual
    manual_pricing: {{input_per_1k: 0.005, output_per_1k: 0.015}}
    quality_tier: 4

router:
  backbone: gemma
  strategy: cost_first
  entropy_threshold: 1.5
  max_new_tokens: 8

device:
  network_rtt_ms: 30
  battery_pct: 85
  temperature_c: 30

pricing:
  cache_path: /tmp/lynsoollm_test_server.json
  cache_ttl_sec: 86400
  zero_price_fallback: 0.01
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                      delete=False, encoding="utf-8") as f:
        f.write(config_yaml)
        cfg_path = f.name

    # 3) 启动 lynsoo server（线程）
    from lynsoollm.server import make_app, LynSooHandler, DEFAULT_MODEL_NAME
    from http.server import ThreadingHTTPServer
    from lynsoollm.executors import AutoExecutor
    from lynsoollm.mock_local_model import MockLocalModel

    server_port = 18770
    app = make_app(cfg_path, device="cpu")
    LynSooHandler.app = app
    LynSooHandler.server_model_name = DEFAULT_MODEL_NAME

    srv = ThreadingHTTPServer(("127.0.0.1", server_port), LynSooHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(1.5)  # 等模型加载完

    yield {
        "base_url": f"http://127.0.0.1:{server_port}",
        "server": srv,
        "app": app,
        "cfg_path": cfg_path,
    }

    srv.shutdown()
    upstream_srv.shutdown()
    os.unlink(cfg_path)


# --------------------------------------------------------------------- #
#  HTTP 工具
# --------------------------------------------------------------------- #
def _post_json(url: str, payload: Dict, headers: Dict = None) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _post_sse(url: str, payload: Dict, headers: Dict = None,
              timeout: float = 30.0, max_chars: int = 8192) -> str:
    """返回完整 SSE 文本（用 socket 读取，直到收到 [DONE] 或 max_chars）。

    urllib.request.urlopen 不支持流式读取，会卡到连接关闭，
    所以这里直接用 socket 拼一个 HTTP/1.1 请求并逐块读取。
    """
    from urllib.parse import urlparse
    p = urlparse(url)
    host = p.hostname
    port = p.port or 80
    path = p.path + ("?" + p.query if p.query else "")

    body = json.dumps(payload).encode("utf-8")
    h = {
        "Host": f"{host}:{port}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Content-Length": str(len(body)),
        "Connection": "close",
    }
    if headers:
        h.update(headers)
    req_line = f"POST {path} HTTP/1.1\r\n"
    req_line += "".join(f"{k}: {v}\r\n" for k, v in h.items())
    req_line += "\r\n"

    import socket
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.sendall(req_line.encode("utf-8") + body)
    chunks: list[bytes] = []
    total = 0
    sock.settimeout(timeout)
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
            total += len(data)
            if total > max_chars:
                break
            # OpenAI SSE 以 [DONE] 结尾
            joined = b"".join(chunks)
            if b"[DONE]" in joined or b"message_stop" in joined \
               or b"response.completed" in joined \
               or b'"finishReason"' in joined and b'"STOP"' in joined:
                # 多读一会儿确保读完最后一个事件
                sock.settimeout(0.5)
                try:
                    while True:
                        more = sock.recv(4096)
                        if not more:
                            break
                        chunks.append(more)
                except (socket.timeout, OSError):
                    pass
                break
    finally:
        sock.close()
    raw = b"".join(chunks).decode("utf-8", errors="ignore")
    # 去掉 HTTP 响应头
    if "\r\n\r\n" in raw:
        raw = raw.split("\r\n\r\n", 1)[1]
    return raw


def _get(url: str) -> Dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


# --------------------------------------------------------------------- #
#  测试用例
# --------------------------------------------------------------------- #
def test_health(lynsoo_server):
    base = lynsoo_server["base_url"]
    r = _get(f"{base}/health")
    assert r["status"] == "ok"
    assert r["model"] == "lynsoo-auto"
    assert r["models_configured"] == 4


def test_list_models_openai_style(lynsoo_server):
    base = lynsoo_server["base_url"]
    r = _get(f"{base}/v1/models")
    assert r["object"] == "list"
    assert any(m["id"] == "lynsoo-auto" for m in r["data"])


def test_list_models_gemini_style(lynsoo_server):
    base = lynsoo_server["base_url"]
    r = _get(f"{base}/v1beta/models")
    assert "models" in r
    assert "lynsoo-auto" in r["models"][0]["name"]


def test_openai_chat_non_stream(lynsoo_server):
    """OpenAI Chat 非流式：用户传 model=lynsoo-auto，内部路由到上游。"""
    base = lynsoo_server["base_url"]
    r = _post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "lynsoo-auto",
            "messages": [{"role": "user", "content": "讲个笑话"}],
            "stream": False,
        },
    )
    assert r["object"] == "chat.completion"
    assert r["model"] == "lynsoo-auto"
    assert "choices" in r
    text = r["choices"][0]["message"]["content"]
    assert len(text) > 0
    # 内部应该接力到了某个上游（DeepSeek 最便宜，cost_first）
    assert r["x_lynsoo_meta"]["cloud_model"] == "mock-deepseek"


def test_openai_chat_stream(lynsoo_server):
    """OpenAI Chat SSE 流式。"""
    base = lynsoo_server["base_url"]
    sse = _post_sse(
        f"{base}/v1/chat/completions",
        {
            "model": "lynsoo-auto",
            "messages": [{"role": "user", "content": "讲个笑话"}],
            "stream": True,
        },
    )
    # 应该有多个 data: 行
    assert "data: " in sse
    assert "data: [DONE]" in sse
    # 应该有 content delta
    assert '"delta"' in sse or "choices" in sse


def test_openai_responses_non_stream(lynsoo_server):
    """OpenAI Responses API 非流式。"""
    base = lynsoo_server["base_url"]
    r = _post_json(
        f"{base}/v1/responses",
        {
            "model": "lynsoo-auto",
            "input": "讲个笑话",
            "stream": False,
        },
    )
    assert r["object"] == "response"
    assert r["status"] == "completed"
    assert "output" in r
    # 提取 output_text
    text = ""
    for item in r["output"]:
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
    assert len(text) > 0


def test_openai_responses_stream(lynsoo_server):
    """OpenAI Responses SSE 流式。"""
    base = lynsoo_server["base_url"]
    sse = _post_sse(
        f"{base}/v1/responses",
        {
            "model": "lynsoo-auto",
            "input": "讲个笑话",
            "stream": True,
        },
    )
    assert "event: response.created" in sse
    assert "event: response.output_text.delta" in sse
    assert "event: response.completed" in sse


def test_anthropic_messages_non_stream(lynsoo_server):
    """Anthropic Messages 非流式。"""
    base = lynsoo_server["base_url"]
    r = _post_json(
        f"{base}/v1/messages",
        {
            "model": "lynsoo-auto",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "讲个笑话"}],
            "stream": False,
        },
        headers={"x-api-key": "any", "anthropic-version": "2023-06-01"},
    )
    assert r["type"] == "message"
    assert r["role"] == "assistant"
    assert "content" in r
    text = r["content"][0]["text"]
    assert len(text) > 0


def test_anthropic_messages_stream(lynsoo_server):
    """Anthropic SSE 流式。"""
    base = lynsoo_server["base_url"]
    sse = _post_sse(
        f"{base}/v1/messages",
        {
            "model": "lynsoo-auto",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "讲个笑话"}],
            "stream": True,
        },
        headers={"x-api-key": "any", "anthropic-version": "2023-06-01"},
    )
    assert "event: message_start" in sse
    assert "event: content_block_delta" in sse
    assert "event: message_stop" in sse


def test_gemini_generate_content_non_stream(lynsoo_server):
    """Gemini generateContent 非流式。"""
    base = lynsoo_server["base_url"]
    r = _post_json(
        f"{base}/v1beta/models/lynsoo-auto:generateContent",
        {
            "contents": [{"role": "user", "parts": [{"text": "讲个笑话"}]}],
        },
    )
    assert "candidates" in r
    parts = r["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts)
    assert len(text) > 0


def test_gemini_generate_content_stream(lynsoo_server):
    """Gemini streamGenerateContent 流式。"""
    base = lynsoo_server["base_url"]
    sse = _post_sse(
        f"{base}/v1beta/models/lynsoo-auto:streamGenerateContent",
        {"contents": [{"role": "user", "parts": [{"text": "讲个笑话"}]}]},
    )
    assert "data: " in sse
    assert "candidates" in sse


def test_prefer_model_override(lynsoo_server):
    """用户在请求里指定 prefer_model，应该绕过 cost_first 走指定模型。"""
    base = lynsoo_server["base_url"]
    r = _post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "lynsoo-auto",
            "messages": [{"role": "user", "content": "讲个笑话"}],
            "stream": False,
            "x_lynsoo_prefer_model": "mock-claude",
        },
    )
    assert r["x_lynsoo_meta"]["cloud_model"] == "mock-claude"
    text = r["choices"][0]["message"]["content"]
    assert "<anthropic" in text  # mock 上游返回的标记


def test_all_protocols_same_model_name(lynsoo_server):
    """所有 4 种协议应该都能用同一个虚拟模型名 lynsoo-auto。"""
    base = lynsoo_server["base_url"]

    # OpenAI Chat
    r1 = _post_json(
        f"{base}/v1/chat/completions",
        {"model": "lynsoo-auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r1["model"] == "lynsoo-auto"

    # OpenAI Responses
    r2 = _post_json(f"{base}/v1/responses", {"model": "lynsoo-auto", "input": "hi"})
    assert r2["model"] == "lynsoo-auto"

    # Anthropic
    r3 = _post_json(
        f"{base}/v1/messages",
        {"model": "lynsoo-auto", "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r3["model"] == "lynsoo-auto"

    # Gemini（路径里的模型名也用 lynsoo-auto）
    r4 = _post_json(
        f"{base}/v1beta/models/lynsoo-auto:generateContent",
        {"contents": [{"parts": [{"text": "hi"}]}]},
    )
    assert "candidates" in r4
