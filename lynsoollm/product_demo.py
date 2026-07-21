"""
product_demo.py
===============
灵枢 LynSooLLM 成品演示脚本。

启动 3 个本地 mock 服务，分别模拟三家厂商的"原生 API"：
    1) OpenAI Responses API       : POST /v1/responses
    2) Anthropic Messages API     : POST /v1/messages
    3) Gemini generateContent API : POST /v1beta/models/{m}:generateContent

然后完整演示：
    1) 用户写配置文件（含 protocol 字段）
    2) 自动拉取定价（manual / models.dev）
    3) 自动算权重 / 优先级（cost_first / quality_first / balanced / manual）
    4) AutoExecutor 根据 provider/endpoint/protocol 自动选执行器
    5) multi-LoRA 注册与环境感知切换
    6) 真实 Gemma-3-270M 路由
    7) 推测式接力：三种原生协议各自跑通
    8) 运行时切换策略 / 设备上下文 / 离线模式
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from .cloud_relay_pool import MockCloudExecutor
from .config import load_config_from_dict, build_app
from .executors import (
    AnthropicMessagesExecutor,
    AutoExecutor,
    GeminiGenerateContentExecutor,
    OpenAIChatExecutor,
    OpenAIResponsesExecutor,
)
from .mock_local_model import MockLocalModel
from .app import LynSooApp


# --------------------------------------------------------------------- #
#  本地 mock 服务：同时模拟三种原生 API 协议
# --------------------------------------------------------------------- #
def start_multi_protocol_mock_server(port: int = 18769) -> HTTPServer:
    """
    启动一个本地 mock 服务，同时响应三种原生协议：
        - POST /v1/responses                                -> OpenAI Responses
        - POST /v1/messages                                 -> Anthropic Messages
        - POST /v1beta/models/{model}:generateContent       -> Gemini generateContent
        - POST /v1/chat/completions                         -> OpenAI Chat（兜底兼容）
    """

    class Handler(BaseHTTPRequestHandler):
        def _read_body(self):
            ln = int(self.headers.get("Content-Length", 0))
            if ln == 0:
                return {}
            return json.loads(self.rfile.read(ln).decode())

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _extract_prompt(req) -> str:
            """从三种协议的请求体里提取用户 prompt。"""
            # OpenAI Chat
            if "messages" in req:
                msgs = req["messages"]
                if msgs and isinstance(msgs, list):
                    return msgs[0].get("content", "")
            # OpenAI Responses
            if "input" in req:
                inp = req["input"]
                if isinstance(inp, str):
                    return inp
                if isinstance(inp, list) and inp:
                    last = inp[-1]
                    if isinstance(last, dict):
                        return last.get("content", "") or str(last)
                return str(inp)
            # Anthropic Messages
            if "messages" in req:
                return req["messages"][0].get("content", "")
            # Gemini
            if "contents" in req:
                c = req["contents"][0]
                parts = c.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""

        def do_POST(self):
            req = self._read_body()
            path = self.path
            prompt = self._extract_prompt(req)[:40]
            model = req.get("model", "unknown")

            # ---------- OpenAI Chat Completions ----------
            if path.endswith("/v1/chat/completions"):
                content = f"<openai_chat:{model}> 续写: {prompt}..."
                self._send({
                    "choices": [
                        {"message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                })
                return

            # ---------- OpenAI Responses API ----------
            if path.endswith("/v1/responses"):
                text = f"<openai_responses:{model}> 续写: {prompt}..."
                self._send({
                    "id": "resp_mock_001",
                    "object": "response",
                    "model": model,
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": text, "annotations": []}
                            ],
                        }
                    ],
                    "output_text": text,
                    "status": "completed",
                })
                return

            # ---------- Anthropic Messages ----------
            if path.endswith("/v1/messages"):
                text = f"<anthropic:{model}> 续写: {prompt}..."
                self._send({
                    "id": "msg_mock_001",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [
                        {"type": "text", "text": text}
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                })
                return

            # ---------- Gemini generateContent ----------
            m = re.match(r"/v1beta/models/([^:]+):generateContent", path)
            if m:
                gemini_model = m.group(1)
                text = f"<gemini:{gemini_model}> 续写: {prompt}..."
                self._send({
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
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10},
                })
                return

            # ---------- 未知路径 ----------
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"unknown endpoint"}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --------------------------------------------------------------------- #
#  本地小模型流式生成器（MockLocalModel 子类）
# --------------------------------------------------------------------- #
class DemoLocalGen(MockLocalModel):
    def stream(self, prompt, max_new_tokens=32):
        for tok, logits in super().stream(prompt, max_new_tokens=max_new_tokens):
            yield tok, logits


# --------------------------------------------------------------------- #
#  演示配置：3 个模型分别走三种原生协议
# --------------------------------------------------------------------- #
def build_demo_config(mock_port: int) -> dict:
    """
    构造演示配置：3 个模型分别使用三种原生 API 协议。

    - mock-gpt4o-responses : provider=openai       + protocol=openai_responses
    - mock-claude          : provider=anthropic    + protocol=anthropic_messages
    - mock-gemini          : provider=gemini       + protocol=gemini_generate
    - mock-deepseek        : provider=deepseek     + protocol=auto（默认 openai_chat）
    """
    base = f"http://127.0.0.1:{mock_port}"
    return {
        "models": [
            # OpenAI 新一代 Responses API
            {
                "name": "mock-gpt4o-responses",
                "provider": "openai",
                "model_id": "gpt-4o",
                "endpoint": f"{base}/v1",
                "api_key": "demo-openai-key",
                "protocol": "openai_responses",
                "pricing_source": "manual",
                "manual_pricing": {"input_per_1k": 0.005, "output_per_1k": 0.015},
                "quality_tier": 4,
            },
            # Anthropic Claude 原生 Messages API
            {
                "name": "mock-claude",
                "provider": "anthropic",
                "model_id": "claude-3.5-sonnet",
                "endpoint": f"{base}/v1",
                "api_key": "demo-anthropic-key",
                "protocol": "anthropic_messages",
                "pricing_source": "manual",
                "manual_pricing": {"input_per_1k": 0.003, "output_per_1k": 0.015},
                "quality_tier": 4,
            },
            # Google Gemini 原生 generateContent API
            {
                "name": "mock-gemini",
                "provider": "gemini",
                "model_id": "gemini-2.5-flash",
                "endpoint": f"{base}/v1beta",
                "api_key": "demo-gemini-key",
                "protocol": "gemini_generate",
                "pricing_source": "manual",
                "manual_pricing": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
                "quality_tier": 3,
            },
            # DeepSeek 走 OpenAI 兼容 Chat Completions（auto 推断）
            {
                "name": "mock-deepseek",
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "endpoint": f"{base}/v1",
                "api_key": "demo-deepseek-key",
                "protocol": "auto",
                "pricing_source": "manual",
                "manual_pricing": {"input_per_1k": 0.0001, "output_per_1k": 0.0003},
                "quality_tier": 3,
            },
        ],
        "router": {
            "backbone": "gemma",
            "strategy": "cost_first",
            "entropy_threshold": 1.5,
            "max_new_tokens": 8,
        },
        "device": {
            "network_rtt_ms": 30,
            "battery_pct": 85,
            "temperature_c": 30,
        },
        "pricing": {
            "cache_path": "/tmp/lynsoollm_demo.json",
            "cache_ttl_sec": 86400,
            "zero_price_fallback": 0.01,
        },
    }


# --------------------------------------------------------------------- #
#  打印工具
# --------------------------------------------------------------------- #
def banner(t: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {t}")
    print("=" * 72)


def print_models(app: LynSooApp) -> None:
    print(f"\n  策略: {app.calculator.strategy}")
    print(f"  {'P':<3}{'名称':<22}{'model_id':<22}{'协议':<22}{'权重':<8}{'成本/1k':<12}{'tier':<5}")
    print("  " + "-" * 80)
    # 通过 AutoExecutor.explain 获取每个模型选中的协议
    for e in sorted(app.registry.list_enabled(), key=lambda x: x.priority):
        p = e.pricing
        cost = f"${p.input_per_1k + p.output_per_1k:.5f}" if p else "n/a"
        # 解释协议
        proto = e.protocol
        if proto == "auto" and isinstance(app.executor, AutoExecutor):
            proto = app.executor.explain(e)["protocol"]
        proto = proto or "auto"
        print(f"  P{e.priority:<2}{e.name:<22}{e.model_id:<22}{proto:<22}"
              f"{e.weight:<8.3f}{cost:<12}{e.quality_tier:<5}")


# --------------------------------------------------------------------- #
#  主演示
# --------------------------------------------------------------------- #
def main() -> int:
    print("灵枢 LynSooLLM —— 成品演示（三家原生 API 协议版）")
    print(f"PyTorch: {torch.__version__}")

    # 1) 启动 mock 服务（一个端口同时支持三种原生协议）
    PORT = 18769
    banner(f"启动本地 mock 服务 (port {PORT})")
    banner_pt2 = "  同时支持: /v1/responses  +  /v1/messages  +  /v1beta/models/:generateContent  +  /v1/chat/completions"
    print(banner_pt2)
    srv = start_multi_protocol_mock_server(PORT)
    time.sleep(0.3)

    # 2) 构造配置 + 启动应用（用 AutoExecutor 自动选协议）
    banner("加载配置 + 自动定价 + AutoExecutor 自动识别协议")
    cfg = load_config_from_dict(build_demo_config(PORT))
    build_app(cfg, fetch_pricing=True, verbose=True)

    local = DemoLocalGen(
        default_tokens=["今天", "天气", "很好", "适合", "去", "公园", "散步", "。"],
        # 在 step 3 熵飙升，触发 Early-Exit
        entropy_schedule=[0.2, 0.3, 0.4, 2.5, 3.0, 2.8, 1.0, 0.5],
    )
    app = LynSooApp(
        cfg,
        local_generator=local,
        executor=AutoExecutor(timeout=5.0),
        device="cpu",
    )
    print_models(app)

    # 3) 场景1: cost_first -> 接力到 deepseek（最便宜，走 openai_chat）
    banner("场景1: cost_first 策略 -> 接力到 mock-deepseek (最便宜)")
    banner_pt2 = "  走 OpenAI Chat Completions /v1/chat/completions"
    print(banner_pt2)
    print("  Prompt: '讲个笑话'")
    print("  流式:")
    for ev in app.stream("讲个笑话"):
        if ev.kind == "decide":
            print(f"    [decide] label={'cloud' if ev.route_label==1 else 'local'} "
                  f"adapter={ev.adapter} route_ms={ev.payload['route_ms']:.0f}")
        elif ev.kind == "token":
            print(f"      token[{ev.step}] {ev.token} H={ev.entropy:.2f} thr={ev.threshold:.2f}")
        elif ev.kind == "exit":
            p = ev.payload
            print(f"      [EXIT@{ev.step}] H={ev.entropy:.2f} -> model={p['model']}")
            print(f"      cloud: {p['cloud_text']}")
        elif ev.kind == "done":
            print(f"    [done] ttft={ev.payload['ttft_ms']:.0f}ms total={ev.payload['total_ms']:.0f}ms")

    # 4) 场景2: quality_first + prefer_model=mock-claude -> 走 Anthropic Messages
    banner("场景2: quality_first + prefer_model=mock-claude")
    banner_pt2 = "  走 Anthropic Messages API /v1/messages (x-api-key + anthropic-version)"
    print(banner_pt2)
    app.switch_strategy("quality_first")
    print_models(app)
    result = app.route("讲个笑话", prefer_model="mock-claude")
    print(f"\n  结果: {result['text']}")
    print(f"  exit_model: {result.get('exit_model')}  (Anthropic 协议)")

    # 5) 场景3: prefer_model=mock-gemini -> 走 Gemini generateContent
    banner("场景3: prefer_model=mock-gemini")
    banner_pt2 = "  走 Gemini generateContent /v1beta/models/{model}:generateContent (x-goog-api-key)"
    print(banner_pt2)
    result = app.route("讲个笑话", prefer_model="mock-gemini")
    print(f"\n  结果: {result['text']}")
    print(f"  exit_model: {result.get('exit_model')}  (Gemini 协议)")

    # 6) 场景4: prefer_model=mock-gpt4o-responses -> 走 OpenAI Responses API
    banner("场景4: prefer_model=mock-gpt4o-responses")
    banner_pt2 = "  走 OpenAI Responses API /v1/responses (input + instructions)"
    print(banner_pt2)
    result = app.route("讲个笑话", prefer_model="mock-gpt4o-responses")
    print(f"\n  结果: {result['text']}")
    print(f"  exit_model: {result.get('exit_model')}  (OpenAI Responses 协议)")

    # 7) 场景5: 弱网+低电量 (multi-LoRA 环境感知)
    banner("场景5: 弱网+低电量+高温 (multi-LoRA 环境感知切换)")
    print("  切换设备上下文: rtt=800ms, battery=15%, temp=42°C")
    app.update_device(network_rtt_ms=800, battery_pct=15, temperature_c=42)
    print(f"  当前 LoRA adapter: {app.engine.router_model.lora_mgr.active}")
    print(f"  selector 决策: {app.engine.selector.last_decision}")
    result = app.route("讲个笑话")
    print(f"  结果: {result['text']}")
    print(f"  exit_model: {result.get('exit_model')}")

    # 8) 场景6: 离线模式 -> 强制本地
    banner("场景6: 离线模式 -> 强制本地，不上云")
    app.update_device(offline=True)
    result = app.route("测试")
    print(f"  结果: {result['text']}")
    print(f"  exited: {result['exited']}  (应为 False)")

    # 9) 场景7: manual 策略 -> 用户手动配置 priority/weight
    banner("场景7: manual 策略 -> 用户手动配置 priority/weight")
    app.update_device(offline=False)
    for e in app.registry.entries:
        if e.name == "mock-claude":
            e.priority = 1
            e.weight = 0.8
        elif e.name == "mock-gpt4o-responses":
            e.priority = 2
            e.weight = 0.15
        else:
            e.priority = 3
            e.weight = 0.05
    app.switch_strategy("manual")
    print_models(app)
    result = app.route("讲个笑话")
    print(f"\n  结果: {result['text']}")
    print(f"  exit_model: {result.get('exit_model')}  (应为 mock-claude，用户手动置顶)")

    # 10) 总结
    banner("成品演示完成 —— 三家原生协议全部支持")
    print("""
  已支持的协议（4 种 + 自动选）:
    1) OpenAI Chat Completions : /v1/chat/completions                 (兼容性最广)
    2) OpenAI Responses API    : /v1/responses                         (新一代)
    3) Anthropic Messages API  : /v1/messages                          (Claude 原生)
    4) Gemini generateContent  : /v1beta/models/{m}:generateContent    (Gemini 原生)
    5) AutoExecutor            : 根据 provider/endpoint/protocol 自动选

  用户接入步骤：
    1. 编辑 config.example.yaml，填入你的模型 endpoint / api_key / protocol
    2. 选择 pricing_source（models_dev / official / custom_api / manual）
    3. 选择 strategy（cost_first / quality_first / balanced / latency_first / manual）
    4. 选择 backbone（gemma / qwen）
    5. 跑：
         from lynsoollm import LynSooApp
         from lynsoollm.executors import AutoExecutor
         app = LynSooApp.from_config('config.yaml',
                                      local_generator=your_local_llm,
                                      executor=AutoExecutor())
         print(app.route('你好')['text'])
""")
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
