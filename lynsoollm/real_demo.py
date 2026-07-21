"""
real_demo.py
============
灵枢 LynSooLLM —— 真实模型版演示。

演示内容：
    1) 加载 Gemma-3-270M 路由模型，展示用户选型
    2) multi-LoRA：注册多个 adapter，演示切换与加权合并
    3) 环境感知：不同 DeviceContext 下 selector 自动选 adapter
    4) 完整路由流程：硬路由 + 推测式接力（用 MockLocalModel 模拟本地流式）
    5) 端侧自进化：基于反馈调整阈值

运行：
    python -m lynsoollm.real_demo
"""

from __future__ import annotations

import sys

import torch

from .adapter_selector import EnvAwareSelector, builtin_rules
from .exit_signal import CloudRelay, RelayContext
from .mock_local_model import MockLocalModel
from .real_router_engine import RealRouterEngine
from .real_router_model import RealRouterModel, ROUTE_CLOUD, ROUTE_LOCAL
from .router import DeviceContext, RouterConfig


def banner(t: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {t}")
    print("=" * 64)


def scene1_user_choice() -> None:
    banner("场景 1：用户选型 —— Gemma-3-270M vs Qwen3.5-0.8B")
    print("两个模型都做路由模型，用户可任选。本演示默认用 Gemma（更轻）。")
    print("切换方式：RealRouterEngine(backbone_name='qwen') 即可。\n")

    m = RealRouterModel(backbone_name="gemma", dtype=torch.float32, device="cpu", use_lora=True)
    info = m.info()
    print(f"  当前基底: {info['backbone']}")
    print(f"  总参数  : {info['total_params_M']} M")
    print(f"  可训练  : {info['trainable_params_M']} M (RouteHead + LoRA)")
    print(f"  词表大小: {info['vocab_size']}")
    print(f"  LoRA池  : {info['lora_adapters']}")


def scene2_multi_lora() -> None:
    banner("场景 2：multi-LoRA 切换与加权合并")
    m = RealRouterModel(backbone_name="gemma", dtype=torch.float32, device="cpu", use_lora=True)

    # 注册 3 个 adapter
    m.lora_mgr.load_adapter("weak_net", path=None)
    m.lora_mgr.load_adapter("low_battery", path=None)
    print(f"  已注册 adapter: {m.lora_mgr.list_adapters()}")

    prompt = "用 Rust 实现一个 MVCC 数据库"
    for name in ["default", "weak_net", "low_battery"]:
        m.activate_adapter(name)
        out = m.route(prompt)
        pl = out.route_prob.flatten().tolist()
        print(f"  [{name:12s}] label={out.route_label} prob=[{pl[0]:.3f},{pl[1]:.3f}] ent={out.entropy:.4f}")

    # 加权合并
    merged = m.lora_mgr.merge_weighted({"default": 0.5, "weak_net": 0.3, "low_battery": 0.2})
    out = m.route(prompt)
    pl = out.route_prob.flatten().tolist()
    print(f"  [{merged[:12]:12s}] label={out.route_label} prob=[{pl[0]:.3f},{pl[1]:.3f}] ent={out.entropy:.4f}  (加权合并)")


def scene3_env_aware() -> None:
    banner("场景 3：环境感知 adapter 选择")

    class LocalGen(MockLocalModel):
        def stream(self, prompt, max_new_tokens=32):
            for tok, logits in super().stream(prompt, max_new_tokens=max_new_tokens):
                yield tok, logits

    local = LocalGen(
        default_tokens=["今天", "天气", "很好", "适合", "去", "公园", "散步", "。"],
        entropy_schedule=[0.2, 0.3, 0.4, 0.8, 0.9, 0.7, 0.3, 0.2],
    )

    def cloud_exec(ctx: RelayContext) -> str:
        return f"{ctx.generated_text}<cloud:gpt-4o>"

    engine = RealRouterEngine(
        backbone_name="gemma",
        local_generator=local,
        cloud_relay=CloudRelay(executor=cloud_exec, model_name="gpt-4o"),
        device_ctx=DeviceContext(network_rtt_ms=30, battery_pct=85, temperature_c=30),
    )
    # 预注册 adapter 以便 selector 命中
    for r in builtin_rules():
        try:
            engine.router_model.lora_mgr.load_adapter(r.name, path=None)
        except ValueError:
            pass

    cases = [
        ("网络良好", DeviceContext(network_rtt_ms=30, battery_pct=85, temperature_c=30)),
        ("弱网+低电量", DeviceContext(network_rtt_ms=800, battery_pct=15, temperature_c=42)),
        ("离线", DeviceContext(offline=True)),
    ]
    for name, ctx in cases:
        engine.device_ctx = ctx
        ev_decide = None
        ev_done = None
        for ev in engine.stream("推荐周末活动"):
            if ev.kind == "decide":
                ev_decide = ev
            elif ev.kind == "done":
                ev_done = ev
        decision = engine.selector.last_decision if engine.selector else None
        print(f"\n  [{name}]")
        print(f"    decision: mode={decision['mode']} adapters={decision['adapters']}")
        print(f"    route_label={ev_decide.route_label} route_ms={ev_decide.payload['route_ms']:.1f}ms")
        print(f"    exited={ev_done.payload['exited']} total_ms={ev_done.payload['total_ms']:.1f}ms")


def scene4_self_evolve() -> None:
    banner("场景 4：端侧自进化 —— 用户反馈调整阈值")
    m = RealRouterModel(backbone_name="gemma", dtype=torch.float32, device="cpu", use_lora=True)
    cfg = RouterConfig(entropy_threshold=1.5)
    print(f"  初始阈值: {cfg.entropy_threshold:.3f}")
    # 模拟反馈循环
    for i in range(5):
        # 假设每次都触发了退出但用户点赞（说明不该上云）
        cfg.entropy_threshold = min(cfg.entropy_threshold + 0.05, 5.0)
    print(f"  5 次点赞反馈后: {cfg.entropy_threshold:.3f}  (上调，更倾向本地)")


def main() -> int:
    print("灵枢 LynSooLLM —— 真实模型版（Gemma-3-270M / Qwen3.5-0.8B + multi-LoRA）")
    print(f"PyTorch: {torch.__version__}")
    scene1_user_choice()
    scene2_multi_lora()
    scene3_env_aware()
    scene4_self_evolve()
    banner("真实模型版演示完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
