"""
demo.py
=======
灵枢 LynSooLLM —— SpeculativeRouter 演示脚本。

运行：
    python -m lynsoollm.demo
或：
    python lynsoollm/demo.py

本脚本演示三种场景：
    1) 正常生成（熵始终低于阈值，全程本地）
    2) 触发 Early-Exit（熵中途飙升 -> 接力云端）
    3) 环境感知：弱网/低电量场景下阈值动态上调，更倾向留在本地
"""

from __future__ import annotations

import sys

import torch

from .router import DeviceContext, RouterConfig, SpeculativeRouter
from .mock_local_model import MockLocalModel
from .exit_signal import CloudRelay, RelayContext


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def scene1_normal() -> None:
    banner("场景 1：低熵稳定生成（全程本地，不触发 Early-Exit）")
    # 通过 entropy_schedule 控制每步熵都很低（< 0.5）
    local = MockLocalModel(
        vocab_size=32,
        default_tokens=["灵", "枢", "已", "就", "绪", "，", "请", "讲", "。"],
        entropy_schedule=[0.1, 0.15, 0.2, 0.18, 0.25, 0.3, 0.2, 0.1, 0.05],
    )
    router = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, min_tokens_before_exit=1),
    )
    print("用户 prompt: 你好")
    print("本地输出: ", end="", flush=True)
    for ev in router.stream("你好"):
        if ev.kind == "token":
            print(f"{ev.token}[H={ev.entropy:.2f}] ", end="", flush=True)
        elif ev.kind == "exit":
            print(f"\n  [Early-Exit @ step {ev.step}, H={ev.entropy:.2f}]")
    diag = router.diagnostics()
    print(f"\n  TTFT={diag['ttft_ms']:.2f}ms  总耗时={diag['total_ms']:.2f}ms")
    print(f"  是否上云: {diag['exited']}")


def scene2_early_exit() -> None:
    banner("场景 2：熵中途飙升 -> 触发 Early-Exit 接力云端")

    def fake_cloud_executor(ctx: RelayContext) -> str:
        return f"{ctx.generated_text}<cloud-补全:GPT-4o>"

    local = MockLocalModel(
        vocab_size=32,
        default_tokens=["今天", "天气", "很好", "适合", "去", "公园", "散步", "。"],
        # 前 3 步低熵，第 4 步熵飙升 -> 触发退出
        entropy_schedule=[0.2, 0.3, 0.4, 2.5, 3.0, 2.8, 1.0, 0.5],
    )
    router = SpeculativeRouter(
        local_model=local,
        cloud_relay=CloudRelay(executor=fake_cloud_executor, model_name="gpt-4o"),
        config=RouterConfig(entropy_threshold=1.5, min_tokens_before_exit=1),
    )
    print("用户 prompt: 推荐个周末活动")
    print("混合输出: ", end="", flush=True)
    for ev in router.stream("推荐个周末活动"):
        if ev.kind == "token":
            print(f"{ev.token}[H={ev.entropy:.2f}] ", end="", flush=True)
        elif ev.kind == "exit":
            cloud = ev.payload.get("cloud_text", "") if ev.payload else ""
            print(f"\n  [Early-Exit @ step {ev.step}, H={ev.entropy:.2f}]")
            print(f"  云端接力补全: {cloud}")
    diag = router.diagnostics()
    print(f"\n  熵轨迹: {diag['entropy_trace']}")
    print(f"  阈值轨迹: {diag['threshold_trace']}")
    print(f"  退出位置: step={diag['exit_step']}, 退出熵={diag['exit_entropy']}")


def scene3_env_aware() -> None:
    banner("场景 3：环境感知 —— 弱网/低电量下阈值动态上调")

    local = MockLocalModel(
        vocab_size=32,
        default_tokens=["今天", "天气", "很好", "适合", "去", "公园", "散步", "。"],
        entropy_schedule=[0.2, 0.3, 0.4, 2.5, 3.0, 2.8, 1.0, 0.5],
    )

    print("\n[3.1] 网络良好 (rtt=30ms, 电量=80%)")
    router_good = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, dynamic_threshold=True),
        device_ctx=DeviceContext(network_rtt_ms=30.0, battery_pct=80.0, temperature_c=30.0),
    )
    print(f"  动态阈值: {router_good._resolve_threshold():.3f}")
    text, sig = router_good.route("测试")
    print(f"  触发退出: {bool(sig)}")

    print("\n[3.2] 弱网+低电量 (rtt=800ms, 电量=15%)")
    router_bad = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, dynamic_threshold=True),
        device_ctx=DeviceContext(network_rtt_ms=800.0, battery_pct=15.0, temperature_c=42.0),
    )
    print(f"  动态阈值: {router_bad._resolve_threshold():.3f}")
    text, sig = router_bad.route("测试")
    print(f"  触发退出: {bool(sig)}  （弱网下阈值上调，更倾向留在本地）")

    print("\n[3.3] 完全离线")
    router_offline = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, dynamic_threshold=True),
        device_ctx=DeviceContext(offline=True),
    )
    print(f"  动态阈值: {router_offline._resolve_threshold()}  (inf=永不退出)")
    text, sig = router_offline.route("测试")
    print(f"  触发退出: {bool(sig)}  （离线下 100% 留在本地）")


def scene4_self_evolve() -> None:
    banner("场景 4：端侧自进化 —— 基于用户反馈微调阈值")
    local = MockLocalModel(
        vocab_size=32,
        default_tokens=["a", "b", "c", "d"],
        entropy_schedule=[0.2, 0.3, 2.5, 3.0],
    )
    router = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5),
    )
    print(f"  初始阈值: {router.config.entropy_threshold:.3f}")
    router.route("测试")
    print(f"  触发退出: {bool(router.last_exit_signal)}")

    # 模拟用户反馈：点赞（说明不该上云）-> 阈值上调
    router.update_from_feedback(liked=True)
    print(f"  点赞后阈值: {router.config.entropy_threshold:.3f}  (上调)")

    # 模拟用户反馈：踩（说明该上云却没上）-> 阈值下调
    router.last_exit_signal = None  # 模拟未触发退出
    router.update_from_feedback(liked=False)
    print(f"  踩后阈值: {router.config.entropy_threshold:.3f}  (下调)")


def main() -> int:
    print("灵枢 LynSooLLM —— 边缘感知与多出口推测式端云协同 LLM 智能路由引擎")
    print(f"PyTorch version: {torch.__version__}")

    scene1_normal()
    scene2_early_exit()
    scene3_env_aware()
    scene4_self_evolve()

    banner("全部场景演示完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
