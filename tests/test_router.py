"""
test_router.py
==============
SpeculativeRouter 单元测试。

运行：
    cd /workspace/lynsoollm && python -m pytest tests/ -v
或：
    python tests/test_router.py
"""

from __future__ import annotations

import math
import sys
import os
from pathlib import Path

# 让 tests/ 能直接 import lynsoollm 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lynsoollm.entropy import compute_entropy, normalized_entropy, token_level_entropy
from lynsoollm.exit_signal import CloudRelay, EarlyExitSignal, RelayContext
from lynsoollm.mock_local_model import MockLocalModel
from lynsoollm.router import (
    DeviceContext,
    RouterConfig,
    SpeculativeRouter,
)


# --------------------------------------------------------------------- #
#  entropy 模块
# --------------------------------------------------------------------- #
def test_entropy_uniform_distribution_is_max():
    """均匀分布的熵应等于 log(vocab_size)。"""
    vocab = 16
    logits = torch.zeros(vocab)
    ent = compute_entropy(logits).item()
    expected = math.log(vocab)
    assert abs(ent - expected) < 1e-4, f"期望 {expected}, 实际 {ent}"


def test_entropy_one_hot_is_zero():
    """极度尖锐分布（一个 token 概率=1）熵应为 0。"""
    logits = torch.tensor([1e6, 0.0, 0.0, 0.0])
    ent = compute_entropy(logits).item()
    assert ent < 1e-3, f"期望 ~0, 实际 {ent}"


def test_normalized_entropy_range():
    """归一化熵应在 [0, 1]。"""
    logits = torch.randn(8, 32)
    ent = normalized_entropy(logits)
    assert ent.shape == (8,)
    assert torch.all(ent >= 0) and torch.all(ent <= 1.0 + 1e-5)


def test_token_level_entropy_bit_base():
    logits = torch.zeros(8)  # 均匀
    ent_bit = token_level_entropy(logits, base="bit").item()
    assert abs(ent_bit - 3.0) < 1e-3, f"期望 3 bits, 实际 {ent_bit}"


# --------------------------------------------------------------------- #
#  exit_signal 模块
# --------------------------------------------------------------------- #
def test_relay_context_merges_prompt():
    ctx = RelayContext(
        prompt="你好",
        generated_tokens=["我", "是"],
        generated_text="我是",
        exit_step=2,
        exit_entropy=2.5,
    )
    assert ctx.merged_prompt() == "你好我是"


def test_relay_context_chat_messages_with_partial():
    """本地有部分输出时，chat_messages 应把本地输出作为 assistant 预填，
    而不是拼到 user prompt 里。"""
    ctx = RelayContext(
        prompt="讲个笑话",
        generated_tokens=["你", "好", "，", "我", "是"],
        generated_text="你好，我是",
        exit_step=5,
        exit_entropy=2.5,
    )
    msgs = ctx.chat_messages()
    assert msgs == [
        {"role": "user", "content": "讲个笑话"},
        {"role": "assistant", "content": "你好，我是"},
    ]
    # Gemini 用 "model" 角色
    msgs_gemini = ctx.chat_messages(assistant_role="model")
    assert msgs_gemini == [
        {"role": "user", "content": "讲个笑话"},
        {"role": "model", "content": "你好，我是"},
    ]


def test_relay_context_chat_messages_hard_route():
    """硬路由上云场景（无本地输出）应只返回 user 消息。"""
    ctx = RelayContext(
        prompt="讲个笑话",
        generated_tokens=[],
        generated_text="",
        exit_step=-1,
        exit_entropy=0.0,
        reason="hard_route_cloud",
    )
    msgs = ctx.chat_messages()
    assert msgs == [{"role": "user", "content": "讲个笑话"}]


def test_cloud_relay_default_executor():
    relay = CloudRelay(model_name="gpt-4o")
    ctx = RelayContext(
        prompt="p", generated_tokens=["a"], generated_text="a",
        exit_step=1, exit_entropy=1.0,
    )
    out = relay.handoff(ctx)
    assert "gpt-4o" in out
    assert "a" in out


def test_cloud_relay_custom_executor():
    def exec_fn(ctx: RelayContext) -> str:
        return f"CLOUD({ctx.generated_text})"
    relay = CloudRelay(executor=exec_fn)
    ctx = RelayContext(
        prompt="p", generated_tokens=["x"], generated_text="x",
        exit_step=1, exit_entropy=1.0,
    )
    assert relay.handoff(ctx) == "CLOUD(x)"


def test_early_exit_signal_bool():
    s1 = EarlyExitSignal(triggered=True, exit_step=3, exit_entropy=2.0)
    s2 = EarlyExitSignal(triggered=False)
    assert bool(s1) is True
    assert bool(s2) is False


# --------------------------------------------------------------------- #
#  SpeculativeRouter 核心行为
# --------------------------------------------------------------------- #
def test_router_no_exit_when_low_entropy():
    """所有 token 熵都低于阈值时，不应触发 Early-Exit。"""
    local = MockLocalModel(
        default_tokens=["a", "b", "c", "d"],
        entropy_schedule=[0.1, 0.2, 0.15, 0.1],
    )
    router = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, min_tokens_before_exit=1),
    )
    text, sig = router.route("test")
    assert sig is None or not sig.triggered
    assert text == "abcd"


def test_router_triggers_exit_when_entropy_exceeds():
    """熵超阈值时应在对应 step 触发 Early-Exit。"""
    # 用中文字符避免与 cloud 占位文本里的字母相撞
    local = MockLocalModel(
        default_tokens=["甲", "乙", "丙", "丁", "戊"],
        entropy_schedule=[0.2, 0.3, 2.5, 3.0, 1.0],  # step 2 触发
    )
    router = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, min_tokens_before_exit=1),
    )
    text, sig = router.route("test")
    assert sig is not None and sig.triggered
    assert sig.exit_step == 2
    assert sig.exit_entropy > 1.5
    # 高熵 token '丙' 不应被吐出（已交给云端）
    assert "丙" not in text
    # 本地已生成 甲, 乙
    assert text.startswith("甲乙")


def test_router_respects_min_tokens_before_exit():
    """min_tokens_before_exit=3 时，前 2 步即使熵高也不应退出。"""
    local = MockLocalModel(
        default_tokens=["a", "b", "c", "d"],
        entropy_schedule=[2.5, 2.5, 2.5, 2.5],
    )
    router = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.0, min_tokens_before_exit=3),
    )
    text, sig = router.route("test")
    assert sig is not None and sig.triggered
    assert sig.exit_step >= 2  # 至少到 step 2 才允许


def test_router_offline_never_exits():
    """离线模式下阈值=inf，永不退出。"""
    local = MockLocalModel(
        default_tokens=["a", "b", "c"],
        entropy_schedule=[3.0, 3.0, 3.0],
    )
    router = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.0, dynamic_threshold=True),
        device_ctx=DeviceContext(offline=True),
    )
    text, sig = router.route("test")
    assert sig is None or not sig.triggered
    assert text == "abc"


def test_router_dynamic_threshold_weak_network():
    """弱网+低电量场景下阈值应上调。"""
    local = MockLocalModel(default_tokens=["a"], entropy_schedule=[0.1])
    router_good = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, dynamic_threshold=True),
        device_ctx=DeviceContext(network_rtt_ms=30.0, battery_pct=80.0, temperature_c=30.0),
    )
    router_bad = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, dynamic_threshold=True),
        device_ctx=DeviceContext(network_rtt_ms=800.0, battery_pct=15.0, temperature_c=42.0),
    )
    t_good = router_good._resolve_threshold()
    t_bad = router_bad._resolve_threshold()
    assert t_bad > t_good, f"弱网场景阈值应更高: {t_bad} vs {t_good}"


def test_router_stream_yields_done_event():
    """stream() 最终必须 yield 一个 done 事件。"""
    local = MockLocalModel(
        default_tokens=["a", "b"],
        entropy_schedule=[0.1, 0.2],
    )
    router = SpeculativeRouter(local_model=local)
    kinds = [ev.kind for ev in router.stream("p")]
    assert kinds[-1] == "done"
    assert "token" in kinds


def test_router_self_evolve_feedback():
    """点赞（且触发了退出）应上调阈值。"""
    local = MockLocalModel(
        default_tokens=["a", "b", "c"],
        entropy_schedule=[0.1, 0.2, 2.5],
    )
    router = SpeculativeRouter(
        local_model=local,
        config=RouterConfig(entropy_threshold=1.5, min_tokens_before_exit=1),
    )
    initial = router.config.entropy_threshold
    router.route("test")
    assert router.last_exit_signal and router.last_exit_signal.triggered

    router.update_from_feedback(liked=True)
    assert router.config.entropy_threshold > initial


def test_router_diagnostics_shape():
    local = MockLocalModel(
        default_tokens=["a", "b", "c"],
        entropy_schedule=[0.1, 0.2, 2.5],
    )
    router = SpeculativeRouter(local_model=local, config=RouterConfig(min_tokens_before_exit=1))
    router.route("test")
    diag = router.diagnostics()
    for key in ("ttft_ms", "total_ms", "entropy_trace", "threshold_trace",
                "exited", "exit_step", "exit_entropy", "current_threshold",
                "device_context"):
        assert key in diag, f"缺少诊断字段: {key}"
    assert len(diag["entropy_trace"]) >= 1


def test_trigger_early_exit_manual():
    """手动触发 Early-Exit 应正确构造信号并调用云端。"""
    local = MockLocalModel(default_tokens=["a"], entropy_schedule=[0.1])
    router = SpeculativeRouter(local_model=local)
    sig = router.trigger_early_exit(
        prompt="p", generated_tokens=["x", "y"], step=2, entropy=2.0, reason="manual"
    )
    assert sig.triggered
    assert sig.exit_step == 2
    assert sig.reason == "manual"
    assert sig.relay_context is not None
    assert sig.relay_context.generated_text == "xy"


# --------------------------------------------------------------------- #
#  入口
# --------------------------------------------------------------------- #
def run_all() -> int:
    tests = [
        ("test_entropy_uniform_distribution_is_max", test_entropy_uniform_distribution_is_max),
        ("test_entropy_one_hot_is_zero", test_entropy_one_hot_is_zero),
        ("test_normalized_entropy_range", test_normalized_entropy_range),
        ("test_token_level_entropy_bit_base", test_token_level_entropy_bit_base),
        ("test_relay_context_merges_prompt", test_relay_context_merges_prompt),
        ("test_cloud_relay_default_executor", test_cloud_relay_default_executor),
        ("test_cloud_relay_custom_executor", test_cloud_relay_custom_executor),
        ("test_early_exit_signal_bool", test_early_exit_signal_bool),
        ("test_router_no_exit_when_low_entropy", test_router_no_exit_when_low_entropy),
        ("test_router_triggers_exit_when_entropy_exceeds", test_router_triggers_exit_when_entropy_exceeds),
        ("test_router_respects_min_tokens_before_exit", test_router_respects_min_tokens_before_exit),
        ("test_router_offline_never_exits", test_router_offline_never_exits),
        ("test_router_dynamic_threshold_weak_network", test_router_dynamic_threshold_weak_network),
        ("test_router_stream_yields_done_event", test_router_stream_yields_done_event),
        ("test_router_self_evolve_feedback", test_router_self_evolve_feedback),
        ("test_router_diagnostics_shape", test_router_diagnostics_shape),
        ("test_trigger_early_exit_manual", test_trigger_early_exit_manual),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n  总计: {passed} 通过, {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
