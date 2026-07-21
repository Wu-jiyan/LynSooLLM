"""
router.py
=========
SpeculativeRouter —— 灵枢 LynSooLLM 的核心路由类。

实现"多出口推测式平滑接力"机制：
    1. 用户输入后，本地小模型零延迟直接启动流式生成；
    2. 路由模型在后台实时监控每步 token 的信息熵；
    3. 一旦熵超过阈值，立即执行 Early-Exit，将已生成上下文
       无缝接力给云端大模型，实现用户无感知的平滑切换。

设计原则：
    - 与具体本地推理引擎解耦（只要能流式 yield (token, logits)）。
    - 与具体云端 API 解耦（通过 CloudRelay 注入）。
    - 端侧可注入环境感知上下文（网络/电量/温度），用于动态调阈值。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch

from .entropy import compute_entropy, normalized_entropy
from .exit_signal import CloudRelay, EarlyExitSignal, RelayContext
from .mock_local_model import MockLocalModel


# --------------------------------------------------------------------- #
#  环境感知上下文
# --------------------------------------------------------------------- #
@dataclass
class DeviceContext:
    """
    设备物理状态。在真实系统中由感知层填入。

    路由器会据此动态调整熵阈值：弱网/低电量时阈值下调，
    更倾向于留在本地；网络良好时阈值上调，更愿意上云。
    """

    network_rtt_ms: float = 50.0          # 网络往返延迟
    battery_pct: float = 80.0             # 电量百分比 0~100
    temperature_c: float = 35.0           # 设备温度
    offline: bool = False                 # 是否完全离线
    cloud_price_per_1k: float = 0.01      # 云端 API 单价（美元/1k token）


# --------------------------------------------------------------------- #
#  路由配置
# --------------------------------------------------------------------- #
@dataclass
class RouterConfig:
    """路由器运行时配置。"""

    entropy_threshold: float = 1.5         # 熵阈值（nats），超过即触发 Early-Exit
    use_normalized: bool = False           # 是否使用归一化熵（0~1）
    min_tokens_before_exit: int = 1        # 至少生成多少 token 后才允许退出
    max_new_tokens: int = 32               # 本地最大生成 token 数
    dynamic_threshold: bool = True         # 是否根据 DeviceContext 动态调整阈值
    # 动态阈值系数：rtt 越高、电量越低，阈值上调（更宽容本地）
    rtt_sensitivity: float = 0.002         # 每毫秒 RTT 增加的阈值
    battery_sensitivity: float = 0.01      # 电量每降 1% 增加的阈值


# --------------------------------------------------------------------- #
#  生成事件（供上层消费）
# --------------------------------------------------------------------- #
@dataclass
class GenerationEvent:
    """流式事件。"""

    kind: str                              # "token" | "exit" | "done"
    step: int = -1
    token: str = ""
    entropy: float = 0.0
    threshold: float = 0.0
    payload: Any = None


# --------------------------------------------------------------------- #
#  核心路由类
# --------------------------------------------------------------------- #
class SpeculativeRouter:
    """
    推测式路由器：边生成边评估信息熵，超阈值即静默打断并接力云端。

    参数:
        local_model : 提供 ``stream(prompt, max_new_tokens)`` 接口的对象。
                      默认使用 MockLocalModel。
        cloud_relay : 云端接力器，默认 CloudRelay()（占位）。
        config      : RouterConfig。
        device_ctx  : 设备环境上下文，用于动态阈值。

    用法:
        >>> router = SpeculativeRouter()
        >>> for ev in router.stream("讲个笑话"):
        ...     if ev.kind == "token":
        ...         print(ev.token, end="", flush=True)
        ...     elif ev.kind == "exit":
        ...         print(f"\\n[Early-Exit @ step {ev.step}, H={ev.entropy:.3f}]")
    """

    def __init__(
        self,
        local_model: Optional[MockLocalModel] = None,
        cloud_relay: Optional[CloudRelay] = None,
        config: Optional[RouterConfig] = None,
        device_ctx: Optional[DeviceContext] = None,
    ) -> None:
        self.local_model = local_model or MockLocalModel()
        self.cloud_relay = cloud_relay or CloudRelay()
        self.config = config or RouterConfig()
        self.device_ctx = device_ctx or DeviceContext()

        # 运行时统计
        self.last_entropy_trace: List[float] = []
        self.last_threshold_trace: List[float] = []
        self.last_exit_signal: Optional[EarlyExitSignal] = None
        self.last_ttft_ms: float = 0.0       # 首字延迟
        self.last_total_ms: float = 0.0

    # ------------------------------------------------------------------ #
    #  动态阈值
    # ------------------------------------------------------------------ #
    def _resolve_threshold(self) -> float:
        """根据 DeviceContext 动态调整熵阈值。"""
        if not self.config.dynamic_threshold:
            return self.config.entropy_threshold

        base = self.config.entropy_threshold
        # 离线：永远不退出（阈值无穷大）
        if self.device_ctx.offline:
            return float("inf")

        adj = 0.0
        # 弱网：rtt 越大越倾向留在本地 -> 阈值上调
        adj += max(0.0, self.device_ctx.network_rtt_ms - 100.0) * self.config.rtt_sensitivity
        # 低电量：阈值上调
        adj += max(0.0, 50.0 - self.device_ctx.battery_pct) * self.config.battery_sensitivity
        # 高温：阈值上调（避免云端往返加热）
        adj += max(0.0, self.device_ctx.temperature_c - 40.0) * 0.02
        return base + adj

    # ------------------------------------------------------------------ #
    #  核心：流式生成 + 实时熵评估
    # ------------------------------------------------------------------ #
    def stream(self, prompt: str) -> Iterator[GenerationEvent]:
        """
        流式生成主入口。

        每生成一个 token：
            1. 计算其 logits 的信息熵；
            2. 若熵 > 阈值且已过最小保留步数，触发 Early-Exit；
            3. 否则正常吐出该 token。

        触发退出时，会构造 RelayContext 并调用 cloud_relay.handoff，
        然后以一个 "exit" 事件 + "done" 事件结束流。
        """
        # 重置统计
        self.last_entropy_trace = []
        self.last_threshold_trace = []
        self.last_exit_signal = None

        threshold = self._resolve_threshold()
        generated_tokens: List[str] = []
        logits_list: List[torch.Tensor] = []
        start_ts = time.perf_counter()
        first_token_ts: Optional[float] = None
        exit_step = -1
        exit_entropy = 0.0

        for step, (token, logits) in enumerate(
            self.local_model.stream(prompt, max_new_tokens=self.config.max_new_tokens)
        ):
            if first_token_ts is None:
                first_token_ts = time.perf_counter()
                self.last_ttft_ms = (first_token_ts - start_ts) * 1000.0

            # 计算熵
            if self.config.use_normalized:
                ent_tensor = normalized_entropy(logits.unsqueeze(0))
            else:
                ent_tensor = compute_entropy(logits.unsqueeze(0))
            entropy = float(ent_tensor.item())
            self.last_entropy_trace.append(entropy)
            self.last_threshold_trace.append(threshold)

            # 判定是否触发 Early-Exit
            should_exit = (
                entropy > threshold
                and step >= self.config.min_tokens_before_exit
            )

            if should_exit:
                exit_step = step
                exit_entropy = entropy
                # 构造接力上下文
                ctx = RelayContext(
                    prompt=prompt,
                    generated_tokens=list(generated_tokens),
                    generated_text="".join(generated_tokens),
                    exit_step=step,
                    exit_entropy=entropy,
                    reason="entropy_threshold_exceeded",
                    metadata={
                        "threshold": threshold,
                        "logits_shape": tuple(logits.shape),
                    },
                )
                # 静默接力云端（这里同步调用，真实系统可异步）
                cloud_text = self.cloud_relay.handoff(ctx)
                self.last_exit_signal = EarlyExitSignal(
                    triggered=True,
                    exit_step=step,
                    exit_entropy=entropy,
                    reason="entropy_threshold_exceeded",
                    relay_context=ctx,
                )
                # 注意：当前 token（高熵那个）不吐给用户，
                # 由云端从其 KV-cache 续写，避免用户看到幻觉 token。
                yield GenerationEvent(
                    kind="exit",
                    step=step,
                    token="",
                    entropy=entropy,
                    threshold=threshold,
                    payload={"cloud_text": cloud_text, "relay_context": ctx},
                )
                break

            # 正常吐出 token
            generated_tokens.append(token)
            logits_list.append(logits)
            yield GenerationEvent(
                kind="token",
                step=step,
                token=token,
                entropy=entropy,
                threshold=threshold,
            )
        else:
            # 未触发退出，正常结束
            pass

        self.last_total_ms = (time.perf_counter() - start_ts) * 1000.0
        yield GenerationEvent(
            kind="done",
            step=exit_step if exit_step >= 0 else len(generated_tokens) - 1,
            entropy=exit_entropy,
            threshold=threshold,
            payload={
                "ttft_ms": self.last_ttft_ms,
                "total_ms": self.last_total_ms,
                "exited": exit_step >= 0,
            },
        )

    # ------------------------------------------------------------------ #
    #  非流式便捷接口
    # ------------------------------------------------------------------ #
    def route(self, prompt: str) -> Tuple[str, Optional[EarlyExitSignal]]:
        """
        非流式调用：返回最终文本（本地 + 可能的云端接力）与退出信号。

        若中途触发 Early-Exit，最终文本 = 本地已生成 + 云端补全。
        """
        local_parts: List[str] = []
        cloud_text = ""
        signal: Optional[EarlyExitSignal] = None

        for ev in self.stream(prompt):
            if ev.kind == "token":
                local_parts.append(ev.token)
            elif ev.kind == "exit":
                signal = self.last_exit_signal
                if ev.payload and "cloud_text" in ev.payload:
                    cloud_text = ev.payload["cloud_text"]
            # done 事件忽略

        final_text = f"{''.join(local_parts)}{cloud_text}"
        return final_text, signal

    # ------------------------------------------------------------------ #
    #  显式打断接口（供外部看门狗使用）
    # ------------------------------------------------------------------ #
    def trigger_early_exit(
        self,
        prompt: str,
        generated_tokens: List[str],
        step: int,
        entropy: float,
        reason: str = "manual",
    ) -> EarlyExitSignal:
        """外部主动触发 Early-Exit，构造信号并完成云端接力。"""
        ctx = RelayContext(
            prompt=prompt,
            generated_tokens=list(generated_tokens),
            generated_text="".join(generated_tokens),
            exit_step=step,
            exit_entropy=entropy,
            reason=reason,
        )
        cloud_text = self.cloud_relay.handoff(ctx)
        signal = EarlyExitSignal(
            triggered=True,
            exit_step=step,
            exit_entropy=entropy,
            reason=reason,
            relay_context=ctx,
        )
        signal.relay_context.metadata["cloud_text"] = cloud_text  # type: ignore[union-attr]
        self.last_exit_signal = signal
        return signal

    # ------------------------------------------------------------------ #
    #  端侧自进化接口（占位：在线感知机更新阈值）
    # ------------------------------------------------------------------ #
    def update_from_feedback(self, liked: bool, entropy_at_exit: Optional[float] = None) -> None:
        """
        基于用户隐式反馈微调路由决策边界（极轻量在线学习）。

        - 若用户点赞但触发了 Early-Exit（即上云了），说明阈值过低，
          应上调以更倾向本地；
        - 若用户踩且未触发 Early-Exit（即留在本地），说明阈值过高，
          应下调以更倾向上云。

        这里仅做阈值层面的标量更新，对应开题中"端侧在线感知机"
        的最小实现，后续可扩展为 270M 模型的 LoRA 微调。
        """
        if entropy_at_exit is None:
            entropy_at_exit = (
                self.last_exit_signal.exit_entropy if self.last_exit_signal else self.config.entropy_threshold
            )

        if liked and self.last_exit_signal and self.last_exit_signal.triggered:
            # 阈值过低（不该上云却上了）-> 上调
            self.config.entropy_threshold = min(
                self.config.entropy_threshold + 0.05, 5.0
            )
        elif (not liked) and not (self.last_exit_signal and self.last_exit_signal.triggered):
            # 阈值过高（该上云却没上）-> 下调
            self.config.entropy_threshold = max(
                self.config.entropy_threshold - 0.05, 0.1
            )

    # ------------------------------------------------------------------ #
    #  诊断信息
    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict:
        return {
            "ttft_ms": round(self.last_ttft_ms, 3),
            "total_ms": round(self.last_total_ms, 3),
            "entropy_trace": [round(e, 4) for e in self.last_entropy_trace],
            "threshold_trace": [round(t, 4) for t in self.last_threshold_trace],
            "exited": bool(self.last_exit_signal and self.last_exit_signal.triggered),
            "exit_step": self.last_exit_signal.exit_step if self.last_exit_signal else -1,
            "exit_entropy": (
                round(self.last_exit_signal.exit_entropy, 4)
                if self.last_exit_signal
                else None
            ),
            "current_threshold": round(self._resolve_threshold(), 4),
            "device_context": {
                "rtt_ms": self.device_ctx.network_rtt_ms,
                "battery_pct": self.device_ctx.battery_pct,
                "temperature_c": self.device_ctx.temperature_c,
                "offline": self.device_ctx.offline,
            },
        }
