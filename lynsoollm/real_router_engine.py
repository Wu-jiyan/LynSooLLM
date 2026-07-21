"""
real_router_engine.py
=====================
SpeculativeRouter 的真实模型版本：用 Gemma-3-270M / Qwen3.5-0.8B 做
路由模型，叠加 multi-LoRA 支持环境感知动态切换，并保留"边生成边评估
信息熵 -> Early-Exit 接力云端"的核心机制。

与 mock 版本（router.py）的关系：
    - router.py             : 原型核心，使用 MockLocalModel，用于流程验证
    - real_router_engine.py : 真实模型版，使用 RealRouterModel + MultiLoRA

工作流（一次路由）：
    1. 用户给出 prompt + 可选 DeviceContext
    2. EnvAwareSelector 根据 ctx 选/合并 LoRA adapter
    3. RealRouterModel.route(prompt) 输出：
         - route_label  : 硬路由（local / cloud）
         - entropy      : 看门狗熵（基底 LM 在 prompt 末尾的归一化熵）
    4. 决策：
         - route_label == cloud 且非离线 → 直接上云（硬路由）
         - route_label == local → 启动本地流式生成，每步算 token 熵，
           超阈值即 Early-Exit 接力云端（推测式软路由）
    5. 端侧自进化：根据用户隐式反馈，调整阈值/微调活跃 LoRA
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from .adapter_selector import EnvAwareSelector
from .entropy import compute_entropy, normalized_entropy
from .exit_signal import CloudRelay, EarlyExitSignal, RelayContext
from .real_router_model import (
    ROUTE_CLOUD,
    ROUTE_LOCAL,
    RealRouterModel,
    RouterOutput,
)
from .router import DeviceContext, RouterConfig


# --------------------------------------------------------------------- #
#  路由事件（与 mock 版兼容，便于上层复用）
# --------------------------------------------------------------------- #
@dataclass
class RealRouterEvent:
    kind: str                       # "decide" | "token" | "exit" | "done"
    step: int = -1
    token: str = ""
    entropy: float = 0.0
    threshold: float = 0.0
    route_label: int = -1
    route_prob: Optional[List[float]] = None
    adapter: str = ""
    payload: Any = None


# --------------------------------------------------------------------- #
#  本地流式生成器抽象（业务侧可注入真实 vLLM/llama.cpp）
# --------------------------------------------------------------------- #
class LocalGenerator:
    """
    本地小模型流式生成接口。

    真实部署时替换为 ONNX Runtime / llama.cpp 封装，
    只要 yield (token_str, logits_tensor) 即可。
    """

    def stream(self, prompt: str, max_new_tokens: int = 32) -> Iterator[Tuple[str, torch.Tensor]]:
        raise NotImplementedError


# --------------------------------------------------------------------- #
#  真实路由引擎
# --------------------------------------------------------------------- #
class RealRouterEngine:
    """
    真实模型版路由引擎。

    参数:
        backbone_name : "gemma" / "qwen" / 自定义路径，用户选型
        local_generator: 本地小模型流式生成器（可选，无则只做硬路由）
        cloud_relay   : 云端接力器
        config        : RouterConfig（沿用 mock 版配置）
        device_ctx    : 设备环境上下文
        use_lora      : 是否启用 multi-LoRA
        enable_selector: 是否启用环境感知 adapter 选择器

    用法:
        engine = RealRouterEngine(backbone_name="gemma")
        for ev in engine.stream("讲个笑话"):
            if ev.kind == "token": print(ev.token, end="", flush=True)
            elif ev.kind == "exit": print(f"\\n[exit@{ev.step}]")
    """

    def __init__(
        self,
        backbone_name: str = "gemma",
        local_generator: Optional[LocalGenerator] = None,
        cloud_relay: Optional[CloudRelay] = None,
        config: Optional[RouterConfig] = None,
        device_ctx: Optional[DeviceContext] = None,
        use_lora: bool = True,
        enable_selector: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or RouterConfig()
        self.device_ctx = device_ctx or DeviceContext()
        self.cloud_relay = cloud_relay or CloudRelay()
        self.local_generator = local_generator

        # 加载真实路由模型（含 multi-LoRA）
        self.router_model = RealRouterModel(
            backbone_name=backbone_name,
            dtype=dtype,
            device=device,
            use_lora=use_lora,
        )

        # 环境感知选择器
        self.selector: Optional[EnvAwareSelector] = None
        if enable_selector and use_lora and self.router_model.lora_mgr:
            # 初始时只有 default adapter，selector 仍可决策（落回 default）
            self.selector = EnvAwareSelector(self.router_model.lora_mgr)

        # 运行时统计
        self.last_router_output: Optional[RouterOutput] = None
        self.last_entropy_trace: List[float] = []
        self.last_threshold_trace: List[float] = []
        self.last_exit_signal: Optional[EarlyExitSignal] = None
        self.last_ttft_ms: float = 0.0
        self.last_total_ms: float = 0.0
        self.last_route_ms: float = 0.0
        self.last_adapter: str = ""

    # ------------------------------------------------------------------ #
    #  动态阈值（沿用 mock 版逻辑，但可被路由模型熵校准）
    # ------------------------------------------------------------------ #
    def _resolve_threshold(self, base_entropy: Optional[float] = None) -> float:
        """根据 DeviceContext + 路由模型给出的参考熵，调整 token 熵阈值。"""
        if not self.config.dynamic_threshold:
            return self.config.entropy_threshold
        if self.device_ctx.offline:
            return float("inf")

        base = self.config.entropy_threshold
        adj = 0.0
        adj += max(0.0, self.device_ctx.network_rtt_ms - 100.0) * self.config.rtt_sensitivity
        adj += max(0.0, 50.0 - self.device_ctx.battery_pct) * self.config.battery_sensitivity
        adj += max(0.0, self.device_ctx.temperature_c - 40.0) * 0.02
        # 若路由模型给出的硬路由是 cloud，阈值适度上调（更信任硬路由决策，
        # 不轻易在本地流式中再次退出）
        if self.last_router_output and self.last_router_output.route_label == ROUTE_CLOUD:
            adj += 0.2
        return base + adj

    # ------------------------------------------------------------------ #
    #  1) 路由决策（硬路由）
    # ------------------------------------------------------------------ #
    def decide(self, prompt: str) -> RouterOutput:
        """
        调用真实路由模型做硬路由决策。
        会先根据 DeviceContext 切/合并 LoRA adapter。
        """
        # 环境感知选 adapter
        if self.selector:
            try:
                self.last_adapter = self.selector.apply(self.device_ctx)
            except Exception:
                # 若 adapter 未注册等情况，退回 default
                self.last_adapter = self.router_model.lora_mgr.active  # type: ignore[union-attr]
        else:
            self.last_adapter = (
                self.router_model.lora_mgr.active if self.router_model.lora_mgr else "none"
            )

        t0 = time.perf_counter()
        out = self.router_model.route(prompt)
        self.last_route_ms = (time.perf_counter() - t0) * 1000.0
        self.last_router_output = out
        return out

    # ------------------------------------------------------------------ #
    #  2) 流式生成 + 推测式接力
    # ------------------------------------------------------------------ #
    def stream(self, prompt: str) -> Iterator[RealRouterEvent]:
        """
        完整流程：硬路由决策 -> 本地流式生成 -> 熵监控 -> Early-Exit。

        事件序列：
            decide -> (token|exit)* -> done
        """
        self.last_entropy_trace = []
        self.last_threshold_trace = []
        self.last_exit_signal = None
        start_ts = time.perf_counter()

        # ---- 硬路由决策 ----
        route_out = self.decide(prompt)
        yield RealRouterEvent(
            kind="decide",
            step=-1,
            entropy=route_out.entropy,
            threshold=0.0,
            route_label=route_out.route_label,
            route_prob=route_out.route_prob.tolist(),
            adapter=self.last_adapter,
            payload={"route_ms": self.last_route_ms},
        )

        # ---- 离线：强制本地 ----
        if self.device_ctx.offline:
            route_label = ROUTE_LOCAL
        else:
            route_label = route_out.route_label

        # ---- 硬路由：直接上云 ----
        if route_label == ROUTE_CLOUD and not self.device_ctx.offline:
            ctx = RelayContext(
                prompt=prompt,
                generated_tokens=[],
                generated_text="",
                exit_step=-1,
                exit_entropy=route_out.entropy,
                reason="hard_route_cloud",
                metadata={"route_prob": route_out.route_prob.tolist()},
            )
            cloud_text = self.cloud_relay.handoff(ctx)
            self.last_exit_signal = EarlyExitSignal(
                triggered=True, exit_step=-1,
                exit_entropy=route_out.entropy,
                reason="hard_route_cloud", relay_context=ctx,
            )
            yield RealRouterEvent(
                kind="exit",
                step=-1,
                entropy=route_out.entropy,
                route_label=ROUTE_CLOUD,
                adapter=self.last_adapter,
                payload={"cloud_text": cloud_text, "hard_route": True},
            )
            self.last_total_ms = (time.perf_counter() - start_ts) * 1000.0
            yield RealRouterEvent(
                kind="done", step=-1, entropy=route_out.entropy,
                payload={
                    "ttft_ms": self.last_route_ms,
                    "total_ms": self.last_total_ms,
                    "exited": True, "hard_route": True,
                },
            )
            return

        # ---- 本地流式生成 + 推测式监控 ----
        if self.local_generator is None:
            # 无本地生成器：直接返回路由结果（仅硬路由可用）
            self.last_total_ms = (time.perf_counter() - start_ts) * 1000.0
            yield RealRouterEvent(
                kind="done", step=-1, entropy=route_out.entropy,
                payload={
                    "ttft_ms": self.last_route_ms,
                    "total_ms": self.last_total_ms,
                    "exited": False, "reason": "no_local_generator",
                },
            )
            return

        threshold = self._resolve_threshold(base_entropy=route_out.entropy)
        generated_tokens: List[str] = []
        first_token_ts: Optional[float] = None
        exit_step = -1
        exit_entropy = 0.0

        for step, (token, logits) in enumerate(
            self.local_generator.stream(prompt, max_new_tokens=self.config.max_new_tokens)
        ):
            if first_token_ts is None:
                first_token_ts = time.perf_counter()
                self.last_ttft_ms = (first_token_ts - start_ts) * 1000.0

            if self.config.use_normalized:
                ent = float(normalized_entropy(logits.unsqueeze(0)).item())
            else:
                ent = float(compute_entropy(logits.unsqueeze(0)).item())
            self.last_entropy_trace.append(ent)
            self.last_threshold_trace.append(threshold)

            should_exit = (
                ent > threshold
                and step >= self.config.min_tokens_before_exit
            )
            if should_exit:
                exit_step = step
                exit_entropy = ent
                ctx = RelayContext(
                    prompt=prompt,
                    generated_tokens=list(generated_tokens),
                    generated_text="".join(generated_tokens),
                    exit_step=step,
                    exit_entropy=ent,
                    reason="entropy_threshold_exceeded",
                    metadata={
                        "threshold": threshold,
                        "route_label": route_out.route_label,
                        "adapter": self.last_adapter,
                    },
                )
                cloud_text = self.cloud_relay.handoff(ctx)
                self.last_exit_signal = EarlyExitSignal(
                    triggered=True, exit_step=step,
                    exit_entropy=ent,
                    reason="entropy_threshold_exceeded",
                    relay_context=ctx,
                )
                yield RealRouterEvent(
                    kind="exit", step=step, token="",
                    entropy=ent, threshold=threshold,
                    route_label=route_out.route_label,
                    adapter=self.last_adapter,
                    payload={"cloud_text": cloud_text, "relay_context": ctx},
                )
                break

            generated_tokens.append(token)
            yield RealRouterEvent(
                kind="token", step=step, token=token,
                entropy=ent, threshold=threshold,
                route_label=route_out.route_label,
                adapter=self.last_adapter,
            )

        self.last_total_ms = (time.perf_counter() - start_ts) * 1000.0
        yield RealRouterEvent(
            kind="done",
            step=exit_step if exit_step >= 0 else len(generated_tokens) - 1,
            entropy=exit_entropy, threshold=threshold,
            route_label=route_out.route_label,
            adapter=self.last_adapter,
            payload={
                "ttft_ms": self.last_ttft_ms,
                "total_ms": self.last_total_ms,
                "route_ms": self.last_route_ms,
                "exited": exit_step >= 0,
                "hard_route": False,
            },
        )

    # ------------------------------------------------------------------ #
    #  非流式
    # ------------------------------------------------------------------ #
    def route(self, prompt: str) -> Tuple[str, Optional[EarlyExitSignal]]:
        parts: List[str] = []
        cloud_text = ""
        signal: Optional[EarlyExitSignal] = None
        for ev in self.stream(prompt):
            if ev.kind == "token":
                parts.append(ev.token)
            elif ev.kind == "exit":
                signal = self.last_exit_signal
                if ev.payload and "cloud_text" in ev.payload:
                    cloud_text = ev.payload["cloud_text"]
        return f"{''.join(parts)}{cloud_text}", signal

    # ------------------------------------------------------------------ #
    #  端侧自进化
    # ------------------------------------------------------------------ #
    def feedback(self, liked: bool, prompt: str = "") -> None:
        """
        用户隐式反馈 -> 阈值标量更新（与 mock 版一致）。
        若启用了 LoRA 且有 personalized adapter，可进一步触发在线训练
        （见 train_router.py）。
        """
        if liked and self.last_exit_signal and self.last_exit_signal.triggered:
            self.config.entropy_threshold = min(self.config.entropy_threshold + 0.05, 5.0)
        elif (not liked) and not (self.last_exit_signal and self.last_exit_signal.triggered):
            self.config.entropy_threshold = max(self.config.entropy_threshold - 0.05, 0.1)

    # ------------------------------------------------------------------ #
    #  诊断
    # ------------------------------------------------------------------ #
    def diagnostics(self) -> Dict:
        return {
            "backbone": self.router_model.backbone_name,
            "ttft_ms": round(self.last_ttft_ms, 3),
            "route_ms": round(self.last_route_ms, 3),
            "total_ms": round(self.last_total_ms, 3),
            "entropy_trace": [round(e, 4) for e in self.last_entropy_trace],
            "threshold_trace": [round(t, 4) for t in self.last_threshold_trace],
            "exited": bool(self.last_exit_signal and self.last_exit_signal.triggered),
            "exit_step": self.last_exit_signal.exit_step if self.last_exit_signal else -1,
            "exit_entropy": (
                round(self.last_exit_signal.exit_entropy, 4)
                if self.last_exit_signal else None
            ),
            "current_threshold": round(self._resolve_threshold(), 4),
            "active_adapter": self.last_adapter,
            "adapter_decision": self.selector.last_decision if self.selector else None,
            "device_context": {
                "rtt_ms": self.device_ctx.network_rtt_ms,
                "battery_pct": self.device_ctx.battery_pct,
                "temperature_c": self.device_ctx.temperature_c,
                "offline": self.device_ctx.offline,
                "cloud_price_per_1k": self.device_ctx.cloud_price_per_1k,
            },
            "router_model_info": self.router_model.info(),
        }
