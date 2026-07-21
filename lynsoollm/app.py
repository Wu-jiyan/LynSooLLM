"""
app.py
======
灵枢 LynSooLLM 成品入口。

一行接入：用户只需写 YAML 配置文件，然后：

    from lynsoollm.app import LynSooApp
    app = LynSooApp.from_config("config.yaml")
    for ev in app.stream("讲个笑话"):
        if ev.kind == "token": print(ev.token, end="", flush=True)
        elif ev.kind == "exit": print(f"\\n[exit->{ev.payload['model']}]")

内部完成：
    1. 解析 YAML（模型连接信息 / 路由策略 / 设备上下文）
    2. PricingFetcher 拉取每个模型的定价（4 源）
    3. PriorityCalculator 算 priority / weight
    4. 加载 RealRouterModel（Gemma / Qwen）+ multi-LoRA
    5. EnvAwareSelector 根据设备上下文选 adapter
    6. CloudRelayPool 按策略选模型并执行接力
    7. 完整"硬路由 + 推测式接力"流程
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

# 离线模式（已下载本地权重）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .adapter_selector import EnvAwareSelector, builtin_rules
from .cloud_relay_pool import CloudExecutorFn, CloudRelayPool, MockCloudExecutor
from .config import AppConfig, build_app, load_config
from .exit_signal import EarlyExitSignal, RelayContext
from .model_registry import ModelRegistry, PriorityCalculator
from .pricing import PricingFetcher
from .real_router_engine import LocalGenerator, RealRouterEngine, RealRouterEvent
from .real_router_model import RealRouterModel, ROUTE_CLOUD, ROUTE_LOCAL
from .router import DeviceContext, RouterConfig


# --------------------------------------------------------------------- #
#  应用封装
# --------------------------------------------------------------------- #
class LynSooApp:
    """
    成品应用类：从配置到运行的完整封装。

    用法:
        app = LynSooApp.from_config("config.yaml")
        # 或者
        app = LynSooApp.from_config("config.yaml",
                                     local_generator=my_local_gen,
                                     executor=my_http_executor)

        for ev in app.stream("讲个笑话"):
            ...
    """

    def __init__(
        self,
        config: AppConfig,
        local_generator: Optional[LocalGenerator] = None,
        executor: Optional[CloudExecutorFn] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = device

        # 构建对象图（若尚未构建）
        if config.registry is None:
            build_app(config, fetch_pricing=True, verbose=True)

        self.registry: ModelRegistry = config.registry  # type: ignore[assignment]
        self.calculator: PriorityCalculator = config.calculator  # type: ignore[assignment]
        self.device_ctx: DeviceContext = config.device_ctx  # type: ignore[assignment]
        self.router_config: RouterConfig = config.router_config  # type: ignore[assignment]

        # 接力池（默认用 AutoExecutor，根据每个 entry 的 protocol 字段自动选执行器）
        self.executor = executor or AutoExecutor()
        self.relay_pool = CloudRelayPool(
            registry=self.registry,
            calculator=self.calculator,
            executor=self.executor,
        )

        # 真实路由模型
        backbone = config.router.get("backbone", "gemma")
        dtype = torch.bfloat16 if backbone == "qwen" else torch.float32
        self.engine = RealRouterEngine(
            backbone_name=backbone,
            local_generator=local_generator,
            cloud_relay=None,  # 我们用 relay_pool 替代
            config=self.router_config,
            device_ctx=self.device_ctx,
            dtype=dtype,
            device=device,
        )

        # 注册配置里声明的 LoRA adapter
        self._load_lora_adapters(config.router.get("lora_adapters", []))

        # 重新计算一次 priority/weight（确保 device_ctx 已就绪）
        self.calculator.compute(self.registry, self.device_ctx)

    # ------------------------------------------------------------------ #
    #  工厂方法
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        config_path: str,
        local_generator: Optional[LocalGenerator] = None,
        executor: Optional[CloudExecutorFn] = None,
        device: str = "cpu",
        fetch_pricing: bool = True,
        verbose: bool = True,
    ) -> "LynSooApp":
        """从 YAML 配置文件构造应用。"""
        cfg = load_config(config_path)
        build_app(cfg, fetch_pricing=fetch_pricing, verbose=verbose)
        return cls(cfg, local_generator=local_generator, executor=executor, device=device)

    # ------------------------------------------------------------------ #
    #  LoRA adapter 加载
    # ------------------------------------------------------------------ #
    def _load_lora_adapters(self, adapters: List[Dict]) -> None:
        """加载配置里声明的 LoRA adapter。"""
        if not adapters or self.engine.router_model.lora_mgr is None:
            return
        for a in adapters:
            name = a.get("name")
            path = a.get("path")
            if not name:
                continue
            try:
                if path and Path(path).exists():
                    # 完整路径（含 adapter_config.json）
                    adapter_path = path
                    if not Path(path, "adapter_config.json").exists():
                        # train_router 保存结构：path/name/adapter_config.json
                        adapter_path = str(Path(path) / name)
                    self.engine.router_model.lora_mgr.load_adapter(name, adapter_path)
                    # 同时加载 route_head（若存在）
                    rh = Path(path) / "route_head.pt"
                    if rh.exists():
                        self.engine.router_model.route_head.load_state_dict(
                            torch.load(rh, weights_only=True)
                        )
                else:
                    # 新建空 adapter（用户后续可训练）
                    self.engine.router_model.lora_mgr.load_adapter(name, path=None)
            except Exception as ex:
                print(f"  [warn] 加载 adapter {name} 失败: {ex}")

    # ------------------------------------------------------------------ #
    #  主流程：流式生成 + 接力
    # ------------------------------------------------------------------ #
    def stream(
        self,
        prompt: str,
        relay_strategy: str = "priority",
        prefer_model: Optional[str] = None,
    ) -> Iterator[RealRouterEvent]:
        """
        流式生成主入口。

        与 RealRouterEngine.stream 兼容，但 exit 事件会携带选中的模型信息。
        """
        # 重置统计
        self.engine.last_entropy_trace = []
        self.engine.last_threshold_trace = []
        self.engine.last_exit_signal = None

        import time
        start_ts = time.perf_counter()

        # 1) 硬路由决策
        route_out = self.engine.decide(prompt)
        yield RealRouterEvent(
            kind="decide",
            step=-1,
            entropy=route_out.entropy,
            route_label=route_out.route_label,
            route_prob=route_out.route_prob.tolist()
                        if route_out.route_prob.numel() <= 2
                        else route_out.route_prob[0].tolist(),
            adapter=self.engine.last_adapter,
            payload={
                "route_ms": self.engine.last_route_ms,
                "models": [e.name for e in self.registry.list_enabled()],
                "strategy": self.calculator.strategy,
            },
        )

        # 离线强制本地
        route_label = ROUTE_LOCAL if self.device_ctx.offline else route_out.route_label

        # 2) 硬路由 -> 上云
        if route_label == ROUTE_CLOUD and not self.device_ctx.offline:
            ctx = RelayContext(
                prompt=prompt,
                generated_tokens=[],
                generated_text="",
                exit_step=-1,
                exit_entropy=route_out.entropy,
                reason="hard_route_cloud",
            )
            result = self.relay_pool.handoff(
                ctx, strategy=relay_strategy, prefer_model=prefer_model
            )
            self.engine.last_exit_signal = EarlyExitSignal(
                triggered=True, exit_step=-1,
                exit_entropy=route_out.entropy,
                reason="hard_route_cloud", relay_context=ctx,
            )
            yield RealRouterEvent(
                kind="exit",
                step=-1,
                entropy=route_out.entropy,
                route_label=ROUTE_CLOUD,
                adapter=self.engine.last_adapter,
                payload={
                    "cloud_text": result.text if result.success else "",
                    "model": result.model_name,
                    "latency_ms": result.latency_ms,
                    "fallback_used": result.fallback_used,
                    "fallback_from": result.fallback_from,
                    "success": result.success,
                    "error": result.error,
                    "hard_route": True,
                },
            )
            total_ms = (time.perf_counter() - start_ts) * 1000.0
            yield RealRouterEvent(
                kind="done", step=-1, entropy=route_out.entropy,
                payload={
                    "ttft_ms": self.engine.last_route_ms,
                    "total_ms": total_ms,
                    "exited": True, "hard_route": True,
                    "cloud_model": result.model_name,
                },
            )
            return

        # 3) 本地流式 + 推测式接力
        if self.engine.local_generator is None:
            total_ms = (time.perf_counter() - start_ts) * 1000.0
            yield RealRouterEvent(
                kind="done", step=-1, entropy=route_out.entropy,
                payload={
                    "ttft_ms": self.engine.last_route_ms,
                    "total_ms": total_ms,
                    "exited": False, "reason": "no_local_generator",
                },
            )
            return

        threshold = self.engine._resolve_threshold(base_entropy=route_out.entropy)
        generated_tokens: List[str] = []
        first_token_ts = None
        exit_step = -1
        exit_entropy = 0.0

        from .entropy import compute_entropy, normalized_entropy

        for step, (token, logits) in enumerate(
            self.engine.local_generator.stream(
                prompt, max_new_tokens=self.engine.config.max_new_tokens
            )
        ):
            if first_token_ts is None:
                first_token_ts = time.perf_counter()
                self.engine.last_ttft_ms = (first_token_ts - start_ts) * 1000.0

            if self.engine.config.use_normalized:
                ent = float(normalized_entropy(logits.unsqueeze(0)).item())
            else:
                ent = float(compute_entropy(logits.unsqueeze(0)).item())
            self.engine.last_entropy_trace.append(ent)
            self.engine.last_threshold_trace.append(threshold)

            should_exit = (
                ent > threshold
                and step >= self.engine.config.min_tokens_before_exit
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
                )
                result = self.relay_pool.handoff(
                    ctx, strategy=relay_strategy, prefer_model=prefer_model
                )
                self.engine.last_exit_signal = EarlyExitSignal(
                    triggered=True, exit_step=step, exit_entropy=ent,
                    reason="entropy_threshold_exceeded", relay_context=ctx,
                )
                yield RealRouterEvent(
                    kind="exit", step=step, token="",
                    entropy=ent, threshold=threshold,
                    route_label=route_out.route_label,
                    adapter=self.engine.last_adapter,
                    payload={
                        "cloud_text": result.text if result.success else "",
                        "model": result.model_name,
                        "latency_ms": result.latency_ms,
                        "fallback_used": result.fallback_used,
                        "success": result.success,
                        "error": result.error,
                        "hard_route": False,
                    },
                )
                break

            generated_tokens.append(token)
            yield RealRouterEvent(
                kind="token", step=step, token=token,
                entropy=ent, threshold=threshold,
                route_label=route_out.route_label,
                adapter=self.engine.last_adapter,
            )

        total_ms = (time.perf_counter() - start_ts) * 1000.0
        self.engine.last_total_ms = total_ms
        yield RealRouterEvent(
            kind="done",
            step=exit_step if exit_step >= 0 else len(generated_tokens) - 1,
            entropy=exit_entropy, threshold=threshold,
            route_label=route_out.route_label,
            adapter=self.engine.last_adapter,
            payload={
                "ttft_ms": self.engine.last_ttft_ms,
                "total_ms": total_ms,
                "route_ms": self.engine.last_route_ms,
                "exited": exit_step >= 0,
                "hard_route": False,
            },
        )

    # ------------------------------------------------------------------ #
    #  非流式便捷接口
    # ------------------------------------------------------------------ #
    def route(self, prompt: str, **kw) -> Dict:
        """非流式：返回完整结果。"""
        local_parts: List[str] = []
        cloud_text = ""
        info: Dict[str, Any] = {}
        for ev in self.stream(prompt, **kw):
            if ev.kind == "token":
                local_parts.append(ev.token)
            elif ev.kind == "exit":
                info["exit_model"] = ev.payload.get("model")
                info["exit_latency_ms"] = ev.payload.get("latency_ms")
                info["hard_route"] = ev.payload.get("hard_route", False)
                cloud_text = ev.payload.get("cloud_text", "")
            elif ev.kind == "done":
                info["ttft_ms"] = ev.payload.get("ttft_ms")
                info["total_ms"] = ev.payload.get("total_ms")
                info["exited"] = ev.payload.get("exited", False)
            elif ev.kind == "decide":
                info["route_label"] = ev.route_label
                info["adapter"] = ev.adapter
                info["route_ms"] = ev.payload.get("route_ms")
        info["text"] = f"{''.join(local_parts)}{cloud_text}"
        return info

    # ------------------------------------------------------------------ #
    #  设备上下文 / 策略 动态切换
    # ------------------------------------------------------------------ #
    def update_device(self, **kwargs) -> None:
        """运行时更新设备上下文，会触发 priority 重算。"""
        for k, v in kwargs.items():
            if hasattr(self.device_ctx, k):
                setattr(self.device_ctx, k, v)
        self.calculator.compute(self.registry, self.device_ctx)

    def switch_strategy(self, strategy: str) -> None:
        """切换路由策略。"""
        self.calculator.strategy = strategy
        self.calculator.compute(self.registry, self.device_ctx)

    # ------------------------------------------------------------------ #
    #  诊断
    # ------------------------------------------------------------------ #
    def info(self) -> Dict:
        return {
            "backbone": self.engine.router_model.backbone_name,
            "strategy": self.calculator.strategy,
            "device": self.device_ctx.__dict__,
            "models": [e.to_dict() for e in self.registry.list_enabled()],
            "lora_adapters": (
                self.engine.router_model.lora_mgr.list_adapters()
                if self.engine.router_model.lora_mgr else []
            ),
            "active_adapter": (
                self.engine.router_model.lora_mgr.active
                if self.engine.router_model.lora_mgr else None
            ),
        }
