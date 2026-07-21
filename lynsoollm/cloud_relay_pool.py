"""
cloud_relay_pool.py
===================
多云端模型接力池：根据 ModelRegistry + PriorityCalculator，
按 priority / weight 选择目标模型并执行接力。

执行器抽象：
    CloudExecutor 是一个可注入的 callable，签名 (entry, ctx) -> str。
    具体执行器实现在 executors.py，覆盖：
        - OpenAIChatExecutor        : /v1/chat/completions
        - OpenAIResponsesExecutor   : /v1/responses
        - AnthropicMessagesExecutor : /v1/messages
        - GeminiGenerateContentExecutor : :generateContent
        - AutoExecutor              : 自动选上面四种

接力策略：
    - "priority"  : 严格按 priority 升序，失败自动降级到下一个
    - "weighted"  : 按 weight 加权随机
    - "round_robin": 轮询（避免单点过载）
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .exit_signal import CloudRelay, RelayContext
from .executors import (
    AnthropicMessagesExecutor,
    AutoExecutor,
    GeminiGenerateContentExecutor,
    HTTPCloudExecutor,
    MockCloudExecutor,
    OpenAIChatExecutor,
    OpenAIResponsesExecutor,
)
from .model_registry import ModelEntry, ModelRegistry, PriorityCalculator


# --------------------------------------------------------------------- #
#  执行器抽象
# --------------------------------------------------------------------- #
CloudExecutorFn = Callable[[ModelEntry, RelayContext], str]


# --------------------------------------------------------------------- #
#  云端接力池
# --------------------------------------------------------------------- #
@dataclass
class RelayResult:
    """一次接力的结果。"""

    success: bool
    model_name: str = ""
    text: str = ""
    latency_ms: float = 0.0
    error: str = ""
    fallback_used: bool = False
    fallback_from: str = ""


class CloudRelayPool:
    """
    多模型接力池。

    用法：
        pool = CloudRelayPool(registry, calculator, executor=HTTPCloudExecutor())
        result = pool.handoff(relay_context, strategy="priority")
        if result.success:
            print(result.text)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        calculator: Optional[PriorityCalculator] = None,
        executor: Optional[CloudExecutorFn] = None,
        max_retries: int = 3,
    ) -> None:
        self.registry = registry
        self.calculator = calculator or PriorityCalculator(strategy="cost_first")
        self.executor: CloudExecutorFn = executor or MockCloudExecutor()
        self.max_retries = max_retries
        # 轮询游标（round_robin 用）
        self._rr_cursor = 0

    # ------------------------------------------------------------------ #
    #  接力主入口
    # ------------------------------------------------------------------ #
    def handoff(
        self,
        ctx: RelayContext,
        strategy: str = "priority",
        prefer_model: Optional[str] = None,
    ) -> RelayResult:
        """
        把 RelayContext 交给云端模型接力。

        参数:
            strategy     : "priority" / "weighted" / "round_robin"
            prefer_model : 优先尝试的模型名（若存在）
        """
        candidates = self._pick_candidates(strategy, prefer_model)
        if not candidates:
            return RelayResult(success=False, error="无可用模型")

        last_err = ""
        for idx, entry in enumerate(candidates):
            if idx >= self.max_retries:
                break
            t0 = time.perf_counter()
            try:
                text = self.executor(entry, ctx)
                latency = (time.perf_counter() - t0) * 1000.0
                entry.last_latency_ms = latency
                entry.call_count += 1
                entry.last_error = None
                return RelayResult(
                    success=True,
                    model_name=entry.name,
                    text=text,
                    latency_ms=latency,
                    fallback_used=idx > 0,
                    fallback_from=candidates[0].name if idx > 0 else "",
                )
            except Exception as ex:
                latency = (time.perf_counter() - t0) * 1000.0
                entry.last_latency_ms = latency
                entry.last_error = f"{type(ex).__name__}: {ex}"
                last_err = entry.last_error
                continue

        return RelayResult(success=False, error=last_err or "all candidates failed")

    # ------------------------------------------------------------------ #
    #  候选选择
    # ------------------------------------------------------------------ #
    def _pick_candidates(
        self, strategy: str, prefer_model: Optional[str]
    ) -> List[ModelEntry]:
        """根据策略返回有序候选列表（含 prefer_model 前置）。"""
        enabled = self.registry.list_enabled()
        if not enabled:
            return []

        # 优先 prefer_model
        if prefer_model:
            pref = [e for e in enabled if e.name == prefer_model]
            rest = [e for e in enabled if e.name != prefer_model]
            if strategy == "priority":
                rest.sort(key=lambda e: e.priority)
            elif strategy == "weighted":
                import random
                rest.sort(key=lambda e: e.weight, reverse=True)
            else:  # round_robin
                pass
            return pref + rest

        if strategy == "priority":
            return sorted(enabled, key=lambda e: e.priority)

        if strategy == "weighted":
            import random
            weights = [e.weight for e in enabled]
            total = sum(weights) or 1.0
            weights = [w / total for w in weights]
            # 加权采样所有候选（不重复）
            pool = list(enabled)
            ordered: List[ModelEntry] = []
            while pool:
                w = [e.weight for e in pool]
                t = sum(w) or 1.0
                w = [x / t for x in w]
                chosen = random.choices(pool, weights=w, k=1)[0]
                ordered.append(chosen)
                pool.remove(chosen)
            return ordered

        if strategy == "round_robin":
            ordered = enabled[self._rr_cursor:] + enabled[:self._rr_cursor]
            self._rr_cursor = (self._rr_cursor + 1) % max(len(enabled), 1)
            return ordered

        raise ValueError(f"未知 strategy: {strategy}")

    # ------------------------------------------------------------------ #
    #  诊断
    # ------------------------------------------------------------------ #
    def info(self) -> Dict:
        return {
            "models": [e.to_dict() for e in self.registry.list_enabled()],
            "executor": type(self.executor).__name__,
            "max_retries": self.max_retries,
        }
