"""
model_registry.py
=================
模型注册表 + 优先级/权重计算器。

ModelRegistry：
    - 管理多个云端模型的连接信息（endpoint / api_key / model_name / provider）
    - 自动通过 PricingFetcher 拉取定价（4 种来源）
    - 缓存定价与计算后的优先级

PriorityCalculator：
    - 根据 定价 + 设备上下文 + 用户策略 生成每个模型的：
        - priority（整数，越小越优先）
        - weight（0~1，加权选择时用）
    - 策略：
        - cost_first     : 优先最便宜
        - quality_first  : 优先质量（按模型规格排序，大模型优先）
        - balanced       : 成本与质量平衡
        - latency_first  : 优先低延迟（按 rtt 排序）
        - manual         : 完全用户配置，不自动计算

设计：
    用户只需在 YAML 里写：
        models:
          - name: gpt-4o
            provider: openai
            endpoint: https://api.openai.com/v1
            api_key: ${OPENAI_API_KEY}
            pricing_source: models_dev   # 或 official / custom_api / manual
            manual_pricing: {input_per_1k: 0.005, output_per_1k: 0.015}
            enabled: true
    系统自动：拉定价 -> 算权重 -> 按权重选模型 -> 接力执行
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .pricing import PricingFetcher, PricingInfo


# --------------------------------------------------------------------- #
#  单个模型条目
# --------------------------------------------------------------------- #
@dataclass
class ModelEntry:
    """一个云端模型的连接与定价信息。"""

    name: str                                   # 用户自定义名（如 "gpt-4o"）
    provider: str = ""                          # 厂商（openai/anthropic/...）
    model_id: str = ""                          # 模型 ID（如 "openai/gpt-4o"）
    endpoint: str = ""                          # API endpoint
    api_key: str = ""                           # API key（已展开环境变量）
    enabled: bool = True
    # 定价来源配置
    pricing_source: str = "models_dev"          # models_dev/official/custom_api/manual
    custom_api_url: Optional[str] = None
    custom_api_headers: Optional[Dict[str, str]] = None
    manual_pricing: Optional[Dict[str, float]] = None
    # 模型规格（用于 quality 排序，可选）
    context_window: int = 0
    quality_tier: int = 0                       # 0=未指定, 1=小, 2=中, 3=大, 4=旗舰
    # API 协议（auto/​openai_chat/openai_responses/anthropic_messages/gemini_generate）
    protocol: str = "auto"
    # 计算结果（由 PriorityCalculator 填充）
    pricing: Optional[PricingInfo] = None
    priority: int = 100                         # 数字越小越优先
    weight: float = 0.0                         # 0~1，加权选择
    # 运行时统计
    last_latency_ms: float = 0.0
    last_error: Optional[str] = None
    call_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "enabled": self.enabled,
            "pricing_source": self.pricing_source,
            "manual_pricing": self.manual_pricing,
            "context_window": self.context_window,
            "quality_tier": self.quality_tier,
            "protocol": self.protocol,
            "pricing": self.pricing.to_dict() if self.pricing else None,
            "priority": self.priority,
            "weight": self.weight,
            "last_latency_ms": self.last_latency_ms,
            "call_count": self.call_count,
            # 出于安全考虑不输出 api_key
        }


# --------------------------------------------------------------------- #
#  策略
# --------------------------------------------------------------------- #
STRATEGIES = {"cost_first", "quality_first", "balanced", "latency_first", "manual"}


# --------------------------------------------------------------------- #
#  模型注册表
# --------------------------------------------------------------------- #
class ModelRegistry:
    """
    管理多个云端模型条目，自动拉取定价。

    用法：
        reg = ModelRegistry()
        reg.add(ModelEntry(name="gpt-4o", model_id="openai/gpt-4o",
                          endpoint="...", api_key="...",
                          pricing_source="models_dev"))
        reg.fetch_all_pricing()
        calc = PriorityCalculator(strategy="cost_first")
        calc.compute(reg, device_ctx)
        # 然后按 reg.entries 的 priority/weight 选模型
    """

    def __init__(self, pricing_fetcher: Optional[PricingFetcher] = None) -> None:
        self.entries: List[ModelEntry] = []
        self.pf = pricing_fetcher or PricingFetcher()

    # ------------------------------------------------------------------ #
    #  CRUD
    # ------------------------------------------------------------------ #
    def add(self, entry: ModelEntry) -> None:
        if self.get(entry.name):
            raise ValueError(f"模型已存在: {entry.name}")
        self.entries.append(entry)

    def get(self, name: str) -> Optional[ModelEntry]:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def remove(self, name: str) -> bool:
        e = self.get(name)
        if e:
            self.entries.remove(e)
            return True
        return False

    def list_enabled(self) -> List[ModelEntry]:
        return [e for e in self.entries if e.enabled]

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ #
    #  批量拉定价
    # ------------------------------------------------------------------ #
    def fetch_all_pricing(self, verbose: bool = False) -> Dict[str, str]:
        """
        为所有 enabled 模型拉取定价。

        返回 {model_name: "ok" / "fail: <reason>"}。
        """
        results: Dict[str, str] = {}
        for e in self.list_enabled():
            try:
                info = self.pf.fetch(
                    model_id=e.model_id or e.name,
                    source=e.pricing_source,
                    provider=e.provider or None,
                    custom_api_url=e.custom_api_url,
                    custom_api_headers=e.custom_api_headers,
                    manual=e.manual_pricing,
                    official_endpoint=e.endpoint,
                    api_key=e.api_key,
                )
                e.pricing = info
                results[e.name] = "ok"
                if verbose:
                    print(f"  [{e.name}] {info.source}: "
                          f"in=${info.input_per_1k:.6f}/1k "
                          f"out=${info.output_per_1k:.6f}/1k")
            except Exception as ex:
                e.last_error = f"{type(ex).__name__}: {ex}"
                results[e.name] = f"fail: {e.last_error}"
                if verbose:
                    print(f"  [{e.name}] FAIL: {e.last_error}")
        return results

    # ------------------------------------------------------------------ #
    #  便捷构造
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config_list(cls, models_cfg: List[Dict],
                         pricing_fetcher: Optional[PricingFetcher] = None) -> "ModelRegistry":
        """从配置文件中的 models 列表构造注册表。"""
        reg = cls(pricing_fetcher=pricing_fetcher)
        for cfg in models_cfg:
            entry = _build_entry_from_cfg(cfg)
            reg.add(entry)
        return reg


def _expand_env(value: str) -> str:
    """展开 ${VAR} 形式的环境变量。"""
    if not isinstance(value, str):
        return value
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)


def _build_entry_from_cfg(cfg: Dict) -> ModelEntry:
    """从单个 model 配置字典构造 ModelEntry。"""
    return ModelEntry(
        name=cfg["name"],
        provider=cfg.get("provider", ""),
        model_id=cfg.get("model_id", cfg.get("name", "")),
        endpoint=_expand_env(cfg.get("endpoint", "")),
        api_key=_expand_env(cfg.get("api_key", "")),
        enabled=cfg.get("enabled", True),
        pricing_source=cfg.get("pricing_source", "models_dev"),
        custom_api_url=cfg.get("custom_api_url"),
        custom_api_headers=cfg.get("custom_api_headers"),
        manual_pricing=cfg.get("manual_pricing"),
        context_window=cfg.get("context_window", 0),
        quality_tier=cfg.get("quality_tier", 0),
        protocol=cfg.get("protocol", "auto"),
        # 用户手动配置的优先级/权重（manual 策略时使用）
        priority=cfg.get("priority", 100),
        weight=cfg.get("weight", 0.0),
    )


# --------------------------------------------------------------------- #
#  优先级与权重计算器
# --------------------------------------------------------------------- #
class PriorityCalculator:
    """
    根据定价 + 策略 + 设备上下文，为注册表中的每个模型生成 priority/weight。

    策略:
        cost_first     : 按 input+output 价格升序，最便宜的 priority=1
        quality_first  : 按 quality_tier 降序，旗舰 priority=1
        balanced       : cost 与 quality 加权（0.5/0.5）
        latency_first  : 按 endpoint 主机 RTT 升序（需先填 last_latency_ms）
        manual         : 完全用 ModelEntry.priority/weight，不自动计算
    """

    def __init__(self, strategy: str = "cost_first",
                 cost_weight: float = 0.5,
                 quality_weight: float = 0.5,
                 rtt_weight: float = 1.0,
                 zero_price_fallback: float = 0.01) -> None:
        """
        参数:
            cost_weight / quality_weight : balanced 策略下的混合权重
            zero_price_fallback          : models.dev 返回 0 时的兜底价（USD/1k）
                                           避免免费模型被无限优先
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略: {strategy}, 可选: {STRATEGIES}")
        self.strategy = strategy
        self.cost_weight = cost_weight
        self.quality_weight = quality_weight
        self.rtt_weight = rtt_weight
        self.zero_price_fallback = zero_price_fallback

    # ------------------------------------------------------------------ #
    #  主计算入口
    # ------------------------------------------------------------------ #
    def compute(self, registry: ModelRegistry, device_ctx=None) -> None:
        """就地填充每个 ModelEntry 的 priority / weight。"""
        entries = registry.list_enabled()
        if not entries:
            return

        if self.strategy == "manual":
            # 用户已填，仅归一化 weight
            total = sum(e.weight for e in entries) or 1.0
            for e in entries:
                e.weight = e.weight / total
            return

        # 计算每个模型的得分（越小越优先）
        scored: List[Tuple[ModelEntry, float]] = []
        for e in entries:
            score = self._score(e, device_ctx)
            scored.append((e, score))

        # 按 score 升序排
        scored.sort(key=lambda x: x[1])

        # priority 从 1 开始递增
        for i, (e, _) in enumerate(scored):
            e.priority = i + 1

        # weight 用 softmax(score) 的反向（score 越小权重越大）
        # 这里用简单的 1/score 归一化
        inv_scores = [1.0 / max(s, 1e-6) for _, s in scored]
        total_inv = sum(inv_scores)
        for (e, _), inv in zip(scored, inv_scores):
            e.weight = inv / total_inv

    # ------------------------------------------------------------------ #
    #  打分
    # ------------------------------------------------------------------ #
    def _score(self, entry: ModelEntry, device_ctx) -> float:
        """返回得分（越小越优先）。"""
        if self.strategy == "cost_first":
            return self._cost_score(entry)
        if self.strategy == "quality_first":
            return -float(entry.quality_tier)  # 取负，越大质量越优先 -> 越小得分
        if self.strategy == "latency_first":
            return max(entry.last_latency_ms, 1.0)
        if self.strategy == "balanced":
            cost = self._cost_score(entry)
            qual = -float(entry.quality_tier) * 10.0  # 放大到与 cost 同量级
            return self.cost_weight * cost + self.quality_weight * qual
        return 100.0

    def _cost_score(self, entry: ModelEntry) -> float:
        """成本得分：input + output 单价之和。"""
        if entry.pricing is None:
            return 1.0  # 无定价信息，给一个中性分
        inp = entry.pricing.input_per_1k
        out = entry.pricing.output_per_1k
        # 0 值兜底（models.dev 未填的情况）
        if inp + out == 0:
            return self.zero_price_fallback * 2
        return inp + out

    # ------------------------------------------------------------------ #
    #  选择单个模型（按权重随机或按 priority 严格）
    # ------------------------------------------------------------------ #
    def pick(
        self,
        registry: ModelRegistry,
        deterministic: bool = False,
        rng_seed: Optional[int] = None,
    ) -> Optional[ModelEntry]:
        """
        从注册表中选一个模型。

        deterministic=True : 返回 priority 最小的（最优先）
        deterministic=False: 按权重加权随机
        """
        entries = registry.list_enabled()
        if not entries:
            return None

        if deterministic:
            return min(entries, key=lambda e: e.priority)

        # 加权随机
        import random as _r
        rng = _r.Random(rng_seed) if rng_seed is not None else _r
        weights = [e.weight for e in entries]
        total = sum(weights)
        if total <= 0:
            return rng.choice(entries)
        weights = [w / total for w in weights]
        return rng.choices(entries, weights=weights, k=1)[0]
