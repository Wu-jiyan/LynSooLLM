"""
adapter_selector.py
===================
环境感知 LoRA adapter 选择器。

提供：
    - 一组"出厂预设 adapter 规则"（weak_net / low_battery / hot_device
      / cheap_cloud / expensive_cloud / personalized）
    - ``EnvAwareSelector`` 类：根据 DeviceContext 从预设里挑 adapter 名，
      并可调用 MultiLoRAManager 激活/合并

策略：
    1. 先按"硬规则"匹配（命中某极端环境即用对应 adapter）
    2. 多个规则同时命中时，按优先级合并（merge_weighted）
    3. 没有任何规则命中 -> 用 "default"
    4. 若有 "personalized" adapter，永远以小权重混入（千人千面）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .multi_lora import AdapterMeta, MultiLoRAManager
from .router import DeviceContext


# --------------------------------------------------------------------- #
#  预设规则
# --------------------------------------------------------------------- #
@dataclass
class AdapterRule:
    """一条环境->adapter 规则。"""

    name: str                                   # adapter 名（需已注册到 manager）
    description: str = ""
    # 命中条件：device 字段 -> (下限, 上限)，闭区间下限、开区间上限
    match: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    priority: int = 10                          # 数值越大越优先
    merge_alpha: float = 1.0                    # 若多规则同时命中，合并时的权重
    personalized_blend: float = 0.0             # 与 personalized 混合时的权重


def builtin_rules() -> List[AdapterRule]:
    """返回一组出厂预设规则。"""
    return [
        AdapterRule(
            name="weak_net",
            description="弱网（RTT > 500ms）: 更倾向本地",
            match={"network_rtt_ms": (500.0, float("inf"))},
            priority=20,
            merge_alpha=0.6,
        ),
        AdapterRule(
            name="low_battery",
            description="低电量（<30%）: 更倾向本地",
            match={"battery_pct": (0.0, 30.0)},
            priority=20,
            merge_alpha=0.6,
        ),
        AdapterRule(
            name="hot_device",
            description="高温（>40℃）: 避免云端往返加热",
            match={"temperature_c": (40.0, float("inf"))},
            priority=15,
            merge_alpha=0.4,
        ),
        AdapterRule(
            name="cheap_cloud",
            description="云端 API 便宜（<0.005/1k）: 更倾向上云",
            match={"cloud_price_per_1k": (0.0, 0.005)},
            priority=10,
            merge_alpha=0.5,
        ),
        AdapterRule(
            name="expensive_cloud",
            description="云端 API 贵（>0.02/1k）: 更倾向本地",
            match={"cloud_price_per_1k": (0.02, float("inf"))},
            priority=10,
            merge_alpha=0.5,
        ),
        AdapterRule(
            name="good_network",
            description="网络极好（RTT < 50ms）: 更倾向上云",
            match={"network_rtt_ms": (0.0, 50.0)},
            priority=5,
            merge_alpha=0.3,
        ),
    ]


# --------------------------------------------------------------------- #
#  环境感知选择器
# --------------------------------------------------------------------- #
class EnvAwareSelector:
    """
    根据 DeviceContext 从规则池里挑 adapter，并驱动 MultiLoRAManager。

    用法：
        mgr = MultiLoRAManager(base_model)
        for r in builtin_rules():
            mgr.load_adapter(r.name, path=...)   # 提前加载好
        sel = EnvAwareSelector(mgr)
        sel.apply(device_ctx)                     # 自动激活/合并
    """

    def __init__(
        self,
        manager: MultiLoRAManager,
        rules: Optional[List[AdapterRule]] = None,
        personalized_blend: float = 0.2,
    ) -> None:
        self.manager = manager
        self.rules = rules if rules is not None else builtin_rules()
        self.personalized_blend = personalized_blend
        self.last_decision: Dict = {}

    # ------------------------------------------------------------------ #
    #  规则匹配
    # ------------------------------------------------------------------ #
    def _match(self, rule: AdapterRule, ctx: DeviceContext) -> bool:
        field_map = {
            "network_rtt_ms": ctx.network_rtt_ms,
            "battery_pct": ctx.battery_pct,
            "temperature_c": ctx.temperature_c,
            "cloud_price_per_1k": ctx.cloud_price_per_1k,
        }
        for key, (lo, hi) in rule.match.items():
            val = field_map.get(key)
            if val is None:
                return False
            if not (lo <= val < hi):
                return False
        return True

    def decide(self, ctx: DeviceContext) -> Dict:
        """
        决策：返回 {mode, adapter(s), weights, reason}。

        mode:
            - "single"  : 单 adapter 激活
            - "merged"  : 多 adapter 加权合并
            - "default" : 无规则命中
        """
        if ctx.offline:
            return {
                "mode": "single",
                "adapters": ["default"],
                "weights": {},
                "reason": "offline -> 强制 default（本地）",
            }

        hits = [(r, r.priority) for r in self.rules if self._match(r, ctx)]
        # 过滤掉 manager 里未注册的
        registered = set(self.manager.list_adapters())
        hits = [(r, p) for r, p in hits if r.name in registered]
        hits.sort(key=lambda x: x[1], reverse=True)

        if not hits:
            return {
                "mode": "default",
                "adapters": ["default"],
                "weights": {},
                "reason": "no rule matched",
            }

        if len(hits) == 1:
            return {
                "mode": "single",
                "adapters": [hits[0][0].name],
                "weights": {hits[0][0].name: 1.0},
                "reason": f"matched: {hits[0][0].name}",
            }

        # 多规则命中 -> 加权合并
        weights = {r.name: r.merge_alpha for r, _ in hits}
        # 归一化
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

        # 混入 personalized（千人千面）
        if "personalized" in registered and self.personalized_blend > 0:
            p = self.personalized_blend
            weights = {k: v * (1 - p) for k, v in weights.items()}
            weights["personalized"] = weights.get("personalized", 0) + p

        return {
            "mode": "merged",
            "adapters": list(weights.keys()),
            "weights": weights,
            "reason": f"merged: {list(weights.keys())}",
        }

    # ------------------------------------------------------------------ #
    #  应用决策（激活或合并）
    # ------------------------------------------------------------------ #
    def apply(self, ctx: DeviceContext) -> str:
        """根据 ctx 决策并应用到 manager，返回最终活跃 adapter 名。"""
        decision = self.decide(ctx)
        self.last_decision = decision

        if decision["mode"] == "single" or decision["mode"] == "default":
            name = decision["adapters"][0]
            self.manager.activate(name)
            return name

        # merged
        merged_name = self.manager.merge_weighted(decision["weights"])
        return merged_name

    # ------------------------------------------------------------------ #
    #  添加新规则（端侧自进化时可用）
    # ------------------------------------------------------------------ #
    def add_rule(self, rule: AdapterRule) -> None:
        # 同名替换
        self.rules = [r for r in self.rules if r.name != rule.name]
        self.rules.append(rule)
