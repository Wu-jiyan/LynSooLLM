"""
multi_lora.py
=============
Multi-LoRA 管理器：支持在同一基底模型上挂载多个 LoRA adapter，
按"路由策略/环境上下文"动态切换或合并。

设计目标：
    1. 单进程内同时持有 N 个 adapter，切换零拷贝（PEFT 的
       ``set_adapter`` 机制，只换活跃 adapter 引用，不重新加载权重）。
    2. 支持"环境感知选择"：根据 DeviceContext（网络/电量/温度）
       从 adapter 池里挑出最合适的一个。
    3. 支持"加权合并"：把多个 adapter 按 alpha 线性合并为一组
       临时权重，用于 AB 过渡或个性化混合（千人千面）。
    4. 支持端侧在线追加：训练完一个新 adapter 后，热加载进池子，
       无需重启路由服务。

adapter 命名约定：
    - "default"      : 未微调基底（实际是个空 LoRA，alpha=0）
    - "weak_net"     : 弱网场景偏保守（更倾向本地）
    - "low_battery"  : 低电量场景偏保守
    - "personalized" : 端侧自进化产出的个性化 adapter
    - 任意用户自定义名
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer


# --------------------------------------------------------------------- #
#  adapter 元信息
# --------------------------------------------------------------------- #
@dataclass
class AdapterMeta:
    """单个 LoRA adapter 的元信息。"""

    name: str
    path: Optional[str] = None            # 已保存 adapter 的本地路径
    description: str = ""
    target_env: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # target_env 形如 {"network_rtt_ms": (500, inf), "battery_pct": (0, 30)}
    # 表示该 adapter 适用的环境区间（闭区间下限、开区间上限）
    alpha: float = 1.0                    # 合并时的权重
    trainable: bool = False               # 是否参与端侧在线训练


# --------------------------------------------------------------------- #
#  环境匹配打分
# --------------------------------------------------------------------- #
def _env_match_score(meta: AdapterMeta, device_ctx) -> float:
    """
    计算 adapter 与当前设备上下文的匹配度（越大越匹配）。

    若 adapter 没声明 target_env，给一个中性分 0.5；
    否则每命中一个区间 +1，未声明字段不参与打分。
    """
    if not meta.target_env:
        return 0.5

    score = 0.0
    hit = 0
    field_map = {
        "network_rtt_ms": device_ctx.network_rtt_ms,
        "battery_pct": device_ctx.battery_pct,
        "temperature_c": device_ctx.temperature_c,
        "cloud_price_per_1k": device_ctx.cloud_price_per_1k,
    }
    for key, (lo, hi) in meta.target_env.items():
        if key not in field_map:
            continue
        val = field_map[key]
        if lo <= val < hi:
            score += 1.0
        else:
            # 落在区间外，按距离衰减
            if val < lo:
                score += max(0.0, 1.0 - (lo - val) / max(lo, 1e-6))
            else:
                score += max(0.0, 1.0 - (val - hi) / max(hi, 1e-6))
        hit += 1
    return score / max(hit, 1)


# --------------------------------------------------------------------- #
#  Multi-LoRA 管理器
# --------------------------------------------------------------------- #
class MultiLoRAManager:
    """
    多 LoRA adapter 管理器。

    用法：
        mgr = MultiLoRAManager(base_model, tokenizer)
        mgr.load_adapter("weak_net", path="/path/to/weak_net_lora")
        mgr.load_adapter("low_battery", path="/path/to/low_battery_lora")
        mgr.activate("weak_net")               # 切换活跃 adapter
        mgr.select_by_device(device_ctx)        # 环境感知自动选

    合并模式：
        mgr.merge_weighted({"weak_net": 0.6, "low_battery": 0.4})
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        lora_cfg: Optional[LoraConfig] = None,
        default_adapter_name: str = "default",
    ) -> None:
        self.base_model = base_model
        self._default_lora_cfg = lora_cfg or LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._adapters: Dict[str, AdapterMeta] = {}
        self._active: str = default_adapter_name

        # 把基底模型包装成 PeftModel（先挂一个 default adapter）
        if not isinstance(base_model, PeftModel):
            self.peft_model = get_peft_model(base_model, self._default_lora_cfg)
            # PEFT 默认把第一个 adapter 命名为 "default"
            self._adapters[default_adapter_name] = AdapterMeta(
                name=default_adapter_name,
                description="基底默认 LoRA（未微调）",
            )
        else:
            self.peft_model = base_model
            # 复用已有 adapter 名
            for name in self.peft_model.peft_config.keys():
                self._adapters[name] = AdapterMeta(name=name)

    # ------------------------------------------------------------------ #
    #  adapter 生命周期
    # ------------------------------------------------------------------ #
    def load_adapter(self, name: str, path: Optional[str] = None,
                     meta: Optional[AdapterMeta] = None) -> None:
        """
        加载一个已训练好的 LoRA adapter 进池子。

        - 若提供 path：从磁盘加载（PEFT ``load_adapter``）。
        - 若未提供 path：新建一个可训练 adapter（用于端侧在线训练）。
        """
        if name in self._adapters:
            raise ValueError(f"adapter '{name}' 已存在")

        if path is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"adapter 路径不存在: {path}")
            self.peft_model.load_adapter(str(p), adapter_name=name)
        else:
            # 新建空 adapter（共享同样的 lora_cfg）
            self.peft_model.add_adapter(name, self._default_lora_cfg)

        self._adapters[name] = meta or AdapterMeta(name=name, path=path)

    def save_adapter(self, name: str, path: str) -> None:
        """把指定 adapter 保存到磁盘。"""
        if name not in self._adapters:
            raise KeyError(f"adapter '{name}' 不存在")
        os.makedirs(path, exist_ok=True)
        self.peft_model.save_pretrained(path, selected_adapters=[name])
        self._adapters[name].path = path

    def remove_adapter(self, name: str) -> None:
        if name == self._active:
            raise ValueError(f"不能删除当前活跃 adapter: {name}")
        if name not in self._adapters:
            raise KeyError(f"adapter '{name}' 不存在")
        self.peft_model.delete_adapter(name)
        del self._adapters[name]

    # ------------------------------------------------------------------ #
    #  活跃 adapter 切换
    # ------------------------------------------------------------------ #
    def activate(self, name: str) -> None:
        """切换活跃 adapter（零拷贝，仅切换引用）。"""
        if name not in self._adapters:
            raise KeyError(f"adapter '{name}' 不存在，已注册: {list(self._adapters)}")
        self.peft_model.set_adapter(name)
        self._active = name

    @property
    def active(self) -> str:
        return self._active

    def list_adapters(self) -> List[str]:
        return list(self._adapters.keys())

    # ------------------------------------------------------------------ #
    #  环境感知选择
    # ------------------------------------------------------------------ #
    def select_by_device(self, device_ctx, topk: int = 1) -> List[str]:
        """
        根据设备上下文挑选最匹配的 adapter，返回 top-k 名单（按分数降序）。
        并自动激活第 1 名。
        """
        scored = [
            (name, _env_match_score(meta, device_ctx))
            for name, meta in self._adapters.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        topk = max(1, topk)
        winners = [n for n, _ in scored[:topk]]
        if winners:
            self.activate(winners[0])
        return winners

    def best_score(self, device_ctx) -> float:
        """返回当前最优 adapter 的匹配分数（不切换）。"""
        if not self._adapters:
            return 0.0
        return max(
            _env_match_score(m, device_ctx) for m in self._adapters.values()
        )

    # ------------------------------------------------------------------ #
    #  加权合并（多 adapter 融合，千人千面）
    # ------------------------------------------------------------------ #
    def merge_weighted(self, weights: Dict[str, float]) -> str:
        """
        把多个 adapter 按权重线性合并为一个新 adapter。

        实现：对每个 LoRA 层，新 A = Σ w_i * A_i，新 B = Σ w_i * B_i。
        返回合并后的 adapter 名（自动激活）。

        注意：合并产物是新建 adapter，原 adapter 不变。
        """
        for name in weights:
            if name not in self._adapters:
                raise KeyError(f"adapter '{name}' 不存在")

        merged_name = "merged_" + "_".join(
            f"{n}{int(w*100)}" for n, w in weights.items()
        )[:48]
        if merged_name in self._adapters:
            self.activate(merged_name)
            return merged_name

        # 新建空 adapter 作为合并目标
        self.peft_model.add_adapter(merged_name, self._default_lora_cfg)

        # 遍历所有 LoRA 层做加权求和
        total_w = sum(weights.values()) or 1.0
        for name, w in weights.items():
            self._add_weighted_layer(merged_name, name, w / total_w)

        self._adapters[merged_name] = AdapterMeta(
            name=merged_name,
            description=f"weighted merge of {weights}",
        )
        self.activate(merged_name)
        return merged_name

    def _add_weighted_layer(self, dst: str, src: str, alpha: float) -> None:
        """把 src adapter 的 LoRA 权重按 alpha 累加到 dst。"""
        for dst_mod in self.peft_model.modules():
            if not isinstance(dst_mod, LoraLayer):
                continue
            # 取出 src 与 dst 的 A/B
            try:
                src_A = self._get_lora_weight(src_mod=dst_mod, name=src, which="A")
                src_B = self._get_lora_weight(src_mod=dst_mod, name=src, which="B")
                dst_A = self._get_lora_weight(src_mod=dst_mod, name=dst, which="A")
                dst_B = self._get_lora_weight(src_mod=dst_mod, name=dst, which="B")
            except (AttributeError, KeyError):
                continue
            if src_A is None or src_B is None:
                continue
            with torch.no_grad():
                dst_A.add_(src_A * alpha)
                dst_B.add_(src_B * alpha)

    @staticmethod
    def _get_lora_weight(src_mod, name: str, which: str):
        """
        从 PEFT 模块里按 adapter 名取 LoRA A/B 权重。
        PEFT 的 LoraLayer 把每个 adapter 的 A/B 存在
        ``lora_A[name].weight`` / ``lora_B[name].weight``，
        其中 lora_A / lora_B 是 ModuleDict。
        """
        container = getattr(src_mod, f"lora_{which}", None)
        if container is None:
            return None
        # ModuleDict 支持按 key 取子模块
        layer = None
        try:
            layer = container[name]
        except (KeyError, TypeError):
            # 某些 PEFT 版本用 .modules 属性（不是方法）
            modules_attr = getattr(container, "_modules", None)
            if isinstance(modules_attr, dict) and name in modules_attr:
                layer = modules_attr[name]
        if layer is None:
            return None
        return getattr(layer, "weight", None)

    # ------------------------------------------------------------------ #
    #  在线训练接口（端侧自进化）
    # ------------------------------------------------------------------ #
    def trainable_adapter(self, name: str) -> torch.nn.Module:
        """把指定 adapter 设为可训练，其余冻结。返回 peft_model。"""
        if name not in self._adapters:
            raise KeyError(f"adapter '{name}' 不存在")
        self.activate(name)
        # 仅当前活跃 adapter 可训练
        for n, p in self.peft_model.named_parameters():
            p.requires_grad = (n.startswith(f"lora_{name}") or
                               (self._active == name and "lora_" in n))
        return self.peft_model

    # ------------------------------------------------------------------ #
    #  诊断
    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, object]:
        return {
            "active": self._active,
            "adapters": [
                {
                    "name": m.name,
                    "path": m.path,
                    "description": m.description,
                    "target_env": {k: list(v) for k, v in m.target_env.items()},
                    "alpha": m.alpha,
                }
                for m in self._adapters.values()
            ],
        }
