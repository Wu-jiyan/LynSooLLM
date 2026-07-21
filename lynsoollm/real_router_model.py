"""
real_router_model.py
====================
真实路由模型封装：基于 Gemma-3-270M / Qwen3.5-0.8B 的基底，
叠加一个路由分类头（RouteHead），输出两类 logits：

    1. route_logits  : [local, cloud] 二分类，决定硬路由去向
    2. entropy_proxy : 用基底 LM head 在 prompt 上的最后一步
                       logits 计算归一化熵，作为"是否触发
                       Early-Exit"的看门狗信号

设计要点：
    - 用户在初始化时选择基底（"gemma" / "qwen" / 自定义路径）
    - 基底冻结，只训分类头 + LoRA（参数量极少，端侧可训）
    - 同时输出 route_logits 和 entropy，供 SpeculativeRouter
      做"硬路由 + 推测式接力"双决策
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# 离线模式（已下载本地权重）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer

from .entropy import compute_entropy, normalized_entropy
from .multi_lora import MultiLoRAManager


# --------------------------------------------------------------------- #
#  默认模型路径与可选清单
# --------------------------------------------------------------------- #
DEFAULT_MODELS = {
    "gemma": "/workspace/models/gemma-3-270m",
    "qwen":  "/workspace/models/Qwen3.5-0.8B",
}

# 路由标签
ROUTE_LOCAL = 0
ROUTE_CLOUD = 1
ROUTE_LABELS = ["local", "cloud"]


# --------------------------------------------------------------------- #
#  路由分类头
# --------------------------------------------------------------------- #
class RouteHead(nn.Module):
    """
    轻量路由分类头：hidden -> [local, cloud]。

    含一层 LayerNorm + Dropout + Linear，参数量约 hidden*2 + hidden。
    对于 Gemma-270M (hidden=640) 约 0.4M；Qwen-0.8B (hidden=1024) 约 1M。
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, H) -> 取最后一步 (B, H)
        h = hidden[:, -1, :]
        h = self.norm(h)
        h = self.dropout(h)
        return self.classifier(h)  # (B, 2)


# --------------------------------------------------------------------- #
#  真实路由模型
# --------------------------------------------------------------------- #
@dataclass
class RouterOutput:
    """路由模型输出。"""

    route_logits: torch.Tensor         # (B, 2) 硬路由分类
    route_label: int                   # argmax
    route_prob: torch.Tensor           # softmax 概率
    entropy: float                     # 基底 LM 在 prompt 末尾的归一化熵
    raw_entropy: float                 # 未归一化熵（nats）
    hidden: Optional[torch.Tensor] = None
    base_logits: Optional[torch.Tensor] = None


class RealRouterModel(nn.Module):
    """
    真实路由模型。

    参数:
        backbone_name : "gemma" / "qwen" / 自定义本地路径
        dtype         : torch.float32 / bfloat16
        device        : "cpu" / "cuda"
        use_lora      : 是否启用 multi-LoRA 管理（默认 True）
        lora_cfg      : 自定义 LoraConfig，否则用默认 r=8

    用法:
        router_model = RealRouterModel(backbone_name="gemma")
        out = router_model.route("帮我写一首诗")
        if out.route_label == ROUTE_CLOUD or out.entropy > 0.7:
            ... # 上云 / 触发 Early-Exit
    """

    def __init__(
        self,
        backbone_name: str = "gemma",
        dtype: Optional[torch.dtype] = None,
        device: str = "cpu",
        use_lora: bool = True,
        lora_cfg=None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        # 解析 backbone 路径
        backbone_path = DEFAULT_MODELS.get(backbone_name, backbone_name)
        if not Path(backbone_path).exists():
            raise FileNotFoundError(
                f"backbone 路径不存在: {backbone_path}。"
                f"内置可选: {list(DEFAULT_MODELS)}"
            )

        self.backbone_name = backbone_name
        self.backbone_path = backbone_path
        self.device = device

        # dtype: Qwen 原生 bfloat16，Gemma float32（270M 内存可控）
        if dtype is None:
            dtype = torch.bfloat16 if "qwen" in backbone_name.lower() else torch.float32
        self.dtype = dtype

        # 加载基底
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_path)
        # Qwen3.5 用专门的 CausalLM 类避免加载 vision 部分
        if "qwen3_5" in (self.tokenizer.name_or_path or "").lower() or \
           "qwen3.5" in backbone_name.lower():
            from transformers import Qwen3_5ForCausalLM
            base = Qwen3_5ForCausalLM.from_pretrained(backbone_path, dtype=dtype)
        else:
            base = AutoModelForCausalLM.from_pretrained(backbone_path, dtype=dtype)

        base.eval()
        for p in base.parameters():
            p.requires_grad = False

        # 取 hidden_size
        hidden_size = base.config.hidden_size if hasattr(base.config, "hidden_size") \
            else base.config.text_config.hidden_size
        self.hidden_size = hidden_size

        # 路由头
        self.route_head = RouteHead(hidden_size)

        # multi-LoRA
        self.use_lora = use_lora
        self.lora_mgr: Optional[MultiLoRAManager] = None
        if use_lora:
            self.lora_mgr = MultiLoRAManager(base, lora_cfg=lora_cfg)
            self.base = self.lora_mgr.peft_model  # 替换为 PeftModel
        else:
            self.base = base

        self.to(device)

    # ------------------------------------------------------------------ #
    #  前向：同时产出路由 logits 与熵
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> RouterOutput:
        """
        前向计算。

        返回:
            - route_logits : 路由二分类 logits
            - entropy      : 基底 LM 在 prompt 末位置的归一化熵（0~1）
        """
        outs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        # hidden_states: tuple, 最后一层 (B, T, H)
        last_hidden = outs.hidden_states[-1]
        # LM head logits（用于熵计算）
        lm_logits = outs.logits if hasattr(outs, "logits") else outs[0]
        # 取最后非 padding 位置
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            idx = lengths.long()
            last_hidden_for_route = last_hidden[torch.arange(last_hidden.size(0)), idx]
            lm_logits_last = lm_logits[torch.arange(lm_logits.size(0)), idx]
        else:
            last_hidden_for_route = last_hidden[:, -1, :]
            lm_logits_last = lm_logits[:, -1, :]

        # 路由头（RouteHead 内部会再取 [:, -1]，这里直接传完整序列）
        route_logits = self.route_head(last_hidden)  # (B, 2)
        route_prob = F.softmax(route_logits, dim=-1)
        # route_label: 单条时取标量；批量时取第 0 条作为代表
        if route_prob.numel() == 2:
            route_label = int(route_prob.argmax(dim=-1).item())
        else:
            route_label = int(route_prob[0].argmax(dim=-1).item())

        # 熵：用 LM logits 的最后一步（批量时取第 0 条作为代表）
        if lm_logits_last.dim() == 1:
            ent_input = lm_logits_last.unsqueeze(0)
        else:
            ent_input = lm_logits_last[0:1]
        ent_tensor = normalized_entropy(ent_input).squeeze(0)
        raw_ent = compute_entropy(ent_input).squeeze(0)

        return RouterOutput(
            route_logits=route_logits,
            route_label=route_label,
            route_prob=route_prob,
            entropy=float(ent_tensor.item()),
            raw_entropy=float(raw_ent.item()),
            hidden=last_hidden if return_hidden else None,
            base_logits=lm_logits_last,
        )

    # ------------------------------------------------------------------ #
    #  高层路由接口
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def route(self, prompt: str, max_length: int = 256) -> RouterOutput:
        """对单个 prompt 做路由推理。"""
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        return self.forward(**enc)

    @torch.no_grad()
    def batch_route(self, prompts: List[str], max_length: int = 256) -> List[RouterOutput]:
        """批量路由（左 padding 对齐）。"""
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)
        out = self.forward(**enc)
        # 批量展开为单条结果列表
        results = []
        for i in range(out.route_logits.size(0)):
            results.append(RouterOutput(
                route_logits=out.route_logits[i],
                route_label=int(out.route_logits[i].argmax().item()),
                route_prob=out.route_prob[i],
                entropy=float(out.entropy if i == 0 else out.entropy),
                raw_entropy=float(out.raw_entropy if i == 0 else out.raw_entropy),
            ))
        return results

    # ------------------------------------------------------------------ #
    #  LoRA 管理（透传给 MultiLoRAManager）
    # ------------------------------------------------------------------ #
    def load_adapter(self, name: str, path: str, **kw) -> None:
        if not self.lora_mgr:
            raise RuntimeError("未启用 multi-LoRA（use_lora=False）")
        self.lora_mgr.load_adapter(name, path, **kw)

    def activate_adapter(self, name: str) -> None:
        if not self.lora_mgr:
            raise RuntimeError("未启用 multi-LoRA")
        self.lora_mgr.activate(name)

    def select_adapter_by_device(self, device_ctx) -> List[str]:
        if not self.lora_mgr:
            return []
        return self.lora_mgr.select_by_device(device_ctx)

    def add_trainable_adapter(self, name: str) -> None:
        if not self.lora_mgr:
            raise RuntimeError("未启用 multi-LoRA")
        self.lora_mgr.load_adapter(name, path=None)

    def save_adapter(self, name: str, path: str) -> None:
        if not self.lora_mgr:
            raise RuntimeError("未启用 multi-LoRA")
        self.lora_mgr.save_adapter(name, path)

    # ------------------------------------------------------------------ #
    #  训练模式：仅 route_head + 活跃 LoRA 可训
    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        """返回可训练参数（路由头 + 当前活跃 LoRA）。"""
        params = list(self.route_head.parameters())
        if self.lora_mgr:
            for n, p in self.lora_mgr.peft_model.named_parameters():
                if "lora_" in n and p.requires_grad:
                    params.append(p)
        return params

    # ------------------------------------------------------------------ #
    #  诊断
    # ------------------------------------------------------------------ #
    def info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": self.backbone_name,
            "path": self.backbone_path,
            "hidden_size": self.hidden_size,
            "dtype": str(self.dtype),
            "device": self.device,
            "total_params_M": round(total / 1e6, 2),
            "trainable_params_M": round(trainable / 1e6, 2),
            "vocab_size": self.tokenizer.vocab_size,
            "lora_adapters": self.lora_mgr.list_adapters() if self.lora_mgr else [],
            "active_adapter": self.lora_mgr.active if self.lora_mgr else None,
        }
