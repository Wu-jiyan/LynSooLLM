"""
mock_local_model.py
===================
本地小模型的 Mock 实现，用于在没有真实 LLM 权重的情况下
验证 SpeculativeRouter 的“边生成边评估”流程。

它支持：
    - 流式逐 token 生成
    - 每步返回 logits（用于熵计算）
    - 通过 ``entropy_schedule`` 精确控制每步熵值，便于单测

真实部署时，可替换为 ONNX Runtime / llama.cpp 的封装，
只要 yield 出 (token_str, logits) 二元组即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch


@dataclass
class MockLocalModel:
    """
    参数:
        vocab_size       : 词表大小。
        default_tokens   : 默认 token 字符串序列（用于演示）。
        entropy_schedule : 可选的每步熵值列表。若提供，则 logits
                           会构造为产生对应熵值的分布；否则使用
                           随机 logits。
        seed             : 随机种子。
    """

    vocab_size: int = 32
    default_tokens: List[str] = field(
        default_factory=lambda: ["你好", "，", "我", "是", "灵", "枢", "路由", "。"]
    )
    entropy_schedule: Optional[List[float]] = None
    seed: int = 42

    def __post_init__(self) -> None:
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)

    # ------------------------------------------------------------------ #
    #  工具：根据目标熵值构造 logits
    # ------------------------------------------------------------------ #
    def _logits_for_target_entropy(self, target_entropy: float, step: int) -> torch.Tensor:
        """
        构造一个 vocab_size 维的 logits，使其 softmax 分布的熵近似
        等于 target_entropy。

        思路：用一个温度 T 控制分布平坦度。T 越大熵越大；T 越小熵越
        小。通过简单二分搜索找到合适的 T。
        """
        import math

        vocab = self.vocab_size
        # 选取一个固定的"偏好 token"作为峰值
        peak = step % vocab
        logits_base = torch.full((vocab,), -10.0)
        logits_base[peak] = 0.0

        # 二分搜索温度 T ∈ [0.01, 10]
        lo, hi = 0.01, 10.0
        target = float(target_entropy)
        for _ in range(20):
            mid = (lo + hi) / 2.0
            scaled = logits_base / mid
            probs = torch.softmax(scaled, dim=-1)
            ent = -(probs * torch.log(probs + 1e-12)).sum().item()
            if ent < target:
                lo = mid  # 升温 -> 增大熵
            else:
                hi = mid  # 降温 -> 减小熵
        return logits_base / ((lo + hi) / 2.0)

    def _random_logits(self) -> torch.Tensor:
        return torch.randn(self.vocab_size, generator=self._generator)

    # ------------------------------------------------------------------ #
    #  流式生成接口
    # ------------------------------------------------------------------ #
    def stream(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        流式生成：每次 yield 一个 (token_str, logits) 二元组。

        真实模型中，logits 是“生成该 token 时所基于的”下一个 token
        分布；这里为简化语义，直接返回“该 token 对应位置”的 logits。
        """
        tokens = list(self.default_tokens)
        if max_new_tokens is not None:
            tokens = tokens[:max_new_tokens]

        for step, tok in enumerate(tokens):
            if self.entropy_schedule is not None and step < len(self.entropy_schedule):
                logits = self._logits_for_target_entropy(self.entropy_schedule[step], step)
            else:
                logits = self._random_logits()
            yield tok, logits

    # ------------------------------------------------------------------ #
    #  便于业务侧替换的统一接口
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_new_tokens: int = 16) -> Tuple[str, List[torch.Tensor]]:
        """非流式：直接返回完整文本与每步 logits 列表。"""
        toks: List[str] = []
        logits_list: List[torch.Tensor] = []
        for tok, logits in self.stream(prompt, max_new_tokens=max_new_tokens):
            toks.append(tok)
            logits_list.append(logits)
        return "".join(toks), logits_list
