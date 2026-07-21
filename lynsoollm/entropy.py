"""
entropy.py
==========
Token 级信息熵（Information Entropy）计算工具。

在推测式接力机制中，路由模型并不直接生成文本，而是实时监控
本地小模型输出 Token 的 logits，计算其归一化分布的香农熵：

    H(p) = - Σ p_i * log(p_i)

当 H 超过阈值时，说明小模型对下一个 Token 高度不确定，
极可能产生幻觉（hallucination），此时触发 Early-Exit。

所有计算基于 PyTorch，可在 CPU/GPU 上运行，单次开销 < 1ms。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    基于 logits 计算香农熵（自然对数底）。

    参数:
        logits : 形状 (..., vocab_size) 的未归一化分数。
        dim    : 词表所在维度，默认最后一维。
        eps    : 数值稳定常数，防止 log(0)。

    返回:
        entropy : 形状 (...) 的逐位置熵值（nats）。
                  若需 bits，可乘以 1/ln(2) ≈ 1.4427。

    示例:
        >>> logits = torch.tensor([[0.1, 0.2, 0.7, -0.3]])
        >>> compute_entropy(logits).item()   # 标量
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits 必须是 torch.Tensor，收到 {type(logits)}")

    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    # H = - Σ p * log p，用 log_softmax 保证数值稳定
    entropy = -torch.sum(probs * log_probs, dim=dim)
    # 防止极小负值（浮点误差）
    return torch.clamp(entropy, min=0.0) + 0.0


def token_level_entropy(
    logits: torch.Tensor, base: str = "nat", eps: float = 1e-12
) -> torch.Tensor:
    """
    与 compute_entropy 等价，但支持指定对数底。

    参数:
        logits : (..., vocab_size)
        base   : "nat"（自然对数）或 "bit"（以 2 为底）。
        eps    : 数值稳定常数。

    返回:
        逐位置熵值张量。
    """
    ent = compute_entropy(logits, dim=-1, eps=eps)
    if base == "bit":
        ent = ent / torch.log(torch.tensor(2.0))
    elif base != "nat":
        raise ValueError(f"不支持的 base: {base}，仅支持 'nat' 或 'bit'")
    return ent


def normalized_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    归一化熵（0~1 区间），便于跨词表大小比较：
        H_norm = H / log(vocab_size)

    越接近 1 表示分布越平坦（越不确定），越接近 0 表示越尖锐（越确定）。
    """
    ent = compute_entropy(logits, dim=-1, eps=eps)
    vocab_size = logits.shape[-1]
    max_ent = torch.log(torch.tensor(float(vocab_size)))
    return ent / (max_ent + eps)


def topk_uncertainty(logits: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 top-k 概率质量集中度，作为熵的补充指标。

    返回:
        topk_mass : top-k 概率之和（越大越确定）。
        margin    : top1 与 top2 概率之差（越大越确定）。
    """
    probs = F.softmax(logits, dim=-1)
    topk_probs, _ = torch.topk(probs, k=k, dim=-1)
    topk_mass = topk_probs.sum(dim=-1)
    if k >= 2:
        margin = topk_probs[..., 0] - topk_probs[..., 1]
    else:
        margin = topk_probs[..., 0]
    return topk_mass, margin
