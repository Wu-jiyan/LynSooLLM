"""
train_router.py
===============
路由模型 SFT + LoRA 微调脚本。

数据构造：
    - 从一组 prompt 模板 + 模拟环境标签合成路由分类样本
    - 每条样本: {"prompt": str, "env": {...}, "label": 0/1}
        label=0 (local): 简单 prompt / 弱网 / 低电量 / 离线
        label=1 (cloud): 复杂 prompt / 网络好 / 电量充足 / 云端便宜
    - 真实场景下可接入 GPT-4 标注或用户隐式反馈

训练目标：
    - 冻结基底，仅训 RouteHead + 当前活跃 LoRA adapter
    - 损失 = 路由分类 CE + 熵正则（让 local 类的 prompt 末位熵低，
      cloud 类的熵高，作为看门狗信号）
    - 训完保存 adapter 到磁盘，供 RealRouterEngine 加载

用法：
    python -m lynsoollm.train_router --backbone gemma --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .real_router_model import RealRouterModel, ROUTE_CLOUD, ROUTE_LOCAL
from .multi_lora import MultiLoRAManager
from .adapter_selector import builtin_rules


# --------------------------------------------------------------------- #
#  合成数据集
# --------------------------------------------------------------------- #
EASY_PROMPTS = [
    "你好",
    "今天天气怎么样",
    "1+1等于几",
    "把这句话翻译成英文：你好",
    "帮我写一个加法函数",
    "现在几点",
    "再见",
    "谢谢",
    "我叫小明",
    "讲个短笑话",
]

HARD_PROMPTS = [
    "用 Rust 实现一个支持 MVCC 的嵌入式 KV 数据库，要求支持事务与快照隔离",
    "请分析 2024 年全球半导体供应链的结构性瓶颈，并给出三个可行的去风险策略",
    "写一首关于量子纠缠的七律，要求押平水韵",
    "设计一个分布式共识算法，能在 Byzantine 容错下达到最终一致性，并分析其复杂度",
    "对比 GPT-4o 与 Claude 3.5 在多步推理任务上的能力差异，给出评测方案",
    "解释 Transformer 中 multi-head attention 的信息瓶颈，并提出改进方案",
    "请写一篇 5000 字的论文综述：神经符号融合的最新进展",
    "实现一个支持 BPE 训练的 tokenizer，要求支持中文与代码混合语料",
    "分析当前 Sora 类视频生成模型的物理一致性缺陷",
    "请设计一个端云协同 LLM 路由系统，要求支持推测式接力",
]


@dataclass
class RouterSample:
    prompt: str
    label: int                       # 0=local, 1=cloud
    env: Dict[str, float] = field(default_factory=dict)


def _random_env(offline: bool = False) -> Dict[str, float]:
    return {
        "network_rtt_ms": random.choice([20, 50, 100, 300, 800, 1500]),
        "battery_pct": random.choice([15, 30, 50, 70, 90, 100]),
        "temperature_c": random.choice([25, 30, 35, 40, 45]),
        "cloud_price_per_1k": random.choice([0.001, 0.005, 0.01, 0.02, 0.05]),
        "offline": 1.0 if offline else 0.0,
    }


def _decide_label(prompt: str, env: Dict[str, float]) -> int:
    """
    合成标签规则：
        - 离线 -> local
        - 弱网(RTT>500) 或 低电量(<30) 或 高温(>40) 或 云端贵(>0.02) -> local
        - 简单 prompt 且不在上述极端环境 -> local
        - 复杂 prompt 且环境良好 -> cloud
        - 复杂 prompt + 弱网 -> local（保命优先）
    """
    if env.get("offline", 0) >= 1.0:
        return ROUTE_LOCAL
    bad_env = (
        env["network_rtt_ms"] > 500
        or env["battery_pct"] < 30
        or env["temperature_c"] > 40
        or env["cloud_price_per_1k"] > 0.02
    )
    if bad_env:
        return ROUTE_LOCAL
    is_hard = prompt in HARD_PROMPTS
    if is_hard:
        return ROUTE_CLOUD
    return ROUTE_LOCAL


def build_dataset(n: int = 200, seed: int = 42) -> List[RouterSample]:
    """合成 n 条训练样本。"""
    random.seed(seed)
    samples: List[RouterSample] = []
    for _ in range(n):
        if random.random() < 0.5:
            prompt = random.choice(EASY_PROMPTS)
        else:
            prompt = random.choice(HARD_PROMPTS)
        env = _random_env(offline=(random.random() < 0.1))
        label = _decide_label(prompt, env)
        samples.append(RouterSample(prompt=prompt, label=label, env=env))
    return samples


class RouterDataset(Dataset):
    """torch Dataset：返回 (input_ids, attention_mask, label)。"""

    def __init__(self, samples: List[RouterSample], tokenizer, max_length: int = 128) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        enc = self.tokenizer(
            s.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "label": torch.tensor(s.label, dtype=torch.long),
        }


# --------------------------------------------------------------------- #
#  训练循环
# --------------------------------------------------------------------- #
def train(
    backbone_name: str = "gemma",
    adapter_name: str = "default",
    n_samples: int = 200,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    entropy_reg: float = 0.1,
    save_dir: Optional[str] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Dict:
    """
    训练 RealRouterModel 的 route_head + 指定 LoRA adapter。

    参数:
        adapter_name : 训练目标 adapter（必须是已注册的可训练 adapter）
        save_dir     : 训完保存 adapter 的目录
    """
    print(f"=== 训练开始: backbone={backbone_name}, adapter={adapter_name} ===")

    # 1) 加载模型
    model = RealRouterModel(
        backbone_name=backbone_name, dtype=dtype, device=device, use_lora=True
    )
    print(f"模型信息: {model.info()}")

    # 2) 若目标 adapter 不存在，新建一个可训练 adapter
    if model.lora_mgr is None:
        raise RuntimeError("use_lora=True 但 lora_mgr 为 None")
    if adapter_name not in model.lora_mgr.list_adapters():
        print(f"新建可训练 adapter: {adapter_name}")
        model.lora_mgr.load_adapter(adapter_name, path=None)
    model.lora_mgr.activate(adapter_name)

    # 3) 准备数据
    samples = build_dataset(n=n_samples)
    n_local = sum(1 for s in samples if s.label == ROUTE_LOCAL)
    n_cloud = len(samples) - n_local
    print(f"数据集: {len(samples)} 条 (local={n_local}, cloud={n_cloud})")

    ds = RouterDataset(samples, model.tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 4) 优化器：仅可训练参数
    params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=lr)

    # 5) 训练循环
    model.train()
    history = {"loss": [], "route_acc": []}
    for ep in range(epochs):
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model.forward(input_ids=input_ids, attention_mask=attn)
            # 路由分类损失
            ce_loss = F.cross_entropy(out.route_logits, labels)

            # 熵正则：用基底 LM 在 prompt 末位的 logits 计算每条样本的归一化熵，
            # 让 local 类熵偏低、cloud 类熵偏高（作为看门狗信号）
            # out.base_logits: (B, vocab) 当 batch 时
            base_logits = out.base_logits
            if base_logits.dim() == 2:
                from .entropy import normalized_entropy as _ne
                ent_per_sample = _ne(base_logits)  # (B,)
            else:
                ent_per_sample = torch.tensor([out.entropy] * labels.size(0),
                                              device=device, dtype=torch.float32)
            target_ent = labels.float() * 0.8 + (1 - labels.float()) * 0.2
            ent_loss = F.mse_loss(ent_per_sample, target_ent)
            loss = ce_loss + entropy_reg * ent_loss
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_correct += (out.route_logits.argmax(dim=-1) == labels).sum().item()
            ep_total += labels.size(0)

        avg_loss = ep_loss / max(len(loader), 1)
        acc = ep_correct / max(ep_total, 1)
        history["loss"].append(avg_loss)
        history["route_acc"].append(acc)
        print(f"  epoch {ep+1}/{epochs}  loss={avg_loss:.4f}  route_acc={acc:.3f}")

    # 6) 保存 adapter（PEFT 会把权重存到 save_dir/adapter_name/，
    #    route_head 与 history 直接放 save_dir 顶层）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_adapter(adapter_name, save_dir)
        # 保存 route_head
        torch.save(model.route_head.state_dict(), Path(save_dir) / "route_head.pt")
        # 保存训练历史
        with open(Path(save_dir) / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"adapter 已保存到: {Path(save_dir) / adapter_name}")
        print(f"route_head 已保存到: {Path(save_dir) / 'route_head.pt'}")

    return history


# --------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="灵枢 LynSooLLM 路由模型微调")
    parser.add_argument("--backbone", choices=["gemma", "qwen"], default="gemma",
                        help="基底模型（用户选型）")
    parser.add_argument("--adapter", default="default",
                        help="训练目标 adapter 名")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy_reg", type=float, default=0.1)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.backbone == "qwen" else torch.float32
    train(
        backbone_name=args.backbone,
        adapter_name=args.adapter,
        n_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        entropy_reg=args.entropy_reg,
        save_dir=args.save_dir,
        device=args.device,
        dtype=dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
