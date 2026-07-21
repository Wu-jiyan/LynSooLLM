"""
exit_signal.py
==============
推测式早期退出（Early-Exit）信号与云端接力上下文封装。

当路由模型检测到本地小模型信息熵超阈值时，会构造一个
``EarlyExitSignal``，其中携带：
    - 已生成的 token 序列
    - 触发退出的位置 / 熵值
    - 上下文（prompt + 已生成内容），用于无缝接力给云端大模型

``CloudRelay`` 是一个可注入的云端大模型执行器抽象，业务侧
可替换为真实的 GPT-4o / Claude API 客户端。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class RelayContext:
    """交接给云端大模型的上下文。"""

    prompt: str
    generated_tokens: List[Any]
    generated_text: str
    exit_step: int                      # 触发 Early-Exit 的 token 步数
    exit_entropy: float                 # 触发时的熵值
    reason: str = "entropy_threshold_exceeded"
    metadata: dict = field(default_factory=dict)

    def merged_prompt(self) -> str:
        """构造给云端模型的完整 prompt（原 prompt + 已生成内容）。

        注意：此方法用于 completion-style API（纯文本拼接）。
        对 chat-style API（OpenAI Chat / Anthropic Messages / Gemini
        generateContent / OpenAI Responses）应改用 ``chat_messages()``，
        把本地已生成文本作为 assistant 预填（prefill），而非拼到 user
        消息里——否则云端模型收到的 user prompt 会被本地部分输出污染。
        """
        return f"{self.prompt}{self.generated_text}"

    def chat_messages(self, assistant_role: str = "assistant") -> list:
        """构造给 chat-style API 的 messages 数组。

        - 用户原始 prompt 作为 user 消息
        - 本地已生成文本作为 assistant 消息（prefill，让云端无缝接续）

        若 generated_text 为空（如 hard_route_cloud 场景），
        只返回 user 消息。

        参数：
            assistant_role: Gemini API 用 "model"，其它 API 用 "assistant"。
        """
        msgs: list = [{"role": "user", "content": self.prompt}]
        if self.generated_text:
            msgs.append({"role": assistant_role, "content": self.generated_text})
        return msgs


@dataclass
class EarlyExitSignal:
    """路由模型发出的“立即退出并接力”信号。"""

    triggered: bool
    exit_step: int = -1
    exit_entropy: float = 0.0
    reason: str = ""
    relay_context: Optional[RelayContext] = None

    def __bool__(self) -> bool:  # 便于 if signal: ...
        return self.triggered


# 云端大模型执行器类型：输入 RelayContext，返回补全文本
CloudExecutor = Callable[[RelayContext], str]


class CloudRelay:
    """
    云端大模型接力器。

    默认实现是一个可注入回调的轻量封装；若未提供 executor，
    则返回占位文本，便于原型阶段验证流程可跑通。
    """

    def __init__(self, executor: Optional[CloudExecutor] = None, model_name: str = "cloud-gpt4o") -> None:
        self._executor = executor
        self.model_name = model_name
        self.last_latency_ms: float = 0.0

    def handoff(self, ctx: RelayContext) -> str:
        """
        将接力上下文交给云端大模型，返回云端补全的完整文本。

        注意：在真实系统中这里会做 KV-cache 复用 / prefix caching，
        本原型仅保证接口语义正确。
        """
        if self._executor is None:
            # 原型占位：在已生成文本后追加云端补全标记
            placeholder = f"[cloud:{self.model_name} 接力补全]"
            return f"{ctx.generated_text}{placeholder}"
        return self._executor(ctx)

    def register(self, executor: CloudExecutor) -> None:
        """运行期替换执行器（例如用户接入真实 API 后）。"""
        self._executor = executor
