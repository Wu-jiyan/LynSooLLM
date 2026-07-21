"""
灵枢 LynSooLLM
==============
边缘感知与多出口推测式端云协同 LLM 智能路由引擎。

核心模块：
    - LynSooApp          : 成品入口（YAML 配置 -> 真实路由 -> 多模型接力）
    - RealRouterEngine   : 真实模型版路由引擎（Gemma-3-270M / Qwen3.5-0.8B + multi-LoRA）
    - RealRouterModel    : 真实路由模型封装（基底 + RouteHead + multi-LoRA）
    - MultiLoRAManager   : 多 LoRA adapter 动态切换 / 加权合并
    - EnvAwareSelector   : 环境感知 adapter 选择器
    - ModelRegistry      : 多云端模型连接管理 + 定价自动拉取
    - PricingFetcher     : models.dev / 官方 / 自定义接口 / 手动 四源定价
    - PriorityCalculator : 根据定价+策略生成权重与优先级
    - CloudRelayPool     : 多云端加权选择 + 接力执行
    - SpeculativeRouter  : 原型路由类（基于 MockLocalModel，用于流程验证）
    - compute_entropy    : 基于 logits 的 token 级信息熵计算
"""

# 原型
from .router import SpeculativeRouter, RouterConfig, DeviceContext
from .entropy import compute_entropy, token_level_entropy, normalized_entropy
from .exit_signal import EarlyExitSignal, RelayContext, CloudRelay
from .mock_local_model import MockLocalModel

# 真实模型
from .real_router_model import RealRouterModel, RouteHead, RouterOutput, ROUTE_LOCAL, ROUTE_CLOUD
from .multi_lora import MultiLoRAManager, AdapterMeta
from .adapter_selector import EnvAwareSelector, AdapterRule, builtin_rules
from .real_router_engine import RealRouterEngine, RealRouterEvent, LocalGenerator

# 成品（多模型 + 定价 + 接力）
from .pricing import PricingFetcher, PricingInfo
from .model_registry import ModelEntry, ModelRegistry, PriorityCalculator
from .executors import (
    OpenAIChatExecutor,
    OpenAIResponsesExecutor,
    AnthropicMessagesExecutor,
    GeminiGenerateContentExecutor,
    AutoExecutor,
    HTTPCloudExecutor,  # 向后兼容别名，等同 OpenAIChatExecutor
    MockCloudExecutor,
)
from .cloud_relay_pool import CloudRelayPool, RelayResult
from .config import AppConfig, load_config, load_config_from_dict, build_app, write_example_config, EXAMPLE_CONFIG
from .app import LynSooApp
from .server import run_server, make_app as make_server_app, DEFAULT_MODEL_NAME
from .admin import create_app as create_admin_app

__all__ = [
    # 成品
    "LynSooApp", "AppConfig", "load_config", "load_config_from_dict", "build_app",
    "write_example_config", "EXAMPLE_CONFIG",
    # HTTP 服务端（路由中转站）
    "run_server", "make_server_app", "DEFAULT_MODEL_NAME",
    # 配置端（Flask）
    "create_admin_app",
    # 定价与注册表
    "PricingFetcher", "PricingInfo",
    "ModelEntry", "ModelRegistry", "PriorityCalculator",
    # 执行器（5 种协议 + Mock）
    "OpenAIChatExecutor", "OpenAIResponsesExecutor",
    "AnthropicMessagesExecutor", "GeminiGenerateContentExecutor",
    "AutoExecutor", "HTTPCloudExecutor", "MockCloudExecutor",
    # 接力池
    "CloudRelayPool", "RelayResult",
    # 真实模型
    "RealRouterEngine", "RealRouterEvent", "LocalGenerator",
    "RealRouterModel", "RouteHead", "RouterOutput",
    "ROUTE_LOCAL", "ROUTE_CLOUD",
    "MultiLoRAManager", "AdapterMeta",
    "EnvAwareSelector", "AdapterRule", "builtin_rules",
    # 原型
    "SpeculativeRouter", "RouterConfig", "DeviceContext",
    "compute_entropy", "token_level_entropy", "normalized_entropy",
    "EarlyExitSignal", "RelayContext", "CloudRelay",
    "MockLocalModel",
]

__version__ = "0.4.0"
