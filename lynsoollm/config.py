"""
config.py
=========
YAML 配置加载器：用户输入模型连接信息、路由策略、设备上下文的入口。

配置示例（config.yaml）：

    models:
      - name: gpt-4o
        provider: openai
        model_id: openai/gpt-4o
        endpoint: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}
        pricing_source: models_dev
        quality_tier: 4
        enabled: true

      - name: claude-sonnet
        provider: anthropic
        model_id: anthropic/claude-3.5-sonnet
        endpoint: https://api.anthropic.com
        api_key: ${ANTHROPIC_API_KEY}
        pricing_source: manual
        manual_pricing: {input_per_1k: 0.003, output_per_1k: 0.015}
        quality_tier: 4

      - name: deepseek-chat
        provider: deepseek
        model_id: deepseek/deepseek-chat
        endpoint: https://api.deepseek.com
        api_key: ${DEEPSEEK_API_KEY}
        pricing_source: models_dev
        quality_tier: 3

      - name: local-fallback
        provider: custom
        endpoint: http://localhost:8080/v1
        pricing_source: manual
        manual_pricing: {input_per_1k: 0.0, output_per_1k: 0.0}
        quality_tier: 1
        priority: 99            # manual 策略下生效
        weight: 0.1

    router:
      backbone: gemma            # gemma / qwen
      strategy: cost_first       # cost_first/quality_first/balanced/latency_first/manual
      entropy_threshold: 1.5
      max_new_tokens: 32
      lora_adapters:
        - name: weak_net
          path: ./checkpoints/weak_net
        - name: low_battery
          path: ./checkpoints/low_battery

    device:
      network_rtt_ms: 50
      battery_pct: 80
      temperature_c: 30
      offline: false
      cloud_price_per_1k: 0.01

    pricing:
      cache_path: ~/.cache/lynsoollm/models_dev.json
      cache_ttl_sec: 86400
      zero_price_fallback: 0.01
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# YAML 在标准库外，做一次软导入
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from .model_registry import ModelEntry, ModelRegistry, PriorityCalculator
from .pricing import PricingFetcher
from .router import DeviceContext, RouterConfig


# --------------------------------------------------------------------- #
#  顶层配置
# --------------------------------------------------------------------- #
@dataclass
class AppConfig:
    """整个应用配置。"""

    models: List[Dict] = field(default_factory=list)
    router: Dict = field(default_factory=dict)
    device: Dict = field(default_factory=dict)
    pricing: Dict = field(default_factory=dict)

    # 以下由 builder 解析后填充
    registry: Optional[ModelRegistry] = None
    calculator: Optional[PriorityCalculator] = None
    router_config: Optional[RouterConfig] = None
    device_ctx: Optional[DeviceContext] = None
    pricing_fetcher: Optional[PricingFetcher] = None


# --------------------------------------------------------------------- #
#  加载与构建
# --------------------------------------------------------------------- #
def load_config(path: str) -> AppConfig:
    """从 YAML 文件加载配置。"""
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，请 pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(
        models=raw.get("models", []),
        router=raw.get("router", {}),
        device=raw.get("device", {}),
        pricing=raw.get("pricing", {}),
    )


def load_config_from_dict(d: Dict) -> AppConfig:
    """从字典加载配置（便于在代码里直接构造）。"""
    if yaml is not None:
        # 借用 yaml 来做 ${VAR} 展开校验之类的，但 dict 入口直接走
        pass
    return AppConfig(
        models=d.get("models", []),
        router=d.get("router", {}),
        device=d.get("device", {}),
        pricing=d.get("pricing", {}),
    )


def build_app(config: AppConfig, fetch_pricing: bool = True,
              verbose: bool = False) -> AppConfig:
    """
    把 AppConfig 构建为可运行的对象图：
        - PricingFetcher
        - ModelRegistry（含定价）
        - PriorityCalculator
        - RouterConfig
        - DeviceContext
    """
    # 1) PricingFetcher
    p_cfg = config.pricing or {}
    pf = PricingFetcher(
        cache_path=p_cfg.get("cache_path") or os.path.expanduser(
            "~/.cache/lynsoollm/models_dev.json"
        ),
        cache_ttl_sec=int(p_cfg.get("cache_ttl_sec", 86400)),
    )
    config.pricing_fetcher = pf

    # 2) ModelRegistry
    registry = ModelRegistry.from_config_list(config.models, pricing_fetcher=pf)
    config.registry = registry

    if fetch_pricing:
        if verbose:
            print("=== 拉取模型定价 ===")
        registry.fetch_all_pricing(verbose=verbose)
    else:
        # 即使不拉远程定价，也要处理 manual 来源（纯本地，不走网络）
        for e in registry.list_enabled():
            if e.pricing is None and e.pricing_source == "manual" and e.manual_pricing:
                e.pricing = pf.fetch(
                    model_id=e.model_id or e.name,
                    source="manual", manual=e.manual_pricing,
                )

    # 3) PriorityCalculator
    r_cfg = config.router or {}
    strategy = r_cfg.get("strategy", "cost_first")
    calc = PriorityCalculator(
        strategy=strategy,
        cost_weight=float(r_cfg.get("cost_weight", 0.5)),
        quality_weight=float(r_cfg.get("quality_weight", 0.5)),
        zero_price_fallback=float(p_cfg.get("zero_price_fallback", 0.01)),
    )
    config.calculator = calc

    # 4) DeviceContext
    d_cfg = config.device or {}
    device_ctx = DeviceContext(
        network_rtt_ms=float(d_cfg.get("network_rtt_ms", 50.0)),
        battery_pct=float(d_cfg.get("battery_pct", 80.0)),
        temperature_c=float(d_cfg.get("temperature_c", 30.0)),
        offline=bool(d_cfg.get("offline", False)),
        cloud_price_per_1k=float(d_cfg.get("cloud_price_per_1k", 0.01)),
    )
    config.device_ctx = device_ctx

    # 5) RouterConfig
    config.router_config = RouterConfig(
        entropy_threshold=float(r_cfg.get("entropy_threshold", 1.5)),
        max_new_tokens=int(r_cfg.get("max_new_tokens", 32)),
        min_tokens_before_exit=int(r_cfg.get("min_tokens_before_exit", 1)),
        dynamic_threshold=bool(r_cfg.get("dynamic_threshold", True)),
    )

    # 6) 计算优先级与权重
    calc.compute(registry, device_ctx)
    return config


# --------------------------------------------------------------------- #
#  生成示例配置文件
# --------------------------------------------------------------------- #
EXAMPLE_CONFIG = """\
# 灵枢 LynSooLLM 配置文件
# 用户只需编辑本文件即可接入新模型
#
# protocol 字段（可选，默认 auto）：
#   openai_chat          : POST /v1/chat/completions         (传统 OpenAI 兼容)
#   openai_responses     : POST /v1/responses                (OpenAI 新一代 Responses API)
#   anthropic_messages   : POST /v1/messages                 (Claude 原生 Messages API)
#   gemini_generate      : POST /v1beta/models/{m}:generateContent (Gemini 原生)
#   auto                 : 根据 provider / endpoint / model_id 自动选

models:
  # OpenAI 新一代 Responses API（推荐用于新项目）
  - name: gpt-4o-responses
    provider: openai
    model_id: openai/gpt-4o
    endpoint: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    protocol: openai_responses          # 强制走 Responses API
    pricing_source: models_dev
    quality_tier: 4

  # OpenAI 传统 Chat Completions（兼容性最广）
  - name: gpt-4o-chat
    provider: openai
    model_id: openai/gpt-4o
    endpoint: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    protocol: openai_chat               # 显式声明（默认也是它）
    pricing_source: models_dev
    quality_tier: 4

  # Anthropic Claude 原生 Messages API
  - name: claude-sonnet
    provider: anthropic                  # provider=anthropic 自动推断 protocol
    model_id: anthropic/claude-3.5-sonnet
    endpoint: https://api.anthropic.com
    api_key: ${ANTHROPIC_API_KEY}
    pricing_source: manual
    manual_pricing:
      input_per_1k: 0.003
      output_per_1k: 0.015
    quality_tier: 4

  # Google Gemini 原生 generateContent API
  - name: gemini-pro
    provider: gemini                     # provider=gemini 自动推断 protocol
    model_id: gemini-2.5-flash
    endpoint: https://generativelanguage.googleapis.com/v1beta
    api_key: ${GEMINI_API_KEY}
    pricing_source: models_dev
    quality_tier: 3

  # DeepSeek（OpenAI 兼容，走 chat/completions）
  - name: deepseek-chat
    provider: deepseek
    model_id: deepseek/deepseek-chat
    endpoint: https://api.deepseek.com
    api_key: ${DEEPSEEK_API_KEY}
    pricing_source: models_dev
    quality_tier: 3

  # 本地兜底（vLLM / Ollama 等 OpenAI 兼容服务）
  - name: local-fallback
    provider: custom
    endpoint: http://localhost:8080/v1
    pricing_source: manual
    manual_pricing:
      input_per_1k: 0.0
      output_per_1k: 0.0
    quality_tier: 1
    priority: 99
    weight: 0.1

router:
  backbone: gemma                       # gemma / qwen，用户选型
  strategy: cost_first                  # cost_first/quality_first/balanced/latency_first/manual
  entropy_threshold: 1.5
  max_new_tokens: 32
  lora_adapters:
    - name: weak_net
      path: ./checkpoints/weak_net
    - name: low_battery
      path: ./checkpoints/low_battery

device:
  network_rtt_ms: 50
  battery_pct: 80
  temperature_c: 30
  offline: false
  cloud_price_per_1k: 0.01

pricing:
  cache_path: ~/.cache/lynsoollm/models_dev.json
  cache_ttl_sec: 86400
  zero_price_fallback: 0.01
"""


def write_example_config(path: str) -> None:
    """生成示例配置文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(EXAMPLE_CONFIG)
