"""
pricing.py
==========
模型定价获取器，支持四种来源：

    1. models_dev   : 从 https://models.dev/api.json 拉取（社区维护，1000+ 模型）
    2. official     : 调用模型厂商官方定价 API（如 OpenAI /v1/pricing 等）
    3. custom_api   : 用户自定义 HTTP 接口（返回 JSON 定价）
    4. manual       : 直接在配置里写死（input_per_1k / output_per_1k）

定价单位统一为：USD per 1K tokens（内部使用，与 DeviceContext.cloud_price_per_1k 对齐）。
models.dev 返回的是 USD/M tokens，需除以 1000 转 1K。

字段统一：
    PricingInfo(input_per_1k, output_per_1k, currency, source, fetched_at, raw)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    import urllib.request as urlreq
    import urllib.error as urlerr
except ImportError:  # pragma: no cover
    urlreq = None  # type: ignore


# --------------------------------------------------------------------- #
#  定价数据结构
# --------------------------------------------------------------------- #
@dataclass
class PricingInfo:
    """统一格式的模型定价信息。"""

    input_per_1k: float           # USD per 1K input tokens
    output_per_1k: float          # USD per 1K output tokens
    currency: str = "USD"
    source: str = "manual"        # models_dev / official / custom_api / manual
    fetched_at: float = 0.0       # unix timestamp
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "input_per_1k": self.input_per_1k,
            "output_per_1k": self.output_per_1k,
            "currency": self.currency,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PricingInfo":
        return cls(
            input_per_1k=float(d.get("input_per_1k", 0.0)),
            output_per_1k=float(d.get("output_per_1k", 0.0)),
            currency=d.get("currency", "USD"),
            source=d.get("source", "manual"),
            fetched_at=float(d.get("fetched_at", 0.0)),
            raw=d.get("raw", {}),
        )


# --------------------------------------------------------------------- #
#  HTTP 工具
# --------------------------------------------------------------------- #
def _http_get_json(url: str, timeout: float = 15.0,
                   headers: Optional[Dict[str, str]] = None) -> Any:
    """用标准库 urllib 拉 JSON，避免引入 requests 依赖。"""
    if urlreq is None:
        raise RuntimeError("urllib 不可用")
    default_headers = {
        "Accept": "application/json",
        "User-Agent": "LynSooLLM/0.2 (https://github.com/lynsoollm)",
    }
    if headers:
        default_headers.update(headers)
    req = urlreq.Request(url, headers=default_headers)
    with urlreq.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data)


# --------------------------------------------------------------------- #
#  定价获取器
# --------------------------------------------------------------------- #
class PricingFetcher:
    """
    多源定价获取器。

    用法：
        pf = PricingFetcher(cache_path="/tmp/models_dev.json")
        info = pf.fetch("openai/gpt-4o", source="models_dev")
        info = pf.fetch("openai/gpt-4o", source="manual",
                        manual={"input_per_1k": 0.005, "output_per_1k": 0.015})
    """

    MODELS_DEV_API = "https://models.dev/api.json"

    def __init__(self, cache_path: Optional[str] = None,
                 cache_ttl_sec: int = 86400) -> None:
        """
        参数:
            cache_path    : models.dev 缓存文件路径，避免每次重新拉取
            cache_ttl_sec : 缓存有效期（秒），默认 1 天
        """
        self.cache_path = cache_path or os.path.expanduser(
            "~/.cache/lynsoollm/models_dev.json"
        )
        self.cache_ttl = cache_ttl_sec
        self._cache: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #
    def fetch(
        self,
        model_id: str,
        source: str = "models_dev",
        provider: Optional[str] = None,
        custom_api_url: Optional[str] = None,
        custom_api_headers: Optional[Dict[str, str]] = None,
        manual: Optional[Dict[str, float]] = None,
        official_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> PricingInfo:
        """
        按指定来源获取定价。

        参数:
            model_id        : 模型 ID（如 "openai/gpt-4o" 或 "gpt-4o"）
            source          : "models_dev" / "official" / "custom_api" / "manual"
            provider        : models.dev 的 provider key（如 "openai"），用于精确匹配
            custom_api_url  : custom_api 模式下的 HTTP 接口
            custom_api_headers: custom_api 模式下的请求头
            manual          : manual 模式下的定价 dict
            official_endpoint: official 模式下的厂商定价 API
            api_key         : official / custom_api 模式下的鉴权 key
        """
        if source == "models_dev":
            return self._fetch_from_models_dev(model_id, provider)
        if source == "official":
            return self._fetch_from_official(model_id, official_endpoint, api_key)
        if source == "custom_api":
            return self._fetch_from_custom_api(
                model_id, custom_api_url, custom_api_headers
            )
        if source == "manual":
            return self._fetch_from_manual(model_id, manual or {})
        raise ValueError(f"未知 source: {source}")

    # ------------------------------------------------------------------ #
    #  1) models.dev
    # ------------------------------------------------------------------ #
    def _load_models_dev_cache(self) -> Dict:
        if self._cache is not None:
            return self._cache

        # 尝试从磁盘缓存读
        if Path(self.cache_path).exists():
            age = time.time() - Path(self.cache_path).stat().st_mtime
            if age < self.cache_ttl:
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
                return self._cache

        # 拉新数据
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        data = _http_get_json(self.MODELS_DEV_API, timeout=30)
        with open(self.cache_path, "w") as f:
            json.dump(data, f)
        self._cache = data
        return data

    def _fetch_from_models_dev(self, model_id: str,
                                provider: Optional[str]) -> PricingInfo:
        data = self._load_models_dev_cache()

        # models.dev 结构：{provider_id: {models: {model_id: {cost: {...}}}}}
        # model_id 形如 "openai/gpt-4o" 或 "gpt-4o"
        target_model = model_id.split("/")[-1]

        for prov_id, prov in data.items():
            if provider and prov_id != provider:
                continue
            models = prov.get("models", {}) if isinstance(prov, dict) else {}
            for m_id, m in models.items():
                # 多种匹配：full id、短 id、name
                if m_id == model_id or m_id == target_model or \
                   m_id.endswith("/" + target_model):
                    cost = m.get("cost") or {}
                    # models.dev 单位 USD/M tokens，转 USD/1K
                    inp = float(cost.get("input", 0)) / 1000.0
                    out = float(cost.get("output", 0)) / 1000.0
                    return PricingInfo(
                        input_per_1k=inp,
                        output_per_1k=out,
                        source="models_dev",
                        fetched_at=time.time(),
                        raw={"provider": prov_id, "model": m_id, "cost": cost},
                    )
        raise LookupError(f"models.dev 未找到模型: {model_id}")

    # ------------------------------------------------------------------ #
    #  2) 官方定价 API
    # ------------------------------------------------------------------ #
    def _fetch_from_official(self, model_id: str,
                              endpoint: Optional[str],
                              api_key: Optional[str]) -> PricingInfo:
        if not endpoint:
            # 厂商内置定价端点（少数支持）
            builtin = self._builtin_official_endpoints(model_id)
            if not builtin:
                raise ValueError(
                    f"official 模式需指定 official_endpoint，且 {model_id} 无内置端点"
                )
            endpoint = builtin
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = _http_get_json(endpoint, headers=headers, timeout=15)
        # 期望格式 {"input_per_1k": x, "output_per_1k": y, ...}
        inp = float(data.get("input_per_1k") or data.get("input_per_1m", 0) / 1000.0)
        out = float(data.get("output_per_1k") or data.get("output_per_1m", 0) / 1000.0)
        return PricingInfo(
            input_per_1k=inp, output_per_1k=out,
            source="official", fetched_at=time.time(), raw=data,
        )

    @staticmethod
    def _builtin_official_endpoints(model_id: str) -> Optional[str]:
        """部分厂商的内置定价端点。"""
        if model_id.startswith("deepseek"):
            return "https://api.deepseek.com/pricing"
        if model_id.startswith("openrouter"):
            return "https://openrouter.ai/api/v1/models"
        return None

    # ------------------------------------------------------------------ #
    #  3) 自定义接口
    # ------------------------------------------------------------------ #
    def _fetch_from_custom_api(
        self, model_id: str, url: Optional[str],
        headers: Optional[Dict[str, str]],
    ) -> PricingInfo:
        if not url:
            raise ValueError("custom_api 模式需指定 custom_api_url")
        # 允许在 url 里用占位符 {model_id}
        full_url = url.replace("{model_id}", model_id)
        data = _http_get_json(full_url, headers=headers or {}, timeout=15)
        inp = float(data.get("input_per_1k", 0))
        out = float(data.get("output_per_1k", 0))
        return PricingInfo(
            input_per_1k=inp, output_per_1k=out,
            source="custom_api", fetched_at=time.time(), raw=data,
        )

    # ------------------------------------------------------------------ #
    #  4) 手动
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fetch_from_manual(model_id: str, manual: Dict[str, float]) -> PricingInfo:
        return PricingInfo(
            input_per_1k=float(manual.get("input_per_1k", 0.0)),
            output_per_1k=float(manual.get("output_per_1k", 0.0)),
            currency=manual.get("currency", "USD"),
            source="manual",
            fetched_at=time.time(),
            raw=manual,
        )

    # ------------------------------------------------------------------ #
    #  缓存控制
    # ------------------------------------------------------------------ #
    def refresh_cache(self) -> None:
        """强制刷新 models.dev 缓存。"""
        if Path(self.cache_path).exists():
            Path(self.cache_path).unlink()
        self._cache = None
        self._load_models_dev_cache()
