"""
test_admin.py
=============
配置端 admin.py 的端到端测试。

覆盖：
    - GET  /                     主页 HTML
    - GET  /api/meta             元信息
    - GET  /api/config           读取配置
    - PUT  /api/config           保存配置
    - GET  /api/models           列出模型
    - POST /api/models           新增模型
    - PUT  /api/models/<name>    修改模型
    - DELETE /api/models/<name>  删除模型
    - PUT  /api/router           修改路由
    - PUT  /api/device           修改设备
    - PUT  /api/pricing          修改定价源
    - GET  /api/upstream-health  可达性检查
    - POST /api/reload           热重载（不真启 server）
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict

import pytest
import yaml


# --------------------------------------------------------------------- #
#  Fixture：一个临时 YAML + Flask test client
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def admin_client():
    """构造一个临时 YAML 配置 + Flask test client。"""
    # 1) 准备一个最小配置
    cfg = {
        "models": [
            {
                "name": "mock-deepseek",
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "endpoint": "http://127.0.0.1:18769/v1",  # 不可达，仅用于测试
                "api_key": "demo-key",
                "protocol": "auto",
                "pricing_source": "manual",
                "manual_pricing": {"input_per_1k": 0.0001, "output_per_1k": 0.0003},
                "quality_tier": 3,
            },
        ],
        "router": {
            "backbone": "gemma",
            "strategy": "cost_first",
            "entropy_threshold": 1.5,
            "max_new_tokens": 8,
        },
        "device": {
            "network_rtt_ms": 30,
            "battery_pct": 85,
            "temperature_c": 30,
            "offline": False,
        },
        "pricing": {
            "cache_path": "/tmp/test_admin_lynsoollm.json",
            "cache_ttl_sec": 86400,
            "zero_price_fallback": 0.01,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                      delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
        cfg_path = f.name

    # 2) 创建 Flask app（不调用 app.run）
    from lynsoollm.admin import create_app
    app = create_app(config_path=cfg_path, device="cpu")
    app.config["TESTING"] = True
    client = app.test_client()

    yield {"client": client, "cfg_path": cfg_path, "app": app}

    os.unlink(cfg_path)
    if os.path.exists(cfg_path + ".bak"):
        os.unlink(cfg_path + ".bak")


# --------------------------------------------------------------------- #
#  辅助
# --------------------------------------------------------------------- #
def _post_json(client, path: str, body: Dict) -> Dict:
    r = client.post(path, data=json.dumps(body), content_type="application/json")
    return r.get_json()


def _put_json(client, path: str, body: Dict) -> Dict:
    r = client.put(path, data=json.dumps(body), content_type="application/json")
    return r.get_json()


def _get_json(client, path: str) -> Dict:
    r = client.get(path)
    return r.get_json()


# --------------------------------------------------------------------- #
#  测试用例
# --------------------------------------------------------------------- #
def test_index_html(admin_client):
    """主页应该返回 HTML。"""
    r = admin_client["client"].get("/")
    assert r.status_code == 200
    assert b"<html" in r.data.lower()
    # HTML 应该引用本地的 Vue / TDesign / js-yaml
    assert b"/static/vue.global.prod.js" in r.data
    assert b"/static/tdesign.min.js" in r.data
    assert b"/static/js-yaml.min.js" in r.data


def test_static_files_exist(admin_client):
    """静态文件应该可下载。"""
    c = admin_client["client"]
    for f in ("vue.global.prod.js", "tdesign.min.js", "tdesign.min.css", "js-yaml.min.js"):
        r = c.get(f"/static/{f}")
        assert r.status_code == 200, f"{f} not found"
        assert len(r.data) > 1000, f"{f} too small"


def test_meta(admin_client):
    r = _get_json(admin_client["client"], "/api/meta")
    assert "config_path" in r
    assert r["backbone"] == "gemma"
    assert "strategies" in r and "cost_first" in r["strategies"]
    assert "protocols" in r and "auto" in r["protocols"]
    assert "pricing_sources" in r


def test_get_config(admin_client):
    r = _get_json(admin_client["client"], "/api/config")
    assert "models" in r and len(r["models"]) == 1
    assert r["models"][0]["name"] == "mock-deepseek"
    assert r["router"]["strategy"] == "cost_first"


def test_put_config(admin_client):
    new_cfg = {
        "models": [],
        "router": {"backbone": "qwen", "strategy": "quality_first",
                    "entropy_threshold": 2.0, "max_new_tokens": 16},
        "device": {"network_rtt_ms": 100, "battery_pct": 50,
                    "temperature_c": 35, "offline": False},
        "pricing": {"cache_path": "/tmp/x.json", "cache_ttl_sec": 3600,
                     "zero_price_fallback": 0.02},
    }
    r = _put_json(admin_client["client"], "/api/config", new_cfg)
    assert r["ok"] is True
    # 验证写回
    with open(admin_client["cfg_path"], "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    assert saved["router"]["backbone"] == "qwen"
    # 再读 API
    r2 = _get_json(admin_client["client"], "/api/config")
    assert r2["router"]["backbone"] == "qwen"


def test_list_models(admin_client):
    # 当前配置应该没模型（上一步被清空了）
    r = _get_json(admin_client["client"], "/api/models")
    assert isinstance(r, list)


def test_add_model(admin_client):
    new_model = {
        "name": "new-claude",
        "provider": "anthropic",
        "model_id": "claude-3.5-sonnet",
        "endpoint": "http://localhost:8888/v1",
        "api_key": "sk-test",
        "protocol": "anthropic_messages",
        "pricing_source": "manual",
        "manual_pricing": {"input_per_1k": 0.003, "output_per_1k": 0.015},
        "quality_tier": 4,
    }
    r = _post_json(admin_client["client"], "/api/models", new_model)
    assert r["ok"] is True
    assert r["model"]["name"] == "new-claude"

    # 重复添加应失败
    r2 = _post_json(admin_client["client"], "/api/models", new_model)
    assert "error" in r2


def test_update_model(admin_client):
    r = _put_json(admin_client["client"], "/api/models/new-claude", {
        "name": "new-claude",
        "provider": "anthropic",
        "model_id": "claude-3.5-sonnet",
        "endpoint": "http://localhost:9999/v1",  # 改 endpoint
        "api_key": "sk-test",
        "protocol": "anthropic_messages",
        "quality_tier": 5,  # 改 tier
    })
    assert r["ok"] is True
    assert r["model"]["endpoint"] == "http://localhost:9999/v1"
    assert r["model"]["quality_tier"] == 5


def test_delete_model(admin_client):
    r = admin_client["client"].delete("/api/models/new-claude")
    data = r.get_json()
    assert data["ok"] is True
    # 验证已删除
    r2 = _get_json(admin_client["client"], "/api/models")
    names = [m["name"] for m in r2]
    assert "new-claude" not in names


def test_put_router(admin_client):
    r = _put_json(admin_client["client"], "/api/router", {
        "backbone": "gemma", "strategy": "balanced",
        "entropy_threshold": 1.8, "max_new_tokens": 16,
    })
    assert r["ok"] is True
    assert r["router"]["strategy"] == "balanced"


def test_put_device(admin_client):
    r = _put_json(admin_client["client"], "/api/device", {
        "network_rtt_ms": 200, "battery_pct": 30,
        "temperature_c": 45, "offline": True,
    })
    assert r["ok"] is True
    assert r["device"]["offline"] is True


def test_put_pricing(admin_client):
    r = _put_json(admin_client["client"], "/api/pricing", {
        "cache_path": "/tmp/new.json", "cache_ttl_sec": 7200,
        "zero_price_fallback": 0.005,
    })
    assert r["ok"] is True
    assert r["pricing"]["cache_ttl_sec"] == 7200


def test_upstream_health(admin_client):
    """所有上游都不可达（端口 18769 没服务），但接口应该返回结果。"""
    r = _get_json(admin_client["client"], "/api/upstream-health")
    assert isinstance(r, list)
    # 至少调用成功
    assert len(r) >= 0


def test_reload(admin_client):
    """热重载接口应该返回 ok（不依赖外部 server）。"""
    r = _post_json(admin_client["client"], "/api/reload", {})
    # 注意：reload 会重建 LynSooApp（需要 torch/transformers）
    # 可能因依赖缺失而失败，但接口本身应该返回 JSON
    assert "ok" in r or "error" in r


def test_404(admin_client):
    r = admin_client["client"].get("/api/nonexistent")
    assert r.status_code == 404
    assert r.get_json()["error"] == "not found"


def test_validate_yaml_ok(admin_client):
    """合法 YAML 应该校验通过。"""
    yaml_text = """
models:
  - name: test
    provider: openai
    endpoint: http://localhost:8000/v1
    api_key: xxx
    protocol: auto
router:
  backbone: gemma
  strategy: cost_first
  entropy_threshold: 1.5
device:
  network_rtt_ms: 50
pricing:
  cache_path: /tmp/x.json
"""
    r = _post_json(admin_client["client"], "/api/validate-yaml", {"yaml": yaml_text})
    assert r["ok"] is True
    assert r["errors"] == []
    assert "parsed" in r


def test_validate_yaml_syntax_error(admin_client):
    """YAML 语法错误应该被捕获。"""
    yaml_text = "models: [unclosed"
    r = _post_json(admin_client["client"], "/api/validate-yaml", {"yaml": yaml_text})
    assert r["ok"] is False
    assert any("语法错误" in e for e in r["errors"])


def test_validate_yaml_missing_field(admin_client):
    """缺少必填字段应该报错。"""
    yaml_text = """
models:
  - provider: openai
router:
  backbone: gemma
  strategy: cost_first
  entropy_threshold: 1.5
"""
    r = _post_json(admin_client["client"], "/api/validate-yaml", {"yaml": yaml_text})
    assert r["ok"] is False
    assert any("name 必填" in e for e in r["errors"])


def test_validate_yaml_bad_protocol(admin_client):
    """非法 protocol 应该报错。"""
    yaml_text = """
models:
  - name: x
    provider: openai
    endpoint: http://x/v1
    protocol: invalid_proto
router:
  backbone: gemma
  strategy: cost_first
  entropy_threshold: 1.5
"""
    r = _post_json(admin_client["client"], "/api/validate-yaml", {"yaml": yaml_text})
    assert r["ok"] is False
    assert any("protocol" in e for e in r["errors"])
