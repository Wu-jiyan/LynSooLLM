"""
admin.py
========
灵枢 LynSooLLM 配置端（Flask 应用）。

定位：独立运行的配置管理服务，与推理服务端 (server.py) 分离：
    - admin.py  端口 8001  管理配置文件 + 提供 Web UI + 一键测试
    - server.py 端口 8000  读 YAML 启动 + 对外暴露推理 API

通信方式：通过共享 config.yaml 文件通信（最简单可靠）。
热重载：admin 保存后写入文件，server.py 通过 mtime 检测或手动重启。

启动：
    python -m lynsoollm.admin -c config.yaml -p 8001

访问：
    浏览器打开 http://localhost:8001

API 列表：
    页面:
        GET  /                       主页（HTML）

    配置:
        GET  /api/config             读完整 YAML 配置（含原始 ${VAR} 占位符）
        PUT  /api/config             保存完整配置（写回 YAML）

    模型池 CRUD:
        GET    /api/models           列出所有模型
        POST   /api/models           新增模型
        PUT    /api/models/<name>    修改模型
        DELETE /api/models/<name>    删除模型

    路由 / 设备 / 定价:
        PUT    /api/router           修改路由配置
        PUT    /api/device           修改设备上下文
        PUT    /api/pricing          修改定价源

    工具:
        POST   /api/test             一键测试路由（输入 prompt 返回结果 + 延迟）
        GET    /api/upstream-health  检查所有上游模型可达性
        POST   /api/reload           通知 server.py 热重载（POST 到 server 的 /admin/reload）
        GET    /api/meta             元信息（路由模型名 / LoRA 列表 / server 状态等）
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import os.path
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# 离线模式（避免拉远程权重）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 确保 lynsoollm 包可被 import（即使直接运行 admin.py）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


# --------------------------------------------------------------------- #
#  YAML 软导入
# --------------------------------------------------------------------- #
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from flask import Flask, jsonify, request, send_from_directory


# --------------------------------------------------------------------- #
#  配置文件读写（保留 ${VAR} 占位符）
# --------------------------------------------------------------------- #
class ConfigStore:
    """
    YAML 配置文件的内存镜像 + 持久化。

    设计要点：
        - 用户在前端填的 api_key 如果是 ${OPENAI_API_KEY}，要原样保存
        - 后端 LynSooApp 启动时会展开 ${VAR}，但前端展示和保存时保留原样
        - 简单做法：直接读写 YAML 文件，不做转换
    """

    def __init__(self, config_path: str):
        self.path = os.path.abspath(config_path)
        self._lock = threading.Lock()
        self._cache: Optional[Dict] = None
        self._mtime: float = 0.0

    def load(self) -> Dict:
        """加载 YAML 配置。若文件不存在返回空骨架。"""
        with self._lock:
            if not os.path.exists(self.path):
                return self._skeleton()
            mtime = os.path.getmtime(self.path)
            if self._cache is not None and mtime == self._mtime:
                return copy.deepcopy(self._cache)
            with open(self.path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            # 规范化骨架
            cfg = {
                "models": raw.get("models", []) or [],
                "router": raw.get("router", {}) or {},
                "device": raw.get("device", {}) or {},
                "pricing": raw.get("pricing", {}) or {},
            }
            self._cache = copy.deepcopy(cfg)
            self._mtime = mtime
            return copy.deepcopy(cfg)

    def save(self, cfg: Dict) -> None:
        """保存配置到 YAML 文件。"""
        with self._lock:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            # 备份旧文件
            if os.path.exists(self.path):
                bak = self.path + ".bak"
                try:
                    os.replace(self.path, bak)
                except OSError:
                    pass
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True,
                               sort_keys=False, default_flow_style=False)
            self._cache = copy.deepcopy(cfg)
            self._mtime = os.path.getmtime(self.path)

    @staticmethod
    def _skeleton() -> Dict:
        return {
            "models": [],
            "router": {
                "backbone": "gemma",
                "strategy": "cost_first",
                "entropy_threshold": 1.5,
                "max_new_tokens": 32,
            },
            "device": {
                "network_rtt_ms": 50,
                "battery_pct": 80,
                "temperature_c": 30,
                "offline": False,
                "cloud_price_per_1k": 0.01,
            },
            "pricing": {
                "cache_path": "~/.cache/lynsoollm/models_dev.json",
                "cache_ttl_sec": 86400,
                "zero_price_fallback": 0.01,
            },
        }


# --------------------------------------------------------------------- #
#  LynSooApp 懒加载（一键测试用）
# --------------------------------------------------------------------- #
class AppLoader:
    """
    懒加载 LynSooApp 实例（用于一键测试）。

    LynSooApp 加载路由模型较慢（Gemma 270M 约 3 秒，Qwen 0.8B 约 8 秒），
    所以缓存一个全局实例。配置变更时通过 reload() 重建。
    """

    def __init__(self, config_path: str, device: str = "cpu"):
        self.config_path = config_path
        self.device = device
        self._app: Optional[Any] = None
        self._lock = threading.Lock()
        self._last_built_at: float = 0.0

    def get(self) -> Any:
        """获取或构建 LynSooApp。"""
        with self._lock:
            if self._app is None:
                self._build()
            return self._app

    def reload(self) -> Any:
        """强制重建 LynSooApp（配置变更后调用）。"""
        with self._lock:
            self._build()
            return self._app

    def _build(self) -> None:
        # 用绝对导入避免直接运行 admin.py 时的 "attempted relative import" 错误
        from lynsoollm.app import LynSooApp
        from lynsoollm.executors import AutoExecutor
        from lynsoollm.mock_local_model import MockLocalModel
        local = MockLocalModel(
            default_tokens=["你好", "，", "我是", "灵枢", "路由", "。"],
            entropy_schedule=[0.2, 0.3, 0.4, 2.5, 3.0, 2.8],
        )
        self._app = LynSooApp.from_config(
            self.config_path,
            local_generator=local,
            executor=AutoExecutor(timeout=30.0),
            device=self.device,
            fetch_pricing=True,
            verbose=False,
        )
        self._last_built_at = time.time()


# --------------------------------------------------------------------- #
#  上游可达性检查
# --------------------------------------------------------------------- #
def check_upstream_health(entry_dict: Dict, timeout: float = 5.0) -> Dict:
    """对一个模型配置做可达性检查（HEAD/GET 根路径或 models 端点）。"""
    name = entry_dict.get("name", "?")
    endpoint = entry_dict.get("endpoint", "")
    provider = (entry_dict.get("provider") or "").lower()
    protocol = entry_dict.get("protocol") or "auto"

    if not endpoint:
        return {"name": name, "ok": False, "status": "no_endpoint", "latency_ms": 0}

    # 推断要探测的 URL
    if provider == "gemini" or "gemini_generate" in protocol or "/v1beta" in endpoint:
        url = endpoint.rstrip("/") + "/models"
    else:
        url = endpoint.rstrip("/") + "/models"

    headers = {"User-Agent": "LynSooLLM-Admin/0.1"}
    api_key = entry_dict.get("api_key", "")
    if api_key and not api_key.startswith("${"):
        if provider == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        elif provider == "gemini":
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = (time.perf_counter() - t0) * 1000.0
            return {
                "name": name, "ok": True,
                "status": f"{resp.status}",
                "latency_ms": round(latency, 1),
            }
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - t0) * 1000.0
        # 401/403 也算"可达"，只是鉴权问题
        ok = e.code in (401, 403)
        return {
            "name": name, "ok": ok,
            "status": f"HTTP {e.code}",
            "latency_ms": round(latency, 1),
            "error": e.reason if not ok else None,
        }
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "name": name, "ok": False,
            "status": "unreachable",
            "latency_ms": round(latency, 1),
            "error": f"{type(e).__name__}: {e}",
        }


# --------------------------------------------------------------------- #
#  通知 server.py 热重载
# --------------------------------------------------------------------- #
def notify_server_reload(server_url: str, token: Optional[str] = None,
                         timeout: float = 3.0) -> Dict:
    """POST 到 server.py 的 /admin/reload 通知热重载。"""
    url = server_url.rstrip("/") + "/admin/reload"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(
            url, data=b"{}", headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status,
                    "response": json.loads(resp.read().decode("utf-8", "ignore"))}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------- #
#  Flask App 工厂
# --------------------------------------------------------------------- #
def create_app(config_path: str,
               device: str = "cpu",
               server_url: Optional[str] = None,
               server_token: Optional[str] = None) -> Flask:
    """
    创建 Flask 配置端应用。

    参数：
        config_path    YAML 配置文件路径
        device         LynSooApp 运行设备（仅用于一键测试）
        server_url     推理服务端 URL（用于热重载通知，可空）
        server_token   推理服务端鉴权 token（可空）
    """
    if yaml is None:
        raise RuntimeError("未安装 PyYAML，请 pip install pyyaml")

    store = ConfigStore(config_path)
    loader = AppLoader(config_path, device=device)

    # 静态资源目录（已下载到本地的 Vue / TDesign / js-yaml）
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    template_dir = os.path.join(os.path.dirname(__file__), "templates")

    app = Flask(__name__, template_folder=template_dir,
                static_folder=static_dir, static_url_path="/static")
    app.config["JSON_AS_ASCII"] = False

    # ----------------------------------------------------------------- #
    #  页面
    # ----------------------------------------------------------------- #
    @app.route("/")
    def index():
        """主页：返回 Vue 单页 HTML。"""
        # 整个页面是静态的 Vue 单页，所有动态数据通过 /api/* 拉取
        # 这里直接返回静态 HTML 文件，避免 Jinja2 与 Vue {{}} 语法冲突
        html_path = os.path.join(template_dir, "admin.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()

    @app.route("/favicon.ico")
    def favicon():
        """避免浏览器 404 报错。"""
        return ("", 204)

    # ----------------------------------------------------------------- #
    #  元信息
    # ----------------------------------------------------------------- #
    @app.route("/api/meta")
    def api_meta():
        """返回路由模型 / LoRA / 服务端状态等元信息。"""
        cfg = store.load()
        backbone = cfg.get("router", {}).get("backbone", "gemma")
        adapters = cfg.get("router", {}).get("lora_adapters", []) or []
        server_status = None
        if server_url:
            try:
                with urllib.request.urlopen(
                    server_url.rstrip("/") + "/health", timeout=2.0
                ) as r:
                    server_status = json.loads(r.read().decode("utf-8", "ignore"))
            except Exception as e:
                server_status = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return jsonify({
            "config_path": config_path,
            "backbone": backbone,
            "lora_adapters": adapters,
            "server_url": server_url or "",
            "server_status": server_status,
            "strategies": ["cost_first", "quality_first", "balanced",
                            "latency_first", "manual"],
            "protocols": ["auto", "openai_chat", "openai_responses",
                           "anthropic_messages", "gemini_generate"],
            "pricing_sources": ["models_dev", "official", "custom_api", "manual"],
            "backbones": ["gemma", "qwen"],
        })

    # ----------------------------------------------------------------- #
    #  YAML 校验
    # ----------------------------------------------------------------- #
    @app.route("/api/validate-yaml", methods=["POST"])
    def api_validate_yaml():
        """校验 YAML 字符串是否能解析为合法配置。"""
        body = request.get_json(force=True, silent=True) or {}
        yaml_text = body.get("yaml", "")
        if not yaml_text.strip():
            return jsonify({"ok": False, "errors": ["YAML 为空"]})
        try:
            parsed = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            return jsonify({"ok": False, "errors": [f"YAML 语法错误: {e}"]})
        if not isinstance(parsed, dict):
            return jsonify({"ok": False, "errors": [f"顶层必须是 dict，实际是 {type(parsed).__name__}"]})

        errors: List[str] = []
        warnings: List[str] = []

        # models 校验
        models = parsed.get("models", [])
        if not isinstance(models, list):
            errors.append("models 必须是 list")
        else:
            for i, m in enumerate(models):
                if not isinstance(m, dict):
                    errors.append(f"models[{i}] 必须是 dict")
                    continue
                if not m.get("name"):
                    errors.append(f"models[{i}].name 必填")
                if not m.get("endpoint") and m.get("provider") != "local":
                    errors.append(f"models[{i}].endpoint 必填（provider=local 除外）")
                proto = m.get("protocol", "auto")
                allowed = {"auto", "openai_chat", "openai_responses",
                            "anthropic_messages", "gemini_generate"}
                if proto not in allowed:
                    errors.append(f"models[{i}].protocol={proto} 非法，可选: {sorted(allowed)}")

        # router 校验
        router = parsed.get("router", {})
        if not isinstance(router, dict):
            errors.append("router 必须是 dict")
        else:
            backbone = router.get("backbone", "gemma")
            if backbone not in ("gemma", "qwen"):
                errors.append(f"router.backbone={backbone} 非法，可选: gemma / qwen")
            strategy = router.get("strategy", "cost_first")
            if strategy not in ("cost_first", "quality_first", "balanced",
                                  "latency_first", "manual"):
                errors.append(f"router.strategy={strategy} 非法")
            eth = router.get("entropy_threshold", 1.5)
            if not isinstance(eth, (int, float)) or eth <= 0:
                errors.append(f"router.entropy_threshold={eth} 必须是正数")

        # device 校验
        device = parsed.get("device", {})
        if not isinstance(device, dict):
            errors.append("device 必须是 dict")

        # pricing 校验
        pricing = parsed.get("pricing", {})
        if not isinstance(pricing, dict):
            errors.append("pricing 必须是 dict")

        # LoRA adapter 路径存在性检查（仅警告）
        for i, a in enumerate(router.get("lora_adapters", []) or []):
            if not isinstance(a, dict):
                continue
            p = a.get("path", "")
            if p and not os.path.exists(p):
                warnings.append(f"router.lora_adapters[{i}].path={p} 不存在")

        return jsonify({
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "parsed": parsed,
        })

    # ----------------------------------------------------------------- #
    #  完整配置
    # ----------------------------------------------------------------- #
    @app.route("/api/config", methods=["GET"])
    def api_get_config():
        return jsonify(store.load())

    @app.route("/api/config", methods=["PUT"])
    def api_put_config():
        body = request.get_json(force=True, silent=True) or {}
        # 基本校验
        for key in ("models", "router", "device", "pricing"):
            if key not in body:
                body[key] = {}
            if key == "models" and not isinstance(body[key], list):
                return jsonify({"error": f"models must be list, got {type(body[key]).__name__}"}), 400
            if key != "models" and not isinstance(body[key], dict):
                return jsonify({"error": f"{key} must be dict"}), 400
        try:
            store.save(body)
            # 触发懒加载重置（下次 get 时重建）
            loader._app = None
            return jsonify({"ok": True, "config": body})
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # ----------------------------------------------------------------- #
    #  模型池 CRUD
    # ----------------------------------------------------------------- #
    @app.route("/api/models", methods=["GET"])
    def api_list_models():
        cfg = store.load()
        return jsonify(cfg.get("models", []))

    @app.route("/api/models", methods=["POST"])
    def api_add_model():
        body = request.get_json(force=True, silent=True) or {}
        name = body.get("name", "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        cfg = store.load()
        # 唯一性
        for m in cfg["models"]:
            if m.get("name") == name:
                return jsonify({"error": f"model {name} already exists"}), 400
        cfg["models"].append(body)
        store.save(cfg)
        loader._app = None
        return jsonify({"ok": True, "model": body}), 201

    @app.route("/api/models/<name>", methods=["PUT"])
    def api_update_model(name: str):
        body = request.get_json(force=True, silent=True) or {}
        cfg = store.load()
        for i, m in enumerate(cfg["models"]):
            if m.get("name") == name:
                # 不允许改 name（避免孤儿引用）
                body["name"] = name
                cfg["models"][i] = body
                store.save(cfg)
                loader._app = None
                return jsonify({"ok": True, "model": body})
        return jsonify({"error": f"model {name} not found"}), 404

    @app.route("/api/models/<name>", methods=["DELETE"])
    def api_delete_model(name: str):
        cfg = store.load()
        for i, m in enumerate(cfg["models"]):
            if m.get("name") == name:
                del cfg["models"][i]
                store.save(cfg)
                loader._app = None
                return jsonify({"ok": True, "deleted": name})
        return jsonify({"error": f"model {name} not found"}), 404

    # ----------------------------------------------------------------- #
    #  路由 / 设备 / 定价 单独更新
    # ----------------------------------------------------------------- #
    @app.route("/api/router", methods=["PUT"])
    def api_put_router():
        body = request.get_json(force=True, silent=True) or {}
        cfg = store.load()
        cfg["router"] = body
        store.save(cfg)
        loader._app = None
        return jsonify({"ok": True, "router": body})

    @app.route("/api/device", methods=["PUT"])
    def api_put_device():
        body = request.get_json(force=True, silent=True) or {}
        cfg = store.load()
        cfg["device"] = body
        store.save(cfg)
        loader._app = None
        return jsonify({"ok": True, "device": body})

    @app.route("/api/pricing", methods=["PUT"])
    def api_put_pricing():
        body = request.get_json(force=True, silent=True) or {}
        cfg = store.load()
        cfg["pricing"] = body
        store.save(cfg)
        loader._app = None
        return jsonify({"ok": True, "pricing": body})

    # ----------------------------------------------------------------- #
    #  一键测试
    # ----------------------------------------------------------------- #
    @app.route("/api/test", methods=["POST"])
    def api_test():
        body = request.get_json(force=True, silent=True) or {}
        prompt = body.get("prompt", "").strip()
        prefer_model = body.get("prefer_model") or None
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        try:
            lynsoo = loader.get()
            # 同步调用（可能耗时几秒）
            result = lynsoo.route(prompt, prefer_model=prefer_model) \
                if prefer_model else lynsoo.route(prompt)
            return jsonify({
                "ok": True,
                "result": {
                    "text": result.get("text", ""),
                    "exit_model": result.get("exit_model"),
                    "exit_latency_ms": result.get("exit_latency_ms"),
                    "ttft_ms": result.get("ttft_ms"),
                    "total_ms": result.get("total_ms"),
                    "exited": result.get("exited", False),
                    "hard_route": result.get("hard_route", False),
                    "adapter": result.get("adapter"),
                    "route_label": result.get("route_label"),
                    "route_ms": result.get("route_ms"),
                },
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # ----------------------------------------------------------------- #
    #  上游可达性检查
    # ----------------------------------------------------------------- #
    @app.route("/api/upstream-health", methods=["GET"])
    def api_upstream_health():
        cfg = store.load()
        results = []
        for m in cfg.get("models", []):
            results.append(check_upstream_health(m))
        return jsonify(results)

    # ----------------------------------------------------------------- #
    #  通知服务端热重载
    # ----------------------------------------------------------------- #
    @app.route("/api/reload", methods=["POST"])
    def api_reload():
        # 1. 重建本地 AppLoader（让一键测试用新配置）
        try:
            loader.reload()
        except Exception as e:
            return jsonify({"ok": False, "error": f"local reload failed: {e}"}), 500

        # 2. 通知 server.py 热重载
        server_result = None
        if server_url:
            server_result = notify_server_reload(server_url, server_token)

        return jsonify({"ok": True, "local": True, "server": server_result})

    # ----------------------------------------------------------------- #
    #  错误处理
    # ----------------------------------------------------------------- #
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not found", "path": request.path}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "internal error", "message": str(e)}), 500

    # 把 store / loader 挂到 app 上，便于测试访问
    app.config["STORE"] = store
    app.config["LOADER"] = loader
    return app


# --------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="灵枢 LynSooLLM 配置端（Flask）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m lynsoollm.admin -c config.yaml -p 8001
    python -m lynsoollm.admin -c config.yaml -p 8001 --server-url http://localhost:8000

使用：
    浏览器打开 http://localhost:8001 编辑配置
    保存后自动写入 config.yaml
    点"通知服务端热重载"让 server.py 重新加载
        """,
    )
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="YAML 配置文件路径（默认: config.yaml）")
    parser.add_argument("-p", "--port", type=int, default=8001,
                        help="监听端口（默认: 8001）")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址（默认: 127.0.0.1，仅本机访问）")
    parser.add_argument("--device", default="cpu",
                        help="一键测试时 LynSooApp 运行设备（默认: cpu）")
    parser.add_argument("--server-url", default=None,
                        help="推理服务端 URL，用于热重载通知（如 http://localhost:8000）")
    parser.add_argument("--server-token", default=None,
                        help="推理服务端鉴权 token")
    parser.add_argument("--debug", action="store_true",
                        help="Flask debug 模式（不推荐生产用）")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.stderr.write(f"[warn] 配置文件不存在: {args.config}，将创建空配置\n")
        # 创建空配置
        store = ConfigStore(args.config)
        store.save(store._skeleton())

    print("=" * 72)
    print("  灵枢 LynSooLLM 配置端 (Flask + Vue 3 + TDesign)")
    print("=" * 72)
    print(f"  配置文件: {os.path.abspath(args.config)}")
    print(f"  监听:     http://{args.host}:{args.port}")
    if args.server_url:
        print(f"  服务端:   {args.server_url}")
    print(f"\n  浏览器打开 http://localhost:{args.port} 开始编辑\n")

    app = create_app(
        config_path=args.config,
        device=args.device,
        server_url=args.server_url,
        server_token=args.server_token,
    )
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
