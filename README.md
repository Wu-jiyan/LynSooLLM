# 灵枢 LynSooLLM

> 边缘感知与多出口推测式端云协同 LLM 智能路由引擎

LynSooLLM 是一个**统一入口的智能路由中转站**：用户用任意标准 SDK（OpenAI / Anthropic / Gemini）接入，对外只看到一个模型 `lynsoo-auto`，内部由路由模型决定走本地小模型还是上云，并在本地输出熵超阈值时无缝接力到云端大模型，让用户感知不到内部路由。

```
用户 (任意 SDK)
   │  统一模型名 lynsoo-auto
   ▼
┌─────────────────────────────────────────────┐
│           LynSooLLM 推理服务端              │
│  ┌──────────────────────────────────────┐  │
│  │  路由模型 (Gemma-3-270M / Qwen3.5-0.8B)│  │
│  │  + Multi-LoRA (弱网/低电量/...)        │  │
│  └──────────────┬───────────────────────┘  │
│                 │ entropy > threshold       │
│       ┌─────────┴─────────┐                │
│       ▼                   ▼                │
│   本地继续            云端接力              │
│   (低延迟)        (assistant prefill)       │
└─────────────────────────────────────────────┘
   │
   ▼
云端模型池 (OpenAI / Anthropic / Gemini / DeepSeek / ...)
```

## 核心特性

- **统一入口**：对外只暴露 `lynsoo-auto`，用户无需改 SDK
- **4 种原生协议**：`openai_chat` / `openai_responses` / `anthropic_messages` / `gemini_generate`
- **推测式早期退出**：本地小模型流式生成时监测 token 级信息熵，超阈值立即接力
- **assistant 预填**：本地部分输出作为 assistant 消息传给云端，不污染用户 prompt
- **边缘感知**：根据网络 RTT / 电量 / 温度 / 离线状态动态选 LoRA adapter
- **多源定价**：`models.dev` / `official` / `custom_api` / `manual` 四种来源
- **自动权重**：根据定价 + 设备上下文 + 策略自动算 priority / weight
- **5 种策略**：`cost_first` / `quality_first` / `balanced` / `latency_first` / `manual`
- **零依赖服务端**：标准库 `http.server`，无需 FastAPI / uvicorn
- **图形化配置端**：Flask + Vue 3 + TDesign，纯图形化配置（无 YAML 模式）

## 架构

| 模块 | 端口 | 说明 |
|---|---|---|
| `server.py` | 8000 | 推理服务端（标准库 HTTP，4 种协议端点 + SSE 流式） |
| `admin.py`  | 8001 | 配置端（Flask + Vue 3 + TDesign 图形化配置） |

两者通过共享 `config.yaml` 通信：admin 保存配置 → 写文件 → 通知 server 热重载。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`torch` / `transformers` / `peft` / `accelerate` / `pyyaml` / `flask`

### 2. 配置

复制示例配置并填入 API key：

```bash
cp config.example.yaml config.yaml
export OPENAI_API_KEY=sk-your-real-key-here
# 或直接编辑 config.yaml 把 ${OPENAI_API_KEY} 替换成真实 key
```

`config.yaml` 已加入 `.gitignore`，不会入库。

### 3. 启动配置端（图形化修改配置）

```bash
python -m lynsoollm.admin -c config.yaml -p 8001
# 浏览器打开 http://localhost:8001
```

### 4. 启动推理服务端（对外提供 API）

```bash
python -m lynsoollm.server -c config.yaml -p 8000
```

### 5. 用任意 SDK 接入

模型名固定为 `lynsoo-auto`，endpoint 指向 `http://localhost:8000`。

**OpenAI SDK**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")
resp = client.chat.completions.create(
    model="lynsoo-auto",
    messages=[{"role": "user", "content": "讲个笑话"}],
)
print(resp.choices[0].message.content)
```

**Anthropic SDK**

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="any")
resp = client.messages.create(
    model="lynsoo-auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "讲个笑话"}],
)
print(resp.content[0].text)
```

**Google GenAI SDK (Gemini)**

```python
from google import genai
client = genai.Client(api_key="any", http_options={"base_url": "http://localhost:8000"})
resp = client.models.generate_content(model="lynsoo-auto", contents="讲个笑话")
print(resp.text)
```

## 配置说明

```yaml
models:                      # 云端模型池
  - name: qwen3.6-27b
    provider: openai         # openai / anthropic / gemini / deepseek / custom
    endpoint: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model_id: qwen3.6-27b
    protocol: auto           # auto / openai_chat / openai_responses /
                             #       anthropic_messages / gemini_generate
    quality_tier: 3          # 0-5，质量分级
    pricing_source: manual   # models_dev / official / custom_api / manual
    manual_pricing:          # pricing_source=manual 时使用
      input_per_1k: 0.0006
      output_per_1k: 0.003
    priority: 100
    weight: 0
    enabled: true

router:                      # 路由模型配置
  backbone: gemma            # gemma (270M) / qwen (0.8B)
  strategy: cost_first       # cost_first / quality_first / balanced /
                             #       latency_first / manual
  entropy_threshold: 1.5     # 触发接力的熵阈值
  max_new_tokens: 32         # 本地最大生成 token 数
  min_tokens_before_exit: 1  # 最少生成多少 token 才允许退出
  dynamic_threshold: true    # 动态调整阈值
  lora_adapters:             # Multi-LoRA adapter 列表
    - name: weak_net
      path: ./checkpoints/weak_net
    - name: low_battery
      path: ./checkpoints/low_battery

device:                      # 设备上下文（影响 adapter 选择 + priority 计算）
  network_rtt_ms: 50
  battery_pct: 80
  temperature_c: 30
  cloud_price_per_1k: 0.01
  offline: false

pricing:                     # 定价源全局配置
  cache_path: ~/.cache/lynsoollm/models_dev.json
  cache_ttl_sec: 86400
  zero_price_fallback: 0.01
```

## 项目结构

```
lynsoollm/
├── lynsoollm/                  # 主包
│   ├── __init__.py
│   ├── app.py                  # 成品入口 LynSooApp
│   ├── server.py               # 推理服务端（标准库 HTTP）
│   ├── admin.py                # 配置端（Flask + Vue 3）
│   ├── config.py               # YAML 配置加载
│   ├── router.py               # SpeculativeRouter（原型）
│   ├── real_router_engine.py   # 真实路由引擎
│   ├── real_router_model.py    # Gemma / Qwen 路由模型封装
│   ├── multi_lora.py           # Multi-LoRA 动态切换
│   ├── adapter_selector.py     # 环境感知 adapter 选择器
│   ├── model_registry.py       # 云端模型注册表 + 优先级计算
│   ├── cloud_relay_pool.py     # 云端接力池（加权选择 + 接力执行）
│   ├── executors.py            # 4 种协议执行器 + AutoExecutor
│   ├── exit_signal.py          # Early-Exit 信号 + RelayContext
│   ├── entropy.py              # token 级信息熵
│   ├── pricing.py              # 4 源定价拉取
│   ├── mock_local_model.py     # 本地模型 Mock（用于流程验证）
│   ├── train_router.py         # 路由模型训练脚本
│   ├── templates/admin.html    # 配置端前端（Vue 3 单页）
│   └── static/                 # 前端依赖（本地化）
│       ├── vue.global.prod.js
│       ├── tdesign.min.js
│       ├── tdesign.min.css
│       └── js-yaml.min.js
├── tests/                      # 单元测试（51 个用例）
│   ├── test_admin.py
│   ├── test_router.py
│   └── test_server.py
├── config.example.yaml         # 示例配置（脱敏）
├── requirements.txt
└── .gitignore
```

## API 端点

### 推理服务端（端口 8000）

| 路径 | 方法 | 说明 |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI Chat 兼容 |
| `/v1/messages` | POST | Anthropic Messages 兼容 |
| `/v1beta/models/{model}:generateContent` | POST | Gemini 兼容 |
| `/v1beta/models/{model}:streamGenerateContent` | POST | Gemini SSE 流式 |
| `/health` | GET | 健康检查 |
| `/admin/reload` | POST | 热重载配置 |

### 配置端（端口 8001）

| 路径 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 配置页面（Vue 单页） |
| `/api/config` | GET / PUT | 读 / 写完整配置 |
| `/api/models` | GET / POST | 列出 / 新增模型 |
| `/api/models/<name>` | PUT / DELETE | 修改 / 删除模型 |
| `/api/router` | PUT | 修改路由配置 |
| `/api/device` | PUT | 修改设备上下文 |
| `/api/pricing` | PUT | 修改定价源 |
| `/api/test` | POST | 一键测试路由 |
| `/api/upstream-health` | GET | 上游可达性检查 |
| `/api/reload` | POST | 通知 server.py 热重载 |
| `/api/meta` | GET | 元信息 |

## 测试

```bash
cd lynsoollm
python -m pytest tests/ -v
```

51 个用例覆盖：路由引擎、Early-Exit 信号、RelayContext、4 种 executor 协议、admin API、server API。

## 工作流程

1. **硬路由决策**：路由模型对 prompt 打分，决定走本地还是直接上云
2. **本地流式生成**：若走本地，边生成边计算 token 级信息熵
3. **熵阈值检测**：熵超阈值 → 立即停止本地生成 → 构造 RelayContext
4. **云端接力**：把 `{user: prompt, assistant: 本地部分输出}` 发给选中的云端模型
5. **无缝接续**：云端从 assistant 预填处接续生成，返回完整文本

## License

MIT
