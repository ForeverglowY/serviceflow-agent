# ServiceFlow：智能客服与售后工单 Agent

基于 FastAPI、LangGraph、DeepSeek、PostgreSQL 和 pgvector 实现的电商客服应用。支持订单与物流工具查询、带来源的政策 RAG、多轮售后申请收集、业务规则预评估、用户确认审计和人工待处理任务登记。

这是一个持续开发中的个人项目，使用模拟订单、商品和售后政策。工单登记不代表退款或换货获批；当前尚未实现真实退款执行或人工客服操作界面。

## 当前功能

- 订单、商品和物流信息查询
- 客服工单创建、预览与查询
- LangGraph 意图分流、内层工具循环与多轮工单子图
- PostgreSQL Checkpointer 保存会话状态，按 thread_id 恢复对话
- 售后字段追问、申请修改、取消、摘要与明确确认
- 普通 Python 退换货预评估：订单状态、商品类别、签收日期和专项审核标记
- 工单与确认快照在同一数据库事务中写入
- 人工待处理队列登记与重复登记防重
- 客户消息结构化意图识别
- DeepSeek Tool Calling
- 订单与物流多工具选择和执行
- 同一轮多个工具调用与连续多轮工具调用
- Pydantic 工具参数和结果校验
- 工具白名单与统一工具注册表
- 最大工具轮数限制
- DeepSeek、数据库连接、连接池和 SQL 执行超时
- Markdown 政策加载、章节切分与幂等向量索引
- PostgreSQL + pgvector 语义检索
- 关键词与向量 RRF 混合检索实验
- 带结构化引用、白名单校验和可信来源的 RAG 回答
- Prompt Injection、政策状态、版本和生效日期安全检查
- 20 篇模拟政策、80 个章节片段和 20 条检索评测
- pytest 自动化测试

## 技术栈

- Python 3.13
- FastAPI、Uvicorn
- Pydantic、Pydantic Settings
- LangGraph、PostgreSQL Checkpointer
- DeepSeek V4 Flash（OpenAI 兼容 API）
- SQLAlchemy 2、Psycopg 3
- PostgreSQL 17
- pgvector、Sentence Transformers
- `BAAI/bge-small-zh-v1.5` 中文 Embedding 模型
- Docker Compose
- pytest
- uv

## Agent 工作流程

外层工作流根据用户意图区分业务查询、政策咨询和售后申请；活跃售后会话优先恢复工单收集状态。

```mermaid
flowchart TD
    U[客户消息] --> E{是否已有活跃售后申请}
    E -->|是| T[恢复工单子图]
    E -->|否| I[意图分类]
    I --> Q[订单与物流工具 Agent]
    I --> R[政策 RAG]
    I --> T
    I --> F[安全回退与需求引导]
    T --> C[收集与合并申请字段]
    C --> M{字段是否齐全}
    M -->|否| A[追问缺失字段]
    M -->|是| B[读取业务事实并执行预评估规则]
    B --> S[展示摘要等待用户确认]
    S -->|修改| C
    S -->|取消| X[结束申请]
    S -->|确认| W[同一事务保存工单与确认快照]
    W --> H{是否需要人工核验}
    H -->|是| P[登记人工待处理任务]
    H -->|否| D[返回工单信息]
```

业务查询分支内部的工具循环：

```mermaid
flowchart TD
    U["客户消息"] --> L["DeepSeek 判断下一步"]
    L -->|"无需业务数据"| A["生成最终回答"]
    L -->|"需要真实数据"| C["返回 Tool Call"]
    C --> V["Python 校验调用类型、工具名和参数"]
    V --> R["工具注册表选择处理器"]
    R --> D["SQLAlchemy 查询 PostgreSQL"]
    D --> S["Pydantic 转换并序列化工具结果"]
    S --> M["以 role=tool 追加到消息历史"]
    M --> L
```

大模型只负责理解问题、选择工具和组织回答。工具参数校验、数据库访问、状态映射、总价计算、超时、错误处理和循环限制均由 Python 后端负责。

## RAG 工作流程

```mermaid
flowchart TD
    Q["客户政策问题"] --> E["Embedding 查询向量"]
    E --> F["状态、分类、商品和生效日期过滤"]
    F --> V["pgvector 相似度检索"]
    V --> C["构造带边界的政策上下文"]
    C --> L["DeepSeek 生成 answer 与 cited_chunk_ids"]
    L --> P["Pydantic 校验结构"]
    P --> W["Python 引用白名单校验"]
    W --> S["Python 根据真实片段生成来源"]
    S --> A["最终可引用回答"]
```

知识文档始终被视为不可信数据。模型不能直接决定政策是否生效，也不能自行生成可信来源；政策过滤、引用范围和来源格式均由 Python 确定性执行。

## 项目结构

```text
serviceflow/
├── api.py              FastAPI 路由和 HTTP 异常转换
├── settings.py         环境变量配置
├── models.py           Pydantic 业务模型
├── data_loader.py      本地 JSON 初始化数据加载
├── db/                 PostgreSQL Engine 和 SQLAlchemy ORM 模型
├── llm/                DeepSeek 客户端、Tool Schema 和结构化模型
├── agent/              Agent 多轮编排、业务工具和工具注册表
├── ticket/             业务规则、工单与确认写入、人工待处理登记
└── rag/                文档、索引、检索、融合和可信回答生成
scripts/
├── seed_data.py        幂等测试数据初始化
├── evaluate_retrieval.py  检索评测入口
└── demos/              分阶段学习和手工验证程序
data/                   订单、商品和物流的本地初始化数据
knowledge/policies/     20 篇模拟售后政策
evaluation/             检索、回答和安全验收资料
tests/                  自动化测试
migrations/             已有数据库的增量 SQL 迁移
compose.yaml            PostgreSQL 本地开发环境
```

`serviceflow/` 是正式应用包；`scripts/` 是命令行入口和演示代码，不属于 FastAPI 的正式调用链。

## 本地运行

### 1. 准备环境

需要安装：

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop 或其他兼容 Docker Compose 的容器运行时

安装项目依赖：

```bash
uv sync
```

### 2. 配置 DeepSeek

复制配置模板：

```bash
cp .env.example .env
```

在 `.env` 中填写真实密钥：

```dotenv
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

DEEPSEEK_TIMEOUT_SECONDS=30
DEEPSEEK_MAX_RETRIES=2

DATABASE_CONNECT_TIMEOUT_SECONDS=5
DATABASE_POOL_TIMEOUT_SECONDS=5
DATABASE_STATEMENT_TIMEOUT_MS=5000
CHECKPOINT_DATABASE_URL=postgresql://agent:agent_password@localhost:5432/agent_study
```

`.env` 已被 Git 忽略，不要提交或公开真实 API Key。

### 3. 启动 PostgreSQL

```bash
docker compose up -d
docker compose ps
```

本地开发数据库配置：

```text
Host: localhost
Port: 5432
Database: agent_study
User: agent
Password: agent_password
```

### 4. 建表和初始化数据

```bash
uv run python -m serviceflow.db.database
uv run python -m scripts.seed_data
uv run python -m serviceflow.rag.indexer
```

初始化程序具有幂等性，重复执行不会重复插入已有订单、商品、物流和未变化的政策片段。

首次建库由 ORM 创建当前业务表；API 启动时会通过 `PostgresSaver.setup()` 创建 checkpoint 表。已有旧数据库需要按文件名顺序执行 `migrations/` 下的 SQL，`create_all()` 不会修改已有表结构。

### 5. 启动 API

```bash
uv run uvicorn serviceflow.api:app --reload
```

服务地址：

- API：<http://127.0.0.1:8000>
- Swagger：<http://127.0.0.1:8000/docs>
- OpenAPI JSON：<http://127.0.0.1:8000/openapi.json>

## API 接口

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/health` | 服务健康检查 |
| `GET` | `/orders/{order_id}` | 查询订单 |
| `GET` | `/products/{product_id}` | 查询商品 |
| `GET` | `/logistics/{order_id}` | 查询物流 |
| `POST` | `/tickets/preview` | 校验并预览工单 |
| `POST` | `/tickets` | 创建工单 |
| `GET` | `/tickets/{ticket_id}` | 查询工单 |
| `POST` | `/agent/intent` | 识别客户消息意图 |
| `POST` | `/agent/chat` | 与智能客服 Agent 对话 |

Agent 请求示例：

```json
{
  "message": "帮我查询订单20260721001，并看看快递到哪里了",
  "thread_id": null
}
```

多轮售后示例：第一轮发送“订单20260721001的耳机左耳没有声音，我想换货”，接口返回确认摘要和 `thread_id`。第二轮发送“确认”并传入同一个 `thread_id`，系统保存工单和确认快照，并在需要核验时登记人工待处理任务。仅咨询“可以退货吗”会进入政策问答，明确要求办理售后才进入工单流程。

## 测试

运行全部测试：

```bash
uv run python -m pytest -q
```

当前基线：

```text
236 passed
```

测试基线验证于 2026-10-03。单元与工作流测试通过 Mock 隔离真实 DeepSeek 和数据库调用；部分 API 集成测试需要已建表并初始化数据的 PostgreSQL。GitHub Actions 自动创建独立测试数据库。Swagger 联调另行验证，不把模拟测试结果视为生产审批效果。

检索评测：

```bash
uv run python -m scripts.evaluate_retrieval
```

当前知识库扩容基线为20篇政策、80个章节片段和20条问题，`Hit@5 = 19/20（95.00%）`。唯一失败是需要同时召回两类政策的多方面问题，相关条款位于第6名；该结果保留为后续查询拆分、多样化召回或 Reranker 的改进基线。

## 安全与可靠性

- API Key 通过环境配置注入，不写入源码或 Git
- 工具名称必须存在于显式注册表中
- 工具参数使用 Pydantic 校验并禁止额外字段
- 工具调用仅支持 SDK 的 Function Tool 类型
- 工具结果由 Python 转换成明确的业务语义
- Agent 最多执行 3 轮工具，避免无限循环和费用失控
- DeepSeek 请求具有超时和有限重试
- PostgreSQL 配置连接、连接池等待和语句执行超时
- 内部异常统一转换，对外返回稳定且不泄露实现细节的错误信息
- 日志不记录 API Key 和完整工具参数
- RAG政策上下文使用明确边界，并声明知识内容属于不可信数据
- 模型回答使用Pydantic结构化校验并禁止额外字段
- 模型引用编号必须属于本次检索结果白名单
- 最终引用名称、章节和来源由Python根据真实片段生成
- 只检索状态有效且已经生效的政策
- 索引同步跟踪正文、模型、版本、日期、状态和过滤元数据变化

## 当前边界与后续计划

- 创建工单的提交幂等键尚在开发中；人工任务登记防重不等于工单创建防重
- 工单写入和人工任务登记为两个事务，跨步骤失败恢复尚未完成
- 用户身份目前使用 USER-DEMO，尚无登录鉴权、人工领取或审批界面
- 售后规则是保守预评估，未知的拆封和质检结果需要人工核验
- 将混合检索或 Reranker 接入正式 RAG 流程
- 增加查询拆分和结果多样化，改善多政策问题召回
- 商品查询、退款资格检查和工单处理工具
- 自动评测集、调用链追踪和成本统计
- Docker Compose 一键启动完整应用
