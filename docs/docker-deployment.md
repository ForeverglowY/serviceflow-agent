# Docker部署（Day24，现有库及独立空库启动均已验证）

## 2026-10-19 当前验证结果

- `serviceflow-agent:local`镜像构建成功；容器uid=10001，torch=2.14.1+cpu且cuda=None，/app/.env不存在。
- 容器运行时的业务库与checkpoint库主机均为postgres。现有PostgreSQL容器没有重建、数据库卷没有删除，没有执行数据初始化。
- 启动api后首次下载Embedding并完成71个权重加载，容器状态healthy。
- `http://127.0.0.1:8001/health`返回200及status=ok，含X-Trace-ID。
- 8001端口的订单接口成功读回20260721001，Swagger页面渲染出接口列表。
- 初次只读检查未调用付费聊天模型；随后用户在8001运行真实HTTP评测，4场景全部REVIEW，对照回答及checkpoint复核均符合本次要求，无新增工单。确认容器内DeepSeek调用成功；独立空库验收见文末。
- 最近本地全量测试312 passed；单测不等于容器业务端到端验收。

## 保留现有开发方式

`docker compose up -d` 仍只启动postgres。API使用app profile，显式启用才启动。
现有数据库卷名、服务名和端口保持不变；本步骤不删除卷、不自动执行旧库ALTER迁移。

## 前置条件

- Docker运行，根目录.env已有真实DeepSeek配置但没有提交Git。
- 现有数据库已按README完成迁移。新数据库可使用下面初始化命令创建当前表结构。
- 首次构建会下载Python依赖（包括PyTorch），可能耗时且镜像较大；容器不能使用Mac的.venv。
- Linux的torch通过uv显式选择官方CPU索引，避免默认CUDA依赖；Mac仍使用默认包源，两个平台的具体锁定版本可能不同。
- 容器第一次加载Embedding会下载模型，需能访问模型源；缓存保留在serviceflow-model-cache卷，构建成功不等于下载模型成功。

## 构建与准备

```bash
docker compose build api
docker compose up -d postgres
docker compose run --rm api python -m scripts.init_container_data
docker compose run --rm api python -m serviceflow.rag.indexer
```

初始化/索引会写入演示数据库，不删除已有业务数据；旧库缺字段时先执行对应迁移，不使用create_all代替ALTER。
没有必要在每次启动API时重复索引，避免把慢任务藏进启动过程。

## 启动与验收

```bash
docker compose --profile app up -d api
docker compose --profile app ps
docker compose logs -f api
```

本机Swagger：http://127.0.0.1:8001/docs 。8001避免与现有本机8000开发服务冲突。

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/orders/20260721001
uv run python -m scripts.evaluate_http --run --port 8001
```

先验证健康、订单和RAG模型加载，再运行HTTP评测（会产生模型调用费用和checkpoint写入）。
评测脚本在Mac运行，使用localhost数据库；容器API使用postgres服务名，二者必须确实连接同一库。
未验证全新空库启动前，不宣称可在全新机器一键部署。

## 原理

- Dockerfile定义API镜像的Python、依赖、代码和启动命令；uv按锁文件安装，排除dev组。
- 容器中的localhost指容器自己，所以数据库主机改为postgres，而不是localhost。
- Compose的environment覆盖env_file里相同变量；业务数据库URL和checkpoint URL都要覆盖。
- Uvicorn在容器内监听0.0.0.0；宿主只绑定127.0.0.1:8001，不向外网公开未鉴权接口。
- depends_on等待postgres健康；API启动会连接checkpoint库并加载Embedding，可能较慢。
- /health证明API启动并可响应，不代表数据库始终健康或外部模型随时可用，healthcheck不是完整业务验收。
- 密钥运行时注入，不COPY进镜像；不对外分享docker compose config或docker inspect完整输出，可能含密钥。
- 镜像使用非root用户；模型命名卷保留缓存。镜像和基础标签仍非摘要固定，不是完全字节级可复现构建。

停止API使用 `docker compose stop api`，不影响现有数据库。不要运行带-v的down来清理：会删除数据库及模型缓存卷。
当前仅本地演示部署，没有生产鉴权、TLS或高可用配置。

## 独立空库验收

compose.fresh.yaml是独立配置，固定搭配项目名serviceflow-fresh使用。数据库没有宿主端口，API为8002；卷fresh-postgres-data与fresh-model-cache都由该项目独立创建。
不使用原来的container_name，不复用日常数据库或模型缓存卷。仅复用已构建的API镜像和本机.env的模型凭据。

```bash
docker compose -p serviceflow-fresh -f compose.fresh.yaml up -d --wait postgres
docker compose -p serviceflow-fresh -f compose.fresh.yaml exec -T postgres psql -U agent -d agent_study -c "SELECT count(*) FROM pg_tables WHERE schemaname='public';"
docker compose -p serviceflow-fresh -f compose.fresh.yaml run --rm --no-deps api python -m scripts.init_container_data
docker compose -p serviceflow-fresh -f compose.fresh.yaml run --rm --no-deps api python -m serviceflow.rag.indexer
docker compose -p serviceflow-fresh -f compose.fresh.yaml up -d --wait api
```

只有首次使用全新卷时表数量才为0；重复使用已存在的验收卷不算再次证明空库部署。
2026-10-19实际初始public表数量为0，初始化商品/订单/物流各20条、索引80条且均512维。重复初始化新增各0，重复索引跳过80。
随后测试API在8002健康，订单接口正确读回20260721001，Swagger浏览器页面展示成功；独立模型缓存加载成功，向量检索召回POL-RETURN-003-003等三条耳机政策。API首次启动创建四张checkpoint相关表。
验收后已停止serviceflow-fresh的两个服务，保留测试数据卷及模型缓存卷，不影响日常8001服务。全量测试313 passed。空库验收未再次调用付费聊天模型；真实模型业务链路依据之前8001的HTTP验收。
这证明当前镜像在新的Linux容器与空库上可初始化，不代表完全没有网络、没有Docker或没有凭据的机器无需准备即可运行。

验收后停止测试环境保留证据和缓存，不删除卷：

```bash
docker compose -p serviceflow-fresh -f compose.fresh.yaml stop
```
