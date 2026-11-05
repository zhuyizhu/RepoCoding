# 实际验收证据

本文件按时间保留验收记录；早期“未验收”描述只对应当时状态。当前结论见末尾“2026-11-06 自动链路实机结果”：带类名线索的完整链路运行成功，原始基准描述未被独立修复。

## 早期基线记录

- pytest：24 passed，1 个 Starlette/anyio 弃用警告，报告 test-results.xml。
- 其中包括 15 个未经改写的 EviState 原核心测试；另有 API/SSE/outbox、真实临时 Git worktree、实际核心 handoff/QZ/数据库绑定映射、修复文件工具边界及默认禁用测试。
- API/数据库用例使用 SQLite 内存库，不能算作 PostgreSQL 集成验证。
- EviState 测试实际调用复制的原核心，使用明确的临时 fixture，不是模型推理或 benchmark 成绩。
- npm run build：TypeScript 与 Vite 通过，32 个模块，已生成 dist。
- 真实 Dubbo checkout：固定基础提交，git status 干净。
- 平台与研究依赖曾完成 pip dry-run 解析；后续实际安装及导入状态另见 import-check.json（若不存在则未验证）。dependency-resolution.json 是初始解析证据，最终要求以 research-requirements.txt 为准。
- Docker Compose、PostgreSQL/Redis/Celery 跨进程链路、CodeLens 推理、Qwen Tool Calling、Maven/Gradle：未进行实机验收。

不能将以上测试汇总成“完整 Agent 已运行成功”。后续应保存一次真实任务的事件、原始候选、EviState 快照、补丁及测试退出码再验收。

## 2026-11-01 最小版实机检查

- 研究依赖实际安装成功；离线导入 `agent.agents.pipeline.LocalizationPipeline` 成功，torch 2.8.0+cpu、faiss 1.12.0、langchain_core 0.3.79。未调用推理。
- 本机启动 FastAPI（端口 8000）和 Vite（端口 5173），HTTP `/health` 返回 200，前端入口返回 200。
- 向 `/api/demo` 提交真实 apache__dubbo-11781 描述和已存在的 URL.java 路径，返回 201。SQLite 持久化结果为 2 个假设、1 条 EXISTS 证据、2 条绑定，Q/Z 同源且分开保存。
- 浏览器点击“运行本地示范”后，页面显示 FILE 存在为 SUPPORTED、FAULT_RELEVANT 为 UNKNOWN、修复与测试未执行，并显示 9 条持久化事件。
- 最小版不运行 CodeLens、Celery、PostgreSQL、Qwen 或 Java 测试；其运行结果不能代表完整 Agent 验收。

## 2026-11-01 PostgreSQL 切换实测

- Docker Desktop 4.97.0、Engine 29.9.2、Compose v5.6.1，Linux containers。
- `docker compose ps`：postgres:16 与 redis:7-alpine 均 healthy；Redis `PING` 返回 `PONG`。
- Alembic `0001_initial` 在 PostgreSQL 成功执行。
- 本机 API `/health` 返回 `postgresql ready`；提交真实本地文件检查任务返回 succeeded，含 2 个假设、1 条证据和 Q/Z 快照；容器中 `select count(*) from tasks` 返回 1。
- 旧 `local-demo.db` 文件保留，历史 SQLite 数据未迁移。Redis/Celery 的跨进程任务链路、CodeLens 模型推理及 Java 容器测试仍未验收。

## 2026-11-02 Qwen 7B 最小修复实测

- 从 Windows 实际请求当时配置的超算端口转发地址 `/v1/models`，返回 `qwen2.5-coder-7b-instruct`；`/v1/chat/completions` 收到真实模型回复。端口转发地址可能变化，重启超算容器后需重新确认。
- 页面样例文件从 `URL.java` 更正为真正含缺陷分支的 `URLStrParser.java`。未修改研究源码或原始 Dubbo checkout。
- `/api/demo` 的 `repair_with_model=true` 实际调用 vLLM，任务 `54f25605-3a59-4918-aaa9-17738aae4a5f` 在 PostgreSQL 中为 `succeeded`：EviState 两个假设、一条文件存在证据、Q/Z 分开保存、11 条事件、一条真实 Git Diff。Diff 仅将两个 `value = name;` 改为 `value = "";`，保存于该任务的隔离 worktree。原始 Dubbo 仓库 `git status --porcelain` 为空。
- 初次模型提案包含无关改动或不唯一的旧文本，程序拒绝写入；收紧为一个方法中的单条替换规则后成功。失败尝试的三个 worktree 已清理。Windows Git worktree 需 `core.longpaths=true`，代码已加。
- `git -c core.longpaths=true diff --check` 通过；`backend/tests` 9 passed；`npm run build` 通过；API `/health` 显示 `postgresql ready`，页面 HTTP 200。实际任务无 `TestResult`，测试阶段 `skipped`，补丁记录 `validated=false`。
- 本机 `mvnw.cmd -v` 因无 `JAVA_HOME` 失败，未声称 Maven/Gradle 验证成功。CodeLens 的完整向量检索依赖本机未缓存的 `BAAI/bge-m3`；试验性的截断图输入曾输出不可靠路径，未接入页面、未作为定位结果保存。Celery 队列也未运行。

## 2026-11-04 容器部署实测

- 页面打不开时，旧 `8000`/`5173` 进程已退出，Docker Desktop Linux 引擎未启动。启动 Docker Desktop 后原数据库卷恢复；本机启动脚本的等待时间从 2 秒改为按端口最多等待 30 秒。
- 新 `compose.deploy.yml` 的 FastAPI 与 React/Nginx 镜像实际构建成功。`repo-seed` 从只读原 Dubbo 仓库克隆固定提交 `d0a1bd014331483c19208b831c4f6b488654a508` 到命名卷；PostgreSQL、API、Web 均 healthy。原 PostgreSQL 命名卷的 6 条历史任务仍能查询。
- 停止本机 Vite/FastAPI 后，容器独立入口 `http://127.0.0.1:8080` 返回 HTTP 200，`/health` 返回 `postgresql ready`。容器内部实际访问用户 vLLM `/v1/models`，返回 `qwen2.5-coder-7b-instruct`。
- 经 Nginx `/api/demo` 提交的公开 Dubbo 样例任务 `2aff04fb-6642-4822-9e1a-596ee036cec3` 成功：2 个候选假设、1 条证据、Q/Z 均存在、11 条事件、1 条 Qwen 生成 Diff；两处 `value = name;` 改为 `value = "";`。容器内原仓库副本 `git status --porcelain` 为空，补丁 `git diff --check` 通过。Java 测试为 0，补丁 `validated=false`。
- `compose.public.yml` 的域名 HTTPS/密码入口是配置模板；实际域名、DNS 和证书签发尚未提供或验收。完整 CodeLens、Celery 和 Java 测试仍不属于当前容器链路。
- Windows `scripts/start-deploy.ps1` 实际完成重新构建、启动和健康轮询，输出 `Ready: http://127.0.0.1:8080`。关闭旧本机 Vite/FastAPI 后容器页面仍返回 200。研究流水线 `/api/tasks` 在部署模式下对合法请求返回 503，不会制造无人处理的排队任务。
- 可选公网配置：使用测试密码哈希执行 Caddy `adapt --validate` 成功；`compose.deploy.yml + compose.public.yml` 的 `config --quiet` 成功。没有用户域名，因此没有启动公网入口或声称证书签发成功。
- 最新改动后再次运行 `backend/tests`：9 passed；`npm run build`：TypeScript 与 Vite 通过。

## 2026-11-04 Java 沙盒修复闭环

- Compose 服务 `postgres`、`redis`、`api`、`validation-worker`、`java-sandbox`、`web` 均启动；`postgres`、`redis`、`api`、`java-sandbox`、`web` 健康检查通过。Celery worker 日志显示已连接 Redis，并注册 `app.validation_worker.validate_patch`。
- API 容器请求用户的 Qwen vLLM `/v1/models` 得到 HTTP 200，模型名 `qwen2.5-coder-7b-instruct`。没有记录或输出模型密钥。
- 一次性 `java-deps` 容器在线准备 Maven 缓存，固定基础提交的 `URLStrParserTest` 2 项通过；正式 `java-sandbox` 处于 internal Docker network，使用 `mvn -o` 在临时副本中执行，不修改挂载的 worktree。
- 通过容器 API 提交真实样例任务 `82f0657f-1c60-4f79-a245-d8a69feb4807`。Qwen 在隔离工作树把 `URLStrParser.java` 中两处 `value = name;` 改为 `value = "";`。Celery 随后运行 `mvn -o -B -Dmaven.repo.local=/cache/m2 -pl dubbo-common -am -Dtest=URLStrParserTest -Dsurefire.failIfNoSpecifiedTests=false test`；`TestResult` 显示 2 项测试通过、退出码 0，任务状态 `succeeded`，补丁 `validated=true`。基础仓库副本 `git status --porcelain` 无输出。
- 原有 2 项测试在修改前也通过，不能单独证明缺陷修好。新增独立回归用例 `sandbox/regression/URLStrParserEmptyValueTest.java`，分别检查解码和编码 URL 的 `key_name=`。同一用例在原版上 2 项失败（实际值 `key_name`），在补丁上 2 项通过；测试只写入一次性容器的临时副本，未改原仓库或保存的补丁。回归用例尚未接入自动任务测试报告。
- 容器内 `PYTHONPATH=/app FULL_PIPELINE_ENABLED=true python -m pytest -p no:cacheprovider -q /app/tests`：9 passed，1 个 Starlette/anyio 弃用警告；测试容器额外挂载 EviState 核心到 `/runtime/evistate`。前端生产构建与 Compose 配置校验通过。
- 重建 API 容器后曾发现 Nginx 固定解析旧容器地址导致 `/health` 返回 502。已改为通过 Docker 内置 DNS 动态解析；重新构建 Web 后，强制重建 API，再从 Web 容器请求 `/health` 得到 HTTP 200 和 `postgresql ready`，React 首页也可读取。
- 未验收自动 CodeLens 定位、新仓库建图及公网证书/域名。当前可称为“人工指定文件的 Qwen 修复与沙盒测试 Demo”，不能称为完整自主定位 Agent 或公网生产服务。

## 2026-11-05 CodeLens 恢复检查（进行中）

- Docker Desktop 中此前中断的 GPU 研究镜像仍存在；BGE-M3 模型缓存命名卷约 3.4 GB。恢复前没有容器在运行。
- 本机向配置的 vLLM `/v1/models` 发起只读请求，返回 `qwen2.5-coder-7b-instruct`。这尚不能证明容器内网络或完整推理可用。
- `docker compose -f compose.deploy.yml -f compose.gpu.yml config --quiet` 通过；前端 `npm run build` 通过；新增的 PowerShell 脚本语法和 Python 包装代码 AST 检查通过。
- 新增 CUDA 前置检查：GPU 模式不可用时拒绝隐式退回 CPU；本地单元级检查证实 `CODELENS_REQUIRE_CUDA=true` 与 CPU 设备冲突会被拒绝。
- GPU 索引目录出现 `index.faiss`（209149997 字节）、`index.pkl`（30002957 字节）和包含 `apache__dubbo` 的 `manifest.json`；目录名与当前图目录及 BGE-M3 模型计算值一致。尚待运行时实际加载与端到端任务检验。
- 当前 Codex 终端连接 Docker Engine 命名管道被权限拒绝；用户已在自己的 PowerShell 尝试启动部署。Windows 本机只有 Python 3.13，而旧 `.testdeps` 的 `pydantic_core` 是 Python 3.12 扩展，因此本轮本机 pytest 在收集阶段失败，不能记为测试通过或业务失败。此前容器内 9 项后端测试通过的记录仍在上节。

## 2026-11-06 镜像拉取阻塞与离线启动准备

- 用户执行 `start-deploy.ps1 -Gpu` 时，Docker Hub 的 `auth.docker.io/token` IPv6 连接超时；`python:3.12-slim`、`nginx:stable-alpine`、`node:24-alpine`、`gradle:8.11.1-jdk17` 元数据未能下载，Compose 在构建阶段退出，服务未启动。
- 新增 `compose.offline.yml` 和 `start-deploy.ps1 -Gpu -Offline`：本地已有镜像时，禁用镜像构建和拉取，挂载当前后端代码及前端构建产物。三份 Compose 文件的配置检查及 PowerShell 语法检查通过。
- 用户运行离线命令后，9 个 Compose 服务启动，API、PostgreSQL、Redis、Java 沙盒健康；`http://127.0.0.1:8080/health` 返回 `status=ok` 与 `postgresql ready`。
- 通过 `/api/tasks` 提交未指定 `file_path` 的真实 Dubbo 样例，任务 `7123d8c5-febc-40d6-821d-041e1861fdf2`。CodeLens 工具调用成功，报告 2 个候选、3 条检索记录和策略轨迹；EviState 工具调用成功并保留 Q/Z 状态。候选为 `src/main/java/org/apache/dubbo/common/url/component/URLParam.java` 与 `src/main/java/org/apache/dubbo/common/URL.java`。这是模型实际输出，不能把它们改称为 `URLStrParser.java`。
- 初次任务在修复阶段阻塞，原因是两个候选缺少 Dubbo 模块前缀 `dubbo-common/`，原有精确路径检查找不到文件。新增仅在仓库内后缀唯一匹配时的路径映射，保留 `native_file_path` 和映射事件；本机合成目录的唯一/歧义/越界检查及真实 Dubbo 两个候选的唯一匹配检查通过。后续重启和复验结果见下节。

## 2026-11-06 自动链路实机结果

- 原始问题重跑 `5bb11547-b48f-42c0-bfba-3f175ad8a98a`：CodeLens 自动生成候选，`candidate_paths_resolved` 事件记录 4 条唯一仓库路径映射，EviState 进入修复阶段。首位是 `URLParam.java`（37,105 字节），超过原先 30,000 字符修复上下文上限，任务阻塞。该上限已调整到 50,000 字符。
- 再次原始问题运行 `24baeb96-349a-4ed3-a992-83e93736878d`：CodeLens 仍选 `URLParam.java`，Qwen 判断给出的文件没有描述中的赋值，返回无变化替换；校验拒绝，任务在修复阶段失败。没有补丁或 Java 测试结果；不能算原始问题解决。
- 诊断任务 `65f361b2-7058-4eac-aa89-201c0ae5cf95`：问题描述额外指出真实类名 `org.apache.dubbo.common.URLStrParser`，但没有提交文件路径。CodeLens 把 `dubbo-common/src/main/java/org/apache/dubbo/common/URLStrParser.java` 排在首位；EviState 记录候选存在性证据并分别保存 Q、Z；Qwen 7B 通过远端 vLLM 生成隔离补丁，将两个 `value = name;` 改为 `value = "";`。
- Java 沙盒执行 `mvn -o -B -Dmaven.repo.local=/cache/m2 -pl dubbo-common -am -Dtest=URLStrParserTest -Dsurefire.failIfNoSpecifiedTests=false test`，退出码 0，2 项测试通过，任务 `succeeded`、补丁 `validated=true`。这些现有测试在补丁前也通过，不能独立证明缺陷修复；上节记录的独立回归用例曾在原版失败、同样的两处修改后通过。本轮自动任务没有运行该回归用例。
- React 页面返回 HTTP 200，`/health` 返回 PostgreSQL ready；API 中可读到候选、路径映射事件、补丁 Diff、测试报告以及 EviState Q/Z。原始 Dubbo Git 仓库 `git status --porcelain` 无输出，补丁只保存于任务工作树。
