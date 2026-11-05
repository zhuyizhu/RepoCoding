# 文件职责

| 文件 | 作用 |
|---|---|
| backend/app/main.py | 任务 REST API、事件查询、SSE 回放与健康检查。 |
| backend/app/models.py | 任务阶段、候选、证据、Support/Attack、工具调用、补丁、测试、事件和 outbox 表；Q/Z 分开保存。 |
| backend/app/schemas.py | 请求与响应校验、相对路径约束。 |
| backend/app/database.py | SQLAlchemy 引擎和会话。 |
| backend/app/config.py | 环境变量配置。 |
| backend/app/events.py | 同事务追加事件。 |
| backend/app/celery_app.py | Redis Broker 和 Beat 定时发布配置。 |
| backend/app/worker.py | outbox 发布、定位调用、重试与 blocked 状态记录。 |
| backend/app/engine/codelens.py | 按已审计 PipelineConfig/run_sync 契约调用真实 CodeLens，校验图、模型设备和依赖。 |
| backend/app/engine/path_resolution.py | 将 CodeLens 的模块相对路径与真实仓库做唯一后缀匹配；歧义时不猜测。 |
| backend/app/worktrees.py | 独立 Git worktree、补丁检查/应用和 diff 采集。 |
| backend/app/sandbox_client.py | Java 容器验证请求客户端。 |
| backend/app/validation_worker.py | Celery 异步运行 Java 测试，写入测试、工具调用、任务状态和事件。 |
| backend/tests/test_api.py | API、outbox 和持久化 SSE 用例。 |
| backend/tests/test_worktrees.py | 实际 Git worktree 隔离用例。 |
| backend/alembic/env.py | 数据库迁移环境。 |
| backend/alembic/versions/0001_initial.py | 首次建表迁移；后续修改应增加新 revision。 |
| frontend/src/App.tsx | 任务创建、进度、候选、证据、Q/Z、Diff、测试和事件页面。 |
| frontend/src/api.ts | REST 与 SSE 地址封装。 |
| frontend/src/types.ts | 前端数据类型。 |
| frontend/src/styles.css | 页面布局与样式。 |
| frontend/src/main.tsx | React 挂载入口。 |
| sandbox/runner.py | 在临时副本执行离线 Maven/Gradle 命令，限制超时并采集测试输出；已验证 Dubbo Maven。 |
| sandbox/regression/URLStrParserEmptyValueTest.java | Dubbo 样例的独立回归用例，验证空参数在原版失败、补丁通过。 |
| compose.deploy.yml、compose.gpu.yml、compose.offline.yml | 常规部署、GPU worker 和使用本地镜像的离线启动配置。 |
| backend/Dockerfile | 平台 Python 依赖镜像，不含研究依赖。 |
| sandbox/Dockerfile | 非 root Java/Maven/Gradle 测试容器镜像。 |
| .env.example | 无密钥的环境配置模板。 |
| samples/create-task.json | 来自真实数据集的 API 请求样例，不含 gold。 |
| source-manifest.json、evistate-source-manifest.json | 本机源码审计记录，默认不纳入公开仓库。 |
| backend/app/engine/evistate.py | 新 CodeLens FILE 投影适配器；实际调用 EviState tracker/barrier，输出快照与 handoff 收据。 |
| backend/app/engine/persist_evistate.py | 分开写入 Q/Z，保存真实证据、绑定和规范事件。 |
| backend/app/engine/repair.py | 最小版 vLLM 单次修复、精确替换和隔离 Diff；也保留尚未实测的 Ollama Tool Calling 完整模式。 |
| backend/research-requirements.txt | 单实例研究 Worker 的可移植依赖组合。 |
| backend/tests/test_evistate.py | 真实核心 handoff、Q/Z 正交及数据库绑定集成测试。 |
| backend/tests/test_repair_tools.py | 修复文件访问约束、测试文件禁止修改与禁用检查。 |
| scripts/preflight.py | 只读检查本地工具、代码、图和 Git 提交，不调用模型。 |
| runtime/codelens、runtime/evistate | 审计源码副本；内部逐文件来源由两个 manifest 记录。 |
| .dockerignore | 排除本机依赖、产物和私密配置。 |
| CLEANUP.md | 已清理内容和可自行清理的生成目录。 |
| scripts/prepare-codelens-index.ps1 | 检查 GPU 并生成或复用 CodeLens 的 BGE-M3/FAISS 索引。 |
| compose.deploy.yml | PostgreSQL、Redis、独立 Dubbo 副本、FastAPI、Celery worker、隔离 Java 测试容器、静态前端与 Maven 缓存准备任务。 |
| deployment/seed_repo.py | 从只读原仓库按固定提交生成 Docker 命名卷中的干净副本。 |
| deployment/Dockerfile.frontend、deployment/nginx.conf | 构建 React 静态文件并由 Nginx 同源代理 API/SSE。 |
| compose.public.yml、deployment/Caddyfile | 可选的域名 HTTPS 与浏览器密码验证。 |
| deployment/public.env.example | 公网域名、用户名、密码哈希的填写模板。 |
| scripts/start-deploy.ps1、scripts/stop-deploy.ps1 | Windows 容器版一键启动与保留数据的停止脚本；支持 `-Gpu -Offline`。 |
| README-DEPLOY.md | 本机开发、容器部署和域名上线的操作手册与验收边界。 |


当前自动链路：App.tsx → main.py `/api/tasks` → Redis/Celery `worker.py` → CodeLens → 唯一路径映射 → EviState → Qwen vLLM 快速修复 → Git worktree → `validation_worker.py` → Java 测试容器 → PostgreSQL → SSE → App.tsx。人工指定文件的旧 `/api/demo` 仍保留供故障定位，页面默认使用自动入口。新仓库建图尚需单独准备并配置挂载。
