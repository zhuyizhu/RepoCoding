# RepoCoding Agent

面向 Java 仓库的缺陷定位与修复 Demo。**CodeLens 是定位主引擎**；EviState 作为插件记录候选与证据状态；Qwen 7B 通过远端 vLLM 接口生成补丁；Java 测试在独立容器中运行。

> 自动 CodeLens 链路已在 Dubbo 的 GPU 索引上实机运行。原始问题描述下 CodeLens 未选中缺陷文件，修复未成功；补充真实 Java 类名线索的诊断任务完成了 `CodeLens → EviState → Qwen → Java 测试`。这证明链路可运行，不代表原始基准问题已由模型独立解决。

## 架构

```text
React / TypeScript → Nginx → FastAPI → PostgreSQL
                                  ↓
                            Celery / Redis
                                  ↓
                     CodeLens → EviState → Qwen
                                              ↓
                                  Git worktree → Java 测试容器
```

CodeLens 使用 Repograph 和 BGE-M3/FAISS 检索，再调用 Qwen 定位。EviState 的工作流状态 Q 和认知状态 Z 分开保存。页面展示任务阶段、候选位置、证据、补丁 Diff、测试结果与可断线回放的事件。

## 本地输入

当前 Compose 样例固定使用 Dubbo，需要：

- 本项目 `D:\RepoCoding\repocoding-agent-demo`，其中 `runtime/codelens` 是 CodeLens 源码，`runtime/evistate` 是 EviState 核心。
- 干净的 Git 仓库 `D:\RepoCoding\repositories\apache__dubbo`，对应版本的图 `D:\RepoCoding\dubbo\3.1-d0a1bd0\graph.pkl` 与 `tags.json`。
- 本项目的 `.env`，从 `.env.example` 复制并填写 Docker 容器可访问的 vLLM `/v1` 地址。不要提交 `.env`。
- Docker Desktop Linux 容器与 NVIDIA GPU 透传，用于快速构建样例图的向量索引。

Qwen 权重保存在超算端，本机通过 HTTP 接口调用。BGE-M3 权重由 CodeLens worker 在 Docker 命名卷中缓存和加载；生成的 FAISS 索引保存在 `deployment-data/vectorstores`。PostgreSQL、仓库副本和 Maven 依赖也存放在 Docker 卷中。已有索引处理新问题时仍需 BGE-M3 生成查询向量。

## Windows 启动

```powershell
Set-Location D:\RepoCoding\repocoding-agent-demo
.\scripts\prepare-codelens-index.ps1
.\scripts\start-deploy.ps1 -Gpu
```

索引脚本会检查 CUDA，避免意外退回 CPU。首次运行可能较久；完成后索引可复用。启动后打开 <http://127.0.0.1:8080>。

如果 Docker Hub 无法访问，但上述应用镜像已经在 Docker Desktop 中构建过，可使用本地镜像启动：

```powershell
.\scripts\start-deploy.ps1 -Gpu -Offline
```

离线选项会检查本地镜像，使用当前 `backend/app` 源码及 `frontend/dist` 页面文件，不重新拉取或构建镜像。修改前端后先在 `frontend` 目录运行 `npm run build`。缺少某个镜像时仍需在 Docker Hub 可访问的网络中构建或导入镜像。此选项只解决容器镜像拉取问题，Java 项目依赖和远端 Qwen 服务仍须可访问。

第一次执行 Java 测试前，还要在线准备 Maven 缓存：

```powershell
docker compose -f compose.deploy.yml --profile java-setup run --rm --no-deps java-deps
```

`frontend/src/sample.json` 提供公开 Dubbo 问题描述。提交任务后，API 保存任务，Celery 调用真实 CodeLens `LocalizationPipeline.run_sync`；现存的 Java 候选文件进入 EviState、Qwen 修复和 Java 测试。失败会作为失败或阻塞记录，不会伪造定位结果。

PostgreSQL 可用 DBeaver 连接 `127.0.0.1:5432`，数据库名、用户名和密码以本地 `.env` 为准。停止服务运行 `.\scripts\stop-deploy.ps1`；不要对有数据的部署执行 `docker compose down -v`。

## 已验证与待验证

- 前端 TypeScript/Vite 构建与 Compose 配置检查通过。
- 人工指定文件的 Dubbo 样例已验证 Qwen 补丁及 Maven 测试；见 `VALIDATION.md`。
- CodeLens GPU 索引已加载，自动候选与完整 `CodeLens → EviState → 修复 → 测试` 链路在带类名线索的诊断任务中通过；原始描述下的定位结果不准确。详细任务 ID、补丁和测试限制见 `VALIDATION.md`。
- 当前 Compose 只映射 Dubbo。新仓库需要同版本 Git 检出、Repograph、索引及 Compose 挂载；测试命令可能需要调整。
- `graph.pkl` 和 FAISS 存储含 Python 序列化数据，只加载可信来源。公网 HTTPS 模板在 `compose.public.yml`，尚未做域名与证书验收。

更多运行细节见 `README-DEPLOY.md`；代码来源与历史验证见 `AUDIT.md`、`VALIDATION.md`。

项目代码按 [Apache-2.0](LICENSE) 发布；Dubbo 源码、Repograph 产物和模型权重不随本仓库分发。
