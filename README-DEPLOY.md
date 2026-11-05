# RepoCoding 部署与验收

本文件记录当前的自动链路。项目仍是单仓库 Dubbo Demo；公网服务与其他仓库未完成实机验收。功能边界和已跑过的任务见 [VALIDATION.md](VALIDATION.md)。

## Windows 本地部署

准备 Docker Desktop Linux 引擎、NVIDIA GPU、`D:\RepoCoding\repositories\apache__dubbo`、`D:\RepoCoding\dubbo\3.1-d0a1bd0` 中的 `graph.pkl`/`tags.json`，以及项目的 `runtime/codelens`、`runtime/evistate`。从 `.env.example` 复制本地 `.env`，配置容器可访问的远端 Qwen vLLM `/v1` 地址和模型名。`.env` 不应提交。

```powershell
Set-Location D:\RepoCoding\repocoding-agent-demo
.\scripts\prepare-codelens-index.ps1
.\scripts\start-deploy.ps1 -Gpu
```

如果 Docker Hub 的镜像拉取失败，但应用镜像已在 Docker Desktop 中存在，可使用当前源码和前端构建产物启动：

```powershell
.\scripts\start-deploy.ps1 -Gpu -Offline
```

离线模式使用 `compose.offline.yml` 挂载当前 `backend/app` 和 `frontend/dist`，不构建或拉取镜像。修改后端后重启对应 worker；修改前端后先在 `frontend` 目录运行 `npm run build`，再重启 Web。离线模式不能替代 Qwen 服务或首次下载 Java 依赖。

首次执行 Java 测试前，在能够访问 Maven 仓库时填充依赖缓存：

```powershell
docker compose -f compose.deploy.yml --profile java-setup run --rm --no-deps java-deps
```

启动后打开 <http://127.0.0.1:8080>，健康检查是 <http://127.0.0.1:8080/health>。页面默认提交 `/api/tasks`：CodeLens 自动定位、EviState 记录证据、Qwen 生成隔离补丁、Java 容器运行 Maven 或 Gradle。任务结果会写入 PostgreSQL，并通过 SSE 回放。DBeaver 可连本机 `127.0.0.1:5432`；数据库名、用户名和密码以 `.env` 为准。

```powershell
docker compose -f compose.deploy.yml -f compose.gpu.yml -f compose.offline.yml ps
docker compose -f compose.deploy.yml -f compose.gpu.yml -f compose.offline.yml logs --tail=100 api codelens-worker validation-worker java-sandbox web
.\scripts\stop-deploy.ps1
```

`stop-deploy.ps1` 保留 PostgreSQL、模型、仓库和 Maven 缓存卷；有任务数据时不要运行 `down -v`。容器部署仅暴露浏览器 `8080`，PostgreSQL `5432` 和 Redis `6379` 绑定本机回环地址。

## Linux 与公网

将项目、可信的 Git 检出和对应 Repograph 按 Compose 所需的相对路径放到服务器。需要可用的 GPU、Qwen vLLM、Maven 依赖缓存和强 PostgreSQL 密码。当前 Compose 固定 Dubbo 的目录与提交；其他仓库要更新挂载、图、索引和测试命令后才能部署。已有数据库卷不会因为修改 `.env` 而自动更改数据库账户密码。

`compose.public.yml` 与 `deployment/Caddyfile` 提供域名 HTTPS 配置模板。实际域名、DNS、证书和公网访问尚未验收；使用前应配置 `deployment/public.env` 中的域名与访问密码哈希，并在目标服务器验证。该文件不应提交。

```bash
docker compose -f compose.deploy.yml -f compose.gpu.yml -f compose.public.yml config --quiet
docker compose -f compose.deploy.yml -f compose.gpu.yml -f compose.public.yml up -d --build
curl -fsS http://127.0.0.1:8080/health
```

项目开源发布前须保留 `.gitignore`，确认没有把 `.env`、模型缓存、数据库卷、仓库副本、Repograph 或 FAISS 索引提交到 Git。
