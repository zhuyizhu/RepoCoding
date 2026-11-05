# 当前真实代码审计

## 输入、来源和缺口

简历技术栈：React/TypeScript，FastAPI/REST/SSE，Celery/Redis，PostgreSQL/SQLAlchemy，Git Worktree，Docker，pytest，Maven/Gradle。

1. 用户提供的 CodeLens 目录已核实；其 README 名称是 JavaLens。原目录没有 Git 元数据。单实例 CLI 为 run_issue_localization.py，需要图、标签、问题描述、模型与 embedding。结构化入口是 LocalizationPipeline(PipelineConfig).run_sync(instance_id, issue_text, tag_rows, few_shots_long_term)，输出 PipelineResult.best_candidates/policy_trajectory 等。CLI 默认长期记忆写入未被本工程调用。config.py 与 localization_llm.py 的 provider 默认值不一致；当前部署显式指定 OpenAI 兼容的 vLLM 服务。
2. `D:\RepoCoding\dubbo` 只有三组 graph.pkl/tags.json，不是源码仓库。已从官方 apache/dubbo 仓库取得 apache__dubbo-11781 的真实 base commit d0a1bd014331483c19208b831c4f6b488654a508，保存于 repositories/apache__dubbo。工作区干净，有 pom.xml。API 样例不含 benchmark gold。
3. `D:\RepoCoding\evistate_orchestrator` 是实验控制器，run_instance.py 依赖原 Linux 上其他 adapter，不是完整机制。其 NEEDS_REPAIR 历史状态未被修改，离线结果未作为在线结果。
4. `D:\RepoCoding\framework` 提供 GraphLocator adapter；evistate_runtime.py 使用硬编码 GraphLocator provenance。它发出的 graphlocator_budgeted_review_signal 与当前核心接受的 heuristic_fault_signal/source_grounded_fault_signal 不匹配；其预算规则还描述 Top-3 而调用方取 Top-5。不能直接换个名字便声称支持 CodeLens。
5. 核心实际位于 `D:\RepoCoding\evistate_magentless_d4j_pilot10\mechanism2`。本工程仅复制 core、schemas、configs、包入口与一个核心测试文件；没有复制实验结果、oracle 或评测标签。

## 已有能力和本次新增

已有：CodeLens 定位 pipeline；EviState 有序事件追踪、不可变证据、规则提取、四类绑定、Q/Z 正交更新及提交屏障。

新增：平台前后端与数据库、outbox/SSE、CodeLens 调用封装、仅对仓库内唯一文件路径做模块前缀映射、保守 FILE 投影的 EviState 在线 handoff、数据库快照映射、受单次调用和精确替换限制的 vLLM Repair、worktree 和 Java runner。源码中还保留未接入当前入口的 Ollama Tool Calling 实验模式。

默认 EviState 规则只认可 EXISTS 的正负证据。FAULT_RELEVANT 的启发式 pilot 规则需显式启用且源码标注 paper_ready=false；本 Demo 没有擅自启用。真实源码文件存在，不等于定位正确。Q 对应 lifecycle/visibility；Z 对应 epistemic_state 及支持/攻击来源。样例中的缺失文件测试仅是自动化测试 fixture，不是 Dubbo 推理结果。

## 实现路线与当前边界

源码审计 → 真实数据/对应 checkout → 平台事件链 → CodeLens/EviState 适配 → 独立 Repair → Java 容器与缓存配置 → 本地自动化测试已按此路线完成。2026-11-06 的实机结果表明 CodeLens GPU 索引、PostgreSQL/Redis/Celery、远端 Qwen、隔离工作树和 Java 沙盒都已在带真实类名线索的 Dubbo 诊断任务中运行。

原始基准描述没有类名线索时，CodeLens 未选中实际缺陷文件，任务没有生成有效补丁；不能宣称已解决该基准实例。现有自动 Maven 测试在补丁前也通过，独立回归用例尚未接入自动报告。其他仓库、通用建图接入与公网域名/HTTPS 未验收。任务 ID 和证据见 VALIDATION.md。
