# 清理与保留

已移除新工程中被真实 Repair 流程替代的 repair_validation.py，以及旧的 EviState 占位检查函数。Docker 构建排除了本机依赖、模型缓存、worktree 和产物，避免把无关大文件打入镜像。

本地运行必须保留：Dubbo 图数据、`repositories/apache__dubbo` 源码、`runtime/codelens` 和 `runtime/evistate` 核心、样例问题。图、仓库副本、索引与含本机路径的源码清单默认不提交公开仓库；已脱敏的测试结论写在 `VALIDATION.md`。

可再生成但暂保留便于继续验证：frontend/node_modules、frontend/dist、.testdeps、.researchdeps、__pycache__、.pytest_cache。若需要节省空间，可在停止相关进程后删除这些本机生成目录；Docker 镜像不依赖它们。

framework、evistate_orchestrator、evistate_graphlocator_qwen32b、evistate_magentless_d4j_pilot10 属于你的原始研究资料。Demo 运行不依赖其中的历史 results/outputs/oracle，但这里可能包含你需要的实验记录，本轮未删除。mechanism2 原件必须先保留，它是当前机制的可追溯来源。
