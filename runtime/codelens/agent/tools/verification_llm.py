from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate

from agent.tools.localization_llm import invoke_llm_with_ollama_fallback


def generate_verification_reflection(
    issue_text: str,
    localization_output: str,
    gold_file_paths: List[str],
    graph_context_excerpt: str,
) -> str:
    """
    Verification agent: compare model output against oracle file list (eval / training only),
    produce concise feedback for the next localization attempt without copying the answer verbatim.
    """
    gold_lines = "\n".join(f"- {g}" for g in sorted(set(gold_file_paths))[:40])
    prompt = ChatPromptTemplate.from_template(
        """你是 Bug 定位结果的「验证与反思」代理。你的输入包含：
1) Issue 描述
2) 当前模型输出（含「怀疑文件」阶段与 Top-K 行号阶段；对比时以行号阶段为主）
3) 参考用的「真实需修改文件」列表（仅用于评估与迭代，勿在反馈中原样罗列为最终答案）
4) GraphRAG 上下文摘录（用于判断模型是否偏离了代码库中的相关区域）

任务：
- 判断当前候选是否可能遗漏了与 Issue 语义强相关、且应在主源码（非测试桩）中修改的位置。
- 指出推理或证据上的弱点（例如：过度依赖测试文件、忽略异常栈中的类名、未覆盖某子系统等）。
- 给出 **下轮定位应优先检索或排序的方向**（用简短条目列出），引导定位模型在 GraphRAG 上下文中重新筛选。

硬性要求：
- 用中文输出，长度控制在 400 字以内。
- 不要输出 markdown 标题或代码块。
- 不要简单重复「真实需修改文件」列表当作反馈；应转化为可操作的检索建议（关键词、模块名、行为描述）。

====================
Issue
{issue_text}

====================
当前模型输出
{localization_output}

====================
参考：真实需修改文件（oracle，仅供对比）
{gold_lines}

====================
GraphRAG 摘录
{graph_context_excerpt}
"""
    )
    text = prompt.format(
        issue_text=issue_text[:8000],
        localization_output=(localization_output or "")[:6000],
        gold_lines=gold_lines or "(无)",
        graph_context_excerpt=(graph_context_excerpt or "")[:4500],
    )
    try:
        resp = invoke_llm_with_ollama_fallback(text, call_site="generate_verification_reflection")
        out = (resp.content or "").strip()
        if len(out) > 1200:
            out = out[:1200] + "…"
        return out
    except Exception:
        return (
            "上一轮候选与 Issue 核心语义对齐不足。请重新阅读 GraphRAG 中与报错栈、API、"
            "配置项相关的 **src/main/java** 下实现类，降低仅含 androidTest/ 或纯测试桩文件的权重；"
            "优先列出直接实现业务逻辑或序列化/反序列化的类与方法。"
        )
