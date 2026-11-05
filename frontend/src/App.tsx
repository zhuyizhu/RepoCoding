import sample from "./sample.json";
import { FormEvent, useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { CreateTask, TaskDetail, TaskEvent, TaskStage, TaskSummary } from "./types";

const initialForm: CreateTask = {
  issue_title: sample.issue_title,
  issue_text: sample.issue_text,
  repo_key: sample.repo_key,
  repository_path: sample.repository_path,
  repograph_path: sample.repograph_path,
};


const stageOrder = ["localization", "evistate", "repair", "tests"];
const stageNames: Record<string, string> = {
  localization: "缺陷定位",
  evistate: "证据状态",
  repair: "补丁生成",
  tests: "自动测试",
  queued: "排队中",
};
const statusNames: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  blocked: "等待配置",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  pending: "待执行",
  skipped: "未执行",
  retrying: "重试中",
};

function summarizeState(snapshot: Record<string, unknown> | null, field: string): string {
  if (!snapshot) return "待 EviState 返回";
  const hypotheses = snapshot.hypotheses as Record<string, Record<string, unknown>> | undefined;
  if (!hypotheses) return "暂无假设";
  return Object.values(hypotheses).map((value, index) => `假设 ${index + 1}: ${String(value[field] ?? "未知")}`).join(" · ");
}

function stageState(stages: TaskStage[], name: string): string {
  return stages.find((stage) => stage.name === name)?.status ?? "pending";
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(
    new Date(value),
  );
}

function eventMessage(event: TaskEvent): string {
  const payload = event.payload;
  if (typeof payload.message === "string") return payload.message;
  if (typeof payload.reason === "string") return payload.reason;
  if (event.event_type === "tests_completed") return payload.passed ? "Java 沙盒测试通过" : "Java 沙盒测试失败，请查看测试报告";
  if (typeof payload.candidate_count === "number") return "返回 " + payload.candidate_count + " 个候选位置";
  return event.event_type.replaceAll("_", " ");
}

export default function App() {
  const [creating, setCreating] = useState(true);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshTasks = useCallback(async () => {
    try {
      const rows = await api.listTasks();
      setTasks(rows);
      if (!activeId && !creating && rows.length) setActiveId(rows[0].id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法连接到服务");
    }
  }, [activeId, creating]);

  useEffect(() => {
    void refreshTasks();
    const timer = window.setInterval(() => void refreshTasks(), 5000);
    return () => window.clearInterval(timer);
  }, [refreshTasks]);

  const refreshDetail = useCallback(async (id: string) => {
    setLoading(true);
    try {
      const [nextDetail, oldEvents] = await Promise.all([api.getTask(id), api.getEvents(id)]);
      setDetail(nextDetail);
      setEvents(oldEvents);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法读取任务");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!activeId) {
      setDetail(null);
      setEvents([]);
      return;
    }
    void refreshDetail(activeId);
  }, [activeId, refreshDetail]);

  useEffect(() => {
    if (!activeId) return;
    const stream = new EventSource(api.eventStreamUrl(activeId));
    const onTaskEvent = (raw: Event) => {
      const message = raw as MessageEvent<string>;
      const next = JSON.parse(message.data) as TaskEvent;
      setEvents((previous) => (previous.some((item) => item.id === next.id) ? previous : [...previous, next]));
      void refreshDetail(activeId);
      void refreshTasks();
    };
    const stop = () => stream.close();
    stream.addEventListener("task_event", onTaskEvent);
    stream.addEventListener("stream_end", stop);
    stream.onerror = () => {
      // EventSource reconnects automatically and sends Last-Event-ID.
    };
    return () => {
      stream.removeEventListener("task_event", onTaskEvent);
      stream.removeEventListener("stream_end", stop);
      stream.close();
    };
  }, [activeId, refreshDetail, refreshTasks]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const created = await api.createTask(form);
      setForm(initialForm);
      await refreshTasks();
      setCreating(false);
      setActiveId(created.task.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "任务创建失败");
    } finally {
      setSubmitting(false);
    }
  }

  const activeSummary = tasks.find((task) => task.id === activeId);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">跳转到主要内容</a>
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">R</div>
          <div>
            <div className="brand-name">RepoCoding</div>
            <div className="brand-caption">代码分析工作台</div>
          </div>
        </div>
        <div className="sidebar-label">工作区</div>
        <button className="nav-item selected" onClick={() => { setCreating(true); setActiveId(null); setDetail(null); }}><span className="nav-icon" aria-hidden="true">＋</span>新建代码检查</button>
        <div className="sidebar-label task-label">任务记录 <span>{tasks.length}</span></div>
        <div className="task-list">
          {tasks.map((task) => (
            <button
              className={"task-nav " + (task.id === activeId ? "active" : "")}
              key={task.id}
              onClick={() => { setCreating(false); setActiveId(task.id); }}
            >
              <span className={"status-dot " + task.status} />
              <span className="task-nav-copy">
                <strong>{task.issue_title}</strong>
                <small>{task.repo_key} · {statusNames[task.status] ?? task.status}</small>
              </span>
            </button>
          ))}
          {!tasks.length && <div className="sidebar-empty">创建任务后会显示在这里</div>}
        </div>
        <div className="sidebar-footer">
          <div className="connection-light" />
          <span>任务事件支持断线回放</span>
        </div>
      </aside>

      <main className="main-panel" id="main-content">
        <header className="topbar">
          <div className="breadcrumb">RepoCoding <span>/</span> 代码检查与修复</div>
          <div className="topbar-right"><span className="local-badge">演示工作区</span><span className="avatar">RC</span></div>
        </header>

        {error && <div className="toast-error" role="alert">{error}<button aria-label="关闭错误提示" onClick={() => setError(null)}>×</button></div>}

        <div className="workspace">
          <section className="page-heading">
            <div>
              <div className="eyebrow">代码分析工作台</div>
              <h1>{activeSummary?.issue_title ?? "代码缺陷定位与自动修复"}</h1>
              <p>CodeLens 从仓库图定位缺陷，EviState 记录证据状态，Qwen 生成补丁并在独立容器中运行 Java 测试。</p>
            </div>
            {detail && <div className={"status-pill " + detail.status}><i />{statusNames[detail.status] ?? detail.status}</div>}
          </section>

          {creating && (
            <section className="panel create-panel">
              <div className="panel-heading">
                <div><span className="step-number">01</span><h2>新建代码检查</h2></div>
                <span className="panel-note">CodeLens 自动检索已准备的 Java 仓库与 Repograph</span>
              </div>
              <form onSubmit={submit} className="task-form">
                <label>任务标题<input required minLength={3} value={form.issue_title} onChange={(e) => setForm({ ...form, issue_title: e.target.value })} placeholder="例如：解析空值时抛出异常" /></label>
                <div className="form-grid">
                  <label>仓库标识<input required value={form.repo_key} onChange={(e) => setForm({ ...form, repo_key: e.target.value })} placeholder="org__repo" /></label>
                  <label>仓库目录<input required value={form.repository_path} onChange={(e) => setForm({ ...form, repository_path: e.target.value })} placeholder="org__repo" /></label>
                  <label>Repograph 目录<input required value={form.repograph_path} onChange={(e) => setForm({ ...form, repograph_path: e.target.value })} placeholder="org__repo" /></label>
                </div>
                <label>缺陷描述<textarea required minLength={10} rows={5} value={form.issue_text} onChange={(e) => setForm({ ...form, issue_text: e.target.value })} placeholder="粘贴 Issue 描述、复现步骤、预期行为和实际行为。" /></label>
                <div className="form-footer"><span>CodeLens 定位 → EviState 证据状态 → Qwen 补丁 → Java 沙盒测试。定位和修复都会调用模型。</span><button className="primary-button" disabled={submitting}>{submitting ? "提交中…" : "开始定位并修复"}<span>↗</span></button></div>
              </form>
            </section>
          )}

          {detail && !creating && (
            <>
              <section className="summary-strip">
                <div><span>仓库</span><strong>{detail.repo_key}</strong></div>
                <div><span>当前阶段</span><strong>{stageNames[detail.current_stage] ?? detail.current_stage}</strong></div>
                <div><span>候选位置</span><strong>{detail.candidates.length}</strong></div>
                <div><span>持久化事件</span><strong>{events.length}</strong></div>
              </section>

              <section className="stage-track panel">
                {stageOrder.map((name, index) => {
                  const state = stageState(detail.stages, name);
                  return <div className={"stage-step " + state} key={name}>
                    <div className="stage-marker">{state === "succeeded" ? "✓" : String(index + 1).padStart(2, "0")}</div>
                    <div><strong>{stageNames[name]}</strong><small>{statusNames[state] ?? state}</small></div>
                    {index < stageOrder.length - 1 && <div className="stage-connector" />}
                  </div>;
                })}
              </section>

              {detail.status === "blocked" && detail.error_message && (
                <section className="integration-notice">
                  <div className="notice-icon">!</div>
                  <div><strong>任务等待运行条件补齐</strong><p>{detail.error_message}</p></div>
                </section>
              )}

              <div className="content-grid">
                <section className="panel candidates-panel">
                  <div className="panel-heading">
                    <div><span className="step-number">02</span><h2>候选定位</h2></div>
                    <span className="count-badge">{detail.candidates.length} 个候选</span>
                  </div>
                  {detail.candidates.length ? detail.candidates.map((candidate) => (
                    <article className="candidate-card" key={candidate.id}>
                      <div className="candidate-top"><span className="rank">#{candidate.rank}</span><span className="confidence">{candidate.source_payload.source === "evistate" ? String(candidate.source_payload.epistemic_state) : Math.round(candidate.confidence * 100) + "%"}</span></div>
                      <h3>{candidate.target}</h3>
                      <div className="candidate-meta">{candidate.target_type}{candidate.file_path && " · " + candidate.file_path}{candidate.function_name && " · " + candidate.function_name}{candidate.line_start && ":" + candidate.line_start}</div>
                      {candidate.hypothesis && <p>{candidate.hypothesis}</p>}
                      <div className="candidate-source">{candidate.source_payload.source === "evistate" ? "EviState · " + String(candidate.source_payload.claim_type) : "CodeLens · 原始定位候选"}</div>
                      {!!candidate.bindings.length && <div className="binding-row">{candidate.bindings.map((binding) => <span key={binding.id} className={"binding " + binding.relation}>{binding.relation} 证据</span>)}</div>}
                    </article>
                  )) : <div className="empty-state"><div className="empty-symbol">⌕</div><strong>{loading ? "正在读取任务…" : "暂无定位结果"}</strong><p>提交后由 CodeLens 根据仓库图与缺陷描述生成候选位置。</p></div>}
                </section>

                <section className="panel evidence-panel">
                  <div className="panel-heading">
                    <div><span className="step-number">03</span><h2>证据链</h2></div>
                    <span className="count-badge">{detail.evidence.length} 条记录</span>
                  </div>
                  {detail.evidence.length ? <div className="evidence-list">
                    {detail.evidence.map((item) => (
                      <article className="evidence-item" key={item.id}>
                        <span className={"evidence-marker " + item.source}>{item.source === "evistate" ? "E" : "C"}</span>
                        <div><div className="evidence-title">{item.kind.replaceAll("_", " ")} <small>{item.source}</small></div><p>{item.content}</p></div>
                      </article>
                    ))}
                  </div> : <div className="empty-state compact"><div className="empty-symbol">⌁</div><strong>证据尚未返回</strong><p>不会用演示数据填充支持/反驳关系。</p></div>}
                  <div className="orthogonal-state">
                    <div><span>Q · 工作流状态</span><strong>{summarizeState(detail.workflow_state_q, "lifecycle")}</strong></div>
                    <div><span>Z · 认知状态</span><strong>{summarizeState(detail.cognitive_state_z, "epistemic_state")}</strong></div>
                  </div>
                </section>
              </div>

              <div className="content-grid lower-grid">
                <section className="panel">
                  <div className="panel-heading"><div><span className="step-number">04</span><h2>补丁 Diff</h2></div></div>
                  {detail.patches.length ? detail.patches.map((patch) => <pre className="diff-view" key={patch.id}>{patch.diff_text}</pre>) : <div className="empty-inline">尚无补丁记录。修复模型服务未配置时不会生成虚假 Diff。</div>}
                </section>
                <section className="panel">
                  <div className="panel-heading"><div><span className="step-number">05</span><h2>测试报告</h2></div></div>
                  {detail.test_results.length ? detail.test_results.map((test) => <div key={test.id}><div className="test-row"><strong>{test.build_tool}</strong><span>{test.command.join(" ")}</span><em className={test.passed ? "passed" : "failed"}>{test.passed ? "通过" : "退出码 " + (test.exit_code ?? "未知")}</em></div><details className="test-log"><summary>查看测试输出</summary><pre>{test.stdout || "（没有标准输出）"}{test.stderr ? "\n\n错误输出：\n" + test.stderr : ""}</pre></details></div>) : <div className="empty-inline">{detail.status === "running" && detail.current_stage === "tests" ? "补丁已生成，正在排队或运行 Java 沙盒测试…" : "尚无 Java 测试结果。"}</div>}
                </section>
              </div>

              <section className="panel timeline-panel">
                <div className="panel-heading"><div><span className="step-number">06</span><h2>执行事件</h2></div><span className="live-indicator"><i /> SSE</span></div>
                <div className="timeline">
                  {events.slice(-12).reverse().map((item) => <div className="timeline-row" key={item.id}><span className="timeline-dot" /><time>{formatDate(item.created_at)}</time><strong>{stageNames[item.stage] ?? item.stage}</strong><span>{eventMessage(item)}</span></div>)}
                  {!events.length && <div className="empty-inline">还没有事件</div>}
                </div>
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
