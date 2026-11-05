export type TaskStatus = "queued" | "running" | "blocked" | "succeeded" | "failed" | "cancelled";

export interface TaskSummary {
  id: string;
  issue_title: string;
  repo_key: string;
  status: TaskStatus;
  current_stage: string;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Candidate {
  id: string;
  rank: number;
  target: string;
  target_type: string;
  file_path: string;
  function_name: string;
  line_start: number | null;
  line_end: number | null;
  confidence: number;
  hypothesis: string;
  source_payload: Record<string, unknown>;
  bindings: { id: string; evidence_id: string; relation: string; source_payload: Record<string, unknown> }[];
}

export interface Evidence {
  id: string;
  candidate_id: string | null;
  source: string;
  kind: string;
  content: string;
  source_payload: Record<string, unknown>;
}

export interface TaskStage {
  name: string;
  status: string;
  detail: Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
}

export interface Patch {
  id: string;
  diff_text: string;
  worktree_path: string;
  source_payload: Record<string, unknown>;
  created_at: string;
}

export interface TestResult {
  id: string;
  build_tool: string;
  command: string[];
  exit_code: number | null;
  passed: boolean | null;
  stdout: string;
  stderr: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TaskDetail extends TaskSummary {
  issue_text: string;
  repository_path: string;
  repograph_path: string;
  workflow_state_q: Record<string, unknown> | null;
  cognitive_state_z: Record<string, unknown> | null;
  stages: TaskStage[];
  candidates: Candidate[];
  evidence: Evidence[];
  patches: Patch[];
  test_results: TestResult[];
}

export interface TaskEvent {
  id: number;
  event_type: string;
  stage: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface CreateTask {
  issue_title: string;
  issue_text: string;
  repo_key: string;
  repository_path: string;
  repograph_path: string;
}
