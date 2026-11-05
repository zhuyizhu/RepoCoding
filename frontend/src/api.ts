import type { CreateTask, TaskDetail, TaskEvent, TaskSummary } from "./types";

const base = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(base + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? ("请求失败 (" + response.status + ")")));
  }
  return response.json() as Promise<T>;
}

export const api = {
  listTasks: () => request<TaskSummary[]>("/api/tasks"),
  getTask: (id: string) => request<TaskDetail>("/api/tasks/" + id),
  getEvents: (id: string, after = 0) =>
    request<TaskEvent[]>("/api/tasks/" + id + "/events?after=" + after),
  inspectLocalFile: (data: CreateTask & { file_path: string; repair_with_model: boolean }) =>
    request<{ task: TaskSummary; event_id: number }>("/api/demo", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  createTask: (data: CreateTask) =>
    request<{ task: TaskSummary; event_id: number }>("/api/tasks", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  eventStreamUrl: (id: string, after = 0) =>
    base + "/api/tasks/" + id + "/stream?after=" + after,
};
