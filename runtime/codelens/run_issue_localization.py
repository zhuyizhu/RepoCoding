from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import config
from agent.agents.pipeline import LocalizationPipeline, PipelineConfig
from agent.tools.issue_memory_tools import memory_few_shots, add_issue_example_to_memory
from agent.utils.localization_metrics import load_repograph_tag_objects
from agent.utils.memory_guard import resolve_vectorstore_path


def read_issue_text(issue_text: str | None, issue_file: str | None) -> str:
    if issue_text and issue_text.strip():
        return issue_text.strip()
    if issue_file:
        p = Path(issue_file)
        return p.read_text(encoding="utf-8").strip()
    raise ValueError("Either --issue-text or --issue-file is required.")


def main():
    parser = argparse.ArgumentParser(description="Policy-driven Java issue localization with GraphRAG.")
    parser.add_argument("--repograph-dir", type=str, default=".", help="Directory containing graph.pkl and tags.json.")
    parser.add_argument("--repo-key", type=str, default=None, help="Metadata key to filter memory/code nodes (e.g. org__repo).")
    parser.add_argument("--issue-text", type=str, default=None, help="Issue description text (can be short).")
    parser.add_argument("--issue-file", type=str, default=None, help="Path to a text file containing the issue.")
    parser.add_argument("--k", type=int, default=20, help="Top-K locations to output.")
    parser.add_argument("--hops", type=int, default=2, help="GraphRAG k-hop radius.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=int(os.environ.get("GRAPHRAG_MAX_CONTEXT_CHARS", "8000") or "8000"),
    )
    parser.add_argument("--few-shots-k", type=int, default=5, help="Few-shot examples number to inject.")
    parser.add_argument(
        "--vectorstore-path",
        type=str,
        default=os.environ.get(
            "JLOC_DATASET_VECTORSTORE_PATH",
            "data/vectorstore_shards/{repo_key}.faiss",
        ),
        help="FAISS path or safe per-repository template containing {repo_key}.",
    )
    parser.add_argument(
        "--embeddings-model",
        type=str,
        default=config.CODE_EMBEDDING_MODEL,
        help="HuggingFace embedding model (default from config.CODE_EMBEDDING_MODEL / env CODE_EMBEDDING_MODEL).",
    )
    parser.add_argument("--rebuild-vectorstore", action="store_true", help="Rebuild FAISS index from repograph.")
    parser.add_argument("--legacy-workflow", action="store_true", help="Run the fixed-threshold workflow ablation.")
    parser.add_argument("--policy-max-steps", type=int, default=8)
    parser.add_argument("--policy-max-tokens", type=int, default=0)
    parser.add_argument("--policy-max-llm-calls", type=int, default=8)
    parser.add_argument("--policy-max-wall-time-seconds", type=float, default=0.0)
    parser.add_argument("--policy-max-graph-nodes", type=int, default=0)
    parser.add_argument("--policy-stop-probability", type=float, default=0.82)
    parser.add_argument("--policy-stable-steps", type=int, default=2)
    parser.add_argument("--policy-initial-outside-probability", type=float, default=0.20)
    parser.add_argument("--policy-expansion-outside-recovery", type=float, default=0.50)
    parser.add_argument("--policy-open-world-recovery-gain", type=float, default=0.35)
    parser.add_argument(
        "--policy-target-granularity",
        choices=("file", "function", "line"),
        default="line",
    )
    parser.add_argument("--policy-calibration-file", type=str, default=None)
    parser.add_argument("--no-persist-memory", action="store_true", help="Do not write localization results into long-term memory.")
    args = parser.parse_args()
    policy_calibration = {}
    if args.policy_calibration_file:
        policy_calibration = json.loads(Path(args.policy_calibration_file).read_text(encoding="utf-8"))

    repograph_dir = args.repograph_dir
    repo_key = args.repo_key or Path(repograph_dir).name or "local_repo"
    vectorstore_path = resolve_vectorstore_path(args.vectorstore_path, repo_key)
    issue_text = read_issue_text(args.issue_text, args.issue_file)

    few = memory_few_shots.invoke(
        {
            "repograph_dir": repograph_dir,
            "repo_key": repo_key,
            "issue_text": issue_text,
            "k": args.few_shots_k,
            "vectorstore_path": vectorstore_path,
            "embeddings_model": args.embeddings_model,
            "rebuild_vectorstore": args.rebuild_vectorstore,
        }
    )

    tags_path = Path(repograph_dir) / "tags.json"
    tag_rows = load_repograph_tag_objects(tags_path)

    k_top = max(3, min(int(args.k), 20))
    max_reflect_rounds = int(
        os.environ.get("JLOC_MAX_REFLECT_ROUNDS", "2") or "2"
    )
    max_reflect_rounds = max(1, min(max_reflect_rounds, 5))

    cfg = PipelineConfig(
        repograph_dir=repograph_dir,
        repo_key=repo_key,
        vectorstore_path=vectorstore_path,
        embeddings_model=args.embeddings_model,
        hops=args.hops,
        max_context_chars=max(1000, args.max_context_chars),
        seed_k=120,
        k_top=k_top,
        max_reflect_rounds=max_reflect_rounds,
        rebuild_vectorstore=args.rebuild_vectorstore,
        policy_driven=not args.legacy_workflow,
        policy_max_steps=max(1, args.policy_max_steps),
        policy_max_tokens=max(0, args.policy_max_tokens),
        policy_max_llm_calls=max(1, args.policy_max_llm_calls),
        policy_max_wall_time_seconds=max(0.0, args.policy_max_wall_time_seconds),
        policy_max_graph_nodes=max(0, args.policy_max_graph_nodes),
        policy_stop_probability=min(1.0, max(0.0, args.policy_stop_probability)),
        policy_stable_steps=max(1, args.policy_stable_steps),
        policy_target_granularity=args.policy_target_granularity,
        policy_initial_outside_probability=min(
            0.95, max(0.0, args.policy_initial_outside_probability)
        ),
        policy_expansion_outside_recovery=min(
            1.0, max(0.0, args.policy_expansion_outside_recovery)
        ),
        policy_open_world_recovery_gain=max(
            0.0, args.policy_open_world_recovery_gain
        ),
        evidence_calibration_slope=float(policy_calibration.get("slope", 0.85)),
        evidence_calibration_intercept=float(policy_calibration.get("intercept", -0.15)),
    )

    pipe = LocalizationPipeline(cfg)
    result = pipe.run_sync(
        instance_id=repo_key,
        issue_text=issue_text,
        tag_rows=tag_rows,
        few_shots_long_term=str(few or ""),
    )
    print(result.best_localization_output)

    if not args.no_persist_memory:
        add_issue_example_to_memory.invoke(
            {
                "repograph_dir": repograph_dir,
                "repo_key": repo_key,
                "issue_text": issue_text,
                "localization_text": result.best_localization_output,
                "vectorstore_path": vectorstore_path,
                "embeddings_model": args.embeddings_model,
                "rebuild_vectorstore": args.rebuild_vectorstore,
            }
        )


if __name__ == "__main__":
    main()
