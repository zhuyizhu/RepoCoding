# JavaLens

JavaLens is a graph-guided, policy-driven framework for fine-grained issue
localization in Java repositories. It maintains a hierarchical fault belief and
selects retrieval, graph expansion, refinement, verification, backtracking, or
stopping actions under explicit inference budgets.

## Main components

- `repograph/`: Java repository graph construction.
- `agent/policy/`: belief update, action/cost contracts, and the value-of-information controller.
- `agent/agents/`: retrieval, localization, and evidence operators used by the controller.
- `run_dataset_localization.py`: batch localization and Top-N evaluation.
- `run_issue_localization.py`: single-issue localization entry point.
- `scripts/`: dataset conversion, graph construction, and experiment scripts.
- `tests/`: regression tests for benchmark isolation, retries, graph caching,
  and source encodings.

Generated repository graphs, model caches, experiment outputs, third-party
repository checkouts, and local credentials are intentionally excluded from
version control.

## Environment

Create the Conda environment and prepare a local configuration file:

```bash
conda env create -f environment.yml
conda activate jloc
cp .env.example .env
```

Configure an OpenAI-compatible model endpoint in `.env`:

```dotenv
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key
LLM_API_URL=http://your-server/v1
```

The `.env` file is ignored and must never be committed.

## Dataset localization

After generating Repographs for the dataset instances, run:

```bash
python run_dataset_localization.py \
  --dataset-jsonl-dir datasets/multi-swe-bench-java \
  --repograph-root resultgraphs/new-multi-swe-bench \
  --output-dir results/localization \
  --topn 1,3,5 \
  --few-shots-k 0 \
  --max-context-chars 8000 \
  --policy-max-steps 8 \
  --policy-max-llm-calls 8 \
  --policy-target-granularity line \
  --stream-components
```

For an initial smoke test, add `--limit 1`. The runner resumes completed
instances by default; use `--no-resume` only when outputs should be replaced.
The policy retrieves finer function/line subgraphs lazily. Use
`--legacy-workflow` only for the fixed-threshold ablation. Token, wall-time, and
retrieved-node limits are available through `--policy-max-tokens`,
`--policy-max-wall-time-seconds`, and `--policy-max-graph-nodes`.
The default terminal target is `line`. Value-based stopping is permitted only
after that resolution is reached; hard step/resource budgets can still stop an
incomplete trajectory. A graph-orphan observation triggers adaptive scope
expansion once, after which the controller can refine to the next level.
For the current Qwen/vLLM endpoint, keep `--max-context-chars 8000`; larger
retrieval blocks plus accumulated trajectory context can exceed the server's
effective prompt window and return HTTP 502.

### WSL memory safety

Dataset runs default to per-`repo_key` FAISS shards under
`data/vectorstore_shards/`. Do not point a memory-constrained WSL instance at the legacy
cross-repository `data/vectorstore.faiss`: its 4+ GiB native index plus metadata,
embedding model, Repograph, and VS Code Server can exhaust both RAM and swap.
An admission guard now refuses an existing FAISS index when it cannot preserve
2 GiB of RAM headroom. The unsafe override `JLOC_DISABLE_MEMORY_GUARD=1` exists
only for substantially larger machines and should not be used on the current 10 GB WSL setup.

All background benchmark runs use tmux plus a systemd user memory scope:

```bash
bash scripts/run_qwen_model_multi_swe_bench_java.sh
tmux ls
tmux attach -t <session-name>
```

The launcher defaults to `MemoryHigh=4G`, `MemoryMax=5G`, and
`MemorySwapMax=2G`. If a task exceeds the hard limit, only that task is killed;
VS Code Server and the WSL VM retain headroom. tmux preserves the terminal
session across ordinary VS Code disconnects, but the cgroup limit is what
protects against OOM.

## Policy-driven search

Each online step records the selected action, action-conditioned observation,
belief before and after the update, the complete set of considered action
values, the next decision, observed cost, and stop reason. A grounded primary
candidate is not treated as negative evidence merely because it is orphaned in
the bounded retrieved subgraph. Same-granularity expansion reconciles newly
retrieved nodes into the categorical candidate posterior, and line grounding
requires an exact match against the retrieved location whitelist. The
deterministic evidence updater exposes calibration parameters but does not
claim that its defaults are calibrated probabilities; fit and evaluate those
parameters on training repositories only.

The default controller uses an **open-world hierarchical belief**. Candidate
probabilities are conditional on the retrieved universe, while a separate
`outside_probability` represents the chance that the true location has not yet
been retrieved. Top-K stopping uses joint probability mass, graph-orphan
observations increase out-of-scope uncertainty, and successful scope expansion
transfers part of that mass to newly observed candidates. Configure this with
`--policy-initial-outside-probability`,
`--policy-expansion-outside-recovery`, and
`--policy-open-world-recovery-gain`.

The supported evaluation datasets are:

- `datasets/multi-swe-bench-java`
- `datasets/defects4j`

The former `abuse-bench-java` and `using-bench-java` datasets are not used.

Fit the scalar evidence calibrator on training-repository trajectories, then
pass it to held-out runs:

```bash
python scripts/fit_policy_calibration.py \
  'results/train/localization_predictions_*.jsonl' \
  --output results/train/policy_calibration.json

python run_dataset_localization.py \
  --policy-calibration-file results/train/policy_calibration.json \
  --dataset-jsonl-dir datasets/multi-swe-bench-java \
  --repograph-root resultgraphs/new-multi-swe-bench
```

## Evaluation protocol

Benchmark runs disable cross-instance issue memory by default to avoid leakage.
Gold locations are used by the evaluator and are kept outside the agent-facing
localization context. Predictions and aggregate metrics are written under the
selected output directory.

## Data and generated artifacts

The repository contains dataset metadata needed by the scripts, but not
downloaded upstream repositories, Defects4J installations, generated graph
pickles, embedding indexes, or experiment outputs. Recreate those artifacts in
the ignored local directories documented by the corresponding scripts.

## Status

This repository contains active research code. Interfaces and experiment
configurations may change while the evaluation is being completed.
