# oxide-workflow

*GPU kernel workflow orchestration with DAG execution, ternary step states, automatic retry, and rollback. Complex GPU operations aren't single kernels — they're pipelines. This crate manages the pipeline.*

## Why This Exists

A real GPU operation is never just one kernel. Training a ternary layer is: load weights → quantize → matmul → add bias → activate → store output. That's 6 kernels, each depending on the previous. If kernel 4 fails, you need to know which outputs are valid and which aren't.

This crate orchestrates kernel DAGs with ternary step states:
- **+1 (Success):** Step completed. Output is valid. Downstream steps can proceed.
- **0 (Pending):** Step hasn't run yet. Waiting for dependencies.
- **-1 (Failed):** Step errored. Output is invalid. Downstream steps are blocked.

The key insight: ternary states enable partial failure handling. Binary (done/not-done) can't distinguish "failed" from "hasn't started yet." With ternary, the orchestrator knows exactly which steps to retry and which to skip.

## Architecture

```
Step "load" (+1) ──→ Step "matmul" (0) ──→ Step "store" (0)
                          ↓
                   Step "bias" (0) ──────────┘
                          ↓
                   Step "activate" (0) ──────┘

If matmul fails (-1):
  bias, activate, store → all skip (dependency failed)
  Retry matmul → if success → resume downstream
```

### Key Types

- **`Workflow`** — Directed acyclic graph of kernel steps. Add steps, declare dependencies, set retry policies.
- **`Step`** — Named kernel launch with ternary state, retry count, timeout, and rollback action.
- **`WorkflowExecutor`** — Runs the DAG: resolves dependencies, launches kernels, handles failures, tracks progress.
- **`Progress`** — Completion tracking: steps done, steps pending, steps failed, estimated time remaining.
- **`RollbackPlan`** — When a step fails, rollback undoes completed steps in reverse dependency order.

## Usage

```rust
use oxide_workflow::*;

let mut wf = Workflow::new();

// Define pipeline
wf.add_step(Step::new("load").with_retry(3));
wf.add_step(Step::new("matmul").depends_on("load"));
wf.add_step(Step::new("bias").depends_on("matmul"));
wf.add_step(Step::new("activate").depends_on("bias"));
wf.add_step(Step::new("store").depends_on("activate"));

// Execute
let executor = WorkflowExecutor::new();
let result = executor.run(wf);

match result {
    WorkflowResult::Complete => println!("All steps done"),
    WorkflowResult::Partial(progress) => {
        println!("Failed at step: {}", progress.last_failed.unwrap());
        println!("Steps completed: {}/{}", progress.done, progress.total);
    }
}
```

## The Deeper Idea

Workflow orchestration is where the oxide stack becomes a real system. Individual kernels are interesting; composing them into pipelines is engineering. The ternary state per step enables partial failure handling that binary (done/not-done) can't express.

This pattern maps directly to `agent-orchestration` (fleet-level workflow) and `ternary-pipeline-parallel` (multi-device pipeline parallelism). The same DAG-with-ternary-states pattern appears at every scale in the ecosystem.

## Related Crates

- `oxide-barrier` — Synchronization between workflow steps
- `oxide-epoch` — Memory reclamation coordinated with workflow phases
- `ternary-pipeline-parallel` — Pipeline parallelism across devices
- `agent-orchestration` — Fleet-level orchestration using the same DAG model
