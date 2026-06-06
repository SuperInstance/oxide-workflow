# oxide-workflow

*GPU kernel workflow orchestration with DAG execution, retry, and rollback.*

## Why This Exists

Complex GPU operations aren't single kernels — they're DAGs of kernel launches. This crate orchestrates those DAGs with ternary step states (success/pending/failed), automatic retry on failure, and rollback for partial completion.

## Architecture

### Key Types

Workflow (DAG of kernel steps), Step (single kernel launch with ternary state), WorkflowExecutor (runs DAG with dependency resolution), Progress (step completion tracking)

### State Machine

```
+1 (Active/Arrived/Allocated)
  ↓ transition event
 0 (Grace/InTransit/Fragmented)
  ↓ transition event
-1 (Reclaimable/NotStarted/Free)
```

## Usage

```rust
use oxide_workflow::*;

let mut wf = Workflow::new(); wf.add_step("compute").depends_on("load"); wf.add_step("store").depends_on("compute"); let result = executor.run(wf);
```

## The Deeper Idea

Workflow orchestration is where the oxide stack becomes a real system. Individual kernels are interesting; composing them into pipelines is engineering. The ternary state per step enables partial failure handling that binary (done/not-done) can't express.

## Related Crates

- `oxide-fleet` — Fleet-level orchestration using these primitives
- `oxide-sandbox` — Safe execution environment built on oxide primitives
- `oxide-slotmap` — Slot-based memory management (complementary allocation strategy)
