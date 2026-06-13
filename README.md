# Oxide Workflow

**DAG-based workflow orchestration** for multi-step GPU kernel pipelines with ternary step states, automatic retry on failure, topological ordering (Kahn's algorithm), rollback support, and progress tracking. Each step resolves to a ternary outcome: complete (+1), running (0), or failed (−1).

## Why It Matters

Complex GPU workloads — training pipelines, inference graphs, data processing chains — require orchestration that handles:

- **Dependencies** — kernels that must run in order (e.g., preprocessor → model → postprocessor).
- **Failure recovery** — transient GPU errors (OOM, CUDA faults) need automatic retry with bounded attempts.
- **Rollback** — when a step fails permanently, completed steps may need to be undone.
- **Progress monitoring** — real-time visibility into what's running, what's done, what's blocked.

This crate provides these primitives without external dependencies.

## How It Works

### Ternary Step States

| State | Value | Meaning |
|-------|-------|---------|
| `Complete` | +1 | Step finished successfully |
| `Running` | 0 | Step is currently executing |
| `Failed` | −1 | Step failed (may retry or abort) |
| `Pending` | −2 | Step hasn't started (waiting on dependencies) |

### DAG Construction

Build a workflow by adding steps with their dependencies:

```rust
let mut dag = WorkflowDAG::new();

dag.add_step(WorkflowStep::new("load_data"));
dag.add_step(WorkflowStep::new("preprocess").depends_on("load_data"));
dag.add_step(WorkflowStep::new("train").depends_on("preprocess").with_max_retries(5));
dag.add_step(WorkflowStep::new("evaluate").depends_on("train"));
```

### Topological Sort (Kahn's Algorithm)

Execution order is resolved via **Kahn's algorithm**:

```
1. Compute in-degree for each node
2. Enqueue all nodes with in-degree 0
3. While queue not empty:
   a. Dequeue node, add to order
   b. For each neighbor: decrement in-degree
   c. If in-degree reaches 0, enqueue
4. If order.size() < node_count → cycle detected (error)
```

| Property | Complexity |
|----------|------------|
| Time | O(V + E) |
| Space | O(V + E) |
| Cycle detection | Included (remaining in-degrees > 0) |

### Execution with Retry and Rollback

The `execute()` method runs steps in dependency order:

```
For each ready step (all dependencies complete):
  1. Mark Running
  2. Execute via callback
  3. If success → mark Complete, continue
  4. If failure:
     a. If retry_count < max_retries → increment retry_count, retry
     b. Else → mark Failed, rollback all completed steps, abort
```

**ExecutionEvent** stream:

| Event | When |
|-------|------|
| `StepStarted(id)` | Step begins executing |
| `StepCompleted(id)` | Step finishes successfully |
| `StepRetrying(id, n)` | Step failed, retry attempt n |
| `StepFailed(id)` | Step failed permanently |
| `RollbackStarted(id)` | Beginning rollback of a completed step |
| `RollbackCompleted(id)` | Rollback finished |
| `WorkflowComplete` | All steps completed |
| `WorkflowFailed(id)` | Workflow aborted due to step id |

### Progress Tracking

```rust
let progress = dag.progress();
// Progress {
//   total_steps: 4,
//   completed: 2,
//   running: 1,
//   failed: 0,
//   pending: 1,
//   percent_complete: 50.0
// }
```

### Complexity

| Operation | Time |
|-----------|------|
| Add step | O(1) |
| Topological sort | O(V + E) |
| Find ready steps | O(V) |
| Execute step | O(1) + callback |
| Progress snapshot | O(V) |

## Quick Start

```rust
use oxide_workflow::{WorkflowDAG, WorkflowStep, ExecutionEvent};

let mut dag = WorkflowDAG::new();
dag.add_step(WorkflowStep::new("a"));
dag.add_step(WorkflowStep::new("b").depends_on("a"));
dag.add_step(WorkflowStep::new("c").depends_on("a"));
dag.add_step(WorkflowStep::new("d").depends_on("b").depends_on("c"));

// Verify topological order
let order = dag.topological_order()?; // ["a", "b", "c", "d"]

// Execute (callback returns true on success)
let events = dag.execute(|step_id| {
    println!("Executing {}", step_id);
    true // success
});

assert!(events.contains(&ExecutionEvent::WorkflowComplete));
```

## API

### `WorkflowDAG`

| Method | Description |
|--------|-------------|
| `new()` | Create empty DAG |
| `add_step(step)` | Register a step |
| `topological_order()` | Kahn's sort; error on cycle |
| `ready_steps()` | Steps with all deps complete |
| `execute(callback)` | Run all steps with retry/rollback |
| `progress()` | `Progress` snapshot |

### `WorkflowStep`

Builder-pattern construction: `.depends_on(id)`, `.with_max_retries(n)`.

### `Progress`

Includes `remaining()` and `percent_remaining()` helpers.

## Architecture Notes

The DAG stores steps in a `HashMap<String, WorkflowStep>` for O(1) lookup. The topological sort allocates temporary adjacency lists and in-degree maps, then uses a `VecDeque` as the Kahn's algorithm worklist.

The `execute()` callback is a generic `FnMut(&str) -> bool`, allowing any execution strategy (synchronous, async via blocking, shell-out to GPU binary).

The **γ + η = C** ternary model: each step resolves to **(γ) Complete (+1)** (success — the step's contribution is locked in) or **(η) Failed (−1)** (permanent failure — triggers rollback). Together γ + η = C covers all terminal states, with Running (0) being the transient state. The Progress struct tracks this partition: `completed` counts γ, `failed` counts η.

## References

1. Kahn, A. B. (1962). "Topological sorting of large networks." *Communications of the ACM*, 5(11), 558–562. — The topological sort algorithm.
2. Dean, J. & Barroso, L. A. (2013). "The tail at scale." *Communications of the ACM*, 56(2), 74–80. — Why retries matter in large pipelines.
3. Karger, P. (2014). "DAGs, tasks, and workflows: A survey of scientific workflow systems." *Procedia Computer Science*, 29, 2276–2285.
4. Malewicz, G. et al. (2010). "Pregel: A system for large-scale graph processing." *SIGMOD 2010*. — DAG-based superstep model.
5. Zaharia, M. et al. (2012). "Resilient Distributed Datasets." *NSDI 2012*. — Lineage-based fault tolerance for DAGs.

## License

MIT
