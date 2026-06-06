# oxide-workflow



## Why This Matters

# oxide-workflow
GPU kernel workflow orchestration with ternary step states.
DAG execution, retry on failure, rollback, and progress tracking.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum StepState
pub fn from_i8
pub fn as_i8
pub struct WorkflowStep
pub fn new
pub fn depends_on
pub fn with_max_retries
pub fn can_retry
pub struct Progress
pub fn remaining
pub fn percent_remaining
pub struct WorkflowDAG
```

## Usage

```toml
[dependencies]
oxide-workflow = "0.1.0"
```

```rust
use oxide_workflow::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-workflow.git
cd oxide-workflow
cargo test    # 11 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 11 |
| Lines of Rust | 566 |
| Public API | 26 items |

## License

Apache-2.0
