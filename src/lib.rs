//! # oxide-workflow
//!
//! GPU kernel workflow orchestration with ternary step states.
//! DAG execution, retry on failure, rollback, and progress tracking.

use std::collections::{HashMap, HashSet, VecDeque};

/// Ternary step state: +1 = complete, 0 = running, -1 = failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepState {
    Complete = 1,
    Running = 0,
    Failed = -1,
    Pending = -2, // not yet started
}

impl StepState {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            1 => Some(StepState::Complete),
            0 => Some(StepState::Running),
            -1 => Some(StepState::Failed),
            _ => None,
        }
    }

    pub fn as_i8(self) -> i8 {
        self as i8
    }
}

impl std::fmt::Display for StepState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepState::Complete => write!(f, "complete(+1)"),
            StepState::Running => write!(f, "running(0)"),
            StepState::Failed => write!(f, "failed(-1)"),
            StepState::Pending => write!(f, "pending"),
        }
    }
}

/// A single step in a GPU kernel workflow.
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    pub id: String,
    pub dependencies: Vec<String>,
    pub state: StepState,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl WorkflowStep {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            dependencies: Vec::new(),
            state: StepState::Pending,
            retry_count: 0,
            max_retries: 3,
        }
    }

    pub fn depends_on(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
}

/// Progress snapshot for a workflow.
#[derive(Debug, Clone)]
pub struct Progress {
    pub total_steps: usize,
    pub completed: usize,
    pub running: usize,
    pub failed: usize,
    pub pending: usize,
    pub percent_complete: f64,
}

impl Progress {
    /// Estimate remaining steps (running + pending).
    pub fn remaining(&self) -> usize {
        self.running + self.pending
    }

    /// Estimate percent remaining.
    pub fn percent_remaining(&self) -> f64 {
        100.0 - self.percent_complete
    }
}

impl std::fmt::Display for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.1}% complete ({}/{}, {} running, {} failed, {} pending)",
            self.percent_complete,
            self.completed,
            self.total_steps,
            self.running,
            self.failed,
            self.pending
        )
    }
}

/// A directed acyclic graph for multi-step kernel pipelines.
#[derive(Debug, Clone)]
pub struct WorkflowDAG {
    steps: HashMap<String, WorkflowStep>,
    /// Optional rollback handler names per step id.
    rollback_order: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionEvent {
    StepStarted(String),
    StepCompleted(String),
    StepFailed(String),
    StepRetrying(String, u32),
    RollbackStarted(String),
    RollbackCompleted(String),
    WorkflowComplete,
    WorkflowFailed(String),
}

impl WorkflowDAG {
    pub fn new() -> Self {
        Self {
            steps: HashMap::new(),
            rollback_order: Vec::new(),
        }
    }

    pub fn add_step(&mut self, step: WorkflowStep) {
        self.steps.insert(step.id.clone(), step);
    }

    pub fn get_step(&self, id: &str) -> Option<&WorkflowStep> {
        self.steps.get(id)
    }

    pub fn get_step_mut(&mut self, id: &str) -> Option<&mut WorkflowStep> {
        self.steps.get_mut(id)
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Resolve topological execution order using Kahn's algorithm.
    /// Returns ordered step IDs or an error if a cycle is detected.
    pub fn topological_order(&self) -> Result<Vec<String>, String> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in self.steps.keys() {
            in_degree.insert(id.as_str(), 0);
            adj.insert(id.as_str(), Vec::new());
        }

        for step in self.steps.values() {
            for dep in &step.dependencies {
                if !self.steps.contains_key(dep) {
                    return Err(format!("dependency '{}' not found for step '{}'", dep, step.id));
                }
                adj.get_mut(dep.as_str()).unwrap().push(&step.id);
                *in_degree.get_mut(step.id.as_str()).unwrap() += 1;
            }
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        for (&id, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(id);
            }
        }

        let mut order = Vec::with_capacity(self.steps.len());
        while let Some(id) = queue.pop_front() {
            order.push(id.to_string());
            if let Some(neighbors) = adj.get(id) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if order.len() != self.steps.len() {
            return Err("cycle detected in workflow DAG".into());
        }

        Ok(order)
    }

    /// Get steps that are ready to execute (all deps complete, state is pending).
    pub fn ready_steps(&self) -> Vec<String> {
        self.steps
            .values()
            .filter(|s| s.state == StepState::Pending || s.state == StepState::Failed && s.can_retry())
            .filter(|s| {
                s.dependencies.iter().all(|dep| {
                    self.steps
                        .get(dep)
                        .map(|d| d.state == StepState::Complete)
                        .unwrap_or(false)
                })
            })
            .map(|s| s.id.clone())
            .collect()
    }

    /// Mark a step as running.
    pub fn start_step(&mut self, id: &str) -> Result<(), String> {
        let step = self
            .steps
            .get_mut(id)
            .ok_or_else(|| format!("step '{}' not found", id))?;
        step.state = StepState::Running;
        Ok(())
    }

    /// Mark a step as complete.
    pub fn complete_step(&mut self, id: &str) -> Result<(), String> {
        let step = self
            .steps
            .get_mut(id)
            .ok_or_else(|| format!("step '{}' not found", id))?;
        step.state = StepState::Complete;
        Ok(())
    }

    /// Mark a step as failed. Returns true if it will be retried.
    pub fn fail_step(&mut self, id: &str) -> Result<bool, String> {
        let step = self
            .steps
            .get_mut(id)
            .ok_or_else(|| format!("step '{}' not found", id))?;
        step.state = StepState::Failed;
        step.retry_count += 1;
        if step.can_retry() {
            step.state = StepState::Pending;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Rollback: undo all completed steps (reverse completion order).
    /// Returns the list of steps that were rolled back.
    pub fn rollback(&mut self) -> Vec<String> {
        let completed: Vec<String> = self
            .steps
            .values()
            .filter(|s| s.state == StepState::Complete)
            .map(|s| s.id.clone())
            .collect();

        let mut rolled_back = Vec::new();
        for id in completed {
            if let Some(step) = self.steps.get_mut(&id) {
                step.state = StepState::Pending;
                step.retry_count = 0;
                rolled_back.push(id);
            }
        }
        rolled_back
    }

    /// Execute the entire workflow with a simulated executor.
    /// The executor returns true for success, false for failure.
    /// Retries failed steps up to max_retries. Rolls back on unrecoverable failure.
    pub fn execute<F>(&mut self, mut executor: F) -> Vec<ExecutionEvent>
    where
        F: FnMut(&str) -> bool,
    {
        let mut events = Vec::new();

        loop {
            let ready = self.ready_steps();
            if ready.is_empty() {
                break;
            }

            for id in ready {
                self.start_step(&id).unwrap();
                events.push(ExecutionEvent::StepStarted(id.clone()));

                if executor(&id) {
                    self.complete_step(&id).unwrap();
                    events.push(ExecutionEvent::StepCompleted(id));
                } else {
                    let will_retry = self.fail_step(&id).unwrap();
                    if will_retry {
                        let step = self.steps.get(&id).unwrap();
                        events.push(ExecutionEvent::StepRetrying(
                            id.clone(),
                            step.retry_count,
                        ));
                    } else {
                        events.push(ExecutionEvent::StepFailed(id.clone()));
                        // Rollback all completed steps
                        let completed_ids: Vec<String> = self
                            .steps
                            .values()
                            .filter(|s| s.state == StepState::Complete)
                            .map(|s| s.id.clone())
                            .collect();
                        for rb_id in &completed_ids {
                            events.push(ExecutionEvent::RollbackStarted(rb_id.clone()));
                        }
                        self.rollback();
                        for rb_id in &completed_ids {
                            events.push(ExecutionEvent::RollbackCompleted(rb_id.clone()));
                        }
                        events.push(ExecutionEvent::WorkflowFailed(id));
                        return events;
                    }
                }
            }
        }

        // Check if all steps completed
        let all_done = self
            .steps
            .values()
            .all(|s| s.state == StepState::Complete);

        if all_done {
            events.push(ExecutionEvent::WorkflowComplete);
        }

        events
    }

    /// Get current progress snapshot.
    pub fn progress(&self) -> Progress {
        let total = self.steps.len();
        let completed = self
            .steps
            .values()
            .filter(|s| s.state == StepState::Complete)
            .count();
        let running = self
            .steps
            .values()
            .filter(|s| s.state == StepState::Running)
            .count();
        let failed = self
            .steps
            .values()
            .filter(|s| s.state == StepState::Failed)
            .count();
        let pending = self
            .steps
            .values()
            .filter(|s| s.state == StepState::Pending)
            .count();

        let percent_complete = if total == 0 {
            100.0
        } else {
            (completed as f64 / total as f64) * 100.0
        };

        Progress {
            total_steps: total,
            completed,
            running,
            failed,
            pending,
            percent_complete,
        }
    }
}

impl Default for WorkflowDAG {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_state_conversion() {
        assert_eq!(StepState::from_i8(1), Some(StepState::Complete));
        assert_eq!(StepState::from_i8(0), Some(StepState::Running));
        assert_eq!(StepState::from_i8(-1), Some(StepState::Failed));
        assert_eq!(StepState::from_i8(-2), None);

        assert_eq!(StepState::Complete.as_i8(), 1);
        assert_eq!(StepState::Running.as_i8(), 0);
        assert_eq!(StepState::Failed.as_i8(), -1);
    }

    #[test]
    fn test_step_builder() {
        let step = WorkflowStep::new("kernel_a")
            .depends_on("kernel_b")
            .with_max_retries(5);
        assert_eq!(step.id, "kernel_a");
        assert_eq!(step.dependencies, vec!["kernel_b"]);
        assert_eq!(step.max_retries, 5);
        assert_eq!(step.state, StepState::Pending);
        assert!(step.can_retry());
    }

    #[test]
    fn test_topological_order_simple() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a"));
        dag.add_step(WorkflowStep::new("b").depends_on("a"));
        dag.add_step(WorkflowStep::new("c").depends_on("b"));

        let order = dag.topological_order().unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_topological_order_diamond() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a"));
        dag.add_step(WorkflowStep::new("b").depends_on("a"));
        dag.add_step(WorkflowStep::new("c").depends_on("a"));
        dag.add_step(WorkflowStep::new("d").depends_on("b").depends_on("c"));

        let order = dag.topological_order().unwrap();
        assert_eq!(order[0], "a");
        assert_eq!(order[3], "d");
        // b and c can be in either order at indices 1 and 2
        let mid: HashSet<_> = order[1..3].iter().cloned().collect();
        assert!(mid.contains("b") && mid.contains("c"));
    }

    #[test]
    fn test_cycle_detection() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a").depends_on("b"));
        dag.add_step(WorkflowStep::new("b").depends_on("a"));

        let result = dag.topological_order();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cycle"));
    }

    #[test]
    fn test_execute_all_success() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("load_data"));
        dag.add_step(WorkflowStep::new("kernel_a").depends_on("load_data"));
        dag.add_step(WorkflowStep::new("kernel_b").depends_on("load_data"));
        dag.add_step(WorkflowStep::new("merge").depends_on("kernel_a").depends_on("kernel_b"));

        let events = dag.execute(|_| true);

        assert!(events.contains(&ExecutionEvent::WorkflowComplete));
        assert!(events.contains(&ExecutionEvent::StepCompleted("merge".into())));
        assert_eq!(dag.progress().percent_complete, 100.0);
    }

    #[test]
    fn test_retry_on_failure() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a").with_max_retries(3));
        dag.add_step(WorkflowStep::new("b").depends_on("a"));

        let mut attempts = HashMap::new();
        let events = dag.execute(|id| {
            if id == "a" {
                let count = attempts.entry(id.to_string()).or_insert(0);
                *count += 1;
                *count >= 3 // succeed on third attempt
            } else {
                true
            }
        });

        assert!(events.contains(&ExecutionEvent::WorkflowComplete));
        let step_a = dag.get_step("a").unwrap();
        assert_eq!(step_a.retry_count, 2); // incremented twice before success
    }

    #[test]
    fn test_rollback_on_failure() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a"));
        dag.add_step(WorkflowStep::new("b").depends_on("a"));
        dag.add_step(WorkflowStep::new("c").depends_on("b").with_max_retries(0));

        let events = dag.execute(|id| id != "c");

        assert!(events.contains(&ExecutionEvent::WorkflowFailed("c".into())));
        assert!(events.contains(&ExecutionEvent::RollbackStarted("a".into())));
        assert!(events.contains(&ExecutionEvent::RollbackStarted("b".into())));

        // After rollback, completed steps should be pending again
        assert_eq!(dag.get_step("a").unwrap().state, StepState::Pending);
        assert_eq!(dag.get_step("b").unwrap().state, StepState::Pending);
    }

    #[test]
    fn test_progress_tracking() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a"));
        dag.add_step(WorkflowStep::new("b").depends_on("a"));
        dag.add_step(WorkflowStep::new("c").depends_on("b"));

        let progress = dag.progress();
        assert_eq!(progress.total_steps, 3);
        assert_eq!(progress.completed, 0);
        assert_eq!(progress.pending, 3);
        assert_eq!(progress.remaining(), 3);

        dag.complete_step("a").unwrap();
        let progress = dag.progress();
        assert!((progress.percent_complete - 33.333).abs() < 1.0);

        dag.execute(|_| true);
        let progress = dag.progress();
        assert_eq!(progress.percent_complete, 100.0);
        assert_eq!(progress.remaining(), 0);
    }

    #[test]
    fn test_missing_dependency() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a").depends_on("nonexistent"));

        let result = dag.topological_order();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_ready_steps() {
        let mut dag = WorkflowDAG::new();
        dag.add_step(WorkflowStep::new("a"));
        dag.add_step(WorkflowStep::new("b").depends_on("a"));
        dag.add_step(WorkflowStep::new("c").depends_on("b"));

        let ready = dag.ready_steps();
        assert_eq!(ready, vec!["a"]);

        dag.complete_step("a").unwrap();
        let ready = dag.ready_steps();
        assert_eq!(ready, vec!["b"]);
    }
}
