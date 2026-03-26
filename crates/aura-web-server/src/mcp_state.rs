//! In-memory state for MCP agent tasks.
//!
//! Tracks agent tasks triggered via the MCP `trigger_agent` tool so that callers
//! can query status via `query_agent_status`.
//!
//! # Design Notes
//!
//! Tasks are stored in a `RwLock<HashMap>` with TTL-based eviction. Completed or
//! failed tasks are automatically removed after [`TASK_TTL_SECS`] seconds, and
//! the store is capped at [`MAX_TASKS`] entries to prevent unbounded growth.

use chrono::Utc;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Maximum number of tasks retained in the store.
const MAX_TASKS: usize = 10_000;

/// Finished tasks are evicted after this many seconds (1 hour).
const TASK_TTL_SECS: i64 = 3600;

/// Lifecycle state of an MCP-triggered agent task.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Agent is still running.
    Running,
    /// Agent finished successfully.
    Complete,
    /// Agent failed with the enclosed message.
    Failed(String),
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Running => write!(f, "running"),
            TaskStatus::Complete => write!(f, "complete"),
            TaskStatus::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

/// A single agent task triggered via MCP.
#[derive(Debug, Clone, Serialize)]
pub struct AgentTask {
    /// Unique task ID returned to the caller.
    pub id: String,
    /// Agent alias used to run this task.
    pub agent: String,
    /// Original prompt supplied by the caller.
    pub prompt: String,
    /// Current lifecycle status.
    pub status: TaskStatus,
    /// Unix timestamp (seconds) when the task was created.
    pub created_at: i64,
    /// Unix timestamp (seconds) when the task completed/failed, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
    /// Final agent response text, available when status is `Complete`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

impl AgentTask {
    pub fn new(id: String, agent: String, prompt: String) -> Self {
        Self {
            id,
            agent,
            prompt,
            status: TaskStatus::Running,
            created_at: Utc::now().timestamp(),
            completed_at: None,
            result: None,
        }
    }
}

/// Shared, thread-safe store of MCP agent tasks.
///
/// Finished tasks are evicted after [`TASK_TTL_SECS`] on each insert, and the
/// total number of entries is capped at [`MAX_TASKS`].
#[derive(Default)]
pub struct McpTaskStore {
    tasks: RwLock<HashMap<String, AgentTask>>,
}

impl McpTaskStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            tasks: RwLock::new(HashMap::new()),
        })
    }

    /// Insert a new task (status: Running).
    ///
    /// Returns `Err` with a message if the store is at capacity even after
    /// evicting expired entries.
    pub async fn insert(&self, task: AgentTask) -> Result<(), String> {
        let mut guard = self.tasks.write().await;
        Self::evict_expired(&mut guard);
        if guard.len() >= MAX_TASKS {
            return Err(format!(
                "Task store at capacity ({MAX_TASKS}). Try again later."
            ));
        }
        guard.insert(task.id.clone(), task);
        Ok(())
    }

    /// Retrieve a task by ID.
    pub async fn get(&self, id: &str) -> Option<AgentTask> {
        let guard = self.tasks.read().await;
        guard.get(id).cloned()
    }

    /// Update a task's status and optional result (called from background worker).
    pub async fn complete(&self, id: &str, result: String) {
        let mut guard = self.tasks.write().await;
        if let Some(task) = guard.get_mut(id) {
            task.status = TaskStatus::Complete;
            task.result = Some(result);
            task.completed_at = Some(Utc::now().timestamp());
        }
    }

    /// Mark a task as failed.
    pub async fn fail(&self, id: &str, reason: String) {
        let mut guard = self.tasks.write().await;
        if let Some(task) = guard.get_mut(id) {
            task.status = TaskStatus::Failed(reason);
            task.completed_at = Some(Utc::now().timestamp());
        }
    }

    /// Remove finished tasks whose `completed_at` is older than [`TASK_TTL_SECS`].
    fn evict_expired(tasks: &mut HashMap<String, AgentTask>) {
        let cutoff = Utc::now().timestamp() - TASK_TTL_SECS;
        tasks.retain(|_, t| match t.status {
            TaskStatus::Running => true,
            _ => t.completed_at.is_none_or(|ts| ts > cutoff),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_insert_and_get() {
        let store = McpTaskStore::new();
        let task = AgentTask::new("t1".into(), "agent".into(), "hello".into());
        store.insert(task).await.unwrap();

        let t = store.get("t1").await.unwrap();
        assert_eq!(t.id, "t1");
        assert_eq!(t.status, TaskStatus::Running);
    }

    #[tokio::test]
    async fn test_get_missing_returns_none() {
        let store = McpTaskStore::new();
        assert!(store.get("nope").await.is_none());
    }

    #[tokio::test]
    async fn test_complete() {
        let store = McpTaskStore::new();
        store
            .insert(AgentTask::new("t1".into(), "a".into(), "p".into()))
            .await
            .unwrap();

        store.complete("t1", "done".into()).await;
        let t = store.get("t1").await.unwrap();
        assert_eq!(t.status, TaskStatus::Complete);
        assert_eq!(t.result.as_deref(), Some("done"));
        assert!(t.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_fail() {
        let store = McpTaskStore::new();
        store
            .insert(AgentTask::new("t1".into(), "a".into(), "p".into()))
            .await
            .unwrap();

        store.fail("t1", "boom".into()).await;
        let t = store.get("t1").await.unwrap();
        assert_eq!(t.status, TaskStatus::Failed("boom".into()));
        assert!(t.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_eviction_removes_old_finished_tasks() {
        let store = McpTaskStore::new();

        // Manually insert a task with an old completed_at timestamp
        {
            let mut guard = store.tasks.write().await;
            let mut old_task = AgentTask::new("old".into(), "a".into(), "p".into());
            old_task.status = TaskStatus::Complete;
            old_task.completed_at = Some(Utc::now().timestamp() - TASK_TTL_SECS - 1);
            guard.insert("old".into(), old_task);

            // Also insert a running task — should not be evicted
            guard.insert(
                "running".into(),
                AgentTask::new("running".into(), "a".into(), "p".into()),
            );
        }

        // Trigger eviction via insert
        store
            .insert(AgentTask::new("new".into(), "a".into(), "p".into()))
            .await
            .unwrap();

        assert!(store.get("old").await.is_none(), "expired task should be evicted");
        assert!(store.get("running").await.is_some(), "running task should survive");
        assert!(store.get("new").await.is_some(), "new task should exist");
    }
}
