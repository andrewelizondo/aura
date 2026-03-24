//! In-memory state for MCP agent tasks.
//!
//! Tracks agent tasks triggered via the MCP `trigger_agent` tool so that callers
//! can query status and inject additional context via `query_agent_status` and
//! `send_context_to_agent`.
//!
//! # Design Notes
//!
//! Tasks are stored in a `RwLock<HashMap>` for simplicity in this prototype.
//! In production this would be backed by a durable store and would support
//! task expiry / cleanup.

use chrono::Utc;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

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
    pub created_at: u64,
    /// Unix timestamp (seconds) when the task completed/failed, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    /// Final agent response text, available when status is `Complete`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Additional context messages pushed by callers via `send_context_to_agent`.
    /// In this prototype they are stored but not yet fed back into a live agent;
    /// a production implementation would forward them to the running stream.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_messages: Vec<String>,
}

impl AgentTask {
    pub fn new(id: String, agent: String, prompt: String) -> Self {
        Self {
            id,
            agent,
            prompt,
            status: TaskStatus::Running,
            created_at: Utc::now().timestamp() as u64,
            completed_at: None,
            result: None,
            context_messages: Vec::new(),
        }
    }
}

/// Shared, thread-safe store of MCP agent tasks.
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
    pub async fn insert(&self, task: AgentTask) {
        let mut guard = self.tasks.write().await;
        guard.insert(task.id.clone(), task);
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
            task.completed_at = Some(Utc::now().timestamp() as u64);
        }
    }

    /// Mark a task as failed.
    pub async fn fail(&self, id: &str, reason: String) {
        let mut guard = self.tasks.write().await;
        if let Some(task) = guard.get_mut(id) {
            task.status = TaskStatus::Failed(reason);
            task.completed_at = Some(Utc::now().timestamp() as u64);
        }
    }

    /// Append a context message to a task.
    /// Returns `false` if the task does not exist.
    pub async fn add_context(&self, id: &str, context: String) -> bool {
        let mut guard = self.tasks.write().await;
        if let Some(task) = guard.get_mut(id) {
            task.context_messages.push(context);
            true
        } else {
            false
        }
    }
}
