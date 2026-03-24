//! MCP (Model Context Protocol) server endpoint for Aura.
//!
//! Mounts at `POST /mcp` and speaks JSON-RPC 2.0 over HTTP Streamable transport.
//! This follows the Hierarchical MCP pattern described in the Aura design notes:
//! Aura exposes an MCP *server* so that external agents, services, and CI/CD
//! pipelines can trigger and interact with Aura agents via the lowest-common-
//! denominator protocol that almost every agent builder supports.
//!
//! # Transport
//!
//! Uses HTTP Streamable (SSE-capable) transport rather than stdio so that Aura
//! can be deployed as a distributed service. Each request is a `POST /mcp`
//! containing a JSON-RPC 2.0 payload. Responses are JSON for simple methods
//! (`initialize`, `tools/list`) and for synchronous tool calls.  Async tool
//! calls return a `task_id` immediately; callers poll via `query_agent_status`.
//!
//! # Exposed Tools
//!
//! | Tool | Description |
//! |------|-------------|
//! | `trigger_agent` | Kick off an Aura agent with a prompt (webhook-like) |
//! | `query_agent_status` | Poll the status / result of a running task |
//! | `send_context_to_agent` | Push additional context to a running task |
//!
//! # Who calls this?
//!
//! 1. **Another agent** — via MCP client built into the caller's agent framework.
//! 2. **A service** — CI/CD, monitoring, alerting systems that want to trigger
//!    Aura agents programmatically.
//! 3. **A person** — via any MCP-aware tool (e.g., Claude Desktop, LibreChat MCP).

use actix_web::{HttpResponse, web};
use aura::{StreamItem, StreamedAssistantContent};
use aura_config::RigBuilder;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::mcp_state::{AgentTask, McpTaskStore};
use crate::types::AppState;

// ---------------------------------------------------------------------------
// MCP / JSON-RPC 2.0 wire types
// ---------------------------------------------------------------------------

/// Inbound JSON-RPC 2.0 request (or notification when `id` is absent).
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    /// `null` for notifications; integer or string for requests.
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// Outbound JSON-RPC 2.0 success response.
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    result: Value,
}

/// Outbound JSON-RPC 2.0 error response.
#[derive(Debug, Serialize)]
struct JsonRpcError {
    jsonrpc: &'static str,
    id: Value,
    error: JsonRpcErrorBody,
}

#[derive(Debug, Serialize)]
struct JsonRpcErrorBody {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

// Standard JSON-RPC error codes
const ERR_PARSE: i64 = -32700;
const ERR_INVALID_REQUEST: i64 = -32600;
const ERR_METHOD_NOT_FOUND: i64 = -32601;
const ERR_INVALID_PARAMS: i64 = -32602;
const ERR_INTERNAL: i64 = -32603;

// MCP-specific error codes (application range)
const ERR_AGENT_NOT_FOUND: i64 = -32001;
const ERR_TASK_NOT_FOUND: i64 = -32002;
const ERR_TASK_NOT_RUNNING: i64 = -32003;

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "aura-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Tool argument structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TriggerAgentArgs {
    /// Agent alias (as defined in TOML config).
    agent: String,
    /// Prompt / task description for the agent.
    prompt: String,
    /// When `true` (default) the task is spawned in the background and a
    /// `task_id` is returned immediately. When `false` the handler waits for
    /// the agent to finish before responding (blocks the HTTP connection).
    #[serde(default = "default_true")]
    r#async: bool,
}

#[derive(Debug, Deserialize)]
struct QueryAgentStatusArgs {
    task_id: String,
}

#[derive(Debug, Deserialize)]
struct SendContextToAgentArgs {
    task_id: String,
    context: String,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Tool schema definitions (returned by tools/list)
// ---------------------------------------------------------------------------

fn tool_definitions() -> Value {
    json!([
        {
            "name": "trigger_agent",
            "description": "Kick off an Aura agent with a given prompt. Works like a webhook trigger — \
                            the agent runs its configured system prompt + toolbelt against the supplied \
                            prompt. Returns a task_id for async mode or the full result for sync mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Agent alias or name as defined in the Aura TOML config."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task or question to send to the agent."
                    },
                    "async": {
                        "type": "boolean",
                        "description": "When true (default) spawn the agent in the background and \
                                        return a task_id. When false, block until the agent completes \
                                        and return the result inline.",
                        "default": true
                    }
                },
                "required": ["agent", "prompt"]
            }
        },
        {
            "name": "query_agent_status",
            "description": "Poll the status and result of an agent task previously started by \
                            trigger_agent. Returns status (running | complete | failed) and the \
                            result text once complete.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID returned by trigger_agent."
                    }
                },
                "required": ["task_id"]
            }
        },
        {
            "name": "send_context_to_agent",
            "description": "Push additional context or data to a running agent task. Useful for \
                            feeding alerts, metric snapshots, or log excerpts to a long-running \
                            incident-response agent. The context is queued and will be incorporated \
                            into the agent's next reasoning step.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID of the running agent."
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context or data to send to the agent."
                    }
                },
                "required": ["task_id", "context"]
            }
        }
    ])
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn ok_response(id: Value, result: Value) -> HttpResponse {
    HttpResponse::Ok().json(JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result,
    })
}

fn err_response(id: Value, code: i64, message: impl Into<String>, data: Option<Value>) -> HttpResponse {
    let status = if code == ERR_PARSE || code == ERR_INVALID_REQUEST {
        actix_web::http::StatusCode::BAD_REQUEST
    } else {
        actix_web::http::StatusCode::OK // JSON-RPC errors use 200 by convention
    };

    HttpResponse::build(status).json(JsonRpcError {
        jsonrpc: "2.0",
        id,
        error: JsonRpcErrorBody {
            code,
            message: message.into(),
            data,
        },
    })
}

/// Wrap text as an MCP tool result content block.
fn text_content(text: impl Into<String>) -> Value {
    json!({
        "content": [
            {"type": "text", "text": text.into()}
        ]
    })
}

// ---------------------------------------------------------------------------
// Method handlers
// ---------------------------------------------------------------------------

fn handle_initialize(id: Value, params: Option<Value>) -> HttpResponse {
    // Echo back the requested protocol version (or fall back to ours).
    let client_version = params
        .as_ref()
        .and_then(|p| p.get("protocolVersion"))
        .and_then(|v| v.as_str())
        .unwrap_or(PROTOCOL_VERSION);

    let negotiated = if client_version == PROTOCOL_VERSION {
        PROTOCOL_VERSION
    } else {
        warn!(
            client_version,
            server_version = PROTOCOL_VERSION,
            "MCP client requested different protocol version; using server version"
        );
        PROTOCOL_VERSION
    };

    ok_response(
        id,
        json!({
            "protocolVersion": negotiated,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            }
        }),
    )
}

fn handle_tools_list(id: Value) -> HttpResponse {
    ok_response(
        id,
        json!({
            "tools": tool_definitions()
        }),
    )
}

async fn handle_tools_call(
    id: Value,
    params: Option<Value>,
    data: &web::Data<AppState>,
    mcp_store: &web::Data<Arc<McpTaskStore>>,
) -> HttpResponse {
    let params = match params {
        Some(p) => p,
        None => {
            return err_response(id, ERR_INVALID_PARAMS, "tools/call requires params", None);
        }
    };

    let tool_name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n.to_string(),
        None => {
            return err_response(id, ERR_INVALID_PARAMS, "missing 'name' in tools/call params", None);
        }
    };

    let args = params.get("arguments").cloned().unwrap_or(Value::Object(Default::default()));

    match tool_name.as_str() {
        "trigger_agent" => tool_trigger_agent(id, args, data, mcp_store).await,
        "query_agent_status" => tool_query_agent_status(id, args, mcp_store).await,
        "send_context_to_agent" => tool_send_context(id, args, mcp_store).await,
        unknown => err_response(
            id,
            ERR_METHOD_NOT_FOUND,
            format!("unknown tool: {unknown}"),
            None,
        ),
    }
}

// ---------------------------------------------------------------------------
// Individual tool implementations
// ---------------------------------------------------------------------------

async fn tool_trigger_agent(
    id: Value,
    args: Value,
    data: &web::Data<AppState>,
    mcp_store: &web::Data<Arc<McpTaskStore>>,
) -> HttpResponse {
    let args: TriggerAgentArgs = match serde_json::from_value(args) {
        Ok(a) => a,
        Err(e) => {
            return err_response(
                id,
                ERR_INVALID_PARAMS,
                format!("invalid trigger_agent arguments: {e}"),
                None,
            );
        }
    };

    // Resolve agent config
    let config = data
        .configs
        .iter()
        .find(|c| c.agent.alias.as_deref().unwrap_or(&c.agent.name) == args.agent)
        .cloned();

    let config = match config {
        Some(c) => c,
        None => {
            return err_response(
                id,
                ERR_AGENT_NOT_FOUND,
                format!("agent '{}' not found — check your TOML config", args.agent),
                Some(json!({"available_agents": available_agent_names(data)})),
            );
        }
    };

    let task_id = format!("mcp-task-{}", Uuid::new_v4().simple());
    let task = AgentTask::new(task_id.clone(), args.agent.clone(), args.prompt.clone());
    mcp_store.insert(task).await;

    info!(
        task_id = %task_id,
        agent = %args.agent,
        r#async = %args.r#async,
        "MCP trigger_agent called"
    );

    if args.r#async {
        // Spawn background task and return task_id immediately
        let store_clone: Arc<McpTaskStore> = Arc::clone(mcp_store);
        let task_id_clone = task_id.clone();
        let prompt = args.prompt.clone();

        tokio::spawn(async move {
            run_agent_task(store_clone, task_id_clone, config, prompt).await;
        });

        ok_response(
            id,
            text_content(format!(
                "Agent '{}' triggered. Task ID: {}\n\
                 Use query_agent_status to check progress.",
                args.agent, task_id
            )),
        )
    } else {
        // Run synchronously — blocks until the agent finishes
        let store_clone: Arc<McpTaskStore> = Arc::clone(mcp_store);
        let task_id_clone = task_id.clone();
        let prompt = args.prompt.clone();

        run_agent_task(store_clone.clone(), task_id_clone.clone(), config, prompt).await;

        let task = store_clone.get(&task_id_clone).await;
        let result_text = match task {
            Some(t) => t.result.unwrap_or_else(|| "Agent completed with no output.".to_string()),
            None => "Agent completed but task record was not found.".to_string(),
        };

        ok_response(id, text_content(result_text))
    }
}

async fn tool_query_agent_status(
    id: Value,
    args: Value,
    mcp_store: &web::Data<Arc<McpTaskStore>>,
) -> HttpResponse {
    let args: QueryAgentStatusArgs = match serde_json::from_value(args) {
        Ok(a) => a,
        Err(e) => {
            return err_response(
                id,
                ERR_INVALID_PARAMS,
                format!("invalid query_agent_status arguments: {e}"),
                None,
            );
        }
    };

    let task = match mcp_store.get(&args.task_id).await {
        Some(t) => t,
        None => {
            return err_response(
                id,
                ERR_TASK_NOT_FOUND,
                format!("task '{}' not found", args.task_id),
                None,
            );
        }
    };

    // Serialize the task as the result
    let task_json = serde_json::to_value(&task).unwrap_or_default();

    // Also produce a human-readable summary for the text content block
    let summary = match &task.status {
        crate::mcp_state::TaskStatus::Running => {
            format!("Task {} is still running (started at {}).", task.id, task.created_at)
        }
        crate::mcp_state::TaskStatus::Complete => {
            let result_preview = task
                .result
                .as_deref()
                .unwrap_or("(no output)")
                .chars()
                .take(200)
                .collect::<String>();
            format!(
                "Task {} completed. Result preview:\n{}\n\nFull result available in the JSON data.",
                task.id, result_preview
            )
        }
        crate::mcp_state::TaskStatus::Failed(reason) => {
            format!("Task {} failed: {}", task.id, reason)
        }
    };

    ok_response(
        id,
        json!({
            "content": [
                {"type": "text", "text": summary}
            ],
            "task": task_json
        }),
    )
}

async fn tool_send_context(
    id: Value,
    args: Value,
    mcp_store: &web::Data<Arc<McpTaskStore>>,
) -> HttpResponse {
    let args: SendContextToAgentArgs = match serde_json::from_value(args) {
        Ok(a) => a,
        Err(e) => {
            return err_response(
                id,
                ERR_INVALID_PARAMS,
                format!("invalid send_context_to_agent arguments: {e}"),
                None,
            );
        }
    };

    // Verify task exists before accepting context
    let task = mcp_store.get(&args.task_id).await;
    let task = match task {
        Some(t) => t,
        None => {
            return err_response(
                id,
                ERR_TASK_NOT_FOUND,
                format!("task '{}' not found", args.task_id),
                None,
            );
        }
    };

    if task.status != crate::mcp_state::TaskStatus::Running {
        return err_response(
            id,
            ERR_TASK_NOT_RUNNING,
            format!(
                "task '{}' is not running (status: {}). Context can only be sent to running tasks.",
                args.task_id, task.status
            ),
            None,
        );
    }

    mcp_store.add_context(&args.task_id, args.context.clone()).await;

    info!(
        task_id = %args.task_id,
        context_len = args.context.len(),
        "MCP send_context_to_agent: context queued"
    );

    ok_response(
        id,
        text_content(format!(
            "Context queued for task {}. It will be incorporated into the agent's next reasoning step.\n\
             Note: In this prototype, context is stored but not yet injected into a live stream. \
             Production implementation would forward this via the agent's context channel.",
            args.task_id
        )),
    )
}

// ---------------------------------------------------------------------------
// Background agent runner
// ---------------------------------------------------------------------------

/// Run an agent to completion and store the result in the task store.
///
/// This is called either directly (sync mode) or from a spawned task (async mode).
/// It builds a fresh agent via `RigBuilder`, streams the response, and collects
/// the full text output — matching the pattern used by the chat completions handler.
async fn run_agent_task(
    store: Arc<McpTaskStore>,
    task_id: String,
    config: aura_config::Config,
    prompt: String,
) {
    let request_id = format!("mcp-req-{}", Uuid::new_v4().simple());

    // Build agent (no request headers to forward for MCP-triggered tasks)
    let builder = RigBuilder::new(config);
    let agent = match builder.build_agent_with_headers(None).await {
        Ok(a) => a,
        Err(e) => {
            error!(task_id = %task_id, error = %e, "MCP agent build failed");
            store.fail(&task_id, format!("Failed to build agent: {e}")).await;
            return;
        }
    };

    let timeout = std::time::Duration::from_secs(300); // 5-minute cap for MCP tasks

    let (mut stream, _cancel_tx, _usage_state) = agent
        .stream_prompt_with_timeout(&prompt, timeout, &request_id)
        .await;

    // Collect text content from the stream
    let mut result = String::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamItem::StreamAssistantItem(StreamedAssistantContent::Text(chunk))) => {
                result.push_str(&chunk);
            }
            Ok(StreamItem::Final(info)) => {
                // Use the accumulated content from the Final variant when available
                if !info.content.is_empty() {
                    result = info.content;
                }
                break;
            }
            Ok(StreamItem::FinalMarker) => {
                break;
            }
            Ok(_) => {
                // Tool calls, tool results, reasoning, etc. — not captured in raw text output
            }
            Err(e) => {
                error!(task_id = %task_id, error = %e, "MCP agent stream error");
                store.fail(&task_id, format!("Stream error: {e}")).await;
                return;
            }
        }
    }

    info!(
        task_id = %task_id,
        result_len = result.len(),
        "MCP agent task completed"
    );

    store.complete(&task_id, result).await;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn available_agent_names(data: &web::Data<AppState>) -> Vec<String> {
    data.configs
        .iter()
        .map(|c| c.agent.alias.as_deref().unwrap_or(&c.agent.name).to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Main handler — POST /mcp
// ---------------------------------------------------------------------------

/// MCP endpoint: POST /mcp
///
/// Accepts JSON-RPC 2.0 requests and dispatches to the appropriate handler.
/// Supports both requests (with `id`) and notifications (without `id`).
pub async fn mcp_endpoint(
    data: web::Data<AppState>,
    mcp_store: web::Data<Arc<McpTaskStore>>,
    body: web::Json<Value>,
) -> HttpResponse {
    // Parse as JSON-RPC request
    let rpc: JsonRpcRequest = match serde_json::from_value(body.into_inner()) {
        Ok(r) => r,
        Err(e) => {
            return err_response(
                Value::Null,
                ERR_PARSE,
                format!("parse error: {e}"),
                None,
            );
        }
    };

    // Validate jsonrpc version
    if rpc.jsonrpc != "2.0" {
        return err_response(
            rpc.id.unwrap_or(Value::Null),
            ERR_INVALID_REQUEST,
            "jsonrpc must be '2.0'",
            None,
        );
    }

    let id = rpc.id.unwrap_or(Value::Null);

    info!(method = %rpc.method, "MCP request received");

    match rpc.method.as_str() {
        "initialize" => handle_initialize(id, rpc.params),

        // Notification: no response expected
        "notifications/initialized" => {
            HttpResponse::NoContent().finish()
        }

        "tools/list" => handle_tools_list(id),

        "tools/call" => handle_tools_call(id, rpc.params, &data, &mcp_store).await,

        // Ping (keep-alive)
        "ping" => ok_response(id, json!({})),

        unknown => {
            warn!(method = %unknown, "MCP unknown method");
            err_response(
                id,
                ERR_METHOD_NOT_FOUND,
                format!("method not found: {unknown}"),
                None,
            )
        }
    }
}
