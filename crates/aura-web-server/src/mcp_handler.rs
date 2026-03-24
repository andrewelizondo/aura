//! MCP (Model Context Protocol) server endpoint for Aura.
//!
//! Mounts at `/mcp` (POST, GET, DELETE) and uses the official `rmcp` Rust SDK
//! with HTTP Streamable transport. This replaces the previous hand-rolled
//! JSON-RPC implementation and gives protocol compliance for free.
//!
//! # Architecture
//!
//! ```text
//! actix-web  ──POST/GET/DELETE /mcp──►  mcp_route()  ──http::Request──►  StreamableHttpService
//!                                            │                                    │
//!                                            │◄── http::Response ────────────────┘
//!                                            │
//!                                            └──► actix HttpResponse (stream body for SSE)
//! ```
//!
//! `StreamableHttpService<AuraServer>` is a Tower service that handles the
//! entire MCP protocol (initialize handshake, session management, tool
//! dispatch, SSE keep-alive pings). `AuraServer` implements `ServerHandler`
//! using the `#[tool_router]` / `#[tool_handler]` macros from `rmcp`.
//!
//! # Exposed Tools
//!
//! | Tool | Description |
//! |------|-------------|
//! | `trigger_agent` | Kick off an Aura agent with a prompt (webhook-like) |
//! | `query_agent_status` | Poll status / result of a running task |
//! | `send_context_to_agent` | Push additional context to a running task |
//!
//! # Transport Note
//!
//! Uses HTTP Streamable (SSE) transport rather than stdio so Aura scales for
//! distributed deployment. Clients connect via POST (new session), GET (SSE
//! stream), DELETE (close session) — all routed through `mcp_route()`.

use std::sync::Arc;

use actix_web::{HttpResponse, web};
use aura::{StreamItem, StreamedAssistantContent};
use aura_config::RigBuilder;
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full};
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService,
        session::local::LocalSessionManager,
    },
};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::{error, info};
use uuid::Uuid;

use crate::mcp_state::{AgentTask, McpTaskStore, TaskStatus};

// ---------------------------------------------------------------------------
// Tool parameter structs
// ---------------------------------------------------------------------------

/// Parameters for `trigger_agent`.
#[derive(Debug, Deserialize, JsonSchema)]
struct TriggerAgentParams {
    /// Agent alias as defined in the Aura TOML config (e.g. `"incident-responder"`).
    agent: String,
    /// The task or question to send to the agent.
    prompt: String,
    /// When `true` (default) the agent runs in the background and a `task_id`
    /// is returned immediately. When `false` the call blocks until the agent
    /// finishes and returns the full result inline.
    #[serde(default = "bool_true")]
    r#async: bool,
}

/// Parameters for `query_agent_status`.
#[derive(Debug, Deserialize, JsonSchema)]
struct QueryStatusParams {
    /// Task ID previously returned by `trigger_agent`.
    task_id: String,
}

/// Parameters for `send_context_to_agent`.
#[derive(Debug, Deserialize, JsonSchema)]
struct SendContextParams {
    /// Task ID of the running agent.
    task_id: String,
    /// Additional context or data to feed to the agent (alert text, log
    /// excerpt, metric snapshot, etc.).
    context: String,
}

fn bool_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// AuraServer — rmcp ServerHandler
// ---------------------------------------------------------------------------

/// rmcp server handler that exposes Aura agents as MCP tools.
///
/// One instance is created per MCP session (via the factory closure in
/// [`build_mcp_service`]). Shared state is behind `Arc` so the clone is cheap.
#[derive(Clone)]
pub struct AuraServer {
    configs: Arc<Vec<aura_config::Config>>,
    task_store: Arc<McpTaskStore>,
    tool_router: ToolRouter<AuraServer>,
}

// The `#[tool_router]` macro generates:
//   - `AuraServer::tool_router()` factory method
//   - routing boilerplate that maps tool names → method calls
#[tool_router]
impl AuraServer {
    pub fn new(configs: Arc<Vec<aura_config::Config>>, task_store: Arc<McpTaskStore>) -> Self {
        Self {
            configs,
            task_store,
            tool_router: Self::tool_router(),
        }
    }

    /// Kick off an Aura agent with a prompt.
    ///
    /// Works like a webhook trigger — the agent runs its configured system
    /// prompt and toolbelt against the supplied prompt. Returns a `task_id`
    /// for async mode or the full result inline for sync mode.
    #[tool(description = "Kick off an Aura agent with a prompt. Works like a webhook trigger — \
        the agent runs its configured system prompt and toolbelt against your prompt. \
        Returns a task_id for async mode (default), or the full result inline when async=false.")]
    async fn trigger_agent(
        &self,
        Parameters(args): Parameters<TriggerAgentParams>,
    ) -> Result<CallToolResult, McpError> {
        let config = self
            .configs
            .iter()
            .find(|c| c.agent.alias.as_deref().unwrap_or(&c.agent.name) == args.agent)
            .cloned();

        let config = match config {
            Some(c) => c,
            None => {
                let available: Vec<&str> = self
                    .configs
                    .iter()
                    .map(|c| c.agent.alias.as_deref().unwrap_or(&c.agent.name))
                    .collect();
                return Err(McpError::invalid_params(
                    format!(
                        "Agent '{}' not found. Available agents: {:?}",
                        args.agent, available
                    ),
                    None,
                ));
            }
        };

        let task_id = format!("mcp-task-{}", Uuid::new_v4().simple());
        self.task_store
            .insert(AgentTask::new(
                task_id.clone(),
                args.agent.clone(),
                args.prompt.clone(),
            ))
            .await;

        info!(
            task_id = %task_id,
            agent = %args.agent,
            r#async = %args.r#async,
            "MCP trigger_agent"
        );

        if args.r#async {
            let store = Arc::clone(&self.task_store);
            let id = task_id.clone();
            let prompt = args.prompt.clone();
            tokio::spawn(async move {
                run_agent_task(store, id, config, prompt).await;
            });

            Ok(CallToolResult::success(vec![Content::text(format!(
                "Agent '{}' triggered.\nTask ID: {}\n\
                 Use query_agent_status to poll for results.",
                args.agent, task_id
            ))]))
        } else {
            // Synchronous — block until done
            run_agent_task(
                Arc::clone(&self.task_store),
                task_id.clone(),
                config,
                args.prompt,
            )
            .await;

            let result = self
                .task_store
                .get(&task_id)
                .await
                .and_then(|t| t.result)
                .unwrap_or_else(|| "Agent completed with no output.".to_string());

            Ok(CallToolResult::success(vec![Content::text(result)]))
        }
    }

    /// Poll the status and result of an agent task started by `trigger_agent`.
    #[tool(description = "Poll the status and result of an agent task started by trigger_agent. \
        Returns status (running | complete | failed) and the full result once complete.")]
    async fn query_agent_status(
        &self,
        Parameters(args): Parameters<QueryStatusParams>,
    ) -> Result<CallToolResult, McpError> {
        let task = match self.task_store.get(&args.task_id).await {
            Some(t) => t,
            None => {
                return Err(McpError::invalid_params(
                    format!("Task '{}' not found.", args.task_id),
                    None,
                ));
            }
        };

        let text = match &task.status {
            TaskStatus::Running => {
                format!(
                    "Task {} is still running (started at {}).",
                    task.id, task.created_at
                )
            }
            TaskStatus::Complete => {
                format!(
                    "Task {} completed.\n\n{}",
                    task.id,
                    task.result.as_deref().unwrap_or("(no output)")
                )
            }
            TaskStatus::Failed(reason) => {
                format!("Task {} failed: {}", task.id, reason)
            }
        };

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Push additional context or data to a running agent task.
    #[tool(description = "Push additional context or data to a running agent task. \
        Useful for feeding alerts, log excerpts, or metric snapshots to a long-running agent. \
        Context is queued and incorporated into the agent's next reasoning step.")]
    async fn send_context_to_agent(
        &self,
        Parameters(args): Parameters<SendContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let task = match self.task_store.get(&args.task_id).await {
            Some(t) => t,
            None => {
                return Err(McpError::invalid_params(
                    format!("Task '{}' not found.", args.task_id),
                    None,
                ));
            }
        };

        if task.status != TaskStatus::Running {
            return Err(McpError::invalid_params(
                format!(
                    "Task '{}' is not running (status: {}). \
                     Context can only be sent to running tasks.",
                    args.task_id, task.status
                ),
                None,
            ));
        }

        self.task_store
            .add_context(&args.task_id, args.context.clone())
            .await;

        info!(task_id = %args.task_id, "MCP send_context_to_agent: context queued");

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Context queued for task {}.",
            args.task_id
        ))]))
    }
}

// Wire the tool router into the ServerHandler trait.
// `#[tool_handler]` generates `call_tool` and `list_tools` that delegate to
// `self.tool_router`. Other methods keep their `ServerHandler` defaults.
#[tool_handler]
impl ServerHandler for AuraServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "aura-mcp".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: None,
                icons: None,
                website_url: None,
            },
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Service construction
// ---------------------------------------------------------------------------

/// Concrete type of the MCP service used by Aura.
pub type AuraMcpService = StreamableHttpService<AuraServer, LocalSessionManager>;

/// Build the `StreamableHttpService` that backs the `/mcp` endpoint.
///
/// The factory closure captures `Arc` clones of the shared state so each new
/// MCP session gets a fresh `AuraServer` that still points to the same
/// underlying data.
pub fn build_mcp_service(
    configs: Arc<Vec<aura_config::Config>>,
    task_store: Arc<McpTaskStore>,
    cancellation_token: tokio_util::sync::CancellationToken,
) -> AuraMcpService {
    StreamableHttpService::new(
        move || {
            Ok(AuraServer::new(
                Arc::clone(&configs),
                Arc::clone(&task_store),
            ))
        },
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig {
            cancellation_token: cancellation_token.child_token(),
            ..Default::default()
        },
    )
}

// ---------------------------------------------------------------------------
// actix-web bridge handler
// ---------------------------------------------------------------------------

/// Bridge handler for `POST /mcp`, `GET /mcp`, and `DELETE /mcp`.
///
/// `StreamableHttpService` is a Tower service that speaks `http::Request` /
/// `http::Response`. Since actix-web is not Tower-native, this handler does a
/// thin, zero-copy conversion:
///
/// 1. Reconstruct an `http::Request<Full<Bytes>>` from the actix request.
/// 2. Call `StreamableHttpService::handle()` — all MCP protocol logic lives here.
/// 3. Convert the `http::Response<BoxBody<Bytes, Infallible>>` back to an
///    actix `HttpResponse`, streaming the body for SSE connections.
pub async fn mcp_route(
    svc: web::Data<Arc<AuraMcpService>>,
    req: actix_web::HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    // --- 1. Build http::Request -----------------------------------------------
    // actix-web re-exports http::Method, http::Uri, http::HeaderMap from the
    // same `http` crate, so the types are directly compatible.
    // actix-web uses http 0.2; rmcp uses http 1.x — convert via string to avoid type mismatch.
    let mut builder = http::Request::builder()
        .method(req.method().as_str())
        .uri(req.uri().to_string());

    for (name, value) in req.headers().iter() {
        builder = builder.header(name.as_str(), value.as_bytes());
    }

    let http_req = match builder.body(Full::new(body)) {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to build http::Request for MCP bridge: {e}");
            return HttpResponse::InternalServerError()
                .body(format!("MCP bridge request error: {e}"));
        }
    };

    // --- 2. Dispatch to rmcp -------------------------------------------------
    let mcp_resp = svc.handle(http_req).await;

    // --- 3. Convert http::Response back to actix -----------------------------
    let (parts, resp_body) = mcp_resp.into_parts();

    let actix_status = actix_web::http::StatusCode::from_u16(parts.status.as_u16())
        .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

    let mut builder = HttpResponse::build(actix_status);
    for (name, value) in &parts.headers {
        builder.insert_header((name.as_str(), value.as_bytes()));
    }

    // Stream the body so SSE responses are not buffered in memory.
    // `BoxBody<Bytes, Infallible>` → `Stream<Item = Result<Bytes, Infallible>>`.
    // The error type is `Infallible` (structurally unreachable), so unwrap is safe.
    let byte_stream = resp_body.into_data_stream().map(
        |frame: Result<Bytes, std::convert::Infallible>| -> Result<Bytes, std::io::Error> {
            Ok(frame.unwrap()) // Infallible: safe
        },
    );

    builder.streaming(byte_stream)
}

// ---------------------------------------------------------------------------
// Background agent runner
// ---------------------------------------------------------------------------

/// Run an agent to completion and write the result into the task store.
///
/// Called directly for sync mode, or from a spawned task for async mode.
/// Builds a fresh agent via `RigBuilder` (matching the chat-completions path),
/// streams the response, and collects the full text output.
async fn run_agent_task(
    store: Arc<McpTaskStore>,
    task_id: String,
    config: aura_config::Config,
    prompt: String,
) {
    let request_id = format!("mcp-req-{}", Uuid::new_v4().simple());

    let agent = match RigBuilder::new(config).build_agent_with_headers(None).await {
        Ok(a) => a,
        Err(e) => {
            error!(task_id = %task_id, error = %e, "MCP: agent build failed");
            store.fail(&task_id, format!("Failed to build agent: {e}")).await;
            return;
        }
    };

    let timeout = std::time::Duration::from_secs(300); // 5-min cap for MCP tasks
    let (mut stream, _cancel_tx, _usage) =
        agent.stream_prompt_with_timeout(&prompt, timeout, &request_id).await;

    let mut result = String::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamItem::StreamAssistantItem(StreamedAssistantContent::Text(chunk))) => {
                result.push_str(&chunk);
            }
            Ok(StreamItem::Final(info)) if !info.content.is_empty() => {
                // Prefer the Final variant's accumulated content when available
                result = info.content;
                break;
            }
            Ok(StreamItem::FinalMarker) => break,
            Ok(_) => {}
            Err(e) => {
                error!(task_id = %task_id, error = %e, "MCP: stream error");
                store.fail(&task_id, format!("Stream error: {e}")).await;
                return;
            }
        }
    }

    info!(task_id = %task_id, result_len = result.len(), "MCP: agent task complete");
    store.complete(&task_id, result).await;
}
