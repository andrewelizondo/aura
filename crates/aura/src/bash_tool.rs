//! Controlled bash execution tool for Aura agents.
//!
//! Provides a sandboxed `execute_bash` tool that agents can invoke to run
//! shell commands with configurable safety controls:
//!
//! - **Allowed commands**: Whitelist of permitted executables
//! - **Working directory**: Restrict execution to a specific directory
//! - **Timeout**: Kill long-running commands
//! - **Output limits**: Truncate large stdout/stderr
//! - **Blocked patterns**: Reject dangerous command patterns

use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;
use tracing::{info, warn};

/// Default timeout for bash commands (30 seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Default maximum output size in bytes (64 KB).
const DEFAULT_MAX_OUTPUT_BYTES: usize = 65_536;

/// Default allowed commands — safe, read-only utilities.
const DEFAULT_ALLOWED_COMMANDS: &[&str] = &[
    "ls", "cat", "head", "tail", "grep", "find", "wc", "sort", "uniq", "echo", "date", "pwd",
    "env", "whoami", "uname", "df", "du", "file", "stat", "basename", "dirname", "realpath",
    "which", "curl", "jq", "sed", "awk", "cut", "tr", "tee", "xargs", "diff", "sha256sum",
    "md5sum", "base64", "tar", "gzip", "gunzip", "zip", "unzip",
];

/// Patterns that should never appear in commands, regardless of allowlist.
const BLOCKED_PATTERNS: &[&str] = &[
    "sudo ",
    "su ",
    " | bash",
    " | sh",
    " | zsh",
    "$(", // command substitution — could bypass allowlist
    "eval ",
    "exec ",
    "> /dev/",
    ">> /dev/",
    "/etc/shadow",
    "/etc/passwd",
    "~/.ssh",
    "rm -rf /",
    "mkfs",
    ":(){",  // fork bomb
    "dd if=", // raw disk ops
];

#[derive(Debug, thiserror::Error)]
pub enum BashError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Command not allowed: {0}")]
    NotAllowed(String),

    #[error("Blocked pattern detected: {0}")]
    BlockedPattern(String),

    #[error("Command timed out after {0} seconds")]
    Timeout(u64),

    #[error("Working directory does not exist: {0}")]
    InvalidWorkDir(String),
}

/// Configuration for the bash execution tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashToolConfig {
    /// Whitelist of allowed command names (e.g. `["ls", "grep", "curl"]`).
    /// If empty, uses the built-in default set of safe commands.
    #[serde(default)]
    pub allowed_commands: Vec<String>,

    /// Working directory for command execution.
    /// Defaults to the process's current directory.
    #[serde(default)]
    pub working_directory: Option<String>,

    /// Maximum wall-clock time for a command in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Maximum combined stdout+stderr size in bytes. Output is truncated beyond this.
    #[serde(default = "default_max_output")]
    pub max_output_bytes: usize,

    /// Additional blocked patterns (appended to built-in list).
    #[serde(default)]
    pub extra_blocked_patterns: Vec<String>,

    /// Environment variables to set for the command.
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
}

fn default_timeout() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

fn default_max_output() -> usize {
    DEFAULT_MAX_OUTPUT_BYTES
}

impl Default for BashToolConfig {
    fn default() -> Self {
        Self {
            allowed_commands: Vec::new(),
            working_directory: None,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            extra_blocked_patterns: Vec::new(),
            env: Default::default(),
        }
    }
}

impl BashToolConfig {
    /// Get the effective allowed commands list (user-specified or defaults).
    fn effective_allowed_commands(&self) -> Vec<String> {
        if self.allowed_commands.is_empty() {
            DEFAULT_ALLOWED_COMMANDS
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            self.allowed_commands.clone()
        }
    }

    /// Get the effective working directory.
    fn effective_work_dir(&self) -> Result<PathBuf, BashError> {
        match &self.working_directory {
            Some(dir) => {
                let path = PathBuf::from(dir);
                if path.is_dir() {
                    Ok(path)
                } else {
                    Err(BashError::InvalidWorkDir(dir.clone()))
                }
            }
            None => std::env::current_dir().map_err(BashError::IoError),
        }
    }

    /// Validate a command string against allowed commands and blocked patterns.
    fn validate_command(&self, command: &str) -> Result<(), BashError> {
        let trimmed = command.trim();

        // Check blocked patterns (built-in + user-configured)
        let lower = trimmed.to_lowercase();
        for pattern in BLOCKED_PATTERNS {
            if lower.contains(&pattern.to_lowercase()) {
                return Err(BashError::BlockedPattern(pattern.to_string()));
            }
        }
        for pattern in &self.extra_blocked_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return Err(BashError::BlockedPattern(pattern.clone()));
            }
        }

        // Extract the base command (first token, or first token in a pipeline).
        // We validate each pipeline segment independently.
        let segments: Vec<&str> = trimmed.split('|').collect();
        let allowed = self.effective_allowed_commands();

        for segment in &segments {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }

            // The command is the first word (skip leading env vars like `FOO=bar cmd`)
            let first_cmd = segment
                .split_whitespace()
                .find(|token| !token.contains('='))
                .unwrap_or("");

            // Strip path prefix to get the binary name
            let binary_name = first_cmd.rsplit('/').next().unwrap_or(first_cmd);

            if binary_name.is_empty() {
                return Err(BashError::NotAllowed("empty command".to_string()));
            }

            if !allowed.iter().any(|a| a == binary_name) {
                return Err(BashError::NotAllowed(format!(
                    "'{}' is not in the allowed commands list",
                    binary_name
                )));
            }
        }

        Ok(())
    }
}

/// Arguments for the execute_bash tool.
#[derive(Debug, Deserialize, Serialize)]
pub struct BashArgs {
    /// The bash command to execute.
    pub command: String,
}

/// Result returned from bash execution.
#[derive(Debug, Serialize)]
pub struct BashOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub truncated: bool,
}

impl std::fmt::Display for BashOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.timed_out {
            writeln!(f, "[TIMED OUT]")?;
        }
        if self.truncated {
            writeln!(f, "[OUTPUT TRUNCATED]")?;
        }
        if !self.stdout.is_empty() {
            write!(f, "{}", self.stdout)?;
        }
        if !self.stderr.is_empty() {
            if !self.stdout.is_empty() {
                writeln!(f)?;
            }
            write!(f, "[stderr] {}", self.stderr)?;
        }
        if self.exit_code != 0 {
            write!(f, "\n[exit code: {}]", self.exit_code)?;
        }
        Ok(())
    }
}

/// The Rig-compatible bash tool that agents can invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashTool {
    pub config: BashToolConfig,
}

impl BashTool {
    pub fn new(config: BashToolConfig) -> Self {
        Self { config }
    }

    /// Execute a validated command and capture output.
    async fn execute(&self, command: &str) -> Result<BashOutput, BashError> {
        self.config.validate_command(command)?;

        let work_dir = self.config.effective_work_dir()?;
        let timeout = Duration::from_secs(self.config.timeout_secs);

        info!(
            command = command,
            work_dir = %work_dir.display(),
            timeout_secs = self.config.timeout_secs,
            "Executing bash command"
        );

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command).current_dir(&work_dir);

        // Set configured environment variables
        for (key, value) in &self.config.env {
            cmd.env(key, value);
        }

        // Prevent interactive prompts
        cmd.env("DEBIAN_FRONTEND", "noninteractive");
        cmd.stdin(std::process::Stdio::null());

        let result = tokio::time::timeout(timeout, cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let mut truncated = false;

                let max = self.config.max_output_bytes;
                if stdout.len() + stderr.len() > max {
                    truncated = true;
                    // Split budget: 80% stdout, 20% stderr
                    let stdout_budget = max * 4 / 5;
                    let stderr_budget = max - stdout_budget;
                    if stdout.len() > stdout_budget {
                        stdout.truncate(stdout_budget);
                        stdout.push_str("\n... [truncated]");
                    }
                    if stderr.len() > stderr_budget {
                        stderr.truncate(stderr_budget);
                        stderr.push_str("\n... [truncated]");
                    }
                }

                let exit_code = output.status.code().unwrap_or(-1);

                info!(
                    exit_code = exit_code,
                    stdout_len = stdout.len(),
                    stderr_len = stderr.len(),
                    truncated = truncated,
                    "Bash command completed"
                );

                Ok(BashOutput {
                    exit_code,
                    stdout,
                    stderr,
                    timed_out: false,
                    truncated,
                })
            }
            Ok(Err(e)) => Err(BashError::IoError(e)),
            Err(_) => {
                warn!(
                    command = command,
                    timeout_secs = self.config.timeout_secs,
                    "Bash command timed out"
                );
                Err(BashError::Timeout(self.config.timeout_secs))
            }
        }
    }
}

impl Tool for BashTool {
    const NAME: &'static str = "execute_bash";
    type Error = BashError;
    type Args = BashArgs;
    type Output = BashOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let allowed = self.config.effective_allowed_commands();
        let allowed_str = allowed.join(", ");

        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Execute a bash command in a controlled environment. \
                 Allowed commands: {}. \
                 Commands are subject to a {} second timeout and {} byte output limit.",
                allowed_str, self.config.timeout_secs, self.config.max_output_bytes
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute. Pipelines are supported (e.g. 'ls -la | grep foo'). Only whitelisted commands are allowed."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.execute(&args.command).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BashToolConfig {
        BashToolConfig::default()
    }

    #[test]
    fn allows_default_commands() {
        let config = default_config();
        assert!(config.validate_command("ls -la").is_ok());
        assert!(config.validate_command("grep -r foo .").is_ok());
        assert!(config.validate_command("cat /tmp/test.txt").is_ok());
        assert!(config.validate_command("echo hello world").is_ok());
    }

    #[test]
    fn allows_pipelines() {
        let config = default_config();
        assert!(config.validate_command("ls -la | grep foo | wc -l").is_ok());
        assert!(config.validate_command("cat file.txt | sort | uniq").is_ok());
    }

    #[test]
    fn rejects_disallowed_commands() {
        let config = default_config();
        assert!(config.validate_command("rm -rf /tmp").is_err());
        assert!(config.validate_command("python script.py").is_err());
        assert!(config.validate_command("gcc hello.c").is_err());
        assert!(config.validate_command("apt install foo").is_err());
    }

    #[test]
    fn rejects_blocked_patterns() {
        let config = default_config();
        assert!(config.validate_command("echo foo | bash").is_err());
        assert!(config.validate_command("sudo ls").is_err());
        assert!(config.validate_command("eval echo hi").is_err());
    }

    #[test]
    fn rejects_command_substitution() {
        let config = default_config();
        assert!(config.validate_command("echo $(whoami)").is_err());
    }

    #[test]
    fn custom_allowed_commands() {
        let config = BashToolConfig {
            allowed_commands: vec!["python3".to_string(), "npm".to_string()],
            ..Default::default()
        };
        assert!(config.validate_command("python3 script.py").is_ok());
        assert!(config.validate_command("npm install").is_ok());
        // Default commands no longer allowed when custom list is set
        assert!(config.validate_command("ls").is_err());
    }

    #[test]
    fn custom_blocked_patterns() {
        let config = BashToolConfig {
            extra_blocked_patterns: vec!["--delete".to_string()],
            ..Default::default()
        };
        assert!(config.validate_command("ls --delete").is_err());
    }

    #[tokio::test]
    async fn executes_simple_command() {
        let tool = BashTool::new(default_config());
        let result = tool.execute("echo hello").await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout.trim(), "hello");
        assert!(!result.timed_out);
    }

    #[tokio::test]
    async fn executes_pipeline() {
        let tool = BashTool::new(default_config());
        let result = tool.execute("echo 'hello world' | wc -w").await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout.trim(), "2");
    }

    #[tokio::test]
    async fn captures_stderr() {
        let tool = BashTool::new(default_config());
        let result = tool.execute("ls /nonexistent_path_12345").await.unwrap();
        assert_ne!(result.exit_code, 0);
        assert!(!result.stderr.is_empty());
    }

    #[tokio::test]
    async fn respects_timeout() {
        let config = BashToolConfig {
            timeout_secs: 1,
            // sleep is not in the default allowed list, so add it
            allowed_commands: vec!["sleep".to_string()],
            ..Default::default()
        };
        let tool = BashTool::new(config);
        let result = tool.execute("sleep 10").await;
        assert!(matches!(result, Err(BashError::Timeout(_))));
    }
}
