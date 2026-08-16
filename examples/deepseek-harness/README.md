# AURA ⇄ DeepSeek Harness integration examples

Working configurations for pairing AURA with
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`),
DeepSeek's open-source plugin-based agent harness. Strategy and rationale:
[`docs/design/deepseek-harness-integration.md`](../../docs/design/deepseek-harness-integration.md).

dsh is a developer preview (written against `@deepseek-ai/dsh` 0.1.0-rc);
expect its config shapes to move.

## Layout

| File | Direction | What it shows |
|------|-----------|---------------|
| [`aura-backend-for-dsh.toml`](aura-backend-for-dsh.toml) | dsh → AURA | AURA agent served as an OpenAI-compatible "model" that dsh consumes; MCP tools stay AURA-side |
| [`deepseek-models-in-aura.toml`](deepseek-models-in-aura.toml) | AURA → DeepSeek | DeepSeek's hosted API as an AURA LLM via `provider = "openai"` + `base_url` |
| [`deepseek-worker-orchestration.toml`](deepseek-worker-orchestration.toml) | AURA → DeepSeek | Mixed fleet: gpt-4o coordinator, DeepSeek worker via per-worker LLM override |
| [`dsh/`](dsh/) | dsh side | Provider route (`settings.yaml`), shared-MCP patch layer, and a delegation-tool bundle plugin |

## Quick start (agent-as-model, the recommended pairing)

1. Serve AURA:

   ```sh
   export OPENAI_API_KEY="sk-..."
   CONFIG_PATH=examples/deepseek-harness/aura-backend-for-dsh.toml \
     cargo run --bin aura-web-server
   ```

   Leave `AURA_CUSTOM_EVENTS` unset — dsh expects a vanilla OpenAI SSE
   stream.

2. Register AURA as a provider in dsh: copy the `llm-pi-ai:` block from
   [`dsh/settings.yaml`](dsh/settings.yaml) into `$DSH_HOME/settings.yaml`
   (hot-reloaded, no restart), then pick the `aura-default` model in the
   dsh Web UI (`npx @deepseek-ai/dsh web`).

Both servers default to loopback and neither enforces auth — front them
with a proxy before crossing a machine boundary.
