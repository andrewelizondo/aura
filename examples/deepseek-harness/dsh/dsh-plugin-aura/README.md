# dsh-plugin-aura

A minimal third-party [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (dsh) bundle plugin that turns dsh into an **operator console** for an [AURA](../../../..) supervisor.

AURA is "systemd for agents": a long-running daemon that loads declaratively configured agent *units* — TOML files composing an LLM, system prompt, MCP tool servers, and (optionally) a graph of orchestration workers. This plugin gives the dsh agent an `auractl`-style toolset over AURA's HTTP surface: list the loaded units, check the daemon's health, and dispatch work to a unit.

Plain ESM JavaScript, no build step.

## The analogy

| systemd world | AURA world |
| --- | --- |
| unit file (`.service`) | agent TOML config |
| the supervisor / init (PID 1) | `aura-web-server` |
| `systemctl list-units` | `aura_units` |
| `systemctl status` | `aura_status` |
| `systemctl start` / `busctl call` | `aura_invoke` |
| dependency graph a unit brings up | orchestration workers |
| sockets/resources attached to a unit | MCP servers |

## Tools

### `aura_units` — list loaded agent units

No parameters. Fetches `/aura/info` and `/v1/models` concurrently and joins them, returning `{ defaultUnit, units: [{ id, model, ownedBy, workers, mcpServers }] }`. Rendered as one block per unit with its workers (and any per-worker model override) and MCP attachments.

The dsh model is told to call this **first**, to discover which units exist and what capabilities (workers, MCP tools) each carries before invoking one.

### `aura_status` — check the supervisor daemon

No parameters. Fetches `/health` and returns it as-is; rendered as a one-liner like `aura 1.4.0 — healthy; session store redis (ping 2ms)`. Use when an invocation fails or before a batch of work.

### `aura_invoke` — dispatch work to a unit

| param | required | meaning |
| --- | --- | --- |
| `prompt` | yes | the complete, self-contained question or task for the AURA agent |
| `unit` | no | agent unit id (its alias/name, as listed by `aura_units`); omit for the configured default |

POSTs to AURA's OpenAI-compatible `/v1/chat/completions` (`stream: false`, `model` = the unit id) and returns `choices[0].message.content`. The unit runs its own MCP tools / RAG / orchestration server-side; dsh sees only the grounded final answer.

Typical dsh-side flow:

1. `aura_units` → discover `sre-orchestrator` carries log-analysis workers and a Mezmo MCP server.
2. `aura_invoke { unit: "sre-orchestrator", prompt: "summarize error spikes in the last hour" }` → grounded answer.
3. On failure: `aura_status` → is the daemon itself degraded?

## Install

Start the AURA backend first:

```sh
CONFIG_PATH=examples/deepseek-harness/aura-backend-for-dsh.toml cargo run --bin aura-web-server
```

Then install the bundle into a dsh profile and verify the layer:

```sh
dsh plugin --profile <p> add ./dsh-plugin-aura
dsh --profile <p> --dump-config     # shows a "# == dsh-plugin-aura" layer
dsh --profile <p>
```

`dsh plugin --profile <p> remove dsh-plugin-aura` removes both the dependency and the layer.

## Config

Set in `cordis.patch.yml` (this bundle's defaults) or override the `aura-delegate` row in your profile's own patch layer:

| key | default | meaning |
| --- | --- | --- |
| `serverURL` | `http://127.0.0.1:8080` | AURA server root; the plugin derives `/v1`, `/health`, and `/aura/info` from it |
| `defaultUnit` | `aura-default` | default agent unit for `aura_invoke` (its alias/name) |
| `apiKeyEnv` | unset | env var holding a bearer token (AURA ignores it today) |
| `timeoutMs` | `120000` | whole-request deadline; a unit may run several MCP tool calls before answering |

Note: AURA enforces **no inbound auth** today (a bearer token is accepted and ignored), and both sides default to loopback — keep it that way unless AURA sits behind an authenticating proxy.

## Alternative: AURA as a full provider

This plugin makes AURA a *toolset* in a dsh agent's toolbox. If you instead want AURA to *be the model* — every turn of a dsh session answered by an AURA unit — register it as a custom `llm-pi-ai` provider route; see [`../settings.yaml`](../settings.yaml) (user plane, hot-reloaded) or [`../cordis.patch.yml`](../cordis.patch.yml) (composition plane).
