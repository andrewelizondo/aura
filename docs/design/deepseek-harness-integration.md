# DeepSeek Harness integration: strategy note

**Status:** design note, current as of 2026-08-16. DeepSeek Harness is a
_developer preview_ (public since 2026-08-13) whose README warns of
compatibility-breaking changes; every dsh-side shape referenced here is
pinned to `@deepseek-ai/dsh` 0.1.0-rc and should be re-verified before
building on it.

## Context

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`)
is DeepSeek's open-source agent harness, MIT-licensed, written in
TypeScript. Its architecture is "everything is a plugin": the model
adapter, tool registry, session log, and the agent loop itself are all
replaceable plugins composed by the [Cordis](https://github.com/cordiverse/cordis)
meta-framework. Configuration is layered YAML: npm "bundles" carry
`cordis.patch.yml` patch layers, profiles stack bundles under
`$DSH_HOME/profiles/<name>/`, and a hot-reloaded `$DSH_HOME/settings.yaml`
is the user-facing plane.

AURA and dsh occupy overlapping but non-identical niches: dsh is an
interactive coding harness (terminal/Web UI, session-centric) with a
deliberately small core; AURA is a declarative server-side agent composer
(TOML → OpenAI-compatible API) with orchestration, scratchpad, HITL, and
RAG built in. The integration question is therefore not "port one to the
other" but "which seams already line up".

## Seam inventory

Both systems speak two common protocols today — OpenAI-compatible chat
completions and MCP — which yields six candidate surfaces:

| # | Direction | Surface | Effort |
|---|-----------|---------|--------|
| A1 | dsh → AURA | AURA as a dsh model provider (`llm-pi-ai` route) | config only |
| A2 | both | Shared MCP servers (one tool plane, two harnesses) | config only |
| A3 | dsh → AURA | dsh tool plugin delegating to an AURA agent | small JS plugin |
| B1 | AURA → DeepSeek | DeepSeek hosted models as an AURA provider | config only |
| B2 | AURA → dsh | dsh tools exposed to AURA over MCP | new dsh plugin (missing piece) |
| B3 | both | Agent-protocol bridge (AURA A2A ⇄ dsh ACP) | future work |

### A1. AURA as a model provider for dsh (recommended entry point)

dsh has no per-vendor OpenAI/Anthropic plugins; its generic adapter is
`@deepseek-ai/dsh-llm-pi-ai`, which accepts hand-declared provider routes
with `api: openai-completions`, a `baseURL` prefix, an `apiKeyEnv`
credential reference, and an explicit `models` list. AURA's web server
already serves exactly that contract: `POST /v1/chat/completions`
(streaming and non-streaming) and `GET /v1/models`
(`crates/aura-web-server/src/main.rs`, routes at the `Router::new()`
block), so dsh's "Fetch available models" discovery works against AURA
unmodified.

This is the **agent-as-model** pattern: dsh selects what it believes is a
model, but the "model" is a fully composed AURA agent — provider LLM, MCP
tools, system prompt, scratchpad, RAG — hidden behind one model id. The
model id dsh requests is the AURA agent's `alias` (falling back to
`name`), which is how `/v1/models` enumerates agents.

Constraints that make this work reliably:

- **Vanilla SSE only.** `AURA_CUSTOM_EVENTS` must stay unset for a server
  consumed by dsh; `aura.*` events are opt-in precisely so foreign
  OpenAI-compatible clients get a clean stream.
- **Context window is declared twice.** pi-ai routes declare
  `contextWindow`/`maxTokens` per model; keep them consistent with the
  AURA agent's `context_window`/`max_tokens`.
- **Auth is a no-op today.** AURA enforces no inbound bearer auth;
  pi-ai still requires an `apiKeyEnv` reference, so any set env var
  satisfies it. Fine on localhost; a real deployment needs an auth layer
  in front of AURA first (see Risks).

### A2. Shared MCP servers

AURA consumes MCP over streamable HTTP/SSE/stdio
(`[mcp.servers.<name>]`, `transport = "http_streamable"`); dsh consumes
MCP via one `@deepseek-ai/dsh-mcp-client` plugin row per server
(`transport: streamable-http` or `stdio`; it has no standalone-SSE
transport). Pointing both at the same streamable-HTTP MCP server gives
both harnesses one tool plane with zero new code. Naming differs by
convention only: dsh prefixes tools as `mcp__<serverName>__<tool>`, AURA
keeps raw names filtered through `mcp_filter` globs.

### A3. dsh delegation tool (`dsh-plugin-aura`)

Where A1 replaces dsh's model wholesale, a delegation tool keeps dsh on
its own model and hands specific sub-questions to an AURA agent as a
single tool call (`aura_query` → `POST /v1/chat/completions`,
`stream: false`). This is a complete third-party dsh bundle — a
`package.json` with `"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }`,
a patch inserting the plugin row, and an ESM `index.js` exporting
`name`/`inject`/`Config`/`apply` that registers the tool via
`ctx.tools.register(defineTool({...}))`. It doubles as the reference
skeleton for any future AURA-flavored dsh plugin (B2).

### B1. DeepSeek hosted models inside AURA

Zero code: DeepSeek's hosted API is OpenAI-compatible, and AURA's
`provider = "openai"` variant takes a `base_url` override
(`crates/aura-config/src/config.rs`, `LlmConfig::OpenAi`;
`examples/reference.toml` documents the pattern for other
OpenAI-compatible vendors). `base_url = "https://api.deepseek.com/v1"`
with `model = "deepseek-chat"` works for a single agent, and per-worker
LLM overrides (`[orchestration.worker.<name>.llm]`, resolved wholesale in
`create_worker` — no field merging) let a mixed fleet run DeepSeek workers
under a coordinator on another provider.

A dedicated `deepseek` provider variant is **not** warranted now. The one
open question is reasoning traffic: `deepseek-reasoner` returns
`reasoning_content` in the OpenAI wire format, and AURA's `openrouter`
variant exists specifically because OpenRouter's `reasoning_details`
differs. Verify `reasoning_content` passthrough on the Rig fork before
recommending `deepseek-reasoner` for production; if it needs wire-format
handling, that is the trigger to add a variant (the checklist is ~7 sites:
config enum + accessors, `ProviderAgent`, builder arm, token counter, CLI
init, examples).

### B2. dsh tools into AURA — the missing piece

dsh ships an MCP *client* but no MCP *server*; nothing today exposes
`ctx.tools` to external consumers (its ACP bridge explicitly rejects
`mcpServers`). Making dsh's tool inventory callable from AURA requires
authoring a dsh plugin that reads `ctx.tools.schemas()` and mounts a
streamable-HTTP MCP endpoint (via `ctx.webServer.register(...)` or its own
listener). That is genuinely new code on the dsh side, well-scoped, and
the natural second deliverable after the examples here; AURA needs no
changes to consume it.

### B3. Agent-protocol bridge

AURA exposes an A2A server (`crates/aura-web-server/src/a2a/`); dsh
exposes ACP and a newline-JSON-RPC SDK runtime, both over stdio. The
protocols do not match, so harness-to-harness sessions (beyond the
model-shaped seam of A1) need a translation shim. Park this until a
concrete use case demands session semantics that A1/A3 cannot express
(mid-session interjection, permission round-trips).

## Recommendation

Adopt in three tiers:

1. **Now (config only, shipped as examples):** A1 + A2 + B1. All three
   are pure configuration against released behavior on both sides. See
   `examples/deepseek-harness/` — AURA TOMLs at the top level, dsh-side
   YAML and the bundle plugin under `dsh/`.
2. **Next (small code, dsh side):** A3 hardening if delegation proves
   useful beyond the example, then B2 (the MCP-server plugin), published
   as an npm bundle with the `dsh-plugin` GitHub topic.
3. **Later, on demand:** B3, plus an upstream contribution registering an
   AURA catalog route in pi-ai once dsh exits developer preview.

## Risks and caveats

- **Preview churn.** dsh is pre-1.0 with promised breaking changes; the
  pi-ai route schema, patch-layer semantics, and plugin API can all move.
  The examples name the rc version they were written against.
- **No auth on either side.** AURA's completion routes are unauthenticated
  and dsh's web server has no TLS/auth/origin policy. Both defaults are
  localhost-only; anything crossing a machine boundary needs a fronting
  proxy before these integrations are exposed.
- **Patch layers replace, not merge.** A dsh patch row replaces the whole
  `config` of the row it overrides — a partial provider tweak in a later
  layer silently drops fields. Prefer `settings.yaml` for user-plane edits.
- **Timeout stacking (A3).** An `aura_query` call nests AURA's own agent
  loop (with MCP calls and retries) inside one dsh tool call; the plugin's
  `timeoutMs` must exceed AURA's worst-case turn, or dsh will abandon
  work AURA completes.

## Pointers

- Examples: `examples/deepseek-harness/README.md`
- dsh source (read-only reference used for this note):
  `github.com/deepseek-ai/deepseek-harness` — `packages/llm/llm-pi-ai`
  (provider routes), `packages/mcp/mcp-client` (MCP rows),
  `docs/user/develop/basic/publish.md` (bundle packaging),
  `docs/architecture.md` (plugin model)
- AURA seams: `crates/aura-web-server/src/main.rs` (routes),
  `crates/aura-config/src/config.rs` (`LlmConfig`, `McpServerConfig`),
  `crates/aura/src/orchestration/orchestrator.rs` (worker LLM override)
