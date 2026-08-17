# DeepSeek Harness (dsh) ⇄ AURA: the dsh side

The AURA-side configs live one directory up (`../aura-backend-for-dsh.toml` starts the OpenAI-compatible backend these examples talk to). This directory holds the three dsh-side integration patterns — they are independent; pick per use case.

## 1. AURA as a model provider — [`settings.yaml`](settings.yaml)

A `$DSH_HOME/settings.yaml` fragment registering AURA as a custom `llm-pi-ai` provider route (`api: openai-completions`, `baseURL: http://127.0.0.1:8080/v1`). dsh then treats each AURA agent as a selectable model; behind that model id AURA composes the LLM, system prompt, and MCP tools. Hot-reloaded, no restart.

**Use when** you want whole dsh sessions answered by an AURA agent — AURA is the brain, dsh is the surface.

## 2. Shared tool plane + composition-plane provider — [`cordis.patch.yml`](cordis.patch.yml)

A profile patch layer with (a) the same provider route declared in the composition plane instead of user settings, and (b) an `@deepseek-ai/dsh-mcp-client` row pointing at the *same* upstream MCP server the AURA example config uses. Both agents then see the same tools: dsh calls them directly as `mcp__docs__<tool>`, AURA uses them server-side.

**Use when** the provider route should ship with a profile/bundle rather than live in user settings, or when dsh and AURA should share one MCP tool source.

## 3. dsh as an AURA operator console — [`dsh-plugin-aura/`](dsh-plugin-aura/)

A complete, minimal third-party bundle plugin (plain ESM, no build step) that treats AURA as "systemd for agents" and gives dsh an `auractl`-style toolset over it: `aura_units` lists the loaded agent units and their capabilities (workers, MCP servers), `aura_status` checks the supervisor daemon's health, and `aura_invoke` dispatches work to a unit and returns its grounded final answer. dsh keeps its own model and tools; AURA's units become dispatchable services.

**Use when** the dsh agent should stay primary, operating the AURA supervisor interactively — discover units, check the daemon, and *delegate* questions needing a unit's data sources (MCP/RAG/orchestration).

|  | dsh's model | AURA's role | dsh sees AURA's tools? |
| --- | --- | --- | --- |
| provider route (1, 2a) | the AURA agent | is the model | no — results only |
| shared MCP (2b) | any | sibling agent | yes — same server, called directly |
| operator-console plugin (3) | any | supervised units | no — unit metadata + final answers only |
