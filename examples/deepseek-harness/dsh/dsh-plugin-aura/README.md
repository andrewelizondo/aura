# dsh-plugin-aura

A minimal third-party [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (dsh) bundle plugin that registers one tool, `aura_query`, letting a dsh agent delegate a sub-question to an [AURA](../../../..) agent as a single tool call.

The AURA agent is a full server-side composition — its own LLM, system prompt, MCP tool servers, and (optionally) multi-agent orchestration. `aura_query` POSTs the prompt to AURA's OpenAI-compatible `/v1/chat/completions` endpoint (`stream: false`) and returns `choices[0].message.content` as the tool result, so the dsh agent sees a grounded final answer without ever seeing AURA's tools.

Plain ESM JavaScript, no build step.

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
| `baseURL` | `http://127.0.0.1:8080/v1` | AURA server base URL |
| `model` | `aura-default` | default AURA agent (its alias/name) |
| `apiKeyEnv` | unset | env var holding a bearer token (AURA ignores it today) |
| `timeoutMs` | `120000` | whole-request deadline; AURA may run several MCP tool calls before answering |

The model can also be chosen per call: `aura_query` takes an optional `model` argument.

## Alternative: AURA as a full provider

This plugin makes AURA *one tool* in a dsh agent's toolbox. If you instead want AURA to *be the model* — every turn of a dsh session answered by the AURA agent — register it as a custom `llm-pi-ai` provider route; see [`../settings.yaml`](../settings.yaml) (user plane, hot-reloaded) or [`../cordis.patch.yml`](../cordis.patch.yml) (composition plane).
