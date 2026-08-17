/**
 * dsh plugin: turns dsh into an interactive operator console for an AURA
 * supervisor ("systemd for agents"). AURA is a long-running daemon that holds
 * declaratively configured agent units (TOML "unit files": LLM, system prompt,
 * MCP tool servers, optional orchestration workers). This plugin registers an
 * auractl-style toolset against it:
 *
 *   - `aura_units`  — list loaded agent units and their capabilities
 *                     (systemctl list-units analog)
 *   - `aura_status` — check the supervisor daemon's health
 *                     (systemctl status analog)
 *   - `aura_invoke` — dispatch work to a unit and return its final answer
 *                     (systemctl start / busctl call analog)
 *
 * Namespace plugin (named exports, no default export).
 * @module dsh-plugin-aura
 */

import z from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'aura'
export const inject = ['tools']

export const Config = z.object({
  // AURA server root; the plugin derives /v1, /health, and /aura/info from it.
  serverURL: z.string().default('http://127.0.0.1:8080'),
  // Default agent unit to dispatch to (unit id = the agent's alias/name in
  // its TOML config).
  defaultUnit: z.string().default('aura-default'),
  // Optional env var holding a bearer token. AURA enforces no inbound auth
  // today, so this stays unset unless AURA sits behind an authenticating proxy.
  apiKeyEnv: z.string(),
  // Whole-request deadline (ms). AURA answers can be slow: a unit may run
  // several MCP tool calls (or a whole orchestration plan) before replying.
  timeoutMs: z.number().default(120000),
})

export function apply(ctx, config) {
  const root = config.serverURL.replace(/\/+$/, '')

  const authHeaders = (extra = {}) => {
    const headers = { ...extra }
    if (config.apiKeyEnv) {
      const key = process.env[config.apiKeyEnv]
      if (!key) {
        throw new Error(`dsh-plugin-aura: apiKeyEnv "${config.apiKeyEnv}" is configured but the environment variable is unset or empty`)
      }
      headers.authorization = `Bearer ${key}`
    }
    return headers
  }

  // Honor caller cancellation AND the configured deadline, whichever fires
  // first (AbortSignal.any adopts the first source's reason).
  const deadline = execSignal => AbortSignal.any([execSignal, AbortSignal.timeout(config.timeoutMs)])

  const getJson = async (path, signal) => {
    const url = `${root}${path}`
    const res = await fetch(url, { headers: authHeaders(), signal })
    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(
        `AURA returned HTTP ${res.status} for GET ${url}`
        + (body ? ` — ${body.slice(0, 500)}` : ''),
      )
    }
    return res.json()
  }

  // register() is effect-based: disposing the plugin fiber unregisters every
  // tool registered here.
  ctx.tools.register(defineTool({
    name: 'aura_units',
    description:
      'List the agent units loaded by the AURA supervisor (systemctl '
      + 'list-units analog): each unit\'s id, model, orchestration workers, '
      + 'and attached MCP tool servers. Use this first to discover which '
      + 'units exist and what capabilities each carries before dispatching '
      + 'work to one with aura_invoke.',
    // defineTool requires a parameters object; this tool takes none.
    parameters: {},
    output: {
      schema: {
        type: 'json',
        description: '{ defaultUnit, units: [{ id, model, ownedBy, workers, mcpServers }] }',
      },
      render: (_args, value) => {
        const lines = []
        for (const unit of value.units) {
          const marker = unit.id === value.defaultUnit ? ' (default)' : ''
          const provider = unit.ownedBy ? `, provider: ${unit.ownedBy}` : ''
          lines.push(`● ${unit.id}${marker} (model: ${unit.model}${provider})`)
          for (const worker of unit.workers) {
            const override = worker.model ? ` [model: ${worker.model}]` : ''
            lines.push(`  └─ worker ${worker.name} — ${worker.description}${override}`)
          }
          for (const [serverName, mcp] of Object.entries(unit.mcpServers)) {
            lines.push(`  └─ mcp ${serverName} (${mcp.transport}) ${mcp.url ?? mcp.command ?? ''}`)
          }
        }
        if (lines.length === 0) lines.push('no agent units loaded')
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    async execute(_args, exec) {
      const signal = deadline(exec.signal)
      const [info, models] = await Promise.all([
        getJson('/aura/info', signal),
        getJson('/v1/models', signal),
      ])
      // Join agent id ↔ model id to attach the provider (`owned_by`).
      const ownedBy = new Map((models?.data ?? []).map(model => [model.id, model.owned_by]))
      return {
        defaultUnit: info?.default_agent ?? null,
        units: (info?.agents ?? []).map(agent => ({
          id: agent.id,
          model: agent.model,
          ownedBy: ownedBy.get(agent.id) ?? null,
          workers: agent.workers ?? [],
          mcpServers: agent.mcp_servers ?? {},
        })),
      }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'aura_status',
    description:
      'Check whether the AURA supervisor daemon is up and healthy (systemctl '
      + 'status analog): version, overall status, and session-store '
      + 'connectivity. Use when an aura_invoke call fails or before '
      + 'dispatching a batch of work.',
    parameters: {},
    output: {
      schema: { type: 'json', description: 'The /health response as-is.' },
      render: (_args, value) => {
        const store = value?.session_store
        const ping = store?.ping
        const pingText = ping?.ok
          ? `ping ${ping.latency_ms}ms`
          : `error: ${ping?.error ?? 'unknown'}`
        return [{
          type: 'text',
          text: `aura ${value?.aura_version ?? '?'} — ${value?.status ?? 'unknown'}; `
            + `session store ${store?.backend ?? '?'} (${pingText})`,
        }]
      },
    },
    async execute(_args, exec) {
      return getJson('/health', deadline(exec.signal))
    },
  }))

  ctx.tools.register(defineTool({
    name: 'aura_invoke',
    description:
      'Dispatch work to an AURA agent unit and wait for its answer '
      + '(systemctl start / busctl call analog). The unit runs its own tools '
      + '(MCP servers, RAG, orchestration workers) on the server side and '
      + 'returns a grounded final answer as text. Use for questions that need '
      + 'a unit\'s data sources rather than your own tools; list units with '
      + 'aura_units first.',
    parameters: {
      prompt: {
        type: 'string',
        required: true,
        description: 'The complete, self-contained question or task for the AURA agent.',
      },
      unit: {
        type: 'string',
        description:
          'Agent unit id to address (its alias/name, as listed by '
          + 'aura_units). Omit for the configured default.',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const unit = args.unit ?? config.defaultUnit
      const signal = deadline(exec.signal)

      const url = `${root}/v1/chat/completions`
      const res = await fetch(url, {
        method: 'POST',
        headers: authHeaders({ 'content-type': 'application/json' }),
        body: JSON.stringify({
          model: unit,
          stream: false,
          messages: [{ role: 'user', content: args.prompt }],
        }),
        signal,
      })

      if (!res.ok) {
        const body = await res.text().catch(() => '')
        throw new Error(
          `aura_invoke: AURA returned HTTP ${res.status} for unit "${unit}" at ${url}`
          + (body ? ` — ${body.slice(0, 500)}` : ''),
        )
      }

      const data = await res.json()
      const content = data?.choices?.[0]?.message?.content
      if (typeof content !== 'string') {
        throw new Error(`aura_invoke: AURA response for unit "${unit}" carried no choices[0].message.content`)
      }
      return content
    },
  }))
}
