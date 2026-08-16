/**
 * dsh plugin: registers an `aura_query` tool that delegates a sub-question to
 * an AURA agent through AURA's OpenAI-compatible /v1/chat/completions endpoint
 * (non-streaming). The AURA agent runs its own MCP tools / RAG server-side;
 * dsh receives only the final assistant text.
 *
 * Namespace plugin (named exports, no default export).
 * @module dsh-plugin-aura
 */

import z from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'aura'
export const inject = ['tools']

export const Config = z.object({
  // AURA server base URL; /chat/completions is appended per request.
  baseURL: z.string().default('http://127.0.0.1:8080/v1'),
  // Default AURA agent (model id = the agent's alias/name in its TOML config).
  model: z.string().default('aura-default'),
  // Optional env var holding a bearer token. AURA enforces no inbound auth
  // today, so this stays unset unless AURA sits behind an authenticating proxy.
  apiKeyEnv: z.string(),
  // Whole-request deadline (ms). AURA answers can be slow: the agent may run
  // several MCP tool calls (or a whole orchestration plan) before replying.
  timeoutMs: z.number().default(120000),
})

export function apply(ctx, config) {
  // register() is effect-based: disposing the plugin fiber unregisters the tool.
  ctx.tools.register(defineTool({
    name: 'aura_query',
    description:
      'Delegate a sub-question to an AURA agent. The agent has its own tools '
      + '(MCP servers, RAG) on the server side and returns a grounded final '
      + 'answer as text. Use for questions that need the AURA agent\'s data '
      + 'sources rather than your own tools.',
    parameters: {
      prompt: {
        type: 'string',
        required: true,
        description: 'The complete, self-contained question or task for the AURA agent.',
      },
      model: {
        type: 'string',
        description:
          'AURA agent to address (its alias/name, e.g. "aura-default"). '
          + 'Omit to use the configured default.',
      },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const model = args.model ?? config.model
      const headers = { 'content-type': 'application/json' }
      if (config.apiKeyEnv) {
        const key = process.env[config.apiKeyEnv]
        if (!key) {
          throw new Error(`aura_query: apiKeyEnv "${config.apiKeyEnv}" is configured but the environment variable is unset or empty`)
        }
        headers.authorization = `Bearer ${key}`
      }

      // Honor caller cancellation AND the configured deadline, whichever fires
      // first (AbortSignal.any adopts the first source's reason).
      const signal = AbortSignal.any([exec.signal, AbortSignal.timeout(config.timeoutMs)])

      const url = `${config.baseURL}/chat/completions`
      const res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model,
          stream: false,
          messages: [{ role: 'user', content: args.prompt }],
        }),
        signal,
      })

      if (!res.ok) {
        const body = await res.text().catch(() => '')
        throw new Error(
          `aura_query: AURA returned HTTP ${res.status} for model "${model}" at ${url}`
          + (body ? ` — ${body.slice(0, 500)}` : ''),
        )
      }

      const data = await res.json()
      const content = data?.choices?.[0]?.message?.content
      if (typeof content !== 'string') {
        throw new Error(`aura_query: AURA response for model "${model}" carried no choices[0].message.content`)
      }
      return content
    },
  }))
}
