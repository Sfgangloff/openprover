# vLLM Thinking and Tool Control

## Goal

Explore practical ways to control:

1. thinking-token budget vs final-output budget, and
2. tool-calling behavior (when tools are allowed, how many calls, and how long tool arguments can grow)

when using vLLM's OpenAI-compatible `/v1/chat/completions` API.

## What vLLM Gives Us Natively

- Reasoning-capable models can expose reasoning separately from answer text (`reasoning` in modern docs; older payloads may still use `reasoning_content`).
- Streaming works for reasoning and content deltas.
- Tool calling supports `tool_choice` values `auto`, `required`, and `none`.
- `parallel_tool_calls=false` limits a response to at most one tool call (0 or 1).
- `max_tokens` / `max_completion_tokens` caps total generated tokens for a request.

## Key Gap

vLLM does **not** provide native, independent budgets like:

- `max_thinking_tokens` and
- `max_output_tokens`

in a single request (unlike `scripts/serve_hf.py` in this repo).

So we need client orchestration.

---

## Option A: Two-Stage Orchestration

### A1) Thinking + Final Answer (no tools)

Use two calls.

Stage 1 (thinking phase):
- enable thinking (model/template-dependent)
- `tool_choice="none"` (if tools are present but should be disabled in this phase)
- `max_tokens = max_thinking_tokens`
- add `stop: ["</think>"]` when model emits explicit think tags

Stage 2 (answer phase):
- disable thinking via `chat_template_kwargs` when supported (for example Qwen3 `enable_thinking=False`)
- `max_tokens = max_output_tokens`
- no `</think>` stop

Pros:
- Hard separation of budgets.
- Deterministic behavior, easy to test.

Cons:
- Extra latency (two requests).
- Requires model/template support for switching thinking behavior cleanly.
- If the model does not emit `</think>`, Stage 1 may end by token cap rather than clean boundary.

### A2) Tool Calling with Budgeted "Deliberation Then Act"

Use two calls for tool-capable turns.

Stage 1 (plan/deliberate):
- thinking enabled
- `tool_choice="none"`
- `max_tokens = max_thinking_tokens`
- optional `stop=["</think>"]`

Stage 2 (tool decision/call):
- thinking disabled (if supported), or stricter system prompt ("no hidden reasoning")
- enable tools (`tool_choice="auto"` or `required`)
- `parallel_tool_calls=false` unless multi-tool fanout is desired
- `max_tokens = tool_phase_tokens` (caps tool JSON + any assistant text)

After executing tool(s), continue with normal tool loop under explicit per-turn budgets.

Pros:
- Strong control over runaway reasoning before tools.
- Lets us enforce separate token envelopes for planning vs tool args/output.

Cons:
- More orchestration complexity.
- Potential context drift between stages if prompts are not stitched carefully.

---

## Option B: Client-Enforced Cutoff During Streaming

Single request with streaming; client monitors deltas and interrupts when budget is hit.

### B1) Thinking + Final Answer

Flow:
- Send one streaming request with reasoning enabled.
- Track an approximate thinking-token count from streamed `reasoning` (or `reasoning_content`) text.
- When count reaches threshold, terminate the HTTP stream.
- Start follow-up request for answer-only with remaining budget.

Pros:
- Lower latency than always two full phases.
- Can stop early exactly when budget is hit.

Cons:
- Requires tokenizer-based counting client-side for good accuracy.
- Mid-token/chunk boundary effects and cancellation races can happen.
- More edge cases than deterministic two-stage calls.

### B2) Tool Calling

Flow:
- Stream tool-capable response.
- Track:
  - reasoning token count
  - tool-argument token count
  - number of tool calls emitted
- Abort when any limit is exceeded, then continue with a constrained follow-up turn:
  - `tool_choice="none"` to force plain text fallback, or
  - `parallel_tool_calls=false` and narrower tool set, or
  - explicit "final answer only" prompt path.

Pros:
- Maximum flexibility; can enforce multiple dynamic limits in one runtime loop.

Cons:
- Most complex option.
- Harder to make behavior predictable across models/parsers.

---

## Tool-Calling Controls We Should Enforce Client-Side

Regardless of Option A or B:

1. **Max tool rounds per user turn**
   - Example: stop after 3 tool-execution loops.
2. **Max tool calls per model response**
   - Use `parallel_tool_calls=false` for strict 0-or-1.
3. **Max tool argument length**
   - Reject/abort if arguments exceed token/char budget.
4. **Tool allowlist by phase**
   - Start narrow; expand only when needed.
5. **Timeout + cancellation**
   - Per request and per tool execution.
6. **Fallback mode**
   - If limits hit, force `tool_choice="none"` and ask for best-effort final answer.

---

## Recommended Path for OpenProver

For reliability, start with **Option A (Two-Stage)** for both reasoning and tool flows:

- It best matches existing deterministic budgeting ideas in `scripts/serve_hf.py`.
- Easier to debug and evaluate.
- Simplifies TUI behavior and archive analysis.

Then optionally add **Option B** as an optimization path behind a feature flag after baseline is stable.

Suggested rollout:

1. Add an orchestration helper in `openprover/llm.py` for two-stage vLLM calls.
2. Add per-turn limits:
   - `max_thinking_tokens`
   - `max_output_tokens`
   - `max_tool_rounds`
   - `max_tool_arg_tokens`
3. Default to `parallel_tool_calls=false`.
4. Normalize both `reasoning` and `reasoning_content` in stream parsing.
5. Add archive fields for limit hits and fallback reasons.

---

## Notes and Caveats

- Reasoning field naming is evolving (`reasoning_content` -> `reasoning`). Clients should accept both for now.
- Behavior depends heavily on model + parser + chat template.
- Tool calling quality is parser/model dependent even when schema validity is guaranteed.
- `stop=["</think>"]` is effective only for models/templates that actually emit that boundary.
