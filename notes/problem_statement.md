Assume the role of an experienced senior AI engineer. Your objective is to complete the codes for the provided files with the assignment description.

# Assignment

Build an agent that passes different target queries. It should build and leverage the four cognitive roles as layers/agents mentioned below in [## The four cognitive roles].

## Example run (expected)

**User query:** "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."

**Description:** This query has three logical goals: search for activities, fetch the weather forecast for Saturday, select an appropriate activity given the weather. The memory carryover happens between goals two and three: the weather fact recorded by Action is read by Decision when reasoning about which activity fits.

**Expected Output**: 
─── iter 1 ───
[perception]    [open] Find 3 family-friendly things to do in Tokyo
                [open] Check Saturday's weather in Tokyo
                [open] Choose the most appropriate activity given the weather
[decision]      TOOL_CALL: web_search({"query": "family-friendly things to do in Tokyo this weekend"})
[action]        → [3 results returned, descriptors recorded]

─── iter 2 ───
[perception]    [done] Find 3 family-friendly things to do in Tokyo
                [open] Check Saturday's weather in Tokyo
                [open] Choose the most appropriate activity given the weather
[decision]      TOOL_CALL: fetch_url({"url": "https://wttr.in/Tokyo?format=...&Saturday"})
[action]        → Saturday forecast: patchy rain, 18C

─── iter 3 ───
[perception]    [done] Find 3 family-friendly things to do in Tokyo
                [done] Check Saturday's weather in Tokyo
                [open] Choose the most appropriate activity given the weather
[decision]      ANSWER: Given Saturday's patchy rain forecast, an indoor
                activity is recommended. From the three options found
                (Ueno Zoo, Tsukiji Outer Market sushi class, Tokyo Skytree),
                the Tsukiji sushi class is most appropriate because it is
                fully indoors and family-oriented.

[done] all 3 goals satisfied

## The four cognitive roles

HEADER: Role |	Responsibility	| Invoked per iteration	 | LLM call?

- Memory |	A typed service that stores facts, preferences, tool outcomes, and scratchpad entries. Exposes read(query, history) and write methods. | Always called for read. Called by Action for record_outcome.	| Only for the ambiguous classifying write. The keyword search read uses no LLM.

- Perception |	The orchestrator. Reads the query, the memory hits, and the history, and emits the current goal list with done flags and optional artifact attachments.	 | Yes, every iteration. | One LLM call routed via auto_route="perception" (pinned to Gemini in this session).

- Decision |	Picks the next action for one bounded goal. Returns either a final answer in plain text, or a single tool call to MCP.	 | Yes, once per iteration when there is an unfinished goal. |	One LLM call routed via auto_route="decision".

- Action	| Dispatches the chosen MCP tool. Pushes large results to the artifact store and returns a short descriptor. |	Only when Decision returns a tool_call. |None. Pure dispatch.

## Read methods

HEADER: Method | What it does | LLM cost
- memory.read(query, history, kinds=None, top_k=8) | Keyword overlap across keywords plus tokens of descriptor. Returns ranked top-k.	|  None. Pure Python.

- memory.filter(kinds=..., goal_id=..., recent=N) | 	Structured filter by kind, goal, recency. | None.

- memory.relevant(query, kinds=..., top_k=5)	| LLM-scored relevance over a kind-filtered candidate pool. Used only when keyword recall is weak. | One gateway call routed auto_route="memory".

## Write methods

HEADER: Method | When | LLM cost
- memory.remember(raw_text, source, run_id, goal_id) | Free-form ambiguous content (user input, observed statement). | One classification call (auto_route="memory", pinned to Gemini). Returns a typed item with kind, keywords, descriptor, and structured value extracted by the LLM.

- memory.record_outcome(tool_call, result_text, artifact_id, ...) | An MCP dispatch returned a result.	| None. Kind is tool_outcome by construction; keywords come from tool name and argument tokens.


## Persistence
- All items live in a single JSON file at state/memory.json. The agent6 loop loads on first read and writes back after every mutation. Across runs, the same JSON file is reused, so preferences and facts persist. Clearing the file resets the agent.

# Required

1. Four code modules with clear separation of concerns: memory.py, perception.py, decision.py, action.py. Plus an agent6.py (or any name) that wires them together in a loop. Plus a schemas.py containing the Pydantic models. Plus the MCP server from earlier sessions.
2. All four target queries must produce correct final answers. The expected answers and iteration counts are documented above. Queries that exceed twice the expected iteration count are not considered passing; tune the prompts and the contracts until convergence is within bounds.
3. Memory must persist across runs in a file under state/. Query C requires the durable-memory behaviour: run 1 records the fact, run 2 reads it.
4. The four cognitive layers must each be backed by typed Pydantic contracts on their inputs and outputs. No free-form dict passing between roles. No regex on LLM output.
5. The LLM gateway V3 must be the substrate for every LLM call. No direct calls to provider SDKs.
6. .The state/ directory must be cleanable between assignment attempts.

# Constraints

- Pydantic v2 on every boundary.
- uv for Python dependency management and execution. No manual virtualenv activation.
- MCP server stdio transport for tool calls. No reimplementing tool dispatch.
- No third-party agentic frameworks (LangGraph, LangChain, CrewAI). The architecture and the contracts are the assignment.
- Work on the existing placeholder files under the folder code/
- Ensure LLM requests are passed through the LLM_GatewayV3