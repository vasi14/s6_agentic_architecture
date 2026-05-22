# Four target queries

## Query A. Shannon Wikipedia (artifact attach test)

**User query**: Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.

**Expected trace**:
─── iter 1 ───
[memory.read]   1 hits
[perception]    [open] Fetch the Wikipedia page for Claude Shannon
                [open] Extract birth date, death date, and three contributions
[decision]      TOOL_CALL: fetch_url({"url": "https://en.wikipedia.org/wiki/Claude_Shannon"})
[action]        → [artifact art:09ff0a67fe264eb9, 263065 bytes] preview: ...

─── iter 2 ───
[memory.read]   2 hits
[perception]    [done] Fetch the Wikipedia page for Claude Shannon
                [open] Extract birth date, death date, and three contributions
                  attach=art:09ff0a67fe264eb9
[attach]        art:09ff0a67fe264eb9 (263065 bytes)
[decision]      ANSWER: Claude Shannon (1916-2001) was an American mathematician...

─── iter 3 ───
[perception]    [done] Fetch the Wikipedia page for Claude Shannon
                [done] Extract birth date, death date, and three contributions

[done] all 2 goals satisfied

FINAL: Birth date: April 30, 1916. Death date: February 24, 2001.
       Three key contributions: (1) A Mathematical Theory of Communication
       (1948), which established the mathematical foundations of digital
       communication; (2) introduction of the bit as the unit of information
       and the concept of entropy; (3) the Shannon limit, the theoretical
       maximum rate at which information can be transmitted over a noisy
       channel.

**Description**:
This query exercises the artifact attach path. The fetched Wikipedia page is roughly 250 KB of markdown. The artifact store receives the bytes; Memory records the handle; Perception identifies the second goal (extraction) as needing the bytes; the loop attaches them to Decision's prompt for that goal; Decision produces the structured answer. Iteration count: 3. The architecture's central property appears in iter 2: Perception sees the artifact handle in the memory hits and sets attach_artifact_id on goal 2. The loop loads the bytes and Decision answers in one call without re-fetching.

## Query B. Tokyo activities with weather constraint (multi-goal plus memory carryover)

**User query**:

Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.

**Expected trace**:

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

**Description**:

This query has three logical goals: search for activities, fetch the weather forecast for Saturday, select an appropriate activity given the weather. The memory carryover happens between goals two and three: the weather fact recorded by Action is read by Decision when reasoning about which activity fits. Iteration count: 6 in the observed run including some web-search refinement. Memory carries the weather fact from iter 2 into Decision's context in iter 3 through the keyword search at the top of each iteration.

## Query C. Mom's birthday (durable memory across two runs)

**User query**:

Run 1: My mom's birthday is 15 May 2026. Remember that and give me
       a calendar reminder for two weeks before and on the day.

Run 2: When is mom's birthday?

**Expected trace**:

Run 1 trace (abbreviated):

[memory.remember]  classified "Mom's birthday is 15 May 2026" as fact
                   keywords: ["mom", "birthday", "may", "2026"]

─── iter 1 ───
[perception]    [open] Remember mom's birthday (15 May 2026)
                [open] Create a reminder for 1 May 2026 (two weeks before)
                [open] Create a reminder for 15 May 2026
[decision]      TOOL_CALL: create_file({"path": "reminders/mom_birthday_2026.txt", ...})
[action]        → ok

... two more iterations creating the reminders ...

FINAL: Reminders created. Mom's birthday on 15 May 2026 is recorded.

Run 2 trace:

─── iter 1 ───
[memory.read]   1 hits
                fact: "Mom's birthday is on 15 May 2026"
[perception]    [open] Answer when mom's birthday is
[decision]      TOOL_CALL: list_dir({"path": "reminders/"})
[action]        → [file: mom_birthday_2026.txt]

─── iter 2 ───
[memory.read]   2 hits
[perception]    [done] Answer when mom's birthday is
[decision]      ANSWER: Mom's birthday is on 15 May 2026.

[done] all 1 goals satisfied

**Description**:

This query exercises the durable-memory contract. The first run classifies the user's statement at the very top of agent6.run(...) via memory.remember(...), producing a fact item with the date and entity extracted. Run 1 then creates reminder files in the sandbox via create_file. Run 2, executed against the same state/ directory, finds the fact through the keyword search and answers directly.

state/memory.json after run 1 contains a fact item with kind="fact", keywords mentioning birthday/mom/may, value containing the date.

Iteration count: 4 for run 1, 2 for run 2. The fact is carried across run boundaries by the persistent state/memory.json file.

## Query D. Asyncio research (multi-source synthesis)

**User query**: Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.

**Expected trace**:

─── iter 1 ───
[decision]      TOOL_CALL: web_search({"query": "Python asyncio best practices"})
[action]        → [3 URLs in descriptors]

─── iter 2 ───
[decision]      TOOL_CALL: fetch_url({"url": "https://result1..."})
[action]        → [artifact art:abc1, 45KB]

─── iter 3 ───
[decision]      TOOL_CALL: fetch_url({"url": "https://result2..."})
[action]        → [artifact art:abc2, 38KB]

─── iter 4 ───
[decision]      TOOL_CALL: fetch_url({"url": "https://result3..."})
[action]        → [artifact art:abc3, 51KB]

─── iter 5 ───
[perception]    [done] Search
                [done] Fetch top 3 results
                [open] Synthesise common advice
                  attach=art:abc1   (Perception picks the most recent
                                     synthesis-relevant artifact)
[attach]        art:abc1 (45KB)
[decision]      ANSWER:
                1. Use asyncio.run() as the program entry point
                2. Prefer asyncio.gather and asyncio.TaskGroup over manual
                   awaits when running multiple coroutines concurrently
                3. Avoid blocking calls in async code; use asyncio.to_thread()
                   for CPU-bound or blocking I/O
                4. Use timeouts on every external call to prevent hangs
                5. Limit concurrency with semaphores when calling rate-limited
                   external services

**Description**:

This query exercises multi-artifact attachment. The agent performs a web search, fetches each of the top three results (producing three artifacts), and then Perception attaches the relevant ones to a synthesis goal. Decision reads the attached content and produces a consolidated list. 

Iteration count: 5 to 7 depending on how many results the agent decides to fetch. Perception's force-attach safety net kicks in for synthesis goals: when the first unfinished goal contains synthesis keywords (synthesise, extract, list, compare, decide) and an artifact exists in memory hits, the implementation attaches the most recent artifact automatically. This guard reduces dependence on the model's reasoning about which artifact is relevant.

