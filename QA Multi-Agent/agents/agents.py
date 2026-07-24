import re

from langchain.tools import tool
from agents_core.models import writer_model, test_explorer_model, reviewer_model
from agents_core.tools import run_python, run_tests, file_reader, playwright_tools
from agents_core.prompts import writer_prompt, test_explorer_prompt, reviewer_prompt
from agents_core.agent_creation import agent
from langchain_core.runnables import RunnableConfig
import asyncio

@tool
async def test_explorer_agent(query: str, config: RunnableConfig):
    """
    Test automation engineer: reads the test case, explores the live site
    via Playwright MCP, writes an auto test in TypeScript + Playwright
    following the codex, and MANDATORILY verifies it with its own run
    via the `run_tests` tool. Returns the result to the supervisor only
    after a green `run_tests` (or after the self-fix budget is exhausted
    with a red result).

    This is an executor agent with self-verification, not an analyst or
    a reviewer. It does not review against the codex (that's
    `reviewer_agent`), does not write documentation (that's
    `writer_agent`), does not edit the codex.

    Tools:
    `file_reader`  | Read the source files you rely on. They usually live in `/app/data/inputs/` or `/app/data/outputs/`. |
    `run_python` | Save the finished auto test. **All** output files are strictly under `/app/data/`. |
    `run_tests ` | Run the auto test for errors on Playwright |


    Args:
        query (str): A self-contained brief from the supervisor. Must
            include:
              - the user's goal in one line;
              - the absolute path to the source test case
                (typically `/app/data/inputs/...`);
              - the target base URL / environment, if known;
              - framework / naming preferences, if any;
              - on a re-run after fixes: the user's list of fixes
                (verbatim);
              - the expected output form: auto test files in
                `/app/data/outputs/tests/<transliterated case name>.ts`.
            Do NOT put the multi-turn conversation with the user
            into `query` — the agent has no context for it.
        config (RunnableConfig): The LangGraph runtime config
            (callbacks, metadata, tags, etc.). Technical fields;
            the agent's content does not depend on them.

    Returns:
        str: The agent's final string response in one of two forms:

        **Green result:**
            **Result:** ✅ Done, run is green
            **File:** `/app/data/outputs/tests/<name>.ts`
            **Run:** passed N / N, duration <ms>
            **Brief:** <1-3 lines, what the test checks>
            **Notes (if any):** <case-vs-reality mismatches, ambiguous
            steps, what to clarify with the case author>

        **Red result (after the attempt budget is exhausted):**
            **Result:** ❌ Not done, run is red after N rounds
            **File:** `/app/data/outputs/tests/<name>.ts`
            **What failed:** <list of failed steps / expect>
            **What I tried to fix:** <what I changed in each round,
            why it didn't help>
            **Suspicion:** <most likely cause — code / product / case>
            **Recommendation:** <what to ask the user>

    Behavior:
        - The agent's workflow is fixed: read the case → explore
          the site via Playwright MCP → write the `.ts` → save to
          `/app/data/outputs/tests/` → run `run_tests` → if red,
          fix and re-run (max 2 rounds) → return the result.
        - The file name is the test case name transliterated, with
          the `.ts` extension.
        - The auto test is wrapped in `test.describe(...)` with the
          scheme `[case №][Zone][Platform] Core-scenario`.
        - Locators are primarily `getByTestId` / `getByRole`;
          `locator(...)` only in edge cases.
        - The ER from the test case is written into `expect(...)`
          VERBATIM.
        - `waitForTimeout(...)` is FORBIDDEN; use element waits /
          network response waits instead.
        - **After saving, MANDATORILY calls `run_tests`**
          (only via this tool, never from the console).
        - If `run_tests` is red — minimal point-wise fix and
          re-run. One fix = one minimal change. Does not
          rewrite the whole test because of a single failing expect.
        - If `run_tests` is red and the cause is a mismatch
          between the test case and reality — **does NOT guess**,
          marks it in the final answer and ends the iteration.
        - **Limit:** a maximum of 3 `run_tests` invocations
          (first + 2 fixes). After the budget is exhausted —
          returns a red result with diagnostics, without trying
          to continue.

    When to use (from the supervisor's point of view):
        - The user has a test case (a file in
          `/app/data/inputs/...`) and wants an auto test.
        - The user explicitly asked to "rewrite / rework an
          existing auto test" — then both the test case and the
          current `.ts` and the list of fixes are passed in
          `query`.

    When NOT to use:
        - There is no test case or the path to it is unknown. The
          agent has nothing to turn into code — escalate to the
          user via `call_human`.
        - You only need to review an existing auto test — that's
          `reviewer_agent`.
        - You need documentation / README / report — that's
          `writer_agent`.

    Constraints:
        - **Does NOT call `call_human`.** If the agent needs
          clarification — it writes that into the returned string
          ("base URL needed", "case step 3 is unclear"), and the
          supervisor escalates to the user itself.
        - **Runs the test ONLY via `run_tests`.** No
          `npx playwright test`, `node ...`, or other manual
          console runs.
        - **Does NOT return "done"** without a green `run_tests`.
          No exceptions. Beautifully written code with a red
          run is a defect, not a result.
        - **Does NOT tune the test for a green run** if it
          contradicts the test case. Better to return a red
          result with a "case mismatch" note.
        - Saves files ONLY to `/app/data/outputs/tests/`.
          Any path outside `/app/data/` is a bug.
        - Does not invent steps / ER that aren't in the case. If
          the case is missing something — it writes strictly to
          the text and marks "clarify with the case author".
        - Does not go into infinite fix: the 2-round
          self-fix limit is hard.

    Gotchas:
        - The agent is stateless from the supervisor's point of
          view, but holds its own memory. `query` must be
          self-contained.
        - The agent may make dozens of MCP calls internally
          (`browser_navigate`, `browser_snapshot`,
          `browser_click`, `browser_evaluate`, ...) — none of
          them make it into the return. If you need to know
          WHAT it tried in the browser, ask explicitly:
          "Before answering, briefly list what you opened /
          clicked / which test-ids you saw".
        - If the case is written ambiguously, the agent does
          NOT guess — it writes strictly to the text and marks
          it. That is not a defect; it's behavior per the
          codex.
        - File naming via transliteration: if the test case
          name contains special characters or Cyrillic, expect
          `[2123] [UZ][Desktop][MainPage] Virtual Keyboard.ts`
          per the rule in the agent's system prompt.
        - `run_tests` can return `status="error"` (the test
          didn't start: syntax / import / config) — that is
          NOT "an expect failed", it's a different class of
          problem. The agent fixes the code until it starts,
          then runs the expects.
        - The supervisor **does not run `run_tests` itself** —
          so the supervisor **cannot confirm a green run**
          without calling `test_explorer_agent`. If the
          supervisor needs a fresh run of an existing test,
          it calls the explorer again with an explicit
          "just run, don't rewrite" note.
    """
    playwright = await playwright_tools()
    tools_explorer = [run_tests, run_python, file_reader] + playwright

    user_session_id = str(config.get("configurable", {}).get("thread_id"))
    rec_limit = int(config.get("recursion_limit"))
    langfuse_handler = config.get("callbacks")

    explorer_config = {
        "configurable": {"thread_id": user_session_id + '_explorer'},
        "recursion_limit": rec_limit,
        "callbacks": langfuse_handler,
    }
    try:
        graph = await agent(test_explorer_model, tools=tools_explorer,
                      sys_prompt=test_explorer_prompt,
                      config=explorer_config, query=query, memory_flag = True)
        response = await graph.ainvoke({"messages": [("user", query)]}, explorer_config)
        clean_content = re.sub(r"<think>.*?</think>", "", response["messages"][-1].content, flags=re.DOTALL).strip()
        return clean_content
    except Exception as e:
        return 'Error: ' + str(e)

@tool
async def writer_agent(query: str, config: RunnableConfig):
    """
    Universal QA writer: turns QA materials (test cases, ready
    auto tests, review reports, run results) into ready-to-use
    text artifacts: test plans, suite READMEs, review reports,
    user-facing recaps, run summaries, coverage matrices, release
    checklists, postmortems.

    This is a writer agent, not a coder and not a reviewer. It
    does not write auto tests, does not review against the
    codex, does not orchestrate. It only formats text from
    materials.

    Args:
        query (str): A self-contained brief from the supervisor.
            Must include:
              - artifact type (test plan / README / review report
                / recap / summary / coverage matrix / release
                checklist / postmortem);
              - audience (developer / tester / manager / the
                user themselves);
              - tone (formal / concise / friendly);
              - length (e.g., "200 words");
              - must-include items (what is mandatory);
              - must-avoid items (what not to write);
              - absolute paths to sources in `/app/data/` that
                the text should be based on.
        config (RunnableConfig): LangGraph runtime config.

    Returns:
        str: The final text of the artifact in markdown. If the
            brief said to save to a file, it uses `run_python`
            and the agent puts this block at the start of the
            string:
                **Saved to:** `/app/data/outputs/<...>.md`
            and prints the absolute path.

    Behavior:
        - Reads sources via `file_reader`. If a source is not
          found — it says so at the start of the answer,
          **before** the artifact text.
        - If the artifact format doesn't fit any of the formats
          listed in the system prompt — it asks in the returned
          string ("format is unclear, please clarify"). The
          supervisor escalates to the user.
        - If data is missing — leaves a placeholder
          `[TODO: <what exactly>]` AND mentions it in the
          answer. Does not invent content.
        - Saves files ONLY to `/app/data/` (typically
          `/app/data/outputs/`). Outside `/app/data/` — lost.
        - Picks the style to fit the brief. No default
          corporate voice.

    When to use (from the supervisor's point of view):
        - The user asks for a test plan, README, review
          report, summary, recap, coverage matrix, release
          checklist, or postmortem.
        - After a `test_explorer_agent → reviewer_agent` run,
          the user asks to "format the result as a readable
          report" or "write a README for the suite".

    When NOT to use:
        - For generating / fixing auto test code — that's
          `test_explorer_agent`.
        - For reviewing against the codex — that's
          `reviewer_agent`.
        - For "rewrite one sentence" — that's not its role.
          The request is too small to launch a subagent;
          better to ask the user via `call_human` whether
          this is even needed.
        - For orchestrating other agents — that's the
          supervisor's job.

    Constraints:
        - **Does NOT call `call_human`.** If the format is
          unclear or data is missing — it mentions it in the
          returned string; the supervisor decides whether to
          escalate.
        - **Does NOT write auto test code.** Even if asked
          to "describe what an auto test would look like"
          — it describes in words, not code.
        - **Does NOT review against the codex.** It does not
          use the terms "blocker / non-blocker", does not
          anchor to file/line, does not issue a "verdict".
          That's `reviewer_agent`'s role.
        - **Does NOT orchestrate.** Does not call other
          agents.
        - **Does NOT invent facts, numbers, IDs, file
          names.** If not in the source — puts
          `[TODO: ...]`.
        - **Does NOT lose specifics** when summarizing:
          numbers, dates, test case IDs, paths to `.ts`
          are preserved verbatim.
        - Saves files ONLY to `/app/data/`.

    Gotchas:
        - The agent is stateless. If `query` refers to
          "the auto test we just generated" — pass the
          absolute path to it explicitly. Otherwise the
          agent will guess.
        - If asked for a suite README and there are no
          `.ts` files in `/app/data/outputs/tests/` yet —
          the agent will write directly that the files
          are missing and ask for a source. It will not
          invent non-existent file names.
        - When summarizing a review report, the agent
          has no right to "soften" the ❌ / ⚠️
          formulations. If the reviewer wrote "blocker",
          the writer writes "blocker". If the reviewer
          wrote "note", the writer writes "note". It
          does not dilute the verdict's semantics.
        - For "user-facing recap" the tone should be
          friendly and brief, but without sliding into
          syrup. If the brief doesn't specify the tone —
          the agent picks a neutral-business default
          and notes at the start that the tone was
          chosen by default.
    """
    try:
        graph = await agent(writer_model, tools=[run_python, file_reader],
                  sys_prompt=writer_prompt,
                  config=config, query=query)
        response = await graph.ainvoke({"messages": [("user", query)]}, config)
        clean_content = re.sub(r"<think>.*?</think>", "", response["messages"][-1].content, flags=re.DOTALL).strip()
        return clean_content
    except Exception as e:
        return 'Error: ' + str(e)
    
@tool
async def reviewer_agent(query: str, config: RunnableConfig):
    """
    Auto test reviewer: statically compares a test case and
    the corresponding `.ts` file. Returns a verdict in one of
    three forms, anchored to file and line.

    This is an auditor agent, not an executor. It does not
    write code, does not fix auto tests, does not change
    test cases. It only reviews the correspondence.

    Args:
        query (str): A self-contained brief from the supervisor.
            Must include:
              - absolute path to the source test case
                (typically `/app/data/inputs/...`);
              - absolute path to the generated auto test
                (typically `/app/data/outputs/tests/<...>.ts`);
              - if needed — context from the automation codex
                (if not built into the agent's system prompt).
            The agent does NOT know what was said in the
            multi-turn conversation with the user — only
            what's in `query`.
        config (RunnableConfig): LangGraph runtime config.

    Returns:
        str: A structured report in a fixed form:
            ```
            ## Verdict
            **Decision:** ✅ Ready to merge / ❌ Fixes required /
                ⚠️ Has notes

            ### Critical Issues (Blockers)
            1. [issue, why it's a blocker, how to fix]

            ### Recommendations (Non-Blockers)
            1. [recommendation for improvement]
            ```
            The verdict is always one of the three. Defects
            and notes — with file and line, without "maybe"
            and "perhaps worth thinking about".

    Behavior:
        - **NEVER runs the test** and does not suggest run
          commands. Only static analysis of the code and the
          case text.
        - Checks five things:
            1. Case coverage completeness (every step and
               ER).
            2. Locator quality (primarily `getByTestId`,
               clear names, lifted to the top).
            3. Matcher quality (correct matcher per check
               type, no over-engineered regex, no hardcode
               in `.toHaveText`).
            4. Code structure (`describe`, one `test`,
               linear `step`, no nested `step`, no helper
               functions inside `test(...)`).
        - Does not evaluate the product / feature — only
          the test's correspondence to the case and the
          codex.
        - Does not flag style that doesn't violate the
          codex.

    When to use (from the supervisor's point of view):
        - **Mandatory** after every auto test generation
          by `test_explorer_agent`, before showing the
          result to the user. No exceptions.
        - On an explicit user request "review my auto
          test" — pass the path to the `.ts` and the
          corresponding test case.

    When NOT to use:
        - On test cases, documents, READMEs, reports.
          The reviewer reviews **only** generated auto
          tests in `/app/data/outputs/tests/`.
        - On raw snippets not formatted as a finished
          auto test in the expected structure (no
          `describe` — the reviewer can't check a
          single rule).
        - For "overall code quality assessment" — that's
          not its role.

    Constraints:
        - **Does NOT call `call_human`.** If clarification
          on the case or code is needed (e.g., an
          ambiguous step) — it writes it in the returned
          string under defects. The supervisor decides
          whether to escalate.
        - **Does NOT run the test** and does not suggest
          run commands.
        - **Does NOT write its own version of the auto
          test** and does not fix someone else's. Review
          only.
        - **Does NOT suggest changing the test case.**
          If something in the case is illogical — that's
          for the case author, not the auto tester.
        - Does not flag stylistic minutiae (import order,
          single vs double quotes, etc.) — only codex
          violations and case mismatches.

    Gotchas:
        - The verdict comes back as a string, not a
          structure. The supervisor parses it itself when
          it needs to extract the list of defects for
          `call_human` to the user.
        - ⚠️ "Has notes" is **not** a green verdict. Even
          if there are no blockers, the supervisor must
          still return to the user and ask whether to
          apply the notes.
        - "Ready to merge" does not mean "perfect". It
          means "no codex violations or case mismatches
          found".
        - The agent is stateless: it does not remember
          what it reviewed earlier in this session. If
          the supervisor re-calls it after fixes — it
          passes both the case and the updated `.ts`,
          and (optionally) the previous list of defects
          as additional context.
    """
    try:
        graph = await agent(reviewer_model, tools=[file_reader],
                  sys_prompt=reviewer_prompt,
                  config=config, query=query)
        response = await graph.ainvoke({"messages": [("user", query)]}, config)
        clean_content = re.sub(r"<think>.*?</think>", "", response["messages"][-1].content, flags=re.DOTALL).strip()
        return clean_content
    except Exception as e:
        return 'Error: ' + str(e)
