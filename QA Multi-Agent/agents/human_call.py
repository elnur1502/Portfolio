from langgraph.types import interrupt
from langchain.tools import tool
import asyncio

@tool
async def call_human(query: str, context=None):
    """
    Pause the orchestration and ask the user a clarifying question.

    This is the supervisor's anti-hallucination tool. It's the
    ONLY way to connect execution with the user in the middle
    of a workflow: the call blocks via `interrupt` in
    LangGraph, the user sees the question, and the function
    returns only after the answer. Use this tool the moment
    you're about to guess, make up, or pick a default that
    MATERIALLY changes the result.

    For the supervisor only. Mini-agents (all team agents:
    `test_explorer_agent`, `reviewer_agent`, `writer_agent`,
    etc.) MUST NOT call this tool — they return their
    result as a string, and the supervisor itself decides
    whether to escalate to the user. If a mini-agent "needs
    clarification", it writes that in its returned string;
    the supervisor then calls `call_human` on its own
    behalf.

    Args:
        query (str): A real question that the user will see.
            Phrase it as a question, not a status update. One
            call — one most blocking unknown. Not several
            questions at once.
        context (str | dict | list, optional): A short
            markdown snippet: what you already know, what
            you tried, why you're asking. Should be easy to
            skim. If None — the user sees only the question.

    Returns:
        str: A string of the form
            "User responded: <user's answer>"

            The raw user response is glued verbatim. The
            supervisor receives it on the next turn and
            continues the workflow.

    Behavior:
        - The call blocks until the user answers (or
          cancels).
        - Do not call any other tool in the same turn as
          `call_human`; the workflow is paused, you will
          continue after the answer.
        - When the answer comes in — briefly confirm
          ("Got it, doing X") and continue. Don't make
          the user re-explain the task.

    When to use:
        - The request is ambiguous and the answer
          MATERIALLY depends on a choice you can't
          reasonably make yourself (topic, audience,
          length, tone, scope).
        - A mini-agent returned "no match" / empty /
          error, and the next step really depends on
          the user.
        - You need information only the user has
          (a private number, a file on their machine,
          a preference that isn't in memory yet).
        - Two mini-agents gave conflicting answers and
          you can't decide yourself who's right.
        - You're about to do something destructive or
          hard to reverse (delete data, send a message,
          place an order) — confirm first.

    When NOT to use:
        - Things a mini-agent can figure out itself
          (table names, internal rules, web search,
          formatting). Re-route or re-prompt the
          mini-agent first.
        - Confirmation theater ("are you sure you want
          to do this?") if the user already said "yes"
          by asking for the task.
        - Several questions at once. Ask the single
          most blocking unknown; the next — after the
          answer to this one.
        - Cosmetic preferences. Pick a reasonable
          default and offer to change it in the final
          answer.

    Constraints:
        - For the supervisor only. Mini-agents must not
          call this tool.
        - `context` (optional) is glued to the question
          as `query + "\n" + str(context)`. If you pass
          a dict — the user will see its string repr;
          for a clean readable output, pass an
          already-formatted markdown string.

    Gotchas:
        - An empty / cancelled answer is also a valid
          return value. Treat it as "the user declined
          to answer" and decide yourself: repeat the
          question, pick a default, or abort the task.
        - The user does NOT see your internal
          scratchpad or the mini-agents' history. If
          the context is too brief — it won't help; if
          too long — they'll skim past. Keep it
          3–6 lines.
    """
    if context is None:
        human_response = interrupt(query)
    else:
        human_response = interrupt(query + '\n' + str(context))
    return 'User responded: ' + str(human_response)
