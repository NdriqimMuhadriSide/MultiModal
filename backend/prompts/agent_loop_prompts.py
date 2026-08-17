"""
Prompt scaffolding shared by every hand-written agent loop.

What lives here is the *shape* of a turn - the tool catalog, the transcript
of what has happened, the step budget, the instruction to act. What does
not live here is any agent's persona or subject-matter advice: those are
the system prompts, and each agent keeps its own (see
prompts/research_prompts.py, prompts/vision_prompts.py).

That split is the useful one. Two agents doing completely different jobs
still need the transcript rendered identically, because the parser that
reads their replies is the same parser - but they need to be told to be
different things.

Originally written as part of prompts/research_prompts.py and generalised
when the vision agent became a second caller. Nothing about the formatting
was research-specific; the module boundary was just drawn a step too far
in.
"""


def format_tool_catalog(tools) -> str:
    """
    Render an agent's tools as the list the model reads.

    Takes the live Tool objects rather than a hand-written block of text, so
    the catalog cannot drift from the dispatch table: a tool added to an
    agent's registry appears here, and one removed from it disappears, with
    no second place to remember to edit.

    Duck-typed (`.name`, `.description`, `.input_spec`) rather than
    importing Tool, because agents/agent_loop.py imports *this* module -
    annotating the parameter would make that a cycle.
    """
    return "\n".join(
        f"- {tool.name}\n"
        f"    what it does: {tool.description}\n"
        f"    input: {tool.input_spec}"
        for tool in tools
    )


def format_scratchpad(steps) -> str:
    """
    Render the steps taken so far as the transcript the model reads back.

    This is the agent's entire memory of its own work. It is rebuilt from
    the step list and re-sent whole on every turn, because the LLM is
    stateless between calls - there is no server-side conversation to append
    to, so "what have I already tried" only exists here.

    Duck-typed on AgentStep for the same reason as format_tool_catalog.
    """
    if not steps:
        return "(nothing yet - this is your first step)"

    blocks = []
    for number, step in enumerate(steps, start=1):
        blocks.append(
            f"Step {number}\n"
            f"Thought: {step.thought}\n"
            f"Action: {step.action_json}\n"
            f"Observation: {step.observation}"
        )
    return "\n\n".join(blocks)


def format_history_block(history) -> str:
    """
    Fold recent conversation turns into the prompt as inert context.

    Passed as prompt *text* rather than as real chat messages: real prior
    turns make the model answer the follow-up conversationally instead of
    producing the Thought/Action pair the loop needs to parse. The routing
    agent this project started with had to do the same thing to get one
    keyword back out of a classification.

    As text it still does the one job it is here for - resolving what "it"
    or "that policy" refers to, so the first tool call is about the right
    subject rather than about four pronouns.
    """
    if not history:
        return ""

    turns = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history
    )
    return (
        "Earlier turns in this conversation, for resolving references only "
        "(do not answer these again):\n"
        f"{turns}\n\n"
    )


def format_loop_prompt(
    question: str,
    tool_catalog: str,
    scratchpad: str,
    steps_left: int,
    history_block: str = "",
    preamble: str = "",
) -> str:
    """
    Build one turn of a loop.

    The ordering is deliberate: stable context first (history, tools), then
    the question, then the part that grows (the scratchpad), then the
    instruction to act. Two reasons. Models attend most reliably to the
    start and the end of a prompt, so the ask goes last where the scratchpad
    cannot bury it; and a stable prefix is what a provider's prompt cache
    can reuse across steps, which a growing block at the top would defeat.

    `steps_left` is in the prompt because a step budget the agent cannot see
    is a budget it cannot plan against - it would spend its last step
    opening a new line of enquiry instead of answering with what it has.

    `preamble` is for a fact about *this run* that the agent needs before it
    plans anything - the vision agent uses it to state that an image is
    attached and what type it is. Empty for agents whose runs differ only by
    the question.
    """
    if steps_left <= 1:
        budget_line = (
            "This is your LAST step. You cannot run another tool. Call "
            "`finish` now with the best answer the evidence above supports."
        )
    else:
        budget_line = (
            f"You have {steps_left} steps left, including this one. Call "
            "`finish` as soon as you can answer - unused steps are not a "
            "waste, but running out before you answer is."
        )

    return (
        f"{history_block}"
        f"{preamble}"
        "Tools available to you:\n"
        f"{tool_catalog}\n"
        "\n"
        f"The question you are working on:\n{question}\n"
        "\n"
        "What you have done so far:\n"
        f"{scratchpad}\n"
        "\n"
        f"{budget_line}\n"
        "\n"
        "Reply with your Thought and Action now."
    )


def format_synthesis_prompt(question: str, scratchpad: str) -> str:
    """
    The forced ending: write the answer from what was gathered.

    Used when a loop hits its step limit without the agent calling `finish`.
    The alternative - returning "I ran out of steps" - throws away
    everything the agent gathered and tells the user about an implementation
    detail they cannot act on. The evidence is already paid for; this spends
    one more LLM call to turn it into the answer it was gathered for.

    Note this prompt asks for prose, not for a Thought/Action pair, and so
    it must be sent WITHOUT the loop's own system prompt - those rules
    demand an Action line and would fight this instruction. Every agent
    passes its own synthesis system prompt instead.
    """
    return (
        f"You were working on this question:\n{question}\n"
        "\n"
        "This is what you gathered:\n"
        f"{scratchpad}\n"
        "\n"
        "You are out of steps. Write the final answer now, using only what "
        "is above. Cite the [E1], [E2] labels inline where they apply. If "
        "what you gathered does not cover part of the question, say which "
        "part is missing rather than filling it in from your own knowledge.\n"
        "\n"
        "Write only the answer - no thoughts, no actions, no preamble."
    )
