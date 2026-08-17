"""
The reasoning loop, written by hand and shared by every agent that needs one.

    ┌─────────────────────────────────────────────────┐
    │  build prompt  (question + tools + scratchpad)  │
    │         ↓                                        │
    │  ask the LLM for one Thought + one Action       │
    │         ↓                                        │
    │  parse the Action out of free text              │
    │         ↓                                        │
    │  run the tool → Observation                     │
    │         ↓                                        │
    │  append (Thought, Action, Observation) ─────────┘
              ↓  ... until a terminal tool, or the budget runs out

Everything the model learns lives in the scratchpad, which is re-sent whole
on every turn. There is no hidden state: the LLM is stateless between
calls, so "what have I already tried" exists only because this module keeps
it and pastes it back.

WHY THIS IS ITS OWN MODULE

It was written inside agents/research_agent.py first, and extracted when
agents/vision_agent.py needed the same machinery. What made the split worth
doing is that the genuinely subtle parts are the ones with nothing
domain-specific in them:

    - `_parse_action` and its brace scanner, which absorb every way a model
      deviates from the format it was given
    - the step ceiling, the consecutive-parse-failure ceiling, and the
      forced synthesis, which are the only reasons a loop ever stops
    - turning a tool's exception into an observation, so a failing tool
      costs a step rather than the run

Two copies of those would drift, and the drift would be invisible until one
agent started hanging. What differs between agents is only configuration:
the tools, the persona, and the policy about when finishing is allowed.

WHAT AN AGENT SUPPLIES

    tools                    a list of Tool, including at least one terminal
    system_prompt            who the agent is and the Thought/Action contract
    synthesis_system_prompt  who it is when writing up, with no Action rule
    max_steps                this agent's own ceiling
    budget                   optional; the shared pool when this loop is one
                             node of a delegating tree (see StepBudget)

DELEGATION

A sub-agent is a Tool whose `run` happens to be another loop. Nothing here
knows about that, and deliberately so - `Tool.run` returning a string is
already the whole interface a delegating tool needs. Two things do have to be
tree-aware, because both are silently wrong otherwise:

    StepBudget   so a 6-step supervisor calling a 6-step specialist is not
                 36 LLM calls
    ToolResult   so the sub-agent's own steps can be attached to the step
                 that caused them, instead of collapsing into one string

WHAT THIS DELIBERATELY DOES NOT USE

The provider's tool-calling API (`tools=[...]`), which would return a
parsed, schema-validated call and delete `_parse_action` entirely.
Production code should generally prefer it. It is not used here for two
reasons: writing the protocol by hand is the point of the exercise, and a
text protocol works against any OpenAI-compatible endpoint regardless of
whether that provider implements tool calling.
"""
import json
import re
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Literal

from agents.run_log import record_run
from prompts.agent_loop_prompts import (
    format_history_block,
    format_loop_prompt,
    format_scratchpad,
    format_synthesis_prompt,
    format_tool_catalog,
)

# Two consecutive unparseable replies ends the loop.
#
# One is worth recovering from - the model is told exactly what was wrong
# and usually fixes it on the next turn, and spending a step on that is
# cheaper than throwing the run away. Two in a row means the model is not
# going to produce the format, and a third attempt is just a third identical
# failure. Consecutive, not cumulative: a single hiccup early should not
# doom an otherwise healthy run.
_MAX_CONSECUTIVE_PARSE_FAILURES = 2

# Recognised aliases for the two Action keys.
#
# Models trained on other agent frameworks reach for those frameworks'
# vocabulary - "action"/"action_input" is ReAct's original spelling, and
# "arguments" is what the OpenAI tool-calling API uses. All three mean the
# same thing here. Accepting them costs two dict lookups and removes a whole
# category of retry.
_TOOL_KEYS = ("tool", "tool_name", "action", "name")
_INPUT_KEYS = ("input", "tool_input", "action_input", "arguments", "args", "parameters")


class ActionParseError(Exception):
    """
    The model's reply did not contain a usable Action.

    An exception rather than a return value because it is genuinely
    exceptional *within a turn* - and because the loop's handling of it is
    not "give up" but "tell the model and try again", which reads far better
    as a `try/except` around one turn than as a status code threaded through
    the parse, the dispatch, and the append.
    """


class ActionRejected(Exception):
    """
    A tool declined the call and wants the model to try again.

    Distinct from a tool *failing*: nothing went wrong, the agent is just
    not allowed to do that yet, or not like that. The message becomes the
    next Observation, so it is written for the model and has to say what to
    do instead.

    One mechanism covers what would otherwise be several special cases in
    the loop - finishing with an empty answer, finishing before looking at
    anything, calling a tool with arguments that don't make sense. Each is a
    policy belonging to the tool that holds it, not to the loop.
    """


@dataclass
class Tool:
    """
    One capability an agent can invoke, and everything the prompt needs to
    describe it.

    `description` and `input_spec` are written for the model, not for a
    developer: they are pasted verbatim into the prompt by
    format_tool_catalog. That is why they read like instructions rather than
    like docstrings.

    A `terminal` tool ends the run, and its `run` returns the final answer
    rather than an observation. It is registered like any other tool so the
    model does not have to learn a second syntax to stop - the loop
    intercepts it on dispatch.

    `stream` is the opt-in for a tool that takes long enough to be worth
    watching - in practice, one that runs another agent. Without it a
    delegation is a single blocking call, and a specialist that thinks for
    four steps is fifteen seconds in which the caller is told nothing at all.
    With it, the sub-agent's steps are forwarded as they land.

    Its contract: yield zero or more StepEvent, then exactly one ToolResult
    last, carrying the observation. Deliberately expressed in this module's
    own types rather than in the sub-agent's, so the loop stays ignorant of
    what is on the other end - a delegating tool adapts, and AgentLoop never
    learns that agents can contain agents.

    A tool with `stream` set must still provide `run`, for callers that do
    not stream. Making it a thin drain of `stream` is the pattern used by
    AgentLoop.run and VisionAgent.run, and for the same reason: two
    implementations of one dispatch drift, and the drift is silent.
    """

    name: str
    description: str
    input_spec: str
    run: Callable[[dict], str]
    terminal: bool = False
    stream: Callable[[dict], Iterator["StepEvent | ToolResult"]] | None = None


@dataclass
class StepBudget:
    """
    Steps remaining across a whole tree of agents, shared by every loop in it.

    A per-loop `max_steps` is the right ceiling for one agent and the wrong
    one for a tree: a 6-step supervisor that can call a 6-step specialist on
    every one of its steps has a worst case of 36 LLM calls, and the worst
    case is what a timeout and a bill are set by. This is the object that
    makes the tree's cost bounded rather than multiplicative.

    It does not replace `max_steps`, it composes with it. Each loop stops at
    whichever comes first - its own ceiling, or the shared pool running dry -
    so one specialist cannot spend everything the supervisor still needs, and
    the tree as a whole still cannot exceed the pool.

    Mutable and shared by reference on purpose. Every `take` is a withdrawal
    visible to every other loop immediately, which is the entire point; a
    copied budget would give each agent its own private allowance and restore
    exactly the multiplication this exists to prevent.
    """

    remaining: int

    def take(self) -> bool:
        """Spend one step. False when there was nothing left to spend."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


@dataclass
class ToolResult:
    """
    A richer return from a tool that ran a whole agent of its own.

    Tools return `str` - that is the interface, and every leaf tool in this
    codebase uses it. A delegating tool needs to say one more thing: here is
    what my sub-agent did, so the trace can show it nested under the step
    that caused it rather than flattened into a single opaque observation.

    Returning this instead of a string is the whole opt-in. The loop checks
    the type at dispatch and every existing tool is unaffected, which is why
    this is an alternative return rather than a change to `Tool.run`'s
    signature.
    """

    observation: str
    children: list["AgentStep"] = field(default_factory=list)


@dataclass
class AgentStep:
    """
    One completed turn, as it will be replayed to the model and shown to the
    user.

    `action_json` is kept as the raw string the model produced rather than
    re-serialised from `tool`/`tool_input`. Replaying the model its own
    words - key order, spacing and all - keeps the transcript consistent
    with what it wrote, and it means the trace shown in a UI is what
    actually happened rather than a tidied reconstruction of it.

    `children` are the steps a delegated sub-agent took to produce this
    step's observation. Empty for every ordinary tool. They are carried in
    the trace but NOT in the scratchpad the model reads back - see
    format_scratchpad: the supervisor asked a question and got an answer, and
    replaying how the specialist arrived at it would spend the supervisor's
    context on reasoning it already delegated precisely so it would not have
    to do it.
    """

    thought: str
    action_json: str
    tool: str
    tool_input: dict
    observation: str
    children: list["AgentStep"] = field(default_factory=list)


@dataclass
class LoopResult:
    answer: str
    # The full trace. A first-class part of the result, not debug output: an
    # answer assembled over several hidden tool calls is not something a
    # user can sanity-check, and this is what makes it checkable.
    steps: list[AgentStep] = field(default_factory=list)
    stopped_because: Literal["finished", "step_limit", "parse_failures"] = "finished"


@dataclass
class StepEvent:
    """
    A turn just completed. Emitted before the next one starts.

    `depth` is 0 for the agent the caller invoked and one greater for each
    level of delegation below it. A consumer that does not care can ignore
    it; one rendering live progress uses it to indent, so a specialist's
    steps read as belonging to the delegation above them rather than as more
    steps the supervisor took.
    """

    step: AgentStep
    depth: int = 0


@dataclass
class ResultEvent:
    """The run ended. Always the last event, exactly once."""

    result: LoopResult


# What `stream` yields. A tagged pair rather than a bare AgentStep, because
# a consumer needs to distinguish "another step happened" from "this is the
# answer" - and a caller that only wants the answer (`run`) has to be able
# to pick it out without counting.
LoopEvent = StepEvent | ResultEvent


def _extract_json_object(text: str) -> str | None:
    """
    Return the first balanced `{...}` in `text`, or None.

    Deliberately a brace scanner and not a regex or a `find('{')` /
    `rfind('}')` pair. Both of those break on the two things models actually
    do: writing prose after the JSON (rfind then swallows a stray brace from
    the prose) and nesting the input object inside the action object (a
    non-greedy regex stops at the inner `}`).

    String-aware, because a brace inside a JSON string value - which is
    exactly what a `finish` answer quoting a document can contain - must not
    change the depth count. Escape-aware for the same reason one level down:
    a `\\"` inside that string is not the end of the string.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    # Unbalanced - the model was cut off mid-object, usually by a token
    # limit while writing a long `finish` answer.
    return None


def _first_present(payload: dict, keys: tuple[str, ...]):
    """Return the value of the first key in `keys` that `payload` has."""
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _parse_action(raw: str) -> tuple[str, str, str, dict]:
    """
    Pull `(thought, action_json, tool, tool_input)` out of the model's reply.

    This function is the whole cost of not using the provider's tool-calling
    API, concentrated in one place. Everything it tolerates, it tolerates
    because a model actually did it:

    - a ```json fence around the action, despite being told not to
    - prose after the closing brace ("...and then I'll summarise.")
    - `"action"`/`"action_input"` instead of `"tool"`/`"input"`
    - a bare string where an object was asked for:
      `{"tool": "search", "input": "refund policy"}`
    - no `input` key at all on a no-argument tool

    What it does NOT do is guess a tool name. A reply naming a tool that
    does not exist is a real error the model has to see and correct, and
    silently rewriting it to the nearest match would hide a prompt problem
    behind a plausible-looking run.

    Raises:
        ActionParseError: with a message written *for the model* - it is fed
            back as the next Observation, so it has to say what to do, not
            just what went wrong.
    """
    text = raw.strip()

    # Strip a surrounding code fence if there is one. Done on the whole
    # reply rather than on the action alone because the model that fences
    # the action usually fences the thought along with it.
    fenced = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    action_marker = re.search(r"Action\s*:", text, re.IGNORECASE)

    # The thought is whatever came before the Action marker, minus its own
    # label. Purely for the transcript - nothing branches on it - so a
    # missing or oddly-formatted thought is never an error.
    if action_marker:
        thought = text[: action_marker.start()]
    else:
        thought = text
    thought = re.sub(r"^\s*Thought\s*:", "", thought, flags=re.IGNORECASE).strip()

    # Search for the JSON from the Action marker onwards when there is one:
    # a thought like `the {policy} section` would otherwise be scanned first
    # and parsed as a malformed action.
    search_from = text[action_marker.end() :] if action_marker else text
    candidate = _extract_json_object(search_from)

    if candidate is None:
        raise ActionParseError(
            "Your last reply contained no Action. Reply again with a "
            "Thought line and then an Action line whose value is a single "
            'JSON object, for example: Action: {"tool": "search", "input": '
            '{"query": "refund window"}}'
        )

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ActionParseError(
            f"Your Action was not valid JSON ({exc.msg}). Reply again with "
            "the Action on a single line as one JSON object, with double "
            "quotes around every key and string value, and no trailing commas."
        ) from exc

    if not isinstance(payload, dict):
        raise ActionParseError(
            "Your Action must be a JSON object with a \"tool\" key, not a "
            f"{type(payload).__name__}."
        )

    tool = _first_present(payload, _TOOL_KEYS)
    if not isinstance(tool, str) or not tool.strip():
        raise ActionParseError(
            'Your Action is missing the "tool" key. It must name one of the '
            "tools you were given."
        )

    tool_input = _first_present(payload, _INPUT_KEYS)
    if tool_input is None:
        # A no-argument tool invoked without an input key is correct usage,
        # not an error.
        tool_input = {}
    elif isinstance(tool_input, str):
        # A bare string for a single-argument tool. Which argument it means
        # is the tool's business, not the parser's, so it is passed through
        # under a reserved key the tools know to look for.
        tool_input = {"__bare__": tool_input}
    elif not isinstance(tool_input, dict):
        raise ActionParseError(
            'The "input" of your Action must be a JSON object, for example '
            '{"query": "refund window"}.'
        )

    return thought, candidate, tool.strip(), tool_input


def _observation_of(returned) -> str:
    """
    Normalise what a tool returned into the observation text.

    Tools return `str`; a delegating one may return `ToolResult` instead.
    Accepting both here is what keeps the dispatch in `stream` from growing a
    second branch for a distinction the rest of the loop does not care about.
    """
    if isinstance(returned, ToolResult):
        return returned.observation
    return returned


def _forward(
    tool: Tool, tool_input: dict
) -> Generator[StepEvent, None, tuple[str, list[AgentStep]]]:
    """
    Run a delegating tool, forwarding its sub-agent's steps as they land.

    Yields each sub-step re-tagged one level deeper, and returns
    `(observation, children)` for the step that caused them.

    The re-tagging is done here rather than by the tool because depth is
    relative: a specialist streaming its own steps has no idea whether it was
    called directly or from three levels up, and should not have to. Each
    layer adds one on the way past, which composes to the right number
    however deep the tree goes.

    A tool whose `stream` ends without a ToolResult yields an empty
    observation, which the model reads as a tool that did nothing and can
    recover from. That is preferable to raising: the contract is documented
    on Tool, and a delegating tool that breaks it should cost a step rather
    than the whole run.
    """
    children: list[AgentStep] = []
    observation = ""

    for event in tool.stream(tool_input):
        if isinstance(event, StepEvent):
            children.append(event.step)
            yield StepEvent(event.step, depth=event.depth + 1)
        elif isinstance(event, ToolResult):
            observation = event.observation
            # A tool that assembled its own child list wins over the one
            # accumulated here - the vision agent, for instance, knows which
            # of its steps are worth showing and this loop does not.
            if event.children:
                children = event.children

    return observation, children


class AgentLoop:
    """
    Runs tools until one of them ends the run or the budget does.

    Stateless between calls: every `run` builds its own step list. Anything
    an agent needs to carry across steps - retrieved evidence, a cache, a
    "have I done X yet" flag - lives on the agent that owns the tools, and
    is that agent's job to reset.
    """

    def __init__(
        self,
        llm_service,
        tools: list[Tool],
        system_prompt: str,
        synthesis_system_prompt: str,
        max_steps: int = 6,
        temperature: float | None = 0.0,
        budget: StepBudget | None = None,
        run_name: str = "agent",
    ) -> None:
        if not any(tool.terminal for tool in tools):
            # Without one there is no way to finish successfully - every run
            # would burn the full budget and end in synthesis. Caught here
            # rather than discovered as a mysterious latency problem.
            raise ValueError("An agent loop needs at least one terminal tool.")

        self._llm_service = llm_service
        self._tools = {tool.name: tool for tool in tools}
        self._system_prompt = system_prompt
        self._synthesis_system_prompt = synthesis_system_prompt
        self._max_steps = max_steps
        # None means this loop is the root of its own tree, so it is also the
        # whole tree and its own ceiling is the only bound that exists. A
        # supervisor passes its shared pool in, and from then on both bounds
        # apply - see StepBudget.
        #
        # Held rather than resolved here: a budget is per-run state, and a
        # loop built once at dependency-injection time is reused across
        # requests. `stream` makes a fresh one per run when there is none.
        self._budget = budget
        self._run_name = run_name
        # Defaults to 0 here rather than to None, unlike LLMService itself.
        # A loop reply is parsed against a grammar on every single turn, so
        # determinism is the correct default for this caller specifically -
        # and a loop that silently varied its tool choices between identical
        # runs would be untestable and unevaluable.
        self._temperature = temperature

    @property
    def tools(self) -> dict[str, Tool]:
        return self._tools

    def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        preamble: str = "",
    ) -> LoopResult:
        """
        Work on `question` and return only the finished result.

        A thin drain of `stream`, deliberately - not a second copy of the
        loop. The routing agent this project started with kept two: its
        `stream` restated the dispatch its graph already encoded, and only a
        test held the two in agreement. Here there is one loop and one set of
        stopping conditions, and a caller that does not care about progress
        simply throws the step events away.

        The same pattern is now used by every `run` in the agents package,
        including each delegating tool's - see agents/supervisor_agent.py.

        Raises:
            ValueError: if `question` is empty.
            RuntimeError: if an LLM call fails (propagated from llm_service).
        """
        result: LoopResult | None = None
        for event in self.stream(question, history=history, preamble=preamble):
            if isinstance(event, ResultEvent):
                result = event.result

        # stream always ends in a ResultEvent - every exit path yields one -
        # so this is unreachable rather than a fallback. Asserted rather
        # than assumed, because a new early return in stream that forgot to
        # emit one would otherwise surface as a None answer far downstream.
        assert result is not None, "stream() ended without a ResultEvent"
        return result

    def stream(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        preamble: str = "",
    ) -> Iterator[LoopEvent]:
        """
        Work on `question`, emitting each turn as it completes.

        Yields a `StepEvent` the moment a step is recorded - after the tool
        has run and its observation is known - and exactly one `ResultEvent`
        last, on every exit path.

        This is the primitive; `run` is built on it. The reason to stream
        steps rather than tokens is that a step is the unit the user
        actually cares about: watching the agent decide to read the text and
        then check a policy is informative, where watching an answer appear
        one word at a time after twenty seconds of silence is not.

        Raises:
            ValueError: if `question` is empty - eagerly, before the first
                yield, so a caller that has already sent HTTP headers is not
                the one to discover it.
            RuntimeError: if an LLM call fails - possibly mid-stream.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty.")

        tool_catalog = format_tool_catalog(self._tools.values())
        history_block = format_history_block(history)

        # The work lives in an inner generator so the validation above runs
        # at call time rather than on the first `next()`. Same reason
        # ai/llm_service.py's stream_response builds its messages eagerly:
        # by the time a caller starts iterating, an HTTP response has
        # usually already committed to 200 and a 400 is no longer available.
        def events() -> Iterator[LoopEvent]:
            steps: list[AgentStep] = []
            consecutive_parse_failures = 0

            # A fresh pool per run when this loop is the root of its own
            # tree; the caller's shared one when it is a node in someone
            # else's. Resolved here rather than in __init__ because a loop
            # built at dependency-injection time is reused across requests,
            # and a budget is per-run state.
            budget = self._budget or StepBudget(self._max_steps)
            # This agent's own ceiling, tracked separately so both bounds
            # apply: the pool stops the tree, this stops one agent inside it
            # from spending everything the others still need.
            own_steps_left = self._max_steps

            with record_run(self._run_name) as run:
                while own_steps_left > 0 and budget.take():
                    own_steps_left -= 1
                    prompt = format_loop_prompt(
                        question=question,
                        tool_catalog=tool_catalog,
                        # Whichever bound bites first, counted inclusive of
                        # the step now being taken. A sub-agent told it has
                        # six steps when the tree has two left plans work it
                        # will not get to finish.
                        steps_left=min(own_steps_left, budget.remaining) + 1,
                        scratchpad=format_scratchpad(steps),
                        history_block=history_block,
                        preamble=preamble,
                    )
                    raw = self._llm_service.generate_response(
                        prompt,
                        system_prompt=self._system_prompt,
                        temperature=self._temperature,
                    )

                    try:
                        thought, action_json, tool_name, tool_input = _parse_action(raw)
                    except ActionParseError as exc:
                        consecutive_parse_failures += 1
                        if consecutive_parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                            # Out of retries on the format. Whatever was
                            # gathered before the failures is still worth an
                            # answer.
                            run.stopped_because = "parse_failures"
                            yield ResultEvent(
                                self._synthesise(question, steps, "parse_failures")
                            )
                            return

                        # Recorded as a step so the model sees its own broken
                        # reply next to the complaint about it - told only
                        # "that was wrong", without the reply in front of it,
                        # models tend to repeat the same mistake.
                        steps.append(
                            AgentStep(
                                thought="(unparseable reply)",
                                action_json=raw.strip()[:400],
                                tool="",
                                tool_input={},
                                observation=str(exc),
                            )
                        )
                        run.record_tool("")
                        yield StepEvent(steps[-1])
                        continue

                    consecutive_parse_failures = 0
                    run.record_tool(tool_name)
                    tool = self._tools.get(tool_name)

                    if tool is None:
                        steps.append(
                            AgentStep(
                                thought=thought,
                                action_json=action_json,
                                tool=tool_name,
                                tool_input=tool_input,
                                observation=(
                                    f'There is no tool called "{tool_name}". The '
                                    f"tools you have are: {', '.join(self._tools)}."
                                ),
                            )
                        )
                        yield StepEvent(steps[-1])
                        continue

                    if tool.terminal:
                        try:
                            answer = tool.run(tool_input)
                        except ActionRejected as exc:
                            # The tool declined - finishing too early, or
                            # with nothing. Its message says what to do next.
                            steps.append(
                                AgentStep(
                                    thought=thought,
                                    action_json=action_json,
                                    tool=tool_name,
                                    tool_input=tool_input,
                                    observation=str(exc),
                                )
                            )
                            yield StepEvent(steps[-1])
                            continue

                        steps.append(
                            AgentStep(
                                thought=thought,
                                action_json=action_json,
                                tool=tool_name,
                                tool_input=tool_input,
                                observation="(final answer)",
                            )
                        )
                        yield StepEvent(steps[-1])
                        run.stopped_because = "finished"
                        yield ResultEvent(
                            LoopResult(
                                answer=answer.strip(),
                                steps=steps,
                                stopped_because="finished",
                            )
                        )
                        return

                    # A tool raising is information, not a crash: the model
                    # can pick different arguments or a different tool next
                    # turn. Only the LLM's own failures (RuntimeError from
                    # llm_service) end the run, since there is no next turn
                    # without it.
                    children: list[AgentStep] = []
                    try:
                        if tool.stream is not None:
                            # A delegating tool. Its sub-agent's steps are
                            # forwarded as they land, so the caller sees
                            # progress during the delegation rather than
                            # after it.
                            observation, children = yield from _forward(
                                tool, tool_input
                            )
                        else:
                            observation = _observation_of(tool.run(tool_input))
                    except ActionRejected as exc:
                        observation = str(exc)
                    except Exception as exc:  # noqa: BLE001 - surfaced to the model
                        observation = f"The {tool_name} tool failed: {exc}"

                    steps.append(
                        AgentStep(
                            thought=thought,
                            action_json=action_json,
                            tool=tool_name,
                            tool_input=tool_input,
                            observation=observation,
                            children=children,
                        )
                    )
                    yield StepEvent(steps[-1])

                # Out of steps with no terminal call - either this agent's
                # own ceiling or the tree's pool. See format_synthesis_prompt
                # for why this writes an answer instead of reporting the
                # limit.
                run.stopped_because = "step_limit"
                yield ResultEvent(self._synthesise(question, steps, "step_limit"))

        return events()

    def _synthesise(
        self,
        question: str,
        steps: list[AgentStep],
        stopped_because: Literal["step_limit", "parse_failures"],
    ) -> LoopResult:
        """
        Write the final answer from the scratchpad, without asking for an action.

        Sent with the synthesis system prompt rather than the loop's own:
        that one requires an Action line on every reply, which is the exact
        opposite of what is wanted here, and leaving it in place reliably
        produces one more Thought/Action pair instead of an answer.
        """
        if not steps:
            # Nothing was gathered - the very first reply was unparseable.
            # There is nothing to synthesise from, and an LLM call here would
            # invite an answer grounded in nothing at all.
            return LoopResult(
                answer=(
                    "I wasn't able to work on this - the reasoning step "
                    "failed before any tool ran. Please try rephrasing it."
                ),
                steps=steps,
                stopped_because=stopped_because,
            )

        answer = self._llm_service.generate_response(
            format_synthesis_prompt(question, format_scratchpad(steps)),
            system_prompt=self._synthesis_system_prompt,
            temperature=self._temperature,
        )
        return LoopResult(
            answer=answer.strip(), steps=steps, stopped_because=stopped_because
        )
