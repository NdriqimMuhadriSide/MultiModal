"""
The bridge: one in-process research run, expressed as an A2A Task.

This is the only file in the package that knows what kind of agent is behind
the protocol. Everything else - the envelope, the types, the dispatch - would
be identical for the vision agent, which is the test of whether the split is
in the right place.

WHAT THE MAPPING ACTUALLY IS

    A2A                          this codebase
    ------------------------     ----------------------------------------
    message text                 the question
    contextId                    conversation_id (memory/conversation_memory)
    task id                      no equivalent - invented here, per call
    artifact "answer"            ResearchResult.answer
    artifact "evidence"          ResearchResult.sources, with [E#] labels
    artifact "trace"             ResearchResult.steps
    status.state                 always terminal, since the run is blocking

WHY THE EVIDENCE IS A SEPARATE ARTIFACT WITH EXPLICIT LABELS

This is the part worth reading twice, because it is the thing that stops
working the moment the agent is on the other end of a socket.

In-process, agents/knowledge_base_tool.py's EvidenceLedger is a single object
shared by reference across the whole tree, so `[E3]` means one specific
passage everywhere. Over HTTP there is no shared object: this process
numbers from its own ledger starting at E1, and so does the caller's, and so
does every other peer. Two agents both hand out `[E1]` for different
passages, the caller merges them, and the citation list now points at the
wrong text with nothing raised anywhere - the exact silent corruption
agents/supervisor_agent.py's docstring calls out, reintroduced by the
network.

Nothing here can fix that alone, because only the caller knows what else it
is merging. What this can do is make the fix *possible*: emit the label
alongside the `chunkId` it stands for, so a caller can re-label into its own
ledger and rewrite the answer text accordingly. Returning the sources as an
ordered list and leaving the correspondence implicit - index i is E(i+1),
which is the in-process convention - would force the caller to rely on an
invariant it cannot verify. The reconciliation itself is phase 4; this is
the artifact it needs to exist.

WHY THE TRACE COMES TOO

The trace is first-class in this project (app/schemas/agent_trace.py: "trust
me, I looked" is not a reasonable thing to ask), and a delegation is exactly
where it matters most, since the caller did not watch any of it happen. It
rides as JSON rather than prose so a caller can render it with the renderer
it already has.

WHAT IT DEPENDS ON

A research service, duck-typed: anything with

    research(question: str, conversation_id: str | None) -> result

where the result carries `.answer`, `.conversation_id`, `.steps`, `.sources`
and `.stopped_because`. app/services/research_service.py's
ResearchChatService is that, and passing it in rather than importing it
keeps this module testable with a stub and keeps the protocol layer from
depending on the app's dependency graph.
"""
import uuid
from datetime import datetime, timezone

from app.schemas.agent_trace import to_step_payload
from a2a.types import Artifact, DataPart, Message, Task, TaskStatus, TextPart

# Artifact names, fixed so a caller can address them.
#
# A caller looks these up by name - it wants the answer, not "the first
# artifact" - so they are part of the contract with anything that consumes
# this agent, and renaming one is a breaking change.
ANSWER_ARTIFACT = "answer"
EVIDENCE_ARTIFACT = "evidence"
TRACE_ARTIFACT = "trace"


def _now() -> str:
    """An RFC 3339 timestamp, which is what the protocol asks for."""
    return datetime.now(timezone.utc).isoformat()


def _agent_message(text: str, task_id: str, context_id: str) -> Message:
    """A message from this agent, for a status that needs words."""
    return Message(
        role="agent",
        parts=[TextPart(text=text)],
        messageId=str(uuid.uuid4()),
        taskId=task_id,
        contextId=context_id,
    )


def _evidence_records(sources) -> list[dict]:
    """
    Pair each retrieved passage with the label the answer cites it by.

    The label is derived from position - index i is [E(i+1)] - because that
    is the invariant EvidenceLedger.sources guarantees and the only place it
    is knowable. Derived *here* and written down explicitly, so that the one
    process which can compute it correctly is the one that does, rather than
    every caller re-deriving it from a convention it has no way to check.
    """
    return [
        {
            "label": f"E{index + 1}",
            "chunkId": source.chunk_id,
            "filename": source.filename,
            "page": source.page,
            "section": source.section,
        }
        for index, source in enumerate(sources)
    ]


class ResearchExecutor:
    """
    Runs the research agent for one A2A message and returns a finished Task.

    Blocking, and terminal in one call: there is no `working` state a caller
    could observe, because there is nothing between the request and the
    answer that this phase can show them. Phase 3 changes that by streaming
    status updates, and it changes it here rather than anywhere else.
    """

    def __init__(self, research_service) -> None:
        self._service = research_service

    def execute(self, question: str, context_id: str | None) -> Task:
        """
        Answer `question` and package the run as a completed Task.

        `context_id` is the caller's conversation, or None to start one. The
        returned task's `contextId` is authoritative either way: the service
        generates an id when given none, and a caller that wants a follow-up
        must send back what it got rather than what it sent.

        Never raises for an agent-level failure. A provider outage comes back
        as a Task in state `failed` with a message saying so, because the
        caller is another agent: a failed task is an observation it can act
        on - answer without this specialist, tell the user, try later - where
        a transport-level error is something it can only propagate. Compare
        agents/agent_loop.py, which turns a tool's exception into an
        observation for exactly the same reason.
        """
        task_id = str(uuid.uuid4())

        try:
            result = self._service.research(
                question=question, conversation_id=context_id
            )
        except RuntimeError as exc:
            # RuntimeError specifically - ai/llm_service.py wraps every
            # provider failure into one, so this is the whole class of
            # "the model was unreachable". A bug in this module should still
            # surface as a 500 rather than as a polite `failed` task that
            # hides it.
            return Task(
                id=task_id,
                contextId=context_id or str(uuid.uuid4()),
                status=TaskStatus(
                    state="failed",
                    message=_agent_message(
                        f"The research agent could not complete this request: {exc}",
                        task_id,
                        context_id or "",
                    ),
                    timestamp=_now(),
                ),
            )

        resolved_context = result.conversation_id

        artifacts = [
            Artifact(
                artifactId=str(uuid.uuid4()),
                name=ANSWER_ARTIFACT,
                description=(
                    "The answer, citing evidence as [E1]-style labels defined "
                    f"in the '{EVIDENCE_ARTIFACT}' artifact."
                ),
                parts=[TextPart(text=result.answer)],
            )
        ]

        # Omitted rather than sent empty when the agent answered without
        # retrieving anything. An empty evidence artifact and an absent one
        # mean different things to a caller merging citations, and "there
        # were no passages" is said more clearly by not claiming to carry any.
        records = _evidence_records(result.sources)
        if records:
            artifacts.append(
                Artifact(
                    artifactId=str(uuid.uuid4()),
                    name=EVIDENCE_ARTIFACT,
                    description=(
                        "Each passage the answer cites, paired with the label "
                        "it is cited by. Labels are local to this run: a "
                        "caller merging several agents' evidence must "
                        "re-label by chunkId and rewrite the answer text."
                    ),
                    parts=[DataPart(data={"evidence": records})],
                )
            )

        artifacts.append(
            Artifact(
                artifactId=str(uuid.uuid4()),
                name=TRACE_ARTIFACT,
                description=(
                    "Every step the agent took to reach the answer, in order."
                ),
                parts=[
                    DataPart(
                        data={"steps": [to_step_payload(step) for step in result.steps]}
                    )
                ],
            )
        )

        return Task(
            id=task_id,
            contextId=resolved_context,
            status=TaskStatus(state="completed", timestamp=_now()),
            artifacts=artifacts,
            # `completed` even when the loop ran out of steps, because there
            # is a real answer either way - agents/agent_loop.py synthesises
            # one from the scratchpad rather than reporting the limit. The
            # difference is reported here instead, because the three
            # outcomes are not equally trustworthy and a caller that renders
            # a truncated run identically to a finished one is hiding that
            # from its reader.
            metadata={
                "stoppedBecause": result.stopped_because,
                "stepCount": len(result.steps),
            },
        )
