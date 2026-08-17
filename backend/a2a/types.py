"""
The A2A wire objects, as Pydantic models.

CAMELCASE, ON PURPOSE

Every other schema in this project is snake_case with a camelCase alias
(see app/schemas/research.py). These are camelCase outright, with no alias,
because they are not this project's API - they are somebody else's spec, and
the only reader that matters is a client written against the A2A
documentation. A snake_case mirror with aliases would add a second spelling
of every field for the benefit of Python code that mostly just forwards them.

THE FOUR SHAPES WORTH UNDERSTANDING

    Part      the unit of content. Text, a file, or arbitrary JSON. A
              message is a list of them, and so is an artifact, which is
              what makes "here is my answer AND the evidence behind it"
              expressible without inventing a field for it.

    Message   one turn of conversation. `role` is "user" or "agent" - and
              note that a *calling agent* sends role="user", because the
              roles are relative to the agent being addressed, not absolute
              positions in a hierarchy.

    Task      a unit of work with a lifecycle. This is the piece with no
              in-process equivalent: a Python call either returns or raises,
              where a Task can be `working` for a minute, ask for more input,
              be fetched again later, or be cancelled. It is closer to a
              WorkManager job than to a function call.

    AgentCard what an agent says it can do. Served at a well-known URL so a
              client can discover it without prior configuration.

TWO IDS, AND WHY BOTH

`id` identifies one unit of work. `contextId` identifies the conversation it
belongs to, and is stable across many tasks. The split is what lets a caller
ask a follow-up - same contextId, new task - and it maps cleanly onto this
project's existing `conversation_id` (memory/conversation_memory.py), which
is why a remote research call gets conversation continuity for free.
"""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# Where a Task can be. The three that matter for phase 1 are `completed`,
# `failed` and `rejected`; the rest are named because a client switches on
# this field and an unlisted value is an error for it, not a surprise.
#
# `input-required` is the interesting one long-term: it is how a remote agent
# says "I need something from the user before I can continue" without failing,
# and it is the reason a Task is not just a slow function call.
TaskState = Literal[
    "submitted",
    "working",
    "input-required",
    "auth-required",
    "completed",
    "canceled",
    "failed",
    "rejected",
    "unknown",
]


# --- Parts -----------------------------------------------------------------


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class FileWithBytes(BaseModel):
    """A file inlined as base64."""

    name: str | None = None
    mimeType: str | None = None
    bytes: str


class FileWithUri(BaseModel):
    """A file the receiver is expected to fetch itself."""

    name: str | None = None
    mimeType: str | None = None
    uri: str


class FilePart(BaseModel):
    """
    A file, either inlined or referenced.

    Unused in phase 1 - the research agent takes text and returns text - and
    present anyway because it is the shape phase 3 needs for the vision
    agent, and because a client reading the card should see a complete type
    model rather than one that grows as this project does.
    """

    kind: Literal["file"] = "file"
    file: FileWithBytes | FileWithUri
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    """
    Structured JSON, for content that is not prose.

    The most load-bearing part type here. An answer is text, but the
    *evidence* behind it is a list of records with chunk ids in it, and
    flattening that into prose would destroy exactly the information a
    calling agent needs to merge citations (see executor.py on why the
    evidence artifact carries labels explicitly).
    """

    kind: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None


# Discriminated on `kind` so Pydantic picks the member by tag rather than by
# trying each in turn. Without the discriminator a DataPart whose `data`
# happened to validate as something else would silently parse as the wrong
# type, which is the class of bug that shows up as a missing field far away.
Part = Annotated[TextPart | FilePart | DataPart, Field(discriminator="kind")]


# --- Messages, artifacts, tasks ---------------------------------------------


class Message(BaseModel):
    """
    One turn addressed to, or produced by, an agent.

    `taskId` and `contextId` are absent on a first message and present on a
    follow-up. Absent means "start something new"; the server assigns both
    and returns them, which is the same contract this project's own endpoints
    already use for `conversation_id` (app/schemas/research.py).
    """

    kind: Literal["message"] = "message"
    role: Literal["user", "agent"]
    parts: list[Part]
    messageId: str
    taskId: str | None = None
    contextId: str | None = None
    metadata: dict[str, Any] | None = None

    def text(self) -> str:
        """
        Every TextPart, joined.

        Non-text parts are dropped rather than rendered. A caller that sent a
        file to a text-only agent should be told the content type is not
        supported (see server.py), not have its file silently summarised into
        the prompt as "[file]".
        """
        return "\n\n".join(
            part.text for part in self.parts if isinstance(part, TextPart)
        ).strip()


class Artifact(BaseModel):
    """
    An output the task produced.

    Distinct from the status message, which says how the task is *going*.
    Artifacts are what it made, they are addressable by id, and there can be
    several - which is the whole reason the answer, the evidence and the
    trace can come back as three named things instead of one blob a caller
    has to parse apart.
    """

    artifactId: str
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    metadata: dict[str, Any] | None = None


class TaskStatus(BaseModel):
    """
    Where the task is, and optionally what it wants to say about that.

    `message` is for a state that needs words - `input-required` explaining
    what it needs, `failed` explaining what broke. It is not where the answer
    goes; that is an artifact.
    """

    state: TaskState
    message: Message | None = None
    timestamp: str | None = None


class Task(BaseModel):
    """
    A unit of work, addressable after the call that created it.

    `history` is the messages exchanged, and is bounded by the caller's
    `historyLength` rather than returned whole - a long conversation
    re-sent on every poll is the same recurring cost this project already
    manages for prompts (see memory/compaction.py).
    """

    kind: Literal["task"] = "task"
    id: str
    contextId: str
    status: TaskStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


# --- Method parameters ------------------------------------------------------


class MessageSendConfiguration(BaseModel):
    """
    How the caller wants the call handled.

    `blocking` is honoured trivially in phase 1: this implementation only
    has a blocking mode, so a caller asking for non-blocking gets a completed
    task rather than a `submitted` one. That is a superset of what it asked
    for, not a violation - it wanted to not wait, and it did not have to.
    """

    acceptedOutputModes: list[str] | None = None
    blocking: bool | None = None
    historyLength: int | None = None


class MessageSendParams(BaseModel):
    """Params for `message/send` and (from phase 3) `message/stream`."""

    message: Message
    configuration: MessageSendConfiguration | None = None
    metadata: dict[str, Any] | None = None


class TaskQueryParams(BaseModel):
    """Params for `tasks/get`."""

    id: str
    historyLength: int | None = None


# --- Agent Card -------------------------------------------------------------


class AgentCapabilities(BaseModel):
    """
    What optional parts of the protocol this agent implements.

    Every field defaults to False, and phase 1 leaves them there. That is the
    point of the card: a client reads `streaming: false` and calls
    `message/send` instead of discovering by trial that `message/stream`
    returns an error. Advertising a capability that is not implemented is
    worse than not having it.
    """

    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentSkill(BaseModel):
    """
    One thing the agent can be asked to do.

    Written for another *model* to read, not for a developer - a calling
    agent decides whether to delegate here based on this text, which makes
    it the same kind of object as `Tool.description` in
    agents/agent_loop.py, and it should be written with the same care.
    `examples` matter more than they look: they are the closest thing the
    protocol has to a few-shot prompt for the routing decision.
    """

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] | None = None
    inputModes: list[str] | None = None
    outputModes: list[str] | None = None


class AgentProvider(BaseModel):
    organization: str
    url: str


class AgentCard(BaseModel):
    """
    The agent's public description, served at a well-known URL.

    Discovery, in the sense an Android intent-filter is discovery: a client
    that knows only the host learns the name, the capabilities, the calling
    convention and the authentication scheme without being configured for
    this specific agent in advance.

    `url` is the JSON-RPC endpoint, and is NOT the URL the card itself was
    fetched from. It has to be the address a *third party* can reach, which
    is why it comes from configuration rather than from the incoming request
    (see card.py): behind a reverse proxy the request's own host is the
    internal one, and a card advertising that sends every future caller to
    an address that does not resolve for them.
    """

    protocolVersion: str
    name: str
    description: str
    url: str
    preferredTransport: str = "JSONRPC"
    version: str
    provider: AgentProvider | None = None
    documentationUrl: str | None = None
    capabilities: AgentCapabilities
    defaultInputModes: list[str]
    defaultOutputModes: list[str]
    skills: list[AgentSkill]
    # Both absent in phase 1, which means "no authentication". Honest rather
    # than convenient: a card that declared a scheme the server does not
    # check would tell a caller its bearer token was being validated when
    # nothing reads it. See card.py.
    securitySchemes: dict[str, Any] | None = None
    security: list[dict[str, list[str]]] | None = None
    supportsAuthenticatedExtendedCard: bool = False
