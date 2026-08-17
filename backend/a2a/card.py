"""
This deployment's Agent Card.

WHAT THE CARD IS FOR

It is the only thing a caller reads before deciding to delegate here. The
`skills` text is therefore written for a model, not for a developer - it is
the remote equivalent of a `Tool.description` in agents/agent_loop.py, and
the supervisor's own tool catalog (agents/supervisor_agent.py) is where to
look for the voice: say what the agent can do, say what it cannot, and give
examples of the questions it exists for.

WHY IT IS BUILT AT REQUEST TIME AND NOT A CONSTANT

Two fields depend on deployment rather than on code - the public URL and
the version - and one depends on what this process has actually implemented
(capabilities). A module-level constant would freeze all three at import,
and the URL in particular is the field most likely to be wrong: it must be
the address a *third party* can reach, not the one this process binds to.

WHERE IT IS SERVED

`/.well-known/agent-card.json`, at the site root - deliberately NOT under
this project's `/api/v1` prefix. The path is part of the protocol, the same
way `/.well-known/openid-configuration` is: a client that has to be told
where the card lives has not discovered anything.
"""
from app.core.config import settings
from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill

# The A2A revision these types were written against. Sent so a client can
# refuse a peer it cannot speak to, and pinned in configuration rather than
# hardcoded so a spec bump is a deploy rather than a patch.
DEFAULT_PROTOCOL_VERSION = "0.3.0"

RESEARCH_SKILL = AgentSkill(
    id="research_documents",
    name="Multi-hop document research",
    description=(
        "Answer a question about the documents ingested into this knowledge "
        "base, searching the corpus several times and letting each search be "
        "shaped by what the last one returned. Use it for anything that turns "
        "on what the documents actually say - comparisons between documents, "
        "multi-part questions, and questions whose second search depends on "
        "the first search's result. It cannot answer from general knowledge, "
        "reach the internet, or look at images; if the corpus does not cover "
        "the question it says so rather than guessing. The answer cites "
        "evidence with [E1]-style labels, and the labelled passages come back "
        "alongside it as a separate artifact."
    ),
    tags=["rag", "documents", "research", "retrieval", "citations"],
    # Not decoration. These are the closest thing the protocol has to
    # few-shot examples for another agent's routing decision, and they are
    # taken from the questions agents/research_agent.py was actually built
    # for - a single-retrieval question would be routed here wastefully.
    examples=[
        "How does our refund policy differ from the returns policy?",
        "Which of these reports mentions the Q3 shortfall, and what does it say?",
        "What does the handbook say about the process the contract requires?",
    ],
    inputModes=["text/plain"],
    outputModes=["text/plain", "application/json"],
)


def build_agent_card() -> AgentCard:
    """
    Describe this agent as it is right now.

    Capabilities are reported from what is implemented, not from what is
    planned: `streaming` stays False until server.py answers
    `message/stream`, because a caller that believes the advertisement will
    call it and get an error instead of an answer.
    """
    provider = (
        AgentProvider(
            organization=settings.a2a_provider_organization,
            url=settings.a2a_provider_url,
        )
        if settings.a2a_provider_organization
        else None
    )

    return AgentCard(
        protocolVersion=settings.a2a_protocol_version,
        name=settings.a2a_agent_name,
        # The one-liner a client sees in a list of agents. The detail belongs
        # on the skill, which is what a routing decision actually reads.
        description=(
            "Answers questions about an ingested document corpus by searching "
            "it repeatedly, following up on what it finds, and citing the "
            "passages it used."
        ),
        url=f"{settings.a2a_public_base_url.rstrip('/')}{settings.a2a_rpc_path}",
        preferredTransport="JSONRPC",
        version=settings.a2a_agent_version,
        provider=provider,
        capabilities=AgentCapabilities(
            # Phase 3. Until then, honestly false.
            streaming=False,
            # Needs a task store that outlives the process - see
            # a2a/task_store.py on why the in-memory one is not that.
            pushNotifications=False,
            stateTransitionHistory=False,
        ),
        # What this agent accepts and produces by default, in the absence of
        # a per-skill override. Text in; text out, plus the JSON artifacts
        # carrying evidence and the trace (see a2a/executor.py).
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[RESEARCH_SKILL],
        # Absent on purpose, and the single most important thing to change
        # before this is reachable from anywhere but localhost. An empty
        # `security` means unauthenticated: any caller who can reach the port
        # can spend this deployment's LLM budget and read its corpus. Phase 5
        # declares a scheme here AND enforces it in app/api/a2a.py - the card
        # is an advertisement, never a control.
        securitySchemes=None,
        security=None,
    )
