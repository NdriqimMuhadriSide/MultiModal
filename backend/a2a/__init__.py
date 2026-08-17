"""
A2A (Agent2Agent) — the protocol layer that lets agents in *different
processes* call each other.

WHAT THIS IS, AND WHAT IT IS NOT

agents/supervisor_agent.py already does agent-to-agent delegation: its tools
are other agents, and a specialist is reached by a Python call. That works
because everything shares one process - one `StepBudget` object, one
`EvidenceLedger`, one exception type.

This package is the same relationship over HTTP, where none of that is true.
It is a *transport*, not an orchestration framework: nothing here decides
what to delegate or when. A remote agent still arrives at AgentLoop as a
`Tool` (agents/agent_loop.py) and the loop never learns the difference.

LAYOUT

    types.py       the wire objects - Agent Card, Message, Part, Task
    jsonrpc.py     the JSON-RPC 2.0 envelope and its error codes
    card.py        this deployment's Agent Card, built from settings
    task_store.py  where a Task lives between `message/send` and `tasks/get`
    executor.py    the bridge: an in-process agent run -> an A2A Task
    server.py      method dispatch, with no web framework in it

app/api/a2a.py is the FastAPI shim over server.py, and is deliberately thin -
the protocol is testable without a TestClient, and phase 3 (streaming) changes
server.py rather than the route.

WHY THIS IS HAND-WRITTEN

There is an official `a2a-sdk` for Python, and a production system with no
teaching goal should probably use it. It is not used here for the same
reason agents/agent_loop.py does not use the provider's tool-calling API:
writing the protocol out is the point, and the whole of it that this project
needs is a few hundred lines of Pydantic plus a dispatch table.

WHAT PHASE 1 COVERS

    GET  /.well-known/agent-card.json   discovery
    POST /a2a/v1  message/send          run the research agent, blocking
    POST /a2a/v1  tasks/get             read a finished task back

Deliberately absent, and reported honestly as unsupported rather than
half-implemented (see server.py):

    message/stream          phase 3 - the card advertises streaming: false
    tasks/cancel            nothing here is long-running enough to cancel yet
    push notifications      needs a task store that outlives the process
    authentication          phase 5 - see card.py on securitySchemes
"""
