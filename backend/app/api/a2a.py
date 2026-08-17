"""
The two HTTP routes A2A needs, and nothing else.

MOUNTED AT THE SITE ROOT, NOT UNDER /api/v1

`/.well-known/agent-card.json` is a fixed path in the protocol, the same way
`/.well-known/openid-configuration` is: a client that has to be told where
the card lives has not discovered anything. Putting it behind this project's
own version prefix would make it undiscoverable by every generic A2A client,
which is the entire point of the well-known path. See app/main.py, where
this router is included with no prefix.

The RPC path is configurable (A2A_RPC_PATH) because it is not fixed by the
spec - the card's `url` field is what tells a caller where to send, and
a2a/card.py builds that from the same setting, so the two cannot drift.

WHY THESE ROUTES ARE ASYNC WHEN EVERY OTHER ROUTE HERE IS NOT

Two reasons, and the second is the one that matters:

    the raw body is needed. A payload that is not JSON at all must come back
    as -32700, and FastAPI's usual model binding would have answered 422
    before this function ran.

    an agent run blocks for seconds. Awaiting the body puts this route on the
    event loop, so calling the blocking handler directly would stall every
    other request in the process for the length of an LLM call.
    `run_in_threadpool` is what puts it back where the project's synchronous
    routes already run.
"""
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from a2a.card import build_agent_card
from a2a.executor import ResearchExecutor
from a2a.jsonrpc import PARSE_ERROR, JSONRPCError, error_response
from a2a.server import A2AServer
from a2a.task_store import get_task_store
from app.core.config import settings
from app.services.research_service import (
    ResearchChatService,
    get_research_chat_service,
)

router = APIRouter(tags=["a2a"])

AGENT_CARD_PATH = "/.well-known/agent-card.json"


@router.get(AGENT_CARD_PATH)
def agent_card() -> JSONResponse:
    """
    Serve this agent's card.

    Unauthenticated by design, and the one route that should stay that way
    even after phase 5: the card is how a caller learns which credentials to
    present, so requiring credentials to read it is a loop with no entry.
    What the card must therefore never contain is anything private - it
    describes capabilities, not corpus contents.

    `exclude_none` so unset optional fields are absent rather than null,
    matching the spec's own examples.
    """
    return JSONResponse(build_agent_card().model_dump(exclude_none=True))


@router.post(settings.a2a_rpc_path)
async def a2a_rpc(
    request: Request,
    research_chat_service: ResearchChatService = Depends(get_research_chat_service),
) -> JSONResponse:
    """
    The JSON-RPC endpoint. Answers 200 for everything the protocol covers.

    Including protocol failures - see a2a/jsonrpc.py. The only thing this
    layer decides is that a body which is not JSON gets -32700 with a null
    id, because there is no id to echo when the envelope itself did not
    parse.

    The one non-200 comes from the dependency: an unconfigured deployment
    (no API key, no retriever) raises 503 before this body runs. That is
    correct rather than a leak. A2A puts transport-level conditions at the
    transport level - authentication failures are 401, not a JSON-RPC error -
    and "this agent is not available at all" belongs in the same category as
    those, not in an `error` member describing a request that was never the
    problem.

    The server object is built per request rather than held as a module
    singleton, so the executor wraps whatever the dependency resolved this
    time - which is what lets a test swap the service without the route
    holding the first one forever.
    """
    body = await request.body()
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(
            error_response(
                None, JSONRPCError(PARSE_ERROR, "Request body is not valid JSON.", data=str(exc))
            )
        )

    server = A2AServer(
        executor=ResearchExecutor(research_chat_service), task_store=get_task_store()
    )
    # In the threadpool, not inline: this route is async so it can read the
    # raw body, and `handle` runs an agent for several seconds. Calling it
    # directly would stall the event loop - and with it every other request
    # in the process - for the length of an LLM call.
    return JSONResponse(await run_in_threadpool(server.handle, payload))
