"""
Method dispatch for the A2A endpoint.

NO WEB FRAMEWORK IN HERE

`handle` takes an already-parsed JSON body and returns a JSON-RPC response
dict. That is the whole surface. app/api/a2a.py is the FastAPI shim over it
and stays about ten lines, which buys two things:

    the protocol is testable by calling one function, with no TestClient
    and no route table in the way

    phase 3 (message/stream) is a change to this file plus a route that
    knows how to emit SSE - not a rewrite of the dispatch, which is where
    the parameter validation and the error mapping live

WHAT IT REFUSES, AND HOW

Every unimplemented method is answered with -32004 UnsupportedOperation and
a message naming what to use instead. Not a 404, and not silence: a calling
agent that asked for streaming needs to learn it should send `message/send`,
and the card already told it so (a2a/card.py sets `streaming: false`). The
error exists for the client that did not read the card.

STATUS CODES

Every response below leaves HTTP at 200, including the failures. See
a2a/jsonrpc.py - a JSON-RPC error is a successful HTTP response carrying an
`error` member, and returning 4xx for one breaks clients that read the body.
"""
import logging

from pydantic import ValidationError

from a2a.jsonrpc import (
    CONTENT_TYPE_NOT_SUPPORTED,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
    JSONRPCError,
    JSONRPCRequest,
    error_response,
    success_response,
)
from a2a.types import MessageSendParams, Task, TaskQueryParams

logger = logging.getLogger(__name__)

# Methods the spec defines that this phase does not implement, each with the
# advice a caller needs. Answered explicitly rather than falling through to
# "no such method", because the distinction matters: "I have never heard of
# that" and "I know that one and cannot do it yet" lead a client to different
# next moves.
_NOT_IMPLEMENTED = {
    "message/stream": (
        "This agent does not support streaming yet - its card advertises "
        "capabilities.streaming = false. Use message/send."
    ),
    "tasks/cancel": (
        "This agent runs every task to completion within the request that "
        "created it, so there is no window in which a task can be cancelled."
    ),
    "tasks/resubscribe": (
        "This agent does not support streaming yet, so there is no stream to "
        "resubscribe to. Use tasks/get to read a task by id."
    ),
    "tasks/pushNotificationConfig/set": (
        "This agent does not support push notifications - its card advertises "
        "capabilities.pushNotifications = false."
    ),
    "tasks/pushNotificationConfig/get": (
        "This agent does not support push notifications - its card advertises "
        "capabilities.pushNotifications = false."
    ),
    "agent/getAuthenticatedExtendedCard": (
        "This agent has no authenticated extended card. The public card at "
        "/.well-known/agent-card.json is the whole description."
    ),
}


class A2AServer:
    """
    Turns one JSON-RPC payload into one JSON-RPC response.

    Stateless per call, in the same sense every agent here is: the only
    thing that survives between calls is the task store, and it is passed in
    rather than owned.
    """

    def __init__(self, executor, task_store) -> None:
        self._executor = executor
        self._tasks = task_store

    def handle(self, payload) -> dict:
        """
        Dispatch `payload` and return the response body. Never raises.

        Never, deliberately. Anything that escaped would become a 500 with no
        `id` and no `error` member, which is the one response shape a
        JSON-RPC client has no handling for - so an unexpected bug here would
        surface at the caller as a transport failure rather than as the
        internal error it is.
        """
        request_id = payload.get("id") if isinstance(payload, dict) else None

        try:
            request = JSONRPCRequest.model_validate(payload)
        except ValidationError as exc:
            return error_response(
                request_id,
                JSONRPCError(
                    INVALID_REQUEST,
                    "Not a valid JSON-RPC 2.0 request.",
                    data=exc.errors(include_url=False),
                ),
            )

        try:
            return success_response(request.id, self._dispatch(request))
        except JSONRPCError as exc:
            return error_response(request.id, exc)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.exception("Unhandled error in A2A method %s", request.method)
            return error_response(
                request.id,
                JSONRPCError(
                    # -32603, not a task in state `failed`: a failed task
                    # means the agent ran and could not finish, and reporting
                    # a bug in the dispatch that way would tell the caller
                    # something untrue about its request.
                    -32603,
                    f"Internal error handling {request.method}.",
                    data=str(exc),
                ),
            )

    def _dispatch(self, request: JSONRPCRequest):
        if request.method == "message/send":
            return self._message_send(request.params or {})
        if request.method == "tasks/get":
            return self._tasks_get(request.params or {})

        advice = _NOT_IMPLEMENTED.get(request.method)
        if advice:
            raise JSONRPCError(UNSUPPORTED_OPERATION, advice)

        raise JSONRPCError(
            METHOD_NOT_FOUND, f'This agent has no method "{request.method}".'
        )

    # ---- Methods -------------------------------------------------------------

    def _message_send(self, raw_params: dict) -> dict:
        """
        Run the agent on the caller's message and return the finished Task.

        Returns a Task rather than a Message - both are legal replies to
        `message/send`. A Task, because this run produces more than prose:
        the evidence and the trace are artifacts, and a bare Message has
        nowhere to put them.
        """
        params = _validate(MessageSendParams, raw_params)
        question = params.message.text()

        if not question:
            # Two different failures wearing the same symptom, told apart
            # because they need different fixes at the caller. Parts that
            # exist but carry no text is a content-type problem - it sent a
            # file to a text-only agent. No usable parts at all is a
            # malformed call.
            if params.message.parts:
                raise JSONRPCError(
                    CONTENT_TYPE_NOT_SUPPORTED,
                    "This agent reads text only. Send the question as a text "
                    "part; it cannot accept files or structured data as input.",
                )
            raise JSONRPCError(
                INVALID_PARAMS,
                "The message carried no question. Send at least one text part.",
            )

        task = self._executor.execute(
            question=question, context_id=params.message.contextId
        )

        # The caller's own message, echoed into the task's history, with the
        # ids the server assigned filled in. This is what makes `tasks/get`
        # worth calling for anything other than the artifacts: a task read
        # back later says what it was asked, not only what it answered.
        task.history = [
            params.message.model_copy(
                update={"taskId": task.id, "contextId": task.contextId}
            )
        ]

        self._tasks.save(task)
        return _serialise(task, _history_limit(params.configuration))

    def _tasks_get(self, raw_params: dict) -> dict:
        """Read a task back by id."""
        params = _validate(TaskQueryParams, raw_params)

        task = self._tasks.get(params.id)
        if task is None:
            # A single error for two causes - never existed, or evicted (see
            # a2a/task_store.py). They are indistinguishable to this process
            # and the caller's move is the same for both: re-send the work.
            raise JSONRPCError(
                TASK_NOT_FOUND,
                f'No task with id "{params.id}". It may have expired - this '
                "agent keeps recent tasks only.",
            )

        return _serialise(task, params.historyLength)


# ---- Helpers -----------------------------------------------------------------


def _validate(model, raw_params: dict):
    """
    Parse `raw_params` into `model`, or raise the JSON-RPC error for it.

    One place rather than a try/except per method, so every method reports a
    bad shape identically - including the field-level detail, which is the
    part that makes a -32602 actionable rather than just discouraging.
    """
    try:
        return model.model_validate(raw_params)
    except ValidationError as exc:
        raise JSONRPCError(
            INVALID_PARAMS,
            "The params for this method are not valid.",
            data=exc.errors(include_url=False),
        ) from exc


def _history_limit(configuration) -> int | None:
    return configuration.historyLength if configuration else None


def _serialise(task: Task, history_length: int | None) -> dict:
    """
    Project a Task onto the wire, honouring the caller's history limit.

    `exclude_none` so optional fields the protocol treats as absent are
    absent, rather than present-and-null. Not cosmetic: a client that
    distinguishes "no status message" from "a null status message" is within
    its rights, and sending nulls makes this agent's replies noisier than the
    spec's own examples for no benefit.

    The limit is applied to a copy. Truncating `task.history` in place would
    permanently shorten the stored task, so a caller that asked for the last
    message once would silently get a lobotomised task on every later
    `tasks/get` - a cache mutated by a read.
    """
    if history_length is not None and history_length >= 0:
        task = task.model_copy(
            update={"history": task.history[-history_length:] if history_length else []}
        )
    return task.model_dump(exclude_none=True)
