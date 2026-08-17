"""
The JSON-RPC 2.0 envelope A2A rides on, and the errors it can carry.

Separate from types.py because none of it is about agents. A2A picked
JSON-RPC as its primary transport; the envelope, the id-echoing rule and the
error-code registry are the same whatever the methods happen to be, and
keeping them apart means the A2A-specific file is only about A2A.

THE ONE RULE THAT IS EASY TO GET WRONG

A JSON-RPC error is a *successful* HTTP response carrying an `error` member.
It is not a 4xx. A peer that returns 400 for "no such method" breaks clients
that read the body, because there is no body shape defined for a 400 - and
the caller then cannot tell "your method name was wrong" apart from "a proxy
in the middle rejected you". Everything below therefore leaves the HTTP
status at 200 and puts the failure in the payload.

The exception is a request that could not be parsed as JSON at all, where
there is no `id` to echo back. The spec's answer is `"id": null`, which is
what `error_response` produces when given None.
"""
from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Standard JSON-RPC 2.0 codes -------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# --- A2A's own extensions, in the -32000..-32099 implementation-defined band -
#
# Named rather than inlined at the raise site: a client branches on the
# number, so a typo is a silently different error, and these are the numbers
# the spec assigns.
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005
INVALID_AGENT_RESPONSE = -32006


class JSONRPCError(Exception):
    """
    A failure that should reach the caller as an `error` member.

    An exception rather than a return value because it is raised deep in
    method handling - inside a parameter check, inside the executor - and
    threading a result union back out through every layer would put a
    branch at each one for something only the outermost frame acts on.

    `data` is optional and free-form. It is where a detail useful to a
    *developer* goes; `message` is the one-line summary and is the only
    part a client is expected to show.
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_payload(self) -> dict:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload


class JSONRPCRequest(BaseModel):
    """
    One incoming call.

    `id` is optional in JSON-RPC - a request without one is a *notification*,
    which the server must not answer. A2A defines no notifications, so an
    absent id is treated as an ordinary null id and answered anyway: a client
    that omitted it by accident gets told what went wrong instead of silence,
    and one that omitted it on purpose is not something this protocol
    produces.

    `params` is typed loosely here and validated per-method. The alternative -
    a discriminated union over `method` - would put every method's shape in
    the envelope model, which is exactly the coupling this file exists to
    avoid.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] | None = Field(default=None)


def success_response(request_id: str | int | None, result: Any) -> dict:
    """Wrap `result` as a JSON-RPC success, echoing the caller's id."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(request_id: str | int | None, error: JSONRPCError) -> dict:
    """Wrap `error` as a JSON-RPC failure, echoing the caller's id."""
    return {"jsonrpc": "2.0", "id": request_id, "error": error.to_payload()}
