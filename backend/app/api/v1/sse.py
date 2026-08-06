"""
Server-Sent Event framing, shared by the streaming endpoints.

One helper and one set of headers, in one place, because /chat/stream and
/agent/ask/stream have to agree on the wire format exactly - the frontend
parses both with the same reader.
"""
import json

# Headers every streaming response needs.
#
# Without them a proxy (or the browser itself) is free to hold the response
# until it completes and hand it over in one piece - which silently
# reintroduces exactly the latency these endpoints exist to remove, while
# still looking like a working stream in tests.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_event(event: dict) -> str:
    """
    Frame one payload as a Server-Sent Event.

    The trailing blank line is the message terminator - without it the
    browser buffers the frame forever waiting for the rest.

    `json.dumps` matters for more than convenience: SSE is a line-delimited
    format, so a raw newline inside a model's token would split one event
    into two malformed ones. Escaping it to \\n is what makes arbitrary
    generated text safe to put on this wire.
    """
    return f"data: {json.dumps(event)}\n\n"
