"""
Tests for the A2A JSON-RPC surface.

Mostly against A2AServer directly rather than through a TestClient, because
a2a/server.py deliberately has no web framework in it: `handle` takes a
parsed body and returns a response body, so the protocol can be tested
without a route table in the way. The HTTP shim gets its own small section
at the bottom for the two things only it can be wrong about - the raw-body
parse and the dependency wiring.

The research agent is stubbed throughout. What is under test is the wire
contract, not the reasoning - that is tests/test_research_agent.py.
"""
import uuid

from fastapi.testclient import TestClient

from a2a.executor import ResearchExecutor
from a2a.jsonrpc import (
    CONTENT_TYPE_NOT_SUPPORTED,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    TASK_NOT_FOUND,
    UNSUPPORTED_OPERATION,
)
from a2a.server import A2AServer
from a2a.task_store import InMemoryTaskStore
from agents.agent_loop import AgentStep
from app.core.config import settings
from app.main import app
from app.schemas.rag import RAGChatSource
from app.services.research_service import (
    ResearchChatResult,
    get_research_chat_service,
)


class StubResearchService:
    """Stands in for the memory-aware ResearchChatService the executor wraps."""

    def __init__(
        self,
        answer: str = "The refund window is 30 days [E1].",
        sources=None,
        steps=None,
        stopped_because: str = "finished",
        raises: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.answer = answer
        self.sources = sources or []
        self.steps = steps or []
        self.stopped_because = stopped_because
        self.raises = raises

    def research(self, question: str, conversation_id: str | None = None):
        self.calls.append((question, conversation_id))
        if self.raises:
            raise self.raises
        return ResearchChatResult(
            conversation_id=conversation_id or "generated-conversation",
            answer=self.answer,
            steps=self.steps,
            sources=self.sources,
            stopped_because=self.stopped_because,
        )


def _source(chunk_id: str, filename: str = "handbook.pdf", page: int = 3):
    return RAGChatSource(
        filename=filename, page=page, chunk_id=chunk_id, section="4. Refunds"
    )


def _server(stub: StubResearchService | None = None):
    stub = stub or StubResearchService()
    return stub, A2AServer(
        executor=ResearchExecutor(stub), task_store=InMemoryTaskStore()
    )


def _send(server, text: str = "What is the refund window?", **message_extra) -> dict:
    message = {
        "kind": "message",
        "role": "user",
        "messageId": str(uuid.uuid4()),
        "parts": [{"kind": "text", "text": text}],
    }
    message.update(message_extra)
    return server.handle(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "message/send",
            "params": {"message": message},
        }
    )


def _artifacts(task: dict) -> dict:
    return {artifact["name"]: artifact for artifact in task["artifacts"]}


# ---- message/send ------------------------------------------------------------


def test_send_returns_a_completed_task_carrying_the_answer():
    stub, server = _server()

    response = _send(server)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "req-1"
    task = response["result"]
    assert task["kind"] == "task"
    assert task["status"]["state"] == "completed"

    answer = _artifacts(task)["answer"]
    assert answer["parts"][0]["text"] == stub.answer


def test_the_question_reaches_the_agent_as_the_message_text():
    stub, server = _server()

    _send(server, text="How does the refund policy differ from returns?")

    assert stub.calls == [("How does the refund policy differ from returns?", None)]


def test_context_id_maps_onto_the_conversation_and_is_echoed_back():
    stub, server = _server()

    task = _send(server, contextId="conv-42")["result"]

    # A follow-up from the same caller must land in the same conversation,
    # which is what makes "and what about the other one?" resolvable.
    assert stub.calls == [("What is the refund window?", "conv-42")]
    assert task["contextId"] == "conv-42"


def test_a_first_call_gets_the_conversation_id_the_service_generated():
    _, server = _server()

    task = _send(server)["result"]

    # The server's id is authoritative: a caller that wants continuity must
    # send back what it received, not what it sent (which was nothing).
    assert task["contextId"] == "generated-conversation"


def test_evidence_artifact_pairs_each_label_with_the_chunk_it_stands_for():
    stub, server = _server(
        StubResearchService(sources=[_source("chunk-a"), _source("chunk-b")])
    )

    task = _send(server)["result"]
    evidence = _artifacts(task)["evidence"]["parts"][0]["data"]["evidence"]

    # The whole reason this artifact exists: over HTTP the caller has its own
    # ledger starting at E1, so it needs the label->chunk mapping explicitly
    # to re-label without colliding. Position alone is an invariant it cannot
    # verify.
    assert [record["label"] for record in evidence] == ["E1", "E2"]
    assert [record["chunkId"] for record in evidence] == ["chunk-a", "chunk-b"]
    assert evidence[0]["filename"] == "handbook.pdf"


def test_evidence_artifact_is_absent_rather_than_empty_when_nothing_was_retrieved():
    _, server = _server(StubResearchService(sources=[]))

    task = _send(server)["result"]

    # "There were no passages" and "here is an empty list of passages" read
    # differently to a caller merging citations.
    assert "evidence" not in _artifacts(task)


def test_trace_artifact_carries_the_steps_including_nested_ones():
    child = AgentStep(
        thought="search first",
        action_json="{}",
        tool="search",
        tool_input={"query": "refund"},
        observation="[E1] ...",
    )
    parent = AgentStep(
        thought="delegate",
        action_json="{}",
        tool="research_documents",
        tool_input={"question": "refund window"},
        observation="30 days",
        children=[child],
    )
    _, server = _server(StubResearchService(steps=[parent]))

    task = _send(server)["result"]
    steps = _artifacts(task)["trace"]["parts"][0]["data"]["steps"]

    assert len(steps) == 1
    assert steps[0]["tool"] == "research_documents"
    # Same projection the REST endpoints use, so a client has one step shape
    # however it reached the agent.
    assert steps[0]["toolInput"] == '{"question": "refund window"}'
    assert steps[0]["children"][0]["tool"] == "search"


def test_a_truncated_run_still_completes_but_says_so():
    _, server = _server(StubResearchService(stopped_because="step_limit"))

    task = _send(server)["result"]

    # `completed` because there is a real answer - the loop synthesises one
    # rather than reporting the limit - but the caller has to be able to tell
    # a truncated run from a finished one.
    assert task["status"]["state"] == "completed"
    assert task["metadata"]["stoppedBecause"] == "step_limit"


def test_a_provider_failure_is_a_failed_task_not_a_transport_error():
    _, server = _server(
        StubResearchService(raises=RuntimeError("provider unreachable"))
    )

    response = _send(server)

    # The caller is another agent, and a failed task is something it can act
    # on - answer without this specialist, or tell the user. A JSON-RPC error
    # is something it can only propagate.
    assert "error" not in response
    task = response["result"]
    assert task["status"]["state"] == "failed"
    assert "provider unreachable" in task["status"]["message"]["parts"][0]["text"]


def test_the_callers_message_is_kept_in_the_task_history():
    _, server = _server()

    task = _send(server)["result"]

    assert len(task["history"]) == 1
    assert task["history"][0]["role"] == "user"
    # Stamped with the ids the server assigned, so a task read back later is
    # self-describing.
    assert task["history"][0]["taskId"] == task["id"]
    assert task["history"][0]["contextId"] == task["contextId"]


def test_a_message_with_no_text_part_is_invalid_params():
    _, server = _server()

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": "m1",
                    "parts": [],
                }
            },
        }
    )

    assert response["error"]["code"] == INVALID_PARAMS


def test_a_message_carrying_only_a_file_is_a_content_type_error():
    _, server = _server()

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": "m1",
                    "parts": [
                        {
                            "kind": "file",
                            "file": {"mimeType": "image/png", "bytes": "aGk="},
                        }
                    ],
                }
            },
        }
    )

    # Distinct from "you sent nothing": this caller sent something real that
    # a text-only agent cannot read, and needs a different fix.
    assert response["error"]["code"] == CONTENT_TYPE_NOT_SUPPORTED


def test_malformed_params_report_the_offending_field():
    _, server = _server()

    response = server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {}}
    )

    assert response["error"]["code"] == INVALID_PARAMS
    # The detail is what makes -32602 actionable rather than just
    # discouraging.
    assert response["error"]["data"]


# ---- tasks/get ---------------------------------------------------------------


def test_a_task_can_be_read_back_by_id():
    _, server = _server()
    sent = _send(server)["result"]

    response = server.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": sent["id"]}}
    )

    fetched = response["result"]
    assert fetched["id"] == sent["id"]
    assert _artifacts(fetched)["answer"] == _artifacts(sent)["answer"]


def test_an_unknown_task_is_task_not_found():
    _, server = _server()

    response = server.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": "nope"}}
    )

    assert response["error"]["code"] == TASK_NOT_FOUND


def test_history_length_truncates_the_reply_without_shortening_the_stored_task():
    _, server = _server()
    sent = _send(server)["result"]

    trimmed = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/get",
            "params": {"id": sent["id"], "historyLength": 0},
        }
    )["result"]
    assert trimmed["history"] == []

    # A read must not mutate the cache. Truncating in place would silently
    # lobotomise the task for every later fetch.
    again = server.handle(
        {"jsonrpc": "2.0", "id": 3, "method": "tasks/get", "params": {"id": sent["id"]}}
    )["result"]
    assert len(again["history"]) == 1


# ---- Dispatch ----------------------------------------------------------------


def test_streaming_is_refused_with_advice_rather_than_method_not_found():
    _, server = _server()

    response = server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "message/stream", "params": {}}
    )

    # "I know that method and cannot do it yet" leads a client somewhere
    # different from "I have never heard of it".
    assert response["error"]["code"] == UNSUPPORTED_OPERATION
    assert "message/send" in response["error"]["message"]


def test_an_unknown_method_is_method_not_found():
    _, server = _server()

    response = server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "agent/doSomething", "params": {}}
    )

    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_a_bad_envelope_is_invalid_request():
    _, server = _server()

    response = server.handle({"method": "message/send", "jsonrpc": "1.0"})

    assert response["error"]["code"] == INVALID_REQUEST


def test_the_request_id_is_echoed_on_success_and_on_failure():
    _, server = _server()

    assert _send(server)["id"] == "req-1"
    assert (
        server.handle({"jsonrpc": "2.0", "id": 99, "method": "nope"})["id"] == 99
    )


# ---- The HTTP shim -----------------------------------------------------------


def _post(stub, payload=None, content=None):
    app.dependency_overrides[get_research_chat_service] = lambda: stub
    try:
        with TestClient(app) as client:
            if content is not None:
                return client.post(settings.a2a_rpc_path, content=content)
            return client.post(settings.a2a_rpc_path, json=payload)
    finally:
        app.dependency_overrides.pop(get_research_chat_service, None)


def test_the_endpoint_answers_a_real_call_end_to_end():
    stub = StubResearchService()

    response = _post(
        stub,
        {
            "jsonrpc": "2.0",
            "id": "http-1",
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "messageId": "m1",
                    "parts": [{"kind": "text", "text": "What is the refund window?"}],
                }
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["status"]["state"] == "completed"
    assert stub.calls == [("What is the refund window?", None)]


def test_a_body_that_is_not_json_is_a_parse_error_at_200():
    response = _post(StubResearchService(), content=b"{not json")

    # A JSON-RPC error is a successful HTTP response carrying an `error`
    # member. Answering 400 would leave a client with no defined body shape
    # to read.
    assert response.status_code == 200
    body = response.json()
    assert body["error"]["code"] == PARSE_ERROR
    # Nothing to echo when the envelope itself did not parse.
    assert body["id"] is None


def test_a_protocol_error_is_still_http_200():
    response = _post(
        StubResearchService(),
        {"jsonrpc": "2.0", "id": 1, "method": "tasks/cancel", "params": {"id": "x"}},
    )

    assert response.status_code == 200
    assert response.json()["error"]["code"] == UNSUPPORTED_OPERATION
