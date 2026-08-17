"""
Tests for the Agent Card and where it is served.

The card is an advertisement other agents act on without asking, so the
things worth asserting are the ones that mislead rather than break: a
capability claimed but not implemented, a URL nobody else can resolve, a
path a generic client does not look at.
"""
from fastapi.testclient import TestClient

from a2a.card import build_agent_card
from app.api.a2a import AGENT_CARD_PATH
from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_card_is_served_at_the_well_known_path():
    # The path is fixed by the protocol. Serving it under /api/v1 would make
    # it undiscoverable by every client that was not written for this app.
    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    assert AGENT_CARD_PATH == "/.well-known/agent-card.json"


def test_card_advertises_the_rpc_url_a_third_party_can_reach():
    card = client.get(AGENT_CARD_PATH).json()

    # Built from the configured public base URL, not from the request's own
    # host - behind a proxy those differ, and the card is read by callers
    # that never saw this request.
    assert card["url"] == (
        settings.a2a_public_base_url.rstrip("/") + settings.a2a_rpc_path
    )
    assert card["preferredTransport"] == "JSONRPC"


def test_card_does_not_claim_capabilities_that_are_not_implemented():
    card = client.get(AGENT_CARD_PATH).json()

    # The point of the card is that a client can trust it instead of probing.
    # These flip in later phases, together with the methods behind them.
    assert card["capabilities"]["streaming"] is False
    assert card["capabilities"]["pushNotifications"] is False


def test_card_describes_the_research_skill_with_examples():
    card = client.get(AGENT_CARD_PATH).json()

    skills = {skill["id"]: skill for skill in card["skills"]}
    assert "research_documents" in skills

    skill = skills["research_documents"]
    # The examples are what another agent's routing decision reads, so an
    # empty list is a real regression rather than a cosmetic one.
    assert skill["examples"]
    assert "text/plain" in skill["inputModes"]


def test_card_omits_unset_optional_fields_rather_than_sending_null():
    card = client.get(AGENT_CARD_PATH).json()

    # No security scheme is declared in phase 1, and absent says that more
    # clearly than null. It also keeps the card honest: nothing here checks
    # credentials, so nothing here should mention them.
    assert "securitySchemes" not in card
    assert "security" not in card


def test_provider_is_omitted_entirely_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "a2a_provider_organization", "")
    assert build_agent_card().provider is None

    monkeypatch.setattr(settings, "a2a_provider_organization", "Example Org")
    monkeypatch.setattr(settings, "a2a_provider_url", "https://example.org")
    provider = build_agent_card().provider
    assert provider is not None and provider.organization == "Example Org"
