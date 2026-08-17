"""
Research agent endpoint.

Pure HTTP layer: parses the request, runs the multi-hop loop, maps domain
errors to HTTP status codes, and returns the answer together with the trace
that produced it.

Distinct from POST /agent/ask, which routes to exactly one tool and returns
one answer. This one may search the corpus several times, and the questions
it exists for - comparisons, multi-part questions, anything whose second
search depends on the first search's result - are exactly the ones the
single-hop router cannot answer at any retrieval quality.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.agent_trace import to_step_models
from app.schemas.research import (
    ResearchRequest,
    ResearchResponse,
)
from app.services.research_service import (
    ResearchChatService,
    get_research_chat_service,
)

router = APIRouter(tags=["research"])


@router.post("/research/ask", response_model=ResearchResponse)
def research(
    request: ResearchRequest,
    research_chat_service: ResearchChatService = Depends(get_research_chat_service),
) -> ResearchResponse:
    """
    Research a question over the ingested documents across several hops.

    Slower than POST /rag/chat by design - it makes one LLM call per step
    plus one per search - so it is a separate endpoint rather than a flag on
    that one. A question answerable from a single retrieval should still go
    to /rag/chat and get its answer in one round trip.
    """
    try:
        result = research_chat_service.research(
            question=request.question, conversation_id=request.conversation_id
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return ResearchResponse(
        question=request.question,
        answer=result.answer,
        conversation_id=result.conversation_id,
        steps=to_step_models(result.steps),
        sources=result.sources,
        stopped_because=result.stopped_because,
    )
