"""
Aggregates all v1 endpoint routers into a single APIRouter.

New feature routers (chat, vision, documents, agents, ...) should be
registered here as they're built in later tasks.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    agent,
    audio,
    chat,
    documents,
    health,
    rag,
    research,
    stream,
    vision,
    vision_agent,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(vision.router)
api_router.include_router(documents.router)
api_router.include_router(rag.router)
api_router.include_router(agent.router)
api_router.include_router(research.router)
api_router.include_router(vision_agent.router)
api_router.include_router(audio.router)
api_router.include_router(stream.router)
