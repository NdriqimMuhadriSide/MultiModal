"""
FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    # A2A is mounted at the site root, with no version prefix, because
    # /.well-known/agent-card.json is a fixed path in the protocol - a
    # generic A2A client looks there and nowhere else. See app/api/a2a.py.
    #
    # Registered conditionally rather than gated inside the handlers: when
    # A2A_ENABLED is false there is no unauthenticated JSON-RPC endpoint in
    # the route table at all, which is a stronger statement than one that
    # returns an error.
    if settings.a2a_enabled:
        from app.api.a2a import router as a2a_router

        app.include_router(a2a_router)

    return app


app = create_app()
