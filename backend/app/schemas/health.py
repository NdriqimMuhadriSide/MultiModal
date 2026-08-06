"""
Pydantic response models for the health check endpoint.
"""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
