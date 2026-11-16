# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Remote policy servers a model can run on, one per backend."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from schemas.hardware import InferenceBackend


class RemoteServer(BaseModel):
    """Where a remote policy server listens, as seen from the Studio backend."""

    host: str = Field(..., min_length=1, max_length=253, description="Host name or IP address, e.g. 172.18.0.1")
    port: int = Field(..., ge=1, le=65535, description="TCP port, e.g. 18765")

    @field_validator("host")
    @classmethod
    def _plain_host(cls, host: str) -> str:
        host = host.strip()
        if not host or "://" in host or "/" in host or " " in host:
            raise ValueError("Enter a host name or IP address only, without http:// or a path")
        return host


class RemoteInferenceConfig(BaseModel):
    """The remote server set for each backend; a backend without one is left out."""

    servers: dict[InferenceBackend, RemoteServer] = Field(default_factory=dict)


class RemoteServerCheckRequest(RemoteServer):
    """A server to try; with ``model_id`` and ``backend``, also whether it runs that model's export."""

    model_id: UUID | None = None
    backend: InferenceBackend | None = None


class RemoteServerCheck(BaseModel):
    """What a remote policy server answered to ``GET /info``."""

    ok: bool
    url: str
    latency_ms: float | None = None
    server_host: str | None = Field(None, description="The server's own host name (its container id)")
    backend: str | None = None
    device: str | None = None
    fingerprint: str | None = None
    same_model: bool | None = Field(None, description="Whether it runs the model's export; null when not asked")
    error: str | None = None
