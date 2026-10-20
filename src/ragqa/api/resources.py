"""Long-lived clients owned by the API process, created in the lifespan."""

import asyncio
from dataclasses import dataclass

import aio_pika
import httpx
from aio_pika.abc import AbstractRobustConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ragqa.settings import Settings


class AmqpConnector:
    """Opens the RabbitMQ connection on first use, so the API can start before the broker."""

    def __init__(self, url: str, connection_name: str) -> None:
        self._url = url
        self._name = connection_name
        self._connection: AbstractRobustConnection | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> AbstractRobustConnection:
        async with self._lock:
            if self._connection is None or self._connection.is_closed:
                self._connection = await aio_pika.connect_robust(
                    self._url, timeout=5, client_properties={"connection_name": self._name}
                )
            return self._connection

    async def close(self) -> None:
        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()


@dataclass
class Resources:
    http: httpx.AsyncClient
    # Replaced by the storage/postgres session factory in slice 2.
    engine: AsyncEngine
    amqp: AmqpConnector

    @classmethod
    def create(cls, settings: Settings) -> "Resources":
        return cls(
            http=httpx.AsyncClient(timeout=httpx.Timeout(5.0)),
            engine=create_async_engine(settings.postgres_dsn, pool_pre_ping=True),
            amqp=AmqpConnector(settings.rabbitmq_url, settings.service_name),
        )

    async def close(self) -> None:
        await self.amqp.close()
        await self.engine.dispose()
        await self.http.aclose()
