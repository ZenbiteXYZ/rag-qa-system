"""FastAPI dependencies: the objects the lifespan stored on `app.state`."""

from typing import Annotated

from fastapi import Depends, Request

from ragqa.api.resources import Resources
from ragqa.settings import Settings


def _resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


ResourcesDep = Annotated[Resources, Depends(_resources)]
SettingsDep = Annotated[Settings, Depends(_settings)]
