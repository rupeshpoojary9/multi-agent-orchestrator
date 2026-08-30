"""Foundational agents under orchestration.

The Web Agent drives a legacy web UI; the API Agent drives a typed enterprise
REST surface. Both expose the same `Agent` contract (a `spec` + `run`) so the
orchestrator routes to either without caring how the work gets done — one
integration contract over heterogeneous systems.
"""

from .api_agent import APIAgent
from .base import Agent
from .web_agent import WebAgent

__all__ = ["Agent", "APIAgent", "WebAgent"]
