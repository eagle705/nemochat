"""Explicit opt-in Lens/NVTX instrumentation; importing alone has no side effects."""
from .bridge import Bridge

__all__ = ["Bridge"]
