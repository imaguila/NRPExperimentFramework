"""
Domain plugin package.

Plugin implementations must be imported explicitly from their modules
to avoid circular imports during initialization of the domain core.

Example:
    from src.domain.plugins.nrp import NRPPlugin
"""
from .nrp import NRPPlugin

