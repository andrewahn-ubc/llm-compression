"""Probe-guided transformer depth compression experiments."""

from .config import ExperimentConfig, load_config
from .layout import Layout, Span

__all__ = ["ExperimentConfig", "Layout", "Span", "load_config"]
__version__ = "0.1.0"
