"""Outils de base pour le pipeline de detection x-ray."""

from .config import ProjectConfig, load_config
from .pipeline import prepare_workspace

__all__ = ["ProjectConfig", "load_config", "prepare_workspace"]
__version__ = "0.1.0"
