"""
KrishiMitra - Knowledge Base Module

Exposes key classes, validators, custom exceptions, and helpers.

Author:
    Antigravity AI
"""

from __future__ import annotations

from knowledge_base.exceptions import (
    KnowledgeBaseError,
    DatabaseNotFoundError,
    ValidationError,
    DiseaseNotFoundError,
    CropNotFoundError,
)
from knowledge_base.validators import DatabaseValidator
from knowledge_base.knowledge_base_manager import KnowledgeBaseManager, build_prompt_context

__all__ = [
    "KnowledgeBaseError",
    "DatabaseNotFoundError",
    "ValidationError",
    "DiseaseNotFoundError",
    "CropNotFoundError",
    "DatabaseValidator",
    "KnowledgeBaseManager",
    "build_prompt_context",
]
