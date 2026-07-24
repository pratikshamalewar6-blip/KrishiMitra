"""
KrishiMitra - Knowledge Base Custom Exceptions

Defines custom exceptions for database loading, validation, and querying.

Author:
    Antigravity AI
"""

from __future__ import annotations


class KnowledgeBaseError(Exception):
    """Base exception for all Knowledge Base errors."""
    pass


class DatabaseNotFoundError(KnowledgeBaseError, FileNotFoundError):
    """Raised when the disease database JSON file cannot be found."""
    pass


class ValidationError(KnowledgeBaseError):
    """Raised when the database fails schema or business logic validation."""
    pass


class DiseaseNotFoundError(KnowledgeBaseError, KeyError):
    """Raised when a queried disease class is not found in the database."""
    pass


class CropNotFoundError(KnowledgeBaseError, KeyError):
    """Raised when a queried crop is not found in the database."""
    pass
