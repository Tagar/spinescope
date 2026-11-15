"""Structured findings for spine MRI reads (human or AI): JSON Schema, validation, multi-reader
consensus and rendering.

Typical flow: a reader fills ``<reader>.json`` following :func:`reader_instructions`, each file is checked
with :func:`validate`, :func:`merge` builds the consensus, and :func:`render_markdown` /
:func:`render_html` produce the reading report.
"""

from .instructions import reader_instructions
from .merge import Consensus, merge
from .render import render_html, render_markdown
from .schema import SCHEMA_VERSION, field_path, load_schema
from .validate import ValidationIssue, ValidationResult, validate

__all__ = [
    "SCHEMA_VERSION",
    "Consensus",
    "ValidationIssue",
    "ValidationResult",
    "field_path",
    "load_schema",
    "merge",
    "reader_instructions",
    "render_html",
    "render_markdown",
    "validate",
]
