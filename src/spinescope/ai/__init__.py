"""AI integration: reading briefs and reader scopes (``briefs``) and the MCP server (``mcp_server``).

Research and education only. Nothing here sends data anywhere; the MCP server runs locally over stdio.
"""

from .briefs import (
    CALIBRATION_GUIDANCE,
    STRUCTURE_FAMILIES,
    ReadingTask,
    brief,
    geometry_facts,
    reading_tasks,
    series_table,
    verification_errors,
)

__all__ = [
    "CALIBRATION_GUIDANCE",
    "STRUCTURE_FAMILIES",
    "ReadingTask",
    "brief",
    "geometry_facts",
    "reading_tasks",
    "series_table",
    "verification_errors",
]
