"""Plugin files: manifests parse, skills have the frontmatter Claude Code needs, and the MCP entry matches
the installed console script."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "plugins" / "claude-code"
PLUGIN = ROOT / "spinescope"
SKILLS = ("spine-read", "spine-findings", "spine-viewer", "spine-inspect")


def test_plugin_manifest_and_marketplace_parse():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "spinescope" and re.fullmatch(r"[a-z0-9-]+", manifest["name"])
    assert manifest["version"] and manifest["description"]
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert [p["name"] for p in market["plugins"]] == ["spinescope"]
    assert (ROOT / market["plugins"][0]["source"]).resolve() == PLUGIN.resolve()


@pytest.mark.parametrize("skill", SKILLS)
def test_skill_frontmatter(skill: str):
    text = (PLUGIN / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = text.split("---\n", 2)[1]
    fields = dict(line.split(": ", 1) for line in front.strip().splitlines() if ": " in line)
    assert fields["name"] == skill
    assert len(fields["description"]) > 40
    # every skill must state the research-only and no-upload rules
    assert "Research and education only" in text
    assert "upload" in text.lower()


def test_spine_read_states_the_concurrency_cap_and_disclaimer():
    text = (PLUGIN / "skills" / "spine-read" / "SKILL.md").read_text(encoding="utf-8")
    assert "at most **4 at a time**" in text
    assert "not a diagnosis" in text
    assert "Never upload patient data anywhere" in text


def test_findings_skill_states_validation_status_and_wording_rules():
    text = (PLUGIN / "skills" / "spine-findings" / "SKILL.md").read_text(encoding="utf-8")
    for word in ("validated", "partly_validated", "experimental", "not assessable", "consider correlation"):
        assert word in text
    assert "only lower trust" in text
    assert "not a diagnosis" in text


def test_mcp_registration_matches_console_script():
    cfg = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))
    server = cfg["mcpServers"]["spinescope"]
    assert server["command"] == "spinescope-mcp"
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'spinescope-mcp = "spinescope.ai.mcp_server:main"' in pyproject
