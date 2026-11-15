"""Viewer review panel: the status banner and the applicability line, run under node on the phantom's anchors.

The banner must follow the same policy as spinescope.anatomy.summary: confirmed (automatic or reviewed), probable,
unconfirmed. Skipped without node.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spinescope.anatomy import label_spine
from spinescope.config import DEFAULT
from spinescope.viewer.build import _asset
from spinescope.viewer.export import label_policy

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

RUNNER = r"""
const fs = require('fs');
const L = require(process.argv[2]);
const spec = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = {};
for (const [name, args] of Object.entries(spec.probes)) {
  const [fn, a, policy] = args;
  if (fn === 'banner') out[name] = L.banner(a, policy);
  else if (fn === 'applicabilityLine') out[name] = L.applicabilityLine(a);
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def _run(probes: dict[str, list[Any]], tmp: Path) -> dict[str, Any]:
    (tmp / "labels.js").write_text(_asset("labels.js"), encoding="utf-8")
    (tmp / "run.js").write_text(RUNNER, encoding="utf-8")
    (tmp / "spec.json").write_text(json.dumps({"probes": probes}), encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(tmp / "run.js"), str(tmp / "labels.js"), str(tmp / "spec.json"), str(tmp / "out.json")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads((tmp / "out.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def anchors(phantom_study) -> dict[str, Any]:
    return label_spine(phantom_study, backend="classical").to_dict()


def _with(doc: dict[str, Any], **numbering: Any) -> dict[str, Any]:
    d = copy.deepcopy(doc)
    d["numbering"].update(numbering)
    return d


def test_banner_levels_for_each_status(anchors, tmp_path):
    pol = label_policy(DEFAULT)
    probes = {
        "probable": ["banner", _with(anchors, status="probable", confidence=0.75), pol],
        "confirmed_auto": [
            "banner",
            _with(anchors, status="confirmed", confidence=0.95, corrections=[]),
            pol,
        ],
        "confirmed_reviewed": [
            "banner",
            _with(
                anchors,
                status="confirmed",
                confidence=0.95,
                corrections=[{"action": "review", "source": "human", "who": "x", "why": "", "detail": {}}],
            ),
            pol,
        ],
        "unconfirmed": ["banner", _with(anchors, status="unconfirmed", confidence=0.3), pol],
    }
    res = _run(probes, tmp_path)
    assert res["probable"]["level"] == "probable"
    assert "to be confirmed" in res["probable"]["text"]
    assert res["confirmed_auto"]["level"] == "confirmed"
    assert "automatic" in res["confirmed_auto"]["text"]
    assert res["confirmed_reviewed"]["text"] == "Levels confirmed by a reviewer."
    assert res["unconfirmed"]["level"] == "unconfirmed"
    assert "positions only" in res["unconfirmed"]["text"]


def test_applicability_line_names_verdict_sacrum_and_experimental_conus(anchors, tmp_path):
    res = _run({"line": ["applicabilityLine", anchors, None]}, tmp_path)
    line = res["line"]
    assert line.startswith("Applicability: lumbar")
    assert "levels labelled" in line
    assert "Sacrum NOT in view" in line  # the phantom's field of view does not hold the sacrum
    assert "(experimental)" in line


def test_applicability_line_without_record(tmp_path):
    res = _run({"none": ["applicabilityLine", {"numbering": {}, "discs": {}}, None]}, tmp_path)
    assert res["none"] == "Applicability: not assessed."


def test_applicability_line_for_unlabelled_study(anchors, tmp_path):
    d = copy.deepcopy(anchors)
    d["applicability"]["labelled"] = False
    res = _run({"x": ["applicabilityLine", d, None]}, tmp_path)
    assert "levels NOT labelled" in res["x"]
