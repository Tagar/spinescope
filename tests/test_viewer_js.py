"""Viewer JavaScript: syntax, and the browser geometry maths checked against the Python geometry.

Skipped when node is not installed. The browser code runs the same functions (math.js) that the
page uses; node evaluates them on cases generated in Python and the results are compared.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from spinescope.anatomy.anchors import Anchors
from spinescope.geometry import Stack, intersection_segment
from spinescope.viewer.build import _asset
from spinescope.viewer.export import build_payload
from spinescope.viewer.reformat import plan_reformat

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

#: Runs the page's maths on the cases in argv[3] and writes the results to argv[4].
NODE_RUNNER = r"""
const fs = require('fs');
const M = require(process.argv[2]);
const payload = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const cases = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const S = {};
payload.series.forEach(s => { S[s.key] = s; });
const out = { pix: [], lps: [], proj: [], lines: [], resample: null, decode: null };

cases.pix.forEach(c => out.pix.push(M.pixToLps(S[c.key], c.k, c.r, c.c)));
cases.lps.forEach(c => {
  const g = S[c.key];
  const p = M.lpsToPix(g, c.X);
  const n = M.nearestSlice(g, c.X);
  out.lps.push({ k: p.k, r: p.r, c: p.c, nk: n.k, noff: n.off });
});
cases.proj.forEach(c => out.proj.push(M.projectToSlice(S[c.key], c.k, c.X)));
cases.lines.forEach(c => {
  const a = S[c.a], b = S[c.b];
  const dB = M.dot(b.ipp[c.kb], b.normal);
  const L = M.planeLine(a, c.ka, b.normal, dB);
  const seg = M.clipLine(L, a.rows, a.cols);
  out.lines.push({ L: [L.a, L.b, L.e], seg: seg });
});
if (cases.resample) {
  const T = S[cases.resample.key], src = S[cases.resample.src];
  M.decodeVoxels(src.enc).then(q => {
    const o = new Float32Array(T.rows * T.cols);
    M.resampleSlice(T, cases.resample.k, src, q, o);
    out.resample = Array.from(o);
    return M.decodeVoxels(cases.decode.key ? S[cases.decode.key].enc : src.enc);
  }).then(q => {
    let sum = 0;
    for (let i = 0; i < q.length; i++) sum += q[i];
    out.decode = { n: q.length, sum: sum, samples: cases.decode.idx.map(i => q[i]) };
    fs.writeFileSync(process.argv[5], JSON.stringify(out));
  }).catch(e => { console.error(e); process.exit(1); });
} else {
  fs.writeFileSync(process.argv[5], JSON.stringify(out));
}
"""


def _stack_from_record(rec: dict) -> Stack:
    """Python Stack with the browser's geometry (zero voxels: only the geometry is compared)."""
    return Stack(
        key=rec["key"],
        vol=np.zeros((rec["n"], rec["rows"], rec["cols"]), dtype=np.float32),
        ipp=np.array(rec["ipp"], dtype=np.float64),
        row_dir=np.array(rec["row_dir"]),
        col_dir=np.array(rec["col_dir"]),
        row_mm=rec["row_mm"],
        col_mm=rec["col_mm"],
        thickness=rec["thickness"],
    )


@pytest.fixture(scope="module")
def js_result(phantom, phantom_study, tmp_path_factory):
    """Run the browser maths in node once; returns (payload, cases, results, python stacks)."""
    root = tmp_path_factory.mktemp("viewer_js")
    payload = build_payload(phantom_study, Anchors.from_truth(phantom.truth), title="t")
    recs = {s["key"]: s for s in payload["series"]}
    stacks = {k: _stack_from_record(r) for k, r in recs.items()}
    rng = np.random.default_rng(1)

    cases: dict = {"pix": [], "lps": [], "proj": [], "lines": []}
    keys = list(recs)
    for key in keys:
        rec = recs[key]
        for k in sorted({0, rec["n"] // 2, rec["n"] - 1}):
            for r, c in ((0.0, 0.0), (rec["rows"] / 2, rec["cols"] / 2), (3.25, 7.5), (rec["rows"] - 1, 2.0)):
                cases["pix"].append({"key": key, "k": float(k), "r": r, "c": c})
        st = stacks[key]
        lo, hi = st.corners(0).min(axis=0), st.corners(st.n - 1).max(axis=0)
        for X in lo + (hi - lo) * rng.uniform(-0.1, 1.1, size=(6, 3)):
            cases["lps"].append({"key": key, "X": X.tolist()})
            cases["proj"].append({"key": key, "k": int(rng.integers(st.n)), "X": X.tolist()})
    # reference lines: slice ka of a, cut by slice kb of b
    pairs = [
        ("sag_t2", "ax_t2"),
        ("ax_t2", "sag_t2"),
        ("ax_t2", "mpr_cor"),
        ("mpr_cor", "sag_t2"),
        ("sag_t2", "mpr_cor"),
        ("sag_t1", "ax_t1"),
    ]
    for a, b in pairs:
        for ka in (0, recs[a]["n"] // 2, recs[a]["n"] - 1):
            for kb in (0, recs[b]["n"] // 2):
                cases["lines"].append({"a": a, "ka": ka, "b": b, "kb": kb})
    ref = plan_reformat(phantom_study, "cor")
    assert ref is not None
    cases["resample"] = {"key": "mpr_cor", "src": "ax_t2", "k": ref.n // 2}
    cases["decode"] = {"key": "ax_t2", "idx": [0, 1, 777, 4095, 20000, 36 * 128 * 128 - 1]}

    cpath, rpath, ppath, outpath = (root / n for n in ("case.json", "payload.json", "run.js", "out.json"))
    cpath.write_text(json.dumps(cases))
    ppath.write_text(NODE_RUNNER)
    (root / "payload.json").write_text(json.dumps(payload))
    math_path = str(Path(__file__).parents[1] / "src/spinescope/viewer/assets/math.js")
    proc = subprocess.run(
        [NODE, str(ppath), math_path, str(rpath), str(cpath), str(outpath)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(outpath.read_text())
    return payload, cases, res, stacks, phantom_study


def test_js_syntax_is_valid():
    for name in ("math.js", "app.js"):
        path = Path(__file__).parents[1] / "src/spinescope/viewer/assets" / name
        proc = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_inline_scripts_are_valid():
    # the page's own scripts (as inlined) must parse too
    import re
    import tempfile

    html = _asset("viewer.html")
    assert "__MATH__" in html and "__APP__" in html
    with tempfile.TemporaryDirectory() as d:
        for name in ("math.js", "app.js"):
            p = Path(d) / name
            p.write_text(_asset(name), encoding="utf-8")
            proc = subprocess.run([NODE, "--check", str(p)], capture_output=True, text=True)
            assert proc.returncode == 0, proc.stderr
    assert re.search(r"<script>\s*__MATH__", html)


def test_click_maths_match_python(js_result):
    _payload, cases, res, stacks, _study = js_result
    for c, got in zip(cases["pix"], res["pix"], strict=True):
        expect = stacks[c["key"]].pix_to_lps(c["k"], c["r"], c["c"])
        assert np.allclose(got, expect, atol=1e-6), c
    for c, got in zip(cases["lps"], res["lps"], strict=True):
        st = stacks[c["key"]]
        k, r, col = st.lps_to_pix(np.array(c["X"]))
        assert got["k"] == pytest.approx(k, abs=1e-6)
        assert got["r"] == pytest.approx(r, abs=1e-6)
        assert got["c"] == pytest.approx(col, abs=1e-6)
        nk, off = st.nearest_slice(np.array(c["X"]))
        assert got["nk"] == nk
        assert got["noff"] == pytest.approx(off, abs=1e-6)
    for c, got in zip(cases["proj"], res["proj"], strict=True):
        st = stacks[c["key"]]
        rel = np.array(c["X"]) - st.ipp[c["k"]]
        assert got["r"] == pytest.approx(rel @ st.col_dir / st.row_mm, abs=1e-6)
        assert got["c"] == pytest.approx(rel @ st.row_dir / st.col_mm, abs=1e-6)
        assert got["off"] == pytest.approx(rel @ st.normal, abs=1e-6)


def test_reference_lines_match_python_intersections(js_result):
    _payload, cases, res, stacks, _study = js_result
    checked = 0
    for c, got in zip(cases["lines"], res["lines"], strict=True):
        a, b = stacks[c["a"]], stacks[c["b"]]
        seg = got["seg"]
        if seg is None:
            continue
        (c0, r0), (c1, r1) = seg
        # both JS endpoints lie on slice ka of a and on plane kb of b
        for cc, rr in ((c0, r0), (c1, r1)):
            X = a.pix_to_lps(c["ka"], rr, cc)
            assert float(X @ b.normal) == pytest.approx(float(b.ipp[c["kb"]] @ b.normal), abs=1e-6)
        # the Python intersection (clipped to both images) lies on the same line
        py = intersection_segment(a, c["ka"], b, c["kb"])
        if py is None:
            continue
        L_a, L_b, L_e = got["L"]
        for rr, cc in py:
            assert L_a * cc + L_b * rr == pytest.approx(L_e, abs=1e-6)
            assert min(c0, c1) - 1e-6 <= cc <= max(c0, c1) + 1e-6
            assert min(r0, r1) - 1e-6 <= rr <= max(r0, r1) + 1e-6
        checked += 1
    assert checked >= 10


def test_reformat_resampling_matches_python(js_result):
    _payload, cases, res, stacks, study = js_result
    spec = cases["resample"]
    T = stacks[spec["key"]]
    src = study[spec["src"]]
    k = spec["k"]
    r_idx, c_idx = np.meshgrid(np.arange(T.vol.shape[1]), np.arange(T.vol.shape[2]), indexing="ij")
    pts = np.array([T.pix_to_lps(k, r, c) for r, c in zip(r_idx.ravel(), c_idx.ravel(), strict=True)])
    ks, rs, cs = (np.empty(len(pts)) for _ in range(3))
    inside = np.zeros(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        ks[i], rs[i], cs[i] = src.lps_to_pix(p)
        inside[i] = src.contains(p)
    ref = map_coordinates(src.vol.astype(np.float64), [ks, rs, cs], order=1, mode="nearest")
    ref = np.where(inside, ref, 0.0)
    got = np.array(res["resample"])
    assert got.shape == ref.shape
    # codes equal values because the phantom volumes are integers in 16-bit range
    vmin = float(src.vol.min())
    diff = np.abs(got - (ref - vmin))
    inside_pix = inside & (ref > 0)
    assert diff[inside_pix].max() < 1e-2
    assert np.mean(inside) > 0.2  # the reformat really overlaps the source


def test_browser_decode_matches_encoded_voxels(js_result):
    payload, cases, res, _stacks, study = js_result
    rec = next(s for s in payload["series"] if s["key"] == "ax_t2")
    vol = study["ax_t2"].vol.astype(np.float64)
    q = np.round((vol - rec["enc"]["vmin"]) / rec["enc"]["scale"]).astype(np.int64).ravel()
    assert res["decode"]["n"] == q.size
    assert res["decode"]["sum"] == pytest.approx(float(q.sum()), rel=1e-9)
    for i, v in zip(cases["decode"]["idx"], res["decode"]["samples"], strict=True):
        assert v == q[i]
