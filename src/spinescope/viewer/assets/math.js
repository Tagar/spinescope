/* spinescope viewer maths: pure functions, no DOM. Mirrors spinescope.geometry.Stack.
 * Geometry objects g carry: n, rows, cols, row_mm, col_mm, thickness, row_dir, col_dir, normal,
 * ipp (n arrays of 3 LPS mm), positions (ipp . normal). Coordinates: LPS mm; pixel (r, c) is the
 * pixel centre at integer index. Runs in browsers and under node (module.exports). */
var SSMath = (function () {
  'use strict';

  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }
  function unit(v) { var m = Math.sqrt(dot(v, v)); return [v[0] / m, v[1] / m, v[2] / m]; }

  // origin of (fractional) slice k: lerp of ipp, extrapolated linearly past the ends (as Stack._origin)
  function origin(g, k) {
    if (g.n === 1) {
      var s = k * g.thickness;
      return [0, 1, 2].map(function (i) { return g.ipp[0][i] + s * g.normal[i]; });
    }
    var k0 = clamp(Math.floor(k), 0, g.n - 2), f = k - k0, a = g.ipp[k0], b = g.ipp[k0 + 1];
    return [0, 1, 2].map(function (i) { return a[i] * (1 - f) + b[i] * f; });
  }

  // slice index of a plane offset d = X . normal (as Stack.slice_index); fractional, extrapolated
  function sliceIndex(g, d) {
    var pos = g.positions, n = g.n;
    if (n === 1) return (d - pos[0]) / (g.thickness || 1);
    if (d <= pos[0]) return (d - pos[0]) / (pos[1] - pos[0]);
    if (d >= pos[n - 1]) return n - 1 + (d - pos[n - 1]) / (pos[n - 1] - pos[n - 2]);
    var lo = 0, hi = n - 1, m;
    while (hi - lo > 1) { m = (lo + hi) >> 1; if (pos[m] <= d) lo = m; else hi = m; }
    return lo + (d - pos[lo]) / (pos[hi] - pos[lo]);
  }

  // slice k (fractional), row r, column c -> LPS mm (as Stack.pix_to_lps)
  function pixToLps(g, k, r, c) {
    var o = origin(g, k);
    return [0, 1, 2].map(function (i) {
      return o[i] + c * g.col_mm * g.row_dir[i] + r * g.row_mm * g.col_dir[i];
    });
  }

  // LPS mm -> {k, r, c} (k fractional, not clamped), as Stack.lps_to_pix
  function lpsToPix(g, X) {
    var k = sliceIndex(g, dot(X, g.normal)), o = origin(g, k);
    var rel = [X[0] - o[0], X[1] - o[1], X[2] - o[2]];
    return { k: k, r: dot(rel, g.col_dir) / g.row_mm, c: dot(rel, g.row_dir) / g.col_mm };
  }

  // nearest slice plane to a point, and the distance (mm) from that plane
  function nearestSlice(g, X) {
    var d = dot(X, g.normal), best = 0, bd = Infinity, k, e;
    for (k = 0; k < g.n; k++) {
      e = Math.abs(g.positions[k] - d);
      if (e < bd) { bd = e; best = k; }
    }
    return { k: best, off: bd };
  }

  // (r, c) of a point projected onto slice k's pixel grid, plus its signed offset (mm) from that plane
  function projectToSlice(g, k, X) {
    var o = g.ipp[k], rel = [X[0] - o[0], X[1] - o[1], X[2] - o[2]];
    return { r: dot(rel, g.col_dir) / g.row_mm, c: dot(rel, g.row_dir) / g.col_mm, off: dot(rel, g.normal) };
  }

  // The line where slice k of g meets the plane {X : X . m = dB}, as a*c + b*r = e in index coords
  function planeLine(g, k, m, dB) {
    var o = g.ipp[k];
    return {
      a: g.col_mm * dot(g.row_dir, m),
      b: g.row_mm * dot(g.col_dir, m),
      e: dB - dot(o, m)
    };
  }

  // Clip a*c + b*r = e to the image rectangle c in [-0.5, cols-0.5], r in [-0.5, rows-0.5].
  // Returns [[c, r], [c, r]] (both on the border) or null.
  function clipLine(L, rows, cols) {
    var a = L.a, b = L.b, e = L.e, n2 = a * a + b * b;
    if (n2 < 1e-18) return null;
    var nn = Math.sqrt(n2), p0 = [a * e / n2, b * e / n2], d = [-b / nn, a / nn];
    var lims = [[-0.5, cols - 0.5], [-0.5, rows - 0.5]], t0 = -Infinity, t1 = Infinity, i;
    for (i = 0; i < 2; i++) {
      var p = p0[i], di = d[i], lo = lims[i][0], hi = lims[i][1];
      if (Math.abs(di) < 1e-12) { if (p < lo || p > hi) return null; continue; }
      var ta = (lo - p) / di, tb = (hi - p) / di;
      if (ta > tb) { var tmp = ta; ta = tb; tb = tmp; }
      t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
    }
    if (!(t0 <= t1) || !isFinite(t0) || !isFinite(t1)) return null;
    return [[p0[0] + t0 * d[0], p0[1] + t0 * d[1]], [p0[0] + t1 * d[0], p0[1] + t1 * d[1]]];
  }

  // anatomical letter of a patient direction (+x = L, +y = P, +z = S), dominant axis
  function dirLabel(v) {
    var ax = 0, m = Math.abs(v[0]), i;
    for (i = 1; i < 3; i++) if (Math.abs(v[i]) > m) { m = Math.abs(v[i]); ax = i; }
    return v[ax] >= 0 ? ['L', 'P', 'S'][ax] : ['R', 'A', 'I'][ax];
  }
  function opposite(ch) { return { L: 'R', R: 'L', P: 'A', A: 'P', S: 'I', I: 'S' }[ch]; }

  function histogram(q) {
    var h = new Float64Array(65536), j;
    for (j = 0; j < q.length; j++) h[q[j]]++;
    return h;
  }
  // value (vmin + code * scale) at percentile p of the code histogram
  function percentile(hist, total, vmin, scale, p) {
    var target = p / 100 * total, cum = 0, v;
    for (v = 0; v < hist.length; v++) { cum += hist[v]; if (cum >= target) return vmin + v * scale; }
    return vmin + (hist.length - 1) * scale;
  }

  // Resample plane T's slice k from source volume S (geometry) with codes Q (row-major, [k][r][c]).
  // Trilinear, same validity rule as Stack.contains (half a voxel beyond the source). Outside: 0.
  // Writes codes (float, source units) into out (length T.rows * T.cols).
  function resampleSlice(T, k, S, Q, out) {
    var o = origin(T, k), rd = T.row_dir, cd = T.col_dir, rm = T.row_mm, cm = T.col_mm;
    var n = S.n, R = S.rows, C = S.cols, r, c, X, p, ks, rs, cs, k0, k1, r0, r1, c0, c1, fk, fr, fc, v;
    var sk, sr, sc;
    for (r = 0; r < T.rows; r++) {
      for (c = 0; c < T.cols; c++) {
        X = [o[0] + c * cm * rd[0] + r * rm * cd[0], o[1] + c * cm * rd[1] + r * rm * cd[1],
             o[2] + c * cm * rd[2] + r * rm * cd[2]];
        p = lpsToPix(S, X);
        ks = p.k; rs = p.r; cs = p.c;
        if (!(ks >= -0.5 && ks <= n - 0.5 && rs >= -0.5 && rs <= R - 0.5 && cs >= -0.5 && cs <= C - 0.5)) {
          out[r * T.cols + c] = 0;
          continue;
        }
        sk = clamp(ks, 0, n - 1); sr = clamp(rs, 0, R - 1); sc = clamp(cs, 0, C - 1);
        k0 = Math.floor(sk); r0 = Math.floor(sr); c0 = Math.floor(sc);
        k1 = Math.min(k0 + 1, n - 1); r1 = Math.min(r0 + 1, R - 1); c1 = Math.min(c0 + 1, C - 1);
        fk = sk - k0; fr = sr - r0; fc = sc - c0;
        v = 0;
        v += (1 - fk) * (1 - fr) * (1 - fc) * Q[(k0 * R + r0) * C + c0];
        v += (1 - fk) * (1 - fr) * fc * Q[(k0 * R + r0) * C + c1];
        v += (1 - fk) * fr * (1 - fc) * Q[(k0 * R + r1) * C + c0];
        v += (1 - fk) * fr * fc * Q[(k0 * R + r1) * C + c1];
        v += fk * (1 - fr) * (1 - fc) * Q[(k1 * R + r0) * C + c0];
        v += fk * (1 - fr) * fc * Q[(k1 * R + r0) * C + c1];
        v += fk * fr * (1 - fc) * Q[(k1 * R + r1) * C + c0];
        v += fk * fr * fc * Q[(k1 * R + r1) * C + c1];
        out[r * T.cols + c] = v;
      }
    }
    return out;
  }

  // Decode a payload {n, rows, cols, enc: 'deflate;shuffle-u16le', data: base64} into Uint16Array codes.
  function decodeVoxels(p) {
    if (typeof DecompressionStream === 'undefined' || typeof Blob === 'undefined')
      return Promise.reject(new Error('This browser cannot decompress the voxel data (needs DecompressionStream: ' +
        'Chrome/Edge 80+, Safari 16.4+ or Firefox 113+). Open the viewer in a current browser.'));
    var bin = atob(p.data), comp = new Uint8Array(bin.length), i;
    for (i = 0; i < bin.length; i++) comp[i] = bin.charCodeAt(i);
    var stream = new Blob([comp]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Response(stream).arrayBuffer().then(function (buf) {
      var raw = new Uint8Array(buf), N = p.n * p.rows * p.cols, q = new Uint16Array(N), j;
      if (raw.length !== 2 * N) throw new Error('voxel data has an unexpected size');
      for (j = 0; j < N; j++) q[j] = raw[j] | (raw[N + j] << 8);
      return q;
    });
  }

  return {
    dot: dot, clamp: clamp, unit: unit, origin: origin, sliceIndex: sliceIndex,
    pixToLps: pixToLps, lpsToPix: lpsToPix, nearestSlice: nearestSlice, projectToSlice: projectToSlice,
    planeLine: planeLine, clipLine: clipLine, dirLabel: dirLabel, opposite: opposite,
    histogram: histogram, percentile: percentile, resampleSlice: resampleSlice, decodeVoxels: decodeVoxels
  };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = SSMath;
