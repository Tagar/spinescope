/* spinescope viewer UI. Reads window.SPINESCOPE_DATA (see export.py); uses SSMath (math.js) and
 * SSLabels (labels.js: the label corrections, which mirror spinescope.anatomy.anchors). */
(function () {
  'use strict';
  var D = window.SPINESCOPE_DATA, M = SSMath, L = SSLabels;
  var MAXP = 4, PAD = 26, PANE_COLORS = ['#ff7b7b', '#5ec8ff', '#a8e063', '#d59cff'];
  var TIER_COLOR = { high: '#66bb6a', medium: '#ffca28', low: '#ef5350' };
  var PRESETS = { T1: [1, 99.5], T2: [0.5, 98.5], STIR: [0.5, 95], Auto: [0.5, 99.5] };
  var MONO = 'ui-monospace, Menlo, Consolas, monospace';
  var HISTORY_MAX = 100;
  var POLICY = D.policy || { who: 'viewer user', confHigh: 0.8, confLow: 0.5, confHuman: 0.95, confAi: 0.85 };
  var SER = {}, ORDER = [];
  D.series.forEach(function (s) {
    SER[s.key] = s;
    ORDER.push(s.key);
    s.q = null; s.hist = null; s.total = 0; s.loading = false; s.error = null; s.slices = new Map();
  });
  var $ = function (id) { return document.getElementById(id); };
  var panesEl = $('panes'), statusEl = $('status');
  var state = {
    layout: D.layout || 3, focus: 0, cursor: null, hover: '', msg: '', panes: [],
    model: D.anchors ? L.clone(D.anchors) : null,  // the labels being reviewed (Anchors.to_dict shape)
    history: [], discs: [], addMode: null
  };
  var rafPending = false;

  // ------------------------------------------------------------------ helpers
  function scheduleRender() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(function () { rafPending = false; renderAll(); });
  }
  function setMsg(m) { state.msg = m; updateStatus(); }
  function fmt3(v) { return '(' + v.map(function (x) { return x.toFixed(1); }).join(', ') + ')'; }
  function spacingOf(g) { return g.n > 1 ? Math.abs(g.positions[1] - g.positions[0]) : g.thickness; }
  function orientName(g) {
    var n = g.normal, ax = 0, m = Math.abs(n[0]), i;
    for (i = 1; i < 3; i++) if (Math.abs(n[i]) > m) { m = Math.abs(n[i]); ax = i; }
    var tilt = Math.acos(Math.min(1, m)) * 180 / Math.PI;
    return ['sagittal', 'coronal', 'axial'][ax] + (tilt > 3 ? ', tilt ' + Math.round(tilt) + ' deg' : '');
  }
  // the series that owns the voxels (a reformat has none of its own)
  function root(key) { var s = SER[key]; return s.reformat ? SER[s.reformat.source] : s; }
  function presetName(key) {
    var c = SER[key].contrast;
    return c === 't1' ? 'T1' : c === 't2' ? 'T2' : c === 'stir' ? 'STIR' : 'Auto';
  }
  function wrapText(ctx, text, x, y, maxW, lh) {
    var words = String(text).split(' '), line = '', yy = y;
    words.forEach(function (w) {
      var test = line ? line + ' ' + w : w;
      if (ctx.measureText(test).width > maxW && line) { ctx.fillText(line, x, yy); line = w; yy += lh; }
      else line = test;
    });
    if (line) ctx.fillText(line, x, yy);
  }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // ------------------------------------------------------------------ data
  function ensure(key) {
    var r = root(key);
    if (r.q || r.loading || r.error) return;
    r.loading = true;
    M.decodeVoxels(r.enc).then(function (q) {
      r.q = q; r.hist = M.histogram(q); r.total = q.length; r.loading = false;
      scheduleRender();
    }, function (e) {
      r.loading = false; r.error = e.message;
      setMsg(key + ': ' + e.message);
      scheduleRender();
    });
  }
  // codes of slice k of series key (float codes; reformats are resampled here, cached per slice)
  function sliceArray(key, k) {
    var s = SER[key], r = root(key);
    if (!r.q) return null;
    var R = s.rows, C = s.cols;
    if (!s.reformat) return r.q.subarray(k * R * C, (k + 1) * R * C);
    var hit = s.slices.get(k);
    if (hit) return hit;
    var out = new Float32Array(R * C);
    M.resampleSlice(s, k, r, r.q, out);
    if (s.slices.size >= 48) s.slices.delete(s.slices.keys().next().value);
    s.slices.set(k, out);
    return out;
  }
  function windowFor(key, name) {
    var r = root(key);
    if (!r.q) return null;
    var pr = PRESETS[name] || PRESETS.Auto, vmin = r.enc.vmin, sc = r.enc.scale;
    var lo = M.percentile(r.hist, r.total, vmin, sc, pr[0]), hi = M.percentile(r.hist, r.total, vmin, sc, pr[1]);
    if (!(hi - lo > 1e-6)) hi = lo + 1;
    return { lo: lo, hi: hi };
  }

  // ------------------------------------------------------------------ panes
  function pixelAt(p, cx, cy) {
    if (!p.view || !p.key) return null;
    var rect = p.canvas.getBoundingClientRect(), x = cx - rect.left, y = cy - rect.top;
    var s = SER[p.key], v = p.view;
    var c = (x - v.ox) / (s.col_mm * v.m) - 0.5, r = (y - v.oy) / (s.row_mm * v.m) - 0.5;
    if (r < -0.5 || r > s.rows - 0.5 || c < -0.5 || c > s.cols - 0.5) return null;
    return { r: r, c: c };
  }
  function makePane(i) {
    var n = i + 1;
    var p = { i: i, key: null, k: 0, win: null, zoom: 1, panX: 0, panY: 0, wheel: 0, view: null, note: '',
      off: document.createElement('canvas'), drag: null, hoverKey: '' };
    var pane = document.createElement('section');
    pane.className = 'pane';
    pane.tabIndex = 0;
    pane.setAttribute('aria-label', 'Pane ' + n);
    pane.innerHTML = '<header class="phead"><select class="psel"></select><div class="pinfo"></div></header>' +
      '<div class="cv"><canvas></canvas></div>' +
      '<div class="pfoot"><input class="slider" type="range" min="0" max="0" value="0"><span class="pnote"></span></div>';
    p.el = pane;
    p.sel = pane.querySelector('.psel');
    p.sel.id = 'pane-' + n + '-series';
    p.sel.name = 'pane_' + n + '_series';
    p.sel.setAttribute('aria-label', 'Series shown in pane ' + n);
    p.info = pane.querySelector('.pinfo');
    p.canvas = pane.querySelector('canvas');
    p.canvas.setAttribute('aria-hidden', 'true');
    p.slider = pane.querySelector('.slider');
    p.slider.id = 'pane-' + n + '-slice';
    p.slider.name = 'pane_' + n + '_slice';
    p.slider.setAttribute('aria-label', 'Slice in pane ' + n);
    p.pnote = pane.querySelector('.pnote');
    p.ctx = p.canvas.getContext('2d');
    p.offCtx = p.off.getContext('2d');
    ORDER.forEach(function (key) {
      var o = document.createElement('option');
      o.value = key;
      o.textContent = SER[key].label;
      o.title = SER[key].label;
      p.sel.appendChild(o);
    });
    p.sel.addEventListener('change', function () { setSeries(i, p.sel.value); });
    p.slider.addEventListener('input', function () { setSlice(i, +p.slider.value); });
    pane.addEventListener('focus', function () { setFocus(i); });
    pane.addEventListener('mousedown', function () { setFocus(i); }, true);

    var cv = p.canvas;
    cv.addEventListener('mousedown', function (ev) {
      if (ev.button !== 0 && ev.button !== 1) return;
      ev.preventDefault();
      pane.focus({ preventScroll: true });
      p.drag = { x0: ev.clientX, y0: ev.clientY, x: ev.clientX, y: ev.clientY, moved: false,
        pan: ev.button === 1 || ev.shiftKey || ev.altKey };
    });
    cv.addEventListener('mousemove', function (ev) {
      if (p.drag || !p.key) return;
      var px = pixelAt(p, ev.clientX, ev.clientY);
      if (!px) { if (state.hover) { state.hover = ''; updateStatus(); } return; }
      var s = SER[p.key], arr = sliceArray(p.key, p.k), rr = Math.round(px.r), cc = Math.round(px.c);
      var val = arr ? root(p.key).enc.vmin + arr[rr * s.cols + cc] * root(p.key).enc.scale : NaN;
      state.hover = 'pane ' + n + ' ' + p.key + ' slice ' + (p.k + 1) + ' row ' + rr + ' col ' + cc +
        ' value ' + (isNaN(val) ? '-' : Math.round(val)) + ' LPS ' + fmt3(M.pixToLps(s, p.k, px.r, px.c));
      updateStatus();
    });
    cv.addEventListener('mouseleave', function () { state.hover = ''; updateStatus(); });
    cv.addEventListener('dblclick', function () {
      p.zoom = 1; p.panX = 0; p.panY = 0;
      p.win = p.key ? windowFor(p.key, presetName(p.key)) : null;
      scheduleRender();
    });
    cv.addEventListener('wheel', function (ev) {
      ev.preventDefault();
      if (!p.key) return;
      setFocus(i);
      if (ev.ctrlKey || ev.metaKey) {
        p.zoom = M.clamp(p.zoom * Math.exp(-ev.deltaY * 0.01), 0.5, 16);
        scheduleRender();
        return;
      }
      p.wheel += ev.deltaY;
      if (Math.abs(p.wheel) >= 25) {
        var step = p.wheel > 0 ? 1 : -1;
        p.wheel = 0;
        setSlice(i, p.k + step);
      }
    }, { passive: false });
    cv.addEventListener('contextmenu', function (ev) { ev.preventDefault(); });
    panesEl.appendChild(pane);
    return p;
  }

  window.addEventListener('mousemove', function (ev) {
    state.panes.forEach(function (p) {
      var d = p.drag;
      if (!d) return;
      var dx = ev.clientX - d.x, dy = ev.clientY - d.y;
      if (!d.moved && Math.abs(ev.clientX - d.x0) + Math.abs(ev.clientY - d.y0) > 4) d.moved = true;
      if (!d.moved) return;
      d.x = ev.clientX; d.y = ev.clientY;
      if (d.pan) {
        p.panX += dx; p.panY += dy;
      } else if (p.win) {
        var W = p.win.hi - p.win.lo, L0 = (p.win.hi + p.win.lo) / 2;
        W = Math.max(W * Math.exp(dx * 0.005), 1e-3);
        L0 += dy * W / 400;
        p.win = { lo: L0 - W / 2, hi: L0 + W / 2 };
      }
      scheduleRender();
    });
  });
  window.addEventListener('mouseup', function () {
    state.panes.forEach(function (p) {
      var d = p.drag;
      if (!d) return;
      p.drag = null;
      if (!d.moved && !d.pan) clickPane(p, d.x0, d.y0);
    });
    scheduleRender();
  });

  function clickPane(p, cx, cy) {
    var px = pixelAt(p, cx, cy);
    if (!px || !p.key) return;
    var centre = M.pixToLps(SER[p.key], p.k, px.r, px.c);
    if (state.addMode) {
      if (SER[p.key].plane !== 'sag') {
        setCorrMsg('Place disc ' + state.addMode.label + ' on a sagittal pane (this one is ' + SER[p.key].plane +
          ').', 'err');
        return;
      }
      placeDisc(centre);
      return;
    }
    state.cursor = centre;
    syncPanes(p.i);
  }
  // every visible pane except `skip` moves to the slice nearest the cursor
  function syncPanes(skip) {
    if (!state.cursor) { scheduleRender(); return; }
    for (var j = 0; j < state.layout; j++) {
      var q = state.panes[j];
      if (j === skip || !q.key) continue;
      q.k = M.nearestSlice(SER[q.key], state.cursor).k;
    }
    scheduleRender();
  }
  function setFocus(i) { state.focus = i; scheduleRender(); }
  function focusPane(i) {
    var n = state.layout;
    state.focus = ((i % n) + n) % n;
    state.panes[state.focus].el.focus({ preventScroll: true });
    scheduleRender();
  }
  function setSlice(i, k) {
    var p = state.panes[i];
    if (!p.key) return;
    p.k = M.clamp(Math.round(k), 0, SER[p.key].n - 1);
    setFocus(i);
  }
  function setSeries(i, key) {
    var p = state.panes[i], s = SER[key];
    p.key = key; p.win = null; p.zoom = 1; p.panX = 0; p.panY = 0; p.wheel = 0;
    p.k = state.cursor ? M.nearestSlice(s, state.cursor).k : Math.floor(s.n / 2);
    ensure(key);
    setFocus(i);
  }
  function setLayout(n) {
    state.layout = n;
    if (state.focus >= n) state.focus = 0;
    for (var j = 0; j < n; j++) {
      if (!state.panes[j].key) setSeries(j, D.panes[j]);
    }
    scheduleRender();
  }
  function applyPreset(name) {
    var p = state.panes[state.focus];
    if (!p.key) return;
    if (!root(p.key).q) { setMsg('Wait for the focused series to load, then apply the preset.'); return; }
    p.win = windowFor(p.key, name);
    scheduleRender();
  }
  // nearest disc whose centre projects inside slice k (axial panes show only this one)
  function nearestDisc(g, k) {
    var best = null;
    state.discs.forEach(function (D1) {
      var pr = M.projectToSlice(g, k, D1.center);
      if (pr.r < -0.5 || pr.r > g.rows - 0.5 || pr.c < -0.5 || pr.c > g.cols - 0.5) return;
      if (!best || Math.abs(pr.off) < Math.abs(best.off)) best = { D: D1, off: pr.off };
    });
    return best;
  }

  // ------------------------------------------------------------------ rendering
  function rectsOverlap(a, b) {
    return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
  }
  function renderPane(p) {
    var cv = p.canvas, ctx = p.ctx, dpr = window.devicePixelRatio || 1;
    var cw = cv.clientWidth, ch = cv.clientHeight;
    if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) {
      cv.width = Math.round(cw * dpr);
      cv.height = Math.round(ch * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, cw, ch);
    p.sel.value = p.key;
    var s = SER[p.key], r = root(p.key);
    p.slider.max = s.n - 1;
    p.slider.value = p.k;
    p.info.textContent = 'slice ' + (p.k + 1) + '/' + s.n + (p.win ? '   W ' + Math.round(p.win.hi - p.win.lo) +
      '  L ' + Math.round((p.win.hi + p.win.lo) / 2) : '') + '   ' + orientName(s) + '   ' +
      s.col_mm.toFixed(2) + ' x ' + s.row_mm.toFixed(2) + ' mm' +
      (s.reformat ? '   REFORMAT from ' + s.reformat.source : '');
    var arr = sliceArray(p.key, p.k);
    if (!arr) {
      p.view = null;
      ensure(p.key);
      ctx.font = '13px ' + MONO;
      ctx.fillStyle = r.error ? '#ff9e9e' : '#9a9a9a';
      wrapText(ctx, r.error || 'Loading ' + p.key + ' ...', 14, 24, cw - 28, 18);
      p.pnote.textContent = '';
      return;
    }
    if (!p.win) p.win = windowFor(p.key, presetName(p.key));
    var R = s.rows, C = s.cols, N = R * C, vmin = r.enc.vmin, sc = r.enc.scale;

    // windowed slice -> offscreen canvas (1 pixel per matrix element)
    if (p.off.width !== C || p.off.height !== R) { p.off.width = C; p.off.height = R; }
    var id = p.offCtx.createImageData(C, R), d = id.data;
    var lo = p.win.lo, k255 = 255 / (p.win.hi - p.win.lo), j, v, o;
    for (j = 0; j < N; j++) {
      v = (vmin + arr[j] * sc - lo) * k255;
      v = v < 0 ? 0 : (v > 255 ? 255 : v);
      o = j * 4;
      d[o] = v; d[o + 1] = v; d[o + 2] = v; d[o + 3] = 255;
    }
    p.offCtx.putImageData(id, 0, 0);

    // fit the image to the pane keeping the aspect ratio from the pixel spacing, then zoom and pan
    var availW = Math.max(cw - 2 * PAD, 10), availH = Math.max(ch - 2 * PAD, 10);
    var fit = Math.min(availW / (C * s.col_mm), availH / (R * s.row_mm));
    var m = fit * p.zoom, W = C * s.col_mm * m, H = R * s.row_mm * m;
    var ox = (cw - W) / 2 + p.panX, oy = (ch - H) / 2 + p.panY;
    p.view = { ox: ox, oy: oy, m: m };
    var Lx = ox, T = oy, RR = ox + W, B = oy + H;
    function X(c) { return ox + (c + 0.5) * s.col_mm * m; }
    function Y(rr) { return oy + (rr + 0.5) * s.row_mm * m; }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(p.off, ox, oy, W, H);

    // ---- overlays: lines inside the image; labels sit on the image edge where a line leaves it
    var placed = [], hidden = 0;
    ctx.font = '12px ' + MONO;
    function seg(sg, color, dash, lw) {
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(X(sg[0][0]), Y(sg[0][1])); ctx.lineTo(X(sg[1][0]), Y(sg[1][1])); ctx.stroke();
      ctx.restore();
    }
    function label(sg, text, color) {
      var w = ctx.measureText(text).width + 8, h = 16, cands = [];
      sg.forEach(function (e) {
        var ex = X(e[0]), ey = Y(e[1]);
        var onL = Math.abs(ex - Lx) < 1, onR = Math.abs(ex - RR) < 1, onT = Math.abs(ey - T) < 1;
        [0, 18, -18, 36, -36, 54, -54, 72, -72, 90, -90].forEach(function (t) {
          if (onL || onR) {
            cands.push({ x: onL ? Lx + 4 : RR - w - 4, y: M.clamp(ey + t - h / 2, T + 2, B - h - 2) });
          } else {
            cands.push({ x: M.clamp(ex + t - w / 2, Lx + 2, RR - w - 2), y: onT ? T + 4 : B - h - 4 });
          }
        });
      });
      for (var q = 0; q < cands.length; q++) {
        var rect = { x: cands[q].x, y: cands[q].y, w: w, h: h }, clash = false;
        for (var z = 0; z < placed.length; z++) if (rectsOverlap(rect, placed[z])) { clash = true; break; }
        if (clash) continue;
        placed.push(rect);
        ctx.fillStyle = 'rgba(0,0,0,0.72)';
        ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
        ctx.fillStyle = color;
        ctx.fillText(text, rect.x + 4, rect.y + 12);
        return;
      }
      hidden++;
    }

    // planes of the other panes, as lines across this slice
    for (var jj = 0; jj < state.layout; jj++) {
      var other = state.panes[jj];
      if (jj === p.i || !other.key) continue;
      var g2 = SER[other.key], dB = M.dot(g2.ipp[other.k], g2.normal);
      var s2 = M.clipLine(M.planeLine(s, p.k, g2.normal, dB), R, C);
      if (!s2) continue;
      seg(s2, PANE_COLORS[jj], [], 1.5);
      label(s2, 'P' + (jj + 1) + ' ' + other.key + ' #' + (other.k + 1), PANE_COLORS[jj]);
    }

    // disc levels from the anchors: all discs on oblique/sagittal/coronal panes, the nearest on axial ones.
    // Colour = confidence tier (the text label says the level, so colour is never the only cue).
    var axial = Math.abs(s.normal[2]) > 0.7, nd = null;
    if (axial) nd = nearestDisc(s, p.k);
    var discs = axial ? (nd ? [nd.D] : []) : state.discs;
    discs.forEach(function (D1) {
      var sg = M.clipLine(M.planeLine(s, p.k, D1.normal, M.dot(D1.center, D1.normal)), R, C);
      if (!sg) return;
      var col = TIER_COLOR[L.tier(D1.confidence, POLICY)];
      seg(sg, col, [6, 4], 1.2);
      label(sg, D1.label, col);
    });

    // 3D cursor
    var note = '';
    if (state.cursor) {
      var pc = M.projectToSlice(s, p.k, state.cursor);
      if (pc.r >= -0.5 && pc.r <= R - 0.5 && pc.c >= -0.5 && pc.c <= C - 0.5) {
        var far = Math.abs(pc.off) > spacingOf(s) / 2 + 1e-6;
        ctx.save();
        ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 1; ctx.setLineDash(far ? [4, 4] : []);
        ctx.beginPath();
        ctx.moveTo(Lx, Y(pc.r)); ctx.lineTo(RR, Y(pc.r));
        ctx.moveTo(X(pc.c), T); ctx.lineTo(X(pc.c), B);
        ctx.stroke();
        ctx.beginPath(); ctx.arc(X(pc.c), Y(pc.r), 4, 0, 2 * Math.PI); ctx.stroke();
        ctx.restore();
        if (far) note = 'cursor ' + pc.off.toFixed(1) + ' mm off this slice';
      } else {
        note = 'cursor outside this slice';
      }
    }
    if (state.addMode && s.plane === 'sag') note = (note ? note + '; ' : '') + 'click to place ' + state.addMode.label;
    if (hidden) note = (note ? note + '; ' : '') + hidden + ' label' + (hidden > 1 ? 's' : '') + ' hidden (overlap)';
    p.note = note;
    p.pnote.textContent = note;

    // orientation letters live in the padding around the image (from the geometry, not hard-coded)
    var rl = M.dirLabel(s.row_dir), bl = M.dirLabel(s.col_dir);
    ctx.save();
    ctx.font = 'bold 15px ' + MONO;
    ctx.fillStyle = '#e8e8e8'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(rl, cw - PAD / 2, ch / 2);
    ctx.fillText(M.opposite(rl), PAD / 2, ch / 2);
    ctx.fillText(bl, cw / 2, ch - PAD / 2);
    ctx.fillText(M.opposite(bl), cw / 2, PAD / 2);
    ctx.restore();
    p.info.title = p.info.textContent;
  }

  // Grid placement: three panes put the first (sagittal, by default) in a tall column on the left, so the
  // tall spine image gets the full height; the other two share the right-hand column.
  function placePanes() {
    var n = state.layout;
    state.panes.forEach(function (p, j) {
      var st = p.el.style;
      if (n === 3 && j < 3) {
        st.gridColumn = j === 0 ? '1' : '2';
        st.gridRow = j === 0 ? '1 / span 2' : String(j);
      } else {
        st.gridColumn = '';
        st.gridRow = '';
      }
    });
    if (n === 3) {
      panesEl.style.gridTemplateColumns = 'minmax(0, 3fr) minmax(0, 2fr)';
      panesEl.style.gridTemplateRows = 'minmax(0, 1fr) minmax(0, 1fr)';
    } else if (n === 4) {
      panesEl.style.gridTemplateColumns = 'repeat(2, minmax(0, 1fr))';
      panesEl.style.gridTemplateRows = 'repeat(2, minmax(0, 1fr))';
    } else {
      panesEl.style.gridTemplateColumns = 'repeat(' + n + ', minmax(0, 1fr))';
      panesEl.style.gridTemplateRows = 'minmax(0, 1fr)';
    }
  }

  function renderAll() {
    state.panes.forEach(function (p, i) { p.el.style.display = i < state.layout ? '' : 'none'; });
    placePanes();
    for (var i = 0; i < state.layout; i++) renderPane(state.panes[i]);
    state.panes.forEach(function (p, j) {
      p.el.classList.toggle('focus', j === state.focus && j < state.layout);
    });
    document.querySelectorAll('#layout button').forEach(function (b) {
      var on = +b.dataset.n === state.layout;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    var used = {};
    state.panes.slice(0, state.layout).forEach(function (p) { used[p.key] = true; });
    document.querySelectorAll('#series li').forEach(function (li) {
      li.classList.toggle('on', !!used[li.dataset.key]);
    });
    updateStatus();
  }
  function updateStatus() {
    var lines = [];
    if (state.msg) lines.push(state.msg);
    if (state.cursor) {
      lines.push('3D cursor at LPS ' + fmt3(state.cursor) + ' mm (x = L, y = P, z = S)');
      state.panes.slice(0, state.layout).forEach(function (p, j) {
        if (!p.key) return;
        var s = SER[p.key], pc = M.projectToSlice(s, p.k, state.cursor);
        lines.push('  pane ' + (j + 1) + ' ' + p.key + ' slice ' + (p.k + 1) + ': ' +
          (Math.abs(pc.off) > spacingOf(s) / 2 ? pc.off.toFixed(1) + ' mm off-plane' : 'on slice') +
          (p.note ? ' (' + p.note + ')' : ''));
      });
    } else {
      lines.push('No 3D cursor. Click an image to place one; the other panes move to the nearest slice.');
    }
    if (state.hover) lines.push(state.hover);
    statusEl.textContent = lines.join('\n');
  }

  // ------------------------------------------------------------------ labels: status and banner
  function refreshDiscs() { state.discs = L.discsOf(state.model); }
  function tierWord(c) { var t = L.tier(c, POLICY); return t === 'high' ? 'high' : t === 'medium' ? 'medium' : 'low'; }

  function renderLabels() {
    var model = state.model, num = model ? (model.numbering || {}) : {};
    var b = L.banner(model, POLICY), bn = $('banner');
    bn.hidden = !b.text;
    bn.className = b.level;
    bn.textContent = b.text;

    $('method').textContent = model
      ? 'Method: ' + (model.method || 'not recorded') + (num.method ? '. Numbering: ' + num.method + '.' : '')
      : 'No automatic labels for this study.';

    var numline = $('numline');
    numline.textContent = '';
    if (model) {
      var c = Number(num.confidence || 0);
      numline.appendChild(document.createTextNode('Numbering confidence '));
      numline.appendChild(el('span', 'tier ' + L.tier(c, POLICY), tierWord(c) + ' ' + c.toFixed(2)));
      if (num.status) numline.appendChild(document.createTextNode('  status: ' + num.status));
    }

    var wl = $('warnlist');
    wl.textContent = '';
    if (model) {
      var seen = {};
      (model.warnings || []).forEach(function (w) {
        if (seen[w]) return;
        seen[w] = true;
        wl.appendChild(el('li', '', w));
      });
    }

    var dl = $('disclist');
    dl.textContent = '';
    state.discs.forEach(function (d) {
      var li = el('li'), dot = el('span', 'dot');
      dot.style.background = TIER_COLOR[L.tier(d.confidence, POLICY)];
      dot.setAttribute('aria-hidden', 'true');
      var btn = el('button', '', d.label + '   ' + Math.round(100 * d.confidence) + '% ' + tierWord(d.confidence));
      btn.type = 'button';
      btn.title = 'Jump the 3D cursor to ' + d.label;
      btn.addEventListener('click', function () { state.cursor = d.center.slice(); syncPanes(-1); });
      li.appendChild(dot); li.appendChild(btn);
      dl.appendChild(li);
    });
    if (!state.discs.length) dl.appendChild(el('li', 'hint', 'No discs to show.'));

    renderLog();
    fillSelects();
    setCorrEnabled(!!model);
  }
  function fillSelects() {
    [['ren-from', 'Choose disc'], ['del-disc', 'Choose disc']].forEach(function (pair) {
      var sel = $(pair[0]), prev = sel.value;
      sel.textContent = '';
      sel.appendChild(new Option(pair[1], ''));
      state.discs.forEach(function (d) { sel.appendChild(new Option(d.label, d.label)); });
      sel.value = state.discs.some(function (d) { return d.label === prev; }) ? prev : '';
    });
  }
  function renderLog() {
    var ol = $('corrlog');
    ol.textContent = '';
    var entries = (state.model && state.model.numbering && state.model.numbering.corrections) || [];
    entries.forEach(function (e) {
      var li = el('li', '', e.what || e.action);
      li.appendChild(document.createTextNode(' '));
      li.appendChild(el('small', '', (e.why ? '"' + e.why + '" ' : '') + (e.when ? '(' + e.when + ')' : '')));
      ol.appendChild(li);
    });
  }
  function setCorrEnabled(on) {
    ['shift-up', 'shift-down', 'ren-go', 'del-go', 'add-go', 'undo', 'confirm', 'export'].forEach(function (id) {
      $(id).disabled = !on && id !== 'undo';
    });
    $('ren-from').disabled = !on; $('ren-to').disabled = !on; $('del-disc').disabled = !on;
    $('add-label').disabled = !on;
    if (!on) setCorrMsg('This study has no labels to correct. Labels come from label_spine(study).', '');
  }
  function setCorrMsg(text, kind) {
    var m = $('corr-msg');
    m.textContent = text;
    m.className = 'fieldmsg' + (kind ? ' ' + kind : '');
  }
  function reasonText() { return $('corr-why').value.trim(); }

  // ------------------------------------------------------------------ corrections
  function opts(extra) {
    var o = { who: POLICY.who, why: reasonText(), when: new Date().toISOString(), source: 'human',
      numConfidence: POLICY.confHuman, confHuman: POLICY.confHuman, confAi: POLICY.confAi };
    for (var k in extra || {}) o[k] = extra[k];
    return o;
  }
  // Apply one correction to a copy of the labels. On failure the labels, history and undo are unchanged.
  function applyCorrection(fn) {
    if (!state.model) { setCorrMsg('This study has no labels to correct.', 'err'); return false; }
    if (!reasonText()) {
      setCorrMsg('Type a reason in the Reason box first. It is saved with the change.', 'err');
      $('corr-why').focus();
      return false;
    }
    var next = L.clone(state.model);
    try {
      fn(next);
    } catch (e) {
      setCorrMsg('Not applied: ' + e.message, 'err');
      return false;
    }
    state.history.push(state.model);
    if (state.history.length > HISTORY_MAX) state.history.shift();
    state.model = next;
    refreshDiscs();
    renderLabels();
    scheduleRender();
    var last = next.numbering.corrections[next.numbering.corrections.length - 1];
    setCorrMsg('Applied: ' + (last ? last.what : 'change') + '.', 'ok');
    return true;
  }
  function undo() {
    if (!state.history.length) { setCorrMsg('Nothing to undo.', 'err'); return; }
    state.model = state.history.pop();
    refreshDiscs();
    renderLabels();
    scheduleRender();
    setCorrMsg('Undone.', 'ok');
  }
  function shiftBy(n) { applyCorrection(function (a) { L.shift(a, n, opts()); }); }
  function renameDisc() {
    var from = $('ren-from').value, to = $('ren-to').value.trim();
    if (!from) { setCorrMsg('Choose the disc to rename.', 'err'); return; }
    if (!to) { setCorrMsg('Type the new label, for example L5-L6.', 'err'); $('ren-to').focus(); return; }
    var map = {};
    map[from] = to;
    if (applyCorrection(function (a) { L.relabel(a, map, opts()); })) $('ren-to').value = '';
  }
  function deleteDisc() {
    var label = $('del-disc').value;
    if (!label) { setCorrMsg('Choose the disc to delete.', 'err'); return; }
    applyCorrection(function (a) { L.drop(a, label, opts()); });
  }
  function startAdd() {
    if (state.addMode) { cancelAdd(); return; }
    var label = $('add-label').value.trim();
    if (!state.model) { setCorrMsg('This study has no labels to correct.', 'err'); return; }
    if (!L.isDiscLabel(label)) { setCorrMsg('Type the new disc label first, for example L4-L5.', 'err'); $('add-label').focus(); return; }
    if (state.model.discs[label]) { setCorrMsg('Disc ' + label + ' already exists.', 'err'); return; }
    if (!reasonText()) { setCorrMsg('Type a reason in the Reason box first.', 'err'); $('corr-why').focus(); return; }
    state.addMode = { label: label };
    $('add-go').textContent = 'Cancel';
    $('add-go').setAttribute('aria-pressed', 'true');
    setCorrMsg('Click the centre of disc ' + label + ' on a sagittal pane. Esc cancels.', '');
    scheduleRender();
  }
  function cancelAdd() {
    state.addMode = null;
    $('add-go').textContent = 'Place';
    $('add-go').setAttribute('aria-pressed', 'false');
    setCorrMsg('Placement cancelled.', '');
    scheduleRender();
  }
  function placeDisc(centre) {
    var label = state.addMode.label;
    if (applyCorrection(function (a) { L.add(a, label, centre, opts({ clicked: true })); })) {
      state.addMode = null;
      $('add-go').textContent = 'Place';
      $('add-go').setAttribute('aria-pressed', 'false');
      $('add-label').value = '';
    }
    scheduleRender();
  }
  function confirmLevels() { applyCorrection(function (a) { L.confirm(a, opts()); }); }

  function exportLabels() {
    if (!state.model) { setCorrMsg('This study has no labels to export.', 'err'); return; }
    var doc;
    try {
      doc = L.exportAnchors(state.model, opts());
    } catch (e) {
      setCorrMsg(e.message, 'err');
      $('corr-why').focus();
      return;
    }
    var text = JSON.stringify(doc, null, 1) + '\n';
    var link = $('export-link');
    if (link.getAttribute('href')) URL.revokeObjectURL(link.getAttribute('href'));
    link.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    link.download = 'spinescope_anchors.json';
    link.hidden = false;
    link.click();
    var n = (doc.numbering.corrections || []).length - 1;
    setCorrMsg('Exported ' + n + ' recorded change' + (n === 1 ? '' : 's') + ' with a provenance entry for this export.', 'ok');
  }

  // ------------------------------------------------------------------ sidebar
  function buildSidebar() {
    $('title').textContent = D.title;
    $('disclaimer').textContent = D.disclaimer;
    var info = D.info || {}, parts = [];
    if (info.Manufacturer) parts.push(info.Manufacturer);
    if (info.ManufacturerModelName) parts.push(info.ManufacturerModelName);
    if (info.MagneticFieldStrength) parts.push(info.MagneticFieldStrength + ' T');
    $('scanner').textContent = parts.length ? 'Scanner (technical): ' + parts.join(', ') : '';

    var ul = $('series');
    ORDER.forEach(function (key) {
      var s = SER[key], li = document.createElement('li');
      li.dataset.key = key;
      li.title = s.label;
      li.tabIndex = 0;
      li.setAttribute('role', 'button');
      li.setAttribute('aria-label', 'Show ' + s.label + ' in the focused pane');
      var a = document.createElement('div'), b = document.createElement('div');
      a.innerHTML = '<b></b> <span></span>';
      a.querySelector('b').textContent = key;
      a.querySelector('span').textContent = s.reformat ? 'reformat' : (s.localizer ? 'localizer' : '');
      b.className = 'sub';
      b.textContent = s.n + ' slices, ' + s.cols + 'x' + s.rows + ' px, ' + s.col_mm.toFixed(2) + ' mm, ' +
        orientName(s);
      li.appendChild(a); li.appendChild(b);
      li.addEventListener('click', function () { setSeries(state.focus, key); });
      li.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); setSeries(state.focus, key); }
      });
      ul.appendChild(li);
    });

    var layoutEl = $('layout');
    [1, 2, 3, 4].forEach(function (n) {
      var b = document.createElement('button');
      b.type = 'button';
      b.dataset.n = n;
      b.textContent = n + (n === 1 ? ' pane' : ' panes');
      b.addEventListener('click', function () { setLayout(n); });
      layoutEl.appendChild(b);
    });
    var presetEl = $('presets');
    ['T1', 'T2', 'STIR', 'Auto'].forEach(function (name) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = name;
      b.addEventListener('click', function () { applyPreset(name); });
      presetEl.appendChild(b);
    });

    $('shift-up').addEventListener('click', function () { shiftBy(1); });
    $('shift-down').addEventListener('click', function () { shiftBy(-1); });
    $('ren-go').addEventListener('click', renameDisc);
    $('ren-to').addEventListener('keydown', function (ev) { if (ev.key === 'Enter') { ev.preventDefault(); renameDisc(); } });
    $('del-go').addEventListener('click', deleteDisc);
    $('add-go').addEventListener('click', startAdd);
    $('add-label').addEventListener('keydown', function (ev) { if (ev.key === 'Enter') { ev.preventDefault(); startAdd(); } });
    $('undo').addEventListener('click', undo);
    $('confirm').addEventListener('click', confirmLevels);
    $('export').addEventListener('click', exportLabels);
    $('clear').addEventListener('click', function () { state.cursor = null; scheduleRender(); });

    var refs = D.series.filter(function (s) { return s.reformat; });
    $('reformatnote').hidden = !refs.length;
    $('reformatnote').textContent = refs.map(function (s) {
      return s.plane.toUpperCase() + ' reformat: no ' + s.plane + ' series in this study, so this plane is ' +
        'resampled in your browser from ' + s.reformat.source + ' (trilinear, ' + s.reformat.px_mm.toFixed(2) +
        ' mm, coverage ' + Math.round(100 * s.reformat.coverage) + '%). Not a separately acquired series.';
    }).join(' ');

    $('helpbtn').addEventListener('click', openHelp);
    $('helpclose').addEventListener('click', closeHelp);
    $('help').addEventListener('click', function (ev) { if (ev.target === $('help')) closeHelp(); });
  }
  function openHelp() { $('help').hidden = false; $('helpclose').focus(); }
  function closeHelp() {
    if ($('help').hidden) return;
    $('help').hidden = true;
    $('helpbtn').focus();
  }

  // ------------------------------------------------------------------ keyboard
  // Form controls keep their own keys. Slice keys act on the focused pane only when focus is on the page
  // or in the panes, so arrows still scroll the sidebar when it has the focus.
  document.addEventListener('keydown', function (ev) {
    var t = ev.target, tag = t && t.tagName, k = ev.key;
    var helpOpen = !$('help').hidden;
    if (helpOpen) {
      if (k === 'Escape') { ev.preventDefault(); closeHelp(); }
      else if (k === 'Tab') { ev.preventDefault(); $('helpclose').focus(); }
      return;
    }
    if (k === 'Escape' && state.addMode) { cancelAdd(); return; }
    if ((ev.ctrlKey || ev.metaKey) && (k === 'z' || k === 'Z') && !ev.shiftKey && !isTyping(t)) {
      ev.preventDefault();
      undo();
      return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (isTyping(t) || tag === 'BUTTON' && (k === 'Enter' || k === ' ')) return;
    if (k === '?') { ev.preventDefault(); openHelp(); return; }
    if (k === '1' || k === '2' || k === '3' || k === '4') { setLayout(+k); ev.preventDefault(); return; }
    if (k === '[' || k === ']') { focusPane(state.focus + (k === ']' ? 1 : -1)); ev.preventDefault(); return; }
    if (k === 'c' || k === 'C') { state.cursor = null; scheduleRender(); return; }
    var onPage = !t || t === document.body || panesEl.contains(t);
    if (!onPage) return;
    var i = state.focus, p = state.panes[i];
    if (!p || !p.key) return;
    var n = SER[p.key].n, step = null;
    if (k === 'ArrowUp' || k === 'ArrowLeft') step = -1;
    else if (k === 'ArrowDown' || k === 'ArrowRight') step = 1;
    else if (k === 'PageUp') step = -5;
    else if (k === 'PageDown') step = 5;
    if (step !== null) { setSlice(i, p.k + step); ev.preventDefault(); return; }
    if (k === 'Home') { setSlice(i, 0); ev.preventDefault(); return; }
    if (k === 'End') { setSlice(i, n - 1); ev.preventDefault(); return; }
    if (k === '+' || k === '=') { p.zoom = M.clamp(p.zoom * 1.25, 0.5, 16); scheduleRender(); return; }
    if (k === '-') { p.zoom = M.clamp(p.zoom / 1.25, 0.5, 16); scheduleRender(); return; }
    if (k === '0') { p.zoom = 1; p.panX = 0; p.panY = 0; scheduleRender(); }
  });
  // text fields, selects and sliders keep their own keys (a range slider keeps arrows)
  function isTyping(t) {
    if (!t) return false;
    if (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT') return true;
    return t.tagName === 'INPUT' && (t.type !== 'button' && t.type !== 'checkbox');
  }

  // ------------------------------------------------------------------ start
  buildSidebar();
  for (var i = 0; i < MAXP; i++) state.panes.push(makePane(i));
  for (i = 0; i < MAXP; i++) {
    var key = D.panes[i];
    state.panes[i].key = key;
    state.panes[i].k = Math.floor(SER[key].n / 2);
    ensure(key);
  }
  refreshDiscs();
  renderLabels();
  if (window.ResizeObserver) new ResizeObserver(function () { scheduleRender(); }).observe(panesEl);
  window.addEventListener('resize', function () { scheduleRender(); });
  setLayout(state.layout);
})();
