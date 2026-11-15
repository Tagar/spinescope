/* spinescope viewer label review: pure functions on an Anchors dict (the shape of Anchors.to_dict()).
 * Mirrors spinescope.anatomy.anchors (shift_labels, relabel, drop_disc, add_disc, record_correction) so
 * an export loads with Anchors.from_json and carries the same levels Python would give. No DOM.
 * Every function that changes labels mutates the anchors object it is given and returns it; the page
 * passes a clone so a failed step leaves the current labels untouched. Runs in browsers and under node.
 */
var SSLabels = (function () {
  'use strict';

  // counted vertebrae, S1 = 0 (same order as spinescope.anatomy.numbering.VERTEBRA_NAMES)
  var NAMES = ['S1', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4',
    'T3', 'T2', 'T1'];
  var INDEX = {};
  NAMES.forEach(function (n, i) { INDEX[n] = i; });
  var DISC_RE = /^([TLS]\d{1,2})-([TLS]\d{1,2})$/;
  var TEXT_MAX = 200;
  var SOURCES = ['programmatic', 'ai_reader', 'human'];
  var RANGE_ERR = 'leaves the counted range S1..T1';

  function has(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }
  function clamp01(x) { return Math.min(1, Math.max(0, x)); }
  function round3(x) { return Math.round(x * 1000) / 1000; }
  function trunc(s) { return String(s === undefined || s === null ? '' : s).trim().slice(0, TEXT_MAX); }

  function isDiscLabel(s) {
    var m = DISC_RE.exec(String(s));
    return m !== null && [m[1], m[2]].every(function (v) { return has(INDEX, v) || v === 'L6'; });
  }
  // (upper, lower) of a disc label such as L4-L5; throws for anything else
  function splitDisc(label) {
    var m = DISC_RE.exec(String(label));
    if (m === null || !isDiscLabel(label)) {
      throw new Error('not a disc label: "' + label + '" (expected like L4-L5; levels T1..S1)');
    }
    return [m[1], m[2]];
  }
  function step(name, n) {
    if (!has(INDEX, name)) throw new Error('cannot shift vertebra ' + name + ': it is outside the counted range S1..T1');
    var j = INDEX[name] + n;
    if (j < 0 || j >= NAMES.length) throw new Error('shifting ' + name + ' by ' + n + ' levels ' + RANGE_ERR);
    return NAMES[j];
  }
  // a disc label or a vertebra name shifted by n levels (positive = cranial)
  function shiftLevel(level, n) {
    if (String(level).indexOf('-') >= 0) {
      var p = splitDisc(level);
      return step(p[0], n) + '-' + step(p[1], n);
    }
    return step(level, n);
  }
  function median(xs) {
    var s = xs.slice().sort(function (a, b) { return a - b; }), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function unit(v) { var m = Math.sqrt(dot(v, v)); return [v[0] / m, v[1] / m, v[2] / m]; }

  // disc labels ordered superior -> inferior (Anchors.disc_labels)
  function discOrder(a) {
    return Object.keys(a.discs).sort(function (x, y) { return a.discs[y].center[2] - a.discs[x].center[2]; });
  }
  // display records for the page: superior first, unit normals, confidences
  function discsOf(a) {
    if (!a) return [];
    return discOrder(a).map(function (lab) {
      var d = a.discs[lab];
      return { label: lab, center: d.center.slice(), normal: unit(d.normal), confidence: d.confidence,
        height_mm: d.height_mm };
    });
  }
  function clone(a) { return JSON.parse(JSON.stringify(a)); }

  // Record one provenance entry (Anchors.record_correction) and optionally set numbering confidence.
  // who / why / when / what are free text or a timestamp supplied by the caller; source defaults to 'human'.
  function record(a, action, opts, detail, confidence) {
    var source = opts.source || 'human';
    if (SOURCES.indexOf(source) < 0) throw new Error('correction source must be one of ' + SOURCES.join(', '));
    if (!opts.when) throw new Error('a correction needs a when timestamp');
    var entry = {
      action: action,
      who: trunc(opts.who || 'programmatic'),
      source: source,
      why: trunc(opts.why),
      detail: detail || {},
      labels_after: discOrder(a),
      what: trunc(opts.what || action),
      when: String(opts.when)
    };
    if (!a.numbering) a.numbering = {};
    if (!Array.isArray(a.numbering.corrections)) a.numbering.corrections = [];
    a.numbering.corrections.push(entry);
    if (confidence !== undefined && confidence !== null) a.numbering.confidence = round3(clamp01(confidence));
    if (opts.warning) a.warnings.push(opts.warning);
    return a;
  }
  // numbering confidence to set after a correction (opts.numConfidence); undefined leaves it unchanged
  function conf(opts) { return opts.numConfidence === undefined ? undefined : opts.numConfidence; }

  // rename discs and vertebrae atomically (targets validated first), then the conus level
  function rename(a, discMap, vertMap, conusMap) {
    Object.keys(discMap).forEach(function (k) { splitDisc(discMap[k]); });
    var newDiscs = Object.keys(a.discs).map(function (k) { return has(discMap, k) ? discMap[k] : k; });
    var newVerts = Object.keys(a.vertebrae).map(function (k) { return has(vertMap, k) ? vertMap[k] : k; });
    if (new Set(newDiscs).size !== newDiscs.length) throw new Error('relabelling would give two discs the same label');
    if (new Set(newVerts).size !== newVerts.length) throw new Error('relabelling would give two vertebrae the same name');
    var discs = {}, verts = {};
    Object.keys(a.discs).forEach(function (k) { discs[has(discMap, k) ? discMap[k] : k] = a.discs[k]; });
    Object.keys(a.vertebrae).forEach(function (k) { verts[has(vertMap, k) ? vertMap[k] : k] = a.vertebrae[k]; });
    a.discs = discs;
    a.vertebrae = verts;
    if (a.conus && a.conus.level) {
      var level = String(a.conus.level);
      if (conusMap && has(conusMap, level)) a.conus.level = conusMap[level];
      else if (has(discMap, level)) a.conus.level = discMap[level];
      else if (has(vertMap, level)) a.conus.level = vertMap[level];
    }
    return a;
  }

  // Rename every level by n (+1 = cranial: L5-S1 becomes L4-L5). Disc names must stay in S1..T1 (throws
  // before any change). Vertebrae and a conus level that leave the range are dropped with a warning.
  function shift(a, offset, opts) {
    var n = Math.trunc(Number(offset));
    if (!isFinite(n)) throw new Error('shift needs a whole number of levels');
    if (n === 0) return record(a, 'shift', opts, { offset: 0 }, conf(opts));
    var discMap = {}, vertMap = {}, conusMap = {}, dropped = [], conusLost = false;
    var example = discOrder(a)[0];  // read before the rename: it is the old label
    Object.keys(a.discs).forEach(function (lab) { discMap[lab] = shiftLevel(lab, n); });
    Object.keys(a.vertebrae).forEach(function (name) {
      try { vertMap[name] = shiftLevel(name, n); } catch (e) { dropped.push(name); }
    });
    if (a.conus && a.conus.level) {
      try { conusMap[String(a.conus.level)] = shiftLevel(String(a.conus.level), n); } catch (e) { conusLost = true; }
    }
    dropped.forEach(function (name) { delete a.vertebrae[name]; });
    rename(a, discMap, vertMap, conusMap);
    if (conusLost && a.conus) {
      a.conus.level = null;
      a.warnings.push('conus level dropped by a shift outside the counted range');
    }
    var detail = { offset: n, discs: discMap };
    if (dropped.length) detail.dropped_vertebrae = dropped;
    var dirWord = n > 0 ? 'up' : 'down';
    var what = 'shift all labels ' + dirWord + ' ' + Math.abs(n) + ' level' + (Math.abs(n) > 1 ? 's' : '') +
      (example ? ' (' + example + ' becomes ' + discMap[example] + ')' : '');
    return record(a, 'shift', extend(opts, { what: what }), detail, conf(opts));
  }

  // Rename chosen discs (L5-S1 -> L5-L6) or vertebrae by name. Unknown keys throw; validated first.
  function relabel(a, mapping, opts) {
    var discMap = {}, vertMap = {}, pairs = [];
    Object.keys(mapping).forEach(function (old) {
      var nu = String(mapping[old]);
      if (has(a.discs, old)) discMap[old] = nu;
      else if (has(a.vertebrae, old)) vertMap[old] = nu;
      else throw new Error('no disc or vertebra "' + old + '" to relabel');
      pairs.push(old + ' to ' + nu);
    });
    rename(a, discMap, vertMap, {});
    var detail = { mapping: {} };
    Object.keys(mapping).forEach(function (old) { detail.mapping[old] = String(mapping[old]); });
    return record(a, 'relabel', extend(opts, { what: 'rename ' + pairs.join(', ') }), detail, conf(opts));
  }

  // Remove a disc that is not a disc. Throws if absent.
  function drop(a, label, opts) {
    if (!has(a.discs, label)) throw new Error('no disc "' + label + '" to delete');
    delete a.discs[label];
    return record(a, 'drop', extend(opts, { what: 'delete disc ' + label }), { label: label }, conf(opts));
  }

  // Add a disc at centre (LPS mm). normal defaults to the nearest disc's (as Anchors.add_disc); height
  // defaults to the median of the other discs (8 mm without any); confidence defaults from the source.
  function add(a, label, center, opts) {
    splitDisc(label);
    if (has(a.discs, label)) throw new Error('disc "' + label + '" already exists');
    var c = Array.prototype.slice.call(center);
    if (c.length !== 3 || !c.every(function (v) { return typeof v === 'number' && isFinite(v); })) {
      throw new Error('center must be three finite LPS coordinates in mm');
    }
    var labs = discOrder(a), n;
    if (opts.normal === undefined || opts.normal === null) {
      if (!labs.length) throw new Error('normal is required when there are no discs to copy it from');
      var best = labs[0], bd = Infinity;
      labs.forEach(function (lab) {
        var d = a.discs[lab].center, dist = Math.hypot(d[0] - c[0], d[1] - c[1], d[2] - c[2]);
        if (dist < bd) { bd = dist; best = lab; }
      });
      n = a.discs[best].normal.slice();
    } else {
      n = Array.prototype.slice.call(opts.normal);
      if (n.length !== 3 || Math.hypot(n[0], n[1], n[2]) === 0) throw new Error('normal must be a non-zero 3-vector');
    }
    n = unit(n);
    if (n[2] < 0) n = [-n[0], -n[1], -n[2]];  // superior-pointing, as in every other disc
    var heights = labs.map(function (lab) { return a.discs[lab].height_mm; }).filter(function (h) { return !!h; });
    var height = opts.height_mm !== undefined && opts.height_mm !== null ? Number(opts.height_mm)
      : (heights.length ? median(heights) : 8.0);
    var source = opts.source || 'human';
    var defConf = source === 'human' ? opts.confHuman : source === 'ai_reader' ? opts.confAi : 0.5;
    var cf = opts.confidence !== undefined && opts.confidence !== null ? opts.confidence : defConf;  // disc, not numbering
    a.discs[label] = {
      center: c.map(Number), normal: n, anterior: null, posterior: null, height_mm: height,
      confidence: clamp01(cf),
      evidence: ['added ' + label + ': ' + (opts.why || 'no reason given') + ' (source ' + source + ')']
    };
    var detail = { label: label, center: c.map(Number) };
    return record(a, 'add', extend(opts, { what: 'add disc ' + label + ' at LPS (' + c.map(function (v) {
      return v.toFixed(1);
    }).join(', ') + ') mm' + (opts.clicked ? ' (clicked on a sagittal pane)' : '') }), detail, conf(opts));
  }

  // Reviewer confirms the current levels: a provenance entry, confidence, numbering status 'confirmed'.
  function confirm(a, opts) {
    record(a, 'confirm', extend(opts, { what: 'confirm the vertebral levels as reviewed' }), {},
      conf(opts));
    a.numbering.status = 'confirmed';
    return a;
  }

  // The exported document: a copy of the labels plus one export entry for this file. Requires a reason.
  function exportAnchors(a, opts) {
    if (!trunc(opts.why)) throw new Error('enter a reason before exporting (it is saved with the export)');
    var out = clone(a);
    var corr = (out.numbering.corrections || []).length;
    record(out, 'export', extend(opts, { what: 'exported the reviewed labels (' + corr + ' recorded change' +
      (corr === 1 ? '' : 's') + ')' }), { corrections: corr }, undefined);
    return out;
  }

  function extend(opts, more) {
    var o = {}, k;
    for (k in opts) if (has(opts, k)) o[k] = opts[k];
    for (k in more) if (has(more, k)) o[k] = more[k];
    return o;
  }

  // Tier of a confidence against the policy bands: 'high' | 'medium' | 'low'
  function tier(c, policy) {
    return c >= policy.confHigh ? 'high' : c >= policy.confLow ? 'medium' : 'low';
  }
  // True when a reader or a human reviewed the levels (a provenance record from a reviewer).
  function reviewed(a) {
    var cs = (a.numbering && a.numbering.corrections) || [];
    for (var i = 0; i < cs.length; i++) {
      if (cs[i] && (cs[i].source === 'ai_reader' || cs[i].source === 'human')) return true;
    }
    return false;
  }
  // Banner for the review panel. level: 'none' | 'confirmed' | 'probable' | 'unconfirmed' | 'ok'.
  // Same policy as anatomy/summary.py: confirmed = use the names; probable = names to be confirmed;
  // unconfirmed = positions only until a reviewer confirms.
  function banner(a, policy) {
    if (!a) return { level: 'none', text: 'No labels in this study. Automatic labelling (label_spine) gave none.' };
    var num = a.numbering || {}, c = Number(num.confidence || 0), status = num.status;
    if (status === 'confirmed') {
      return { level: 'confirmed', text: reviewed(a) ? 'Levels confirmed by a reviewer.'
        : 'Numbering status: confirmed (automatic). Use the level names.' };
    }
    if (status === 'probable') {
      return { level: 'probable', text: 'LEVELS PROBABLE (numbering confidence ' + c.toFixed(2) +
        '): the level names are to be confirmed; mark them so in any output.' };
    }
    if (c < policy.confLow || status === 'unconfirmed') {
      return { level: 'unconfirmed', text: 'LABELS UNCONFIRMED. Numbering confidence ' + c.toFixed(2) +
        ' (below ' + policy.confLow.toFixed(2) + '). Use disc positions only; check the lowest disc against the ' +
        'sacrum and confirm the levels before using their names.' };
    }
    return { level: 'ok', text: status ? 'Numbering status: ' + status + '.' : '' };
  }
  // One line for the review panel: applicability verdict, sacrum and top level in view, conus (experimental).
  function applicabilityLine(a) {
    var ap = (a && a.applicability) || {};
    if (!a || !ap.verdict) return 'Applicability: not assessed.';
    var cov = ap.coverage || {}, s = 'Applicability: ' + ap.verdict +
      (ap.labelled ? ' (levels labelled).' : ' (levels NOT labelled).');
    if (cov.sacrum_covered === true) s += ' Sacrum in view.';
    else if (cov.sacrum_covered === false) s += ' Sacrum NOT in view.';
    if (cov.top_level_in_view) s += ' Top level in view: ' + cov.top_level_in_view + '.';
    var cn = a.conus;
    s += ' Conus: ' + (cn && cn.z !== undefined && cn.z !== null ? 'near ' + (cn.level || 'unknown') : 'not located') +
      ' (experimental).';
    return s;
  }

  return {
    NAMES: NAMES, isDiscLabel: isDiscLabel, splitDisc: splitDisc, shiftLevel: shiftLevel,
    discOrder: discOrder, discsOf: discsOf, clone: clone, shift: shift, relabel: relabel, drop: drop,
    add: add, confirm: confirm, exportAnchors: exportAnchors, tier: tier, banner: banner,
    applicabilityLine: applicabilityLine
  };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = SSLabels;
