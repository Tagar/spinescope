# lab19_surface: labelling on every user-facing surface

Branch `lab/lab19_surface`. Owner: the surfaces, not the labeller. Files: `anatomy/summary.py` (new, the one
wording), `cli.py` (`label`, `inspect --label`), `ai/mcp_server.py` (status fields), `ai/briefs.py` (status and
abstain policy in briefs and reader scopes), `viewer/assets/` (banner, applicability line), the three Claude Code
skills, `data/anchors.schema.json` (schema 1.1, validated in `tests/test_anchors_schema.py`).

Shared-file edits: `anatomy/verify.py` (`review_status`: a reviewed correction now sets the numbering status, so the
status matches the applicability record; an AI answer is at most `probable`, `unsure` keeps `unconfirmed`), and
`pyproject.toml` (`jsonschema` in the dev extra).

## The abstain policy (one definition, quoted everywhere)

| status | level names | what the surfaces say |
|---|---|---|
| confirmed | use them | banner "LEVELS CONFIRMED" (automatic, or by a reviewer) |
| probable | use them, each marked "to be confirmed" | banner "LEVELS PROBABLE" |
| unconfirmed | positions only; verify first (`verify_labels`, or `verify-labels --answer`) | banner "LEVELS UNCONFIRMED"; briefs list positions and the automatic name, flagged |

The conus is experimental on every surface (`conus: ... (EXPERIMENTAL, not validated)`).

## Where each fact appears

- CLI `label`: banner, policy, applicability, field of view, conus, lumbosacral check, votes, levels table, warnings.
- CLI `inspect --label`: one line; `--json` adds the `labelling` block.
- MCP `get_anchors`: all anchor fields plus `labelling` (structured) and `summary` (text).
- MCP `verify_labels`: `status`, `status_text`, `policy`, `applicability`, `field_of_view` in the metadata.
- MCP `apply_label_verification`: the same fields, after the answer. An answer that cannot be checked against the
  bone (sacrum not in view) stays `unconfirmed`.
- Briefs and `tasks`: a "LEVEL NAMING" line at the top, a status block in section 3, and the policy in every scope
  that names levels.
- Viewer review panel: banner (confirmed automatic / reviewed, probable, unconfirmed) and an applicability line.

## Tests

`tests/test_labelling_surfaces.py` (CLI, MCP, briefs, summary, review status, skills),
`tests/test_labelling_surfaces_viewer.py` (banner and applicability under node),
`tests/test_anchors_schema.py` (schema 1.1 against learned, classical, reviewed, truth and 1.0 documents; every
dataclass field declared; bad documents refused). All run on the phantom; the identifying phantom values and the
input path are asserted absent from every output.
