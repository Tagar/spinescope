# spinescope Claude Code plugin

A Claude Code plugin that lets an AI assistant work on a local spine MRI study: open it, label the levels, render
and measure, run an orchestrated multi-reader structured read with a consensus report, and build the offline
viewer. It is research and education software. It does not diagnose, and its output must not be used for clinical
decisions.

## What is included

```
plugins/claude-code/
  README.md                          this file
  .claude-plugin/marketplace.json    local marketplace that lists the plugin (Option B below)
  spinescope/                        the plugin itself (load this directory)
    .claude-plugin/plugin.json       plugin manifest
    .mcp.json                        registers the spinescope-mcp server (stdio)
    skills/
      spine-read/SKILL.md            orchestrated read: briefs, reader subagents, validation, merge, verification
      spine-viewer/SKILL.md          build the offline HTML viewer and open it
      spine-inspect/SKILL.md         list series, classifications and gaps (read-only)
```

Skills are namespaced by the plugin name, so they appear as `/spinescope:spine-read`, `/spinescope:spine-viewer` and
`/spinescope:spine-inspect`. The MCP tools appear as `mcp__spinescope__<tool>`.

## Requirements

- Python 3.10 or newer, and Claude Code.
- spinescope with the MCP extra, installed in an environment whose `bin` directory is on your `PATH`:

```
python3 -m venv ~/.venvs/spinescope
~/.venvs/spinescope/bin/pip install "spinescope[mcp]"
export PATH="$HOME/.venvs/spinescope/bin:$PATH"     # or put this in your shell profile
spinescope --version
spinescope-mcp --help
```

Until spinescope is on PATH, the plugin's MCP server cannot start. `.mcp.json` runs the bare command
`spinescope-mcp`. If you prefer not to change `PATH`, edit `.mcp.json` and put the absolute path there, for example
`"command": "/absolute/path/to/venv/bin/spinescope-mcp"`.

The MCP extra pins `mcp` below 2.0, because the server uses the 1.x FastMCP API.

## Installing the plugin

Pick one of these.

### Option A: load it for one session (no install)

From a clone of the repository:

```
claude --plugin-dir /path/to/spinescope/plugins/claude-code/spinescope
```

### Option B: add it as a local marketplace

From a clone of the repository, inside Claude Code:

```
/plugin marketplace add /path/to/spinescope/plugins/claude-code
/plugin install spinescope@spinescope-local
```

The marketplace is defined in `plugins/claude-code/.claude-plugin/marketplace.json` (name `spinescope-local`). Run
`/plugin` to see what is installed and its status.

After installing, check that the MCP server is connected with `/mcp`. You should see `spinescope` with its tools.
If it is not connected, check that `spinescope-mcp` runs in a terminal (see Requirements).

## Using it

Ask in plain language, or call the skills:

- `/spinescope:spine-inspect /path/to/study`: list the series and how they were classified.
- `/spinescope:spine-viewer /path/to/study /path/to/output`: build the HTML viewer and open it.
- `/spinescope:spine-read /path/to/study`: run the orchestrated read (reader subagents, at most 4 at a time),
  validate, merge, verify the top findings, and report.

The command line does the same setup work without Claude Code:

```
spinescope inspect STUDY --json
spinescope brief STUDY --out brief.txt
spinescope tasks STUDY --outdir tasks/          # tasks.json and prompts/<reader>.txt
spinescope viewer STUDY viewer-out/
spinescope mcp --output-dir mcp-out/            # the same server the plugin starts
```

`STUDY` is a DICOM directory (any layout) or a NIfTI, MetaImage or NRRD volume.

## Rules for using it on patient data

These apply to every skill. The skills repeat them to the assistant.

- **Research and education only.** Outputs are descriptions with confidence levels. They are not diagnoses.
- **Never upload patient data anywhere.** Do not send studies, images, viewers, findings or reports to web services,
  third-party APIs, shared drives, chats or issue trackers. Keep the output on your machine.
- **Only process data you are allowed to process.** Use de-identified, public or synthetic data unless your
  institution's policy permits an AI tool to read the study.
- **PHI.** The tools never return DICOM attributes outside the technical allowlist and never return input file paths.
  They do not remove burned-in text on pixels, so check images before you share anything.
- **Testing.** The repository's tests use only the synthetic phantom and public, licensed datasets.

## How a read is organised

1. `open_study` gives an opaque `study_id`. The path is not returned.
2. `get_anchors` runs the automatic level labelling, with confidences.
3. `spinescope tasks` writes one brief and one scope per reader: each disc level, each structure across levels, a
   holistic read, and a verifier that runs after the merge.
4. Reader subagents each take one scope and write one findings file. They never see each other's output.
5. Each file is validated against the findings schema. `merge_findings` builds a consensus with disagreements listed.
6. A verifier tries to refute the top findings on the images.
7. The report states what was supported, refuted or unclear, with the limits and the research disclaimer.

## Troubleshooting

- `spinescope-mcp: command not found` in `/mcp`: the environment's `bin` is not on the PATH Claude Code sees. Use
  the absolute path in `.mcp.json`, or start Claude Code from a shell where the PATH is set.
- `spinescope mcp needs the mcp extra`: reinstall with `pip install "spinescope[mcp]"`.
- A reader reports that it has no image tools: its subagent cannot reach the MCP server. Its output must not be used.
- A level is missing from the read: check `get_anchors` for warnings and confidences. Automatic labelling fails
  loudly on unusual studies; label by hand instead of guessing.

## Development

The MCP server and the briefs are plain Python in `src/spinescope/ai/`. The plugin is a thin layer over them. To test
the server without Claude Code, run `spinescope-mcp` and speak MCP over stdio, or run the unit tests:
`pytest tests/test_ai_mcp.py tests/test_ai_briefs.py tests/test_ai_cli.py`.

Research and education only. This is not a diagnosis.
