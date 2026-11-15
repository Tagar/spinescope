---
name: spine-viewer
description: Build the offline spinescope HTML viewer for a local spine MRI study (every series, in-browser reformats, disc lines) and open it for the user. Use when the user wants to look at a spine MRI study interactively or asks for a viewer. Research and education only.
---

# spine-viewer

Builds one HTML file that the user opens in a browser. The file works offline and embeds the study's image pixels,
so treat it as patient data.

## Guardrails (apply to every step)

- Research and education only. The viewer shows images; it does not diagnose.
- Never upload the viewer, the study, or a screenshot of it to a web service, shared drive or chat. Keep the output
  local, and do not put it in a synced folder unless the user's policy allows that.
- Use an output directory the user chooses. Do not put it in a repository or a shared location.
- The page title must not contain patient information. Use a neutral title such as "Study A", or leave it empty.

## Steps

1. Get the study path (a DICOM directory or a volume file) and an output directory from the user. If no output
   directory is given, use `./spinescope-viewer-out` in the current working directory and say so.
2. Build the viewer. With the `spinescope` MCP server connected, call `build_viewer` with the `study_id` from
   `open_study` and the output directory. Otherwise run:
   `spinescope viewer "<PATH>" "<OUTDIR>" --title "<neutral title>"`
   Add `--no-anchors` to skip automatic level labelling if it is slow or not needed.
3. Open the file for the user: `open "<OUTDIR>/spinescope_viewer.html"` on macOS, or `xdg-open` on Linux. If that
   fails, give the user the path to open in a browser.
4. Tell the user what the viewer contains: every series; planes with no native series are computed in the browser
   and labelled as reformats; disc lines appear only when level labelling ran. The review panel shows the numbering
   banner (confirmed, probable or unconfirmed), the applicability verdict, whether the sacrum is in view, and the
   conus line, which is experimental. Say that an unconfirmed banner means the level names must not be used yet.

End with: "Research and education only. This is not a diagnosis."
