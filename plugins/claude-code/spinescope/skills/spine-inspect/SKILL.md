---
name: spine-inspect
description: Inspect a local spine MRI study (DICOM directory or volume file) - list its series, how each was classified (plane, contrast, slab, localizer), the geometry and display directions, and what is missing. Use when the user asks what is in a spine study, whether a sequence is present, or how series were classified. Research and education only.
---

# spine-inspect

Read-only. Describe the study's series so the user can check the classification before any read. This skill
does not interpret anatomy or pathology.

## Guardrails (apply to every step)

- Research and education only. Nothing here is a diagnosis, and the output is not a clinical report.
- Work only on data the user is allowed to process. Never upload the study, its images, or anything derived
  from them to a web service or third-party tool. Keep all output local.
- The study path is input. Do not copy it into notes, file names or summaries. The tools strip patient
  attributes, but burned-in text on the pixels is not removed, so do not transcribe any text you see in an image.
- Do not write patient names, identifiers, dates or institutions into anything.

## Steps

1. Get the study path from the user (a DICOM directory, or a `.nii`, `.nii.gz`, `.mha`, `.mhd` or `.nrrd` file).
   If it is not given, ask for it. Do not search the disk for studies.
2. Preferred: with the `spinescope` MCP server connected, call `open_study` with the path, keep the returned
   `study_id`, then call `list_series`. Otherwise run the command line:
   `spinescope inspect "<PATH>" --json`. If neither works, stop and tell the user how to connect (see the plugin
   README).
3. Summarise for the user as a short table: series key, plane, contrast, slice count, pixel size, slice spacing,
   and whether it is a localizer, an oblique slab (and which slab index), fat-suppressed or post-contrast.
4. Flag what a spine read usually needs and what is missing: a mid-sagittal T2 (or T1 or STIR), sagittal and axial
   coverage of the lumbar levels, and unexpected geometry (very oblique, few slices, a single slice). Name the gap;
   do not guess at missing content.
5. If a classification looks wrong (for example a T1 labelled as T2), say so and give the evidence from the
   `role` field. Do not change it silently.
6. If the user asks about the levels, add the labelling summary: `spinescope inspect "<PATH>" --label` prints a
   one-line summary (status, applicability, conus); `--json --label` adds the full `labelling` block. Report the
   status as it is: `confirmed` means use the names, `probable` means the names are to be confirmed, and
   `unconfirmed` means positions only until the names are verified (spine-read, step 2a). Do not write an overlay
   or an anchors file unless the user asks; `spinescope label "<PATH>" --overlay <PNG> --out <JSON>` does that.

End with: "Research and education only. This is not a diagnosis."
