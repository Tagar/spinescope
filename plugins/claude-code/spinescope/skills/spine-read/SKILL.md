---
name: spine-read
description: Orchestrated AI read of one local spine MRI study - open it, label the levels, generate the reading brief and independent reader scopes, fan out reader subagents (concurrency capped), validate each findings file, merge them into a consensus, adversarially verify the top findings, and report with the research disclaimer. Use when the user asks for an AI read, a multi-reader read, or a structured findings report for a spine MRI. Research and education only; never a diagnosis.
---

# spine-read

An orchestrated, multi-reader structured read of one spine MRI study. The orchestrator (you) does not read
images itself beyond checking reader output and, in step 2a, the level-label packet. Reader subagents each look at one scope and write a findings file.
The orchestrator validates, merges, verifies the strongest findings, and reports.

## Guardrails (apply to every step, and to every subagent you launch)

- **Research and education only.** The output describes what is visible and how confident the readers are. It is
  not a diagnosis, not a clinical report, and must not be used for clinical decisions. Say so in the final report.
- **Never upload patient data anywhere.** The study, its images, the viewer, the prompts that contain study
  material, the findings and the report stay on this machine. Do not send them to a web service, a third-party
  API, a paste site, a shared drive, a chat or an issue tracker. Do not attach them to a message. Use only the
  spinescope tools and local files.
- **Only process data the user is allowed to process.** If the study is not de-identified, or the user has not
  said that their policy allows an AI tool to read it, stop and ask before reading.
- **PHI rules.** Output from the spinescope tools has no patient attributes and no file paths, except the opaque
  `study_id`. Do not write patient names, identifiers, dates, institutions, physicians, accession numbers or file
  paths of the input into any file, prompt, log, subagent message or summary. Burned-in text on the image pixels
  is not removed by the tools; do not transcribe it. If you see something that looks like identifying text on an
  image, tell the user and leave it out of the output.
- **Subagents follow the same rules.** Tell every subagent this section in its prompt.
- **Do not overwrite** an existing output directory without the user's go-ahead. Use a new directory per study.

## Inputs and outputs

- `STUDY`: the DICOM directory or volume file the user gives you.
- `OUT`: a new, empty output directory for this read (default: `./spinescope-read-<short id>` in the current
  working directory; say where it is). Layout: `OUT/tasks/` (index and prompts), `OUT/findings/` (one JSON per
  reader), `OUT/verify/` (verifier output), `OUT/consensus/` (`consensus.json`, `report.md`).

## Steps

1. **Check the environment.** Confirm the spinescope MCP tools are available (`open_study` and the others). If
   they are not, the command line still runs the setup steps, but readers cannot look at images, so do not
   continue to step 4 without them. Tell the user how to connect (see the plugin README) and stop.

2. **Open and label.** Call `open_study` with `STUDY`. Keep the `study_id`. Call `list_series`; check the series
   and their roles. Call `get_anchors`; note the levels and their confidences and any warnings. If there are no
   sagittal series or no anchors, say so: the read then has no level tasks.
   Read `labelling` and `summary` from the response: `labelling.status` (`confirmed`, `probable` or `unconfirmed`),
   the abstain policy, the applicability verdict, the field-of-view coverage (is the sacrum in view?) and the conus
   line (always experimental). Carry these into the brief and the final report.

## Abstain policy for the level names (applies to every step and every reader)

The numbering status decides what a reader may do with the level names:

- `confirmed`: use the level names.
- `probable`: use the level names, each marked "to be confirmed" in its notes.
- `unconfirmed`: positions only. No level name goes into any finding until a HUMAN reviewer has confirmed the
  numbering (viewer correction tools or `apply_label_verification` with source `human`). An AI review can never
  raise the status (see 2a). If it stays `unconfirmed`, keep the disc positions and say so in notes.

## 2a. Verify the level labels before fan-out

The automatic labels are often wrong: the count from the sacrum can be off by one level, and a disc can be missed
or invented. Readers take their level names from the prompts, so verify the labels first. You are the reader for
this step.

**What this step can and cannot do (measured):** in a blinded experiment on 196 public cases
(docs/labeling/lab24_ai_verify.md) an AI reader fixed 0 of 49 wrong numberings and miscorrected 0 of 147 right
ones, while flagging some cases as unsure. So an AI review is a way to surface doubt, not to confirm names:
applying an AI answer can only LOWER the status/confidence (an AI-proposed relabel is kept as a suggestion and the
status becomes `unconfirmed`). Only a human review can raise it.

**When to run it:** optional. Run it when the status is `probable` or `unconfirmed` and you want a second look that
may downgrade the status; never present an AI "confirmation" as confirmation in the report.

1. Build the packet: `spinescope verify-labels "<STUDY>" --outdir "<OUT>/verify/packet"`. (MCP: `verify_labels`
   returns the images and the question inline, with the status and its policy in its metadata.) It writes `packet.json` (question, rules, answer format, predicted
   levels) and PNGs: the mid-sagittal T2 and T1 with the predicted discs, the localizer with the predicted levels
   when there is one, and a zoom on the sacrum below the lowest disc.
2. Look at each PNG (Read it). Count from the sacrum upward, as the question in `packet.json` asks. Do not
   transcribe any text burned into the pixels.
3. Write `OUT/verify/answer.json` with exactly these keys: `lowest_disc_is` (`"L5-S1"`, `"L4-L5"`, `"L5-L6"` or
   `"unsure"`), `missing` (true names of discs with no label), `extra` (labels as shown that are not discs),
   `transitional` (`true`, `false` or `null`) and `notes` (one sentence). `lowest_disc_is` is the true name of the
   lowest disc shown after you remove the extras: a disc marked L5-S1 that is really L4-L5 gets `"L4-L5"`.
   Answer `"unsure"` when the sacrum is not visible or the count is doubtful; an unsure answer changes nothing.
   A body below the lowest label does not prove that the label is a disc. Check that a disc signal lies between
   the label and the body, and that the label is not on bowel or muscle. When in doubt, answer `"unsure"`.
4. Apply it: `spinescope verify-labels "<STUDY>" --answer "<OUT>/verify/answer.json" --out "<OUT>/verify/anchors.json"`
   (MCP: `apply_label_verification(study_id, answer_json)`). A rejected answer says why; fix the file and rerun.
   The response carries the new `status`. An AI answer is at most `probable`, and `unsure` keeps the status
   `unconfirmed`. Give the readers that new status, not the automatic one, when you set the policy in fan-out step 4.
   Missing discs are placed between their neighbours and marked as not measured, with low confidence.
5. Every later step uses the verified anchors. Pass `--anchors "<OUT>/verify/anchors.json"` to `spinescope tasks`,
   `brief` and `viewer`. Subagents open the study with `open_study(STUDY, anchors_path="<OUT>/verify/anchors.json")`.
6. In the final report, state the label review: the lowest-disc answer, any discs dropped or placed, the resulting
   numbering confidence and the warnings.

3. **Generate the briefs and reader scopes.** Run the CLI so the prompts are files that subagents can read. Use
   the verified anchors from step 2a:
   `spinescope tasks "<STUDY>" --outdir "<OUT>/tasks" --anchors "<OUT>/verify/anchors.json"`
   Without `--anchors` the CLI labels the study again and the prompts use the automatic labels.
   This writes `OUT/tasks/tasks.json` (one entry per reader: `id`, `kind`, `reader`, `levels`, `groups`,
   `series`, `depends_on`) and `OUT/tasks/prompts/<reader>.txt` (the complete brief plus that reader's scope and
   the findings format). Read `tasks.json`. The kinds are:
   - `level`: one disc level, all its fields (independent).
   - `structure`: one structure across all levels (independent).
   - `holistic`: global sections and the array sections (independent).
   - `verify`: the adversarial verifier. It depends on `merge` and runs in step 7, not now.

   Use the `spinescope` CLI from the same Python environment that has spinescope installed (the one the MCP
   server runs in). `which spinescope` shows where it is. If the CLI is missing, use `brief()` and `tasks()` from
   the MCP server instead and write the prompts to files yourself.

4. **Fan out the independent readers, with a concurrency cap.** Launch one subagent per independent task (every
   `kind` other than `verify`). Run at most **4 at a time**: send up to 4 Agent calls in one message, wait for
   all of them, then send the next batch. Use a general-purpose subagent. Each subagent prompt must:
   - Include the guardrails above (research only, no upload, PHI rules).
   - Include the abstain policy for the study's status (from `labelling.status`), in these words: confirmed = use the
     level names; probable = use them, each marked "to be confirmed"; unconfirmed = positions only, and call
     `verify_labels` before writing any level name. A reader that cannot call `verify_labels` must not write level
     names for an unconfirmed study.
   - Give the path of its prompt file, `OUT/tasks/prompts/<reader>.txt`, and say: read it in full and follow it
     exactly. Its reader name is `<reader>`, and it must write exactly one file, `OUT/findings/<reader>.json`.
   - Give `STUDY` and the verified anchors file `OUT/verify/anchors.json`, and say: open it with
     `open_study(STUDY, anchors_path=...)` to get its own `study_id` (do not reuse another reader's),
     and use the spinescope MCP tools to look at the images it needs. Each image it relies on must be a PNG it
     actually rendered and cites as evidence.
   - Say: validate the file before stopping, using the check in the brief (or `validate_findings`), and fix every
     error. Only a valid file counts.
   - Say: return one line only, in the form `<reader>: valid|invalid, <n> fields filled, <m> null`. Do not return
     the JSON or any image content.

   A reader with no MCP tools cannot look at images. If a subagent reports that it has no image tools, do not
   accept fields from it; rerun that reader later, or drop it and say so in the report.

5. **Validate and collect.** For each `OUT/findings/*.json` run:
   `python -c "from spinescope.report import validate; r = validate('<FILE>'); print(r.ok); print(*r.error_messages(), sep='\n')"`
   Rerun any invalid reader once with the errors; drop it if it still fails, and record that in the report.

6. **Merge to a consensus.** Call `merge_findings` with the valid files and `outdir = OUT/consensus`. Or run:
   `python -c "from spinescope.report import merge, render_markdown; c = merge('<OUT>/findings'); c.to_json('<OUT>/consensus/consensus.json'); open('<OUT>/consensus/report.md', 'w').write(render_markdown(c))"`
   Read the disagreements and the indeterminate list from `consensus.json`. Do not average or override a reader's
   value by hand. Record disagreements as they are.

7. **Adversarial verification of the top findings.** Pick the top 10 items from `consensus.json`: first the
   disagreements (largest spread first), then the highest-confidence positive findings with a clinical or
   research-relevant meaning (for example central canal stenosis, nerve root contact, Modic type, listhesis).
   Write the list to `OUT/verify/targets.json`. Launch **one** verifier subagent (the `verify` task, prompt
   `OUT/tasks/prompts/verify_1.txt`) with the path of `targets.json`, the consensus, the study, and the instruction
   to try to refute each target on the images, writing `OUT/verify/verify.json`. Check it with:
   `python -c "import json; from spinescope.ai.briefs import verification_errors as v; d = json.load(open('<OUT>/verify/verify.json')); print(v(d) or 'ok')"`
   Fix or rerun once if it fails.

8. **Final report.** Give the user, in chat:
   - The study in one line (technical only: modality, field strength, number of series, levels found) and the
     `OUT` path. No file paths of the input, no patient attributes.
   - The labelling status (`confirmed`, `probable` or `unconfirmed`), the abstain policy it implied, the
     applicability verdict, whether the sacrum is in view, and that the conus is experimental.
   - The reader count: launched, valid, dropped, and why.
   - The consensus: the main findings per level and the global findings, each with agreement across readers and
     the confidence. Show the disagreements explicitly.
   - The verification: for each target, `supported`, `refuted` or `unclear`, with its confidence and a one-line reason.
   - Limits: automatic level labels and their confidence, missing sequences, readers that were dropped, anything
     the readers could not assess.
   - The path to `OUT/consensus/report.md`.

   End with this disclaimer, verbatim: "Research and education only. This AI-assisted read is not a diagnosis,
   has not been reviewed by a clinician, and must not be used for clinical decisions."

## Things not to do

- Do not give a diagnosis, a treatment recommendation or a prognosis, even if a finding looks important.
- Do not launch more than 4 reader subagents at once.
- Do not let a reader see another reader's findings; each works from its own prompt only.
- Do not upload, paste or attach study material anywhere outside this machine.
- Do not overwrite a previous `OUT` directory.
