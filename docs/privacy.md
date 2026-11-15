# Privacy: threat model, what is removed, what is not

spinescope reads MRI studies of people. Its outputs (JSON, text, HTML viewers, PNG images, findings, briefs,
logs, error messages) are often shared: with an AI service, in a GitHub issue, in a paper. This page states what
the library promises, how that promise is tested, and what remains your responsibility.

## Threat model

| Threat | Example | Covered by |
|---|---|---|
| Patient identity in DICOM text attributes | `PatientName`, `PatientID`, `OtherPatientIDs`, `PatientBirthName` | allowlist reader (`KEEP_TAGS`): never read into the model |
| Institution, staff, device | `InstitutionName`, `ReferringPhysicianName`, `OperatorsName`, `StationName`, `DeviceSerialNumber` | allowlist |
| Dates and times | `StudyDate`, `AcquisitionDate`, `PatientBirthDate` | allowlist |
| UIDs (study, series, instance) | `StudyInstanceUID` | never exposed; only a 12-character hash of the study UID (`study_id`) |
| Free text the technologist typed | `SeriesDescription`, `ProtocolName`, `SequenceName`, `ReceiveCoilName`, `ContrastBolusAgent` | scrubbed to protocol vocabulary before any output (see below) |
| Private tags | vendor blocks with names or comments | allowlist: private tags are never read |
| Input file and folder names | `/data/Smithson_Jon_1970/...`, a NIfTI named after a patient | paths never leave the library; error messages and reports show no file names |
| Burned-in text in pixels | a screenshot or annotated localizer with a name on it | **not covered**: see "Not removed" |
| Logs | debug records that quote a path or a series text | no path or free text in any log record (tested at DEBUG) |
| Exception messages | `FileNotFoundError: ... under /data/Smithson...` | messages report types and counts, not paths; the CLI replaces the input path it was given |

The phantom (`spinescope.testing.phantom`) carries fake identifiers on purpose, so tests can prove the rules.

## What is removed

* **Every DICOM attribute outside the allowlist** `spinescope.io.KEEP_TAGS`. The list is technical (modality, field
  strength, sequence, TR/TE/TI, flip angle, image type, slice thickness, coil, body part, series number...). Adding
  an attribute to it is a privacy decision: it needs a test in `tests/test_privacy_canaries.py`.
* **Private `meta` keys** (names starting with `_`, for example the source file list) are never serialised.
* **UIDs.** Study, series and instance UIDs are not in any output. The `study_id` shown to users is the first 12
  hex characters of a SHA-256 of the study UID: stable across runs, so two outputs of the same study can be
  linked by it. Treat it as a pseudonym.
* **Input paths.** No path given to `load_study`, the CLI or the MCP server is echoed back. The MCP server only
  hands out an opaque `study_id`. Rendered files are saved under names built from a counter and the `study_id`.
* **File names of findings and evidence images.** Merge reports label sources `file 1`, `file 2`, ...; evidence
  images are labelled `image <8 hex characters>` (a hash of the cited path).
* **Benchmark errors.** Path-like tokens in a benchmark case error are replaced with `<path>`.
* **PNG metadata.** Images are written without text chunks. The canary scan reads every PNG the tests produce,
  including the text chunks PIL decodes.

## Free text: the policy

Series descriptions carry real information (contrast and plane words such as `SAG T2`, and level names such as
`L4-5 OBL AX` that help number the discs). They are also the attribute a technologist is most likely to fill with
a name.

Policy: the classifier reads the raw text in memory, as it always has. **Every output** passes it through
`spinescope.privacy.public_text`, which keeps only tokens from a protocol vocabulary (contrast, plane, sequence, coil
and common protocol words), level names (`L4-L5`, `T12-L1`, `L5-S1`) and small numbers with units (`3-PL`, `3.5MM`).
Every other token becomes `*`:

| Raw `SeriesDescription` | Shown in outputs |
|---|---|
| `SAG T2 FSE` | `SAG T2 FSE` |
| `OBL AX T2 L4-5` | `OBL AX T2 L4-5` |
| `Smithson T2 STIR 20010203` | `* T2 STIR *` |
| `SAG T2 ZQCANARY_SeriesDescription` | `SAG T2 *` |

Trade-off, stated plainly:

* **Cost.** A description that is not protocol wording loses that wording in outputs. Brief, viewer and inspect
  show `*` for it. The classifier still sees the words, so labelling is unaffected.
* **Benefit.** A name, an ID, a date or a site code typed into the description cannot reach a report, a viewer or
  an AI prompt, even though the vocabulary is only as good as its list.
* **Maintenance.** When a new classifier keyword is added to `spinescope/series.py`, add it to `_VOCAB` in
  `spinescope/privacy.py`, or outputs will hide it.

`ContrastBolusAgent` is read only by the classifier (any non-empty value marks a post-contrast series) and is not
part of any output. The canary test does not put text in it, because that would turn every test series into a
post-contrast one; its value is a drug name, not an identifier.

## What is not removed

1. **Pixels.** Spinescope does not read text in images. A secondary capture or a derived export can show a name
   burned into the pixels; a rendered slice, viewer or PNG of that image shows it too. Secondary captures and colour
   screenshots are excluded from anatomy (`io/dicom.py`), but a localizer or a derived MR image is not. When a
   series is marked `DERIVED` (`ImageType`), spinescope logs a warning and adds it to `study.report`. Check annotated localizers yourself
   before sharing a rendered image. We considered a text-region detector and did not ship one: it would need its own
   validation, and it would give a false sense of safety.
2. **Anatomy and the face of the data.** Age, sex, weight, rare device serials and scan dates are quasi-identifiers.
   Only `PatientAge`, `PatientSex` and `PatientWeight` are outside the allowlist; `StudyDate` and times are not
   written, but the scanner, the field strength, the coil and the protocol together can narrow a study down. Share
   less than you think you need.
3. **Your own text.** Reader notes in findings, `--title` for the viewer, and file names you give to the outputs
   are yours. The library does not inspect them.
4. **Contrast-agent names and the sequence name in the description** are scrubbed to vocabulary, but a vendor protocol
   name that happens to be in the vocabulary is shown as is.
5. **Dataset tests and benchmark tables** report aggregate numbers only. Public datasets (SPIDER, Sudirman) are
   CC BY 4.0; do not add real patient data to the repository.

## Your responsibilities

* Treat every output of a real study as sensitive, including the images, the viewer, the JSON and the brief.
* Before sharing, look at the outputs. The viewer and the PNGs show pixels, which spinescope cannot scrub.
* Keep study folders outside the repository, and do not put patient names in folder or file names; they are
  scrubbed from outputs, but they still sit on your disk.
* Use the `--out`, `--outdir` and `--overlay` options with a directory you control.
* Sending an output to an AI service is a disclosure to that service. Check your organisation's rules first.
* Report a privacy problem as described in [SECURITY.md](../SECURITY.md).

## How it is tested

`tests/test_privacy_canaries.py` builds a study where every text-capable attribute, a private tag, the folder name
and the raw UIDs carry unique canaries (`tests/_canary.py`). It then runs every surface and scans every byte it
produces for those canaries:

* load and inspect (text and JSON), the study summary, the series table, the stack summaries;
* labelling: the anchors JSON, labelling summary, summary lines;
* analysis modules (CSV, JSON, PNG), the offline viewer (including base64-embedded thumbnails, decoded and scanned),
  the findings report (Markdown and HTML), reading briefs and reader tasks;
* every CLI command (stdout, stderr, every file written), every MCP tool (JSON, PNG pixels and text chunks, saved
  PNGs, merge output), forced errors (tool errors, CLI stderr);
* every log record at DEBUG level.

A negative control proves that the scanner fails on a planted canary, in text, in a PNG text chunk and in an
embedded image. The test runs in the default suite and in its own CI job (`.github/workflows/ci.yml`, job
`privacy`). A new output surface needs a new entry in the test; a new MCP tool fails `test_every_mcp_tool_is_covered`.

Known gaps of that test: burned-in pixel text (no OCR), numeric fields other than the listed canaries, and the
`study_id` (a deliberate hash).

## Reporting

Privacy problems: see [SECURITY.md](../SECURITY.md). Do not attach patient images to a public issue.
