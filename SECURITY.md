# Security policy

## Supported versions

Only the latest release and the `main` branch receive fixes.

## Reporting a security problem

Use GitHub's private vulnerability reporting for this repository (the Security tab, "Report a vulnerability").
Do not open a public issue for a vulnerability.

## Reporting a privacy problem

spinescope is built to keep patient identity out of everything it writes or returns (see
[docs/privacy.md](docs/privacy.md) for the threat model, what is removed, and what is not). A privacy problem is:

* identifying text (a name, an ID, a date, an institution, a physician, a serial number, a UID, a file or folder
  name) appearing in any output: JSON, text, HTML, PNG (pixels or text chunks), CSV, a log record or an error message;
* a DICOM attribute outside `spinescope.io.KEEP_TAGS` reaching an output;
* a free-text attribute (`SeriesDescription`, `ProtocolName`, ...) shown in an output without the protocol
  scrubbing in `spinescope.privacy`.

Please report it privately, through the same GitHub channel as above, and include:

1. the spinescope version or commit, and the command or MCP tool that produced the output;
2. the output file or message with the identifying text replaced by a placeholder, and where the text came from
   (for example "PatientName appears in brief.txt");
3. if possible, a reproduction on the synthetic phantom. `tests/_canary.py` writes a study with canaries in every text
   attribute, which is a good starting point for a minimal case.

Do not attach patient images, DICOM files, or any output that still contains real identifiers. If a real study
has leaked, say so in the report without the data, and remove the leaked output from wherever it was shared.

A fix comes with a regression case in `tests/test_privacy_canaries.py`, and the changelog says what changed.

## Scope

spinescope is research and education software. It does not diagnose. Its privacy controls limit what it reveals; they
do not make shared images anonymous. Burned-in text in pixels is not read or removed (see docs/privacy.md).
