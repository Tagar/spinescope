# spinescope

Geometry-aware spine MRI analysis for humans and AI agents.

> **Research and education only. Not a medical device. Not for diagnosis or treatment decisions.**
> Outputs can be wrong. Always rely on a qualified radiologist's report.

spinescope loads a spine MRI study from any DICOM folder (or NIfTI / MetaImage volumes), works out
what every series is (sagittal T2, axial T1, STIR, angled slabs, localizers...), labels the vertebral
levels automatically, and links all series in patient space so the same point can be shown in every
image. On top of that it provides measurements, per-level reading sheets, an offline browser viewer
with 3-plane cross-referencing, a structured findings format with multi-reader consensus, and tools for
AI assistants (an MCP server and Claude Code skills) to perform and audit structured reads.

## Status

Alpha. The API will change. See [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
pip install "spinescope @ git+https://github.com/Tagar/spinescope"          # core
pip install "spinescope[all] @ git+https://github.com/Tagar/spinescope"     # + JPEG DICOM, NIfTI/MHA, MCP
```

## Quick start

```python
import spinescope as ss

study = ss.load_study("path/to/dicom_folder")
print(study.summary())  # sag_t2, sag_t1, sag_stir, ax_t2, ax_t2_slab1..5, loc_sag ...
sag = study["sag_t2"]
xyz = sag.pix_to_lps(7, 256, 256)  # pixel -> patient LPS mm
print(study["ax_t2"].nearest_slice(xyz))
```

```bash
spinescope inspect path/to/dicom_folder
```

## Coordinates

All geometry is DICOM patient LPS in millimetres (+x patient left, +y posterior, +z head). Images are
re-oriented for display: sagittal shows anterior on the left and head at the top; axial and coronal
show the patient's left on the right of the screen (radiological convention).

## Privacy

spinescope keeps only an allowlist of technical DICOM attributes (`spinescope.io.KEEP_TAGS`); patient
names, IDs, dates, institutions and physicians never enter its data model or outputs. Still treat
every output derived from a real patient's images as sensitive, and never commit patient data.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
