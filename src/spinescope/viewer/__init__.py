"""Offline HTML viewer: every series of a study, 1-4 panes, in-browser reformats, 3D cursor.

Example::

    from spinescope import load_study
    from spinescope.viewer import build_viewer
    path = build_viewer(load_study("study_dir"), "out/", anchors=None, title="Study 1")
"""

from .build import build_viewer, render_html

__all__ = ["build_viewer", "render_html"]
