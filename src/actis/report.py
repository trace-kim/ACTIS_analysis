"""Build a portable, offline comparison workspace for a reference study."""

import json
from html import escape
from importlib.resources import files
from string import Template
from time import perf_counter

from .files import native
from .report_data import write_viewer_data
from .report_details import details_views


def measurement_notice(study, frames):
    if not study.regions:
        discovery = getattr(study, "discovery", {})
        count = discovery.get("candidate_centers")
        candidates = (
            f"Discovery found {count:,} candidate bright centers in acquisition 01. "
            if count is not None
            else ""
        )
        return (
            candidates
            + "No periodic region qualified. The hole fits and region-based tiles were skipped. "
            "Whole-acquisition brightness is available for every input."
        )
    if not any(
        frame["qualified"][method]
        for frame in frames
        for method in ("contour", "model", "differential", "flux")
    ):
        return (
            "No per-hole measurement qualified across these acquisitions. "
            "Select a method to see its coverage and exclusion reasons."
        )
    return ""


def make_report(
    out,
    images,
    info,
    study,
    summary,
    splits,
    spatial,
    stress,
    comparisons,
    performance,
    synthetic,
    synthetic_settings=None,
    viewer_data=None,
):
    start = perf_counter()
    manifest = (
        viewer_data
        if viewer_data is not None
        else write_viewer_data(out, images, info, study)
    )
    notice = measurement_notice(study, manifest["frames"])
    precision, diagnostics = details_views(
        study,
        info,
        summary,
        splits,
        spatial,
        stress,
        comparisons,
        performance,
        synthetic,
        assets=out / "images",
    )
    if synthetic_settings:
        s = synthetic_settings
        provenance = (
            '<section class="detail-card"><h3>Simulated acquisition geometry</h3>'
            f"<p>The generator used {escape(str(s['shape']))} openings with a nominal "
            f"CD of {s['cd_nm']:g} nm, a {escape(str(s['lattice']))} lattice, "
            f"{s['pitch_px']:g} native pixels per period and a {s['angle_deg']:g}° rotation. "
            "The acquisition pitch, orientation, optical blur and detector response are simulation settings. "
            "Lasertec published actinic review images of programmed 80 nm square contacts in Figure 10. "
            '<a href="https://www.spie.org/documents/Membership/BacusNewsletters/BACUS-Newsletter-January-2020.pdf#page=6">'
            "Miyai et al., Figure 10</a></p></section>"
        )
        diagnostics = provenance + diagnostics

    # A JSON script element still recognizes the HTML end tag inside a JSON string.
    embedded = json.dumps(
        native(manifest), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    embedded = (
        embedded.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    )
    resources = files("actis")
    for asset in ("report.css", "report.js"):
        (out / asset).write_bytes(resources.joinpath(asset).read_bytes())
    template = Template(resources.joinpath("report.html").read_text(encoding="utf-8"))
    page = template.substitute(
        manifest=embedded,
        provenance="Synthetic demo" if synthetic else "Reference data",
        site_count=f"{len(manifest['sites']):,}",
        frame_count=len(images),
        frame_size=f"{manifest['width']:,} × {manifest['height']:,}",
        pixel_nm=f"{study.pixel_nm:g}",
        precision=precision,
        diagnostics=diagnostics,
        qualification_notice=(
            '<div class="detail-card" role="status"><p>'
            + escape(notice)
            + ' <a href="#run">Run details</a></p></div>'
            if notice
            else ""
        ),
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    return perf_counter() - start
