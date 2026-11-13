"""Figs. 8 and 9 as vector SVG, from the PDFs that the steps fig8 and fig9
write (results/figures/r29_fig8_fidelity_h160.pdf, r29_fig9_glassbox.pdf).

1. PyMuPDF's SVG of the PDF page: fitz.open(pdf)[0].get_svg_image(
   text_as_path=True), the text drawn as glyph outlines.
2. Masks as clip paths. Word's PDF export draws an SVG <mask> as a raster.
   Every mask here must be an opaque white shape, and such a mask equals a
   clip to that shape. So each <mask id=X> becomes <clipPath id=X> with the
   same children, and each mask="url(#X)" becomes clip-path="url(#X)". Any
   other kind of mask stops the script.
3. A larger user space. Word draws a stroke at least about 1.16 user units
   wide. So the content is wrapped in <g transform="scale(8)">, and the
   width, height and viewBox of the root are multiplied by 8. The drawing is
   the same, but each stroke is 8 times more user units wide.

Writes results/svg/<name>.svg and prints the SHA-256 of each, and whether it
equals the file that was there before (the shipped copy). Needs PyMuPDF
(pinned in requirements.txt).

    python fig_vector.py
"""
import hashlib
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "results" / "figures"
OUT_DIR = ROOT / "results" / "svg"
NAMES = ("r29_fig8_fidelity_h160", "r29_fig9_glassbox")
SCALE = 8.0


def path_bbox(d):
    """bounding box of an axis-aligned path (M, L, H, V, Z)"""
    xs, ys = [], []
    for cmd, args in re.findall(r"([MHVLZ])([^MHVLZ]*)", d):
        v = [float(x) for x in re.findall(r"-?\d*\.?\d+", args)]
        if cmd in "ML":
            xs += v[0::2]
            ys += v[1::2]
        elif cmd == "H":
            xs += v
        elif cmd == "V":
            ys += v
    return min(xs), min(ys), max(xs), max(ys)


def conv_mask(m, count):
    head, body = m.group(1), m.group(2)
    mid = re.search(r'id="([^"]+)"', head).group(1)
    attrs = dict(re.findall(r'([\w:-]+)="([^"]*)"', head))
    assert set(attrs) <= {"id", "fill", "style", "maskUnits", "x", "y", "width", "height"}, attrs
    assert attrs.get("style", "mask-type:luminance") == "mask-type:luminance", attrs
    kids = re.findall(r"<(\w+)\b([^>]*)/>", body)
    assert kids and re.sub(r"<(\w+)\b[^>]*/>", "", body).strip() == "", body[:200]
    for tag, ka in kids:
        assert tag in ("path", "rect"), tag
        assert "opacity" not in ka, ka
        fill = re.search(r'fill="([^"]*)"', ka)
        fill = fill.group(1) if fill else attrs.get("fill")
        assert fill and fill.lower() in ("white", "#fff", "#ffffff"), (mid, fill)
    if attrs.get("maskUnits") == "userSpaceOnUse":
        x, y, w, h = (float(attrs[k]) for k in ("x", "y", "width", "height"))
        for tag, ka in kids:
            x0, y0, x1, y1 = path_bbox(re.search(r'd="([^"]*)"', ka).group(1))
            assert x <= x0 and y <= y0 and x1 <= x + w and y1 <= y + h, (mid, (x0, y0, x1, y1), (x, y, w, h))
    count[0] += 1
    return f'<clipPath id="{mid}">{body}</clipPath>'


def fix(s, scale):
    """masks to clip paths, then the user space scaled by `scale`"""
    count = [0]
    s2 = re.sub(r"<mask\b([^>]*)>(.*?)</mask>", lambda m: conv_mask(m, count), s, flags=re.S)
    for el in re.findall(r"<\w+\b[^>]*\bmask=\"url\(#[^)]+\)\"[^>]*>", s2):
        assert "clip-path=" not in el, el[:200]
    s2, n_refs = re.subn(r'\bmask="url\(#([^)]+)\)"', r'clip-path="url(#\1)"', s2)
    assert "<mask" not in s2 and "mask=" not in s2.replace("clip-path=", "")
    root = re.search(r"<svg\b[^>]*>", s2)
    tag = root.group(0)
    w = float(re.search(r'\bwidth="([\d.]+)"', tag).group(1))
    h = float(re.search(r'\bheight="([\d.]+)"', tag).group(1))
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', tag).group(1).split()]
    assert vb[:2] == [0.0, 0.0] and abs(vb[2] - w) < 1e-9 and abs(vb[3] - h) < 1e-9, (w, h, vb)
    fmt = lambda v: f"{v:.6g}"
    tag2 = re.sub(r'\bwidth="[\d.]+"', f'width="{fmt(w * scale)}"', tag, count=1)
    tag2 = re.sub(r'\bheight="[\d.]+"', f'height="{fmt(h * scale)}"', tag2, count=1)
    tag2 = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {fmt(w * scale)} {fmt(h * scale)}"', tag2, count=1)
    end = s2.rindex("</svg>")
    s3 = s2[:root.start()] + tag2 + f'<g transform="scale({fmt(scale)})">' + s2[root.end():end] + "</g>" + s2[end:]
    ET.fromstring(s3.encode("utf8"))                     # still well-formed XML
    return s3, count[0], n_refs


def main():
    print(f"PyMuPDF {fitz.VersionBind}, MuPDF {fitz.VersionFitz}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in NAMES:
        pdf = PDF_DIR / f"{name}.pdf"
        if not pdf.is_file():
            print(f"{pdf} is missing: run the steps fig8 and fig9 first")
            return 1
        svg = fitz.open(pdf)[0].get_svg_image(text_as_path=True)
        svg = svg.replace("\r\n", "\n").replace("\r", "\n")
        s3, n_masks, n_refs = fix(svg, SCALE)
        out = OUT_DIR / f"{name}.svg"
        before = hashlib.sha256(out.read_bytes()).hexdigest() if out.is_file() else None
        out.write_text(s3, encoding="utf8", newline="\n")
        now = hashlib.sha256(out.read_bytes()).hexdigest()
        same = before == now
        ok &= before is None or same
        print(f"{out.name}: masks to clip paths {n_masks} (refs {n_refs}), {out.stat().st_size:,} bytes, "
              f"sha256 {now}; equal to the file that was there: {same if before else 'no file before'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
