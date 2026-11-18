"""Stroke-to-outline conversion ("stroke to path").

Cutting machines and similar consumers need a fillable solid outline, not
a centerline plus a ``stroke-width``.  This module computes, for a given
centerline `Path` and stroke width, closed `Path` objects tracing the
boundary of the stroked region.

The main entry point is `stroke_outline`.

Notes on the geometry
---------------------
* Curved segments (quadratic/cubic Beziers and arcs) are flattened to
  polylines by adaptive subdivision before offsetting, so curve joins
  inherit the (stable) polyline join behavior.
* Round line joins and round/square line caps are emitted as true `Arc`
  segments; everything else is emitted as `Line` segments.
* Self-intersecting centerlines (e.g. hairpins, figure-eights) produce
  well-defined, possibly self-intersecting outlines; no boolean union is
  performed.  Filled with the nonzero rule they cover the stroked region.
"""

from math import isfinite

from .path import Path, Line, Arc

VALID_CAPS = ("butt", "round", "square")
VALID_JOINS = ("miter", "round", "bevel")

# |sin(theta)| between consecutive unit tangents below this is "parallel".
_PARALLEL_TOL = 1e-9
# |sin(theta)| below this (~0.57 degrees) is treated as a straight join;
# this keeps flattened curves from sprouting a join artifact per vertex.
_STRAIGHT_TOL = 0.01


def stroke_outline(path, width, cap="butt", join="miter", miter_limit=4.0,
                   tolerance=None, max_depth=12):
    """Return closed `Path`s outlining the region a stroke of `path` covers.

    Parameters
    ----------
    path : Path
        The centerline.  It is never modified.  Each continuous subpath is
        stroked independently.
    width : float
        Stroke width.  Must be positive and finite; zero or negative
        widths raise `ValueError`.
    cap : {"butt", "round", "square"}
        Line cap style used at the ends of open subpaths.
    join : {"miter", "round", "bevel"}
        Line join style used at vertices (including the seam vertex of
        closed subpaths).
    miter_limit : float
        As in SVG: the miter is clipped to a bevel when the miter length
        exceeds ``miter_limit * width``.  Must be >= 1.
    tolerance : float, optional
        Flatness tolerance used when flattening curved segments.  Defaults
        to 0.1% of ``min(path size, width)`` so the outline error stays
        small relative to the stroke width.
    max_depth : int
        Maximum subdivision depth used while flattening.  If a curve does
        not converge within this depth, the chord of the remaining piece
        is used (a stable, documented fallback -- no exception is raised).

    Returns
    -------
    list of Path
        Closed outlines.  Each open subpath yields one closed outline;
        each closed subpath yields two (the outer and the inner boundary
        loop).  Degenerate input (empty path, or subpaths with no extent)
        yields fewer or no outlines; an empty list is a valid result.

    Raises
    ------
    ValueError
        For non-positive/non-finite `width`, unknown `cap`/`join` styles,
        ``miter_limit < 1``, non-positive `tolerance`, or non-finite
        coordinates in `path`.
    """
    if cap not in VALID_CAPS:
        raise ValueError("cap must be one of %r, got %r" % (VALID_CAPS, cap))
    if join not in VALID_JOINS:
        raise ValueError("join must be one of %r, got %r"
                         % (VALID_JOINS, join))
    if not (isfinite(miter_limit) and miter_limit >= 1):
        raise ValueError("miter_limit must be a finite number >= 1, got %r"
                         % (miter_limit,))
    try:
        width = float(width)
    except (TypeError, ValueError):
        raise ValueError("width must be a positive, finite number, got %r"
                         % (width,))
    if not isfinite(width) or width <= 0:
        raise ValueError("width must be a positive, finite number, got %r"
                         % (width,))
    if tolerance is not None and not tolerance > 0:
        raise ValueError("tolerance must be positive, got %r" % (tolerance,))
    max_depth = max(1, int(max_depth))

    if len(path) == 0:
        return []
    for seg in path:
        for z in (seg.start, seg.end):
            if not (isfinite(z.real) and isfinite(z.imag)):
                raise ValueError("path contains non-finite coordinates: %r"
                                 % (z,))

    xmin, xmax, ymin, ymax = path.bbox()
    scale = max(xmax - xmin, ymax - ymin, width)
    if scale == 0:  # all segments are points; nothing to stroke
        return []
    tol = tolerance if tolerance is not None else 1e-3 * min(scale, width)
    eps = 1e-9 * scale  # zero-length / duplicate-point threshold
    h = width / 2.0

    outlines = []
    for subpath in path.continuous_subpaths():
        pts = _flatten_subpath(subpath, tol, max_depth, eps)
        if len(pts) < 2:
            continue
        closed = abs(pts[-1] - pts[0]) <= eps
        if closed:
            pts.pop()  # drop the duplicated closing point
            if len(pts) < 3:
                closed = False  # degenerate ring; stroke it as an open line
        if closed:
            dirs = _unit_vectors(pts, closed=True)
            loops = [_left_chain_closed(pts, dirs, h, join, miter_limit)]
            rpts = pts[::-1]
            rdirs = _unit_vectors(rpts, closed=True)
            loops.append(_left_chain_closed(rpts, rdirs, h, join,
                                            miter_limit))
            for items in loops:
                outline = _items_to_path(items, h, eps)
                if outline is not None:
                    outlines.append(outline)
        else:
            dirs = _unit_vectors(pts, closed=False)
            items = _open_outline_items(pts, dirs, h, cap, join, miter_limit)
            outline = _items_to_path(items, h, eps)
            if outline is not None:
                outlines.append(outline)
    return outlines


# Flattening ##################################################################

def _segment_vanishes(seg, eps):
    """True if `seg` has no extent (a zero-length segment, up to `eps`)."""
    if abs(seg.end - seg.start) > eps:
        return False
    xmin, xmax, ymin, ymax = seg.bbox()
    return (xmax - xmin) <= eps and (ymax - ymin) <= eps


def _flatten_subpath(subpath, tol, max_depth, eps):
    """Flatten a continuous subpath to a list of distinct points."""
    pts = []
    for seg in subpath:
        if _segment_vanishes(seg, eps):
            continue  # skip zero-length segments
        if not pts:
            pts.append(seg.start)
        _flatten_segment(seg, tol, max_depth, pts)
    clean = []
    for p in pts:
        if not clean or abs(p - clean[-1]) > eps:
            clean.append(p)
    return clean


def _flatten_segment(seg, tol, depth, out):
    """Append points of `seg` (excluding its start) to `out`."""
    if depth <= 0 or _is_flat(seg, tol):
        # Depth exhausted without converging: fall back to the chord.
        out.append(seg.point(1))
        return
    first, second = seg.split(0.5)
    _flatten_segment(first, tol, depth - 1, out)
    _flatten_segment(second, tol, depth - 1, out)


def _is_flat(seg, tol):
    if isinstance(seg, Line):
        return True
    p0 = seg.start
    chord = seg.end - p0
    chord_len = abs(chord)
    for t in (0.25, 0.5, 0.75):
        w = seg.point(t) - p0
        if chord_len == 0:
            dist = abs(w)
        else:
            dist = abs(chord.real * w.imag - chord.imag * w.real) / chord_len
        if dist > tol:
            return False
    return True


# Offset polyline construction #################################################

def _unit_vectors(pts, closed):
    n = len(pts)
    count = n if closed else n - 1
    dirs = []
    for i in range(count):
        d = pts[(i + 1) % n] - pts[i]
        dirs.append(d / abs(d))
    return dirs


def _offset_intersection(a, din, b, dout):
    """Intersection of the line through `a` along `din` with the line
    through `b` along `dout`; None if (numerically) parallel."""
    denom = din.real * dout.imag - din.imag * dout.real
    if abs(denom) < _PARALLEL_TOL:
        return None
    w = b - a
    s = (w.real * dout.imag - w.imag * dout.real) / denom
    return a + s * din


def _join_items(p, din, dout, h, join, miter_limit):
    """Outline items on the left-of-travel side at the vertex `p` whose
    incoming/outgoing unit tangents are `din`/`dout`.

    Items are ("pt", z) for a polyline vertex and ("arc", z) for a
    circular arc of radius `h` ending at `z` (always sweep=False, i.e.
    clockwise, which is the only direction outer joins/caps ever need).
    """
    a = p + h * din * 1j
    b = p + h * dout * 1j
    cross = din.real * dout.imag - din.imag * dout.real
    if abs(cross) < _STRAIGHT_TOL:
        dot = din.real * dout.real + din.imag * dout.imag
        if dot > 0:
            return [("pt", a)]  # (nearly) straight: no join artifact
        # 180-degree reversal: both sides wrap the tip.  A miter here
        # would sit at infinity, so miter degrades to a bevel.
        if join == "round":
            return [("pt", a), ("arc", b)]
        return [("pt", a), ("pt", b)]
    if cross > 0:
        # Left side is the inside of the turn: use the intersection of
        # the two offset lines, clamped so sharp reversals stay bounded.
        m = _offset_intersection(a, din, b, dout)
        if m is None or abs(m - p) > miter_limit * h:
            return [("pt", a), ("pt", b)]
        return [("pt", m)]
    # Left side is the outside of the turn.
    if join == "round":
        return [("pt", a), ("arc", b)]
    if join == "miter":
        # SVG: clip when miterLength/stroke-width > miter_limit, where
        # miterLength/stroke-width == |m - p| / h.
        m = _offset_intersection(a, din, b, dout)
        if m is not None and abs(m - p) <= miter_limit * h:
            return [("pt", m)]
    return [("pt", a), ("pt", b)]  # bevel (or miter past its limit)


def _left_chain_open(pts, dirs, h, join, miter_limit):
    """Left-of-travel offset chain of an open polyline, start to end."""
    items = [("pt", pts[0] + h * dirs[0] * 1j)]
    for i in range(1, len(pts) - 1):
        items.extend(_join_items(pts[i], dirs[i - 1], dirs[i], h, join,
                                 miter_limit))
    items.append(("pt", pts[-1] + h * dirs[-1] * 1j))
    return items


def _left_chain_closed(pts, dirs, h, join, miter_limit):
    """Left-of-travel offset loop of a closed (cyclic) polyline."""
    items = []
    for i in range(len(pts)):
        items.extend(_join_items(pts[i], dirs[i - 1], dirs[i], h, join,
                                 miter_limit))
    return items


def _open_outline_items(pts, dirs, h, cap, join, miter_limit):
    """All outline items (cyclically ordered) for an open polyline."""
    if cap == "square":
        # A square cap is a butt cap on a centerline extended by h.
        pts = ([pts[0] - h * dirs[0]] + list(pts[1:-1])
               + [pts[-1] + h * dirs[-1]])
    left = _left_chain_open(pts, dirs, h, join, miter_limit)
    # The right side of a polyline is the left side of its reversal.
    rpts = pts[::-1]
    rdirs = _unit_vectors(rpts, closed=False)
    right = _left_chain_open(rpts, rdirs, h, join, miter_limit)
    items = list(left)
    if cap == "round":
        items.append(("arc", right[0][1]))  # end cap: left -> right
    items.extend(right)
    if cap == "round":
        items.append(("arc", left[0][1]))  # start cap: right -> left
    return items


# Outline assembly #############################################################

def _items_to_path(items, h, eps):
    """Turn a cyclic list of outline items into a closed `Path`."""
    if len(items) < 2:
        return None
    z0 = items[0][1]
    z = z0
    segs = []
    for kind, p in items[1:]:
        if abs(p - z) <= eps:
            continue  # drop degenerate pieces
        if kind == "arc":
            segs.append(Arc(z, h + 1j * h, 0.0, False, False, p))
        else:
            segs.append(Line(z, p))
        z = p
    if not segs:
        return None
    gap = abs(z - z0)
    if gap > eps:
        segs.append(Line(z, z0))
    elif z != z0:
        # Snap the tiny closing gap so the outline is exactly closed.
        segs[-1] = Line(segs[-1].start, z0)
    return Path(*segs)
