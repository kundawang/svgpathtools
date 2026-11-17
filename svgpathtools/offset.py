"""Contour (parallel-curve) offsetting of paths, e.g. to compensate for the
radius of a CNC engraving/milling tool.

The main entry point is :func:`offset_path`, which offsets every segment of a
continuous path by a signed distance and stitches the pieces together with
round or bevel joins.  :func:`offset_segment` offsets a single segment.

Conventions
-----------
* The offset direction follows the library's normal convention,
  ``seg.normal(t) == -1j * seg.unit_tangent(t)`` (the "right hand rule"
  normal).  A positive ``distance`` shifts the curve toward
  ``point(t) + distance * normal(t)``.
* Only ``Line`` and ``CubicBezier`` segments are supported.  Convert arcs
  first, e.g. with ``Path.approximate_arcs_with_cubics()``.
* The result is a plain ``Path`` of ``Line``/``CubicBezier``/``Arc`` segments
  (``Arc`` only appears for round joins), so ``point()``, ``length()``,
  ``d()`` etc. all work as usual.  A closed input yields a closed output.

Failure and degenerate-segment rules
------------------------------------
* ``tolerance`` must be a positive, finite real number and ``distance`` a
  finite real number; anything else raises ``ValueError``.
* ``join`` must be ``"round"`` or ``"bevel"``; anything else raises
  ``ValueError``.
* Segments of zero length are dropped (they do not contribute to the
  contour).  A path with no nonzero-length segment raises ``OffsetError``.
* A segment whose derivative vanishes (a cusp) raises ``OffsetError`` --
  the offset normal is not defined there.
* If ``|distance|`` reaches the minimum radius of curvature on the offset
  side (the offset curve collapses onto itself), ``OffsetError`` is raised:
  the requested offset is too large for the geometry.
* Self-intersecting input paths raise ``OffsetError``.
* At an "inside" corner (the offset side faces the inside of the turn) the
  two neighbouring offset pieces are trimmed at their intersection; if no
  intersection is found a bevel segment is inserted as a fallback.
* If the cubic-offset approximation cannot meet ``tolerance`` within
  ``max_depth`` bisections, ``OffsetError`` is raised (ask for a looser
  tolerance).
"""

from __future__ import annotations

import math
from numbers import Real

from .path import Path, Line, CubicBezier, QuadraticBezier, Arc

__all__ = ["OffsetError", "offset_path", "offset_segment"]


class OffsetError(ValueError):
    """Raised when a path cannot be offset as requested."""


# Interior parameter values used to gauge the cubic-offset approximation
# error and to probe for cusps / curvature collapse.
_SAMPLE_TS = tuple(k / 8.0 for k in range(1, 8))

# Tangents whose cross product magnitude is below this are treated as
# parallel (smooth joint, no join segment needed).
_PARALLEL_EPS = 1e-9

# A derivative magnitude below this (absolute) marks a cusp.
_CUSP_EPS = 1e-12

# Offset speed factor |r_d'|/|r'| = 1 + d*kappa must stay above this.
_COLLAPSE_EPS = 1e-9


def _validate_inputs(distance, tolerance, join):
    if not isinstance(distance, Real) or isinstance(distance, bool) \
            or not math.isfinite(distance):
        raise ValueError("distance must be a finite real number, got "
                         "{!r}".format(distance))
    if not isinstance(tolerance, Real) or isinstance(tolerance, bool) \
            or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive, finite real number, "
                         "got {!r}".format(tolerance))
    if join not in ("round", "bevel"):
        raise ValueError("join must be 'round' or 'bevel', got "
                         "{!r}".format(join))
    return float(distance), float(tolerance)


def _is_zero_length(seg):
    if isinstance(seg, Line):
        return seg.start == seg.end
    if isinstance(seg, CubicBezier):
        return seg.start == seg.end == seg.control1 == seg.control2
    return False


def _signed_curvature(seg, t):
    """Curvature signed w.r.t. the library normal: the offset speed factor
    at parameter t is ``1 + distance * _signed_curvature(seg, t)``."""
    dz = seg.derivative(t)
    if abs(dz) < _CUSP_EPS:
        raise OffsetError(
            "segment has a cusp (vanishing derivative) at t = {}; the "
            "offset normal is undefined there".format(t))
    ddz = seg.derivative(t, n=2)
    return (dz.real * ddz.imag - dz.imag * ddz.real) / abs(dz) ** 3


def _check_offset_collapse(seg, distance):
    for t in (0.0, 1.0) + _SAMPLE_TS:
        if 1.0 + distance * _signed_curvature(seg, t) <= _COLLAPSE_EPS:
            raise OffsetError(
                "offset distance {!r} is too large: it meets or exceeds the "
                "radius of curvature on the offset side near t = {}".format(
                    distance, t))


def _normal(seg, t):
    try:
        return seg.normal(t)
    except ValueError as exc:
        raise OffsetError(
            "segment has a cusp (undefined tangent) at t = {}: {}".format(
                t, exc))


def _offset_line(seg, distance):
    n = seg.normal()
    return Line(seg.start + distance * n, seg.end + distance * n)


def _offset_cubic(seg, distance, tolerance, depth, max_depth):
    _check_offset_collapse(seg, distance)
    bpoints = seg.bpoints()
    ts = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    controls = [bpoints[i] + distance * _normal(seg, ts[i])
                for i in range(4)]
    candidate = CubicBezier(*controls)
    err = max(abs(candidate.point(t)
                  - (seg.point(t) + distance * _normal(seg, t)))
              for t in _SAMPLE_TS)
    if err <= tolerance:
        return [candidate]
    if depth >= max_depth:
        raise OffsetError(
            "could not approximate the offset within tolerance {!r} after "
            "{} bisections; relax the tolerance".format(tolerance, max_depth))
    left, right = seg.split(0.5)
    return (_offset_cubic(left, distance, tolerance, depth + 1, max_depth) +
            _offset_cubic(right, distance, tolerance, depth + 1, max_depth))


def offset_segment(segment, distance, tolerance=0.01, max_depth=16):
    """Offsets a single segment by a signed distance.

    Parameters
    ----------
    segment : Line | CubicBezier
    distance : float
        Signed offset distance; positive offsets toward
        ``segment.normal(t)``.
    tolerance : float
        Maximum allowed distance between the returned curve and the exact
        offset curve (must be positive).
    max_depth : int
        Bisection limit for the cubic-offset approximation.

    Returns
    -------
    list
        ``[Line]`` for a line input, or a list of ``CubicBezier`` segments
        approximating the exact offset within ``tolerance``.  The list is
        empty for a zero-length input segment.
    """
    distance, tolerance = _validate_inputs(distance, tolerance, "bevel")
    if isinstance(segment, Line):
        return [] if _is_zero_length(segment) else [
            _offset_line(segment, distance)]
    if isinstance(segment, CubicBezier):
        if _is_zero_length(segment):
            return []
        return _offset_cubic(segment, distance, tolerance, 0, max_depth)
    raise TypeError(
        "only Line and CubicBezier segments can be offset, got {}; convert "
        "arcs with Path.approximate_arcs_with_cubics() first".format(
            type(segment).__name__))


def _round_join(point_a, point_b, center, distance):
    """Arc from point_a to point_b around center with radius |distance|."""
    radius = abs(distance)
    mid_dir = (point_a - center) + (point_b - center)
    expected_mid = center + radius * mid_dir / abs(mid_dir)
    best = None
    for sweep in (True, False):
        arc = Arc(point_a, radius + 1j * radius, 0, False, sweep, point_b)
        err = abs(arc.point(0.5) - expected_mid)
        if best is None or err < best[1]:
            best = (arc, err)
    return best[0]


def _trim_join(prev_seg, next_seg):
    """Trims two offset pieces meeting at an inside corner.

    Returns the cropped (prev_seg, next_seg) or None if they do not
    intersect."""
    try:
        intersections = prev_seg.intersect(next_seg)
    except Exception:
        return None
    if not intersections:
        return None
    t1, t2 = min(intersections, key=lambda ts: (1.0 - ts[0]) + ts[1])
    t1 = min(max(t1, 0.0), 1.0)
    t2 = min(max(t2, 0.0), 1.0)
    return prev_seg.cropped(0.0, t1), next_seg.cropped(t2, 1.0)


def _with_start(seg, start):
    if isinstance(seg, Line):
        return Line(start, seg.end)
    if isinstance(seg, CubicBezier):
        return CubicBezier(start, seg.control1, seg.control2, seg.end)
    if isinstance(seg, Arc):
        return Arc(start, seg.radius, seg.rotation, seg.large_arc,
                   seg.sweep, seg.end)
    raise TypeError("unexpected segment type {!r}".format(type(seg)))


def _with_end(seg, end):
    if isinstance(seg, Line):
        return Line(seg.start, end)
    if isinstance(seg, CubicBezier):
        return CubicBezier(seg.start, seg.control1, seg.control2, end)
    if isinstance(seg, Arc):
        return Arc(seg.start, seg.radius, seg.rotation, seg.large_arc,
                   seg.sweep, end)
    raise TypeError("unexpected segment type {!r}".format(type(seg)))


def _check_self_intersection(segments, closed):
    n = len(segments)
    for i in range(n):
        for j in range(i + 2, n):
            if closed and i == 0 and j == n - 1:
                continue
            if segments[i].intersect(segments[j]):
                raise OffsetError(
                    "input path self-intersects (segments {} and {}); "
                    "offsetting self-intersecting paths is not "
                    "supported".format(i, j))


def offset_path(path, distance, tolerance=0.01, join="round", max_depth=16):
    """Offsets a continuous path by a signed distance.

    Parameters
    ----------
    path : Path
        A continuous path of ``Line`` and/or ``CubicBezier`` segments.
    distance : float
        Signed offset distance; positive offsets toward each segment's
        ``normal(t)``.  ``distance == 0`` returns a copy of the input.
    tolerance : float
        Maximum allowed distance between the output and the exact offset
        curve (must be positive and finite).
    join : str
        ``"round"`` (arc of radius ``abs(distance)`` around the joint) or
        ``"bevel"`` (straight segment) for outside corners.
    max_depth : int
        Bisection limit for the cubic-offset approximation.

    Returns
    -------
    Path
        The offset contour.  It is continuous, and closed iff the input is
        closed.  Open ends are left uncapped.

    Raises
    ------
    ValueError
        For invalid ``distance``, ``tolerance`` or ``join`` arguments, or a
        discontinuous input path.
    TypeError
        For unsupported segment types (``Arc``, ``QuadraticBezier``).
    OffsetError
        For self-intersecting input, cusps, zero-length-only paths, or an
        offset distance too large for the local curvature.
    """
    distance, tolerance = _validate_inputs(distance, tolerance, join)
    if not isinstance(path, Path):
        raise TypeError("path must be a Path, got {}".format(
            type(path).__name__))

    # Rule: zero-length segments are dropped; they do not affect the contour.
    segments = [seg for seg in path if not _is_zero_length(seg)]
    if not segments:
        raise OffsetError("path has no nonzero-length segments to offset")
    for seg in segments:
        if not isinstance(seg, (Line, CubicBezier)):
            raise TypeError(
                "only Line and CubicBezier segments can be offset, found "
                "{}; convert arcs with Path.approximate_arcs_with_cubics() "
                "first".format(type(seg).__name__))

    if any(segments[i].end != segments[i + 1].start
           for i in range(len(segments) - 1)):
        raise ValueError("path must be continuous; offset each piece of "
                         "path.continuous_subpaths() separately")
    closed = segments[-1].end == segments[0].start

    _check_self_intersection(segments, closed)

    # Offset each segment independently (a list of pieces per segment).
    offset_pieces = [offset_segment(seg, distance, tolerance, max_depth)
                     for seg in segments]

    n = len(segments)
    joint_count = n if closed else n - 1
    # joins[i] connects offset_pieces[i] to offset_pieces[(i + 1) % n];
    # it may replace the adjoining pieces (inside-corner trim).
    joins = [None] * n
    for i in range(joint_count):
        j = (i + 1) % n
        tangent1 = segments[i].unit_tangent(1)
        tangent2 = segments[j].unit_tangent(0)
        turn = (tangent1.conjugate() * tangent2).imag
        if abs(turn) < _PARALLEL_EPS:
            continue  # smooth joint: offset endpoints already coincide
        prev_piece = offset_pieces[i][-1]
        next_piece = offset_pieces[j][0]
        if turn * distance < 0:
            # Inside corner: trim the two pieces at their intersection.
            trimmed = _trim_join(prev_piece, next_piece)
            if trimmed is not None:
                offset_pieces[i][-1], offset_pieces[j][0] = trimmed
            else:
                # Documented fallback: connect with a bevel segment.
                joins[i] = Line(prev_piece.end, next_piece.start)
        else:
            # Outside corner: insert the requested join.
            joint = segments[i].end
            if join == "round":
                joins[i] = _round_join(prev_piece.end, next_piece.start,
                                       joint, distance)
            else:
                joins[i] = Line(prev_piece.end, next_piece.start)

    out = []
    for i in range(n):
        out.extend(offset_pieces[i])
        if joins[i] is not None:
            out.append(joins[i])

    # Snap each segment's start onto the previous segment's end so the
    # result is exactly continuous (and exactly closed for closed input).
    for i in range(1, len(out)):
        if out[i].start != out[i - 1].end:
            out[i] = _with_start(out[i], out[i - 1].end)
    if closed and out[-1].end != out[0].start:
        out[-1] = _with_end(out[-1], out[0].start)

    return Path(*out)
