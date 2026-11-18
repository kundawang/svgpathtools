"""Adaptive conversion of cubic Bezier curves to quadratic Bezier curves.

Some renderers (e.g. TrueType-based font pipelines) only accept quadratic
Bezier curves.  This module converts the cubic segments of a Path into one
or more QuadraticBezier segments each, such that the geometric deviation
from the original cubic stays below a user-supplied error tolerance.

The main entry points are

* ``cubic_to_quadratics`` -- convert a single CubicBezier segment, and
* ``path_to_quadratics`` -- convert every cubic segment of a Path while
  keeping Line and QuadraticBezier segments (and the subpath order) intact.

Algorithm
---------
Each cubic is first approximated by a single quadratic.  The preferred
control point is the intersection of the cubic's endpoint tangent lines,
which preserves the endpoint tangent directions exactly; it is used
whenever it provably fits.  Otherwise the control point matching the cubic
at t=0.5 is tried (this is also the fallback for parallel/degenerate
tangents, e.g. near cusps and inflection points).  If neither candidate
provably fits, the cubic is split in half (de Casteljau) and the two
halves are processed recursively, until every piece fits or a resource
limit (``max_depth`` / ``max_segments``) is hit, in which case a
QuadraticConversionError is raised.

The error of a candidate is estimated with a guaranteed upper bound on
max_t |C(t) - Q(t)|: the difference D(t) = C(t) - Q(t) is itself a cubic
Bezier (control points 0, D1, D2, 0, where D1, D2 are the inner control
point deltas against the degree-elevated quadratic), so |D| is sampled at
33 equidistant parameter values and a Lipschitz correction
(sup|D'| * sample_spacing / 2, with sup|D'| bounded via the convex hull
property) is added.
"""

from math import isfinite

from .bezier import split_bezier
from .path import Path, Line, QuadraticBezier, CubicBezier, Arc

__all__ = ['QuadraticConversionError', 'cubic_to_quadratics',
           'path_to_quadratics', 'DEFAULT_MAX_DEPTH', 'DEFAULT_MAX_SEGMENTS']

#: Default recursion limit for the adaptive subdivision.  Each level halves
#: the parameter interval, so 12 levels resolve features down to 2**-12 of
#: the original segment's parameter range.
DEFAULT_MAX_DEPTH = 12

#: Default cap on the number of quadratic segments produced per cubic.
DEFAULT_MAX_SEGMENTS = 4096

#: Relative threshold below which two tangent lines count as parallel.
_PARALLEL_RTOL = 1e-12


class QuadraticConversionError(ValueError):
    """Raised when a cubic cannot be approximated within the given limits
    (tolerance invalid, non-finite control points, or the recursion depth /
    segment count limits were exceeded)."""


def _validate_error(error):
    if not isinstance(error, (int, float)) or isinstance(error, bool) \
            or not isfinite(error) or error <= 0:
        raise QuadraticConversionError(
            "error tolerance must be a positive finite number, got %r"
            % (error,))


def _check_finite(points):
    for p in points:
        if not (isfinite(p.real) and isfinite(p.imag)):
            raise QuadraticConversionError(
                "control point %r is not finite" % (p,))


def _tangent_intersection_control(p0, p1, p2, p3):
    """Returns the intersection of the cubic's endpoint tangent lines, to be
    used as the control point of an approximating quadratic (which then
    matches the cubic's endpoint tangent directions exactly).  Returns None
    if either tangent is degenerate or the tangents are (near-)parallel."""
    d_start = p1 - p0  # tangent direction at t=0
    d_end = p2 - p3    # tangent direction at t=1 (pointing backwards)
    len_start = abs(d_start)
    len_end = abs(d_end)
    if len_start == 0 or len_end == 0:
        return None
    cross = d_start.real * d_end.imag - d_start.imag * d_end.real
    if abs(cross) <= _PARALLEL_RTOL * len_start * len_end:
        return None
    w = p3 - p0
    # solve p0 + s*d_start = p3 + u*d_end for s
    s = (w.real * d_end.imag - w.imag * d_end.real) / cross
    c = p0 + s * d_start
    if not (isfinite(c.real) and isfinite(c.imag)):
        return None
    return c


def _midpoint_control(p0, p1, p2, p3):
    """Control point of the quadratic that interpolates the cubic's
    endpoints and its midpoint (t=0.5)."""
    return (3 * (p1 + p2) - (p0 + p3)) / 4


#: Number of intervals used when sampling the error function; the
#: Lipschitz correction uses half the resulting sample spacing.  32 keeps
#: the correction small (the bound overshoots the true error by only a few
#: percent) at negligible cost.
_ERROR_SAMPLES = 32


def _fit_error_bound(p0, p1, p2, p3, c):
    """Guaranteed upper bound on max_t |C(t) - Q(t)| for the cubic
    (p0, p1, p2, p3) and the quadratic (p0, c, p3).

    The difference D(t) = C(t) - Q(t) is a cubic Bezier with control points
    (0, d1, d2, 0), where d1, d2 are the deltas of the inner control points
    against the degree-elevated quadratic.  |D| is sampled at
    _ERROR_SAMPLES + 1 equidistant parameter values and a Lipschitz
    correction is added: sup|D'| <= 3*max(|d1|, |d2 - d1|, |d2|) by the
    convex hull property applied to D'(t).
    """
    d1 = p1 - (p0 + 2 * c) / 3
    d2 = p2 - (2 * c + p3) / 3
    sample_max = 0.0
    for i in range(1, _ERROR_SAMPLES):
        t = i / _ERROR_SAMPLES
        tc = 1 - t
        d = abs(3 * t * tc * tc * d1 + 3 * tc * t * t * d2)
        if d > sample_max:
            sample_max = d
    lip = 3 * max(abs(d1), abs(d2 - d1), abs(d2))
    return sample_max + 0.5 * lip / _ERROR_SAMPLES


def cubic_to_quadratics(cubic, error, max_depth=DEFAULT_MAX_DEPTH,
                        max_segments=DEFAULT_MAX_SEGMENTS,
                        return_t_ranges=False):
    """Approximates a CubicBezier segment with QuadraticBezier segments.

    Args:
        cubic (CubicBezier): the segment to convert.
        error (float): positive upper bound on the geometric deviation
            (max distance between the cubic and its approximation, measured
            at equal parameter values -- hence also a bound on the Hausdorff
            distance).
        max_depth (int): recursion limit for the adaptive subdivision.
        max_segments (int): maximum number of quadratic segments produced;
            a QuadraticConversionError is raised if more would be needed.
        return_t_ranges (bool): if True, each returned item is a tuple
            (segment, (t0, t1)) giving the parameter range of the original
            cubic that the segment approximates.  The ranges tile [0, 1]
            contiguously and in order.

    Returns:
        list of QuadraticBezier (or of (QuadraticBezier, (t0, t1)) tuples).
        The segments join exactly (C0 continuous) and, where the cubic has
        non-degenerate tangents, preserve the endpoint tangent directions
        (G1 continuous joints).  Degenerate (zero-length) cubics convert to
        a single degenerate quadratic.

    Raises:
        QuadraticConversionError: on invalid tolerance, non-finite control
            points, or if max_depth / max_segments would be exceeded.
        TypeError: if `cubic` is not a CubicBezier.
    """
    if not isinstance(cubic, CubicBezier):
        raise TypeError("expected a CubicBezier, got %r" % type(cubic))
    _validate_error(error)
    if max_depth < 0:
        raise QuadraticConversionError("max_depth must be non-negative")
    if max_segments < 1:
        raise QuadraticConversionError("max_segments must be at least 1")

    p0, p1, p2, p3 = cubic.bpoints()
    _check_finite((p0, p1, p2, p3))

    results = []  # list of (QuadraticBezier, t0, t1)

    # Degenerate (zero-length) cubic: a single degenerate quadratic.
    if p0 == p1 == p2 == p3:
        results.append((QuadraticBezier(p0, p0, p0), (0.0, 1.0)))
    else:
        # Iterative deepening with an explicit stack (deepest-first, in
        # parameter order) so Python's recursion limit is never a factor.
        stack = [(p0, p1, p2, p3, 0.0, 1.0, 0)]
        while stack:
            q0, q1, q2, q3, t0, t1, depth = stack.pop()
            # Candidate control points: the endpoint-tangent intersection
            # (preserves tangent directions exactly -- preferred whenever
            # it provably fits) and the midpoint-matching control
            # (fallback, also for parallel/degenerate tangents).
            c = None
            bound = None
            c_tan = _tangent_intersection_control(q0, q1, q2, q3)
            if c_tan is not None:
                bound_tan = _fit_error_bound(q0, q1, q2, q3, c_tan)
                if bound_tan <= error:
                    c, bound = c_tan, bound_tan
            if c is None:
                c_mid = _midpoint_control(q0, q1, q2, q3)
                bound_mid = _fit_error_bound(q0, q1, q2, q3, c_mid)
                if bound_mid <= error:
                    c, bound = c_mid, bound_mid
            if c is not None:
                if len(results) >= max_segments:
                    raise QuadraticConversionError(
                        "conversion needs more than max_segments=%d "
                        "quadratic segments (tolerance too small?)"
                        % max_segments)
                results.append((QuadraticBezier(q0, c, q3), (t0, t1)))
            else:
                if depth >= max_depth:
                    raise QuadraticConversionError(
                        "conversion exceeded max_depth=%d without reaching "
                        "the requested tolerance" % max_depth)
                left, right = split_bezier((q0, q1, q2, q3), 0.5)
                tm = 0.5 * (t0 + t1)
                # push right half first so the left half is processed first
                stack.append((right[0], right[1], right[2], right[3],
                              tm, t1, depth + 1))
                stack.append((left[0], left[1], left[2], left[3],
                              t0, tm, depth + 1))

    if return_t_ranges:
        return results
    return [seg for seg, _ in results]


def path_to_quadratics(path, error, max_depth=DEFAULT_MAX_DEPTH,
                       max_segments=DEFAULT_MAX_SEGMENTS,
                       return_t_ranges=False):
    """Returns a copy of `path` in which every CubicBezier segment has been
    replaced by one or more QuadraticBezier segments (see
    cubic_to_quadratics).  Line and QuadraticBezier segments are kept as-is
    (same objects), and the order of segments -- hence of subpaths -- is
    preserved.  Closed paths stay closed: segment endpoints are preserved
    exactly, and the closed flag is carried over.

    Args:
        path (Path): the path to convert.
        error (float): positive geometric error tolerance (per cubic).
        max_depth (int): per-cubic recursion limit.
        max_segments (int): per-cubic limit on produced segments.
        return_t_ranges (bool): if True, returns a tuple
            (new_path, t_ranges) where t_ranges[i] is the parameter range of
            the original i-th output segment's source segment: (t0, t1) with
            (0.0, 1.0) for segments that were kept unchanged.

    Returns:
        Path, or (Path, list of (float, float)) if return_t_ranges=True.

    Raises:
        NotImplementedError: if the path contains Arc segments.
        QuadraticConversionError: see cubic_to_quadratics.
    """
    if not isinstance(path, Path):
        raise TypeError("expected a Path, got %r" % type(path))
    _validate_error(error)

    new_segments = []
    t_ranges = []
    for seg in path:
        if isinstance(seg, CubicBezier):
            pieces = cubic_to_quadratics(
                seg, error, max_depth=max_depth, max_segments=max_segments,
                return_t_ranges=True)
            for quad, trange in pieces:
                new_segments.append(quad)
                t_ranges.append(trange)
        elif isinstance(seg, (Line, QuadraticBezier)):
            new_segments.append(seg)
            t_ranges.append((0.0, 1.0))
        elif isinstance(seg, Arc):
            raise NotImplementedError(
                "Arc segments are not supported; convert them to cubics "
                "first (e.g. with Path.d() / parse_path round-trip).")
        else:
            raise TypeError("unsupported segment type %r" % type(seg))

    new_path = Path(*new_segments)
    new_path._closed = getattr(path, '_closed', False)
    if return_t_ranges:
        return new_path, t_ranges
    return new_path
