"""Tools for splitting a (possibly self-intersecting) path into edge segments
that meet at shared nodes.

The main entry point is :func:`split_path_at_intersections` (also available as
:meth:`Path.split_at_intersections`).  It finds every point where the path
crosses or touches itself -- both between distinct segments and within a single
segment -- and cuts the path there.  The result is a :class:`SplitPath` whose
``edges`` are ordinary :class:`~svgpathtools.path.Line`,
:class:`~svgpathtools.path.QuadraticBezier`,
:class:`~svgpathtools.path.CubicBezier` or :class:`~svgpathtools.path.Arc`
objects, so the existing :class:`~svgpathtools.path.Path` interface
(``point()``, ``length()``, ``d()``, ...) keeps working unchanged.

Behavior summary
----------------
* Tangent contacts are ordinary intersections: they produce one shared node.
* Collinear, overlapping :class:`~svgpathtools.path.Line` segments are handled
  explicitly; the overlap endpoints become nodes and the result records the
  overlap in ``SplitPath.overlaps``.
* Zero-length segments cannot be cut; they are kept verbatim and their indices
  are reported in ``SplitPath.zero_length_segments``.
* A path without self-intersections is returned unchanged (one edge per input
  segment) and ``SplitPath.intersections`` is empty.
* Illegal ``tol`` values raise :class:`InvalidToleranceError`; finding more
  than ``max_intersections`` distinct intersection nodes raises
  :class:`IntersectionLimitError`.
* Overlapping Bezier/Arc curves (infinitely many intersection points) raise
  :class:`OverlappingGeometryError`.
"""

from __future__ import division, absolute_import, print_function

import numpy as np

from .path import Path, Line, QuadraticBezier, CubicBezier, Arc, \
    is_path_segment
from numpy import angle as _phase


# Tunables ####################################################################

# Depth of the midpoint-subdivision search for intra-segment (cubic)
# self-intersections.  Depth 5 -> 32 sub-curves, which empirically locates
# every crossing to ~1e-6 before Newton refinement.
_CUBIC_SUBDIVISION_DEPTH = 5

# Cap on Newton refinement iterations when polishing a cubic self-crossing.
_NEWTON_MAXITS = 25


# Exceptions ##################################################################

class SplitPathError(Exception):
    """Base class for every error raised by this module."""


class InvalidToleranceError(SplitPathError, ValueError):
    """Raised when ``tol`` is not a finite, strictly positive number."""


class IntersectionLimitError(SplitPathError):
    """Raised when more than ``max_intersections`` distinct intersection
    nodes are found."""


class OverlappingGeometryError(SplitPathError):
    """Raised when two non-linear segments coincide over an interval, i.e.
    they share infinitely many points and cannot be split at a finite set of
    intersection nodes.

    Attributes:
        segment_index1 (int): index of the first segment in the input path.
        segment_index2 (int): index of the second segment in the input path.
    """

    def __init__(self, message, segment_index1, segment_index2):
        super(OverlappingGeometryError, self).__init__(message)
        self.segment_index1 = segment_index1
        self.segment_index2 = segment_index2


# Result containers ###########################################################

class IntersectionPoint(object):
    """One point where the path crosses or touches itself.

    Attributes:
        node (complex): the canonical coordinate of the intersection.
        incidences (list[tuple[int, float]]): every place the path passes
            through this node, as ``(segment_index, segment_parameter)``
            pairs, ordered by segment index then parameter.
    """

    def __init__(self, node, incidences):
        self.node = node
        self.incidences = sorted(incidences)

    def __repr__(self):
        return "IntersectionPoint(node={}, incidences={})".format(
            self.node, self.incidences)


class Overlap(object):
    """An interval shared by two collinear :class:`Line` segments.

    The overlap runs from ``node_start`` to ``node_end`` and is traced by
    ``segment_index1`` on parameters ``(t1_start, t1_end)`` and by
    ``segment_index2`` on parameters ``(t2_start, t2_end)``.
    """

    def __init__(self, node_start, node_end, segment_index1, t1_start, t1_end,
                 segment_index2, t2_start, t2_end):
        self.node_start = node_start
        self.node_end = node_end
        self.segment_index1 = segment_index1
        self.t1_start = t1_start
        self.t1_end = t1_end
        self.segment_index2 = segment_index2
        self.t2_start = t2_start
        self.t2_end = t2_end

    def __repr__(self):
        return ("Overlap(segments {} and {}, from {} to {})".format(
            self.segment_index1, self.segment_index2,
            self.node_start, self.node_end))


class SplitPath(object):
    """A path decomposed into edge segments joined at shared nodes.

    Attributes:
        nodes (tuple[complex]): all nodes (joints, endpoints and intersection
            points) in first-seen order along the original path.
        edges (tuple): the edge segments, in original path order.  The end of
            ``edges[k]`` is the exact same coordinate object as the start of
            the next edge whenever the two edges are consecutive on the input.
        edge_nodes (tuple[tuple[complex, complex]]): ``(start_node, end_node)``
            for each edge, taken from ``nodes``.
        intersections (tuple[IntersectionPoint]): self-intersection nodes, in
            the order they were discovered.
        overlaps (tuple[Overlap]): collinear line-overlap descriptions.
        zero_length_segments (tuple[int]): indices of zero-length input
            segments that were carried over verbatim.
        closed (bool): whether the input path's first and last nodes were
            merged (i.e. the path is closed within ``tol``).
    """

    def __init__(self, nodes, edges, edge_nodes, intersections, overlaps,
                 zero_length_segments, closed):
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.edge_nodes = tuple(edge_nodes)
        self.intersections = tuple(intersections)
        self.overlaps = tuple(overlaps)
        self.zero_length_segments = tuple(zero_length_segments)
        self.closed = bool(closed)
        self._path = None

    # -- container-style convenience ----------------------------------------

    def __len__(self):
        return len(self.edges)

    def __iter__(self):
        return iter(self.edges)

    def __getitem__(self, index):
        return self.edges[index]

    def node_index(self, node):
        """Index of ``node`` in :attr:`nodes` (identity comparison)."""
        return self.nodes.index(node)

    @property
    def path(self):
        """A :class:`~svgpathtools.path.Path` built from :attr:`edges`, so the
        result supports ``point()``, ``length()``, ``d()`` and friends."""
        if self._path is None:
            self._path = Path(*self.edges)
        return self._path

    def __repr__(self):
        return ("SplitPath(nodes={}, edges={}, intersections={}, "
                "overlaps={})".format(len(self.nodes), len(self.edges),
                                      len(self.intersections),
                                      len(self.overlaps)))


# Internal helpers ############################################################

class _NodeRegistry(object):
    """Assigns one canonical node object to every point, merging points that
    lie within ``tol`` of an already-registered node (first-seen wins)."""

    def __init__(self, tol):
        self.tol = tol
        self.nodes = []

    def register(self, point):
        """Return the canonical node for ``point``, creating one if needed."""
        for node in self.nodes:
            if abs(point - node) <= self.tol:
                return node
        self.nodes.append(point)
        return point

    def find(self, point):
        """Return the canonical node for ``point`` or ``None`` if no registered
        node is within ``tol``."""
        for node in self.nodes:
            if abs(point - node) <= self.tol:
                return node
        return None


def _is_zero_length(seg):
    """A segment is zero-length when all its defining points coincide."""
    if isinstance(seg, Line):
        return seg.start == seg.end
    if isinstance(seg, (QuadraticBezier, CubicBezier)):
        first = seg.bpoints()[0]
        return all(p == first for p in seg.bpoints())
    # An Arc asserts start != end, so it can never be zero-length.
    return False


def _validate_tol(tol):
    # Reject booleans explicitly: isinstance(True, int) is True in Python.
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.floating,
                                                      np.integer)):
        raise InvalidToleranceError(
            "tol must be a real number, got {}.".format(type(tol).__name__))
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise InvalidToleranceError(
            "tol must be a finite, strictly positive number, got {}.".format(tol))
    return tol


def _validate_max_intersections(max_intersections):
    if max_intersections is None:
        return None
    if isinstance(max_intersections, bool) or not isinstance(
            max_intersections, (int, np.integer)) or max_intersections <= 0:
        raise ValueError(
            "max_intersections must be a positive integer or None, got "
            "{}.".format(max_intersections))
    return int(max_intersections)


# -- intersection primitives --------------------------------------------------

def _line_line_parameters(line_a, line_b):
    """Return ``(t_a, t_b)`` for the crossing of two non-parallel lines, or
    ``None`` if they are parallel (including collinear)."""
    pa0, pa1 = line_a.start, line_a.end
    pb0, pb1 = line_b.start, line_b.end
    da = pa1 - pa0
    db = pb1 - pb0
    denom = da.real * (-db.imag) - da.imag * (-db.real)
    if np.isclose(denom, 0.0, rtol=0.0, atol=1e-14):
        return None
    diff = pb0 - pa0
    t_a = (diff.real * (-db.imag) - diff.imag * (-db.real)) / denom
    t_b = (da.real * diff.imag - da.imag * diff.real) / denom
    return t_a, t_b


def _collinear_overlap(line_a, line_b, tol):
    """If the two lines are collinear and share more than an endpoint, return
    the overlap as ``(t_a_start, t_a_end, t_b_start, t_b_end)``; otherwise
    ``None``.  A single-point contact (touching endpoints) returns ``None`` so
    it is handled as a regular point intersection."""
    da = line_a.end - line_a.start
    db = line_b.end - line_b.start
    length_a = abs(da)
    # Collinear iff the vector between starts is parallel to da.
    if abs((line_b.start - line_a.start).real * da.imag
           - (line_b.start - line_a.start).imag * da.real) > tol:
        return None

    def param_on_a(point):
        return ((point - line_a.start) * da.conjugate()).real / (length_a ** 2)

    t_b0 = param_on_a(line_b.start)
    t_b1 = param_on_a(line_b.end)
    # Project line_b's endpoints onto line_a's parameter axis; the overlap in
    # line_a parameters is the intersection of [0, 1] with [t_b0, t_b1].
    lo_a = max(0.0, min(t_b0, t_b1))
    hi_a = min(1.0, max(t_b0, t_b1))
    if hi_a - lo_a <= tol / max(length_a, 1e-300):
        return None  # no positive-length overlap

    def to_b(t_a):
        physical = line_a.point(t_a)
        return ((physical - line_b.start) * db.conjugate()).real / (abs(db) ** 2)

    return lo_a, hi_a, to_b(lo_a), to_b(hi_a)


def _arc_line_intersections(arc, line):
    """Arc/Line intersection that also works when the arc is rotated or is a
    genuine ellipse.  Sends the line through ``arc.u1transform`` (an affine
    map taking the arc's ellipse to the unit circle) and solves
    ``|line'(u)|^2 = 1``.

    Returns ``(t_arc, t_line)`` pairs with both parameters in ``[0, 1]``."""
    line_prime = Line(arc.u1transform(line.start),
                      arc.u1transform(line.end))
    p = line_prime.poly()
    v, w = p[1], p[0]  # line'(u) = v*u + w
    # |v*u + w|^2 = 1  ->  |v|^2 u^2 + 2 Re(v conj(w)) u + (|w|^2 - 1) = 0
    a = (v * v.conjugate()).real
    b = 2.0 * (v * w.conjugate()).real
    c = (w * w.conjugate()).real - 1.0
    if np.isclose(a, 0.0, rtol=0.0, atol=1e-14):
        return []
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    sqrt_disc = sqrt_safe(discriminant)
    candidates = [(-b - sqrt_disc) / (2.0 * a),
                  (-b + sqrt_disc) / (2.0 * a)]
    results = []
    for t_line in candidates:
        if not (0.0 <= t_line <= 1.0):
            continue
        unit_point = line_prime.point(t_line)
        t_arc = arc.phase2t(_phase(unit_point))
        if t_arc is not None and 0.0 <= t_arc <= 1.0:
            results.append((t_arc, t_line))
    return results


def sqrt_safe(x):
    return np.sqrt(max(x, 0.0))


def _cubic_self_intersections(seg, tol):
    """Find proper self-crossings of one :class:`CubicBezier`.

    A midpoint subdivision (using the library's own Bezier intersection
    routine) localizes every crossing; a 2-D Newton step on
    ``B(t1) - B(t2) = 0`` then polishes the parameters so the two images
    agree to machine precision.  Cusps (``t1 == t2``) are not crossings and
    are discarded.
    """
    depth = _CUBIC_SUBDIVISION_DEPTH
    n = 2 ** depth
    pieces = []
    for k in range(n):
        t0, t1 = k / n, (k + 1) / n
        pieces.append((t0, t1, seg.cropped(t0, t1)))

    raw = []
    for i in range(n):
        for j in range(i + 2, n):  # neighboring pieces only share a split point
            a0, a1, piece_a = pieces[i]
            b0, b1, piece_b = pieces[j]
            try:
                hits = piece_a.intersect(piece_b, tol=tol)
            except Exception:
                # Subdivision failed to converge: only happens for coincident
                # pieces, which cannot occur for two pieces of one regular
                # cubic.  Skip rather than crash.
                continue
            for ta, tb in hits:
                global_a = a0 + ta * (a1 - a0)
                global_b = b0 + tb * (b1 - b0)
                if all(abs(global_a - other[0]) > 1e-5 for other in raw):
                    raw.append((global_a, global_b))

    polished = []
    for t1, t2 in raw:
        t1, t2 = _newton_refine_self_crossing(seg, t1, t2)
        if t1 is None:
            continue
        # Proper crossing requires two distinct, interior parameters.
        if not (0.0 < t1 < t2 < 1.0) or (t2 - t1) <= max(tol, 1e-10):
            continue
        if abs(seg.point(t1) - seg.point(t2)) <= tol:
            polished.append((t1, t2))
    return polished


def _newton_refine_self_crossing(seg, t1, t2):
    """Newton iteration on F(t1, t2) = B(t1) - B(t2)."""
    for _ in range(_NEWTON_MAXITS):
        p1, p2 = seg.point(t1), seg.point(t2)
        f = p1 - p2
        if abs(f) < 1e-13:
            break
        d1, d2 = seg.derivative(t1), seg.derivative(t2)
        jacobian = np.array([[d1.real, -d2.real],
                             [d1.imag, -d2.imag]])
        determinant = np.linalg.det(jacobian)
        if abs(determinant) < 1e-14:
            break  # singular (cusp-like); keep the subdivision estimate
        delta = np.linalg.solve(jacobian, np.array([f.real, f.imag]))
        t1 -= delta[0]
        t2 -= delta[1]
        if not (0.0 < t1 < 1.0 and 0.0 < t2 < 1.0):
            return None, None
    if t1 > t2:
        t1, t2 = t2, t1
    return t1, t2


def _segment_pair_intersections(seg_i, seg_j, index_i, index_j, tol):
    """Return ``(point_intersections, overlap)`` for two distinct segments.

    ``point_intersections`` is a list of ``(t_i, t_j)`` pairs.  ``overlap`` is
    a 4-tuple of parameters for a positive-length collinear Line overlap, or
    ``None``.
    """
    # Line/Line needs custom handling because the library routine explicitly
    # does not support collinear segments.
    if isinstance(seg_i, Line) and isinstance(seg_j, Line):
        params = _line_line_parameters(seg_i, seg_j)
        if params is None:
            overlap = _collinear_overlap(seg_i, seg_j, tol)
            if overlap is not None:
                return [], overlap
            return [], None
        t_i, t_j = params
        if 0.0 <= t_i <= 1.0 and 0.0 <= t_j <= 1.0:
            return [(t_i, t_j)], None
        return [], None

    # Rotated/elliptic Arc vs. Line is not covered by Arc.intersect in the
    # library, so handle it here through the unit-circle transform.
    if isinstance(seg_i, Arc) and isinstance(seg_j, Line):
        return [(ta, tl) for ta, tl in _arc_line_intersections(seg_i, seg_j)], None
    if isinstance(seg_j, Arc) and isinstance(seg_i, Line):
        return [(tl, ta) for ta, tl in _arc_line_intersections(seg_j, seg_i)], None

    # The library's subdivision intersection routine grows without bound (and
    # eventually raises) when two curves coincide over an interval.  Detect
    # that case up front with a bounded subdivision so the failure is fast and
    # explicit instead of a long hang deep in the Bezier machinery.
    if _curves_overlap(seg_i, seg_j, tol):
        raise OverlappingGeometryError(
            "Segments {} and {} coincide over an interval (infinitely many "
            "intersection points); only collinear Line overlaps can be split "
            "automatically.".format(index_i, index_j),
            index_i, index_j)

    try:
        pairs = seg_i.intersect(seg_j, tol=tol)
    except OverlappingGeometryError:
        raise
    except Exception as exc:
        raise OverlappingGeometryError(
            "Segments {} and {} coincide over an interval (infinite "
            "intersections); only collinear Line overlaps can be split "
            "automatically. Original error: {}".format(
                index_i, index_j, exc), index_i, index_j)
    return list(pairs), None


# Number of interior sample points used to detect a shared interval.  Two
# well-separated points cannot coincide by chance for a single crossing or
# tangency, so two matches are sufficient evidence of an overlap.
_OVERLAP_SAMPLES = 4


def _point_on_segment_interior(point, seg, tol):
    """Return True only when ``point`` lies in the strict interior of ``seg``
    (not at either endpoint), within the absolute tolerance ``tol``."""
    if isinstance(seg, Line):
        t = seg.point_to_t(point)
        if t is None or not (1e-9 < t < 1.0 - 1e-9):
            return False
        return abs(seg.point(t) - point) <= tol
    if isinstance(seg, Arc) and seg.rotation == 0.0:
        t = seg.point_to_t(point)
        if t is not None:
            if not (1e-9 < t < 1.0 - 1e-9):
                return False
            return abs(seg.point(t) - point) <= tol
    for k in range(1, 200):
        t = k / 200.0  # strictly interior samples
        if abs(seg.point(t) - point) <= tol:
            return True
    return False


def _curves_overlap(seg_a, seg_b, tol):
    """Heuristically decide whether two non-Line segments share a
    positive-length geometric interval (i.e. infinitely many points).

    A single crossing or tangency makes at most one point coincide, and an
    endpoint contact none at all in the strict interior.  An overlap makes a
    run of interior points of one segment lie in the strict interior of the
    other.  Several well-separated interior points are therefore tested; two
    matches are required before an overlap is declared.
    """
    scale = max(abs(seg_a.end - seg_a.start),
                abs(seg_b.end - seg_b.start), 1.0)
    on_tol = max(tol, scale * 1e-7)

    def interior_matches(source, target):
        matches = 0
        for k in range(1, _OVERLAP_SAMPLES + 1):
            point = source.point(k / (_OVERLAP_SAMPLES + 1))
            if _point_on_segment_interior(point, target, on_tol):
                matches += 1
        return matches

    try:
        return (interior_matches(seg_a, seg_b) >= 2 or
                interior_matches(seg_b, seg_a) >= 2)
    except Exception:
        # Never let the heuristic mask a real result: defer to the regular
        # intersect path if anything unexpected happens.
        return False


def _make_edge(seg, t0, t1, start_node, end_node):
    """Build the subsegment of ``seg`` on ``[t0, t1]``, forcing its endpoints
    to the shared canonical nodes so consecutive edges really share nodes.

    Bezier segments are split directly with de Casteljau rather than
    :meth:`cropped`, because the library's ``crop_bezier`` uses
    :meth:`radialrange`, which cannot find ``t1`` when ``point(t0)`` and
    ``point(t1)`` coincide -- precisely the self-intersection case.
    """
    if isinstance(seg, Line):
        return Line(start_node, end_node)
    if isinstance(seg, Arc):
        if abs(seg.delta * (t1 - t0)) <= 180.0:
            large_arc = 0
        else:
            large_arc = 1
        return Arc(start_node, radius=seg.radius, rotation=seg.rotation,
                   large_arc=large_arc, sweep=seg.sweep, end=end_node,
                   autoscale_radius=seg.autoscale_radius)
    # Bezier segment: split out [t0, t1] with de Casteljau, then snap the
    # endpoints onto the shared canonical nodes.
    if t0 == 0.0:
        edge = seg.split(t1)[0]
    elif t1 == 1.0:
        edge = seg.split(t0)[1]
    else:
        edge = seg.split(t0)[1].split((t1 - t0) / (1.0 - t0))[0]
    edge.start = start_node
    edge.end = end_node
    return edge


# Main API ####################################################################

def split_path_at_intersections(path, tol=1e-9, max_intersections=1000):
    """Split ``path`` at every self-intersection into shared-node edge
    segments.

    Args:
        path (Path): the path to split.  It must be continuous (see
            :meth:`Path.continuous_subpaths`); every segment must be a
            :class:`Line`, :class:`QuadraticBezier`, :class:`CubicBezier` or
            :class:`Arc`.
        tol (float): absolute distance tolerance.  Points no farther apart
            than ``tol`` are treated as the same node, so a closed path's
            first and last points and nearby intersections are merged
            stably.  Must be a finite, strictly positive number.
        max_intersections (int or None): upper bound on the number of
            distinct interior self-intersection nodes.  When the bound is
            exceeded, :class:`IntersectionLimitError` is raised.  ``None``
            removes the limit.

    Returns:
        SplitPath: the nodes, edges, intersection records and overlap
            records.

    Raises:
        InvalidToleranceError: if ``tol`` is invalid.
        TypeError: if ``path`` contains unsupported segment types.
        IntersectionLimitError: if the intersection cap is exceeded.
        OverlappingGeometryError: if two non-linear segments overlap along an
            interval.
    """
    tol = _validate_tol(tol)
    max_intersections = _validate_max_intersections(max_intersections)

    if not isinstance(path, Path):
        raise TypeError("path must be a Path instance, got {}.".format(
            type(path).__name__))
    for seg in path:
        if not is_path_segment(seg):
            raise TypeError(
                "Unsupported segment type: {}. Only Line, QuadraticBezier, "
                "CubicBezier and Arc are supported.".format(type(seg).__name__))

    # An empty path has no nodes, edges or intersections.
    if len(path) == 0:
        return SplitPath(nodes=[], edges=[], edge_nodes=[], intersections=[],
                         overlaps=[], zero_length_segments=[], closed=False)

    registry = _NodeRegistry(tol)
    # incidences[node] = list of (segment_index, t)
    incidences = {}
    intersection_records = []
    overlaps = []
    zero_length = []

    def add_incidence(node, seg_index, t):
        incidences.setdefault(node, []).append((seg_index, float(t)))

    # -- pass 1: register every endpoint, merging the closed-path seam ------
    # The registry merges any endpoints lying within tol, so a closed path's
    # first and last points (and any near-coincident joints) share one node.
    endpoint_nodes = set()
    for index, seg in enumerate(path):
        start_node = registry.register(seg.start)
        add_incidence(start_node, index, 0.0)
        endpoint_nodes.add(start_node)
    for index, seg in enumerate(path):
        end_node = registry.register(seg.end)
        add_incidence(end_node, index, 1.0)
        endpoint_nodes.add(end_node)
    closed = registry.find(path[-1].end) is registry.find(path[0].start)

    intersection_count = 0

    def note_intersection(node):
        """Account for a newly-registered interior intersection node,
        enforcing the ``max_intersections`` cap."""
        nonlocal intersection_count
        if node in endpoint_nodes or node in intersection_records:
            return
        intersection_count += 1
        if (max_intersections is not None
                and intersection_count > max_intersections):
            raise IntersectionLimitError(
                "Found more than {} distinct self-intersection nodes; raise "
                "max_intersections (or pass None) to continue.".format(
                    max_intersections))
        intersection_records.append(node)

    def register_intersection(point, index_a, t_a, index_b, t_b):
        """Register one crossing/contact.  Both incidences alias to the same
        canonical node even when floating-point error makes the two images
        differ slightly."""
        node = registry.register(point)
        add_incidence(node, index_a, t_a)
        add_incidence(node, index_b, t_b)
        note_intersection(node)
        return node

    # -- pass 2: inter-segment intersections --------------------------------
    for i in range(len(path)):
        seg_i = path[i]
        if _is_zero_length(seg_i):
            if i not in zero_length:
                zero_length.append(i)
            continue
        for j in range(i + 1, len(path)):
            seg_j = path[j]
            if _is_zero_length(seg_j):
                continue
            pairs, overlap_params = _segment_pair_intersections(
                seg_i, seg_j, i, j, tol)
            if overlap_params is not None:
                lo_i, hi_i, lo_j, hi_j = overlap_params
                node_start = registry.register(seg_i.point(lo_i))
                node_end = registry.register(seg_i.point(hi_i))
                add_incidence(node_start, i, lo_i)
                add_incidence(node_end, i, hi_i)
                add_incidence(node_start, j, lo_j)
                add_incidence(node_end, j, hi_j)
                overlaps.append(Overlap(
                    node_start, node_end, i, lo_i, hi_i, j, lo_j, hi_j))
                note_intersection(node_start)
                note_intersection(node_end)
                continue
            for t_i, t_j in pairs:
                point_i = seg_i.point(t_i)
                point_j = seg_j.point(t_j)
                # Snap contacts at a segment endpoint to that endpoint's node.
                t_i, point_i = _snap_to_endpoint(
                    seg_i, t_i, point_i, registry, tol)
                t_j, point_j = _snap_to_endpoint(
                    seg_j, t_j, point_j, registry, tol)
                register_intersection(point_i, i, t_i, j, t_j)

        # -- pass 3: intra-segment self-intersections -----------------------
        if isinstance(seg_i, CubicBezier):
            for t_a, t_b in _cubic_self_intersections(seg_i, tol):
                node = registry.register(seg_i.point(t_a))
                add_incidence(node, i, t_a)
                add_incidence(node, i, t_b)
                note_intersection(node)

    # -- pass 4: cut every segment at its incident nodes --------------------
    edges = []
    edge_nodes = []
    for index, seg in enumerate(path):
        if _is_zero_length(seg):
            node = registry.register(seg.start)
            edges.append(seg)
            edge_nodes.append((node, node))
            continue

        # Collect every (parameter, node) at which this segment is cut.  Two
        # cuts whose parameters agree within a relative tolerance describe the
        # same geometric point (e.g. a joint reported by both adjacent
        # segments) and are merged, keeping the node whose image is closest.
        raw_cuts = [(min(1.0, max(0.0, t)), node)
                    for nd, pairs in incidences.items()
                    for (idx, t) in pairs if idx == index
                    for node in [nd]]
        # Force the physical endpoints to be present.
        raw_cuts.append((0.0, registry.register(seg.start)))
        raw_cuts.append((1.0, registry.register(seg.end)))
        raw_cuts.sort(key=lambda pair: pair[0])

        merged = []  # list of [t, node]
        param_tol = max(1e-10, tol)
        for t, node in raw_cuts:
            if merged and abs(t - merged[-1][0]) <= param_tol:
                t0_prev, node_prev = merged[-1]
                if abs(seg.point(t) - node) < abs(
                        seg.point(t0_prev) - node_prev):
                    merged[-1] = [t, node]
            else:
                merged.append([t, node])

        for (t0, start_node), (t1, end_node) in zip(merged[:-1], merged[1:]):
            if t1 - t0 <= 1e-12:
                # Zero-parameter gap: nothing geometric to emit.
                continue
            edges.append(_make_edge(seg, t0, t1, start_node, end_node))
            edge_nodes.append((start_node, end_node))

    # A node can receive the same (segment, parameter) incidence more than once
    # (e.g. a tangency reported at a repeated quadratic root), so collapse exact
    # duplicates before building the result records.
    for node in list(incidences):
        unique = []
        seen = set()
        for pair in incidences[node]:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)
        incidences[node] = unique

    intersection_points = [
        IntersectionPoint(node, incidences[node])
        for node in intersection_records
    ]

    return SplitPath(
        nodes=registry.nodes,
        edges=edges,
        edge_nodes=edge_nodes,
        intersections=intersection_points,
        overlaps=overlaps,
        zero_length_segments=zero_length,
        closed=closed,
    )


def _snap_to_endpoint(seg, t, point, registry, tol):
    """If an intersection lands within ``tol`` of a segment endpoint, clamp
    its parameter to 0/1 and use the endpoint's registered node."""
    if t <= 1e-10 or abs(point - seg.start) <= tol:
        node = registry.register(seg.start)
        return 0.0, node
    if t >= 1.0 - 1e-10 or abs(point - seg.end) <= tol:
        node = registry.register(seg.end)
        return 1.0, node
    return t, point
