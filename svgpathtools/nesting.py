"""Contour nesting analysis for a set of closed paths.

Given a collection of closed :class:`~svgpathtools.path.Path` objects (for
example the outlines found in a cutting / nesting drawing), this module
reconstructs which contour contains which, assigns each contour its nesting
depth and its even-odd fill/hole identity, and reports every geometric
situation that would make that hierarchy ambiguous as a typed exception.

The hierarchy is based purely on geometric containment, so it is independent
of the orientation (clockwise vs. counter-clockwise) of the input paths and
works uniformly with :class:`Line`, :class:`QuadraticBezier`,
:class:`CubicBezier` and :class:`Arc` segments.

Public entry point: :func:`contour_nesting`.

Error classification
---------------------
Every unresolvable input raises a subclass of
:class:`ContourNestingError` (itself a ``ValueError``):

* :class:`UnclosedContourError`  -- an input path is not closed (or not
  continuous);
* :class:`SelfIntersectingContourError` -- a contour crosses or touches
  itself;
* :class:`DuplicateContourError` -- two contours describe the same geometric
  curve, so neither can be a well-defined parent of the other;
* :class:`ContourIntersectionError` -- two contours cross transversely, so
  their interiors are not nested;
* :class:`ContourContactError` -- two contours share boundary points without
  crossing (tangent or partially overlapping), so the nesting is ambiguous;
* :class:`AmbiguousContainmentError` -- the point-in-contour test cannot be
  decided within the requested tolerance.
"""

from math import cos, sin, pi, sqrt

from .path import Line, Path


__all__ = [
    "ContourNode",
    "contour_nesting",
    "ContourNestingError",
    "UnclosedContourError",
    "SelfIntersectingContourError",
    "DuplicateContourError",
    "ContourIntersectionError",
    "ContourContactError",
    "AmbiguousContainmentError",
]


# Exceptions #################################################################


class ContourNestingError(ValueError):
    """Base class for every error raised by the nesting analysis."""


class UnclosedContourError(ContourNestingError):
    """Raised when an input path is not a single closed contour."""


class SelfIntersectingContourError(ContourNestingError):
    """Raised when a contour crosses or touches itself."""


class DuplicateContourError(ContourNestingError):
    """Raised when two input paths describe the same geometric curve."""


class ContourIntersectionError(ContourNestingError):
    """Raised when two contours cross transversely (not nested)."""


class ContourContactError(ContourNestingError):
    """Raised when two contours share boundary points without crossing."""


class AmbiguousContainmentError(ContourNestingError):
    """Raised when containment cannot be decided within the tolerance."""


# Result type ################################################################


class ContourNode:
    """A node of the contour nesting tree.

    Attributes
    ----------
    index : int
        Position of the contour in the input sequence (never changes, even
        when contours are reordered internally).
    path : Path
        The original (closed) :class:`Path` object.
    parent : ContourNode | None
        The node of the innermost contour that directly contains this one,
        or ``None`` for an outer-most (root) contour.
    children : tuple[ContourNode, ...]
        Directly contained contours, sorted by their input index.
    depth : int
        Nesting depth: ``0`` for root contours, ``1`` for their children,
        and so on.
    is_hole : bool
        Even-odd identity.  ``True`` for odd ``depth`` (a hole of its parent
        region), ``False`` for even ``depth`` (filled material).
    """

    def __init__(self, index, path):
        self.index = index
        self.path = path
        self.parent = None
        self.children = ()
        self.depth = 0

    @property
    def is_hole(self):
        """Even-odd hole identity: ``True`` at odd nesting depth."""
        return self.depth % 2 == 1

    def walk(self):
        """Yield this node and all descendants in pre-order.

        Siblings are visited in input-index order, which makes the output
        independent of the iteration order of any internal data structure.
        """
        yield self
        for child in self.children:
            for node in child.walk():
                yield node

    def __repr__(self):
        return "ContourNode(index={}, depth={}, is_hole={}, children={})".format(
            self.index, self.depth, self.is_hole,
            tuple(c.index for c in self.children))


# Geometry helpers ###########################################################


def _bbox_diagonal(paths):
    """Diagonal of the union bounding box (or 1.0 for empty/degenerate sets)."""
    xmins, xmaxs, ymins, ymaxs = zip(*(p.bbox() for p in paths))
    width = max(xmaxs) - min(xmins)
    height = max(ymaxs) - min(ymins)
    diag = sqrt(width * width + height * height)
    return max(diag, 1.0)


def _dedup_points(points, tol):
    """Filter a sequence of points so consecutive-in-space results collapse."""
    unique = []
    for pt in points:
        if not unique or abs(pt - unique[-1]) > tol:
            unique.append(pt)
    return unique


def _nondegenerate_segments(path):
    """Segments carrying actual geometry (zero-length segments are skipped)."""
    return [seg for seg in path if seg.point(1) != seg.point(0)]


def _self_intersection_points(path, tol):
    """Return contact points where `path` crosses or touches itself."""
    hits = []
    segments = _nondegenerate_segments(path)
    n = len(segments)
    for i in range(n):
        # Each segment is also checked against its own two halves so a
        # single self-intersecting segment (e.g. a looped cubic Bezier) is
        # detected; the halves legitimately meet at their split point.
        half_a, half_b = segments[i].split(0.5)
        candidates = [(j, segments[j], False) for j in range(i + 1, n)]
        candidates.append((i, half_b, True))
        for j, other, is_half in candidates:
            adjacent = not is_half and (
                j == i + 1 or (i == 0 and j == n - 1))
            for t1, t2 in half_a.intersect(other):
                pt = half_a.point(t1)
                # Adjacent segments (and the two halves of one segment)
                # legitimately meet at exactly one shared endpoint.
                if is_half and abs(pt - segments[i].point(0.5)) <= tol:
                    continue
                if adjacent and (abs(pt - segments[i].end) <= tol
                                 or abs(pt - segments[i].start) <= tol):
                    continue
                hits.append(pt)
    return _dedup_points(sorted(hits, key=lambda z: (z.real, z.imag)), tol)


def _distance_to_path(pt, path, tol):
    """Minimum distance from `pt` to any point of `path`.

    Uses adaptive bbox-pruned subdivision and therefore works for every
    segment type (including :class:`Arc`, whose ``radialrange`` is not
    implemented).  Subdivision stops as soon as the bbox lower bound proves
    the segment cannot beat the current best, and never deeper than
    ``max_depth`` levels, so zero-length sub-segments cannot occur.
    """
    best = float("inf")
    max_depth = 40

    def consider(seg, depth):
        nonlocal best
        if seg.point(1) == seg.point(0):
            best = min(best, abs(pt - seg.point(0)))
            return
        lower = _bbox_point_distance(pt, seg.bbox())
        if lower >= best - tol:
            return
        best = min(best, abs(pt - seg.point(0)), abs(pt - seg.point(0.5)),
                   abs(pt - seg.point(1)))
        if best <= tol:
            return
        if depth >= max_depth:
            return
        left, right = seg.split(0.5)
        consider(left, depth + 1)
        if best > tol:
            consider(right, depth + 1)

    for seg in path:
        if best <= tol:
            return best
        consider(seg, 0)
    return best


def _bbox_point_distance(pt, bbox):
    """Lower bound on the distance from `pt` to an axis-aligned bbox."""
    xmin, xmax, ymin, ymax = bbox
    dx = max(xmin - pt.real, 0.0, pt.real - xmax)
    dy = max(ymin - pt.imag, 0.0, pt.imag - ymax)
    return sqrt(dx * dx + dy * dy)


def _boundary_contacts(path_a, path_b, tol):
    """Return ``(point, hit)`` pairs where the two contours meet.

    ``hit`` keeps the intersecting segment indices/parameters so the
    classifier can sample both contours on either side of the contact.
    Contacts coincident within `tol` (e.g. duplicated at a joint) collapse
    to one entry.
    """
    contacts = []
    seen = []
    segs_a = _nondegenerate_segments(path_a)
    segs_b = _nondegenerate_segments(path_b)
    index_a = {id(seg): k for k, seg in enumerate(segs_a)}
    index_b = {id(seg): k for k, seg in enumerate(segs_b)}
    for seg_a in segs_a:
        for seg_b in segs_b:
            for t_a, t_b in seg_a.intersect(seg_b):
                pt = seg_a.point(t_a)
                if any(abs(pt - q) <= tol for q in seen):
                    continue
                seen.append(pt)
                hit = ((t_a, index_a[id(seg_a)]), (t_b, index_b[id(seg_b)]))
                contacts.append((pt, hit))
    contacts.sort(key=lambda item: (item[0].real, item[0].imag))
    return contacts


def _curve_local_points(segments, contact_idx, t_on_seg, deltas):
    """Points on a closed contour a few parameter steps around a contact.

    The contact is located on segment `contact_idx` at parameter
    `t_on_seg`.  Sampling points directly from the curve (rather than
    probing along a single segment tangent) gives the true local flow of
    the contour through a vertex, where a stored tangent may belong to a
    transverse neighbouring edge.  Indices wrap around, which matters when
    the contact sits on the closing joint of the path.
    """
    points = []
    n = len(segments)
    for delta in deltas:
        idx = contact_idx
        t = t_on_seg + delta
        # Wrap across segment boundaries in both directions (closed input).
        while t < 0.0:
            idx = (idx - 1) % n
            t += 1.0
        while t > 1.0:
            idx = (idx + 1) % n
            t -= 1.0
        points.append(segments[idx].point(t))
    return points


def _is_transverse_contact(path_a, path_b, hit, far_pt, tol):
    """True if two contours cross (rather than merely touch) at a contact.

    `hit` is an intersection tuple produced by
    :func:`_boundary_contacts`, carrying segment indices into each path's
    non-degenerate segments and the intersection parameters.  Curves are
    sampled on both sides of the contact along each contour and classified
    against the other contour.  A transverse crossing flips inside/outside
    between the two local sides; a tangential contact (including a shared
    edge or a vertex touch) keeps the non-boundary samples of each contour
    in one region.
    """
    (t_a, idx_a), (t_b, idx_b) = hit
    segs_a = _nondegenerate_segments(path_a)
    segs_b = _nondegenerate_segments(path_b)
    deltas = (-1e-3, -1e-4, 1e-4, 1e-3)

    def regions_for(samples, other):
        regions = set()
        for pt in samples:
            verdict = _point_in_contour(pt, other, far_pt, tol)
            if verdict != "boundary":
                regions.add(verdict)
        return regions

    def flips(regions):
        return "inside" in regions and "outside" in regions

    # A transverse crossing makes BOTH contours switch region as they pass
    # the contact (each curve enters the other's interior on one side and
    # leaves it on the other).  At a tangential / T-junction contact only
    # one contour switches; the other stays on a single side, so requiring
    # both flips separates crossing from touching.
    regions_a = regions_for(
        _curve_local_points(segs_a, idx_a, t_a, deltas), path_b)
    regions_b = regions_for(
        _curve_local_points(segs_b, idx_b, t_b, deltas), path_a)
    return flips(regions_a) and flips(regions_b)


def _is_duplicate(path_a, path_b, tol):
    """True if two closed paths trace the same geometric curve.

    The point-sampling test is symmetric under path reversal and does not
    care how the curve is split into segments.
    """
    for path, other in ((path_a, path_b), (path_b, path_a)):
        for seg in path:
            for t in (0.25, 0.5, 0.75):
                if _distance_to_path(seg.point(t), other, tol) > tol:
                    return False
    return True


def _bboxes_intersect(bbox_a, bbox_b, tol):
    """True if two axis-aligned bounding boxes overlap within `tol`."""
    amin_x, amax_x, amin_y, amax_y = bbox_a
    bmin_x, bmax_x, bmin_y, bmax_y = bbox_b
    return (amin_x <= bmax_x + tol and bmin_x <= amax_x + tol
            and amin_y <= bmax_y + tol and bmin_y <= amax_y + tol)


def _bbox_contains(outer_bbox, inner_bbox, tol):
    """True if `inner_bbox` lies inside `outer_bbox` within `tol`."""
    omin_x, omax_x, omin_y, omax_y = outer_bbox
    imin_x, imax_x, imin_y, imax_y = inner_bbox
    return (imin_x >= omin_x - tol and imax_x <= omax_x + tol
            and imin_y >= omin_y - tol and imax_y <= omax_y + tol)


def _ray_crossings(segments, ray, ray_bbox):
    """Number of intersections between a ray segment and contour segments.

    Segment-level bbox pruning keeps expensive arc/Bezier root solving for
    segments whose bbox actually meets the ray.
    """
    total = 0
    for seg in segments:
        if _bboxes_intersect(seg.bbox(), ray_bbox, 0.0):
            total += len(seg.intersect(ray))
    return total


def _point_in_contour(pt, path, far_pt, tol):
    """Even-odd point-in-contour test by majority vote over several rays.

    Rays are cast from `pt` in fixed, distinct directions.  Three
    well-separated directions are voted on first; the (much more expensive)
    boundary-distance check only runs if the rays disagree, since
    disagreement is the signature of a point on (or within `tol` of) the
    contour.  An unresolvable even split raises
    :class:`AmbiguousContainmentError`.
    """
    segments = _nondegenerate_segments(path)

    def vote(angle):
        direction = complex(cos(angle), sin(angle))
        ray_end = pt + direction * far_pt
        ray = Line(pt, ray_end)
        x0, x1 = sorted((pt.real, ray_end.real))
        y0, y1 = sorted((pt.imag, ray_end.imag))
        n_crossings = _ray_crossings(segments, ray, (x0, x1, y0, y1))
        return n_crossings % 2 == 1

    # Three well-separated directions agree for every non-degenerate input;
    # extra directions are only cast if the first vote is not unanimous.
    angles = (0.0, pi / 3, 2 * pi / 3)
    votes = [vote(angle) for angle in angles]
    if len(set(votes)) == 1:
        return "inside" if votes[0] else "outside"

    if _distance_to_path(pt, path, tol) <= tol:
        return "boundary"

    for angle in (pi / 11, 3 * pi / 11, 5 * pi / 11,
                  7 * pi / 11, 9 * pi / 11):
        votes.append(vote(angle))

    inside_votes = sum(votes)
    outside_votes = len(votes) - inside_votes
    if inside_votes == outside_votes:
        raise AmbiguousContainmentError(
            "Point {} is inside for {} ray directions and outside for {}; "
            "containment could not be decided within tolerance {:.3e}."
            "".format(pt, inside_votes, outside_votes, tol))
    return "inside" if inside_votes > outside_votes else "outside"


def _contour_contains(outer, inner, far_pt, tol):
    """Containment of closed contour `inner` inside closed contour `outer`.

    Returns ``True`` only when every segment midpoint of `inner` lies
    strictly inside `outer`; any midpoint on `outer` is reported as
    ``"boundary"`` (the caller turns boundary cases into contact errors).
    """
    saw_boundary = False
    for seg in inner:
        midpoint = seg.point(0.5)
        verdict = _point_in_contour(midpoint, outer, far_pt, tol)
        if verdict == "boundary":
            saw_boundary = True
        elif verdict == "outside":
            return False
    return "boundary" if saw_boundary else True


# Public API #################################################################


def contour_nesting(paths, tol=None):
    """Build the parent/child nesting tree of a set of closed contours.

    Parameters
    ----------
    paths : iterable[Path]
        Closed, continuous, non-self-intersecting :class:`Path` objects.
        Segment types may be mixed freely (lines, quadratic/cubic Beziers,
        arcs) and path orientation may be either clockwise or
        counter-clockwise.
    tol : float, optional
        Absolute geometric tolerance (in user units) used for coincidence,
        contact and on-boundary decisions.  Defaults to ``1e-9`` times the
        diagonal of the union bounding box.

    Returns
    -------
    roots : tuple[ContourNode, ...]
        Outer-most contours (depth 0), in input order.
    nodes : tuple[ContourNode, ...]
        One :class:`ContourNode` per input path, in the same order as
        ``paths``.  Use ``node.parent``, ``node.children``, ``node.depth``
        and ``node.is_hole`` to inspect the hierarchy.

    Raises
    ------
    UnclosedContourError
        If an input path is empty, discontinuous or not closed.
    SelfIntersectingContourError
        If a contour crosses or touches itself.
    DuplicateContourError
        If two inputs trace the same geometric curve.
    ContourIntersectionError
        If two contours cross each other transversely.
    ContourContactError
        If two contours only touch / partially overlap (nesting ambiguous).
    AmbiguousContainmentError
        If a containment test is numerically undecidable within `tol`.
    """
    paths = tuple(paths)
    if not paths:
        return (), ()

    # --- per-contour validation (before any bbox/tolerance work) ----------
    checked = []
    for index, path in enumerate(paths):
        if not isinstance(path, Path):
            raise TypeError(
                "contour_nesting expects Path objects, got {} at index {}."
                "".format(type(path).__name__, index))
        if len(path) == 0 or not path.iscontinuous() or not path.isclosedac():
            raise UnclosedContourError(
                "Contour at index {} is not a single closed path.".format(
                    index))
        checked.append(path)
    paths = tuple(checked)

    # --- tolerance ---------------------------------------------------------
    if tol is None:
        tol = 1e-9 * _bbox_diagonal(paths)
    tol = float(tol)
    if tol <= 0:
        raise ValueError("tol must be strictly positive, got {}".format(tol))

    # Far enough to leave the union bounding box from any interior point.
    far_pt = 10.0 * _bbox_diagonal(paths) + 10.0

    nodes = []
    for index, path in enumerate(paths):
        self_hits = _self_intersection_points(path, tol)
        if self_hits:
            raise SelfIntersectingContourError(
                "Contour at index {} self-intersects at point {} ({} self "
                "contact point(s) in total).".format(
                    index, self_hits[0], len(self_hits)))
        nodes.append(ContourNode(index, path))

    # --- pair-wise validation ---------------------------------------------
    n = len(nodes)
    bboxes = [node.path.bbox() for node in nodes]
    for i in range(n):
        for j in range(i + 1, n):
            path_i, path_j = nodes[i].path, nodes[j].path
            # Disjoint bounding boxes cannot share a boundary point.
            if not _bboxes_intersect(bboxes[i], bboxes[j], tol):
                continue
            contacts = _boundary_contacts(path_i, path_j, tol)
            if not contacts:
                continue
            if _is_duplicate(path_i, path_j, tol):
                raise DuplicateContourError(
                    "Contours at indices {} and {} describe the same "
                    "geometric path.".format(i, j))
            # Classification at every contact point by sampling both curves
            # on either side of it.  This also handles vertices and shared
            # edges, where a raw segment tangent cannot distinguish a
            # crossing from a tangential touch.
            transverse = any(
                _is_transverse_contact(path_i, path_j, hit, far_pt, tol)
                for _, hit in contacts)
            if transverse:
                raise ContourIntersectionError(
                    "Contours at indices {} and {} cross transversely at "
                    "point {}.".format(i, j, contacts[0][0]))
            raise ContourContactError(
                "Contours at indices {} and {} touch at point {} without "
                "crossing; nesting is ambiguous.".format(i, j,
                                                         contacts[0][0]))

    # --- containment matrix ------------------------------------------------
    # relation[i][j] is True iff contour i contains contour j.
    relation = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # A contour whose bbox reaches outside of i's bbox cannot be
            # contained in i (boundary contacts were rejected above).
            if not _bbox_contains(bboxes[i], bboxes[j], tol):
                continue
            verdict = _contour_contains(
                nodes[i].path, nodes[j].path, far_pt, tol)
            if verdict == "boundary":
                # No boundary contact was detected pairwise above, so a
                # midpoint landing on a boundary means numeric ambiguity.
                raise AmbiguousContainmentError(
                    "Contour {} runs along the boundary of contour {} within "
                    "tolerance {:.3e}.".format(j, i, tol))
            relation[i][j] = verdict

    # --- build the stable parent/child tree --------------------------------
    for j in range(n):
        containers = [i for i in range(n) if relation[i][j]]
        # The parent is the innermost container: the one contained by all
        # the others.  Ties cannot occur for disjoint simple contours.
        parent_idx = None
        for i in containers:
            if all(i == k or relation[k][i] for k in containers):
                parent_idx = i
                break
        if parent_idx is not None:
            nodes[j].parent = nodes[parent_idx]

    # Depth first, then attach children in input order.
    def _depth(node, seen=None):
        if seen is None:
            seen = set()
        if node.index in seen:
            raise ContourNestingError(
                "Cycle detected in contour nesting at index {}.".format(
                    node.index))
        seen.add(node.index)
        return 0 if node.parent is None else 1 + _depth(node.parent, seen)

    for node in nodes:
        node.depth = _depth(node)

    for node in nodes:
        node.children = tuple(
            sorted((c for c in nodes if c.parent is node),
                   key=lambda c: c.index))

    roots = tuple(node for node in nodes if node.parent is None)
    return roots, tuple(nodes)
