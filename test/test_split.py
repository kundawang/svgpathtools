# External dependencies
from __future__ import division, absolute_import, print_function
import unittest

import numpy as np

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path,
    split_path_at_intersections, InvalidToleranceError,
    IntersectionLimitError, OverlappingGeometryError, SplitPath,
)


TOL = 1e-8


def point_to_segment_distance(point, seg):
    """Exact distance for Lines; a fine-sample approximation for curves."""
    if isinstance(seg, Line):
        direction = seg.end - seg.start
        length_sq = (direction * direction.conjugate()).real
        if length_sq == 0.0:
            return abs(point - seg.start)
        t = ((point - seg.start) * direction.conjugate()).real / length_sq
        t = min(1.0, max(0.0, t))
        return abs(point - seg.point(t))
    return min(abs(point - seg.point(t)) for t in np.linspace(0, 1, 200))


def bowtie_path():
    """A closed figure-8 ("bowtie"): a unit square traced once, followed by its
    two diagonals.  The diagonals cross once at the center, so the result has
    four corner nodes, one interior crossing node and ten edge segments."""
    return Path(
        Line(0 + 0j, 100 + 0j),
        Line(100 + 0j, 100 + 100j),
        Line(100 + 100j, 0 + 100j),
        Line(0 + 100j, 0 + 0j),
        Line(0 + 0j, 100 + 100j),
        Line(100 + 100j, 100 + 0j),
        Line(100 + 0j, 0 + 100j),
        Line(0 + 100j, 0 + 0j),
    )


def self_crossing_cubic():
    """A single cubic that genuinely crosses itself (not a cusp)."""
    return CubicBezier(0 + 0j, 10 + 10j, -10 + 10j, 10 + 0j)


def tangent_path():
    """A quadratic arc whose apex is touched, at a single interior point, by a
    horizontal line."""
    return Path(
        QuadraticBezier(0 + 0j, 50 + 100j, 100 + 0j),
        Line(0 + 50j, 100 + 50j),
    )


def overlapping_lines_path():
    """Two collinear lines sharing the interval [20, 100]."""
    return Path(Line(0 + 0j, 100 + 0j), Line(20 + 0j, 120 + 0j))


class TestSplitPath(unittest.TestCase):

    # -- figure-8 fixture ---------------------------------------------------

    def test_figure_eight_nodes_edges_and_reconstruction(self):
        path = bowtie_path()
        result = path.split_at_intersections()

        self.assertIsInstance(result, SplitPath)
        # Four corners + the one diagonal crossing.
        self.assertEqual(len(result.nodes), 5)
        # The 8 input segments are cut at the center crossing: the four
        # diagonals each split in two (8 pieces) and the 4 boundary edges stay
        # whole, giving 10 edges total.
        self.assertEqual(len(result.edges), 10)
        self.assertEqual(len(result.intersections), 1)
        self.assertTrue(result.closed)

        crossing = result.intersections[0]
        self.assertAlmostEqual(crossing.node, 50 + 50j, delta=TOL)
        # The two diagonals each pass once through the center -> two
        # incidences, one per distinct segment.
        self.assertEqual(len(crossing.incidences), 2)
        involved_segments = {inc[0] for inc in crossing.incidences}
        self.assertEqual(len(involved_segments), 2)

    def test_edges_share_node_objects(self):
        result = bowtie_path().split_at_intersections()
        for edge, (start_node, end_node) in zip(result.edges,
                                                result.edge_nodes):
            self.assertIs(edge.start, start_node)
            self.assertIs(edge.end, end_node)
        # Consecutive edges of a continuous input must chain on one node.
        for first, second in zip(result.edge_nodes[:-1], result.edge_nodes[1:]):
            self.assertIs(first[1], second[0])

    def test_figure_eight_rebuilds_original_geometry(self):
        path = bowtie_path()
        result = path.split_at_intersections()
        # Every edge point must lie on some original segment and vice versa.
        # All bowtie edges are Lines, so use exact point-to-line distances.
        for edge in result.edges:
            self.assertIsInstance(edge, Line)
            for u in np.linspace(0, 1, 50):
                point = edge.point(u)
                nearest = min(point_to_segment_distance(point, seg)
                              for seg in path)
                self.assertLess(nearest, 1e-7)
        for seg in path:
            for t in np.linspace(0, 1, 100):
                point = seg.point(t)
                nearest = min(point_to_segment_distance(point, edge)
                              for edge in result.edges)
                self.assertLess(nearest, 1e-7)

    def test_edges_are_exact_subsegments_of_their_source(self):
        # For every input segment, reconstruct its expected cut intervals from
        # the incidence records and confirm an edge of the same type covers
        # each interval to machine precision (duplicate edges are allowed,
        # e.g. the symmetric bowtie).
        path = bowtie_path()
        result = path.split_at_intersections()
        cuts_per_segment = {}
        for ip in result.intersections:
            for seg_index, t in ip.incidences:
                cuts_per_segment.setdefault(seg_index, set()).add(t)
        for seg_index, seg in enumerate(path):
            params = sorted({0.0, 1.0} | cuts_per_segment.get(seg_index, set()))
            for t0, t1 in zip(params[:-1], params[1:]):
                start, end = seg.point(t0), seg.point(t1)
                candidates = [e for e in result.edges
                              if type(e) is type(seg)
                              and abs(e.start - start) < 1e-7
                              and abs(e.end - end) < 1e-7]
                self.assertTrue(candidates, "missing edge for segment "
                                             "{} interval ({}, {})".format(
                    seg_index, t0, t1))
                edge = candidates[0]
                for s in np.linspace(0, 1, 50):
                    self.assertAlmostEqual(
                        edge.point(s), seg.point(t0 + s * (t1 - t0)),
                        delta=1e-7)

    def test_result_path_supports_path_interface_and_d_string(self):
        result = bowtie_path().split_at_intersections()
        rebuilt = result.path
        self.assertIsInstance(rebuilt, Path)
        self.assertEqual(len(rebuilt), len(result.edges))
        d_string = rebuilt.d()
        self.assertTrue(d_string.startswith("M"))
        # Every output edge is a standard segment type.
        for edge in result.edges:
            self.assertTrue(hasattr(edge, "point"))
            self.assertTrue(hasattr(edge, "length"))

    # -- intra-segment cubic self-intersection ------------------------------

    def test_intra_segment_cubic_self_crossing(self):
        cubic = self_crossing_cubic()
        result = Path(cubic).split_at_intersections()

        self.assertEqual(len(result.intersections), 1)
        self.assertEqual(len(result.edges), 3)
        self.assertEqual(len(result.nodes), 3)

        t_a, t_b = (inc[1] for inc in
                    result.intersections[0].incidences)
        self.assertLess(0.0, t_a)
        self.assertLess(t_a, t_b)
        self.assertLess(t_b, 1.0)

        # The three edges reproduce the original cubic exactly on their
        # respective parameter intervals.
        intervals = [(0.0, t_a), (t_a, t_b), (t_b, 1.0)]
        for edge, (u0, u1) in zip(result.edges, intervals):
            for s in np.linspace(0, 1, 100):
                self.assertAlmostEqual(
                    edge.point(s), cubic.point(u0 + s * (u1 - u0)),
                    delta=1e-9)
            self.assertIsInstance(edge, CubicBezier)

    def test_plain_cubic_has_no_self_intersection(self):
        cubic = CubicBezier(0 + 0j, 30 + 30j, 60 + 30j, 100 + 0j)
        result = Path(cubic).split_at_intersections()
        self.assertEqual(len(result.intersections), 0)
        self.assertEqual(len(result.edges), 1)

    def test_cusp_cubic_is_not_split(self):
        # A cusp (derivative vanishes) is not a proper self-crossing.
        cusp = CubicBezier(0 + 0j, 100 + 100j, 0 + 100j, 100 + 0j)
        result = Path(cusp).split_at_intersections()
        self.assertEqual(len(result.intersections), 0)

    # -- tangency fixture ---------------------------------------------------

    def test_tangent_contact_is_one_shared_node(self):
        path = tangent_path()
        result = path.split_at_intersections()
        self.assertEqual(len(result.intersections), 1)
        contact = result.intersections[0]
        self.assertAlmostEqual(contact.node, 50 + 50j, delta=TOL)
        self.assertEqual(len(contact.incidences), 2)
        # Both segments are split once: two edges each.
        self.assertEqual(len(result.edges), 4)
        # Every edge incident to the contact uses the same canonical node.
        contact = result.intersections[0].node
        incident = [n for pair in result.edge_nodes for n in pair
                    if n is contact]
        self.assertTrue(incident)
        self.assertTrue(all(n is contact for n in incident))

    # -- overlapping line fixture -------------------------------------------

    def test_collinear_overlapping_lines(self):
        path = overlapping_lines_path()
        result = path.split_at_intersections()

        self.assertEqual(len(result.overlaps), 1)
        overlap = result.overlaps[0]
        self.assertAlmostEqual(overlap.node_start, 20 + 0j, delta=TOL)
        self.assertAlmostEqual(overlap.node_end, 100 + 0j, delta=TOL)
        self.assertEqual(overlap.segment_index1, 0)
        self.assertEqual(overlap.segment_index2, 1)

        # Edges: [0,20], [20,100], [20,100], [100,120].
        endpoints = [(edge.start, edge.end) for edge in result.edges]
        self.assertEqual(len(endpoints), 4)
        self.assertAlmostEqual(endpoints[1][0], endpoints[2][0], delta=TOL)
        self.assertAlmostEqual(endpoints[1][1], endpoints[2][1], delta=TOL)

    def test_parallel_non_overlapping_lines_have_no_intersection(self):
        path = Path(Line(0 + 0j, 100 + 0j), Line(0 + 200j, 100 + 200j))
        result = path.split_at_intersections()
        self.assertEqual(len(result.intersections), 0)
        self.assertEqual(len(result.overlaps), 0)
        self.assertEqual(len(result.edges), 2)

    # -- zero-length and no-intersection cases ------------------------------

    def test_zero_length_segment_is_kept_and_reported(self):
        path = Path(Line(0 + 0j, 0 + 0j), Line(0 + 0j, 100 + 0j))
        result = path.split_at_intersections()
        self.assertEqual(result.zero_length_segments, (0,))
        self.assertEqual(len(result.edges), 2)
        self.assertEqual(result.edges[0].start, result.edges[0].end)

    def test_path_without_intersections_is_returned_one_edge_per_segment(self):
        path = Path(Line(0 + 0j, 100 + 0j),
                    QuadraticBezier(100 + 0j, 150 + 50j, 200 + 0j))
        result = path.split_at_intersections()
        self.assertEqual(len(result.edges), 2)
        self.assertEqual(len(result.intersections), 0)

    # -- closed-path seam dedup ---------------------------------------------

    def test_closed_path_head_and_tail_share_one_node(self):
        triangle = Path(Line(0 + 0j, 100 + 0j),
                        Line(100 + 0j, 50 + 100j),
                        Line(50 + 100j, 0 + 0j))
        result = triangle.split_at_intersections()
        self.assertTrue(result.closed)
        self.assertEqual(len(result.nodes), 3)
        self.assertIs(result.edge_nodes[0][0], result.edge_nodes[-1][1])

    def test_near_closed_path_merges_within_tolerance(self):
        path = Path(Line(0 + 0j, 100 + 0j),
                    Line(100 + 0j, 50 + 100j),
                    Line(50 + 100j, 1e-11 + 1e-11j))
        result = path.split_at_intersections(tol=1e-9)
        self.assertTrue(result.closed)
        self.assertEqual(len(result.nodes), 3)

    def test_open_path_is_not_closed(self):
        path = Path(Line(0 + 0j, 100 + 0j), Line(100 + 0j, 50 + 100j))
        result = path.split_at_intersections()
        self.assertFalse(result.closed)

    # -- arc support ---------------------------------------------------------

    def test_arc_crossed_by_line(self):
        arc = Arc(0 + 0j, 50 + 50j, 0, 0, 1, 100 + 0j)  # bottom semicircle
        line = Line(50 - 60j, 50 + 20j)
        result = Path(arc, line).split_at_intersections()
        self.assertEqual(len(result.intersections), 1)
        self.assertAlmostEqual(result.intersections[0].node, 50 - 50j,
                               delta=TOL)

    def test_rotated_arc_crossed_by_line(self):
        arc = Arc(0 + 0j, 60 + 30j, 45, 0, 1, 100 + 0j)
        line = Line(arc.center - 80j, arc.center + 80j)
        result = Path(arc, line).split_at_intersections()
        self.assertEqual(len(result.intersections), 1)
        # Output arc pieces keep the arc's ellipse parameters.
        for edge in result.edges:
            if isinstance(edge, Arc):
                self.assertEqual(edge.radius, arc.radius)
                self.assertEqual(edge.rotation, arc.rotation)

    def test_interior_arc_tangency(self):
        arc_a = Arc(0 + 0j, 50 + 50j, 0, 1, 1, 50 + 50j)
        arc_b = Arc(200 + 0j, 50 + 50j, 0, 1, 0, 150 + 50j)
        result = Path(arc_a, arc_b).split_at_intersections()
        self.assertEqual(len(result.intersections), 1)
        self.assertAlmostEqual(result.intersections[0].node, 100 + 0j,
                               delta=1e-7)

    def test_arc_line_tangency_has_deduplicated_incidences(self):
        # A horizontal line tangent to the bottom of a semicircle.  Repeated
        # polynomial roots must not produce duplicate incidence records.
        arc = Arc(0 + 0j, 50 + 50j, 0, 0, 1, 100 + 0j)
        line = Line(0 - 50j, 100 - 50j)
        result = Path(arc, line).split_at_intersections()
        self.assertEqual(len(result.intersections), 1)
        contact = result.intersections[0]
        self.assertAlmostEqual(contact.node, 50 - 50j, delta=1e-7)
        # Exactly one incidence per involved segment (no duplicates).
        segment_indices = [inc[0] for inc in contact.incidences]
        self.assertEqual(sorted(segment_indices), [0, 1])

    # -- validation and limits ----------------------------------------------

    def test_invalid_tolerance_raises(self):
        path = bowtie_path()
        for bad in (0, -1e-9, float("nan"), float("inf"), "1e-9", True):
            with self.assertRaises(InvalidToleranceError):
                path.split_at_intersections(tol=bad)

    def test_invalid_max_intersections_raises(self):
        path = bowtie_path()
        for bad in (0, -1, 1.5, "2", True):
            with self.assertRaises(ValueError):
                path.split_at_intersections(max_intersections=bad)

    def test_intersection_limit_is_enforced(self):
        # Two separate X crossings -> two distinct intersection nodes.
        path = Path(
            Line(0 + 0j, 50 + 50j),
            Line(0 + 50j, 50 + 0j),
            Line(50 + 0j, 100 + 50j),
            Line(50 + 50j, 100 + 0j),
        )
        self.assertEqual(len(path.split_at_intersections().intersections), 2)
        with self.assertRaises(IntersectionLimitError):
            path.split_at_intersections(max_intersections=1)
        # None removes the cap.
        self.assertEqual(
            len(path.split_at_intersections(max_intersections=None)
                .intersections), 2)

    def test_non_path_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            split_path_at_intersections([Line(0, 1)])

    def test_empty_path_returns_empty_split(self):
        result = Path().split_at_intersections()
        self.assertEqual(len(result.edges), 0)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(result.path.d(), "")
        self.assertFalse(result.closed)

    def test_zero_length_bezier_segments_are_reported(self):
        path = Path(
            Line(0 + 0j, 0 + 0j),
            QuadraticBezier(5 + 5j, 5 + 5j, 5 + 5j),
            CubicBezier(7 + 7j, 7 + 7j, 7 + 7j, 7 + 7j),
            Line(5 + 5j, 20 + 20j),
        )
        result = path.split_at_intersections()
        self.assertEqual(result.zero_length_segments, (0, 1, 2))
        self.assertEqual(len(result.edges), 4)

    def test_overlapping_bezier_raises_overlapping_geometry_error(self):
        # Two identical (coincident) non-linear segments overlap on an
        # interval and cannot be represented by a finite set of nodes.
        quad = QuadraticBezier(0 + 0j, 50 + 100j, 100 + 0j)
        path = Path(quad,
                    QuadraticBezier(0 + 0j, 50 + 100j, 100 + 0j))
        with self.assertRaises(OverlappingGeometryError):
            path.split_at_intersections()

    def test_partial_bezier_overlap_raises_quickly(self):
        # A sub-curve coinciding with part of another curve shares infinitely
        # many points and must raise promptly (not hang in subdivision).
        quad = QuadraticBezier(0 + 0j, 50 + 100j, 100 + 0j)
        sub = quad.cropped(0.3, 0.7)
        with self.assertRaises(OverlappingGeometryError):
            Path(quad, sub).split_at_intersections()

    def test_overlap_detection_does_not_fire_on_a_single_crossing(self):
        # Two curved cubics crossing at exactly one interior point must be
        # split normally, never mistaken for an overlapping interval.
        h = CubicBezier(0 + 50j, 30 + 0j, 70 + 100j, 100 + 50j)
        g = CubicBezier(50 + 0j, 20 + 40j, 80 + 60j, 50 + 100j)
        result = Path(h, g).split_at_intersections()
        self.assertEqual(len(result.intersections), 1)
        # The library's subdivision intersection locates the crossing to
        # ~1e-5 on unit-scale geometry, so use its standard test tolerance.
        self.assertAlmostEqual(result.intersections[0].node, 50 + 50j,
                               delta=1e-4)

    def test_two_crossing_circles_yield_two_nodes(self):
        circle_a = Path(Arc(0 + 0j, 50 + 50j, 0, 0, 1, 100 + 0j),
                        Arc(100 + 0j, 50 + 50j, 0, 0, 1, 0 + 0j))
        circle_b = Path(Arc(60 - 40j, 50 + 50j, 0, 0, 1, 160 - 40j),
                        Arc(160 - 40j, 50 + 50j, 0, 0, 1, 60 - 40j))
        result = Path(*circle_a, *circle_b).split_at_intersections()
        self.assertEqual(len(result.intersections), 2)
        # Every intersection is genuinely shared by both circles.
        for point in result.intersections:
            self.assertGreaterEqual(len(point.incidences), 2)


if __name__ == "__main__":
    unittest.main()
