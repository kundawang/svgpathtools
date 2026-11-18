# External dependencies
from __future__ import division, absolute_import, print_function
import unittest

# Internal dependencies
from svgpathtools import (
    Path, Line, QuadraticBezier, CubicBezier, Arc, polygon, parse_path,
    contour_nesting, ContourNestingError, UnclosedContourError,
    SelfIntersectingContourError, DuplicateContourError,
    ContourIntersectionError, ContourContactError,
    AmbiguousContainmentError,
)


# Geometry fixtures ##########################################################


def circle(cx, cy, r, ccw=True):
    """A closed circle built from two half-circle Arc segments."""
    center = complex(cx, cy)
    radius = complex(r, r)
    sweep = ccw
    return Path(
        Arc(center - r, radius, 0, False, sweep, center + r),
        Arc(center + r, radius, 0, False, sweep, center - r))


def rect(x0, y0, x1, y1):
    """A closed axis-aligned rectangle from four Line segments."""
    return polygon(complex(x0, y0), complex(x1, y0),
                   complex(x1, y1), complex(x0, y1))


def mixed_outer():
    """A closed contour mixing Line, CubicBezier and QuadraticBezier."""
    return Path(
        Line(0j, 10 + 0j),
        CubicBezier(10 + 0j, 12 + 3j, 12 + 7j, 10 + 10j),
        Line(10 + 10j, 0 + 10j),
        QuadraticBezier(0 + 10j, -2 + 5j, 0j))


class TestContourNestingTree(unittest.TestCase):
    """Requirement 1 & 4: stable tree, depth and even-odd hole identity."""

    def test_three_level_nesting(self):
        outer, middle, inner = circle(0, 0, 10), circle(0, 0, 6), circle(0, 0, 2)
        roots, nodes = contour_nesting([outer, middle, inner])

        # One root, input order preserved in `nodes`.
        self.assertEqual(len(roots), 1)
        self.assertEqual([node.index for node in nodes], [0, 1, 2])
        self.assertIs(roots[0], nodes[0])

        # Parent / child links.
        self.assertIsNone(nodes[0].parent)
        self.assertIs(nodes[1].parent, nodes[0])
        self.assertIs(nodes[2].parent, nodes[1])
        self.assertEqual(tuple(c.index for c in nodes[0].children), (1,))
        self.assertEqual(tuple(c.index for c in nodes[1].children), (2,))
        self.assertEqual(nodes[2].children, ())

        # Depth and even-odd identity.
        self.assertEqual([node.depth for node in nodes], [0, 1, 2])
        self.assertEqual([node.is_hole for node in nodes],
                         [False, True, False])

    def test_walk_is_stable_pre_order(self):
        outer = rect(0, 0, 20, 20)
        hole_a = circle(5, 10, 2)
        hole_b = circle(15, 10, 2)
        island = circle(15, 10, 0.8)
        roots, nodes = contour_nesting([outer, hole_a, hole_b, island])

        order = [node.index for node in roots[0].walk()]
        self.assertEqual(order, [0, 1, 2, 3])

        depths = {node.index: node.depth for node in nodes}
        holes = {node.index: node.is_hole for node in nodes}
        self.assertEqual(depths, {0: 0, 1: 1, 2: 1, 3: 2})
        self.assertEqual(holes, {0: False, 1: True, 2: True, 3: False})
        # The island belongs to hole_b (index 2), not hole_a.
        self.assertIs(nodes[3].parent, nodes[2])
        # Siblings are sorted by input index.
        self.assertEqual(tuple(c.index for c in nodes[0].children), (1, 2))

    def test_disjoint_contours_are_separate_roots(self):
        roots, nodes = contour_nesting(
            [circle(0, 0, 1), circle(5, 0, 1), circle(10, 0, 1)])
        self.assertEqual(len(roots), 3)
        self.assertTrue(all(node.depth == 0 for node in nodes))
        self.assertTrue(all(not node.is_hole for node in nodes))
        self.assertTrue(all(node.parent is None for node in nodes))

    def test_empty_input(self):
        roots, nodes = contour_nesting([])
        self.assertEqual(roots, ())
        self.assertEqual(nodes, ())

    def test_mixed_segment_types(self):
        # Requirement 2: lines, Beziers and arcs judged geometrically.
        outer = mixed_outer()
        inner = circle(5, 5, 2)
        roots, nodes = contour_nesting([outer, inner])
        self.assertEqual(nodes[0].depth, 0)
        self.assertEqual(nodes[1].depth, 1)
        self.assertTrue(nodes[1].is_hole)
        self.assertIs(nodes[1].parent, nodes[0])

    def test_result_is_independent_of_input_order(self):
        outer = rect(0, 0, 20, 20)
        middle = rect(2, 2, 18, 18)
        inner = circle(10, 10, 3)
        _, forward = contour_nesting([outer, middle, inner])
        _, shuffled = contour_nesting([inner, outer, middle])

        def depth_hole_map(nodes):
            return sorted((node.depth, node.is_hole) for node in nodes)
        self.assertEqual(depth_hole_map(forward), depth_hole_map(shuffled))
        # Each node still reports its own original input index.
        self.assertEqual([node.index for node in shuffled], [0, 1, 2])


class TestOrientationInvariance(unittest.TestCase):
    """Requirement 2: reversing orientation must not change even-odd result."""

    def test_reversed_middle_contour(self):
        outer = circle(0, 0, 10)
        middle = circle(0, 0, 6)
        inner = circle(0, 0, 2)
        _, reference = contour_nesting([outer, middle, inner])
        _, reversed_middle = contour_nesting(
            [outer, middle.reversed(), inner])

        for ref, rev in zip(reference, reversed_middle):
            self.assertEqual(ref.depth, rev.depth)
            self.assertEqual(ref.is_hole, rev.is_hole)
            self.assertEqual(ref.parent is None, rev.parent is None)

    def test_all_contours_reversed(self):
        outer = rect(0, 0, 20, 20)
        middle = rect(2, 2, 18, 18)
        inner = circle(10, 10, 3)
        _, reference = contour_nesting([outer, middle, inner])
        _, flipped = contour_nesting(
            [outer.reversed(), middle.reversed(), inner.reversed()])
        self.assertEqual(
            [n.depth for n in reference], [n.depth for n in flipped])
        self.assertEqual(
            [n.is_hole for n in reference], [n.is_hole for n in flipped])

    def test_clockwise_and_counterclockwise_circles_nest_alike(self):
        ccw = [circle(0, 0, r, ccw=True) for r in (9, 6, 3)]
        cw = [circle(0, 0, r, ccw=False) for r in (9, 6, 3)]
        _, n_ccw = contour_nesting(ccw)
        _, n_cw = contour_nesting(cw)
        self.assertEqual([n.depth for n in n_ccw], [0, 1, 2])
        self.assertEqual([n.depth for n in n_cw], [0, 1, 2])
        self.assertEqual([n.is_hole for n in n_cw],
                         [False, True, False])


class TestErrorClassification(unittest.TestCase):
    """Requirement 3: explicit errors for every ambiguous/bad situation."""

    def test_unclosed_open_path(self):
        open_path = Path(Line(0j, 1 + 1j), Line(1 + 1j, 2j))
        with self.assertRaises(UnclosedContourError):
            contour_nesting([open_path])

    def test_empty_path(self):
        with self.assertRaises(UnclosedContourError):
            contour_nesting([Path()])

    def test_non_path_input(self):
        with self.assertRaises(TypeError):
            contour_nesting([42])

    def test_self_intersecting_bowtie(self):
        # The two diagonals of this quadrilateral cross in the middle.
        bowtie = polygon(0j, 4 + 4j, 4 + 0j, 0 + 4j)
        with self.assertRaises(SelfIntersectingContourError):
            contour_nesting([bowtie])

    def test_self_touching_figure_eight(self):
        # Two square lobes meeting at a single shared vertex.
        figure_eight = polygon(
            0j, 2 + 0j, 2 + 2j, 0 + 2j, 0j, 0 - 2j, 2 - 2j, 2 + 0j)
        with self.assertRaises(SelfIntersectingContourError):
            contour_nesting([figure_eight])

    def test_self_intersecting_loop_inside_bezier(self):
        # The cubic's control points force a loop before the path closes.
        looped = Path(
            CubicBezier(0j, 4 + 2j, -2 + 2j, 2 + 0j),
            Line(2 + 0j, 2 - 2j), Line(2 - 2j, -2j), Line(-2j, 0j))
        with self.assertRaises(SelfIntersectingContourError):
            contour_nesting([looped])

    def test_duplicate_paths(self):
        c = circle(0, 0, 5)
        with self.assertRaises(DuplicateContourError):
            contour_nesting([c, c.reversed()])

    def test_duplicate_with_different_segmentation(self):
        # Same geometric circle expressed through an SVG path string.
        c = circle(0, 0, 5)
        parsed = parse_path(
            "M -5,0 A 5,5 0 1,1 5,0 A 5,5 0 1,1 -5,0")
        with self.assertRaises(DuplicateContourError):
            contour_nesting([c, parsed])

    def test_transverse_rectangle_crossing(self):
        r1 = rect(0, 0, 4, 4)
        r2 = rect(2, -2, 6, 2)
        with self.assertRaises(ContourIntersectionError):
            contour_nesting([r1, r2])

    def test_externally_tangent_circles(self):
        # Requirement 4: tangency is a contact, not a crossing.
        with self.assertRaises(ContourContactError):
            contour_nesting([circle(0, 0, 2), circle(4, 0, 2)])

    def test_internally_tangent_circles(self):
        with self.assertRaises(ContourContactError):
            contour_nesting([circle(0, 0, 6), circle(4, 0, 2)])

    def test_tangent_rectangles_at_vertex(self):
        outer = rect(0, 0, 4, 4)
        inner = rect(1, 1, 3, 4)  # top edge touches the outer top edge
        with self.assertRaises(ContourContactError):
            contour_nesting([outer, inner])

    def test_shared_edge_is_contact_not_crossing(self):
        left = rect(0, 0, 4, 4)
        right = rect(4, 0, 8, 4)  # shares the x=4 edge
        with self.assertRaises(ContourContactError):
            contour_nesting([left, right])

    def test_contact_at_closing_joint_is_classified(self):
        # Outer square starts at (4,4), the point where the inner square
        # touches it; exercises wrap-around sampling at the path joint.
        outer = polygon(4 + 4j, 0 + 4j, 0j, 4 + 0j)
        inner = polygon(4 + 4j, 8 + 4j, 8 + 8j, 4 + 8j)
        with self.assertRaises(ContourContactError):
            contour_nesting([outer, inner])

    def test_non_positive_tolerance_rejected(self):
        outer, inner = circle(0, 0, 5), circle(0, 0, 2)
        with self.assertRaises(ValueError):
            contour_nesting([outer, inner], tol=0)
        with self.assertRaises(ValueError):
            contour_nesting([outer, inner], tol=-1.0)

    def test_all_errors_are_contour_nesting_errors(self):
        for exc in (UnclosedContourError, SelfIntersectingContourError,
                    DuplicateContourError, ContourIntersectionError,
                    ContourContactError, AmbiguousContainmentError):
            self.assertTrue(issubclass(exc, ContourNestingError))
            self.assertTrue(issubclass(exc, ValueError))


class TestToleranceAndGeometry(unittest.TestCase):
    """Requirement 3: tolerance handling and original-geometry preservation."""

    def test_custom_tolerance_accepted(self):
        outer, inner = circle(0, 0, 5), circle(0, 0, 2)
        roots, nodes = contour_nesting([outer, inner], tol=1e-6)
        self.assertEqual(nodes[1].depth, 1)

    def test_node_keeps_original_path_object(self):
        outer, inner = circle(0, 0, 5), circle(0, 0, 2)
        _, nodes = contour_nesting([outer, inner])
        self.assertIs(nodes[0].path, outer)
        self.assertIs(nodes[1].path, inner)

    def test_node_repr_contains_identity(self):
        _, nodes = contour_nesting(
            [circle(0, 0, 5), circle(0, 0, 2)])
        text = repr(nodes[1])
        self.assertIn("index=1", text)
        self.assertIn("depth=1", text)
        self.assertIn("is_hole=True", text)

    def test_arc_containment_matches_known_rings(self):
        # Four concentric rings: depths 0..3, alternating hole identity.
        rings = [circle(0, 0, r) for r in (8, 6, 4, 2)]
        roots, nodes = contour_nesting(rings)
        self.assertEqual([n.depth for n in nodes], [0, 1, 2, 3])
        self.assertEqual([n.is_hole for n in nodes],
                         [False, True, False, True])
        self.assertEqual(len(roots), 1)


if __name__ == "__main__":
    unittest.main()
