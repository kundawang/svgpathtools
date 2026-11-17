"""Tests for svgpathtools.offset (tool-radius contour offsetting)."""

from __future__ import annotations

import math
import unittest

from svgpathtools import (Path, Line, CubicBezier, QuadraticBezier, Arc,
                          parse_path)
from svgpathtools.offset import OffsetError, offset_path, offset_segment


def _sample_points(curve, n=33):
    return [curve.point(i / (n - 1)) for i in range(n)]


def _min_distance_to_polyline(point, polyline):
    """Distance from point to a dense polyline (list of complex points)."""
    best = float("inf")
    for a, b in zip(polyline[:-1], polyline[1:]):
        ab = b - a
        if ab == 0:
            best = min(best, abs(point - a))
            continue
        t = ((point - a).real * ab.real + (point - a).imag * ab.imag) \
            / (abs(ab) ** 2)
        t = min(max(t, 0.0), 1.0)
        best = min(best, abs(point - (a + t * ab)))
    return best


class TestOffsetLine(unittest.TestCase):
    def test_line_offset_is_exact(self):
        line = Line(0, 100)
        (off,) = offset_segment(line, 5)
        self.assertIsInstance(off, Line)
        # normal = -1j * unit_tangent = -1j for a line along +x
        self.assertAlmostEqual(off.start, -5j)
        self.assertAlmostEqual(off.end, 100 - 5j)

    def test_line_offset_negative_distance(self):
        (off,) = offset_segment(Line(0, 100), -2.5)
        self.assertAlmostEqual(off.start, 2.5j)
        self.assertAlmostEqual(off.end, 100 + 2.5j)

    def test_line_path_distance(self):
        path = Path(Line(10 + 20j, 110 + 20j))
        off = offset_path(path, 3, tolerance=1e-6)
        self.assertEqual(len(off), 1)
        for p in _sample_points(off):
            self.assertAlmostEqual(p.imag, 17.0, places=9)
            self.assertTrue(10 <= p.real <= 110)

    def test_zero_length_line_dropped(self):
        self.assertEqual(offset_segment(Line(5 + 5j, 5 + 5j), 1), [])


class TestOffsetCubic(unittest.TestCase):
    def test_straight_cubic_offsets_like_line(self):
        # A cubic that is exactly the straight line y=0, 0<=x<=90.
        cubic = CubicBezier(0, 30, 60, 90)
        pieces = offset_segment(cubic, 4, tolerance=1e-9)
        self.assertEqual(len(pieces), 1)
        for p in _sample_points(pieces[0]):
            self.assertAlmostEqual(p.imag, -4.0, places=6)

    def test_curved_cubic_distance_within_tolerance(self):
        cubic = CubicBezier(0, 50 + 80j, 100 - 80j, 150)
        tol = 1e-3
        pieces = offset_segment(cubic, 6, tolerance=tol)
        original_samples = _sample_points(cubic, 401)
        for piece in pieces:
            for p in _sample_points(piece, 17):
                d = _min_distance_to_polyline(p, original_samples)
                self.assertLessEqual(d, 6 + 10 * tol)
                # and the offset point really came from the offset side
                self.assertGreaterEqual(d, 6 - 10 * tol)

    def test_piece_chain_is_continuous(self):
        cubic = CubicBezier(0, 10 + 60j, 40 - 60j, 100 + 10j)
        pieces = offset_segment(cubic, 3, tolerance=1e-6)
        for a, b in zip(pieces[:-1], pieces[1:]):
            self.assertEqual(a.end, b.start)

    def test_unsupported_segment_types_rejected(self):
        with self.assertRaises(TypeError):
            offset_segment(QuadraticBezier(0, 1, 2), 1)
        with self.assertRaises(TypeError):
            offset_segment(Arc(0, 1 + 1j, 0, False, True, 2j), 1)

    def test_cusp_rejected(self):
        # derivative vanishes at t = 0.5 (cusp: out and back)
        cusp = CubicBezier(0, 1, 0, 1)
        with self.assertRaises(OffsetError):
            offset_segment(cusp, 1, tolerance=1e-3)


class TestJoins(unittest.TestCase):
    def setUp(self):
        # Sharp 90-degree outside corner at 100+0j (offset to the right,
        # which is the outside of this left turn... checked via geometry).
        self.corner = 100 + 0j
        self.elbow = Path(Line(0, self.corner), Line(self.corner, 100 + 100j))

    def test_round_join_arc_radius(self):
        d = 10.0
        off = offset_path(self.elbow, d, join="round")
        arcs = [s for s in off if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 1)
        arc = arcs[0]
        self.assertTrue(off.iscontinuous())
        for p in _sample_points(arc, 9):
            self.assertAlmostEqual(abs(p - self.corner), d, places=9)

    def test_bevel_join(self):
        d = 10.0
        off = offset_path(self.elbow, d, join="bevel")
        self.assertTrue(off.iscontinuous())
        self.assertEqual(len(off), 3)
        bevel = off[1]
        self.assertIsInstance(bevel, Line)
        self.assertAlmostEqual(abs(bevel.start - self.corner), d, places=9)
        self.assertAlmostEqual(abs(bevel.end - self.corner), d, places=9)

    def test_join_connects_exactly(self):
        for join in ("round", "bevel"):
            off = offset_path(self.elbow, 7.5, join=join)
            for a, b in zip(off[:-1], off[1:]):
                self.assertEqual(a.end, b.start)

    def test_inside_corner_is_trimmed(self):
        # Right turn: offsetting to the right puts the corner on the inside.
        elbow = Path(Line(0, 100), Line(100, 100 - 100j))
        off = offset_path(elbow, 5, join="round")
        self.assertTrue(off.iscontinuous())
        # No join segment is inserted at an inside corner: the two offset
        # lines are trimmed to their intersection at (95, -5)... i.e. the
        # single shared endpoint 95-5j.
        self.assertEqual(len(off), 2)
        self.assertAlmostEqual(off[0].end, 95 - 5j, places=9)
        self.assertAlmostEqual(off[1].start, 95 - 5j, places=9)

    def test_smooth_joint_needs_no_join_segment(self):
        path = Path(Line(0, 50), Line(50, 100))
        off = offset_path(path, 4, join="round")
        self.assertEqual(len(off), 2)
        self.assertTrue(off.iscontinuous())


class TestClosedPaths(unittest.TestCase):
    def test_square_outward_round(self):
        square = Path(Line(0, 100), Line(100, 100 + 100j),
                      Line(100 + 100j, 100j), Line(100j, 0))
        off = offset_path(square, 10, join="round")
        self.assertTrue(off.iscontinuous())
        self.assertTrue(off.isclosed())
        arcs = [s for s in off if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 4)
        for arc, corner in zip(arcs, [100, 100 + 100j, 100j, 0]):
            for p in _sample_points(arc, 5):
                self.assertAlmostEqual(abs(p - corner), 10.0, places=9)

    def test_square_inward_bevel(self):
        square = Path(Line(0, 100), Line(100, 100 + 100j),
                      Line(100 + 100j, 100j), Line(100j, 0))
        off = offset_path(square, -10, join="bevel")
        self.assertTrue(off.isclosed())
        # Inward offset of a CCW square: corners are inside corners and get
        # trimmed, leaving 4 straight sides of length 80.
        self.assertEqual(len(off), 4)
        for seg in off:
            self.assertIsInstance(seg, Line)
            self.assertAlmostEqual(seg.length(), 80.0, places=9)

    def test_closed_output_serializes(self):
        square = Path(Line(0, 100), Line(100, 100 + 100j),
                      Line(100 + 100j, 100j), Line(100j, 0))
        off = offset_path(square, 5, join="round")
        d = off.d()
        reparsed = parse_path(d)
        self.assertTrue(reparsed.isclosed())
        self.assertAlmostEqual(reparsed.point(0), off.point(0))
        self.assertAlmostEqual(reparsed.point(1), off.point(1))


class TestArcFixture(unittest.TestCase):
    """Offset of a cubic-approximated circular arc stays concentric."""

    def test_quarter_circle_offset(self):
        radius = 80.0
        arc = Arc(radius, radius + 1j * radius, 0, False, True, 1j * radius)
        path = Path(arc)
        path.approximate_arcs_with_cubics(error=1e-4)
        d = 12.0
        tol = 1e-2
        off = offset_path(path, d, tolerance=tol)
        self.assertTrue(off.iscontinuous())
        center = arc.center
        distances = [abs(p - center) for p in _sample_points(off, 65)]
        # The offset of a circle is a concentric circle; all sampled
        # distances must agree with a single radius within tolerance.
        expected = sum(distances) / len(distances)
        self.assertGreater(abs(expected - radius), d - 2 * tol)
        for dist in distances:
            self.assertAlmostEqual(dist, expected, delta=3 * tol)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.path = Path(Line(0, 100))

    def test_bad_tolerance(self):
        for bad in (0, -1, -1e-9, float("nan"), float("inf"), "0.1", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                offset_path(self.path, 1, tolerance=bad)

    def test_bad_distance(self):
        for bad in (float("nan"), float("inf"), 1 + 1j, "5", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                offset_path(self.path, bad)

    def test_bad_join(self):
        for bad in ("miter", "ROUND", "", None, 1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                offset_path(self.path, 1, join=bad)

    def test_zero_length_only_path(self):
        with self.assertRaises(OffsetError):
            offset_path(Path(Line(3 + 3j, 3 + 3j)), 1)

    def test_discontinuous_path_rejected(self):
        with self.assertRaises(ValueError):
            offset_path(Path(Line(0, 50), Line(60, 100)), 1)

    def test_self_intersecting_path_rejected(self):
        bowtie = Path(Line(0, 100 + 100j), Line(100 + 100j, 100),
                      Line(100, 100j), Line(100j, 0))
        with self.assertRaises(OffsetError):
            offset_path(bowtie, 1)

    def test_oversized_offset_rejected(self):
        # Inward offset larger than the circle's radius collapses.
        # Quarter-circle cubics rotated by exact multiples of 1j so the
        # path is exactly continuous and closed.
        radius = 50.0
        k = 4.0 / 3.0 * math.tan(math.pi / 8)
        quarter = CubicBezier(radius, radius + k * radius * 1j,
                              radius * (k + 1j), radius * 1j)
        circle = Path(*[CubicBezier(*[b * 1j ** m for b in quarter.bpoints()])
                        for m in range(4)])
        self.assertTrue(circle.iscontinuous())
        with self.assertRaises(OffsetError):
            offset_path(circle, -radius * 1.5, tolerance=1e-3)

    def test_arc_input_rejected_with_hint(self):
        arc = Arc(0, 10 + 10j, 0, False, True, 10j)
        with self.assertRaises(TypeError):
            offset_path(Path(arc), 1)

    def test_zero_distance_returns_copy(self):
        path = Path(Line(0, 100), CubicBezier(100, 120, 140, 150))
        off = offset_path(path, 0)
        self.assertEqual(off, path)


if __name__ == "__main__":
    unittest.main()
