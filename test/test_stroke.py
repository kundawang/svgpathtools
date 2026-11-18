import unittest
from copy import deepcopy
from math import cos, sin, radians, pi

from svgpathtools import Path, Line, Arc, CubicBezier, polyline, polygon
from svgpathtools.stroke import stroke_outline


def outline_points(outline, samples_per_seg=1):
    """Sampled points along an outline (endpoints of every segment, plus
    midpoints so arcs contribute)."""
    pts = []
    for seg in outline:
        for k in range(samples_per_seg):
            pts.append(seg.point(k / samples_per_seg))
    return pts


def polygon_area(pts):
    area = 0.0
    for i in range(len(pts)):
        z1 = pts[i]
        z2 = pts[(i + 1) % len(pts)]
        area += z1.real * z2.imag - z2.real * z1.imag
    return abs(area) / 2.0


def outline_area(outline):
    return polygon_area(outline_points(outline, samples_per_seg=64))


class TestStrokePolyline(unittest.TestCase):
    """L-shaped polyline: exact outline vertices, dimensions, joins."""

    def setUp(self):
        # L shape: (0,0) -> (10,0) -> (10,10), stroked with width 2.
        self.L = Path(Line(0, 10), Line(10, 10 + 10j))

    def test_butt_miter_exact_vertices(self):
        outlines = stroke_outline(self.L, 2, cap="butt", join="miter")
        self.assertEqual(len(outlines), 1)
        outline = outlines[0]
        self.assertTrue(outline.isclosed())
        expected = [1j, 9 + 1j, 9 + 10j, 11 + 10j, 11 - 1j, -1j]
        self.assertEqual(len(outline), len(expected))
        for seg, z in zip(outline, expected):
            self.assertAlmostEqual(seg.start.real, z.real)
            self.assertAlmostEqual(seg.start.imag, z.imag)
        # bounding box of the stroke region
        self.assertEqual(outline.bbox(), (0.0, 11.0, -1.0, 10.0))
        # area: 11x2 vertical bar + 9x2 horizontal bar = 40
        self.assertAlmostEqual(outline_area(outline), 40.0)

    def test_bevel_join(self):
        outlines = stroke_outline(self.L, 2, cap="butt", join="bevel")
        self.assertEqual(len(outlines), 1)
        pts = [seg.start for seg in outlines[0]]
        # outer corner is clipped: both bevel endpoints appear
        self.assertIn(11 + 0j, [complex(round(p.real, 9), round(p.imag, 9))
                                for p in pts] + pts)
        self.assertTrue(any(abs(p - 11) < 1e-9 for p in pts))
        self.assertTrue(any(abs(p - (10 - 1j)) < 1e-9 for p in pts))
        # no point farther than h*sqrt(2) from the corner vertex
        self.assertTrue(all(abs(p - 10) <= 2**0.5 + 1e-9
                            for p in pts if abs(p - 10) < 5))

    def test_round_join_uses_arc(self):
        outlines = stroke_outline(self.L, 2, cap="butt", join="round")
        arcs = [s for s in outlines[0] if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 1)
        arc = arcs[0]
        self.assertAlmostEqual(abs(arc.radius), 2**0.5 * 1.0)  # h + 1j*h
        # arc is centered on the vertex (10, 0) with radius h = 1
        self.assertAlmostEqual(abs(arc.center - 10), 0.0, places=9)
        self.assertAlmostEqual(abs(arc.start - 10), 1.0, places=9)
        self.assertAlmostEqual(abs(arc.end - 10), 1.0, places=9)

    def test_miter_limit_clips_to_bevel(self):
        # 170-degree turn: miter tip distance = h / cos(85 deg) ~ 11.47
        d = cos(radians(-170)) + 1j * sin(radians(-170))
        path = Path(Line(0, 10), Line(10, 10 + 5 * d))
        # miter_limit=4 allows 4*h = 4 < 11.47 -> bevel
        beveled = stroke_outline(path, 2, join="miter", miter_limit=4)[0]
        pts = [seg.start for seg in beveled]
        # bevel points sit at distance h=1 from the vertex, cap points at
        # ~10; a miter spike at ~11.47 must not appear
        self.assertTrue(all(abs(p - 10) < 11 for p in pts))
        # miter_limit=12 allows 12 > 11.47 -> miter survives
        mitered = stroke_outline(path, 2, join="miter", miter_limit=12)[0]
        pts = [seg.start for seg in mitered]
        self.assertTrue(any(abs(p - 10) > 11 for p in pts))

    def test_hairpin_reversal_is_stable(self):
        # 180-degree turn: miter would be infinite; result must be bounded.
        path = polyline(0, 10, 10 + 1e-9j, 0 + 1e-9j)
        for join in ("miter", "bevel", "round"):
            outlines = stroke_outline(path, 2, join=join)
            self.assertEqual(len(outlines), 1)
            self.assertTrue(outlines[0].isclosed())
            pts = outline_points(outlines[0])
            self.assertTrue(all(abs(p) < 20 for p in pts))


class TestStrokeCaps(unittest.TestCase):
    def test_butt_cap_line(self):
        outlines = stroke_outline(Path(Line(0, 10)), 2, cap="butt")
        self.assertEqual(len(outlines), 1)
        self.assertEqual(outlines[0].bbox(), (0.0, 10.0, -1.0, 1.0))
        self.assertAlmostEqual(outline_area(outlines[0]), 20.0)

    def test_square_cap_extends_by_half_width(self):
        outlines = stroke_outline(Path(Line(0, 10)), 2, cap="square")
        self.assertEqual(len(outlines), 1)
        self.assertEqual(outlines[0].bbox(), (-1.0, 11.0, -1.0, 1.0))
        self.assertAlmostEqual(outline_area(outlines[0]), 24.0)

    def test_round_cap_uses_semicircle_arcs(self):
        outlines = stroke_outline(Path(Line(0, 10)), 2, cap="round")
        self.assertEqual(len(outlines), 1)
        outline = outlines[0]
        self.assertTrue(outline.isclosed())
        arcs = [s for s in outline if isinstance(s, Arc)]
        self.assertEqual(len(arcs), 2)
        xmin, xmax, ymin, ymax = outline.bbox()
        self.assertAlmostEqual(xmin, -1.0)
        self.assertAlmostEqual(xmax, 11.0)
        self.assertAlmostEqual(ymin, -1.0)
        self.assertAlmostEqual(ymax, 1.0)
        # area = rectangle + two semicircles = 20 + pi
        self.assertAlmostEqual(outline_area(outline), 20 + pi, places=2)


class TestStrokeArc(unittest.TestCase):
    """Stroking a circular arc: outline stays between the two offset
    circles and has the expected radial dimensions."""

    def test_quarter_circle(self):
        # Quarter circle of radius 10 centered at the origin.
        arc = Arc(10 + 0j, 10 + 10j, 0.0, False, True, 0 + 10j)
        path = Path(arc)
        outlines = stroke_outline(path, 2, cap="butt", join="round")
        self.assertEqual(len(outlines), 1)
        outline = outlines[0]
        self.assertTrue(outline.isclosed())
        radii = [abs(p) for p in outline_points(outline, samples_per_seg=4)]
        # every outline point lies between the inner and outer offset arcs
        self.assertGreaterEqual(min(radii), 9.0 - 0.05)
        self.assertLessEqual(max(radii), 11.0 + 0.05)
        # and both offset radii are actually reached
        self.assertLess(min(radii), 9.0 + 0.05)
        self.assertGreater(max(radii), 11.0 - 0.05)

    def test_arc_stroke_bbox(self):
        arc = Arc(10 + 0j, 10 + 10j, 0.0, False, True, 0 + 10j)
        outline = stroke_outline(Path(arc), 2, cap="round")[0]
        xmin, xmax, ymin, ymax = outline.bbox()
        self.assertAlmostEqual(xmin, -1.0, places=2)
        self.assertAlmostEqual(xmax, 11.0, places=2)
        self.assertAlmostEqual(ymin, -1.0, places=2)
        self.assertAlmostEqual(ymax, 11.0, places=2)


class TestStrokeClosedLoop(unittest.TestCase):
    def test_square_loop_two_outlines(self):
        loop = polygon(0, 10, 10 + 10j, 10j)
        outlines = stroke_outline(loop, 2, join="miter")
        self.assertEqual(len(outlines), 2)
        bboxes = sorted(o.bbox() for o in outlines)
        # outer boundary loop (smaller xmin sorts first)
        self.assertEqual(bboxes[0], (-1.0, 11.0, -1.0, 11.0))
        # inner boundary loop
        self.assertEqual(bboxes[1], (1.0, 9.0, 1.0, 9.0))
        for o in outlines:
            self.assertTrue(o.isclosed())
        areas = sorted(outline_area(o) for o in outlines)
        self.assertAlmostEqual(areas[0], 64.0)
        self.assertAlmostEqual(areas[1], 144.0)

    def test_closed_loop_join_at_seam(self):
        # The seam vertex (start == end) gets a proper join, not a cap.
        loop = polygon(0, 10, 10 + 10j, 10j)
        outlines = stroke_outline(loop, 2, join="round")
        self.assertEqual(len(outlines), 2)
        arcs = [s for o in outlines for s in o if isinstance(s, Arc)]
        # 4 rounded corners on the outer loop only
        self.assertEqual(len(arcs), 4)


class TestStrokeRobustness(unittest.TestCase):
    def test_original_path_unchanged(self):
        path = Path(Line(0, 10), CubicBezier(10, 20, 20 + 10j, 5 + 5j),
                    Arc(5 + 5j, 3 + 3j, 0.0, False, True, 1 + 1j))
        snapshot = deepcopy(path)
        stroke_outline(path, 2, cap="round", join="round")
        self.assertEqual(path, snapshot)
        self.assertEqual(path.d(), snapshot.d())

    def test_reversed_path_same_region(self):
        path = Path(Line(0, 10), Line(10, 10 + 10j), Line(10 + 10j, 3 + 4j))
        forward = stroke_outline(path, 2, cap="square", join="bevel")
        backward = stroke_outline(path.reversed(), 2, cap="square",
                                  join="bevel")
        self.assertEqual(len(forward), len(backward))
        for f, b in zip(forward, backward):
            self.assertAlmostEqual(outline_area(f), outline_area(b))
            fb, bb = f.bbox(), b.bbox()
            for x, y in zip(fb, bb):
                self.assertAlmostEqual(x, y)

    def test_zero_and_negative_width_raise(self):
        path = Path(Line(0, 10))
        for bad in (0, -1, -0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                stroke_outline(path, bad)

    def test_invalid_styles_raise(self):
        path = Path(Line(0, 10))
        with self.assertRaises(ValueError):
            stroke_outline(path, 2, cap="triangle")
        with self.assertRaises(ValueError):
            stroke_outline(path, 2, join="arcs")
        with self.assertRaises(ValueError):
            stroke_outline(path, 2, miter_limit=0.5)

    def test_zero_length_segments_skipped(self):
        path = Path(Line(5 + 5j, 5 + 5j), Line(5 + 5j, 10 + 5j),
                    Line(10 + 5j, 10 + 5j))
        outlines = stroke_outline(path, 2)
        self.assertEqual(len(outlines), 1)
        self.assertEqual(outlines[0].bbox(), (5.0, 10.0, 4.0, 6.0))

    def test_all_degenerate_path_returns_empty(self):
        self.assertEqual(stroke_outline(Path(Line(3, 3)), 2), [])
        self.assertEqual(stroke_outline(Path(), 2), [])

    def test_self_intersecting_path_is_stable(self):
        # figure-eight style self-crossing polyline
        path = polyline(0, 10 + 10j, 10, 0 + 10j, 5 + 5j)
        outlines = stroke_outline(path, 2, join="round", cap="round")
        self.assertEqual(len(outlines), 1)
        self.assertTrue(outlines[0].isclosed())
        xmin, xmax, ymin, ymax = outlines[0].bbox()
        self.assertLessEqual(xmin, 0)
        self.assertGreaterEqual(xmax, 10)

    def test_huge_coordinates(self):
        # horizontal segment along the real axis at x ~ 1e12
        path = Path(Line(1e12 + 0j, 1e12 + 10))
        outlines = stroke_outline(path, 2)
        self.assertEqual(len(outlines), 1)
        xmin, xmax, ymin, ymax = outlines[0].bbox()
        self.assertAlmostEqual(xmin, 1e12, delta=1e-3)
        self.assertAlmostEqual(xmax, 1e12 + 10, delta=1e-3)
        self.assertAlmostEqual(ymin, -1.0, delta=1e-3)
        self.assertAlmostEqual(ymax, 1.0, delta=1e-3)

    def test_non_finite_coordinates_raise(self):
        with self.assertRaises(ValueError):
            stroke_outline(Path(Line(0, complex(float("inf"), 0))), 2)

    def test_flattening_fallback_is_defined(self):
        # max_depth=1 forces the non-converged chord fallback; the result
        # must still be a closed outline (coarse, but well-defined).
        path = Path(CubicBezier(0, 0, 10 + 10j, 10))
        outlines = stroke_outline(path, 2, max_depth=1)
        self.assertEqual(len(outlines), 1)
        self.assertTrue(outlines[0].isclosed())

    def test_curve_joins_smooth(self):
        # A cubic curving gently: no join spikes on the offset outline.
        path = Path(CubicBezier(0, 10 + 2j, 20 + 2j, 30))
        outlines = stroke_outline(path, 2, join="round")
        self.assertEqual(len(outlines), 1)
        pts = outline_points(outlines[0])
        # offset of a shallow curve stays within a thin band around it
        self.assertTrue(all(-1.5 <= p.imag <= 3.5 for p in pts))


if __name__ == "__main__":
    unittest.main()
