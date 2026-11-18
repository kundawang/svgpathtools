"""Tests for svgpathtools.cubic2quad (cubic -> quadratic conversion)."""

import unittest
from math import inf, nan

from svgpathtools import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                          parse_path)
from svgpathtools.cubic2quad import (QuadraticConversionError,
                                     cubic_to_quadratics, path_to_quadratics)


def dense_sample_error(cubic, quads_with_ranges, samples_per_quad=257):
    """Max distance between the cubic and its quadratic approximation,
    comparing points at equal (interpolated) parameter values."""
    worst = 0.0
    for quad, (t0, t1) in quads_with_ranges:
        for i in range(samples_per_quad):
            s = i / (samples_per_quad - 1)
            d = abs(quad.point(s) - cubic.point(t0 + (t1 - t0) * s))
            worst = max(worst, d)
    return worst


def unit(z):
    return z / abs(z)


# An S-shaped cubic (inflection point at t=0.5).
S_CURVE = CubicBezier(0 + 0j, 0 + 100j, 100 - 100j, 100 + 0j)

# A high-curvature cubic with a tight loop.
LOOPY = CubicBezier(0 + 0j, 300 + 0j, -300 + 200j, 0 + 100j)

# A cubic with a cusp at t=0.5 (derivative vanishes there).
CUSP = CubicBezier(0 + 0j, 1 + 1j, 1 - 1j, 0 + 2j)


class TestCubicToQuadratics(unittest.TestCase):

    def test_s_curve_error_and_endpoints(self):
        for error in (1.0, 0.1, 0.01):
            quads = cubic_to_quadratics(S_CURVE, error,
                                        return_t_ranges=True)
            self.assertGreaterEqual(len(quads), 1)
            # dense-sampling error check
            self.assertLessEqual(dense_sample_error(S_CURVE, quads), error)
            # endpoints preserved exactly
            self.assertEqual(quads[0][0].start, S_CURVE.start)
            self.assertEqual(quads[-1][0].end, S_CURVE.end)
            # endpoint tangent directions preserved as closely as
            # possible (the S-curve's inflection forces the midpoint
            # fallback on some pieces, so allow a small deviation that
            # shrinks with the tolerance)
            self.assertLess(
                abs(unit(quads[0][0].derivative(0))
                    - unit(S_CURVE.derivative(0))), 0.05)
            self.assertLess(
                abs(unit(quads[-1][0].derivative(1))
                    - unit(S_CURVE.derivative(1))), 0.05)

    def test_gentle_cubic_exact_tangents(self):
        # quarter-circle-like cubic: no inflection, well-separated
        # tangents, so the tangent-intersection control is used and
        # endpoint tangent directions are preserved exactly
        k = 0.5522847498307936
        arc = CubicBezier(1 + 0j, 1 + k * 1j, k + 1j, 0 + 1j)
        quads = cubic_to_quadratics(arc, 0.01, return_t_ranges=True)
        self.assertLessEqual(dense_sample_error(arc, quads), 0.01)
        self.assertAlmostEqual(
            abs(unit(quads[0][0].derivative(0)) - unit(arc.derivative(0))),
            0)
        self.assertAlmostEqual(
            abs(unit(quads[-1][0].derivative(1)) - unit(arc.derivative(1))),
            0)
        # interior joints are exactly G1 as well
        for a, b in zip(quads, quads[1:]):
            self.assertAlmostEqual(
                abs(unit(a[0].derivative(1)) - unit(b[0].derivative(0))), 0)

    def test_loopy_curve_error(self):
        for error in (1.0, 0.1):
            quads = cubic_to_quadratics(LOOPY, error, return_t_ranges=True)
            self.assertLessEqual(dense_sample_error(LOOPY, quads), error)

    def test_cusp_converges(self):
        quads = cubic_to_quadratics(CUSP, 1e-3, return_t_ranges=True)
        self.assertLessEqual(dense_sample_error(CUSP, quads), 1e-3)
        self.assertEqual(quads[0][0].start, CUSP.start)
        self.assertEqual(quads[-1][0].end, CUSP.end)

    def test_continuity_and_t_ranges(self):
        quads = cubic_to_quadratics(S_CURVE, 0.5, return_t_ranges=True)
        # exact C0 joints
        for a, b in zip(quads, quads[1:]):
            self.assertEqual(a[0].end, b[0].start)
        # G1 joints (tangent directions agree at interior joins, up to the
        # small deviation the midpoint fallback can introduce)
        for a, b in zip(quads, quads[1:]):
            da, db = a[0].derivative(1), b[0].derivative(0)
            if abs(da) > 0 and abs(db) > 0:
                self.assertLess(abs(unit(da) - unit(db)), 0.05)
        # t-ranges tile [0, 1] contiguously and in order
        self.assertEqual(quads[0][1][0], 0.0)
        self.assertEqual(quads[-1][1][1], 1.0)
        for a, b in zip(quads, quads[1:]):
            self.assertEqual(a[1][1], b[1][0])
            self.assertLess(a[1][0], a[1][1])

    def test_smaller_error_means_more_segments(self):
        n1 = len(cubic_to_quadratics(S_CURVE, 1.0))
        n2 = len(cubic_to_quadratics(S_CURVE, 0.01))
        self.assertLess(n1, n2)

    def test_zero_length_cubic(self):
        cubic = CubicBezier(3 + 4j, 3 + 4j, 3 + 4j, 3 + 4j)
        quads = cubic_to_quadratics(cubic, 0.1, return_t_ranges=True)
        self.assertEqual(len(quads), 1)
        quad, trange = quads[0]
        self.assertEqual(quad.start, quad.control)
        self.assertEqual(quad.control, quad.end)
        self.assertEqual(quad.start, 3 + 4j)
        self.assertEqual(trange, (0.0, 1.0))

    def test_nearly_zero_length_cubic(self):
        p = 1e-9
        cubic = CubicBezier(0, p + 0j, 0 + 1j * p, p + 1j * p)
        quads = cubic_to_quadratics(cubic, 1e-6)
        self.assertEqual(len(quads), 1)

    def test_quadratic_cubic_converts_to_single_quadratic(self):
        # a cubic that is exactly a (degree-elevated) quadratic
        quad = QuadraticBezier(0 + 0j, 1 + 2j, 3 + 0j)
        cubic = CubicBezier(quad.start,
                            (quad.start + 2 * quad.control) / 3,
                            (2 * quad.control + quad.end) / 3,
                            quad.end)
        quads = cubic_to_quadratics(cubic, 1e-9)
        self.assertEqual(len(quads), 1)
        self.assertAlmostEqual(abs(quads[0].control - quad.control), 0)

    def test_invalid_tolerance(self):
        for bad in (0, -1, nan, inf, -inf):
            with self.assertRaises(QuadraticConversionError):
                cubic_to_quadratics(S_CURVE, bad)

    def test_non_finite_control_points(self):
        for bad in (nan + 0j, inf + 0j, 0 + inf * 1j, complex(nan, nan)):
            cubic = CubicBezier(0 + 0j, bad, 1 + 1j, 2 + 0j)
            with self.assertRaises(QuadraticConversionError):
                cubic_to_quadratics(cubic, 0.1)

    def test_max_segments_limit(self):
        with self.assertRaises(QuadraticConversionError):
            cubic_to_quadratics(S_CURVE, 1e-6, max_segments=2)

    def test_max_depth_limit(self):
        with self.assertRaises(QuadraticConversionError):
            cubic_to_quadratics(S_CURVE, 1e-9, max_depth=2)
        # with enough depth the same tolerance succeeds
        cubic_to_quadratics(S_CURVE, 1e-9, max_depth=40,
                            max_segments=10 ** 6)

    def test_tiny_tolerance_terminates(self):
        quads = cubic_to_quadratics(S_CURVE, 1e-8, max_depth=40,
                                    max_segments=10 ** 6,
                                    return_t_ranges=True)
        self.assertLessEqual(dense_sample_error(S_CURVE, quads), 1e-8)

    def test_wrong_input_type(self):
        with self.assertRaises(TypeError):
            cubic_to_quadratics(Line(0, 1), 0.1)


class TestPathToQuadratics(unittest.TestCase):

    def test_lines_and_quadratics_kept_and_order_preserved(self):
        line = Line(0 + 0j, 1 + 0j)
        quad = QuadraticBezier(1 + 0j, 2 + 1j, 3 + 0j)
        cubic = CubicBezier(3 + 0j, 4 + 2j, 5 - 2j, 6 + 0j)
        path = Path(line, quad, cubic)
        new_path, t_ranges = path_to_quadratics(path, 0.1,
                                                return_t_ranges=True)
        # original segments kept (same objects), in order
        self.assertIs(new_path[0], line)
        self.assertIs(new_path[1], quad)
        self.assertTrue(all(isinstance(s, QuadraticBezier)
                            for s in new_path[2:]))
        self.assertEqual(t_ranges[0], (0.0, 1.0))
        self.assertEqual(t_ranges[1], (0.0, 1.0))
        # no cubics remain, path stays continuous
        self.assertFalse(any(isinstance(s, CubicBezier) for s in new_path))
        self.assertTrue(new_path.iscontinuous())
        self.assertEqual(new_path.start, path.start)
        self.assertEqual(new_path.end, path.end)

    def test_subpath_order_preserved(self):
        path = parse_path("M 0 0 C 0 1 1 1 1 0 "
                          "M 10 10 L 11 10 "
                          "M 20 20 C 20 21 21 21 21 20")
        new_path = path_to_quadratics(path, 0.01)
        old_subs = path.continuous_subpaths()
        new_subs = new_path.continuous_subpaths()
        self.assertEqual(len(old_subs), len(new_subs))
        for old, new in zip(old_subs, new_subs):
            self.assertEqual(old.start, new.start)
            self.assertEqual(old.end, new.end)
        # the untouched middle subpath is identical
        self.assertEqual(list(old_subs[1]), list(new_subs[1]))

    def test_closed_path_stays_closed(self):
        path = parse_path("M 0 0 C 0 50 50 50 50 0 "
                          "C 50 -50 0 -50 0 0 Z")
        self.assertTrue(path.iscontinuous())
        self.assertTrue(path.isclosed())
        new_path = path_to_quadratics(path, 0.1)
        self.assertTrue(new_path.iscontinuous())
        self.assertTrue(new_path.isclosed())

    def test_whole_path_error(self):
        # dense-sampling check of every converted cubic of a mixed path
        cubic1 = CubicBezier(0 + 0j, 0 + 100j, 100 - 100j, 100 + 0j)
        cubic2 = CubicBezier(100 + 0j, 200 + 150j, 50 + 200j, 150 + 50j)
        path = Path(cubic1, cubic2)
        error = 0.05
        new_path, t_ranges = path_to_quadratics(path, error,
                                                return_t_ranges=True)
        src = {id(cubic1): cubic1, id(cubic2): cubic2}
        # rebuild per-cubic piece lists from the tiling of [0, 1]
        i = 0
        for cubic in (cubic1, cubic2):
            pieces = []
            while i < len(t_ranges):
                pieces.append((new_path[i], t_ranges[i]))
                done = t_ranges[i][1] == 1.0
                i += 1
                if done:
                    break
            self.assertLessEqual(dense_sample_error(cubic, pieces), error)
        self.assertEqual(i, len(new_path))
        del src

    def test_arc_raises(self):
        arc = Arc(0 + 0j, 1 + 1j, 0, False, True, 2 + 0j)
        with self.assertRaises(NotImplementedError):
            path_to_quadratics(Path(arc), 0.1)

    def test_empty_path(self):
        new_path = path_to_quadratics(Path(), 0.1)
        self.assertEqual(len(new_path), 0)


if __name__ == '__main__':
    unittest.main()
