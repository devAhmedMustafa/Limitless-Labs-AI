import unittest
from SkeletonScan.main import calc_angle

class TestCalcAngle(unittest.TestCase):
    def test_right_angle(self):
        a = (1, 0)
        b = (0, 0)
        c = (0, 1)
        angle = calc_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=5)

    def test_straight_angle(self):
        a = (1, 0)
        b = (0, 0)
        c = (-1, 0)
        angle = calc_angle(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=5)

    def test_acute_angle(self):
        a = (1, 1)
        b = (0, 0)
        c = (1, 0)
        angle = calc_angle(a, b, c)
        self.assertAlmostEqual(angle, 45.0, places=5)

    def test_obtuse_angle(self):
        a = (-1, 1)
        b = (0, 0)
        c = (1, 0)
        angle = calc_angle(a, b, c)
        self.assertAlmostEqual(angle, 135.0, places=5)