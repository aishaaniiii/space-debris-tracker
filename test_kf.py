from kf import KF
import unittest

class testKF(unittest.TestCase):
    def test_can_construct(self):
        x = 0.2
        v = 0.5

        kf = KF(x_init = x, v_init = v)
        self.assertAlmostEqual(kf.x[0], x)
        self.assertAlmostEqual(kf.x[1], v)