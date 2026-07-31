import kf as python_kf

import os 
cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys 
sys.path.append(cur_file_path + '/../cpp/build')

import kf_cpp as cpp_kf

import numpy as np
import unittest
from parameterized import parameterized_class

@parameterized_class([
    {"KF": python_kf.KF},
    {"KF": cpp_kf.KF}
])

#once we have actual data to work with, we can also put in numerical tests
class testKF(unittest.TestCase):
    def test_can_construct(self):
        x = 0.2
        v = 0.5

        kf = self.KF(x,v, 1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    def test_predict(self):
        x = 0.2
        v = 0.5

        kf = self.KF(x,  v, 1.2)
        #runnning predict several times to see effect of accumulating uncertainty
        for i in range(10):
            det_1 = np.linalg.det(kf.cov) #if the uncertainity increases, so should the determinant of the predict matrix 
            kf.predict( 0.1)
            det_2 = np.linalg.det(kf.cov)
            print(det_1, det_2)
            self.assertGreater(det_2,det_1) #should come out as true

        #also checks if mean and covariance are of right dimensions 
        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2,))

    def test_update(self): #not a complete test yet
        x = 0.2
        v = 0.5

        kf = self.KF(x,  v,  1.2)

        det_1 = np.linalg.det(kf.cov)
        kf.update( 0.8, 0.5)
        det_2 = np.linalg.det(kf.cov)

        self.assertLess(det_2,det_1)

