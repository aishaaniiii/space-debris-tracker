import matplotlib.pyplot as plt
import numpy as np 
from unittest 

class KF: 
    def init(self, x_init, v_init):
        #mean of the state
        self.x = np.array([x_init, v_init])
        #covaraince of the state
        self.P = np.eye(2) #ask what this does


kf = KF(x_init = 0.2, v_init = 0.5)

#this is a block just to test our code 

class testKF(inittest.TestCase):
    def test_can_construct(self):
        x = 0.2
        v = 0.5

        kf = KF(x_init = x, v_init = v)
        self.assertAlmostEqual(kf.x[0] = x)
        self.assertAlmostEqual(kf.x[1] = v)