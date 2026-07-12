import numpy as np

class KF: 
    def __init__(self, x_init, v_init):
        #mean of the state
        self.x = np.array([x_init, v_init])
        #covaraince of the state
        self.P = np.eye(2) #this makes an identity matrix, 2 on the diagonal, initial covariance matrix 