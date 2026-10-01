import numpy as np


#this modification will help later when we add 3 dims.. i think 
iX = 0
iY = 1
iZ = 2
iU = 3
iV = 4
iW = 5
n_var = iW + 1
class KF: 
    def __init__(self, pos_init: np.ndarray, vel_init: np.ndarray, a_var: float) -> None: #self makes it be a specific instance of KF, so the array is stored there instead of the whole class
        self._x = np.zeros(n_var)
        #forcing the input to be floats
        pos_init = np.asarray(pos_init, dtype=float)
        vel_init = np.asarray(vel_init, dtype=float)
        #mean of the state
        #position
        self._x[iX] = pos_init[iX]
        self._x[iY] = pos_init[iY]
        self._x[iZ] = pos_init[iZ]
        #velocity
        self._x[iU] = vel_init[iX]
        self._x[iV] = vel_init[iY]
        self._x[iW] = vel_init[iZ]
         #the underscore makes this variable private
        #acceleration variance 
        self._a_var = a_var 
        #covaraince of the state
        self._P = np.eye(n_var) #this makes an identity matrix, 2 on the diagonal, initial covariance matrix 


    #predict  
    def predict(self, dt: float) -> None:
        #calling predict means that we integrate, which means our uncertainity icnrease
        #equations to predict 
        #new x = Fx
        #new P = F P Ft + G Gt a
        #F Matrix
        F = np.eye(n_var)
        F[iX,iU] = dt
        F[iY,iV] = dt
        F[iZ,iW] = dt

        #G Matrix
        G = np.zeros((n_var,3)) #could make this into a loop maybe 
        #changes in x axis 
        G[iX,0] = 0.5*dt**2
        G[iU,0] = dt

        #changes in y axis
        G[iY,1] = 0.5*dt**2
        G[iV,1] = dt

        #changes in z axis 
        G[iZ,2] = 0.5*dt**2
        G[iW,2] = dt

        #F = np.array([[1, dt], [0,1]]) #matrix from the equation 
        #G = np.array([[0.5*dt**2],[dt]])
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * (self._a_var)
        #updating the position and the covariance matrix 
        self._x = new_x
        self._P = new_P
        # pass #just a place holder while the function is empty.

    def update(self, meas_val: np.ndarray, meas_var: float) :
        #equations: 
        #y = z - Hx updated position matrix considering measurements 
        # i think here the y and the z is saying its the difference between the mean and the observed
        #S = H P Ht + R
        #K = P Ht s^-1 
        #x_z = x_k + K y, updated position considering measurement values 
        #P_z = (I - KH)Pk : updated covariance matrix considering the measurement values 
        #H = np.array([1,0]).reshape((1,2))
        H = np.zeros((3,n_var))
        H[0,iX] = 1
        H[1,iY] = 1
        H[2,iZ] = 1
        z = np.asarray(meas_val)
        R = np.eye(3) * meas_var

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))
        up_x = self._x + K.dot(y)
        up_P = (np.eye(n_var) - K.dot(H)).dot(self._P)

        self._x = up_x 
        self._P = up_P


    #time for time evolution: 
 
    #working with the mean of position and velocity 
    #using property lets us call kf.pos instead of kf.pos()
    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x
    
    @property #with the underscore, the private ones can change but a property is more stable since it is public
    def pos(self) -> np.ndarray:
        return ([self._x[iX],self._x[iY],self._x[iZ]])
        
    @property
    def vel(self) -> np.ndarray:
        return ([self._x[iU],self._x[iV],self._x[iW]])