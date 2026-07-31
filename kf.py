import numpy as np


#this modification will help later when we add 3 dims.. i think 
iX = 0
iV = 1
n_var = iV + 1
class KF: 
    def __init__(self, x_init: float, v_init: float, a_var: float) -> None: #self makes it be a specific instance of KF, so the array is stored there instead of the whole class
        self._x = np.zeros(n_var)
        #mean of the state
        self._x[iX] = x_init
        self._x[iV] = v_init
         #the underscore makes this variable private
        #acceleration variance 
        self._a_var = a_var 
        #covaraince of the state
        self._P = np.eye(2) #this makes an identity matrix, 2 on the diagonal, initial covariance matrix 


    #predict  
    def predict(self, dt: float) -> None:
        #calling predict means that we integrate, which means our uncertainity icnrease
        #equations to predict 
        #new x = Fx
        #new P = F P Ft + G Gt a
        F = np.eye(n_var)
        F[iX,iV] = dt
        G = np.zeros((2,1))
        G[iX] = 0.5*dt**2
        G[iV] = dt
        #F = np.array([[1, dt], [0,1]]) #matrix from the equation 
        #G = np.array([[0.5*dt**2],[dt]])
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * (self._a_var)
        #updating the position and the covariance matrix 
        self._x = new_x
        self._P = new_P
        # pass #just a place holder while the function is empty.

    def update(self, meas_val: float, meas_var: float) :
        #equations: 
        #y = z - Hx updated position matrix considering measurements 
        #S = H P Ht + R
        #K = P Ht s^-1 
        #x_z = x_k + K y, updated position considering measurement values 
        #P_z = (I - KH)Pk : updated covariance matrix considering the measurement values 
        #H = np.array([1,0]).reshape((1,2))
        H = np.zeros((1,n_var))
        H[iX] = 1
        z = np.array([meas_val])
        R = np.array([meas_var])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))
        up_x = self._x + K.dot(y)
        up_P = (np.eye(2) - K.dot(H)).dot(self._P)

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
    def pos(self) -> float:
        return self._x[iX]
        
    @property
    def vel(self) -> float:
        return self._x[iV]