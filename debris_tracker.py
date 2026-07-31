import matplotlib.pyplot as plt
import numpy as np 
from kf import KF

plt.ion()
plt.figure()

# real measured values! 
real_x = 0.0
meas_var = 0.1 ** 2 #simulating noise 
real_v = 0.9

kf = KF(x_init = 0.0, v_init = 1.0, a_var = 0.1)
DT = 0.1
N = 1000
MEAS_N = 20


means = []
covs = []
real_xs = []
real_vs = []

for i in range(N):
    #just for tseting/fun purpses, varying the velocity half way:
    if i > 500:
        real_v *=0.9
        
    covs.append(kf.cov)
    means.append(kf.mean)

    real_x = real_x + DT*real_v

    kf.predict(dt = DT)

    if i != 0 and i % MEAS_N == 0:
        kf.update(meas_val = real_x + np.random.randn()*np.sqrt(meas_var), meas_var = meas_var) #randint stuff is the generated noise for now 

    #adding updates with measurements, the uncertaintiy should become bounded
    real_xs.append(real_x)
    real_vs.append(real_v)

#plotting the position and velocity, with uncertainty
plt.subplot(2,1,1)
plt.title('Position')
plt.plot([mean[0] for mean in means], 'r' )
#this plots within 2 standard deviaiton of the mean 
plt.plot(real_xs, 'k')
plt.plot([mean[0] - 2*np.sqrt(cov[0,0])for mean, cov in zip(means, covs)], 'r-.') #zip just takes the index numebr we are interested in
plt.plot([mean[0] + 2*np.sqrt(cov[0,0])for mean, cov in zip(means, covs)], 'r-.') #zip just takes the index numebr we are interested in


plt.subplot(2,1,2)
plt.title('Velocity')
plt.plot([mean[1] for mean in means], 'b' )
plt.plot(real_vs, 'k')
plt.plot([mean[1] - 2*np.sqrt(cov[1,1])for mean, cov in zip(means, covs)], 'b-.') #zip just takes the index numebr we are interested in
plt.plot([mean[1] + 2*np.sqrt(cov[1,1])for mean, cov in zip(means, covs)], 'b-.') #zip just takes the index numebr we are interested in


plt.show() #graph shows that whenever there is a measurement, the uncertainity drops!! so cool 
plt.ginput()