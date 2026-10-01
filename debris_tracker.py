import matplotlib.pyplot as plt
import numpy as np 
from kf import KF

#getting space track space debris data 
from dotenv import load_dotenv #this is for the password and username
import os

load_dotenv()
spacetrack_user = os.getenv("SPACETRACK_USERNAME")
spacetrack_pass = os.getenv("SPACETRACK_PASSWORD")

from spacetrack import SpaceTrackClient 
from sgp4.api import Satrec, jday

#plt.ion()
#plt.figure()

# real measured values! 
real_x = np.array([0.0, 0.0, 0.0])
meas_var = 0.1 ** 2 #simulating noise 
real_v = np.array([0.9,0.4,0.1]) #completely arbitrary right now

#tle captures orbital parameters, inlcination, eccentricity, how elliptical and mean motion
#TLE gives the recipe while sgp4 module lets us convert it into xyz that can be used with the Kalman filter
#starting query will be the ISS! YAY!
st = SpaceTrackClient(identity=spacetrack_user, password=spacetrack_pass)
#the norad id is what chooses the object to track

#choosing a non decayed debris
tle_data = st.gp(norad_cat_id=33655, orderby='epoch desc',limit = 1, format='tle')
print(tle_data)

#example output: 
#1 33655U 99025DEQ 26273.55513619  .00019231  00000-0  10433-2 0  9991
#2 33655  98.8014  24.8857 0032061 119.5904 240.8525 15.14175888  9439

#splitting the lines: 
lines = tle_data.strip().split('\n')
line1 = lines[0]
line2 = lines[1]
satrec = Satrec.twoline2rv(line1,line2) #satrec gets the actual position at a time
#time of position is given in Julian dates, jday to convert
#jd and fr are due to how large this number is, jd is the whole number part and fr is the fraction so 0.5 for 12pm
#error_code is to see if the calc succeeded

kf = KF(pos_init = np.zeros(3), vel_init = np.array([1.0,0.5,0.2]), a_var = 0.1)
DT = 1.0 #seconds between timesteps 
N = 1000
dt_days = DT/86400.0 #fraction of the day
jd, fr = jday(2026,10,1,12,0,0) #todays date to get todays pos
MEAS_N = 20

for i in range(N):
    fr += dt_days #update timestep

    error_code, position, velocity = satrec.sgp4(jd,fr)



means = []
covs = []
real_poss = []
real_vels = []

for i in range(N):
    #just for tseting/fun purpses, varying the velocity half way:
    #if i > 500:
    #    real_v *=0.9
    fr += dt_days #update timestep
    
    error_code, position, velocity = satrec.sgp4(jd,fr)
    real_x = np.array(position)
    reak_v = np.array(velocity)
        
    covs.append(kf.cov)
    means.append(kf.mean)

    #real_x = real_x + DT*real_v

    kf.predict(dt = DT)

    if i != 0 and i % MEAS_N == 0:
        noise_meas = real_x + np.random.randn(3)*np.sqrt(meas_var) #generating noise, in all three dimenions randomly
        kf.update(meas_val = noise_meas, meas_var = meas_var) #randint stuff is the generated noise for now 

    #adding updates with measurements, the uncertaintiy should become bounded
    real_poss.append(real_x)
    real_vels.append(real_v)

#plotting the position and velocity, with uncertainty
#we are going ot have a 3D overview plot and a 2D plot as well 
#coverting means and covs into np arrays 

real_poss = np.array(real_poss)
means = np.array(means)
covs = np.array(covs)

fig = plt.figure(figsize=(12,8))

#3D plot 
ax3d = fig.add_subplot(2,3, (2), projection = '3d') #positioning the subplot
ax3d.set_box_aspect([1,1,1])
ax3d.plot(real_poss[:,0],real_poss[:,1],real_poss[:,2], 'k', label='True Path')
ax3d.plot(means[:,0],means[:,1],means[:,2],'r--', label='KF Estimate')
ax3d.set_title('3D Trajectory')
ax3d.legend()

#2D plots for each axis
axis_label = ['x','y','z'] #need to consdier units maybe, but that depnds on what space tracker has 
#using a loop so that the same code doesnt need to be repeated 29302 times

for i, label in enumerate(axis_label):
    ax = fig.add_subplot(2,3, 4 + i)
    ax.set_title(f'{label}')
    ax.plot(means[:,i],'r')
    ax.plot(real_poss[:+i],'k')
    #now also plotting the uncertainity, in the form of standard deiviation
    ax.plot(means[:,i] -2*np.sqrt(covs[:,i,i]),'c:')
    ax.plot(means[:,i] +2*np.sqrt(covs[:,i,i]),'c:')

plt.tight_layout()
plt.show()
plt.ginput()

#plt.subplot(2,1,1)
#plt.title('Position')
#plt.plot([mean[0] for mean in means], 'r' )
#this plots within 2 standard deviaiton of the mean 
#plt.plot(real_xs, 'k')
#plt.plot([mean[0] - 2*np.sqrt(cov[0,0])for mean, cov in zip(means, covs)], 'r-.') #zip just takes the index numebr we are interested in
#plt.plot([mean[0] + 2*np.sqrt(cov[0,0])for mean, cov in zip(means, covs)], 'r-.') #zip just takes the index numebr we are interested in


#plt.subplot(2,1,2)
#plt.title('Velocity')
#plt.plot([mean[1] for mean in means], 'b' )
#plt.plot(real_vs, 'k')
#plt.plot([mean[1] - 2*np.sqrt(cov[1,1])for mean, cov in zip(means, covs)], 'b-.') #zip just takes the index numebr we are interested in
#plt.plot([mean[1] + 2*np.sqrt(cov[1,1])for mean, cov in zip(means, covs)], 'b-.') #zip just takes the index numebr we are interested in


#plt.show() #graph shows that whenever there is a measurement, the uncertainity drops!! so cool 
#plt.ginput()