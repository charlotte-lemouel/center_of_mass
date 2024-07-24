import center_of_mass

# The example data file is loaded
import pickle
input_file     = '..\\examples\\example_data.pkl'
pickle_file    = open(input_file,'rb')
data           = pickle.load(pickle_file)
pickle_file.close()
sex                 = data['sex']                # 'female' or 'male'
Labels              = data['Labels']             # list of marker labels
Kinematic_frequency = data['Position_frequency'] # in Hertz
Position            = data['Position']           # dictionary with, for each marker, the position in meters (numpy.ndarray of shape (3,duration*Position_frequency)
Force_frequency     = data['Force_frequency']    # in Hertz
GroundReactionForce = data['GroundReactionForce']# in Newtons (numpy.ndarray of shape (3,duration*Force_frequency)

# Calculation of the center of mass from the kinematics
kinematics    = center_of_mass.Kinematics(Position, Labels, sex)
Kinematic_com = kinematics.calculate_CoM()

# The person's mass is determined as the median vertical ground reaction force during the initial 1.5 seconds of quiet standing
import numpy
mass = numpy.median(GroundReactionForce[2,:int(1.5*Force_frequency)])/9.81

# The estimates of the center of mass position and velocity are calculated through optimal combination of the force and position information
Pos_estimate, Vel_estimate, Frequency = center_of_mass.estimator.optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass)

# Visualisation
Acceleration = center_of_mass.estimator.com_acceleration(GroundReactionForce, mass)
Acc_subsampled, Pos_subsampled, Frequency = center_of_mass.estimator.subsample_two_signals(Acceleration, Force_frequency, Kinematic_com, Kinematic_frequency)

import matplotlib.pyplot as plt
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
NbOfDimensions, NbOfSamples = numpy.shape(Acc_subsampled)
time = numpy.arange(NbOfSamples)/Frequency

fig, axes = plt.subplots(3,NbOfDimensions)
for dim in range(NbOfDimensions):
    axes[0,dim].plot(time, Pos_subsampled[dim], color = 'k', label = 'measurement')
    axes[0,dim].plot(time, Pos_estimate[dim], color = 'b', label = 'estimate')
    axes[1,dim].plot(time, Vel_estimate[dim], color = 'b')
    axes[2,dim].plot(time, Acc_subsampled[dim], color = 'k')
axes[0,0].legend()
labels = ['Position (m)','Velocity (m/s)',r'Acceleration $(m/s^2)$']
for l in range(3):
    axes[l,0].set_ylabel(labels[l])
for t in range(NbOfDimensions):
    axes[2,t].set_xlabel('Time (s)')
plt.show()
    