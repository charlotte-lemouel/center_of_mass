import center_of_mass

# The example data file is loaded
import pickle
input_file     = '..\\examples\\example_data.pkl'
pickle_file    = open(input_file,'rb')
data           = pickle.load(pickle_file)
pickle_file.close()
sex            = data['sex']                # 'female' or 'male'
Labels         = data['Labels']             # list of marker labels
Position       = data['Position']           # dictionary with, for each marker, the position in meters (numpy.ndarray of shape (3,duration*Position_frequency)

# We collect the kinematic data into a Kinematics object, which generates additional attributes: Joint_centers, SegmentCoordinateSystem, SegmentLength and SegmentOrigin.
kinematics    = center_of_mass.Kinematics(Position, Labels, sex)

# Calculation of the center of mass
Kinematic_com = kinematics.calculate_CoM()

# Visualisation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

instant = 500 # timepoint at which the data will be visualised
xlim = [1,2.8]
ylim = [-0.9,0.9]
zlim = [0,1.8]

fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")
for marker in kinematics.Labels:
    position = kinematics.Position[marker][:,instant]
    ax.scatter(position[0],position[1],position[2], color = 'k', s=0.5)
for segment in list(kinematics.SegmentCoordinateSystem.keys()):
    origin   = kinematics.SegmentOrigin[segment][:,instant]
    if segment == 'Head':
        endpoint = origin + kinematics.SegmentLength[segment]*kinematics.SegmentCoordinateSystem[segment][1][:,instant]
    elif segment.endswith('Foot'):
        endpoint = origin + kinematics.SegmentLength[segment]*kinematics.SegmentCoordinateSystem[segment][0][:,instant]  
    else:
        endpoint = origin - kinematics.SegmentLength[segment]*kinematics.SegmentCoordinateSystem[segment][1][:,instant] 
    ax.scatter(origin[0],origin[1],origin[2], label = segment)
    ax.plot([origin[0],endpoint[0]],[origin[1],endpoint[1]],[origin[2],endpoint[2]]) 
com = Kinematic_com[:,500]   
ax.scatter(com[0],com[1],com[2], color = 'k', s=40, label = 'CoM')
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)
plt.show()