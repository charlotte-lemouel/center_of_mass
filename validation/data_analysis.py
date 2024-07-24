import numpy
import functions
import center_of_mass

trials = ['2.2','2.3']
models = ['3D-segments','Long-axis','hips']

Weight    = numpy.load('data\weight.npz')
Frequency = 500 # in Hertz

## Calculation of the center of mass
# From the kinematics
import center_of_mass
Sex = {'A':'female','B':'male'}
Reduced_markers = ['LASI','RASI','LPSI','RPSI','LSHO','RSHO','LELL','RELL','LWRU','RWRU','LFLE','RFLE','LLMAL','RLMAL'] # marker set for the calculation of the Long-axis model
# Parameters for combining kinematic and kinetic information: 
# Standard deviation of the noise in the position measurement
P_stds = list(numpy.arange(0.0002,0.004,0.00005))
P_stds.extend(list(numpy.arange(0.004,0.0205,0.0005)))
# The standard deviation of the noise in the force measurement in each direction is estimated as the root mean square of the force signal in that direction when the forceplates are unloaded.
Unloaded_force = numpy.load('data\\unloaded_force.npz')['Unloaded_force']
Force_std      = numpy.zeros((3))
for dim in range(3):
    Force_std[dim] = numpy.sqrt(numpy.mean(Unloaded_force[:,dim]**2))

## Parameters for determining the onset of flight and stance phases
flight_cutoff  = 0.1  # fraction of the weight used to determine the onset of flight and stance phases
flight_cutoff2 = 0.01 # fraction of the weight used to refine the onset of flight and stance phases
min_flight     = 20   # Minimum number of samples of the flight phase
min_stance     = 100  # Minimum number of samples of the stance phase
duration       = 50   # Number of samples used to calculate the periodic vertical force for refining the onsets
# The first ten steps and the last fives steps are removed since the subject is accelerating then decelerating
Remove = {'start':10,'end':5}

## Acceleration of the CoM during flight
Flight_values = {'kinematic':{},'estimate':{}}
for model in models:
    Flight_values['kinematic'][model] = [[], [], []]
    Flight_values['estimate'][model]  = [[[] for _ in range(len(P_stds))],[[] for _ in range(len(P_stds))],[[] for _ in range(len(P_stds))]]
import scipy
window_length = 3
polyorder     = 2
delta         = 1/Frequency  

## Periodic CoM position 
periodic_position = {'kinetic':{},'kinematic':{},'estimate':{}}
periodic_position['P_stds'] = P_stds
for subject in ['A','B']:
    for trial in trials:
        print(subject, trial)
        periodic_position['kinematic'][subject+trial] = {'3D-segments':{},'Long-axis':{},'hips':{}}
        periodic_position['estimate'][subject+trial]  = {'3D-segments':{},'Long-axis':{},'hips':{}}
        
        weight   = Weight[subject]
        Mass     = weight/9.81
        Force    = numpy.load('data\\preprocessed\\'+subject+'\\'+trial+'-Force.npz')['Force']
        
        ## Center of Mass
        # From the kinematics
        sex = Sex[subject]
        Position = numpy.load('data\\preprocessed\\'+subject+'\\'+trial+'-Motion.npz')
        Labels   = list(Position.keys())
        CoM = {}
        # Using the full marker set of Dumas et al. 2007
        kinematics         = center_of_mass.Kinematics(Position, Labels, sex)
        CoM['3D-segments'] = kinematics.calculate_CoM()
        # Using the simplified marker set of Tisserand et al. 2016
        kinematics         = center_of_mass.Kinematics(Position, Reduced_markers, sex)
        CoM['Long-axis']   = kinematics.calculate_CoM()
        # Using the midpoint of the four hip markers
        CoM['hips']   = 0.25*(Position['LASI']+Position['RASI']+Position['LPSI']+Position['RPSI'])
        # Combining kinematic and kinetic information 
        Estimate = {}
        for model in CoM.keys():
            print('CoM estimate for '+model+' model')
            Estimate[model] = numpy.zeros((len(P_stds),3,numpy.shape(CoM[model])[1]))
            for p, Position_std in enumerate(P_stds):
                CoM_position, CoM_velocity, Frequency = center_of_mass.estimator.optimal_combination(Force, Frequency, CoM[model], Frequency, Mass, Force_std = Force_std, Position_std = Position_std)
                Estimate[model][p] = CoM_position
                
        ## The onsets of flight and stance phases are determined as the instants when the vertical force crosses the threshold flight_cutoff*weight
        print('Flight and stance onsets')
        Vertical_force = Force[2]
        Flight_onsets  = []
        Stance_onsets  = []
        stance_onset   = 0
        while numpy.min(Vertical_force[stance_onset+min_stance:]) < flight_cutoff*weight:
            flight_onset = stance_onset+min_stance + numpy.argmax(Vertical_force[stance_onset+min_stance:] < flight_cutoff*weight)
            stance_onset = flight_onset+min_flight + numpy.argmax(Vertical_force[flight_onset+min_flight:] > flight_cutoff*weight)
            # Flight phases which are shorter than the minimal duration are not included because in that case the determination of stance_onset is inaccurate
            if stance_onset - flight_onset > min_flight:
                Flight_onsets.append(flight_onset)
                Stance_onsets.append(stance_onset)
                
        # The first and last few steps are removed to ensure the subject is in steady state
        Flight_onsets  = Flight_onsets[Remove['start']:-Remove['end']]
        Stance_onsets  = Stance_onsets[Remove['start']:-Remove['end']]

        # The side of the first step is determined by averaging the lateral force during the first stance phase
        MeanLateralForce0 = numpy.mean(Force[0,Stance_onsets[0]:Flight_onsets[1]])
        if MeanLateralForce0 > 0:
            initial_side = 'Left'
        else:
            initial_side = 'Right'          
            
        # The flight phase is determined more precisely by 
        # - adding a given number of samples (flight_remove) to all flight onsets
        # - subtracting a given number of samples (stance_remove) to all stance onsets
        # These numbers are determined by averaging the vertical force across all steps and looking at when the average crosses the threshold flight_cutoff2*weight
        Flight_mean_vertical_force = functions.average_across_steps(Force,Flight_onsets,trial,initial_side,duration)[2]
        Stance_mean_vertical_force = functions.average_across_steps(Force,Stance_onsets,trial,initial_side,duration)[2]
        # Instants when the mean force crosses the threshold flight_cutoff2*weight
        flight_remove  = numpy.argmax(Flight_mean_vertical_force[duration:] < weight*flight_cutoff2)
        stance_remove  = numpy.argmax(Stance_mean_vertical_force[:duration][::-1] < weight*flight_cutoff2)
        Flight_onsets += flight_remove
        Stance_onsets -= stance_remove

            
        ## Acceleration during flight 
        print('Acceleration during flight')
        for model in models:
            # Kinematic center of mass
            CoM_acceleration    = scipy.signal.savgol_filter(CoM[model], window_length, polyorder, deriv=2, delta=delta) 
            flight_acceleration = functions.flight_values(CoM_acceleration, Flight_onsets, Stance_onsets)
            for dim in range(3):
                Flight_values['kinematic'][model][dim].extend(flight_acceleration[dim])
            # CoM estimator combining kinematic and forceplate information
            Estimate_acceleration = scipy.signal.savgol_filter(Estimate[model], window_length, polyorder, deriv=2, delta=delta) 
            for p in range(len(P_stds)):
                flight_acceleration   = functions.flight_values(Estimate_acceleration[p], Flight_onsets, Stance_onsets)
                for dim in range(3):
                    Flight_values['estimate'][model][dim][p].extend(flight_acceleration[dim])

        ## Periodic CoM position
        print('Periodic CoM position')
        Step_durations = Stance_onsets[1:] - Stance_onsets[:-1]
        MedianDuration = int(numpy.median(Step_durations))
        # for running trials, a gait cycle corresponds to two steps
        pre_samples  = MedianDuration
        post_samples = MedianDuration
        Duration     = pre_samples + post_samples
        
        # Kinetic Periodic Position
        MeanForce            = functions.average_across_steps(Force, Flight_onsets, trial, initial_side, pre_samples+1, post_samples = post_samples-1)
        MeanAcceleration     = MeanForce/Mass # in m/s^2
        MeanAcceleration[2] -= 9.81 # the acceleration due to gravity is removed to obtain the CoM acceleration
        # To obtain a periodic velocity signal, the mean acceleration over the trial is removed, then acceleration is integrated
        ZeroMeanAcceleration = MeanAcceleration - numpy.array([numpy.mean(MeanAcceleration, axis = -1)]).T
        MeanVelocity         = numpy.cumsum(ZeroMeanAcceleration, axis = 1)/Frequency
        # To obtain a periodic position signal, the mean velocity over the trial is removed, then velocity is integrated
        ZeroMeanVelocity     = MeanVelocity - numpy.array([numpy.mean(MeanVelocity, axis = -1)]).T
        MeanPosition         = numpy.cumsum(ZeroMeanVelocity, axis = 1)/Frequency
        ZeroMeanPosition     = MeanPosition - numpy.array([numpy.mean(MeanPosition, axis = -1)]).T
        periodic_position['kinetic'][subject+trial] = ZeroMeanPosition
        
        # Kinematic and Estimate Periodic Positions for the three models
        for model in models:

            # Kinematics
            MeanPosition = functions.average_across_steps(CoM[model], Flight_onsets, trial, initial_side, pre_samples, post_samples = post_samples+1)
            # To obtain a periodic position signal, we subtract a linear function of time 
            Trial_difference = MeanPosition[:,-1] - MeanPosition[:,0]
            for dim in range(3):
                MeanPosition[dim] -= numpy.arange(Duration+1)/Duration*Trial_difference[dim]
            MeanPosition     = MeanPosition[:,:-1]    
            ZeroMeanPosition = MeanPosition - numpy.array([numpy.mean(MeanPosition, axis = -1)]).T
            periodic_position['kinematic'][subject+trial][model] = ZeroMeanPosition
            
            # Estimator
            periodic_position['estimate'][subject+trial][model]  = numpy.zeros((len(P_stds),3,Duration))
            for p in range(len(P_stds)):
                MeanPosition = functions.average_across_steps(Estimate[model][p], Flight_onsets, trial, initial_side, pre_samples, post_samples = post_samples+1)
                # To obtain a periodic position signal, we subtract a linear function of time 
                Trial_difference = MeanPosition[:,-1] - MeanPosition[:,0]
                for dim in range(3):
                    MeanPosition[dim] -= numpy.arange(Duration+1)/Duration*Trial_difference[dim]
                MeanPosition     = MeanPosition[:,:-1]
                ZeroMeanPosition = MeanPosition - numpy.array([numpy.mean(MeanPosition, axis = -1)]).T    
                periodic_position['estimate'][subject+trial][model][p] = ZeroMeanPosition

## Saving the variables
import pickle
# Flight acceleration
for CoM_calculation in ['kinematic','estimate']:
    for model in models:
        Flight_values[CoM_calculation][model] = numpy.array(Flight_values[CoM_calculation][model])
pickle_file = open('data\\flight_acceleration.pkl','wb')
pickle.dump(Flight_values, pickle_file)
pickle_file.close()

# Periodic CoM position
pickle_file = open('data\\periodic_positions.pkl','wb')
pickle.dump(periodic_position, pickle_file)
pickle_file.close()

# The Kinetic Error is estimated by applying the estimator to the empty forceplate measurements
nb_of_segments, nb_of_dims, unloaded_duration = numpy.shape(Unloaded_force)
KineticDrift = numpy.zeros((len(P_stds),2*nb_of_segments,3,unloaded_duration))
KinematicCoM = numpy.zeros((3,unloaded_duration))
for s, subject in enumerate(['A','B']):
    Mass = Weight[subject]/9.81
    for segment in range(nb_of_segments):
        Force = Unloaded_force[segment] + Mass*numpy.array([[0,0,9.81]]).T # the net ground reaction force is modelled as the sum of the unloaded force and the opposite of the person's weight 
        for i, Position_std in enumerate(P_stds):
            CoM_position, CoM_velocity, Frequency = center_of_mass.estimator.optimal_combination(Force, Frequency, KinematicCoM, Frequency, Mass, Force_std = Force_std, Position_std = Position_std)
            KineticDrift[i,nb_of_segments*s+segment] = CoM_position
numpy.savez('data\\kinetic_drift.npz',KineticDrift = KineticDrift)
            
            