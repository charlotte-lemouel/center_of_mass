import numpy
import functions
import center_of_mass

Weight      = numpy.load('data\weight.npz')
models      = ['3D-segments','Long-axis','Hips_1','Hips_2']
Frequency   = 1000
subjects    = []
Trials      = {}
Foot_marker = {}        
Kinematic_Frequency = {}

# Dataset 1 from Wojtusch & von Stryk 2015 https://doi.org/10.1109/HUMANOIDS.2015.7363534 
for subject in ['A','B']:
    Trials[subject] = ['2.2','2.3']
    subjects.append(subject)
    Foot_marker[subject] = 'CAL2'
    Kinematic_Frequency[subject] = 500
# Dataset 2 from Seethapathi & Srinavasan 2019 https://doi.org/10.7554/eLife.38371
for subject in range(8):
    Trials[str(subject)] = ['5','7','9']
    subjects.append(str(subject))
    Foot_marker[str(subject)]    = '_Foot_Heel'   
    Kinematic_Frequency[str(subject)] = 100
    
## Parameters for combining kinematic and kinetic information: 
# Standard deviation of the noise in the position measurement
P_stds = list(numpy.arange(0.0002,0.005,0.0001))
P_stds.extend(list(numpy.arange(0.005,0.0155,0.001)))
P_stds.append(0.02)
# The standard deviation of the noise in the force measurement in each direction is estimated as the root mean square of the force signal in that direction when the forceplates are unloaded.
Unloaded_force = numpy.load('data\\unloaded_force.npz')['Unloaded_force']
Force_std      = numpy.zeros((3))
for dim in range(3):
    Force_std[dim] = numpy.sqrt(numpy.mean(Unloaded_force[:,dim]**2))    
    
## Parameters for determining the onset of flight and stance phases
flight_cutoff  = 0.07  # fraction of the weight used to determine the onset of flight and stance phases
flight_cutoff2 = 0.01 # fraction of the weight used to refine the onset of flight and stance phases
duration       = 0.1  # duration (in seconds) used to calculate the periodic vertical force for refining the onsets
flight_samples = int(duration*Frequency)
# In dataset 1, the person first stands still on the forceplates, then starts running as the treadmill gradually accelerates.
# At the end of the trial, the treadmill gradually decelerates until the person stands still again. 
# To ensure that stance and swing onsets are determined correctly, we impose a minimum number of samples for the stance and flight phases
min_flight     = 40   # Minimum number of samples of the flight phase
min_stance     = 100  # Minimum number of samples of the stance phase
# To ensure that the running speed is constant, the first ten steps and the last fives steps are removed
Remove = {'start':10,'end':5}

## Acceleration of the CoM during flight
Flight_values = {'kinematic':{},'estimate':{}}
for model in models:
    Flight_values['kinematic'][model] = [[], [], []]
    Flight_values['estimate'][model]  = [[[] for _ in range(len(P_stds))],[[] for _ in range(len(P_stds))],[[] for _ in range(len(P_stds))]]
import scipy
window_length = 3
polyorder     = 2
# delta     = 1/Frequency

## Calculation of the center of mass from the kinematics
Sex    = {'A':'female','B':'male'}
Reduced_markers = ['LASI','RASI','LPSI','RPSI','LSHO','RSHO','LELL','RELL','LWRU','RWRU','LFLE','RFLE','LLMAL','RLMAL'] # marker set for the calculation of the Long-axis model

## Periodic CoM position 
periodic_position = {'kinetic':{},'kinematic':{},'estimate':{}}
for method in ['kinematic','estimate']:
    for model in models:
        periodic_position[method][model] = {}
periodic_position['P_stds'] = P_stds

Leg_width_explained_var  = {'kinematic':{},'estimate':{}}
for method in ['kinematic','estimate']:
    for model in models:
        Leg_width_explained_var[method][model]  = []

for subject in subjects:
    marker    = Foot_marker[subject]
    for trial in Trials[subject]:
        print(subject, trial)
        
        weight   = Weight[subject]
        Mass     = weight/9.81
        Force    = numpy.load('data\\preprocessed\\'+subject+'\\'+trial+'-Force.npz')['Force']
        
        ## Center of Mass
        # From the kinematics
        Position = numpy.load('data\\preprocessed\\'+subject+'\\'+trial+'-Motion.npz')
        Position_upsampled = numpy.load('data\\preprocessed\\'+subject+'\\'+trial+'-MotionUpsampled.npz')
        KinematicCoM = {}
        KinematicCoM_upsampled = {}
        
        # Dataset 1: three kinematic models of increasing complexity are calculated
        if subject in ['A','B']:
            sex = Sex[subject]
            Labels   = list(Position.keys())
            # Using the full marker set of Dumas et al. 2007
            kinematics         = center_of_mass.Kinematics(Position, Labels, sex)
            KinematicCoM['3D-segments'] = kinematics.calculate_CoM()
            # Using the simplified marker set of Tisserand et al. 2016
            kinematics         = center_of_mass.Kinematics(Position, Reduced_markers, sex)
            KinematicCoM['Long-axis']   = kinematics.calculate_CoM()
            # Using the midpoint of the four hip markers
            KinematicCoM['Hips_1']   = 0.25*(Position['LASI']+Position['RASI']+Position['LPSI']+Position['RPSI'])
            
            # Upsampled
            # Using the full marker set of Dumas et al. 2007
            kinematics         = center_of_mass.Kinematics(Position_upsampled, Labels, sex)
            KinematicCoM_upsampled['3D-segments'] = kinematics.calculate_CoM()
            # Using the simplified marker set of Tisserand et al. 2016
            kinematics         = center_of_mass.Kinematics(Position_upsampled, Reduced_markers, sex)
            KinematicCoM_upsampled['Long-axis']   = kinematics.calculate_CoM()
            # Using the midpoint of the four hip markers
            KinematicCoM_upsampled['Hips_1']   = 0.25*(Position_upsampled['LASI']+Position_upsampled['RASI']+Position_upsampled['LPSI']+Position_upsampled['RPSI'])
            NrOfSamples   = numpy.shape(Force)[1]
            
        # Dataset 2: only the hips model is calculated
        else:
            KinematicCoM['Hips_2']= 0.25*(Position['Hip_BackLeft']+Position['Hip_BackMid']+Position['Hip_BackRight']+Position['Hip_Front'])
            
            # Upsampled
            KinematicCoM_upsampled['Hips_2']= 0.25*(Position_upsampled['Hip_BackLeft']+Position_upsampled['Hip_BackMid']+Position_upsampled['Hip_BackRight']+Position_upsampled['Hip_Front'])
            # Force and kinematics are truncated to have the same duration
            NrOfSamples   = min(numpy.shape(Force)[1],numpy.shape(KinematicCoM_upsampled['Hips_2'])[1])
            Force         = Force[:,:NrOfSamples]
            KinematicCoM_upsampled['Hips_2'] = KinematicCoM_upsampled['Hips_2'][:,:NrOfSamples]
            
        KinematicVelocity_upsampled = {}
        for model in KinematicCoM_upsampled.keys():
            KinematicVelocity_upsampled[model] = (KinematicCoM_upsampled[model][:,1:]-KinematicCoM_upsampled[model][:,:-1])*Frequency
        
        # Combining kinematic and kinetic information 
        EstimatePosition = {}
        EstimateVelocity = {}
        for model in KinematicCoM_upsampled.keys():
            print('CoM estimate for '+model+' model')
            EstimatePosition[model] = numpy.zeros((len(P_stds),3,numpy.shape(KinematicCoM_upsampled[model])[1]))
            EstimateVelocity[model] = numpy.zeros((len(P_stds),3,numpy.shape(KinematicCoM_upsampled[model])[1]))
            for p, Position_std in enumerate(P_stds):
                EstimatePosition[model][p], EstimateVelocity[model][p], Frequency = center_of_mass.estimator.optimal_combination(Force, Frequency, KinematicCoM_upsampled[model], Frequency, Mass, Force_std = Force_std, Position_std = Position_std)

        ## The onsets of flight and stance phases are determined as the instants when the vertical force crosses the threshold flight_cutoff*weight
        print('Flight and stance onsets')
        Vertical_force = Force[2]
        # NrOfSamples    = len(Vertical_force)
        
        Flight_onsets  = []
        Stance_onsets  = []
        if Vertical_force[0] < flight_cutoff*weight:
            stance_onset = numpy.argmax(Vertical_force > flight_cutoff*weight)
        else:
            stance_onset = - min_stance 
        while stance_onset+min_stance < NrOfSamples - 1:
            if numpy.min(Vertical_force[stance_onset+min_stance:]) < flight_cutoff*weight:
                flight_onset = stance_onset+min_stance + numpy.argmax(Vertical_force[stance_onset+min_stance:] < flight_cutoff*weight)
                if flight_onset+min_flight < NrOfSamples - 1:
                    stance_onset = flight_onset+min_flight + numpy.argmax(Vertical_force[flight_onset+min_flight:] > flight_cutoff*weight)
                    # Flight phases which are shorter than the minimal duration are not included because in that case the determination of stance_onset is inaccurate
                    if stance_onset - flight_onset > min_flight:
                        Flight_onsets.append(flight_onset)
                        Stance_onsets.append(stance_onset)
                else:
                    stance_onset = NrOfSamples
            else:
                stance_onset = NrOfSamples
        Flight_onsets  = numpy.array(Flight_onsets)
        Stance_onsets  = numpy.array(Stance_onsets)
                
        # Dataset 1: The first and last few steps are removed to ensure the subject is in steady state
        if subject in ['A','B']:
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
        Flight_mean_vertical_force = functions.average_across_steps(Force,Flight_onsets[(Flight_onsets > flight_samples)*(Flight_onsets < NrOfSamples - flight_samples)],trial,initial_side,flight_samples)[2]
        Stance_mean_vertical_force = functions.average_across_steps(Force,Stance_onsets[(Stance_onsets > flight_samples)*(Stance_onsets < NrOfSamples - flight_samples)],trial,initial_side,flight_samples)[2]
        # Instants when the mean force crosses the threshold flight_cutoff2*weight
        flight_remove  = numpy.argmax(Flight_mean_vertical_force[flight_samples:] < weight*flight_cutoff2)
        stance_remove  = numpy.argmax(Stance_mean_vertical_force[:flight_samples][::-1] < weight*flight_cutoff2)
        Flight_onsets += flight_remove
        Stance_onsets -= stance_remove
        
        Flight_onsets_kinematics = numpy.array(numpy.ceil(Flight_onsets*Kinematic_Frequency[subject]/Frequency), dtype = 'int')
        Stance_onsets_kinematics = numpy.array(Stance_onsets*Kinematic_Frequency[subject]/Frequency, dtype = 'int')
            
        ## Acceleration during flight 
        print('Acceleration during flight')
        for model in KinematicCoM.keys():
            # Kinematic center of mass: not upsampled
            CoM_acceleration    = scipy.signal.savgol_filter(KinematicCoM[model], window_length, polyorder, deriv=2, delta=1/Kinematic_Frequency[subject]) 
            flight_acceleration = functions.flight_values(CoM_acceleration, Flight_onsets_kinematics, Stance_onsets_kinematics, initial_side)
            for dim in range(3):
                Flight_values['kinematic'][model][dim].extend(flight_acceleration[dim])
            # CoM estimator combining kinematic and forceplate information: upsampled
            Estimate_acceleration = scipy.signal.savgol_filter(EstimatePosition[model], window_length, polyorder, deriv=2, delta=1/Frequency) 
            for p in range(len(P_stds)):
                flight_acceleration   = functions.flight_values(Estimate_acceleration[p], Flight_onsets, Stance_onsets, initial_side)
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
        Flight_onsets_cropped = Flight_onsets[(Flight_onsets > pre_samples)*(Flight_onsets < NrOfSamples - post_samples)]
        first_stance_crop     = Stance_onsets[numpy.argmax(Stance_onsets > Flight_onsets_cropped[0])]
        # The side of the first step is determined by averaging the lateral force during the first stance phase
        MeanLateralForce_crop = numpy.mean(Force[0,first_stance_crop:Flight_onsets_cropped[1]])
        if MeanLateralForce_crop > 0:
            initial_side_crop = 'Left'
        else:
            initial_side_crop = 'Right'          
        
        # Kinetic Periodic Position
        MeanForce            = functions.average_across_steps(Force, Flight_onsets_cropped, trial, initial_side_crop, pre_samples+1, post_samples = post_samples-1)
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
        
        # Kinematic and Estimate Periodic Positions
        for model in KinematicCoM.keys():

            # Kinematics
            MeanPosition = functions.average_across_steps(KinematicCoM_upsampled[model], Flight_onsets_cropped, trial, initial_side_crop, pre_samples, post_samples = post_samples+1)
            # To obtain a periodic position signal, we subtract a linear function of time 
            Trial_difference = MeanPosition[:,-1] - MeanPosition[:,0]
            for dim in range(3):
                MeanPosition[dim] -= numpy.arange(Duration+1)/Duration*Trial_difference[dim]
            MeanPosition     = MeanPosition[:,:-1]    
            ZeroMeanPosition = MeanPosition - numpy.array([numpy.mean(MeanPosition, axis = -1)]).T
            periodic_position['kinematic'][model][subject+trial] = ZeroMeanPosition
            
            # Estimator
            periodic_position['estimate'][model][subject+trial]  = numpy.zeros((len(P_stds),3,Duration))
            for p in range(len(P_stds)):
                MeanPosition = functions.average_across_steps(EstimatePosition[model][p], Flight_onsets_cropped, trial, initial_side_crop, pre_samples, post_samples = post_samples+1)
                # To obtain a periodic position signal, we subtract a linear function of time 
                Trial_difference = MeanPosition[:,-1] - MeanPosition[:,0]
                for dim in range(3):
                    MeanPosition[dim] -= numpy.arange(Duration+1)/Duration*Trial_difference[dim]
                MeanPosition     = MeanPosition[:,:-1]
                ZeroMeanPosition = MeanPosition - numpy.array([numpy.mean(MeanPosition, axis = -1)]).T    
                periodic_position['estimate'][model][subject+trial][p] = ZeroMeanPosition

        ## Foot placement
        print('Foot placement')
        CoM_state_pre = {}
        Leg_width     = {}
        for model in KinematicCoM.keys():    
            CoM_state_pre[model] = {'kinematic':{'L':[],'R':[]},'estimate':{'L':[],'R':[]}}
            Leg_width[model]     = {'kinematic':{'L':[],'R':[]},'estimate':{'L':[],'R':[]}}
            
        for f, flight_onset in enumerate(Flight_onsets):
            touchdown_side = functions.next_stance_side(f, initial_side)[0]
            stance_onset = Stance_onsets[f]
                
            for model in KinematicCoM.keys():     
                # Kinematics
                apex = flight_onset + numpy.argmax(KinematicCoM_upsampled[model][2,flight_onset:stance_onset])
                CoM_state_pre[model]['kinematic'][touchdown_side].append([KinematicVelocity_upsampled[model][0,apex],KinematicVelocity_upsampled[model][1,apex],KinematicCoM_upsampled[model][2,apex]])
                Leg_width[model]['kinematic'][touchdown_side].append(Position_upsampled[touchdown_side+marker][0,stance_onset] - KinematicCoM_upsampled[model][0,stance_onset])
                
                # Estimator
                apex = flight_onset + numpy.argmax(EstimatePosition[model][:,2,flight_onset:stance_onset], axis = 1)
                CoM_state_pre[model]['estimate'][touchdown_side].append([EstimateVelocity[model][numpy.arange(len(P_stds)),0,apex],EstimateVelocity[model][numpy.arange(len(P_stds)),1,apex],EstimatePosition[model][numpy.arange(len(P_stds)),2,apex]])
                Leg_width[model]['estimate'][touchdown_side].append(Position_upsampled[touchdown_side+marker][0,stance_onset] - EstimatePosition[model][:,0,stance_onset])
                
        for model in KinematicCoM.keys():
            for side in ['L','R']:
                for method in ['kinematic','estimate']:
                    CoM_state_pre[model][method][side] = numpy.array(CoM_state_pre[model][method][side])
                    Leg_width[model][method][side] = numpy.array(Leg_width[model][method][side])

                    CoM_state_pre[model][method][side] -= numpy.array([numpy.mean(CoM_state_pre[model][method][side], axis = 0)])
                    Leg_width[model][method][side] -= numpy.array([numpy.mean(Leg_width[model][method][side], axis = 0)])
                    
                # Kinematics
                parameters, explained_var = functions.linear_fit_2D(CoM_state_pre[model]['kinematic'][side],Leg_width[model]['kinematic'][side])
                Leg_width_explained_var['kinematic'][model].append(explained_var[0])
                
                # Estimator: you need one regression per p_std
                explained_vars_width = []
                for p in range(len(P_stds)):
                    parameters, explained_var = functions.linear_fit_2D(CoM_state_pre[model]['estimate'][side][:,:,p],Leg_width[model]['estimate'][side][:,p])
                    explained_vars_width.append(explained_var[0])
                
                Leg_width_explained_var['estimate'][model].append(explained_vars_width)  

        
## Saving the variables
print('Saving the variables')

import pickle
# Leg width explained variance
for CoM_calculation in ['kinematic','estimate']:
    for model in models:
        Leg_width_explained_var[CoM_calculation][model]  = numpy.array(Leg_width_explained_var[CoM_calculation][model])
pickle_file = open('data\\foot_placement_width.pkl','wb')
pickle.dump(Leg_width_explained_var, pickle_file)
pickle_file.close()

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

# # The Kinetic Error is estimated by applying the estimator to the empty forceplate measurements
# Frequency = 500
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
            