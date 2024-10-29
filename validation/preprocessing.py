import numpy, os, scipy, functions

input_path = 'data\\raw\\'

# We generate the folders in which the pre-processed data will be exported
if not os.path.exists('data\\preprocessed'):
    os.mkdir('data\\preprocessed')
    for subject in ['A','B']:
        os.mkdir('data\\preprocessed\\'+subject)
    for subject in range(8):
        os.mkdir('data\\preprocessed\\'+str(subject))
        
        
weight = {}
        
### Dataset 1 from Wojtusch & von Stryk 2015 https://doi.org/10.1109/HUMANOIDS.2015.7363534 ###
        
### Force pre-processing
print('Force pre-processing')
# Synchronisation of the force and motion measurements
'''The time lag between force and kinematic measurements was estimated through a separate series of measurements 
and modeled as linear function (personnal communication from Janis Wojtusch).
The intercept and gradient of this linear function are the following:'''
intercept = 0.017767380272882      # in seconds
gradient  = -5.713883360543501e-05 # dimensionless

Force_frequency     = 1000 # Hertz
Kinematic_frequency = 500 # Hertz

# Parameters for determining when the force plates are not loaded
loaded_cutoff = 30  # in Newton, used to determine when the subject walks onto and off of the treadmill
loaded_remove = 100 # number of samples @ 500 Hz to remove before the vertical force reaches the loaded_cutoff, and after it drops below it.
# The force measurements when the platforms are unloaded are saved and analysed separately to determine the Kinetic Error (estimation error due to the forceplate noise).
Unloaded_durations   = []
Unloaded_forceplates = []

## The quiet standing periods at the start and end of the trial are determined 
# - to truncate the trial 
# - and to calculate the subject's weight from the vertical force
stance_cutoff = 500 # in Newton, used to determine when the subject stands quietly at the start and end of the trial
stance_remove = 1000 # number of samples @ 1000 Hz to remove before the vertical force reaches the stance_cutoff, and after it drops below it.
Truncate = {}
Weight   = {'A':[],'B':[]}
        
for subject in ['A','B']: 
    for trial in ['2.2','2.3']:

        ## The matlab file is loaded
        input_file  = input_path+subject+'\\'+trial+'-RawForce'
        data        = scipy.io.loadmat(input_file)

        ## The force is extracted and reshaped into a numpy array of shape (3, NrOfSamples)
        Fx          = data['Fx'][:,0]
        NbOfSamples = len(Fx)
        RawForce    = numpy.zeros((3, NbOfSamples))
        RawForce[0] = Fx                                # lateral  direction
        RawForce[1] = data['Fy'][:,0]                   # forwards direction 
        RawForce[2] = data['FzR'][:,0]+data['FzL'][:,0] # vertical direction

        ## The force is synchronised to the motion data 
        originalSamplePoints    = numpy.arange(0,NbOfSamples)/Force_frequency
        compensatedSamplePoints = originalSamplePoints - (intercept + gradient * originalSamplePoints)
        for dim in range(3):
            RawForce[dim] = numpy.interp(originalSamplePoints, compensatedSamplePoints, RawForce[dim])

        ## The force is subsampled at the kinematic frequency 
        # Force_subsampled = functions.subsample_one_signal(RawForce, Force_frequency, Kinematic_frequency)
        # NbOfSamples      = numpy.shape(RawForce)[1]
           
        ## Drift removal
        '''The forceplate drift in each direction is approximated by an affine function of time:  offset + slope*time
        through a least-squares fit on the forceplate measurements:
        - at the start of the trial, before the subject steps onto the treadmill
        - and at the end of the trial, after the subject steps off the treadmill
        '''        
        # Instant when the person steps onto the treadmill (@ 500 Hz): loaded_remove samples before the vertical force exceeds loaded_cutoff
        loaded   = numpy.argmax(RawForce[2] > loaded_cutoff) - loaded_remove
        # The same analysis is applied to the time-reversed force to determine when the subject steps off the platforms
        unloaded = NbOfSamples - numpy.argmax(RawForce[2,::-1] > loaded_cutoff) + loaded_remove

        # The force when the platform is unloaded is fit by an affine function of time: F = offset + slope x time
        # This drift is then removed from the force measurement during the whole trial
        times          = numpy.arange(NbOfSamples)
        indices        = (times < loaded)+(times > unloaded)
        unloaded_times = times[indices]
        Input          = numpy.array([unloaded_times, numpy.ones(len(unloaded_times))]).T
        Force          = numpy.zeros((3,NbOfSamples))
        for dim in range(3):
            slope, offset = numpy.linalg.lstsq(Input,RawForce[dim,indices],rcond=None)[0]
            drift         = offset + slope*times
            Force[dim]    = RawForce[dim] - drift

        ## The force measurements when the platforms are unloaded are saved and analysed separately to determine the error due to the forceplate noise.
        Unloaded_durations.append(loaded)
        Unloaded_forceplates.append(Force[:,:loaded])
        Unloaded_durations.append(NbOfSamples-unloaded)
        Unloaded_forceplates.append(Force[:,unloaded:])

        ## The quiet standing periods at the start and end of the trial are determined 
        # - to truncate the trial 
        # - and to calculate the subject's weight from the vertical force

        # start of the initial quiet standing period
        stance_start = numpy.argmax(Force[2] > stance_cutoff) + stance_remove
        # end of the initial quiet standing period
        run_start    = stance_start+numpy.argmax(Force[2,stance_start:] < stance_cutoff)

        # The same analysis is applied to the time-reversed force to determine the start and end of the final quiet standing period
        rev_stance_start = numpy.argmax(Force[2,::-1] > stance_cutoff) + stance_remove
        rev_run_start    = rev_stance_start + numpy.argmax(Force[2,::-1][rev_stance_start:] < stance_cutoff)

        # start of the final quiet standing period
        run_end    = NbOfSamples - rev_run_start
        # end of the final quiet standing period
        stance_end = NbOfSamples - rev_stance_start
        
        # The start and end of the trial are rounded to even numbers so that position and force are aligned
        stance_start = int(int(stance_start*Kinematic_frequency/Force_frequency)*Force_frequency/Kinematic_frequency)
        stance_end   = int(int(stance_end*Kinematic_frequency/Force_frequency)*Force_frequency/Kinematic_frequency)

        ForceTruncated = Force[:,stance_start:stance_end]
        Truncate[subject+trial] = [stance_start,stance_end]

        # The vertical force during the quiet standing periods is used to calculate the weight
        Weight[subject].extend(list(Force[2,stance_start:run_start]))
        Weight[subject].extend(list(Force[2,run_end:stance_end]))

        numpy.savez('data\\preprocessed\\'+subject+'\\'+trial+'-Force.npz', Force = ForceTruncated, Frequency = Kinematic_frequency)

numpy.savez('data\\preprocessed\\truncate.npz', **Truncate) 

# The weight of each subject is determined from the quiet stance periods
for subject in ['A','B']:
    weight[subject] = numpy.median(Weight[subject])
    
# The force measurements when the platforms are unloaded are saved and analysed separately to determine the error due to the forceplate noise.
unloaded_duration = numpy.min(Unloaded_durations)
NbOfSegments      = len(Unloaded_durations)
Unloaded_force    = numpy.zeros((NbOfSegments,3,unloaded_duration))
for segment in range(NbOfSegments):
    Unloaded_force[segment] = Unloaded_forceplates[segment][:,:unloaded_duration]
numpy.savez('data\\unloaded_force.npz', Unloaded_force = Unloaded_force)


### Kinematics pre-processing 
print('Kinematics pre-processing')

# The marker labels are renamed
New_labels = {'TRA_L':'HEADL',                                                                                                                
'TRA_R':'HEADR',                                                                                                                
'GLA':'HEADF',                                                                                                                      
'ACR_L':'LSHO',                                                                                                                
'ACR_R':'RSHO',                                                                                                                
'LHC_L':'LELL',                                                                                                                
'LHC_R':'RELL',                                                                                                                
'WRI_L':'LWRU',                                                                                                                
'WRI_R':'RWRU',                                                                                                                
'SUP':'STERNSUP',                                                                                                                  
'C7':'C7',                                                                                                                   
'T8':'T8',                                                                      
'T12':'T12',                                                                                                             
'ASIS_L':'LASI',                                                                                                               
'ASIS_R':'RASI',                                                                                                               
'PSIS_L':'LPSI',                                                                                                               
'PSIS_R':'RPSI',                                                                                                               
'PS':'PS',                                                                                                       
'GTR_L':'LFTC',                                                                                                                
'GTR_R':'RFTC',                                                                                                                
'LFC_L':'LFLE',                                                                                                                
'LFC_R':'RFLE',                                                                                                                
'MFC_L':'LFME',                                                                                                                
'MFC_R':'RFME',                                                                                                                
'LM_L':'LLMAL',                                                                                                                 
'LM_R':'RLMAL',                                                                                                                 
'MM_L':'LMMAL',                                                                                                                 
'MM_R':'RMMAL',                                                                                                                 
'CAL_L':'LCAL2',                                                                                                                
'CAL_R':'RCAL2',                                                                                                                
'MT2_L':'LDMT2',                                                                                                                
'MT2_R':'RDMT2',                                                                                                                
'MT5_L':'LDMT5',                                                                                                                
'MT5_R':'RDMT5',                                                                                                                
'HAL_L':'LDH',                                                                                                                
'HAL_R':'RDH'}

Invisible = {}

for subject in ['A','B']:
    for trial in ['2.2','2.3']:
        print(subject, trial)

        start, stop    = Truncate[subject+trial]
        position_start = int(start*Kinematic_frequency/Force_frequency)
        position_end   = int(stop*Kinematic_frequency/Force_frequency)
        print('loading the data')
        input_file  = input_path+subject+'\\'+trial+'-RawMotion'
        data        = functions.loadmat(input_file)

        for key in data.keys():
            if key.startswith('Motion'):
                Key = key 

        Trajectories = data[Key]['Trajectories']['Labeled']
        Labels       = Trajectories['Labels']
        Data         = numpy.array(Trajectories['Data'])

        print('Renaming and gap-filling the markers')
        Position           = {}
        Position_upsampled = {}
        for nb, label in enumerate(Labels):
            if label in New_labels.keys():
                
                new_label = New_labels[label]
                position  = Data[nb,:3]/1000 # the position data is converted from millimeters to meters
                
                # How many samples are invisible?
                invisible = numpy.isnan(numpy.sum(position,axis = 0)[position_start:position_end+1])
                Invisible[subject+trial+new_label] = invisible
                if numpy.sum(invisible) > 0: 
                    print(new_label, 'invisible for %i samples' %numpy.sum(invisible))
                    visibility = invisible == False
                    position[:,position_start:position_end+1] = functions.gap_filling(position[:,position_start:position_end+1], visibility)
                Position[new_label] = position[:,position_start:position_end]  
                
                position_upsampled  = functions.upsample_one_signal(position, Kinematic_frequency, Force_frequency) # the data is upsampled to the force frequency
                Position_upsampled[new_label] = position_upsampled[:,start:stop] # the data is truncated from the start of the initial standing period to the end of the final standing period
            else:
                print(label,' doesnt have a corresponding label')
                        
        numpy.savez('data\\preprocessed\\'+subject+'\\'+trial+'-Motion.npz', **Position)
        numpy.savez('data\\preprocessed\\'+subject+'\\'+trial+'-MotionUpsampled.npz', **Position_upsampled)
        Invisible[subject+trial] = numpy.array(Invisible[subject+trial])
numpy.savez('data\\invisible.npz', **Invisible)

        
### Dataset 2 from Seethapathi & Srinavasan 2019 https://doi.org/10.7554/eLife.38371 ###
trials   = ['5','7','9']
subjects = range(8)

### Force pre-processing
print('Force pre-processing')

Force_frequency     = 1000 # Hertz
Kinematic_frequency = 100 # Hertz

loaded_cutoff = 30  # in Newton

for trial in trials:
    print('loading speed', trial)
    ## The matlab file is loaded
    data   = functions.loadmat(input_path+'RunningForceDataWithCoP_2pt'+trial+'mps.mat')
    for subject in subjects: 
        print('loading subject', subject)
        ## The force is extracted and reshaped into a numpy array of shape (3, NrOfSamples)
        forcedata = data['RunningForceDataWithCoP'][subject]
        forcedata = functions._todict(forcedata)
        
        print('calculations')
        NbOfSamples = len(forcedata['LForceX'])
        RawForce    = numpy.zeros((3, NbOfSamples))
        for side in ['L','R']:
            for d, dimension in enumerate(['X','Y','Z']): # X: lateral, Y: forwards, Z: vertical
                RawForce[d] += forcedata[side + 'Force' + dimension]

        ## The force is subsampled at the kinematic frequency 
        # Force_subsampled = functions.subsample_one_signal(RawForce, Force_frequency, Kinematic_frequency)
        # NbOfSamples      = numpy.shape(Force_subsampled)[1]
           
        ## Drift removal
        '''The forceplate drift in each direction is approximated by an affine function of time:  offset + slope*time
        through a least-squares fit on the forceplate measurements during the flight phases of running
        '''        
        # The force during flight is fit by an affine function of time: F = offset + slope x time
        # This drift is then removed from the force measurement during the whole trial
        times          = numpy.arange(NbOfSamples)
        indices        = RawForce[2] < loaded_cutoff
        unloaded_times = times[indices]
        Input          = numpy.array([unloaded_times, numpy.ones(len(unloaded_times))]).T
        Force          = numpy.zeros((3,NbOfSamples))
        for dim in range(3):
            slope, offset = numpy.linalg.lstsq(Input,RawForce[dim,indices],rcond=None)[0]
            drift         = offset + slope*times
            Force[dim]    = RawForce[dim] - drift
            
        numpy.savez('data\\preprocessed\\'+str(subject)+'\\'+trial+'-Force.npz', Force = Force, Frequency = Kinematic_frequency)

for subject in subjects: 
    Weight[subject] = []
for trial in trials: 
    print(trial)
    for subject in subjects: 
        Force = numpy.load('data\\preprocessed\\'+str(subject)+'\\'+trial+'-Force.npz')['Force']
        NbOfSamples = numpy.shape(Force)[1]
        
        ## The weight is calculated as the mean vertical force from the first stance onset to the last stance onset
        # stance onsets are determined as the instants when the vertical force crosses loaded_cutoff
        stance_start = numpy.arange(1,NbOfSamples)[(Force[2,:-1] < loaded_cutoff)*(Force[2,1:] > loaded_cutoff)]
        Weight[subject].extend(list(Force[2,stance_start[0]:stance_start[-1]]))

# The weight of each subject is determined by averaging over all trials of that subject
for subject in subjects:
    weight[str(subject)] = numpy.mean(Weight[subject])
numpy.savez('data\weight.npz',**weight)


# # ### Kinematics pre-processing 
print('Kinematics pre-processing')
for trial in trials:
    print(trial, 'loading the data')
    input_file  = input_path+ 'RunningMotionData_2pt'+trial+'mps.mat'
    Motiondata  = functions.loadmat(input_file)
    # for subject in [4]:
    for subject in subjects:
        print(subject)
        
        motiondata    = Motiondata['RunningMotionData'][subject]
        motiondata    = functions._todict(motiondata)
        Labels        = motiondata['MarkerNames']
        AllMarkerData = numpy.array(motiondata['AllMarkerData'])
        MarkerTimes   = motiondata['Times']

        Position      = {}
        Position_upsampled = {}
        for m, marker in enumerate(Labels):
            position = AllMarkerData[:,3*m:3*(m+1)].T
            Position[marker] = position
            Position_upsampled[marker] = functions.upsample_one_signal(position, Kinematic_frequency, Force_frequency)
                        
        numpy.savez('data\\preprocessed\\'+str(subject)+'\\'+trial+'-Motion.npz', **Position)
        numpy.savez('data\\preprocessed\\'+str(subject)+'\\'+trial+'-MotionUpsampled.npz', **Position_upsampled)
        
# The feet markers were swapped between left and right sides for subject 4: this is corrected here       
def Other_side(side):
    if side == 'L':
        return 'R'
    elif side == 'R':
        return 'L'
subject = 4
for trial in trials:
    for file in ['-Motion.npz', '-MotionUpsampled.npz']:
        Position    = numpy.load('data\\preprocessed\\'+str(subject)+'\\'+trial+file)
        NewPosition = {}
        for marker in Position.keys():
            if 'Foot' in marker:
                NewPosition[Other_side(marker[0])+marker[1:]] = Position[marker]
            else:
                NewPosition[marker] = Position[marker]
                
        numpy.savez('data\\preprocessed\\'+str(subject)+'\\'+trial+file, **NewPosition)    