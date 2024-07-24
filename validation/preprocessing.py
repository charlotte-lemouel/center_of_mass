import numpy, os, scipy, functions, center_of_mass

# We generate the folder in which the pre-processed data will be exported
if not os.path.exists('data\\preprocessed'):
    os.mkdir('data\\preprocessed')
    for subject in ['A','B']:
        os.mkdir('data\\preprocessed\\'+subject)
        
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
stance_remove = 500 # number of samples @ 500 Hz to remove before the vertical force reaches the stance_cutoff, and after it drops below it.
Truncate = {}
Weight   = {'A':[],'B':[]}
        
for subject in ['A','B']: 
    for trial in ['2.2','2.3']:

        ## The matlab file is loaded
        input_file  = 'data\\raw\\'+subject+'\\'+trial+'-RawForce'
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
        Force_subsampled = center_of_mass.estimator.subsample_one_signal(RawForce, Force_frequency, Kinematic_frequency)
        NbOfSamples      = numpy.shape(Force_subsampled)[1]
           
        ## Drift removal
        '''The forceplate drift in each direction is approximated by an affine function of time:  offset + slope*time
        through a least-squares fit on the forceplate measurements:
        - at the start of the trial, before the subject steps onto the treadmill
        - and at the end of the trial, after the subject steps off the treadmill
        '''        
        # Instant when the person steps onto the treadmill (@ 500 Hz): loaded_remove samples before the vertical force exceeds loaded_cutoff
        loaded   = numpy.argmax(Force_subsampled[2] > loaded_cutoff) - loaded_remove
        # The same analysis is applied to the time-reversed force to determine when the subject steps off the platforms
        unloaded = NbOfSamples - numpy.argmax(Force_subsampled[2,::-1] > loaded_cutoff) + loaded_remove

        # The force when the platform is unloaded is fit by an affine function of time: F = offset + slope x time
        # This drift is then removed from the force measurement during the whole trial
        times          = numpy.arange(NbOfSamples)
        indices        = (times < loaded)+(times > unloaded)
        unloaded_times = times[indices]
        Input          = numpy.array([unloaded_times, numpy.ones(len(unloaded_times))]).T
        Force          = numpy.zeros((3,NbOfSamples))
        for dim in range(3):
            slope, offset = numpy.linalg.lstsq(Input,Force_subsampled[dim,indices],rcond=None)[0]
            drift         = offset + slope*times
            Force[dim]    = Force_subsampled[dim] - drift

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

        ForceTruncated = Force[:,stance_start:stance_end]
        Truncate[subject+trial] = [stance_start,stance_end]

        # The vertical force during the quiet standing periods is used to calculate the weight
        Weight[subject].extend(list(Force[2,stance_start:run_start]))
        Weight[subject].extend(list(Force[2,run_end:stance_end]))

        numpy.savez('data\\preprocessed\\'+subject+'\\'+trial+'-Force.npz', Force = ForceTruncated, Frequency = Kinematic_frequency)

# The force measurements when the platforms are unloaded are saved and analysed separately to determine the error due to the forceplate noise.
unloaded_duration = numpy.min(Unloaded_durations)
NbOfSegments      = len(Unloaded_durations)
Unloaded_force    = numpy.zeros((NbOfSegments,3,unloaded_duration))
for segment in range(NbOfSegments):
    Unloaded_force[segment] = Unloaded_forceplates[segment][:,:unloaded_duration]
numpy.savez('data\\unloaded_force.npz', Unloaded_force = Unloaded_force)

# The weight of each subject is determined from the quiet stance periods
weight = {}
for subject in ['A','B']:
    weight[subject] = numpy.median(Weight[subject])
numpy.savez('data\weight.npz',**weight)

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

for subject in ['A','B']:
    for trial in ['2.2','2.3']:
        print(subject, trial)

        start, stop = Truncate[subject+trial]
        print('loading the data')
        input_file  = 'data\\raw\\'+subject+'\\'+trial+'-RawMotion'
        data        = functions.loadmat(input_file)

        for key in data.keys():
            if key.startswith('Motion'):
                Key = key 

        Trajectories = data[Key]['Trajectories']['Labeled']
        Labels       = Trajectories['Labels']
        Data         = numpy.array(Trajectories['Data'])

        print('Renaming and gap-filling the markers')
        Position   = {}
        for nb, label in enumerate(Labels):
            if label in New_labels.keys():
                
                new_label = New_labels[label]
                position  = Data[nb,:3, start:stop] # the data is truncated from the start of the initial standing period to the end of the final standing period
                position  = position/1000 # the position data is converted from millimeters to meters
                
                # the sections when the marker is invisible are gap-filled by linear interpolation 
                invisible = numpy.isnan(numpy.sum(position,axis = 0))
                if numpy.sum(invisible) > 0: 
                    print(new_label, 'invisible for %i samples' %numpy.sum(invisible))
                    visibility          = invisible == False
                    Position[new_label] = functions.gap_filling(position, visibility)
                else:
                    Position[new_label] = position
            
            else:
                print(label,' doesnt have a corresponding label')
                        
        numpy.savez('data\\preprocessed\\'+subject+'\\'+trial+'-Motion.npz', **Position)