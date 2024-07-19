import numpy

# The Center of Mass acceleration is calculated by subtracting the weight from the ground reaction force and dividing by mass. 
# By default, the vertical axis is assumed to correspond to the third row of the Ground Reaction Force, and to be oriented upwards. 
# However, this can be changed through the arguments 'vertical' and 'direction'.
def com_acceleration(GroundReactionForce, mass, gravity_direction = numpy.array([0,0,-1])):
    '''Calculates the CoM acceleration from the Ground reaction force and mass 
    
    Parameters
    ----------
    GroundReactionForce: (NbOfDimensions,NbOfSamples) numpy.ndarray
        Ground reaction force (in Newton)
    mass: float 
        subject's mass (in kg)
    gravity_direction: (NbOfDimensions) numpy.ndarray, optional
        direction of the gravity vector used to subtract the subject's weight, default is numpy.array([0,0,-1])

    Returns
    -------
    Acceleration: (NbOfDimensions,NbOfSamples) numpy.ndarray
        Acceleration of the Center of Mass (in m/s^2)
    '''
    ## The net force is the sum of the ground reaction force and the person's weight 
    NetForce = GroundReactionForce + mass*9.81*numpy.array([gravity_direction]).T
    ## Acceleration is force divided by mass
    Acceleration = NetForce/mass
    return Acceleration
    
# The position and acceleration signals must be synchronised before applying the filtering. 
# If they have a different sampling frequency, they are sub-sampled at a common frequency. 
# The user can specify a desired sub-sampling frequency (which must be a common divisor of both Position_frequency and Acceleration_frequency). 
# Otherwise, the greatest common divisor is used as the sub-sampling frequency.    
def subsample_one_signal(signal, signal_frequency, sub_frequency):
    '''Subsample a signal at a given Frequency
    
    Parameters
    ----------
    signal: (NbOfDimensions, NbOfSamples) numpy.ndarray
        Signal to subsample
    signal_frequency: int 
        Sampling frequency (in Hertz) of the signal
    sub_frequency: int 
        Desired sub-sampling frequency (in Hertz)
        
    Returns
    -------   
    signal_subsampled: (NbOfDimensions, NbOfSamples_sub) numpy.ndarray
        Subsampled signal
    '''
    # The sub-sampling frequency must be a divisor of the signal frequency:
    if signal_frequency % sub_frequency != 0:
        raise ValueError('The sub-sampling frequency is not a divisor of the signal frequency. This would lead to synchronisation issues when sub-sampling.')
    else:
        if signal_frequency == sub_frequency:
            return signal
        else:    
            bin_size          = int(signal_frequency/sub_frequency) # the signal will be averaged over bins of this size
            NbOfDimensions    = numpy.shape(signal)[0]
            NbOfSamples_sub   = int(numpy.shape(signal)[1]/bin_size)
            signal_truncated  = signal[:,:bin_size*NbOfSamples_sub] 
            signal_subsampled = numpy.zeros((NbOfDimensions,NbOfSamples_sub))
            for dim in range(NbOfDimensions):
                signal_reshape         = signal_truncated[dim].reshape(NbOfSamples_sub,bin_size)
                signal_subsampled[dim] = numpy.mean(signal_reshape, axis = 1)   
            return signal_subsampled

def subsample_two_signals(signal1, frequency1, signal2, frequency2, sub_frequency = None):
    '''Subsample two signals at a common frequency
    
    Parameters
    ----------
    signal1: (NbOfDimensions,NbOfSamples1) numpy.ndarray
        First signal
    frequency1: int 
        Sampling frequency (in Hertz) of the first signal
    signal2: (NbOfDimensions,NbOfSamples2) numpy.ndarray
        Second signal
    frequency2: int 
        Sampling frequency (in Hertz) of the second signal
    sub_frequency: int, optional
        Desired sub-sampling frequency (in Hertz), default is None
        
    Returns
    -------   
    signal1_subsampled: (NbOfDimensions,NbOfSamples_sub) numpy.ndarray
        Subsampled first signal
    signal2_subsampled: (NbOfDimensions,NbOfSamples_sub) numpy.ndarray
        Subsampled second signal
    sub_frequency: int 
        Subsampling frequency (in Hertz)
        If the sub-sampling frequency was not specified by the user, this is the greatest common divisor of frequency1 and frequency2
    '''
    if not sub_frequency:
        ## The frequency at which both signals will be subsampled is the greatest common divisor of the two frequencies
        import math
        sub_frequency = math.gcd(int(frequency1), int(frequency2))
        
    signal1_subsampled = subsample_one_signal(signal1, frequency1, sub_frequency)
    signal2_subsampled = subsample_one_signal(signal2, frequency2, sub_frequency)
    
    # Truncate the signals so that they have the same length
    NbOfSamples_sub = min(numpy.shape(signal1_subsampled)[1],numpy.shape(signal2_subsampled)[1])
    signal1_subsampled = signal1_subsampled[:,:NbOfSamples_sub]
    signal2_subsampled = signal2_subsampled[:,:NbOfSamples_sub]
    
    return signal1_subsampled, signal2_subsampled, sub_frequency

# The optimal estimator gains depend on the dimensionless ratio of position to acceleration noise
def estimator_gains(Force_std, Position_std, Frequency, mass):
    '''Calculates the optimal estimator gains according to the measurement errors and sampling frequency
    
    Parameters
    ----------
    Force_std: float or (NbOfDimensions,) numpy.ndarray
        Standard deviation of the error in Ground reaction force (in N) (can be provided for each dimension independently)
    Position_std: float or (NbOfDimensions,) numpy.ndarray
        Standard deviation of the error in CoM position obtained from the kinematics (in m) (can be provided for each dimension independently)
    Frequency: int
        Sampling frequency of the (sub-sampled) CoM position and acceleration 
    mass: float
        Mass of the subject (in kg)
        
    Returns
    -------
    l1: float or (NbOfDimensions,) numpy.ndarray
        Optimal estimator gain for position (dimensionless)
    l2: float or (NbOfDimensions,) numpy.ndarray
        Optimal estimator gain for velocity (dimensionless)
    '''
    Acceleration_std = Force_std/mass
    ratio = Position_std/Acceleration_std*Frequency**2
    l2    = numpy.array((4*ratio+1 - numpy.sqrt(1+8*ratio))/(4*ratio**2))
    l1    = numpy.array(1 - ratio**2*l2**2)
    if not numpy.array(((Position_std > 0)*(Force_std > 0))).all():
        if numpy.array((Force_std == 0)*(Position_std == 0)).any():
            raise ValueError('Either Force_std or Position_std must be strictly positive')
        elif numpy.array(Force_std == 0).any(): 
            l1[Force_std == 0] = 0
            l2[Force_std == 0] = 0
        elif numpy.array(Position_std == 0).any():
            l1[Position_std == 0] = 1
            l2[Position_std == 0] = 2
        else:
            raise ValueError('Force_std and Position_std must be positive')
    return l1, l2
   
# Unless they are specified by the user, the initial estimates of position and velocity are obtained as a least-squares fit on the first few samples of the position measurement.    
def initial_conditions(Pos_measurement, T, initial_samples = 10):
    '''The initial estimates of position and velocity are obtained as a least-squares fit on the first few samples of the position measurement.
    
    Parameters
    ----------
    Pos_measurement: (NbOfSamples,) numpy.ndarray
        Position measurement (in m)
    T: float 
        duration (in seconds) between successive samples (i.e. 1/Sampling_frequency)
    initial_samples: int, optional   
        Number of samples used to estimate initial position and velocity (default is 10)
        
        
    Returns
    -------
    pos_initial: float
        Initial position estimate (in m)
    vel_initial: float
        Initial velocity estimate (in m/s)
    '''
    Coefficients = numpy.array([numpy.ones(initial_samples),numpy.arange(initial_samples)*T]).T
    [pos_initial, vel_initial] = numpy.linalg.lstsq(Coefficients, Pos_measurement[:initial_samples], rcond=None)[0]
    return pos_initial, vel_initial 
    
def estimator(Acc_measurement, Pos_measurement, l1, l2, T, Initial_conditions):
    '''Estimation of the position and velocity, given (noisy) measurements of position and acceleration,
    estimator gains, and initial conditions
    
    Parameters
    ----------
    Acc_measurement: (NbOfSamples,) numpy.ndarray
        Acceleration measurement (in m/s^2)
    Pos_measurement: (NbOfSamples,) numpy.ndarray
        Position measurement (in m)
    l1: float or (NbOfDimensions,) numpy.ndarray
        Position estimator gain (dimensionless)
    l2: float or (NbOfDimensions,) numpy.ndarray
        Velocity estimator gain (dimensionless)
    T: float 
        duration (in seconds) between successive samples (i.e. 1/Sampling_frequency)
    Initial_conditions: (2,) numpy.ndarray
        initial estimates of position (in m) and velocity (in m/s)
    
    Returns
    -------
    Pos_estimate: (NbOfSamples,) numpy.ndarray
        Position estimate (in m)
    Vel_estimate: (NbOfSamples,) numpy.ndarray
        Velocity estimate (in m/s)
    '''
    Pos_estimate    = numpy.zeros(len(Pos_measurement))
    Vel_estimate    = numpy.zeros(len(Pos_measurement))
    Pos_estimate[0], Vel_estimate[0] = Initial_conditions
    for k in range(len(Pos_measurement)-1):
        pos_prediction    = Pos_estimate[k]+T*Vel_estimate[k]+T**2/2*Acc_measurement[k]
        vel_prediction    = Vel_estimate[k]+T*Acc_measurement[k]
        Pos_estimate[k+1] = (1-l1)*pos_prediction   +l1*Pos_measurement[k+1]
        Vel_estimate[k+1] = vel_prediction +l2/T*(Pos_measurement[k+1]-pos_prediction)
    return Pos_estimate, Vel_estimate    
    
def estimator_backandforth(Acc_measurement, Pos_measurement, l1, l2, Frequency, Initial_conditions = None, Final_conditions = None, initial_samples = 10):
    '''The estimator is applied, for each dimension separately, both forwards and backwards in time, and the forwards and backwards estimates are then merged.
    
    Parameters
    ----------
    Acc_measurement: (NbOfDimensions, NbOfSamples) numpy.ndarray
        Acceleration measurement (in m/s^2)
    Pos_measurement: (NbOfDimensions, NbOfSamples) numpy.ndarray
        Position measurement (in m)
    l1: float or (NbOfDimensions,) numpy.ndarray
        Position estimator gain (dimensionless)
    l2: float or (NbOfDimensions,) numpy.ndarray
        Velocity estimator gain (dimensionless)
    Frequency: float 
        Sampling frequency (in Hertz)
    Initial_conditions: (NbOfDimensions,2) numpy.ndarray, optional
        Initial estimates of position (in m) and velocity (in m/s), used when the estimator is applied forwards in time (default is None).
        If None, the initial conditions are determined by a least-squares fit on the first few samples.
    Final_conditions: (NbOfDimensions,2) numpy.ndarray, optional
        Final estimates of position (in m) and velocity (in m/s), used when the estimator is applied backwards in time (default is None).
        If None, the final conditions are determined by a least-squares fit on the first few samples.
    initial_samples: int, optional
        Number of samples used to estimate initial and final position and velocity (default is 10)
    
    Returns
    -------
    Pos_estimate: (NbOfDimensions, NbOfSamples) numpy.ndarray
        Position estimate (in m)
    Vel_estimate: (NbOfDimensions, NbOfSamples) numpy.ndarray
        Velocity estimate (in m/s)
    '''
    NbOfDimensions, NbOfSamples = numpy.shape(Pos_measurement)
    Pos_estimate_forw  = numpy.zeros((NbOfDimensions, NbOfSamples))
    Vel_estimate_forw  = numpy.zeros((NbOfDimensions, NbOfSamples))
    Pos_estimate_back  = numpy.zeros((NbOfDimensions, NbOfSamples))
    Vel_estimate_back  = numpy.zeros((NbOfDimensions, NbOfSamples))
    
    # If a single value is given for l1 and l2, these are used for all dimensions
    if len(numpy.shape(l1)) == 0: 
        l1 = l1*numpy.ones((NbOfDimensions))
    if len(numpy.shape(l2)) == 0:     
        l2 = l2*numpy.ones((NbOfDimensions))
    
    for dim in range(NbOfDimensions):
        # If the user does not specify the initial estimate of position and velocity, these are estimated from the first few samples of position 
        if numpy.sum(Initial_conditions==None) > 0:
            Initial_conditions_dim = initial_conditions(Pos_measurement[dim], 1/Frequency, initial_samples = initial_samples)
        else:
            Initial_conditions_dim = Initial_conditions[dim]
        if numpy.sum(Final_conditions==None) > 0:
            Final_conditions_dim = initial_conditions(Pos_measurement[dim,::-1], 1/Frequency, initial_samples = initial_samples)
        else:
            Final_conditions_dim = Final_conditions[dim]
        Pos_estimate_forw[dim], Vel_estimate_forw[dim] = estimator(Acc_measurement[dim],      Pos_measurement[dim],      l1[dim], l2[dim], 1/Frequency, Initial_conditions_dim)
        Pos_estimate_back[dim], Vel_estimate_back[dim] = estimator(Acc_measurement[dim,::-1], Pos_measurement[dim,::-1], l1[dim], l2[dim], 1/Frequency, Final_conditions_dim)
    Pos_estimate = 0.5*(Pos_estimate_forw+Pos_estimate_back[:,::-1])
    Vel_estimate = 0.5*(Vel_estimate_forw-Vel_estimate_back[:,::-1])    
    return Pos_estimate, Vel_estimate
    
def optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass,
    Force_std = 2, Position_std = 0.002, gravity_direction = numpy.array([0,0,-1]), sub_frequency = None, Initial_conditions = None, Final_conditions = None, initial_samples = 10):
    '''Combines the Center of Mass position obtained from kinematic measurements with Ground reaction force to estimate the Center of Mass position and velocity
    
    Parameters
    ----------
    GroundReactionForce: (NbOfDimensions,NbOfSamples1) numpy.ndarray
        Ground reaction force (in Newton)
    Force_frequency: int 
        Sampling frequency (in Hertz) of the Ground reaction force
    Kinematic_com: (NbOfDimensions,NbOfSamples2) numpy.ndarray
        Center of Mass position obtained from kinematic measurements (in m)
    Kinematic_frequency: int 
        Sampling frequency (in Hertz) of the kinematics
    mass: float 
        subject's mass (in kg)
    Force_std: float or (NbOfDimensions,) numpy.ndarray, optional
        Standard deviation of the error in Ground reaction force (in N) (can be provided for each dimension independently), default is 2 N
    Position_std: float or (NbOfDimensions,) numpy.ndarray, optional
        Standard deviation of the error in CoM position obtained from the kinematics (in m) (can be provided for each dimension independently), default is 0.002 m
    gravity_direction: (NbOfDimensions) numpy.ndarray, optional
        direction of the gravity vector used to subtract the subject's weight, default is numpy.array([0,0,-1])
    sub_frequency: int, optional 
        Desired sub-sampling frequency (in Hertz), default is None
    Initial_conditions: (NbOfDimensions,2) numpy.ndarray, optional
        Initial estimates of position (in m) and velocity (in m/s), used when the estimator is applied forwards in time (default is None).
        If None, the initial conditions are determined by a least-squares fit on the first few samples.
    Final_conditions: (NbOfDimensions,2) numpy.ndarray, optional
        Final estimates of position (in m) and velocity (in m/s), used when the estimator is applied backwards in time (default is None).
        If None, the final conditions are determined by a least-squares fit on the first few samples.
    initial_samples: int, optional
        Number of samples used to estimate initial and final position and velocity (default is 10)
    
    Returns
    -------
    Pos_estimate: (NbOfDimensions, NbOfSamples) numpy.ndarray
        Position estimate (in m)
    Vel_estimate: (NbOfDimensions, NbOfSamples) numpy.ndarray
        Velocity estimate (in m/s)
    Frequency: int 
        Sampling frequency of the position and velocity estimates
    '''
    
    
    # The CoM acceleration is calculated from the Ground reaction force and mass 
    Acceleration = com_acceleration(GroundReactionForce, mass, gravity_direction = gravity_direction)

    # Position and Acceleration are sub-sampled at a common frequency
    Acc_subsampled, Pos_subsampled, Frequency = subsample_two_signals(Acceleration, Force_frequency, Kinematic_com, Kinematic_frequency, sub_frequency = sub_frequency)
    
    # The optimal estimator gains at that Frequency are calculated
    l1, l2 = estimator_gains(Force_std, Position_std, Frequency, mass)
    
    ## The estimator is applied both forwards in time and backwards, and the forwards and backwards estimates are merged.
    Pos_estimate, Vel_estimate = estimator_backandforth(Acc_subsampled, Pos_subsampled, l1, l2, Frequency, Initial_conditions = Initial_conditions, Final_conditions = Final_conditions, initial_samples = initial_samples)
    
    return Pos_estimate, Vel_estimate, Frequency
