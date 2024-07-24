%Estimation of the whole body center of mass by optimal combination of the ground reaction force and the kinematic CoM

addpath(genpath('..\src-matlab'));

load example_data;

[Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass);
% visualisation
Acceleration = com_acceleration(GroundReactionForce, mass);
[Acc_subsampled, Pos_subsampled, Frequency] = subsample_two_signals(Acceleration, Force_frequency, Kinematic_com, Kinematic_frequency);
visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, Frequency, 'Default values');

%% Optional arguments

% Measurement errors
Position_std = 0.0035; % standard deviation (in m) of the error in CoM position obtained from the kinematics alone (in m)
Force_std    = 1;      % standard deviation of the error in force obtained from the forceplates (in N)
[Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass, 'Position_std', Position_std, 'Force_std',Force_std);
visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, Frequency, 'Measurement errors');

% The orientation of the gravity vector in the reference frame must be correctly specified.
% By default, the code assumes that the first two dimensions (x,y) are horizontal, and the third dimension (z) is vertical, oriented upwards. 
% If this is not the case, then the direction of gravity must be specified
% as gravity_direction: for example [0,-1,0] if the y-axis is vertical, directed upwards (instead of the z-axis).
% The code then calculates the CoM acceleration by subtracting the person’s weight from the Ground reaction force.
gravity_direction = [0,-1,0]; % if the y-axis is vertical, directed upwards (instead of the z-axis)
[Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass, 'gravity_direction', gravity_direction);
visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, Frequency, 'Orientation of gravity');

% Initial conditions
% By default, the initial position and speed for the estimator are obtained by a least-squares fit of the first and last ten samples of the kinematic Center of Mass. 
% This number of samples can be modified by specifying the optional argument “initial_samples”.
initial_samples = 20;
[Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass, 'initial_samples', initial_samples);
visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, Frequency, 'Initial conditions: number of samples');

% Alternately, the user may wish to use a custom method to determine the initial conditions. 
% In this case, they can specify the desired initial conditions for both the forwards estimator using the optional argument “Initial_conditions”, and for the backwards estimator using the optional argument “Final_conditions”.
% In the example data, the subject is initially static. We can specify an initial velocity of zero.
initial_position = Kinematic_com(:,1);
%initial_velocity = zeros(3);
Initial_conditions = [initial_position, [0,0,0]'];
[Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass, 'Initial_conditions',Initial_conditions);
visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, Frequency, 'Initial conditions: initial velocity');

% In the example data, the kinematics are sampled at 200 Hz and the force at 1000 Hz. The code automatically selects a sub-sampling frequency of 200 Hz for the two signals (corresponding to the greatest common divisor). 
% However, the user can specify any common divisor as the sub-sampling frequency through the optional argument 'sub_frequency'.
sub_frequency = 100;
[Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass, 'sub_frequency', sub_frequency);
[Acc_subsampled, Pos_subsampled, Frequency] = subsample_two_signals(Acceleration, Force_frequency, Kinematic_com, Kinematic_frequency, sub_frequency);
visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, sub_frequency, 'Sub-sampling frequency');
