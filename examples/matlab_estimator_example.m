%Estimation of the whole body center of mass by optimal combination of the ground reaction force and the kinematic CoM

addpath(genpath('..\src-matlab'));

% Parameters for calculating optimal estimation 
Position_std = 0.0035; % standard deviation (in m) of the error in CoM position obtained from the kinematics alone (in m)
Force_std    = 2;      % standard deviation of the error in force obtained from the forceplates (in N)

load example_data;

[Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Force_std, Kinematic_com, Kinematic_frequency, Position_std, mass);

% visualisation
Acceleration = com_acceleration(GroundReactionForce, mass);
[Acc_subsampled, Pos_subsampled, Frequency] = subsample_two_signals(Acceleration, Force_frequency, Kinematic_com, Kinematic_frequency);
visualise(Acc_subsampled, Pos_subsampled, Pos_estimate, Vel_estimate, Frequency);