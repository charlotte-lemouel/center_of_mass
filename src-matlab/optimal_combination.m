function [Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass, varargin)
% [Pos_estimate, Vel_estimate, Frequency] = optimal_combination(GroundReactionForce, Force_frequency, Kinematic_com, Kinematic_frequency, mass, varargin)
%
% Combines the Center of Mass position obtained from kinematic measurements with Ground reaction force to estimate the Center of Mass position and velocity
%
% Parameters
% ----------
% GroundReactionForce: (NbOfDimensions,NbOfSamples1) double array
%	 Ground reaction force (in Newton)
% Force_frequency: int 
%	 Sampling frequency (in Hertz) of the Ground reaction force
% Kinematic_com: (NbOfDimensions,NbOfSamples2) double array
%	 Center of Mass position obtained from kinematic measurements (in m)
% Kinematic_frequency: int 
% 	Sampling frequency (in Hertz) of the kinematics
% mass: float 
%	 subject's mass (in kg)
% Force_std: float, optional
%	 Standard deviation of the error in Ground reaction force, default is 2 (in N)
% Position_std: float, optional
% 	Standard deviation of the error in CoM position obtained from the kinematics, default is 0.002 (in m)
% gravity_direction: (NbOfDimensions) double array, optional
%	 direction of the gravity vector used to subtract the subject's weight, default is [0,0,-1]
% sub_frequency: int, optional 
%	 Desired sub-sampling frequency (in Hertz), default is []
% Initial_conditions: (NbOfDimensions,2) double array, optional
%	 Initial estimates of position (in m) and velocity (in m/s), used when the estimator is applied forwards in time (default is None).
%	 If None, the initial conditions are determined by a least-squares fit on the first few samples.
% Final_conditions: (NbOfDimensions,2) double array, optional
%	 Final estimates of position (in m) and velocity (in m/s), used when the estimator is applied backwards in time (default is None).
% 	 If None, the final conditions are determined by a least-squares fit on the first few samples.
% initial_samples: int, optional
% 	 Number of samples used to estimate initial and final position and velocity (default is 10)
%
% Returns
% -------
% Pos_estimate: (NbOfDimensions, NbOfSamples) double array
% 	Position estimate (in m)
% Vel_estimate: (NbOfDimensions, NbOfSamples) double array
% 	Velocity estimate (in m/s)
% Frequency: int 
% 	Sampling frequency of the position and velocity estimates

% Optional arguments
if ismember('Force_std',varargin(1:2:end))
    Force_std = varargin{find(strcmp('Force_std',varargin))+1};
else
    Force_std = 2; % default, in N
end
if ismember('Position_std',varargin(1:2:end))
    Position_std = varargin{find(strcmp('Position_std',varargin))+1};
else
    Position_std = 0.002; % default, in m
end
if ismember('gravity_direction',varargin(1:2:end))
    gravity_direction = varargin{find(strcmp('gravity_direction',varargin))+1};
else
    gravity_direction = [0,0,-1]; % default
end
if ismember('sub_frequency',varargin(1:2:end))
    sub_frequency = varargin{find(strcmp('sub_frequency',varargin))+1};
else
    sub_frequency = []; % default
end
if ismember('Initial_conditions',varargin(1:2:end))
    Initial_conditions = varargin{find(strcmp('Initial_conditions',varargin))+1};
else
    Initial_conditions = []; % default
end
if ismember('Final_conditions',varargin(1:2:end))
    Final_conditions = varargin{find(strcmp('Final_conditions',varargin))+1};
else
    Final_conditions = []; % default
end
if ismember('initial_samples',varargin(1:2:end))
    initial_samples = varargin{find(strcmp('initial_samples',varargin))+1};
else
    initial_samples = 10; % default
end

% The CoM acceleration is calculated from the Ground reaction force and mass 
Acceleration = com_acceleration(GroundReactionForce, mass, gravity_direction);

% Position and Acceleration are sub-sampled at a common frequency
[Acc_subsampled, Pos_subsampled, Frequency] = subsample_two_signals(Acceleration, Force_frequency, Kinematic_com, Kinematic_frequency, sub_frequency);

% The optimal estimator gains at that Frequency are calculated
[l1, l2] = estimator_gains(Force_std, Position_std, Frequency, mass);

% The estimator is applied both forwards in time and backwards, and the forwards and backwards estimates are merged.
[Pos_estimate, Vel_estimate] = estimator_backandforth(Acc_subsampled, Pos_subsampled, l1, l2, Frequency, 'Initial_conditions', Initial_conditions, 'Final_conditions', Final_conditions, 'initial_samples', initial_samples);

end