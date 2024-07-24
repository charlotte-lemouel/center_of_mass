function [l1, l2] = estimator_gains(Force_std, Position_std, Frequency, mass)
% [l1, l2] = estimator_gains(Force_std, Position_std, Frequency, mass)
%
% Calculates the optimal estimator gains according to the measurement errors and sampling frequency
%
% Parameters
% ----------
% Force_std: float
%	 Standard deviation of the error in Ground reaction force (in N)
% Position_std: float
% 	Standard deviation of the error in CoM position obtained from the kinematics (in m)
% Frequency: int
% 	Sampling frequency of the (sub-sampled) CoM position and acceleration 
% mass: float
% 	Mass of the subject (in kg)
	
% Returns
% -------
% l1: float 
%	 Optimal estimator gain for position (dimensionless)
% l2: float 
% 	Optimal estimator gain for velocity (dimensionless)

if Force_std > 0 && Position_std > 0
    Acceleration_std = Force_std / mass;
    ratio = Position_std / Acceleration_std * Frequency^2;
    l2 = (4 * ratio + 1 - sqrt(1 + 8 * ratio)) / (4 * ratio^2); % optimal estimator gain for velocity (dimensionless)
    l1 = 1 - ratio^2 * l2^2;  % optimal estimator gain for position (dimensionless)
else
    if Force_std == 0 && Position_std == 0
        error('Either Force_std or Position_std must be strictly positive');
    elseif Force_std == 0
        l1 = 0;
        l2 = 0;
    elseif Position_std == 0
        l1 = 1;
        l2 = 2;
    else
        error('Force_std and Position_std must be positive');
    end
end

end