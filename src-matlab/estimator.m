function [Pos_estimate, Vel_estimate]  = estimator(Acc_measurement, Pos_measurement, l1, l2, T, Initial_conditions)

% Estimation of the position and velocity, given (noisy) measurements of position and acceleration,
% estimator gains, and initial conditions
%
% Parameters
% ----------
% Acc_measurement: (NbOfSamples,1) double array
%     Acceleration measurement (in m/s^2)
% Pos_measurement: (NbOfSamples,1) double array
%     Position measurement (in m)
% l1: double or (NbOfDimensions,1) double array
%     Position estimator gain (dimensionless)
% l2: double or (NbOfDimensions,1) double array
%     Velocity estimator gain (dimensionless)
% T: double
%     duration (in seconds) between successive samples (i.e. 1/Sampling_frequency)
% Initial_conditions: (2,1) double array
%     initial estimates of position (in m) and velocity (in m/s)
%
% Returns
% -------
% Pos_estimate: (NbOfSamples,1) double array
%     Position estimate (in m)
% Vel_estimate: (NbOfSamples,1) double array
%     Velocity estimate (in m/s)

Pos_estimate = zeros(length(Pos_measurement), 1);
Vel_estimate = zeros(length(Pos_measurement), 1);
Pos_estimate(1) = Initial_conditions(1);
Vel_estimate(1) = Initial_conditions(2);

for p = 1:length(Pos_measurement) - 1
    pos_prediction = Pos_estimate(p) + T * Vel_estimate(p) + T^2 / 2 * Acc_measurement(p);
    vel_prediction = Vel_estimate(p) + T * Acc_measurement(p);
    Pos_estimate(p + 1) = (1 - l1) * pos_prediction + l1 * Pos_measurement(p + 1);
    Vel_estimate(p + 1) = vel_prediction + l2 / T * (Pos_measurement(p + 1) - pos_prediction);
end

end