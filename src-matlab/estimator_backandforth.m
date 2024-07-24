function [Pos_estimate, Vel_estimate] = estimator_backandforth(Acc_measurement, Pos_measurement, l1, l2, Frequency, varargin)
% [Pos_estimate, Vel_estimate] = estimator_backandforth(Acc_measurement, Pos_measurement, l1, l2, Frequency, varargin)
%
% The estimator is applied, for each dimension separately, both forwards and backwards 
% in time, and the forwards and backwards estimates are then merged.
%
% Parameters
% -------
% Acc_measurement: (NbOfDimensions, NbOfSamples) double array
%     Acceleration measurement (in m/s^2)
% Pos_measurement: (NbOfDimensions, NbOfSamples) double array
%     Position measurement (in m)
% l1: double or (NbOfDimensions, 1) double array
%     Position estimator gain (dimensionless)
% l2: double or (NbOfDimensions, 1) double array
%     Velocity estimator gain (dimensionless)
% Frequency: double
%     Sampling frequency (in Hertz)
% Initial_conditions: (NbOfDimensions,2) double array, optional
%     Initial estimates of position (in m) and velocity (in m/s), used when the estimator is applied forwards 
%     in time (default is []). If empty, the initial conditions are determined by a least-squares fit on the 
%     first few samples.
% Final_conditions: (NbOfDimensions,2) double array, optional
%     Final estimates of position (in m) and velocity (in m/s), used when the estimator is applied forwards 
%     in time (default is []). If empty, the initial conditions are determined by a least-squares fit on the 
%     first few samples.
% initial_samples: int, optional
%     Number of samples used to estimate initial and final position and velocity (default is 10)
%
% Returns
% -------
% Pos_estimate: (NbOfDimensions, NbOfSamples) double array
%     Position estimate (in m)
% Vel_estimate: (NbOfDimensions, NbOfSamples) double array
%     Velocity estimate (in m/s)

% Optional arguments
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

% Initialization
[NbOfDimensions, NbOfSamples] = size(Pos_measurement);
Pos_estimate_forw = zeros(NbOfDimensions, NbOfSamples);
Vel_estimate_forw = zeros(NbOfDimensions, NbOfSamples);
Pos_estimate_back = zeros(NbOfDimensions, NbOfSamples);
Vel_estimate_back = zeros(NbOfDimensions, NbOfSamples);

% If a single value is given for l1 and l2, these are used for all dimensions
if numel(l1) == 1
    l1 = l1 * ones(1, NbOfDimensions);
end
if numel(l2) == 1
    l2 = l2 * ones(1, NbOfDimensions);
end

for dim = 1:NbOfDimensions
    % If the user does not specify the initial estimate of position and velocity, these are estimated from the first few samples of position
    if isempty(Initial_conditions)
        Initial_conditions_dim = initial_conditions(Pos_measurement(dim, :), 1/Frequency, initial_samples);     
    else
        Initial_conditions_dim = Initial_conditions(dim, :);
    end
    if isempty(Final_conditions)
        Final_conditions_dim = initial_conditions(fliplr(Pos_measurement(dim, :)), 1/Frequency, initial_samples);
    else
        Final_conditions_dim = Final_conditions(dim, :);
    end
    [Pos_estimate_forw(dim, :), Vel_estimate_forw(dim, :)] = estimator(Acc_measurement(dim, :), Pos_measurement(dim, :), l1(dim), l2(dim), 1 / Frequency, Initial_conditions_dim);
    [Pos_estimate_back(dim, :), Vel_estimate_back(dim, :)] = estimator(fliplr(Acc_measurement(dim, :)), fliplr(Pos_measurement(dim, :)), l1(dim), l2(dim), 1 / Frequency, Final_conditions_dim);
end

Pos_estimate = 0.5 * (Pos_estimate_forw + fliplr(Pos_estimate_back));
Vel_estimate = 0.5 * (Vel_estimate_forw - fliplr(Vel_estimate_back));

end