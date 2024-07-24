function Initial_conditions = initial_conditions(Pos_measurement, T, initial_samples)
% Initial_conditions = initial_conditions(Pos_measurement, T, initial_samples)
%
% The initial estimates of position and velocity are obtained as a least-squares fit 
% on the first few samples of the position measurement.
%
% Parameters
% ----------
% Pos_measurement: (NbOfSamples,1) double array
%     Position measurement (in m)
% T: double
%     duration (in seconds) between successive samples (i.e. 1/Sampling_frequency)
% initial_samples: int, optional
%     Number of samples used to estimate initial position and velocity (default is 10)
%
% Returns
% -------
% pos_initial: double
%     Initial position estimate (in m)
% vel_initial: double
%     Initial velocity estimate (in m/s)


if nargin < 3
    initial_samples = 10; % Default value if initial_samples is not provided
end

Coefficients = [ones(initial_samples, 1), (0:initial_samples-1)' * T];
Initial_conditions = Coefficients \ Pos_measurement(1:initial_samples)'; 

end