function [signal1_subsampled, signal2_subsampled, sub_frequency] = subsample_two_signals(signal1, frequency1, signal2, frequency2, sub_frequency)
% Subsample two signals at a common frequency
%
% Parameters
% ----------
% signal1: (NbOfDimensions, NbOfSamples1) array
%     First signal
% frequency1: int 
%     Sampling frequency (in Hertz) of the first signal
% signal2: (NbOfDimensions, NbOfSamples2) array
%     Second signal
% frequency2: int 
%     Sampling frequency (in Hertz) of the second signal
% sub_frequency: int, optional
%     Desired sub-sampling frequency (in Hertz), default is None
%
% Returns
% -------   
% signal1_subsampled: (NbOfDimensions, NbOfSamples_sub) array
%     Subsampled first signal
% signal2_subsampled: (NbOfDimensions, NbOfSamples_sub) array
%     Subsampled second signal
% sub_frequency: int 
%     Subsampling frequency (in Hertz)
%     If the sub-sampling frequency was not specified by the user, this is the greatest common divisor of frequency1 and frequency2

if nargin < 5 || isempty(sub_frequency)
    % The frequency at which both signals will be subsampled is the greatest common divisor of the two frequencies
    sub_frequency_int32 = gcd(frequency1, frequency2);
    sub_frequency       = double(sub_frequency_int32);
end

% Subsample the signals
signal1_subsampled = subsample_one_signal(signal1, frequency1, sub_frequency);
signal2_subsampled = subsample_one_signal(signal2, frequency2, sub_frequency);

% Truncate the signals so that they have the same length
NbOfSamples_sub = min(size(signal1_subsampled, 2), size(signal2_subsampled, 2));
signal1_subsampled = signal1_subsampled(:, 1:NbOfSamples_sub);
signal2_subsampled = signal2_subsampled(:, 1:NbOfSamples_sub);
end
