function signal_subsampled = subsample_one_signal(signal, signal_frequency, sub_frequency)
% Subsample a signal at a given Frequency
%
% Parameters
% ----------
% signal: (NbOfDimensions, NbOfSamples) array
%     Signal to subsample
% signal_frequency: int 
%     Sampling frequency (in Hertz) of the signal
% sub_frequency: int 
%     Desired sub-sampling frequency (in Hertz)
%
% Returns
% -------   
% signal_subsampled: (NbOfDimensions, NbOfSamples_sub) array
%     Subsampled signal


% The sub-sampling frequency must be a divisor of the signal frequency:
if mod(signal_frequency, sub_frequency) ~= 0
    error('The sub-sampling frequency is not a divisor of the signal frequency. This would lead to synchronization issues when sub-sampling.');
else
    if signal_frequency == sub_frequency
        signal_subsampled = signal;
    else
        bin_size = floor(signal_frequency / sub_frequency); % the signal will be averaged over bins of this size
        [NbOfDimensions, NbOfSamples] = size(signal);
        NbOfSamples_sub = floor(NbOfSamples / bin_size);
        signal_truncated = signal(:, 1:bin_size*NbOfSamples_sub);
        signal_subsampled = zeros(NbOfDimensions, NbOfSamples_sub);
        for dim = 1:NbOfDimensions
            signal_reshape = reshape(signal_truncated(dim, :), bin_size, NbOfSamples_sub);
            signal_subsampled(dim, :) = mean(signal_reshape, 1);
        end
    end
end

end 
