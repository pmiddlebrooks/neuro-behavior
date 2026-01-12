function firingRate = calculate_firing_rate_from_spikes(spikeTimes, spikeClusters, neuronIDs, timeRange)
% CALCULATE_FIRING_RATE_FROM_SPIKES - Calculate firing rate from spike times
%
% Variables:
%   spikeTimes - Vector of all spike times (seconds)
%   spikeClusters - Vector of neuron IDs for each spike
%   neuronIDs - Which neurons to include (vector of IDs)
%   timeRange - [startTime, endTime] in seconds
%
% Returns:
%   firingRate - Total firing rate in spikes per second (sum across all neurons)

    % Filter spikes to time range and neurons
    validSpikes = spikeTimes >= timeRange(1) & ...
                  spikeTimes < timeRange(2) & ...
                  ismember(spikeClusters, neuronIDs);
    
    totalSpikes = sum(validSpikes);
    totalTime = timeRange(2) - timeRange(1);
    
    firingRate = totalSpikes / totalTime;
end
