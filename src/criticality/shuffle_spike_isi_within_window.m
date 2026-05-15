function [surrTimes, surrClusters] = shuffle_spike_isi_within_window( ...
    spikeTimes, spikeClusters, neuronIds, timeRange)
% SHUFFLE_SPIKE_ISI_WITHIN_WINDOW ISI-shuffle surrogate spikes within a window
%
% Variables:
%   spikeTimes - All spike times (seconds)
%   spikeClusters - Neuron ID per spike
%   neuronIds - Neuron IDs to include
%   timeRange - [startTime, endTime] in seconds
%
% Goal:
%   Cambrainha et al. 2025 surrogate: shuffle inter-spike intervals independently
%   per unit within the window, preserving each unit's spike count.
%
% Returns:
%   surrTimes, surrClusters - Surrogate spike trains in the window

    validMask = spikeTimes >= timeRange(1) & spikeTimes < timeRange(2) ...
        & ismember(spikeClusters, neuronIds);
    surrTimes = spikeTimes(validMask);
    surrClusters = spikeClusters(validMask);

    for iUnit = 1:numel(neuronIds)
        unitId = neuronIds(iUnit);
        unitMask = surrClusters == unitId;
        unitSpikes = sort(surrTimes(unitMask));
        if numel(unitSpikes) < 2
            continue;
        end
        isiVec = diff(unitSpikes);
        isiVec = isiVec(randperm(numel(isiVec)));
        % n spikes -> n-1 ISIs; anchor first spike time, rebuild remainder
        shuffledSpikes = zeros(size(unitSpikes));
        shuffledSpikes(1) = unitSpikes(1);
        shuffledSpikes(2:end) = unitSpikes(1) + cumsum(isiVec);
        surrTimes(unitMask) = shuffledSpikes;
    end
end
