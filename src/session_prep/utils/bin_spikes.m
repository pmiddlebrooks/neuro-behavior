function dataMat = bin_spikes(spikeTimes, spikeClusters, neuronIds, timeRange, binSize)
% BIN_SPIKES Bin spike trains into a count matrix for a time range
%
% Variables:
%   spikeTimes - Spike times (seconds)
%   spikeClusters - Neuron ID per spike
%   neuronIds - Neuron IDs to include (column order in output)
%   timeRange - [startTime, endTime] in seconds
%   binSize - Bin width in seconds
%
% Returns:
%   dataMat - [nBins x nNeurons] spike counts

    if numel(timeRange) ~= 2 || timeRange(2) <= timeRange(1)
        error('timeRange must be [startTime, endTime] with endTime > startTime');
    end

    if binSize <= 0
        error('binSize must be positive');
    end

    if isempty(neuronIds)
        error('neuronIds cannot be empty');
    end

    numBins = ceil((timeRange(2) - timeRange(1)) / binSize);
    dataMat = zeros(numBins, numel(neuronIds), 'single');

    validSpikes = spikeTimes >= timeRange(1) & spikeTimes < timeRange(2);
    spikeTimesFiltered = spikeTimes(validSpikes);
    spikeClustersFiltered = spikeClusters(validSpikes);

    binEdges = timeRange(1):binSize:timeRange(2);
    if binEdges(end) < timeRange(2)
        binEdges(end + 1) = timeRange(2);
    end

    for n = 1:numel(neuronIds)
        neuronId = neuronIds(n);
        neuronSpikes = spikeTimesFiltered(spikeClustersFiltered == neuronId);

        if isempty(neuronSpikes)
            continue;
        end

        spikeCounts = histcounts(neuronSpikes, binEdges);

        if numel(spikeCounts) == numBins
            dataMat(:, n) = spikeCounts(:);
        elseif numel(spikeCounts) < numBins
            dataMat(1:numel(spikeCounts), n) = spikeCounts(:);
        else
            dataMat(:, n) = spikeCounts(1:numBins)';
        end
    end
end
