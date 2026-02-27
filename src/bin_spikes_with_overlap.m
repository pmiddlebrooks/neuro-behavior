function [dataMat, binCenters] = bin_spikes_with_overlap(spikeTimes, spikeClusters, neuronIDs, timeRange, binSize, stepSize)
% BIN_SPIKES_WITH_OVERLAP - Bin spike data with optional overlapping windows
%
% Bins spikes into windows of length binSize, stepped by stepSize. Sum of
% spikes in each window is returned per neuron. If stepSize equals binSize
% or stepSize is omitted, uses non-overlapping bins (standard procedure).
%
% Variables:
%   spikeTimes    - Vector of all spike times (seconds)
%   spikeClusters - Vector of neuron IDs for each spike
%   neuronIDs     - Which neurons to include (vector of IDs, same order as output columns)
%   timeRange     - [startTime, endTime] in seconds
%   binSize       - Window/bin size in seconds
%   stepSize      - Step between successive bins in seconds (optional). If omitted or
%                   equals binSize, uses non-overlapping bins.
%
% Returns:
%   dataMat    - [nBins x nNeurons] matrix of spike counts per bin
%   binCenters - [nBins x 1] center time of each bin (seconds)
%
% Example:
%   % Overlapping: 0.1 s windows, step 0.02 s
%   [dataMat, t] = bin_spikes_with_overlap(spikeTimes, spikeClusters, neuronIDs, [0 60], 0.1, 0.02);
%   % Non-overlapping: same as bin_spikes
%   [dataMat, t] = bin_spikes_with_overlap(spikeTimes, spikeClusters, neuronIDs, [0 60], 0.1);

    if length(timeRange) ~= 2 || timeRange(2) <= timeRange(1)
        error('timeRange must be [startTime, endTime] with endTime > startTime');
    end
    if binSize <= 0
        error('binSize must be positive');
    end
    if isempty(neuronIDs)
        error('neuronIDs cannot be empty');
    end

    tStart = timeRange(1);
    tEnd = timeRange(2);
    useOverlap = nargin >= 6 && stepSize < binSize && stepSize > 0;

    if ~useOverlap
        % Non-overlapping: same as standard bin_spikes
        binEdges = tStart:binSize:tEnd;
        if binEdges(end) < tEnd
            binEdges(end + 1) = tEnd;
        end
        numBins = length(binEdges) - 1;
        dataMat = zeros(numBins, length(neuronIDs), 'single');
        validSpikes = spikeTimes >= tStart & spikeTimes < tEnd;
        spikeTimesF = spikeTimes(validSpikes);
        spikeClustersF = spikeClusters(validSpikes);

        for n = 1:length(neuronIDs)
            neuronID = neuronIDs(n);
            neuronSpikes = spikeTimesF(spikeClustersF == neuronID);
            if ~isempty(neuronSpikes)
                counts = histcounts(neuronSpikes, binEdges);
                nCounts = length(counts);
                if nCounts >= numBins
                    dataMat(:, n) = counts(1:numBins);
                else
                    dataMat(1:nCounts, n) = counts(:);
                end
            end
        end
        binCenters = tStart + binSize * ((1:numBins)' - 0.5);
        return;
    end

    % Overlapping bins: windows [binStart, binStart+binSize), binStart = tStart + (i-1)*stepSize
    lastBinStart = tEnd - binSize;
    if lastBinStart < tStart
        dataMat = zeros(0, length(neuronIDs), 'single');
        binCenters = zeros(0, 1);
        return;
    end
    binStarts = (tStart : stepSize : lastBinStart)';
    numBins = length(binStarts);
    dataMat = zeros(numBins, length(neuronIDs), 'single');
    validSpikes = spikeTimes >= tStart & spikeTimes < tEnd;
    spikeTimesF = spikeTimes(validSpikes);
    spikeClustersF = spikeClusters(validSpikes);

    for n = 1:length(neuronIDs)
        neuronID = neuronIDs(n);
        neuronSpikes = spikeTimesF(spikeClustersF == neuronID);
        for b = 1:numBins
            left = binStarts(b);
            right = left + binSize;
            dataMat(b, n) = sum(neuronSpikes >= left & neuronSpikes < right);
        end
    end

    binCenters = binStarts + binSize / 2;
end
