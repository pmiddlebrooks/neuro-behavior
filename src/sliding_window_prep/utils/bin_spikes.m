function dataMat = bin_spikes(spikeTimes, spikeClusters, neuronIDs, timeRange, binSize)
% BIN_SPIKES - Bin spike data on-demand from spike times and IDs
%
% This function bins spike data at a specified bin size without requiring
% a pre-computed dataMat. This allows for memory-efficient, on-demand binning
% at different bin sizes per area.
%
% Variables:
%   spikeTimes - Vector of all spike times (seconds)
%   spikeClusters - Vector of neuron IDs for each spike
%   neuronIDs - Which neurons to include (vector of IDs, same order as output columns)
%   timeRange - [startTime, endTime] in seconds
%   binSize - Bin size in seconds
%
% Returns:
%   dataMat - [nBins x nNeurons] matrix of spike counts
%
% Example:
%   % Bin spikes for a specific area
%   aID = dataStruct.idMatIdx{1};  % M23 neuron indices
%   neuronIDs = dataStruct.idLabel{1};  % Actual neuron IDs
%   timeRange = [0, 3600];  % First hour
%   binSize = 0.02;  % 20ms bins
%   dataMat = bin_spikes(spikeData.spikeTimes, spikeData.spikeClusters, ...
%                        neuronIDs, timeRange, binSize);

    % Validate inputs
    if length(timeRange) ~= 2 || timeRange(2) <= timeRange(1)
        error('timeRange must be [startTime, endTime] with endTime > startTime');
    end
    
    if binSize <= 0
        error('binSize must be positive');
    end
    
    if isempty(neuronIDs)
        error('neuronIDs cannot be empty');
    end
    
    % Calculate number of bins
    numBins = ceil((timeRange(2) - timeRange(1)) / binSize);
    
    % Initialize output matrix in single precision for memory efficiency
    dataMat = zeros(numBins, length(neuronIDs), 'single');
    
    % Filter spikes to time range
    validSpikes = spikeTimes >= timeRange(1) & spikeTimes < timeRange(2);
    spikeTimesFiltered = spikeTimes(validSpikes);
    spikeClustersFiltered = spikeClusters(validSpikes);
    
    % Create bin edges
    binEdges = timeRange(1):binSize:timeRange(2);
    % Ensure last edge is at least at endTime
    if binEdges(end) < timeRange(2)
        binEdges(end+1) = timeRange(2);
    end
    
    % Bin spikes for each neuron
    for n = 1:length(neuronIDs)
        neuronID = neuronIDs(n);
        
        % Get spikes for this neuron in time range
        neuronSpikes = spikeTimesFiltered(spikeClustersFiltered == neuronID);
        
        if ~isempty(neuronSpikes)
            % Bin the spikes
            spikeCounts = histcounts(neuronSpikes, binEdges);
            
            % Ensure we have the right number of bins (handle edge case)
            if length(spikeCounts) == numBins
                dataMat(:, n) = spikeCounts(:);
            elseif length(spikeCounts) < numBins
                % Pad with zeros if needed
                dataMat(1:length(spikeCounts), n) = spikeCounts(:);
            else
                % Truncate if needed (shouldn't happen, but be safe)
                dataMat(:, n) = spikeCounts(1:numBins)';
            end
        end
    end
end
