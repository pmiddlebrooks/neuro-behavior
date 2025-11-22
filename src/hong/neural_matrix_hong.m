function [dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix_hong(data, opts)
% NEURAL_MATRIX_HONG - Converts spike time data into a binned matrix representation for Hong dataset
%
% This function takes spike data from the Hong whisker detection task dataset
% and bins spikes into a matrix based on the specified options.
%
% INPUTS:
%   data - Structure containing the following fields:
%       .spikeTimes    - Vector of spike timestamps in seconds
%       .spikeClusters - Vector of neuron IDs (cluster IDs) corresponding to each spike
%       .spikeDepths   - Vector of spike depths for each spike
%       .ci            - Table containing cluster metadata with cluster_id and group columns
%       .T             - (Optional) Behavior table with startTime_oe (trial start times in seconds)
%   opts - Structure containing the following options:
%       .method          - Method of creating dataMat: 'standard', 'useOverlap', or 'gaussian'
%       .frameSize       - Bin size for final dataMat (seconds)
%       .shiftAlignFactor - Time shift applied to spike bins (default: 0)
%       .collectStart    - Start time for data collection (seconds, default: 0)
%       .collectEnd      - End time for data collection (seconds, or empty for all data)
%       .removeSome      - Flag to filter out neurons based on firing rate
%       .minFiringRate   - Minimum firing rate threshold (for removal if enabled)
%       .maxFiringRate   - Maximum firing rate threshold (for removal if enabled)
%       .firingRateCheckTime - Time window to check firing rates (seconds)
%       For method = 'useOverlap'
%       .windowSize      - Window size for summing spikes in overlapping binning
%       .windowAlign     - Window alignment ('center', 'left', or 'right', default: 'center')
%       For method = 'gaussian'
%       .gaussWidth      - if method is gaussian, width of the kernel in ms
%
% OUTPUTS:
%   dataMat     - Binned spike matrix (time bins x neurons)
%   idLabels    - List of neuron IDs (cluster IDs) included in dataMat
%   areaLabels  - List of brain area labels for included neurons ('S1' or 'SC')
%   rmvNeurons  - Logical vector indicating neurons removed based on firing rate criteria
%
% Variables:
%   spikeTimesSec - Spike times in seconds
%   uniqueClusters - Unique cluster IDs that are 'good' or 'mua'
%   clusterDepths - Mean depth for each cluster

% Check if required fields exist in data struct
if ~isfield(data, 'spikeTimes') || ~isfield(data, 'spikeClusters') || ~isfield(data, 'spikeDepths') || ~isfield(data, 'ci')
    error('Required fields (spikeTimes, spikeClusters, spikeDepths, ci) not found in data struct.');
end

% Extract data from struct
spikeTimesSec = data.spikeTimes;
spikeClusters = data.spikeClusters;
spikeDepths = data.spikeDepths;
cg = data.ci;

% Determine collection time window
if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
    firstSecond = 0;
else
    firstSecond = opts.collectStart;
end

if ~isfield(opts, 'collectEnd') || isempty(opts.collectEnd)
    % Use maximum trial end time if T is provided, otherwise use last spike time
    if isfield(data, 'T') && isfield(data.T, 'endTime_oe')
        lastSecond = max(data.T.endTime_oe);
    else
        % Estimate from last spike time
        lastSecond = max(spikeTimesSec);
    end
else
    lastSecond = firstSecond + opts.collectEnd;
end

% Filter spikes to collection window
spikeMask = spikeTimesSec >= firstSecond & spikeTimesSec < lastSecond;
spikeTimesSec = spikeTimesSec(spikeMask);
spikeClusters = spikeClusters(spikeMask);
spikeDepthsFiltered = spikeDepths(spikeMask);

% Get unique clusters that are 'good' or 'mua'
validGroups = {'good', 'mua'};
validClusterMask = ismember(cg.group, validGroups);
validClusterIds = cg.cluster_id(validClusterMask);

% Filter to only include spikes from valid clusters
clusterMask = ismember(spikeClusters, validClusterIds);
spikeTimesSec = spikeTimesSec(clusterMask);
spikeClusters = spikeClusters(clusterMask);
spikeDepthsFiltered = spikeDepthsFiltered(clusterMask);

% Get unique cluster IDs that have spikes in the collection window
uniqueClusters = unique(spikeClusters);
nClusters = length(uniqueClusters);

% Determine area for each cluster based on mean spike depth
% S1: depth >= 2000, SC: depth < 2000
clusterAreas = cell(nClusters, 1);
for i = 1:nClusters
    clusterId = uniqueClusters(i);
    clusterDepthMask = (spikeClusters == clusterId);
    meanDepth = mean(spikeDepthsFiltered(clusterDepthMask));
    
    if meanDepth >= 2000
        clusterAreas{i} = 'S1';
    else
        clusterAreas{i} = 'SC';
    end
end

% Define time edges and number of frames
if isfield(opts, 'method') && strcmp(opts.method, 'gaussian')
    numFrames = ceil((lastSecond - firstSecond) / opts.frameSize);
else
    baseStep = opts.frameSize;
    timeEdgesGlobal = firstSecond:baseStep:lastSecond;
    numFrames = max(0, length(timeEdgesGlobal) - 1);
end

% Preallocate data matrix
if numFrames > 60 * 60 / 0.001
    dataMat = int8(zeros(numFrames, nClusters));
else
    dataMat = zeros(numFrames, nClusters);
end

% Set up area labels
idLabels = uniqueClusters;
areaLabels = clusterAreas;

% Handle different binning methods
if isfield(opts, 'method') && strcmp(opts.method, 'useOverlap')
    warning('You are using overlapping bins, so dataMat is in firing rate units (sp/s)');
    windowSize = opts.windowSize;
    if ~isfield(opts, 'windowAlign')
        opts.windowAlign = 'center';
    end
end

if isfield(opts, 'method') && strcmp(opts.method, 'gaussian')
    totalTimeMs = (lastSecond - firstSecond) * 1000;
    
    % Define Gaussian kernel
    kernelRange = round(5 * opts.gaussWidth);
    timeVec = -kernelRange:kernelRange;
    gaussKernel = exp(-timeVec.^2 / (2 * opts.gaussWidth^2));
    gaussKernel = gaussKernel / sum(gaussKernel);
    
    % Create high-resolution firing rate matrix (ms resolution)
    spikeRateMs = zeros(totalTimeMs, nClusters);
end

% Loop through each cluster and bin spike counts
for i = 1:nClusters
    clusterId = uniqueClusters(i);
    iSpikeTime = spikeTimesSec(spikeClusters == clusterId);
    
    % Shift spike times relative to collection start
    iSpikeTime = iSpikeTime - firstSecond;
    
    if isfield(opts, 'method')
        switch opts.method
            case 'useOverlap'
                % Overlapping bin edges
                timeEdges = timeEdgesGlobal - firstSecond;
                dataMatTemp = zeros(length(timeEdges) - 1, 1);
                
                % Apply sliding window approach
                for j = 1:length(timeEdges)-1
                    switch opts.windowAlign
                        case 'center'
                            winStart = timeEdges(j) - windowSize / 2;
                            winEnd = timeEdges(j) + windowSize / 2;
                        case 'left'
                            winStart = timeEdges(j);
                            winEnd = timeEdges(j) + windowSize;
                        case 'right'
                            winStart = timeEdges(j) - windowSize;
                            winEnd = timeEdges(j);
                    end
                    
                    % Count spikes in window and convert to rate
                    dataMatTemp(j) = sum(iSpikeTime >= winStart & iSpikeTime < winEnd) ./ windowSize;
                end
                
                dataMat(:, i) = dataMatTemp;
                
            case 'gaussian'
                % Convert spike times to binary spike train
                spikeTrain = zeros(totalTimeMs, 1);
                spikeIndices = round(iSpikeTime * 1000);
                validIndices = spikeIndices >= 1 & spikeIndices <= totalTimeMs;
                spikeTrain(spikeIndices(validIndices)) = 1;
                
                % Convolve with Gaussian kernel
                smoothedRate = conv(spikeTrain, gaussKernel, 'same');
                spikeRateMs(:, i) = smoothedRate * 1000; % Convert to Hz
                
            case 'standard'
                % Standard non-overlapping binning
                timeEdges = timeEdgesGlobal - firstSecond;
                if isfield(opts, 'shiftAlignFactor') && opts.shiftAlignFactor ~= 0
                    timeEdges = timeEdges + opts.shiftAlignFactor;
                end
                [iSpikeCount, ~] = histcounts(iSpikeTime, timeEdges);
                if isfield(opts, 'shiftAlignFactor') && opts.shiftAlignFactor ~= 0
                    iSpikeCount = [0 iSpikeCount(1:end-1)];
                end
                dataMat(:, i) = iSpikeCount';
        end
    else
        % Default to standard binning
        timeEdges = timeEdgesGlobal - firstSecond;
        [iSpikeCount, ~] = histcounts(iSpikeTime, timeEdges);
        dataMat(:, i) = iSpikeCount';
    end
end

% Handle gaussian method final binning
if isfield(opts, 'method') && strcmp(opts.method, 'gaussian')
    for b = 1:numFrames
        realTimeStart = (b-1) * opts.frameSize * 1000;
        binStart = round(realTimeStart) + 1;
        binEnd = min(round(realTimeStart + opts.frameSize * 1000), totalTimeMs);
        if binEnd >= binStart
            dataMat(b, :) = mean(spikeRateMs(binStart:binEnd, :), 1);
        end
    end
end

% Remove neurons that do not meet firing rate criteria
rmvNeurons = false(nClusters, 1);
if isfield(opts, 'removeSome') && opts.removeSome
    if ~isfield(opts, 'firingRateCheckTime')
        checkTime = 5 * 60; % Default 5 minutes
    else
        checkTime = opts.firingRateCheckTime;
    end
    
    checkFrames = floor(checkTime / opts.frameSize);
    checkFrames = min(checkFrames, numFrames);
    
    if checkFrames > 0
        if ~isfield(opts, 'method') || strcmp(opts.method, 'standard')
            meanStart = sum(dataMat(1:checkFrames, :), 1) ./ checkTime;
            meanEnd = sum(dataMat(end-checkFrames+1:end, :), 1) ./ checkTime;
        else
            meanStart = mean(dataMat(1:checkFrames, :), 1);
            meanEnd = mean(dataMat(end-checkFrames+1:end, :), 1);
        end
        
        keepStart = meanStart >= opts.minFiringRate & meanStart <= opts.maxFiringRate;
        keepEnd = meanEnd >= opts.minFiringRate & meanEnd <= opts.maxFiringRate;
        
        % Get rid of units with crazy bursting (likely multi-units)
        if ~isfield(opts, 'method') || strcmp(opts.method, 'standard')
            tooCrazy = max(dataMat ./ opts.frameSize, [], 1) > 3000;
        else
            tooCrazy = max(dataMat, [], 1) > 1000;
        end
        
        rmvNeurons = ~(keepStart & keepEnd) | tooCrazy;
        fprintf('\nkeeping %d of %d neurons\n', sum(~rmvNeurons), length(rmvNeurons));
        
        dataMat(:, rmvNeurons) = [];
        idLabels(rmvNeurons) = [];
        areaLabels(rmvNeurons) = [];
    end
end

end

