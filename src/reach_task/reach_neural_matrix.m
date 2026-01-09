function [dataMat, idLabels, areaLabels, rmvNeurons] = reach_neural_matrix(data, opts)
% NEURAL_MATRIX_MARK_DATA - Converts spike time data into a binned matrix representation
%
% This function takes in a data structure containing spike times, spike IDs,
% and behavior durations, and bins the spikes into a matrix based on the specified
% options. It allows both non-overlapping binning and an optional sliding window
% approach with overlapping bins.
%
% INPUTS:
%   data - Structure containing the following fields:
%       .idchan    - xxxxx
%       .CSV - xxxx
%   opts - Structure containing the following options:
%       .method          - Method of creating dataMat: 'standard', 'useOverlap', or 'gaussian'
%       .frameSize       - Bin size for final dataMat
%       .shiftAlignFactor - Time shift applied to spike bins (default: 0)
%       .useNeurons      - List of neuron IDs to include
%       .removeSome      - Flag to filter out neurons based on firing rate
%       .minFiringRate   - Minimum firing rate threshold (for removal if enabled)
%       .maxFiringRate   - Maximum firing rate threshold (for removal if enabled)
%       For method = 'useOverlap'
%       .windowSize      - Window size for summing spikes in overlapping binning (stepSize is same as frameSize
%       .windowAlign     - Window alignment ('center', 'left', or 'right', default: 'center')
%       For method = 'gaussian'
%       .gaussWidth      - if method is gaussian, width of the kernel in ms
%
% OUTPUTS:
%   dataMat     - Binned spike matrix (time bins x neurons)
%   idLabels    - List of neuron IDs included in dataMat
%   areaLabels  - List of brain area labels for included neurons
%   rmvNeurons  - Logical vector indicating neurons removed based on firing rate criteria
%


if opts.shiftAlignFactor ~= 0
    warning('neural_matrix.m: You shifted the spike bin time w.r.t. the behavior start time: make sure you wanted to do this.')
end

% Determine total duration in frames
if ~isempty(opts.collectStart)
firstSecond = opts.collectStart;
else
    firstSecond = 0;
end
if ~isempty(opts.collectEnd)
    lastSecond = firstSecond + opts.collectEnd;
else
    lastSecond = ceil(max(data.CSV(:,1)));
end

% Define time edges and number of frames based on method to avoid off-by-one
if isfield(opts, 'method') && strcmp(opts.method, 'gaussian')
    % Will compute numFrames later for gaussian
    numFrames = ceil((lastSecond-firstSecond) / opts.frameSize);
else
    % For 'standard' and 'useOverlap', base frames on explicit edges
    baseStep = opts.frameSize;
    timeEdgesGlobal = firstSecond:baseStep:lastSecond;
    numFrames = max(0, length(timeEdgesGlobal) - 1);
end

useNeurons = find(data.idchan(:,end) ~= 0 & ismember(data.idchan(:,4), [1 2])); % Keep everything that isn't corpus collosum
idLabels = data.idchan(useNeurons, 1);
brainAreas = data.idchan(useNeurons, end);
areaLabels = cell(size(brainAreas));
areaLabels(brainAreas == 1) = {'M23'};
areaLabels(brainAreas == 2) = {'M56'};
areaLabels(brainAreas == 3) = {'DS'};
areaLabels(brainAreas == 4) = {'VS'};

% Preallocate data matrix
if numFrames > 60 * 60 / 0.001
    dataMat = int8(zeros(numFrames, length(idLabels)));
else
    dataMat = zeros(numFrames, length(idLabels));
end

if strcmp(opts.method, 'useOverlap')
    warning('You are using overlapping bins, so dataMat is in firing rate units (sp/s)')
    % Define window and step size
    windowSize = opts.windowSize; % Summing window size
    stepSize = opts.frameSize; % Step size for shifting the window
    if ~isfield(opts, 'windowAlign')
        opts.windowAlign = 'center'; % Default alignment
    end
end
if strcmp(opts.method, 'gaussian')
    totalTimeMs = (lastSecond-firstSecond) * 1000; % Ensure time resolution in ms

    % Define Gaussian kernel for convolution
    kernelRange = round(5 * opts.gaussWidth); % 5 SD range ensures near-zero tails
    timeVec = -kernelRange:kernelRange;
    gaussKernel = exp(-timeVec.^2 / (2 * opts.gaussWidth^2));
    gaussKernel = gaussKernel / sum(gaussKernel); % Normalize

    % Create high-resolution firing rate matrix (ms resolution)
    spikeRateMs = zeros(totalTimeMs, length(idLabels));
end

% Loop through each neuron and bin spike counts or rates
for i = 1:length(idLabels)
    iSpikeTime = data.CSV(data.CSV(:,2) == idLabels(i), 1);

    switch opts.method
        case 'useOverlap'
            % Overlapping bin edges (reuse global edges for consistent sizing)
            timeEdges = timeEdgesGlobal; % firstSecond:stepSize:lastSecond
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

                % Count spikes in the current window
                % dataMatTemp(j) = sum(iSpikeTime >= winStart & iSpikeTime < winEnd);
                dataMatTemp(j) = sum(iSpikeTime >= winStart & iSpikeTime < winEnd) ./ windowSize;
            end

            % Store in dataMat
            dataMat(:, i) = dataMatTemp;
        case 'standard'
            % Standard non-overlapping binning (reuse global edges)
            timeEdges = timeEdgesGlobal;
            [iSpikeCount, ~] = histcounts(iSpikeTime, timeEdges);
            dataMat(:, i) = iSpikeCount';
        case 'gaussian'
            % Convert spike times to binary spike train
            error('Something is bad about gaussian in mark data version of this function')
            spikeTrain = zeros(totalTimeMs, 1);
            spikeTrain(round(iSpikeTime * 1000)) = 1;

            % Convolve with Gaussian kernel to get smoothed firing rate
            smoothedRate = conv(spikeTrain, gaussKernel, 'same'); % divide by 1000 to get smoothed rate
            spikeRateMs(:, i) = smoothedRate * 1000; % Convert to Hz

    end

end

    if strcmp(opts.method, 'gaussian')
         % Loop through neurons and generate spike rate vectors
           for b = 1:numFrames
                realTimeStart = (b-1) * opts.frameSize * 1000;
                binStart = round(realTimeStart) + 1;
                binEnd = min(round(realTimeStart + opts.frameSize * 1000), totalTimeMs);
                dataMat(b, :) = mean(spikeRateMs(binStart:binEnd, :), 1);
            end
    end


% Remove neurons that do not meet firing rate criteria
if opts.removeSome
    checkTime = 5 * 60; % Check 5min for now, convert to input variable later maybe
    checkTime = opts.firingRateCheckTime; % Check 5min for now, convert to input variable later maybe
    
    % Calculate duration of dataMat
    dataMatDuration = numFrames * opts.frameSize;
    
    % If firingRateCheckTime is greater than dataMat duration, use dataMat duration
    if checkTime > dataMatDuration
        checkTime = dataMatDuration;
    end
    
    checkFrames = floor(checkTime / opts.frameSize);
    if ~strcmp(opts.method, 'standard')
        meanStart = mean(dataMat(1:checkFrames, :), 1);
        meanEnd = mean(dataMat(end-checkFrames+1:end, :), 1);
    else
        meanStart = sum(dataMat(1:checkFrames, :), 1) ./ checkTime;
        meanEnd = sum(dataMat(end-checkFrames+1:end, :), 1) ./ checkTime;
    end
    keepStart = meanStart >= opts.minFiringRate & meanStart <= opts.maxFiringRate;
    keepEnd = meanEnd >= opts.minFiringRate & meanEnd <= opts.maxFiringRate;

    % Get rid of units that had crazy bursting during the recording (most
    % likely multi-units)
    if ~strcmp(opts.method, 'standard')
    tooCrazy = max(dataMat, [], 1) > 1000;
    else
    tooCrazy = max(dataMat ./ opts.frameSize, [], 1) > 3000;
    end
    rmvNeurons = ~(keepStart & keepEnd) | tooCrazy;
    fprintf('\nkeeping %d of %d neurons\n', sum(~rmvNeurons), length(rmvNeurons));
    dataMat(:, rmvNeurons) = [];
    idLabels(rmvNeurons) = [];
    areaLabels(rmvNeurons) = [];
end

end





