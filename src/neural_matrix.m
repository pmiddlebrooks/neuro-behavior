function [dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix(data, opts)
% NEURAL_MATRIX - Converts spike time data into a binned matrix representation
%
% This function takes in a data structure containing spike times, spike IDs,
% and behavior durations, and bins the spikes into a matrix based on the specified
% options. It allows both non-overlapping binning and an optional sliding window
% approach with overlapping bins.
%
% INPUTS:
%   data - Structure containing the following fields:
%       .spikeTimes    - Vector of spike timestamps
%       .spikeClusters - Vector of neuron IDs corresponding to each spike
%       .bhvDur        - Behavior duration (time span of recording)
%       .ci            - Table containing neuron metadata (id, area, etc.)
%   opts - Structure containing the following options:
%       .method          - Method of creating dataMat: 'standard', 'useOverlap', or 'gaussian'
%       .frameSize       - Bin size for final dataMat
%       .shiftAlignFactor - Time shift applied to spike bins (default: 0)
%       .useNeurons      - List of neuron IDs to include
%       .removeSome      - Flag to filter out neurons based on firing rate
%       .minFiringRate   - Minimum firing rate threshold (for removal if enabled)
%       .maxFiringRate   - Maximum firing rate threshold (for removal if enabled)
%       For method = 'useOverlap'
%       .windowSize      - Window size for summing spikes in overlapping binning (using stepSize = opts.frameSize)
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
numFrames = (opts.collectEnd - opts.collectStart) / opts.frameSize;

% Define neuron IDs
if ismember('id', data.ci.Properties.VariableNames)
    idLabels = data.ci.id(opts.useNeurons);
else
    idLabels = data.ci.cluster_id(opts.useNeurons);
end

% Preallocate data matrix
if numFrames > 45 * 60 / 0.001
    dataMat = single(zeros(numFrames, length(opts.useNeurons)));
else
    dataMat = zeros(numFrames, length(opts.useNeurons));
end

areaLabels = {};

if strcmp(opts.method, 'useOverlap')
    warning('You are using overlapping bins, so dataMat is in firing rate units (sp/s)')
    % Define window and step size
    windowSize = opts.windowSize; % Summing window size
    if ~isfield(opts, 'windowAlign')
        opts.windowAlign = 'center'; % Default alignment
    end
end
if strcmp(opts.method, 'gaussian')
    totalTimeMs = ceil(sum(data.bhvDur)*1000); % Ensure time resolution in ms

    % Define Gaussian kernel for convolution
    kernelRange = round(5 * opts.gaussWidth); % 5 SD range ensures near-zero tails
    timeVec = -kernelRange:kernelRange;
    gaussKernel = exp(-timeVec.^2 / (2 * opts.gaussWidth^2));
    gaussKernel = gaussKernel / sum(gaussKernel); % Normalize

    % Create high-resolution firing rate matrix (ms resolution)
    spikeRateMs = zeros(totalTimeMs, length(idLabels));
end

% Loop through each neuron and bin spike counts
for i = 1:length(idLabels)
    iSpikeTime = data.spikeTimes(data.spikeClusters == idLabels(i));

    switch opts.method
        case 'useOverlap'
            % Overlapping bin edges
            timeEdges = 0:opts.frameSize:numFrames * opts.frameSize;
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
            % Standard non-overlapping binning
            timeEdges = 0:opts.frameSize:numFrames * opts.frameSize;
            if opts.shiftAlignFactor ~= 0
                timeEdges = timeEdges + opts.shiftAlignFactor;
            end
            [iSpikeCount, ~] = histcounts(iSpikeTime, timeEdges);
            if opts.shiftAlignFactor ~= 0
                iSpikeCount = [0 iSpikeCount(1:end-1)];
            end
            dataMat(:, i) = iSpikeCount';
        case 'gaussian'
            % Convert spike times to binary spike train
            spikeTrain = zeros(totalTimeMs, 1);
            spikeTrain(round(iSpikeTime * 1000)) = 1;

            % Convolve with Gaussian kernel to get smoothed firing rate
            smoothedRate = conv(spikeTrain, gaussKernel, 'same'); % divide by 1000 to get smoothed rate
            spikeRateMs(:, i) = smoothedRate * opts.fsSpike; % Convert to Hz
            spikeRateMs(:, i) = smoothedRate * 1000; % Convert to Hz

    end

    
    % Track brain areas
    areaLabels = [areaLabels, data.ci.area(opts.useNeurons(i))];
end

    if strcmp(opts.method, 'gaussian')
         % Loop through neurons and generate spike rate vectors
           for b = 1:numFrames
                % binStart = (b-1) * binWidthMs + 1;
                % binEnd = min(binStart + binWidthMs - 1, totalTime);
                realTimeStart = (b-1) * opts.frameSize * 1000;
                binStart = round(realTimeStart) + 1;
                binEnd = min(round(realTimeStart + opts.frameSize * 1000), totalTimeMs);
                dataMat(b, :) = mean(spikeRateMs(binStart:binEnd, :), 1);
            end
    end


% Remove neurons that do not meet firing rate criteria
rmvNeurons = [];
if opts.removeSome
    checkTime = opts.firingRateCheckTime;
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
    rmvNeurons = ~(keepStart & keepEnd);
    fprintf('\nkeeping %d of %d neurons\n', sum(~rmvNeurons), length(rmvNeurons));
    dataMat(:, rmvNeurons) = [];
    idLabels(rmvNeurons) = [];
    areaLabels(rmvNeurons) = [];
end
end





