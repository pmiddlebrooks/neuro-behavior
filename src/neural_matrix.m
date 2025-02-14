% function [dataMat, idLabels, areaLabels, rmvNeurons] = neural_matrix(data, opts)
%
%
%     if opts.shiftAlignFactor ~= 0
%     warning('neural_matrix.m:  You shifted the spike bin time w.r.t. the behavior start time: make sure you wanted to do this.')
% end
%
% % checkTime = 5 * 60;
% durFrames = ceil(sum(data.bhvDur) / opts.frameSize);
%
% if ismember('id',data.ci.Properties.VariableNames)
%     idLabels = data.ci.id(opts.useNeurons);
% else
%     idLabels = data.ci.cluster_id(opts.useNeurons);
% end
%
% % Preallocate data matrix: If it's huge, make it a int matrix (reduce memory)
% if durFrames > 30*60/.001
%     dataMat = int8(zeros(durFrames, length(opts.useNeurons)));
% else
%     dataMat = zeros(durFrames, length(opts.useNeurons));
% end
%
% % dataMat = int8([]);
% areaLabels = {};
% % rmvNeurons = [];
% % Loop through each neuron and put spike numbers within each frame
% for i = 1 : length(idLabels)
%
%     % iSpikeTimes = data.spikeTimes(data.spikeClusters == opts.useNeurons(i));
%     iSpikeTime = data.spikeTimes(data.spikeClusters == idLabels(i));
%
%
%     % % Don't add neurons out of min-max firing rate
%     % meanStart = sum(iSpikeTime >= checkTime) ./ checkTime;
%     % meanEnd = sum(iSpikeTime > opts.collectFor - checkTime) ./ checkTime;
%     %
%     % keepStart = meanStart >= opts.minFiringRate & meanStart <= opts.maxFiringRate;
%     % keepEnd = meanEnd >= opts.minFiringRate & meanEnd <= opts.maxFiringRate;
%     % if keepStart && keepEnd
%
%     timeEdges = 0 : opts.frameSize : sum(data.bhvDur); % Create edges of bins from 0 to max time
% if opts.shiftAlignFactor ~= 0
%     timeEdges = timeEdges + opts.shiftAlignFactor;
% end
%
%     % Count the number of spikes in each bin
%     [iSpikeCount, ~] = histcounts(iSpikeTime, timeEdges);
% % If we shifted where the spike count bin is, account for that here
%     if opts.shiftAlignFactor ~= 0
%     iSpikeCount = [0 iSpikeCount(1:end-1)];
% end
%
%     dataMat(:, i) = iSpikeCount';
%     % dataMat = [dataMat, iSpikeCount'];
%
%     % Keep track which neurons (columns) are in which brain area
%     areaLabels = [areaLabels, data.ci.area(opts.useNeurons(i))];
%     % else
%     %     rmvNeurons = [rmvNeurons, i];
%     % end
% end
%
%
%
% % return
% %% Remove neurons that are out of min-max firing rates
% %
% % check mean rates at the beginning and end of data to ensure somewhat
% % stable isolation and neurons are within firirng rate range
% rmvNeurons = [];
% if opts.removeSome
%
%     checkTime = 5 * 60;
%     checkFrames = floor(checkTime / opts.frameSize);
%
%     meanStart = sum(dataMat(1:checkFrames, :), 1) ./ checkTime;
%     meanEnd = sum(dataMat(end-checkFrames+1:end, :), 1) ./ checkTime;
%
%     keepStart = meanStart >= opts.minFiringRate & meanStart <= opts.maxFiringRate;
%     keepEnd = meanEnd >= opts.minFiringRate & meanEnd <= opts.maxFiringRate;
%
%     rmvNeurons = ~(keepStart & keepEnd);
%     fprintf('\nkeeping %d of %d neurons\n', sum(~rmvNeurons), length(rmvNeurons))
%
%     % nrnIdx = find(rmvNeurons);
%     % for i = 1 : length(nrnIdx)
%     % plot(dataMat(:,nrnIdx(i)))
%     % [nrnIdx(i) meanStart(nrnIdx(i)) sum(dataMat(8000:10000,nrnIdx(i)) / checkTime) meanEnd(nrnIdx(i))]
%     % end
%
%     % meanRates = sum(dataMat, 1) / (size(dataMat, 1) * opts.frameSize);
%     % sum(meanRates >= opts.minFiringRate & meanRates <= opts.maxFiringRate)
%
%     dataMat(:,rmvNeurons) = [];
%     idLabels(rmvNeurons) = [];
%     areaLabels(rmvNeurons) = [];
%     % imagesc(dataMat')
%
% end
%


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
%       .frameSize       - Bin size for non-overlapping binning
%       .shiftAlignFactor - Time shift applied to spike bins (default: 0)
%       .useNeurons      - List of neuron IDs to include
%       .removeSome      - Flag to filter out neurons based on firing rate
%       .minFiringRate   - Minimum firing rate threshold (for removal if enabled)
%       .maxFiringRate   - Maximum firing rate threshold (for removal if enabled)
%       .useOverlappingBins - Flag to enable overlapping binning
%       .windowSize      - Window size for summing spikes in overlapping binning
%       .stepSize        - Step size for shifting the sliding window
%       .windowAlign     - Window alignment ('center', 'left', or 'right', default: 'center')
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
numFrames = ceil(sum(data.bhvDur) / opts.frameSize);

% Define neuron IDs
if ismember('id', data.ci.Properties.VariableNames)
    idLabels = data.ci.id(opts.useNeurons);
else
    idLabels = data.ci.cluster_id(opts.useNeurons);
end

% Preallocate data matrix
if numFrames > 30 * 60 / 0.001
    dataMat = int8(zeros(numFrames, length(opts.useNeurons)));
else
    dataMat = zeros(numFrames, length(opts.useNeurons));
end

areaLabels = {};

% Check if overlapping bins are used
useOverlap = isfield(opts, 'useOverlappingBins') && opts.useOverlappingBins;

if useOverlap
warning('You are using overlapping bins, so dataMat is in firing rate units (sp/s)')
% Define window and step size
    windowSize = opts.windowSize; % Summing window size
    stepSize = opts.stepSize; % Step size for shifting the window
    if ~isfield(opts, 'windowAlign')
        opts.windowAlign = 'center'; % Default alignment
    end
end

% Loop through each neuron and bin spike counts
for i = 1:length(idLabels)
    iSpikeTime = data.spikeTimes(data.spikeClusters == idLabels(i));

    if useOverlap
        % Overlapping bin edges
        timeEdges = 0:stepSize:sum(data.bhvDur); % Define steps
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
    else
        % Standard non-overlapping binning
        timeEdges = 0:opts.frameSize:sum(data.bhvDur);
        if opts.shiftAlignFactor ~= 0
            timeEdges = timeEdges + opts.shiftAlignFactor;
        end
        [iSpikeCount, ~] = histcounts(iSpikeTime, timeEdges);
        if opts.shiftAlignFactor ~= 0
            iSpikeCount = [0 iSpikeCount(1:end-1)];
        end
        dataMat(:, i) = iSpikeCount';
    end

    % Track brain areas
    areaLabels = [areaLabels, data.ci.area(opts.useNeurons(i))];
end

% Remove neurons that do not meet firing rate criteria
rmvNeurons = [];
if opts.removeSome
    checkTime = 5 * 60;
    checkFrames = floor(checkTime / opts.frameSize);
    if useOverlap
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





