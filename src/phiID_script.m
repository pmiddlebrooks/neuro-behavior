%% Build 4-area PCA inputs for PhiID (spontaneous spiking)
% Variables:
%   - phiIDPath: absolute path to local PhiID toolbox
%   - sessionName: spontaneous session folder name
%   - binSize: spike-count bin size [s]
%   - analysisDurationSec: total analyzed duration from collectStart [s]
% Goal:
%   Load 30 minutes of spontaneous spiking data, bin spikes at 20 ms, run
%   PCA independently within each area (M23, M56, DS, VS), and create a
%   4-column matrix of first principal components for PhiID input.
% Output:
%   - phiInputMat: [nBins x 4] matrix with columns [M23, M56, DS, VS]
%   - areaPca: struct array with PCA details for each area

%% User settings
phiIDPath = '/Users/paulmiddlebrooks/Projects/toolboxes/PhiID';
sessionName = 'ag112321_1';
collectStartSec = 0;
analysisDurationSec = 30 * 60;
binSize = 0.02;
saveOutput = true;

% Keep area order fixed for downstream PhiID use.
areaNames = {'M23', 'M56', 'DS', 'VS'};

%% Paths and dependencies
paths = get_paths;
addpath(genpath(phiIDPath));

%% Load spontaneous spike-time data
opts = neuro_behavior_options;
opts.collectStart = collectStartSec;
opts.collectEnd = collectStartSec + analysisDurationSec;
opts.removeSome = false; % start with all good+mua units, matching current pipeline

spikeData = load_spike_times('spontaneous', paths, sessionName, opts);

%% Bin spikes and run PCA in each area
timeRange = [opts.collectStart, opts.collectEnd];
numAreas = numel(areaNames);
areaPca = repmat(struct( ...
    'areaName', '', ...
    'neuronIDs', [], ...
    'binnedSpikes', [], ...
    'firstPc', [], ...
    'explainedVariance', [], ...
    'coeff', []), numAreas, 1);

firstPcByArea = cell(numAreas, 1);

for areaIdx = 1:numAreas
    thisAreaName = areaNames{areaIdx};
    isAreaNeuron = strcmp(spikeData.neuronAreas, thisAreaName);
    neuronIDs = spikeData.neuronIDs(isAreaNeuron);

    if isempty(neuronIDs)
        error('No neurons found in %s for session %s.', thisAreaName, sessionName);
    end

    spikeMat = bin_spikes(spikeData.spikeTimes, spikeData.spikeClusters, neuronIDs, timeRange, binSize);
    spikeMat = double(spikeMat);

    % Z-score each neuron so PCA reflects shared temporal structure rather
    % than only absolute firing-rate magnitude differences.
    spikeMatZ = zscore(spikeMat, 0, 1);
    spikeMatZ(:, any(~isfinite(spikeMatZ), 1)) = 0;

    [coeff, score, ~, ~, explained] = pca(spikeMatZ, 'Centered', true);
    firstPc = score(:, 1);

    firstPcByArea{areaIdx} = firstPc;
    areaPca(areaIdx).areaName = thisAreaName;
    areaPca(areaIdx).neuronIDs = neuronIDs;
    areaPca(areaIdx).binnedSpikes = spikeMat;
    areaPca(areaIdx).firstPc = firstPc;
    areaPca(areaIdx).explainedVariance = explained(1);
    areaPca(areaIdx).coeff = coeff(:, 1);

    fprintf('%s: nNeurons=%d, nBins=%d, PC1 explained=%.2f%%\n', ...
        thisAreaName, numel(neuronIDs), size(spikeMat, 1), explained(1));
end

%% Build final 4-column matrix for PhiID
numBinsEachArea = cellfun(@numel, firstPcByArea);
if numel(unique(numBinsEachArea)) ~= 1
    error('Area PC vectors have mismatched lengths: %s', mat2str(numBinsEachArea));
end

phiInputMat = [firstPcByArea{1}, firstPcByArea{2}, firstPcByArea{3}, firstPcByArea{4}];
timeAxisSec = opts.collectStart + ((1:size(phiInputMat, 1))' - 0.5) * binSize;

fprintf('\nphiInputMat created: %d bins x %d areas\n', size(phiInputMat, 1), size(phiInputMat, 2));
fprintf('Columns: %s\n', strjoin(areaNames, ', '));

%% Save output (optional)
if saveOutput
    saveDir = fullfile(paths.spontaneousResultsPath, sessionName);
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    savePath = fullfile(saveDir, sprintf('phiID_inputs_%s_bin%.3f_30min.mat', sessionName, binSize));
    save(savePath, 'phiInputMat', 'areaPca', 'areaNames', 'binSize', ...
        'sessionName', 'timeAxisSec', 'collectStartSec', 'analysisDurationSec', '-v7.3');
    fprintf('Saved PhiID inputs to:\n%s\n', savePath);
end
