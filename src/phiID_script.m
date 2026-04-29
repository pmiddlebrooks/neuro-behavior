%% Build PhiID lagged inputs from area PCA (spontaneous spiking)
% Variables:
%   - phiIDPath: absolute path to local PhiID toolbox
%   - sessionName: spontaneous session folder name
%   - binSize: spike-count bin size [s]
%   - analysisDurationSec: total analyzed duration from collectStart [s]
%   - explainVarThreshold: cumulative explained variance threshold for PCA
% Goal:
%   Load spontaneous spiking data, bin spikes, run PCA in selected areas,
%   retain enough PCs to explain at least explainVarThreshold variance, and
%   construct lagged PhiID inputs:
%       X1, X2 at time t
%       Y1, Y2 at time t+1 (one-bin future of X1 and X2)
% Output:
%   - X1, X2, Y1, Y2 matrices for PhiID
%   - areaPca struct with PCA details for each requested area

%% User settings
phiIDPath = 'E:\Projects\toolboxes\PhiID';
sessionName = 'ag112321_1';
collectStartSec = 0;
analysisDurationSec = 30 * 60;
binSize = 0.02;
saveOutput = true;
explainVarThreshold = 0.85;

% Area names for X1 and X2. Use M23 as current M1 label in this dataset.
areaNameX1 = 'M56';
areaNameX2 = 'DS';
areaNames = {areaNameX1, areaNameX2};

% Paths and dependencies
paths = get_paths;
addpath(genpath(phiIDPath));

%% Load spontaneous spike-time data
opts = neuro_behavior_options;
opts.collectStart = collectStartSec;
opts.collectEnd = collectStartSec + analysisDurationSec;
opts.removeSome = true; % start with all good+mua units, matching current pipeline
opts.minFiringRate = .5; % 0.7;

spikeData = load_spike_times('spontaneous', paths, sessionName, opts);

%% Bin spikes and run PCA in each selected area
timeRange = [opts.collectStart, opts.collectEnd];
numAreas = numel(areaNames);
areaPca = repmat(struct( ...
    'areaName', '', ...
    'neuronIDs', [], ...
    'binnedSpikes', [], ...
    'scores', [], ...
    'coeff', [], ...
    'explained', [], ...
    'cumExplained', [], ...
    'numComponentsUsed', []), numAreas, 1);

scoreByArea = cell(numAreas, 1);
numBinsByArea = zeros(numAreas, 1);
if explainVarThreshold <= 1
    explainVarThresholdPct = explainVarThreshold * 100;
else
    explainVarThresholdPct = explainVarThreshold;
end

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
    cumExplained = cumsum(explained);
    numComponentsUsed = find(cumExplained >= explainVarThresholdPct, 1, 'first');
    if isempty(numComponentsUsed)
        numComponentsUsed = size(score, 2);
    end

    scoreByArea{areaIdx} = score(:, 1:numComponentsUsed);
    numBinsByArea(areaIdx) = size(score, 1);
    areaPca(areaIdx).areaName = thisAreaName;
    areaPca(areaIdx).neuronIDs = neuronIDs;
    areaPca(areaIdx).binnedSpikes = spikeMat;
    areaPca(areaIdx).scores = score(:, 1:numComponentsUsed);
    areaPca(areaIdx).coeff = coeff(:, 1:numComponentsUsed);
    areaPca(areaIdx).explained = explained(:);
    areaPca(areaIdx).cumExplained = cumExplained(:);
    areaPca(areaIdx).numComponentsUsed = numComponentsUsed;

    fprintf('%s: nNeurons=%d, nBins=%d, PCs used=%d, cumulative explained=%.2f%%\n', ...
        thisAreaName, numel(neuronIDs), size(spikeMat, 1), ...
        numComponentsUsed, cumExplained(numComponentsUsed));
end

%% Build X1/X2 and one-bin-future Y1/Y2
if numel(unique(numBinsByArea)) ~= 1
    error('Area PCA score matrices have mismatched bin counts: %s', mat2str(numBinsByArea));
end

X1Full = scoreByArea{1};
X2Full = scoreByArea{2};
numComponentsPhiID = min(size(X1Full, 2), size(X2Full, 2));
if numComponentsPhiID < 1
    error('No PCA components available for PhiID inputs.');
end

% PhiIDFull requires all four inputs to have identical column count.
X1Full = X1Full(:, 1:numComponentsPhiID);
X2Full = X2Full(:, 1:numComponentsPhiID);

% Build lagged pairs in time-major format, then transpose to features x time
% for PhiIDFull input convention.
X1TimeMajor = X1Full(1:end-1, :);
X2TimeMajor = X2Full(1:end-1, :);
Y1TimeMajor = X1Full(2:end, :);
Y2TimeMajor = X2Full(2:end, :);

X1 = X1TimeMajor';
X2 = X2TimeMajor';
Y1 = Y1TimeMajor';
Y2 = Y2TimeMajor';

timeAxisSec = opts.collectStart + ((1:size(X1Full, 1))' - 0.5) * binSize;
timeAxisSec = timeAxisSec(1:end-1);

fprintf('\nConstructed lagged PhiID inputs:\n');
fprintf('Shared PCA dimensions used for PhiID: %d\n', numComponentsPhiID);
fprintf('X1 (%s, features x time): %d x %d\n', areaNameX1, size(X1, 1), size(X1, 2));
fprintf('X2 (%s, features x time): %d x %d\n', areaNameX2, size(X2, 1), size(X2, 2));
fprintf('Y1 (%s, +1 bin, features x time): %d x %d\n', areaNameX1, size(Y1, 1), size(Y1, 2));
fprintf('Y2 (%s, +1 bin, features x time): %d x %d\n', areaNameX2, size(Y2, 1), size(Y2, 2));

%% Run PhiID
phiOutput = PhiIDFull(X1, X2, Y1, Y2);

%% Save output (optional)
if saveOutput
    saveDir = fullfile(paths.spontaneousResultsPath, sessionName);
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    savePath = fullfile(saveDir, sprintf('phiID_inputs_lagged_%s_bin%.3f.mat', sessionName, binSize));
    save(savePath, 'X1', 'X2', 'Y1', 'Y2', 'phiOutput', 'areaPca', 'areaNameX1', ...
        'areaNameX2', 'explainVarThreshold', 'numComponentsPhiID', 'binSize', 'sessionName', ...
        'timeAxisSec', 'collectStartSec', 'analysisDurationSec', '-v7.3');
    fprintf('Saved lagged PhiID inputs to:\n%s\n', savePath);
end