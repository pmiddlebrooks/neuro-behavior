%% Build PhiID lagged inputs from grouped-area PCA (spontaneous spiking)
% Variables:
%   - phiIDPath: absolute path to local PhiID toolbox
%   - sessionName: spontaneous session folder name
%   - binSize: spike-count bin size [s]
%   - analysisDurationSec: total analyzed duration from collectStart [s]
%   - explainVarThreshold: cumulative explained variance threshold for PCA
% Goal:
%   Load spontaneous spiking data, bin spikes in selected areas, combine
%   areas into X1/X2 groups, run PCA per group, retain enough PCs to explain
%   at least explainVarThreshold variance, and
%   construct lagged PhiID inputs:
%       X1, X2 at time t
%       Y1, Y2 at time t+yLagBins (future of X1 and X2)
% Output:
%   - X1, X2, Y1, Y2 matrices for PhiID
%   - groupPca struct with PCA details for X1 and X2 groups

%% User settings
phiIDPath = 'E:\Projects\toolboxes\PhiID';
phiIDPath = 'Users/pmiddlebrooks/Projects/toolboxes/PhiID';
sessionName = 'ag112321_1';
collectStartSec = 0;
analysisDurationSec = 30 * 60;
binSize = 0.025;
yLagBins = 1;
saveOutput = true;
explainVarThreshold = 0.85;

% Area groups for X1 and X2. Each group can contain one or more areas.
% Example:
%   areaNamesX1 = {'M56', 'M23'};
%   areaNamesX2 = {'DS'};
areaNamesX1 = {'M23','M56'};
areaNamesX2 = {'DS'};
areaNames = unique([areaNamesX1, areaNamesX2], 'stable');

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

%% Bin spikes in each selected area
timeRange = [opts.collectStart, opts.collectEnd];
numAreas = numel(areaNames);
areaData = repmat(struct( ...
    'areaName', '', ...
    'neuronIDs', [], ...
    'binnedSpikes', []), numAreas, 1);

spikeMatByArea = cell(numAreas, 1);
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

    spikeMatByArea{areaIdx} = spikeMat;
    numBinsByArea(areaIdx) = size(spikeMat, 1);
    areaData(areaIdx).areaName = thisAreaName;
    areaData(areaIdx).neuronIDs = neuronIDs;
    areaData(areaIdx).binnedSpikes = spikeMat;

    fprintf('%s: nNeurons=%d, nBins=%d\n', ...
        thisAreaName, numel(neuronIDs), size(spikeMat, 1));
end

%% Build X1/X2 and lagged-future Y1/Y2
if numel(unique(numBinsByArea)) ~= 1
    error('Area PCA score matrices have mismatched bin counts: %s', mat2str(numBinsByArea));
end

X1SpikeMat = combine_area_spikes(areaNamesX1, areaNames, spikeMatByArea, 'X1');
X2SpikeMat = combine_area_spikes(areaNamesX2, areaNames, spikeMatByArea, 'X2');

[X1Full, x1Coeff, x1Explained, x1CumExplained, x1NumComponentsUsed] = ...
    run_group_pca(X1SpikeMat, explainVarThresholdPct);
[X2Full, x2Coeff, x2Explained, x2CumExplained, x2NumComponentsUsed] = ...
    run_group_pca(X2SpikeMat, explainVarThresholdPct);

groupPca = struct();
groupPca.X1 = struct('areaNames', {areaNamesX1}, 'coeff', x1Coeff, ...
    'explained', x1Explained, 'cumExplained', x1CumExplained, ...
    'numComponentsUsed', x1NumComponentsUsed);
groupPca.X2 = struct('areaNames', {areaNamesX2}, 'coeff', x2Coeff, ...
    'explained', x2Explained, 'cumExplained', x2CumExplained, ...
    'numComponentsUsed', x2NumComponentsUsed);

if yLagBins < 1 || yLagBins ~= round(yLagBins)
    error('yLagBins must be a positive integer.');
end
if size(X1Full, 1) <= yLagBins
    error('Not enough time bins (%d) for yLagBins=%d.', size(X1Full, 1), yLagBins);
end

% Build lagged pairs in time-major format, then transpose to features x time
% for PhiIDFull input convention.
X1TimeMajor = X1Full(1:end-yLagBins, :);
X2TimeMajor = X2Full(1:end-yLagBins, :);
Y1TimeMajor = X1Full(1+yLagBins:end, :);
Y2TimeMajor = X2Full(1+yLagBins:end, :);

X1 = X1TimeMajor';
X2 = X2TimeMajor';
Y1 = Y1TimeMajor';
Y2 = Y2TimeMajor';

timeAxisSec = opts.collectStart + ((1:size(X1Full, 1))' - 0.5) * binSize;
timeAxisSec = timeAxisSec(1:end-yLagBins);

fprintf('\nConstructed lagged PhiID inputs:\n');
fprintf('X1 areas: %s\n', strjoin(areaNamesX1, ', '));
fprintf('X2 areas: %s\n', strjoin(areaNamesX2, ', '));
fprintf('X1 (features x time): %d x %d\n', size(X1, 1), size(X1, 2));
fprintf('X2 (features x time): %d x %d\n', size(X2, 1), size(X2, 2));
fprintf('X1 PCs used: %d (%.2f%% explained)\n', x1NumComponentsUsed, x1CumExplained(x1NumComponentsUsed));
fprintf('X2 PCs used: %d (%.2f%% explained)\n', x2NumComponentsUsed, x2CumExplained(x2NumComponentsUsed));
fprintf('Y lag: %d bins (%.4f s)\n', yLagBins, yLagBins * binSize);
fprintf('Y1 (+%d bins, features x time): %d x %d\n', yLagBins, size(Y1, 1), size(Y1, 2));
fprintf('Y2 (+%d bins, features x time): %d x %d\n', yLagBins, size(Y2, 1), size(Y2, 2));

%% Run PhiID
phiOutput = PhiIDFull(X1, X2, Y1, Y2)

%% Print PhiID aggregate summaries
atomNames = fieldnames(phiOutput);
atomValues = struct2array(phiOutput);
tdmiSum = sum(atomValues);

% Directional transfer entropy aggregates (README convention for X1->Y2).
teX1ToX2 = phiOutput.xtr + phiOutput.str + phiOutput.xty + phiOutput.sty;
teX2ToX1 = phiOutput.ytr + phiOutput.str + phiOutput.ytx + phiOutput.stx;

% AIS-like aggregates for each side (analog of I(X_t^i ; X_{t+1}^i)).
aisLikeX1 = phiOutput.rtr + phiOutput.rtx + phiOutput.xtr + phiOutput.xtx;
aisLikeX2 = phiOutput.rtr + phiOutput.rty + phiOutput.ytr + phiOutput.yty;

% Six-class taxonomy aggregates (Mediano et al. 2025).
storageSum = phiOutput.rtr + phiOutput.xtx + phiOutput.yty + phiOutput.sts;
transferSum = phiOutput.xty + phiOutput.ytx;
copySum = phiOutput.xtr + phiOutput.ytr;
erasureSum = phiOutput.rtx + phiOutput.rty;
downwardSum = phiOutput.stx + phiOutput.sty + phiOutput.str;
upwardSum = phiOutput.xts + phiOutput.yts + phiOutput.rts;

fprintf('\nPhiID aggregate summary (nats):\n');
fprintf('TDMI (sum of all atoms): %.6f\n', tdmiSum);
fprintf('TE X1->X2: %.6f\n', teX1ToX2);
fprintf('TE X2->X1: %.6f\n', teX2ToX1);
fprintf('AIS-like X1: %.6f\n', aisLikeX1);
fprintf('AIS-like X2: %.6f\n', aisLikeX2);
fprintf('Class sums: storage=%.6f, transfer=%.6f, copy=%.6f, erasure=%.6f, downward=%.6f, upward=%.6f\n', ...
    storageSum, transferSum, copySum, erasureSum, downwardSum, upwardSum);
fprintf('Atom-wise check (%d atoms): %s\n', numel(atomNames), strjoin(atomNames', ', '));

%% Save output (optional)
if saveOutput
    saveDir = fullfile(paths.spontaneousResultsPath, sessionName);
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    savePath = fullfile(saveDir, sprintf('phiID_inputs_lagged_%s_bin%.3f.mat', sessionName, binSize));
    save(savePath, 'X1', 'X2', 'Y1', 'Y2', 'phiOutput', 'areaData', 'groupPca', 'areaNamesX1', ...
        'areaNamesX2', 'explainVarThreshold', 'binSize', 'sessionName', ...
        'timeAxisSec', 'collectStartSec', 'analysisDurationSec', 'yLagBins', '-v7.3');
    fprintf('Saved lagged PhiID inputs to:\n%s\n', savePath);
end

function combinedSpikeMat = combine_area_spikes(areaNamesToCombine, availableAreaNames, spikeMatByArea, groupLabel)
% Variables:
%   - areaNamesToCombine: cell array of area names for one PhiID group
%   - availableAreaNames: all area names with computed binned spikes
%   - spikeMatByArea: cell array of binned spike matrices (time x neurons)
%   - groupLabel: string label for error messages ('X1' or 'X2')
% Goal:
%   Concatenate binned spike matrices across requested areas to build one
%   PhiID group before PCA.

if isempty(areaNamesToCombine)
    error('%s area list is empty.', groupLabel);
end

combinedSpikeMat = [];
for areaIdx = 1:numel(areaNamesToCombine)
    thisAreaName = areaNamesToCombine{areaIdx};
    matchIdx = find(strcmp(availableAreaNames, thisAreaName), 1, 'first');
    if isempty(matchIdx)
        error('Area %s was requested for %s but was not found.', thisAreaName, groupLabel);
    end
    combinedSpikeMat = [combinedSpikeMat, spikeMatByArea{matchIdx}]; %#ok<AGROW>
end

if isempty(combinedSpikeMat)
    error('No spike features available for %s.', groupLabel);
end
end

function [groupScore, groupCoeff, explained, cumExplained, numComponentsUsed] = ...
    run_group_pca(spikeMat, explainVarThresholdPct)
% Variables:
%   - spikeMat: binned spike matrix (time x neurons) for one group
%   - explainVarThresholdPct: target cumulative explained variance [%]
% Goal:
%   Run PCA on a grouped spike matrix and keep PCs up to threshold.

spikeMatZ = zscore(spikeMat, 0, 1);
spikeMatZ(:, any(~isfinite(spikeMatZ), 1)) = 0;
[groupCoeff, score, ~, ~, explained] = pca(spikeMatZ, 'Centered', true);
cumExplained = cumsum(explained);
numComponentsUsed = find(cumExplained >= explainVarThresholdPct, 1, 'first');
if isempty(numComponentsUsed)
    numComponentsUsed = size(score, 2);
end
groupScore = score(:, 1:numComponentsUsed);
end