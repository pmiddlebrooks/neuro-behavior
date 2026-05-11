%% Build PhiID lagged inputs from grouped-area PCA (spontaneous spiking)
% Variables:
%   - phiIDPath: absolute path to local PhiID toolbox (genpath adds root + all subdirs)
%   - jidtJarPath: full path to JIDT infodynamics.jar (required for PhiIDFull Java MI).
%     PhiID .m path is not enough — set this or place the jar under phiIDPath / toolboxes/jidt.
%     Release: https://github.com/jlizier/jidt/releases
%   - sessionName: spontaneous session folder name
%   - binSizesToTest: vector of spike-count bin sizes [s] (default 0.01:0.01:0.05)
%   - analysisDurationSec: total analyzed duration from collectStart [s]
%   - explainVarThreshold: cumulative explained variance threshold for PCA
%   - yLagBins: lag in bins (physical lag in seconds scales with binSize)
% Goal:
%   Load spontaneous spiking data, loop over binSizesToTest, bin spikes in
%   selected areas, combine areas into X1/X2 groups, run PCA per group, build
%   lagged PhiID inputs, run PhiIDFull, and aggregate six taxonomy classes
%   (storage, transfer, copy, erasure, downward, upward). Plot class sums vs
%   bin size; optionally save sweep struct and last-bin detailed variables.

%% User settings
paths = get_paths;
phiIDPath = fullfile(paths.homePath, 'toolboxes', 'PhiID');
% JIDT jar (Java Information Dynamics Toolkit). Example:
  jidtJarPath = fullfile(paths.homePath, 'toolboxes', 'infodynamics-dist-1.6.1', 'infodynamics.jar');
% jidtJarPath = fullfile(paths.homePath, 'toolboxes', 'infodynamics-dist-1.6.1');

sessionName = 'ag112321_1';
collectStartSec = 0;
analysisDurationSec = 30 * 60;
% Bin-size sweep for PhiID taxonomy (nats). Physical lag = yLagBins * binSize changes with binSize.
binSizesToTest = 0.01:0.01:0.05;
yLagBins = 1;
saveOutput = false;
explainVarThreshold = 0.85;
plotBinSizeSweep = true;

% Area groups for X1 and X2. Each group can contain one or more areas.
% Example:
%   areaNamesX1 = {'M56', 'M23'};
%   areaNamesX2 = {'DS'};
areaNamesX1 = {'M23','M56'};
areaNamesX2 = {'DS'};
areaNames = unique([areaNamesX1, areaNamesX2], 'stable');

% Paths and dependencies — PhiID root + every subfolder (e.g. private/) on MATLAB path
if ~exist(phiIDPath, 'dir')
    error('PhiID folder not found: %s\nSet phiIDPath to your local PhiID clone.', phiIDPath);
end
addpath(genpath(phiIDPath));
ensure_jidt_java_classpath(jidtJarPath, phiIDPath, paths);

scriptDir = fileparts(mfilename('fullpath'));
swPrepUtilsPath = fullfile(scriptDir, 'sliding_window_prep', 'utils');
if exist(swPrepUtilsPath, 'dir')
    addpath(swPrepUtilsPath);
end

%% Load spontaneous spike-time data
opts = neuro_behavior_options;
opts.collectStart = collectStartSec;
opts.collectEnd = collectStartSec + analysisDurationSec;
opts.removeSome = true; % start with all good+mua units, matching current pipeline
opts.minFiringRate = .5; % 0.7;

spikeData = load_spike_times('spontaneous', paths, sessionName, opts);

timeRange = [opts.collectStart, opts.collectEnd];
numAreas = numel(areaNames);
if explainVarThreshold <= 1
    explainVarThresholdPct = explainVarThreshold * 100;
else
    explainVarThresholdPct = explainVarThreshold;
end

if yLagBins < 1 || yLagBins ~= round(yLagBins)
    error('yLagBins must be a positive integer.');
end

binSizesToTest = binSizesToTest(:)';
nSweep = numel(binSizesToTest);
storageSweep = nan(nSweep, 1);
transferSweep = nan(nSweep, 1);
copySweep = nan(nSweep, 1);
erasureSweep = nan(nSweep, 1);
downwardSweep = nan(nSweep, 1);
upwardSweep = nan(nSweep, 1);
tdmiSweep = nan(nSweep, 1);
phiOutputLast = [];
groupPcaLast = [];
areaDataLast = [];
X1 = []; X2 = []; Y1 = []; Y2 = [];
timeAxisSec = [];
binSize = binSizesToTest(end);

for sweepIdx = 1:nSweep
    binSize = binSizesToTest(sweepIdx);
    fprintf('\n=== PhiID sweep %d/%d: binSize = %.3f s ===\n', sweepIdx, nSweep, binSize);

    spikeMatByArea = cell(numAreas, 1);
    numBinsByArea = zeros(numAreas, 1);
    areaData = repmat(struct('areaName', '', 'neuronIDs', [], 'binnedSpikes', []), numAreas, 1);

    for areaIdx = 1:numAreas
        thisAreaName = areaNames{areaIdx};
        isAreaNeuron = strcmp(spikeData.neuronAreas, thisAreaName);
        neuronIDs = spikeData.neuronIDs(isAreaNeuron);

        if isempty(neuronIDs)
            error('No neurons found in %s for session %s.', thisAreaName, sessionName);
        end

        spikeMat = bin_spikes(spikeData.spikeTimes, spikeData.spikeClusters, neuronIDs, timeRange, binSize);
        spikeMat = double(spikeMat);

        spikeMatByArea{areaIdx} = spikeMat;
        numBinsByArea(areaIdx) = size(spikeMat, 1);
        areaData(areaIdx).areaName = thisAreaName;
        areaData(areaIdx).neuronIDs = neuronIDs;
        areaData(areaIdx).binnedSpikes = spikeMat;

        fprintf('  %s: nNeurons=%d, nBins=%d\n', thisAreaName, numel(neuronIDs), size(spikeMat, 1));
    end

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

    if size(X1Full, 1) <= yLagBins
        warning('Skipping binSize=%.3f: only %d bins (need > yLagBins=%d).', binSize, size(X1Full, 1), yLagBins);
        continue;
    end

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

    fprintf('  X1 PCs=%d, X2 PCs=%d, lag=%d bins (%.4f s)\n', ...
        x1NumComponentsUsed, x2NumComponentsUsed, yLagBins, yLagBins * binSize);

    phiOutput = PhiIDFull(X1, X2, Y1, Y2);
    tax = phi_id_taxonomy_sums(phiOutput);

    storageSweep(sweepIdx) = tax.storage;
    transferSweep(sweepIdx) = tax.transfer;
    copySweep(sweepIdx) = tax.copy;
    erasureSweep(sweepIdx) = tax.erasure;
    downwardSweep(sweepIdx) = tax.downward;
    upwardSweep(sweepIdx) = tax.upward;
    tdmiSweep(sweepIdx) = tax.tdmi;

    phiOutputLast = phiOutput;
    groupPcaLast = groupPca;
    areaDataLast = areaData;
end

%% Print last successful PhiID taxonomy (same definitions as loop)
if ~isempty(phiOutputLast)
    taxLast = phi_id_taxonomy_sums(phiOutputLast);
    fprintf('\nPhiID taxonomy at last binSize=%.3f s (nats):\n', binSize);
    fprintf('  TDMI=%.6f storage=%.6f transfer=%.6f copy=%.6f erasure=%.6f downward=%.6f upward=%.6f\n', ...
        taxLast.tdmi, taxLast.storage, taxLast.transfer, taxLast.copy, taxLast.erasure, ...
        taxLast.downward, taxLast.upward);
end

%% Plot taxonomy vs binSize
if plotBinSizeSweep && nSweep > 0
    figSweep = figure('Color', 'w');
    hold on;
    plot(binSizesToTest, storageSweep, '-o', 'LineWidth', 1.5, 'DisplayName', 'storage');
    plot(binSizesToTest, transferSweep, '-s', 'LineWidth', 1.5, 'DisplayName', 'transfer');
    plot(binSizesToTest, copySweep, '-^', 'LineWidth', 1.5, 'DisplayName', 'copy');
    plot(binSizesToTest, erasureSweep, '-v', 'LineWidth', 1.5, 'DisplayName', 'erasure');
    plot(binSizesToTest, downwardSweep, '-d', 'LineWidth', 1.5, 'DisplayName', 'downward');
    plot(binSizesToTest, upwardSweep, '-*', 'LineWidth', 1.5, 'DisplayName', 'upward');
    xlabel('bin size (s)');
    ylabel('information (nats)');
    title(sprintf('PhiID taxonomy vs bin size | %s | yLag=%d bins', sessionName, yLagBins), 'Interpreter', 'none');
    legend('Location', 'eastoutside');
    grid on;
    hold off;
end

%% Save output (optional)
if saveOutput
    saveDir = fullfile(paths.spontaneousResultsPath, sessionName);
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    binSizeSweep = struct( ...
        'binSizesToTest', binSizesToTest, ...
        'storage', storageSweep, ...
        'transfer', transferSweep, ...
        'copy', copySweep, ...
        'erasure', erasureSweep, ...
        'downward', downwardSweep, ...
        'upward', upwardSweep, ...
        'tdmi', tdmiSweep);

    savePathSweep = fullfile(saveDir, sprintf('phiID_binSize_sweep_%s_yLag%d.mat', sessionName, yLagBins));
    save(savePathSweep, 'binSizeSweep', 'phiOutputLast', 'groupPcaLast', 'areaDataLast', ...
        'X1', 'X2', 'Y1', 'Y2', 'areaNamesX1', 'areaNamesX2', 'explainVarThreshold', ...
        'sessionName', 'timeAxisSec', 'collectStartSec', 'analysisDurationSec', 'yLagBins', ...
        'opts', '-v7.3');
    fprintf('Saved bin-size sweep to:\n%s\n', savePathSweep);
end

function ensure_jidt_java_classpath(jidtJarPath, phiIDPath, paths)
% ENSURE_JIDT_JAVA_CLASSPATH  Put JIDT infodynamics.jar on MATLAB's Java dynamic classpath.
%
% Variables:
%   jidtJarPath - explicit path to infodynamics.jar, or '' to search common locations
%   phiIDPath   - PhiID toolbox root (jar may live here or next to it)
%   paths       - from get_paths (homePath for toolboxes/jidt)
% Goal:
%   PhiIDFull uses javaObject('infodynamics.measures.continuous.gaussian....').
%   Those classes live in infodynamics.jar; addpath(genpath) does NOT load Java classes.

testClass = 'infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian';
try
    javaObject(testClass);
    return;
catch
end

candidates = {};
if nargin >= 1 && ~isempty(jidtJarPath)
    candidates{end+1} = char(jidtJarPath); %#ok<AGROW>
end
toolboxRoot = fileparts(phiIDPath);
candidates{end+1} = fullfile(phiIDPath, 'infodynamics.jar'); %#ok<AGROW>
candidates{end+1} = fullfile(phiIDPath, 'jars', 'infodynamics.jar'); %#ok<AGROW>
candidates{end+1} = fullfile(phiIDPath, 'jidt', 'infodynamics.jar'); %#ok<AGROW>
candidates{end+1} = fullfile(toolboxRoot, 'jidt', 'infodynamics.jar'); %#ok<AGROW>
candidates{end+1} = fullfile(toolboxRoot, 'JIDT', 'infodynamics.jar'); %#ok<AGROW>
if nargin >= 3 && isfield(paths, 'homePath')
    candidates{end+1} = fullfile(paths.homePath, 'toolboxes', 'jidt', 'infodynamics.jar'); %#ok<AGROW>
    candidates{end+1} = fullfile(paths.homePath, 'toolboxes', 'JIDT', 'infodynamics.jar'); %#ok<AGROW>
end

jarFound = '';
for iCand = 1:numel(candidates)
    cand = candidates{iCand};
    if isempty(cand)
        continue;
    end
    if exist(cand, 'file') == 2
        jarFound = cand;
        break;
    end
end

if isempty(jarFound)
    error(['PhiIDFull needs JIDT (infodynamics.jar) on the Java classpath — addpath is not enough.\n', ...
           '1) Download a release: https://github.com/jlizier/jidt/releases\n', ...
           '2) Copy infodynamics.jar to e.g. fullfile(paths.homePath,''toolboxes'',''jidt'')\n', ...
           '   OR set jidtJarPath in phiID_script.m to the full path of infodynamics.jar\n', ...
           'Searched from phiIDPath: %s'], phiIDPath);
end

dynList = javaclasspath('-dynamic');
alreadyThere = false;
for iDyn = 1:numel(dynList)
    if ~isempty(strfind(lower(dynList{iDyn}), 'infodynamics.jar')) %#ok<STREMP>
        alreadyThere = true;
        break;
    end
end
if ~alreadyThere
    javaaddpath(jarFound);
end

try
    javaObject(testClass);
catch ME
    error('After javaaddpath(''%s''), Java still cannot load %s:\n%s', jarFound, testClass, ME.message);
end
fprintf('JIDT on Java classpath: %s\n', jarFound);
end

function tax = phi_id_taxonomy_sums(phiOutput)
% PHI_ID_TAXONOMY_SUMS  Six-class PhiID aggregates + TDMI from PhiIDFull output (nats).
%
% Variables:
%   phiOutput - struct returned by PhiIDFull (atom fields rtr, xtx, ...).
% Goal:
%   Map atoms to Mediano et al. 2025 taxonomy sums used in phiID_script.

tax = struct();
tax.storage = phiOutput.rtr + phiOutput.xtx + phiOutput.yty + phiOutput.sts;
tax.transfer = phiOutput.xty + phiOutput.ytx;
tax.copy = phiOutput.xtr + phiOutput.ytr;
tax.erasure = phiOutput.rtx + phiOutput.rty;
tax.downward = phiOutput.stx + phiOutput.sty + phiOutput.str;
tax.upward = phiOutput.xts + phiOutput.yts + phiOutput.rts;
atomVals = struct2array(phiOutput);
tax.tdmi = sum(atomVals(:));
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