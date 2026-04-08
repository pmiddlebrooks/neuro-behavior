%%
% criticality_spontaneous_clusters
% Computes d2 criticality for spontaneous behavior clusters defined like
% spontaneous_behavior_clusters.m (same rules / outputs).
%
% Workflow:
%   1) Run spontaneous_behavior_clusters.m for your session so the workspace
%      contains clusterResults, labelsUsed, timeAxis, minDur, sessionName
%      (and optionally includeBhv for labels in plots).
%   2) Run this script (or set those variables then run).
%
% Variables:
%   useNeuralWindowClusterSpan - true: d2 window = cluster time span [tStart,tEnd]
%                                false: fixed duration = minDur centered on cluster midpoint
%   windowBuffer - keep neural window inside [timeRange] by this margin (s)
%   plotAreaIdx - [] = all areasToTest; else one or more area indices (e.g. [1 3])
%   pOrder, critType, normalizeD2, nShuffles - d2 settings
%   useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple - optional
%   ethogramMinDur - x-axis half-width uses minDur from clusters by default; override here if needed
%   maxClustersEthogram - max rows per group in ethogram (random subsample if more)
%
% Goal:
%   One d2 value per cluster per area; compare distributions across cluster groups;
%   color ethogram of behavior labels aligned on cluster center over [-minDur/2, +minDur/2].

%% ============================= Preconditions =============================
fprintf('\n=== criticality_spontaneous_clusters ===\n');

requiredVars = {'sessionName', 'clusterResults', 'labelsUsed', 'timeAxis', 'minDur'};
missingList = {};
for v = 1:numel(requiredVars)
    vn = requiredVars{v};
    if exist(vn, 'var') ~= 1
        missingList{end + 1} = vn; %#ok<AGROW>
    end
end
if ~isempty(missingList)
    error('Run spontaneous_behavior_clusters.m first (missing: %s).', strjoin(missingList, ', '));
end
if isempty(clusterResults)
    error('clusterResults is empty. Re-run spontaneous_behavior_clusters.m.');
end

%% ============================= Configuration =============================
useNeuralWindowClusterSpan = true;  % false => fixed neural window length = minDur, cluster-centered
windowBuffer = 0.5;

useOptimalBinWindowFunction = false;
binSizeManual = 0.025;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;

pOrder = 10;
critType = 2;
normalizeD2 = false;
nShuffles = 8;

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.2;

pcaFlag = 0;

plotAreaIdx = [];  % [] = all areasToTest

ethogramMinDur = [];  % [] => use minDur for ethogram x-limits
maxClustersEthogram = 80;
ethogramFigureId = 32060;
d2CompareFigureId = 32061;

includeM2356 = false;

makePlots = true;

%% ============================= Neural data load =============================
sessionType = 'spontaneous';
opts = neuro_behavior_options;
opts.frameSize = 0.001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;
if ~isfield(opts, 'fsBhv') || isempty(opts.fsBhv)
    opts.fsBhv = 30;
end

dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

areas = dataStruct.areas;
numAreas = numel(areas);
if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

if isempty(plotAreaIdx)
    areasToPlot = areasToTest(:)';
else
    plotAreaIdx = plotAreaIdx(:)';
    if ~all(ismember(plotAreaIdx, areasToTest))
        error('plotAreaIdx contains indices not in areasToTest.');
    end
    areasToPlot = plotAreaIdx;
end

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end
tRecStart = timeRange(1);
tRecEnd = timeRange(2);

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'sliding_window_prep', 'utils'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'criticality'));

if includeM2356
    idxM23 = find(strcmp(areas, 'M23'));
    idxM56 = find(strcmp(areas, 'M56'));
    if ~isempty(idxM23) && ~isempty(idxM56) && ~any(strcmp(areas, 'M2356'))
        areas{end+1} = 'M2356';
        dataStruct.areas = areas;
        dataStruct.idMatIdx{end+1} = [dataStruct.idMatIdx{idxM23}(:); dataStruct.idMatIdx{idxM56}(:)];
        if isfield(dataStruct, 'idLabel')
            dataStruct.idLabel{end+1} = [dataStruct.idLabel{idxM23}(:); dataStruct.idLabel{idxM56}(:)];
        end
        numAreas = numel(areas);
        m2356Idx = numAreas;
        if ~ismember(m2356Idx, areasToTest)
            areasToTest = [areasToTest, m2356Idx];
        end
        if isempty(plotAreaIdx)
            areasToPlot = areasToTest;
        end
    end
end

%% ============================= Bin sizes =============================
binSize = zeros(1, numAreas);
if useOptimalBinWindowFunction
    for a = areasToPlot
        neuronIDs = dataStruct.idLabel{a};
        firingRateHz = calculate_firing_rate_from_spikes( ...
            dataStruct.spikeTimes, dataStruct.spikeClusters, neuronIDs, timeRange);
        [binSize(a), ~] = find_optimal_bin_and_window(firingRateHz, minSpikesPerBin, minBinsPerWindow);
    end
else
    binSize(:) = binSizeManual;
end

% Binned matrices per area (once per session); optional PCA on binned data
binnedDataPerArea = cell(1, numAreas);
for a = areasToPlot
    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, min(6, forDim));
        nDimUse = 1:forDim;
        aDataMat = score(:, nDimUse) * coeff(:, nDimUse)' + mu;
    end
    binnedDataPerArea{a} = aDataMat;
end

%% ============================= Cluster bookkeeping =============================
nGroups = numel(clusterResults);
if nGroups < 1
    error('clusterResults is empty.');
end

if isempty(ethogramMinDur) || ~isfinite(ethogramMinDur)
    ethogramWindowSec = minDur;
else
    ethogramWindowSec = ethogramMinDur;
end

frameDtBhv = median(diff(timeAxis(:)));
if ~(isfinite(frameDtBhv) && frameDtBhv > 0)
    error('Could not estimate behavior frame spacing from timeAxis.');
end
nBinsEthogram = max(3, round(ethogramWindowSec / frameDtBhv));
relTimeEthogram = linspace(-ethogramWindowSec / 2, ethogramWindowSec / 2, nBinsEthogram);

groupLabels = cell(1, nGroups);
for g = 1:nGroups
    if isfield(clusterResults{g}, 'includeLabels')
        groupLabels{g} = sprintf('G%d %s', g, mat2str(clusterResults{g}.includeLabels));
    else
        groupLabels{g} = sprintf('Group %d', g);
    end
end

%% ============================= d2 per cluster per area =============================
d2ByGroupArea = cell(nGroups, numAreas);  % {g}{a} = column vector
metaByGroup = cell(nGroups, 1);  % cluster time windows used for neural win

for g = 1:nGroups
    resG = clusterResults{g};
    nClust = size(resG.clusterTimeWindow, 1);
    metaByGroup{g} = nan(nClust, 4);  % tStart tEnd center neuralDur

    for a = 1:numAreas
        d2ByGroupArea{g, a} = nan(nClust, 1);
    end

    if nClust < 1
        continue;
    end

    for c = 1:nClust
        tStart = resG.clusterTimeWindow(c, 1);
        tEnd = resG.clusterTimeWindow(c, 2);
        centerSec = (tStart + tEnd) / 2;
        if useNeuralWindowClusterSpan
            neuralDur = tEnd - tStart;
            winStart = tStart;
            winEnd = tEnd;
        else
            neuralDur = minDur;
            winStart = centerSec - neuralDur / 2;
            winEnd = centerSec + neuralDur / 2;
        end

        metaByGroup{g}(c, :) = [tStart, tEnd, centerSec, neuralDur];

        if neuralDur <= 0
            continue;
        end
        if winStart < tRecStart + windowBuffer || winEnd > tRecEnd - windowBuffer
            continue;
        end

        for a = areasToPlot(:)'
            aDataMat = binnedDataPerArea{a};
            numTimePoints = size(aDataMat, 1);

            [startIdx, endIdx] = calculate_window_indices_from_center( ...
                centerSec, neuralDur, binSize(a), numTimePoints);
            if startIdx < 1 || endIdx > numTimePoints || endIdx <= startIdx
                continue;
            end

            wDataMat = aDataMat(startIdx:endIdx, :);
            d2Val = compute_d2_from_window_matrix(wDataMat, pOrder, critType, ...
                normalizeD2, nShuffles, useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple);
            d2ByGroupArea{g, a}(c) = d2Val;
        end
    end
end

%% ============================= Summary print =============================
fprintf('\n=== d2 cluster counts (finite values) per group x area ===\n');
for a = areasToPlot(:)'
    fprintf('Area %s\n', areas{a});
    for g = 1:nGroups
        v = d2ByGroupArea{g, a};
        nFin = sum(isfinite(v));
        fprintf('  %s: n=%d / %d clusters\n', groupLabels{g}, nFin, numel(v));
    end
end

%% ============================= Plots: d2 comparison =============================
if ~makePlots
    fprintf('makePlots=false; skipping figures.\n');
    return;
end

nAreasPlot = numel(areasToPlot);
figD2 = figure(d2CompareFigureId);
clf(figD2);
set(figD2, 'Color', 'w', 'Name', 'd2 by behavior cluster group', 'NumberTitle', 'off');
if isprop(figD2, 'WindowState')
    figD2.WindowState = 'maximized';
end

colorsGroup = lines(max(2, nGroups));

for plotIdx = 1:nAreasPlot
    a = areasToPlot(plotIdx);
    subplot(nAreasPlot, 1, plotIdx);
    hold on;

    allD2 = [];
    allGrp = [];
    for g = 1:nGroups
        v = d2ByGroupArea{g, a}(:);
        mask = isfinite(v);
        allD2 = [allD2; v(mask)]; %#ok<AGROW>
        allGrp = [allGrp; g * ones(sum(mask), 1)]; %#ok<AGROW>
    end

    if isempty(allD2)
        title(sprintf('%s: no finite d2', areas{a}), 'Interpreter', 'none');
        axis off;
        continue;
    end

    boxplot(allD2, allGrp, 'Labels', groupLabels, 'Symbol', '', 'Notch', 'on');

    for g = 1:nGroups
        v = d2ByGroupArea{g, a}(:);
        mask = isfinite(v);
        if ~any(mask)
            continue;
        end
        xJitter = g + 0.08 * randn(sum(mask), 1);
        scatter(xJitter, v(mask), 18, colorsGroup(g, :), 'filled', ...
            'MarkerFaceAlpha', 0.35, 'MarkerEdgeColor', 'none');
    end

    ylabel('d2');
    title(sprintf('%s: d2 per cluster', areas{a}), 'Interpreter', 'none');
    grid on;
    hold off;
end

sgtitle(sprintf('Spontaneous clusters d2 | %s | neuralWin=%s', sessionName, ...
    ternary_plot_label(useNeuralWindowClusterSpan, 'cluster span', sprintf('fixed %.3g s (minDur)', minDur))), ...
    'Interpreter', 'none');

%% ============================= Plots: ethograms =============================
behaviorIdList = -1:15;
if exist('colors_for_behaviors', 'file')
    behaviorColors = colors_for_behaviors(behaviorIdList);
else
    behaviorColors = lines(numel(behaviorIdList));
end

figEth = figure(ethogramFigureId);
clf(figEth);
set(figEth, 'Color', 'w', 'Name', 'Cluster ethograms (behavior)', 'NumberTitle', 'off');
if isprop(figEth, 'WindowState')
    figEth.WindowState = 'maximized';
end

nColsFig = min(3, nGroups);
nRowsFig = ceil(nGroups / nColsFig);

for g = 1:nGroups
    subplot(nRowsFig, nColsFig, g);
    hold on;

    resG = clusterResults{g};
    nClust = size(resG.clusterTimeWindow, 1);
    if nClust < 1
        title(sprintf('%s: no clusters', groupLabels{g}), 'Interpreter', 'none');
        axis off;
        continue;
    end

    rowOrder = 1:nClust;
    if nClust > maxClustersEthogram
        rowOrder = randperm(nClust, maxClustersEthogram);
        nClust = maxClustersEthogram;
    end

    labelMat = nan(nClust, nBinsEthogram);
    rr = 0;
    for k = rowOrder(:)'
        rr = rr + 1;
        tStart = resG.clusterTimeWindow(k, 1);
        tEnd = resG.clusterTimeWindow(k, 2);
        centerSec = (tStart + tEnd) / 2;
        absT = centerSec + relTimeEthogram;
        labelRow = interp1(timeAxis(:), double(labelsUsed(:)), absT(:), 'nearest', 'extrap');
        labelMat(rr, :) = labelRow;
    end

    rgbImg = ones(nClust, nBinsEthogram, 3);
    for rowIdx = 1:nClust
        for colIdx = 1:nBinsEthogram
            labelVal = labelMat(rowIdx, colIdx);
            mapColorIdx = find(behaviorIdList == labelVal, 1);
            if isempty(mapColorIdx)
                rgbImg(rowIdx, colIdx, :) = [1 1 1];
            else
                rgbImg(rowIdx, colIdx, :) = behaviorColors(mapColorIdx, :);
            end
        end
    end

    image(relTimeEthogram, 1:nClust, rgbImg);
    set(gca, 'YDir', 'normal');
    ylim([0.5, nClust + 0.5]);
    xlim([-ethogramWindowSec / 2, ethogramWindowSec / 2]);
    xlabel('Time rel. cluster center (s)');
    ylabel('Cluster #');
    title(sprintf('%s (n=%d)', groupLabels{g}, nClust), 'Interpreter', 'none');
    plot([0, 0], ylim, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    grid on;
    hold off;
end

sgtitle(sprintf('Behavior ethogram around cluster center | window=%.3g s | %s', ...
    ethogramWindowSec, sessionName), 'Interpreter', 'none');

fprintf('\nDone criticality_spontaneous_clusters.\n');

%% ============================= Local functions =============================
function d2Val = compute_d2_from_window_matrix(wDataMat, pOrder, critType, normalizeD2, nShuffles, ...
    useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple)
% compute_d2_from_window_matrix
% Variables:
%   wDataMat - [timeBins x neurons] counts or activity
%   pOrder, critType - AR / d2 parameters
%   normalizeD2, nShuffles - shuffle normalization
%   useSubsampling and related - optional neuron subsampling
% Goal:
%   Scalar d2 from mean population activity (or mean across subsamples).

    d2Val = nan;
    if isempty(wDataMat) || size(wDataMat, 1) < pOrder + 2
        return;
    end

    if useSubsampling
        nNeuronsTotal = size(wDataMat, 2);
        minNeuronsRequired = round(nNeuronsSubsample * minNeuronsMultiple);
        if nNeuronsTotal < minNeuronsRequired
            return;
        end
        nThisSub = min(nNeuronsSubsample, nNeuronsTotal);
        wPopMatrix = nan(nSubsamples, size(wDataMat, 1));
        for ss = 1:nSubsamples
            if nThisSub == nNeuronsTotal
                neuronIdx = 1:nNeuronsTotal;
            else
                neuronIdx = randperm(nNeuronsTotal, nThisSub);
            end
            subMat = wDataMat(:, neuronIdx);
            wPopMatrix(ss, :) = mean(subMat, 2)';
        end
        wPopActivity = mean(wPopMatrix, 1)';
    else
        wPopActivity = mean(wDataMat, 2);
    end

    try
        [varphi, ~] = myYuleWalker3(double(wPopActivity), pOrder);
        d2Raw = getFixedPointDistance2(pOrder, critType, varphi);
    catch
        d2Raw = nan;
    end

    if ~normalizeD2
        d2Val = d2Raw;
        return;
    end

    if isnan(d2Raw)
        return;
    end

    numNeurons = size(wDataMat, 2);
    numTimeBins = size(wDataMat, 1);
    d2Shuffled = nan(1, nShuffles);
    for shuffleIdx = 1:nShuffles
        permutedDataMat = zeros(size(wDataMat));
        for neuronIdx = 1:numNeurons
            shiftAmount = randi(numTimeBins);
            permutedDataMat(:, neuronIdx) = circshift(wDataMat(:, neuronIdx), shiftAmount);
        end
        if useSubsampling
            wPopMatrixS = nan(nSubsamples, numTimeBins);
            nThisSub = min(nNeuronsSubsample, numNeurons);
            for ss = 1:nSubsamples
                if nThisSub == numNeurons
                    idxN = 1:numNeurons;
                else
                    idxN = randperm(numNeurons, nThisSub);
                end
                wPopMatrixS(ss, :) = mean(permutedDataMat(:, idxN), 2)';
            end
            permutedPop = mean(wPopMatrixS, 1)';
        else
            permutedPop = mean(permutedDataMat, 2);
        end
        try
            [varphiPerm, ~] = myYuleWalker3(double(permutedPop), pOrder);
            d2Shuffled(shuffleIdx) = getFixedPointDistance2(pOrder, critType, varphiPerm);
        catch
            d2Shuffled(shuffleIdx) = nan;
        end
    end
    meanShuffled = nanmean(d2Shuffled);
    if isfinite(meanShuffled) && meanShuffled > 0
        d2Val = d2Raw / meanShuffled;
    end
end

function s = ternary_plot_label(cond, trueStr, falseStr)
% ternary_plot_label — string helper for plot titles.

    if cond
        s = trueStr;
    else
        s = falseStr;
    end
end
