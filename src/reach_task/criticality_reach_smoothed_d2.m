%%
% criticality_reach_smoothed_d2
% Per-reach population mean traces (popActivity): remove slow local ramps by subtracting a
% time-local mean smoothing (mean over bins within ±meanWidthSec/2 s of each bin).
% Compare d2 on raw pop traces vs residuals (pop minus smooth).
%
% Depends on sessionType, sessionName (choose_task_and_session.m or workspace).
%
% Configuration (edit below):
%   meanWidthSec — full width (s) for local smoothing; default 0.1 => ±0.05 s around each bin center.
%   reachWindowSec — reach-centered analysis window width (same gating as criticality_reach_ramp_d2).
%   pOrder, critType, normalizeD2, nShuffles, meanSubtract, useLog10D2
%   windowBuffer, binSizeManual
%   saveResults — if true, write results struct to .mat under saveDir.

%% ============================= Data loading =============================
fprintf('\n=== criticality_reach_smoothed_d2: load reach spikes ===\n');

if ~exist('sessionType', 'var')
    error(['sessionType must be defined. Run choose_task_and_session.m first ', ...
           'or set sessionType in workspace.']);
end
if ~exist('sessionName', 'var')
    error(['sessionName must be defined. Run choose_task_and_session.m first ', ...
           'or set sessionName in workspace.']);
end

opts = neuro_behavior_options;
opts.frameSize = 0.001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;

dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

if ~isfield(dataStruct, 'dataR')
    error('dataStruct.dataR missing; need reach times.');
end

dataR = dataStruct.dataR;
reachStart = dataR.R(:, 1) / 1000; % ms -> s

areas = dataStruct.areas;
numAreas = numel(areas);
if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'sliding_window_prep', 'utils'));

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end
tRecStart = timeRange(1);
tRecEnd = timeRange(2);

%% ============================= Configuration =============================
pOrder = 10;
critType = 2;
normalizeD2 = false;
nShuffles = 10;
meanSubtract = false;

useLog10D2 = false;

reachWindowSec = 6;

meanWidthSec = 0.1;

rng(42, 'twister');

windowBuffer = 0.5;
binSizeManual = 0.025;
makePlots = true;
saveResults = false;

binSize = zeros(1, numAreas);
binSize(:) = binSizeManual;

%% ============================= Part 1: d2 raw vs mean-smooth residual =============================
d2Pop = cell(1, numAreas);
d2Residual = cell(1, numAreas);
eligibleReach = cell(1, numAreas);
tRelPopSec = cell(1, numAreas);
popMeanTraceRaw = cell(1, numAreas);
popMeanTraceResidual = cell(1, numAreas);

for aIdx = areasToTest
    neuronIDs = dataStruct.idLabel{aIdx};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(aIdx));
    nTimeBins = size(aDataMat, 1);
    nNeuronsHere = size(aDataMat, 2);
    neuronCols = 1:nNeuronsHere;

    fprintf(['Area %s: reach-centered d2 — raw pop vs local-mean subtraction ', ...
        '(meanWidth=%.3fs)\n'], areas{aIdx}, meanWidthSec);

    [popTraceCell, windowMatCell, elig] = collect_reach_centered_window_cells(aDataMat, ...
        nTimeBins, reachStart, reachWindowSec, binSize(aIdx), ...
        tRecStart, tRecEnd, windowBuffer, neuronCols);
    eligibleReach{aIdx} = elig;

    [tRelPopSec{aIdx}, popMeanTraceRaw{aIdx}] = mean_trace_across_reach_cells(popTraceCell, ...
        reachWindowSec, binSize(aIdx));

    popResidualCell = build_pop_residual_cells_local(popTraceCell, reachWindowSec, ...
        binSize(aIdx), meanWidthSec);

    [~, popMeanTraceResidual{aIdx}] = mean_trace_across_reach_cells(popResidualCell, ...
        reachWindowSec, binSize(aIdx));

    d2RowRaw = compute_d2_by_window_cells_ramp(popTraceCell, windowMatCell, pOrder, critType, ...
        normalizeD2, nShuffles, meanSubtract);
    d2Pop{aIdx} = d2RowRaw(:);

    d2RowResid = compute_d2_by_window_cells_mean_smooth_residual(popTraceCell, windowMatCell, ...
        pOrder, critType, normalizeD2, nShuffles, meanSubtract, reachWindowSec, ...
        binSize(aIdx), meanWidthSec);
    d2Residual{aIdx} = d2RowResid(:);
end

if useLog10D2
    for aIdx = areasToTest
        d2Pop{aIdx} = apply_log10_safe_smoothed(d2Pop{aIdx});
        d2Residual{aIdx} = apply_log10_safe_smoothed(d2Residual{aIdx});
    end
end

%% ============================= Plots =============================
if makePlots
    if useLog10D2
        yLab = 'mean log_{10}(d2)';
    else
        yLab = 'mean d2';
    end

    d2AxisData = cell(1, 2 * numel(areasToTest));
    popAxisData = cell(1, 2 * numel(areasToTest));
    idxAxis = 1;
    for ia = 1:numel(areasToTest)
        aAxis = areasToTest(ia);
        eligAxis = eligibleReach{aAxis}(:);
        d2ValsRaw = d2Pop{aAxis}(:);
        d2ValsResid = d2Residual{aAxis}(:);
        d2AxisData{idxAxis} = d2ValsRaw(eligAxis & isfinite(d2ValsRaw)); idxAxis = idxAxis + 1;
        d2AxisData{idxAxis} = d2ValsResid(eligAxis & isfinite(d2ValsResid)); idxAxis = idxAxis + 1;
    end
    d2YLimShared = compute_shared_ylim_smoothed(d2AxisData, 0.08);
    if isempty(d2YLimShared)
        d2YLimShared = [0, 1];
    end

    idxAxis = 1;
    for ia = 1:numel(areasToTest)
        aAxis = areasToTest(ia);
        popAxisData{idxAxis} = popMeanTraceRaw{aAxis}; idxAxis = idxAxis + 1;
        popAxisData{idxAxis} = popMeanTraceResidual{aAxis}; idxAxis = idxAxis + 1;
    end
    popYLimShared = compute_shared_ylim_smoothed(popAxisData, 0.08);
    if isempty(popYLimShared)
        popYLimShared = [0, 1];
    end

    nPlotAreas = numel(areasToTest);
    figure('Color', 'w', 'Name', 'Raw pop vs local-mean residual d2', 'NumberTitle', 'off');
    for ia = 1:nPlotAreas
        aPlot = areasToTest(ia);
        subplot(1, max(nPlotAreas, 1), ia);
        hold on;
        plot_bar_d2_raw_vs_residual(d2Pop{aPlot}, d2Residual{aPlot}, eligibleReach{aPlot}, ...
            yLab);
        ylim(d2YLimShared);
        title(sprintf('%s', areas{aPlot}), 'Interpreter', 'none');
    end
    sgtitle(sprintf(['d2: raw pop vs local-mean subtracted | meanWidth=%.3fs | %s'], ...
            meanWidthSec, sessionName), 'Interpreter', 'none');
    outPngBars = fullfile(saveDir, sprintf('criticality_reach_smoothed_d2_bars_%s.png', sessionName));
    exportgraphics(gcf, outPngBars, 'Resolution', 300);
    fprintf('Saved: %s\n', outPngBars);

    figure('Color', 'w', 'Name', 'Mean pop traces — raw vs residual', 'NumberTitle', 'off');
    for ia = 1:nPlotAreas
        aPlot = areasToTest(ia);
        subplot(1, max(nPlotAreas, 1), ia);
        plot_pop_activity_raw_vs_residual(tRelPopSec{aPlot}, ...
            popMeanTraceRaw{aPlot}, popMeanTraceResidual{aPlot}, areas{aPlot});
        ylim(popYLimShared);
    end
    sgtitle(sprintf(['Mean pop trace (cross-reach avg) vs residual | meanWidth=%.3fs | ', ...
                     'W=%.2fs | %s'], meanWidthSec, reachWindowSec, sessionName), ...
        'Interpreter', 'none');
    outPngPop = fullfile(saveDir, sprintf('criticality_reach_smoothed_d2_pop_traces_%s.png', ...
            sessionName));
    exportgraphics(gcf, outPngPop, 'Resolution', 300);
    fprintf('Saved: %s\n', outPngPop);
end

%% ============================= Save =============================
results = struct();
results.sessionName = sessionName;
results.areas = areas;
results.areasToTest = areasToTest;
results.reachWindowSec = reachWindowSec;
results.meanWidthSec = meanWidthSec;
results.d2Pop = d2Pop;
results.d2Residual = d2Residual;
results.eligibleReach = eligibleReach;
results.tRelPopSec = tRelPopSec;
results.popMeanTraceRaw = popMeanTraceRaw;
results.popMeanTraceResidual = popMeanTraceResidual;
results.params = struct('reachWindowSec', reachWindowSec, 'meanWidthSec', meanWidthSec, ...
    'windowBuffer', windowBuffer, 'pOrder', pOrder, 'critType', critType, ...
    'normalizeD2', normalizeD2, 'nShuffles', nShuffles, 'meanSubtract', meanSubtract, ...
    'useLog10D2', useLog10D2, 'saveResults', saveResults);

if saveResults
    resultsPath = fullfile(saveDir, sprintf('criticality_reach_smoothed_d2_%s.mat', sessionName));
    save(resultsPath, 'results');
    fprintf('Saved results: %s\n', resultsPath);
end

%% ============================= Local functions =============================
function tf = window_contains_other_reaches_ramp(winStart, winEnd, reachStartAll, idxExcept)
% window_contains_other_reaches_ramp — other reach onsets inside [winStart, winEnd].
    reachCol = reachStartAll(:);
    nReachAll = numel(reachCol);
    maskOthers = true(nReachAll, 1);
    if idxExcept >= 1 && idxExcept <= nReachAll
        maskOthers(idxExcept) = false;
    end
    rs = reachCol(maskOthers);
    tf = any(rs >= winStart & rs <= winEnd);
end

function [startIdx, endIdx, okFlag] = window_indices_strict_ramp(centerTime, windowSec, binSizeLoc, ...
    numTimePoints)
% window_indices_strict_ramp — bin indices for symmetric windowSec around centerTime.
    centerIdx = round(centerTime / binSizeLoc) + 1;
    winSamples = round(windowSec / binSizeLoc);
    if winSamples < 1
        winSamples = 1;
    end
    halfWinBins = round(winSamples / 2);
    startIdx = centerIdx - halfWinBins + 1;
    endIdx = startIdx + winSamples - 1;
    okFlag = startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx;
end

function [popTraceCell, windowMatCell, eligibleMask] = collect_reach_centered_window_cells( ...
    aDataMat, nTimeBins, reachStart, reachWindowSec, binSizeLoc, tRecStart, tRecEnd, ...
    windowBuffer, neuronCols)
% collect_reach_centered_window_cells
% Goal: one [bins x nNeurons] window per reach, centered on reach onset.

    halfRampWin = reachWindowSec / 2;
    nReachHere = numel(reachStart);
    popTraceCell = cell(1, nReachHere);
    windowMatCell = cell(1, nReachHere);
    eligibleMask = false(nReachHere, 1);
    subMat = aDataMat(:, neuronCols);

    for rIx = 1:nReachHere
        tReach = reachStart(rIx);
        ws = tReach - halfRampWin;
        we = tReach + halfRampWin;
        if ws < tRecStart + windowBuffer || we > tRecEnd - windowBuffer
            continue;
        end
        if window_contains_other_reaches_ramp(ws, we, reachStart, rIx)
            continue;
        end
        [idx0, idx1, idxOk] = window_indices_strict_ramp(tReach, reachWindowSec, binSizeLoc, ...
            nTimeBins);
        if ~idxOk
            continue;
        end
        wMat = subMat(idx0:idx1, :);
        popTraceCell{1, rIx} = mean(wMat, 2);
        windowMatCell{1, rIx} = wMat;
        eligibleMask(rIx) = true;
    end
end

function [timeRelSec, meanPopTrace] = mean_trace_across_reach_cells(popTraceCell, ...
        reachWindowSec, binSizeLoc)
% mean_trace_across_reach_cells — average per-reach pop traces across reaches.

    nReachHere = size(popTraceCell, 2);
    sumTrace = [];
    numAcc = 0;

    for rIx = 1:nReachHere
        traceRow = popTraceCell{1, rIx};
        if isempty(traceRow)
            continue;
        end
        if isempty(sumTrace)
            sumTrace = zeros(size(traceRow));
        elseif ~isequal(size(sumTrace), size(traceRow))
            continue;
        end
        sumTrace = sumTrace + traceRow;
        numAcc = numAcc + 1;
    end

    if numAcc < 1
        timeRelSec = [];
        meanPopTrace = [];
        return;
    end

    meanPopTrace = sumTrace / numAcc;
    nBinsTot = numel(meanPopTrace);
    winSamples = round(reachWindowSec / binSizeLoc);
    halfWinBins = round(winSamples / 2);
    rowIdx = (1:nBinsTot)';
    timeRelSec = (rowIdx - halfWinBins) * binSizeLoc;
end

function tBinCenters = bin_centers_for_reach_window(reachWindowSec, binSizeLoc, nBins)
% bin_centers_for_reach_window — times (s relative to reach) at bin centers, matches mean_trace.
    winSamples = round(reachWindowSec / binSizeLoc);
    halfWinBins = round(winSamples / 2);
    rowIdx = (1:nBins)';
    tBinCenters = (rowIdx - halfWinBins) * binSizeLoc;
end

function smoothTrace = mean_smoothed_pop_trace_local(popTraceVec, reachWindowSec, ...
    binSizeLoc, meanWidthSec)
% mean_smoothed_pop_trace_local
% Goal: at each bin, mean of signal over bins whose centers lie within ±meanWidthSec/2 of this bin center.
% Variables: popTraceVec — column [nBins x 1].

    traceCol = double(popTraceVec(:));
    nBins = numel(traceCol);
    if nBins < 1
        smoothTrace = traceCol;
        return;
    end
    tCenters = bin_centers_for_reach_window(reachWindowSec, binSizeLoc, nBins);
    halfWide = meanWidthSec / 2;
    epsTol = max(1e-12 * meanWidthSec, 1e-15);
    smoothTrace = zeros(nBins, 1);
    for bIx = 1:nBins
        inWindow = abs(tCenters - tCenters(bIx)) <= halfWide + epsTol;
        smoothTrace(bIx) = mean(traceCol(inWindow), 'omitnan');
    end
end

function residTrace = subtract_mean_smooth_from_pop_trace(popTraceVec, reachWindowSec, ...
    binSizeLoc, meanWidthSec)
% subtract_mean_smooth_from_pop_trace — pop minus local mean-smoothed estimator (column vector).

    smoothCol = mean_smoothed_pop_trace_local(popTraceVec, reachWindowSec, binSizeLoc, meanWidthSec);
    residTrace = double(popTraceVec(:)) - smoothCol;
end

function popResidualCellOut = build_pop_residual_cells_local(popTraceCell, reachWindowSec, ...
    binSizeLoc, meanWidthSec)
% build_pop_residual_cells_local — residual pop traces with same layout as popTraceCell.

    [numPos, nReachHere] = size(popTraceCell);
    popResidualCellOut = cell(size(popTraceCell));
    for kPos = 1:numPos
        for rIx = 1:nReachHere
            tr = popTraceCell{kPos, rIx};
            if isempty(tr)
                continue;
            end
            popResidualCellOut{kPos, rIx} = subtract_mean_smooth_from_pop_trace(tr, ...
                reachWindowSec, binSizeLoc, meanWidthSec);
        end
    end
end

function d2Mat = compute_d2_by_window_cells_mean_smooth_residual(popTraceCell, windowMatCell, ...
    pOrder, critType, normalizeD2Flag, numShuffles, meanSubtractFlag, reachWindowSec, ...
    binSizeLoc, meanWidthSec)
% compute_d2_by_window_cells_mean_smooth_residual
% Goal: d2 after subtracting local mean smoothed baseline from population mean trace (each reach),
%       and same preprocessing on shuffle-mean traces when normalizeD2Flag.

    [numPos, nReachHere] = size(popTraceCell);
    d2Mat = nan(numPos, nReachHere);

    for kPos = 1:numPos
        reachValidMask = false(1, nReachHere);
        nBinsReference = nan;
        for rIx = 1:nReachHere
            traceRow = popTraceCell{kPos, rIx};
            if ~isempty(traceRow)
                if isnan(nBinsReference)
                    nBinsReference = numel(traceRow);
                end
                if numel(traceRow) == nBinsReference
                    reachValidMask(rIx) = true;
                end
            end
        end
        if ~any(reachValidMask)
            continue;
        end

        validIx = find(reachValidMask);
        numValid = numel(validIx);
        popMat = nan(numValid, nBinsReference);
        for jj = 1:numValid
            rawTraceVec = popTraceCell{kPos, validIx(jj)}(:);
            residTraceVec = subtract_mean_smooth_from_pop_trace(rawTraceVec, reachWindowSec, ...
                binSizeLoc, meanWidthSec);
            popMat(jj, :) = residTraceVec.';
        end
        if meanSubtractFlag
            meanPerBinAcrossReaches = nanmean(popMat, 1);
            popMat = popMat - meanPerBinAcrossReaches;
        end

        d2RawCol = nan(numValid, 1);
        for jj = 1:numValid
            d2RawCol(jj) = compute_d2_from_pop_trace_smoothed(popMat(jj, :)', pOrder, critType);
        end

        if ~normalizeD2Flag
            d2Mat(kPos, validIx) = d2RawCol;
            continue;
        end

        d2ShuffledAll = nan(numValid, numShuffles);
        for shuffleIx = 1:numShuffles
            shuffledPopMatIter = nan(numValid, nBinsReference);
            for jj = 1:numValid
                wMatWin = windowMatCell{kPos, validIx(jj)};
                if isempty(wMatWin)
                    continue;
                end
                nBinsWin = size(wMatWin, 1);
                nNeuronsWin = size(wMatWin, 2);
                permutedMat = zeros(size(wMatWin));
                for unitIx = 1:nNeuronsWin
                    permutedMat(:, unitIx) = circshift(wMatWin(:, unitIx), randi(nBinsWin));
                end
                shuffledMeanTrace = mean(permutedMat, 2);
                residShuffled = subtract_mean_smooth_from_pop_trace(shuffledMeanTrace, ...
                    reachWindowSec, binSizeLoc, meanWidthSec);
                shuffledPopMatIter(jj, :) = residShuffled.';
            end
            if meanSubtractFlag
                shMeanPerBin = nanmean(shuffledPopMatIter, 1);
                shuffledPopMatIter = shuffledPopMatIter - shMeanPerBin;
            end
            for jj = 1:numValid
                d2ShuffledAll(jj, shuffleIx) = compute_d2_from_pop_trace_smoothed( ...
                    shuffledPopMatIter(jj, :)', pOrder, critType);
            end
        end

        meanShuffledReaches = nanmean(d2ShuffledAll, 2);
        d2NormColumn = nan(numValid, 1);
        for jj = 1:numValid
            if isfinite(d2RawCol(jj)) && isfinite(meanShuffledReaches(jj)) && ...
               meanShuffledReaches(jj) > 0
                d2NormColumn(jj) = d2RawCol(jj) / meanShuffledReaches(jj);
            end
        end
        d2Mat(kPos, validIx) = d2NormColumn;
    end
end

function d2Val = compute_d2_from_pop_trace_smoothed(popTrace, pOrder, critType)
% compute_d2_from_pop_trace_smoothed — scalar d2 from population trace vector.

    d2Val = nan;
    if isempty(popTrace) || numel(popTrace) < pOrder + 2
        return;
    end
    try
        [varPhi, ~] = myYuleWalker3(double(popTrace), pOrder);
        d2Val = getFixedPointDistance2(pOrder, critType, varPhi);
    catch
        d2Val = nan;
    end
end

function d2MatOut = compute_d2_by_window_cells_ramp(popTraceCell, windowMatCell, pOrder, critType, ...
    normalizeD2Flag, numShuffles, meanSubtractFlag)
% compute_d2_by_window_cells_ramp — verbatim logic from criticality_reach_ramp_d2.

    [numPos, nReachHere] = size(popTraceCell);
    d2MatOut = nan(numPos, nReachHere);

    for kPos = 1:numPos
        reachValidMask = false(1, nReachHere);
        nBinsReference = nan;
        for rIx = 1:nReachHere
            traceRow = popTraceCell{kPos, rIx};
            if ~isempty(traceRow)
                if isnan(nBinsReference)
                    nBinsReference = numel(traceRow);
                end
                if numel(traceRow) == nBinsReference
                    reachValidMask(rIx) = true;
                end
            end
        end
        if ~any(reachValidMask)
            continue;
        end

        validIx = find(reachValidMask);
        numValid = numel(validIx);
        popMat = nan(numValid, nBinsReference);
        for jj = 1:numValid
            popMat(jj, :) = popTraceCell{kPos, validIx(jj)}(:)';
        end
        if meanSubtractFlag
            meanPerBinAcrossReaches = nanmean(popMat, 1);
            popMat = popMat - meanPerBinAcrossReaches;
        end

        d2RawCol = nan(numValid, 1);
        for jj = 1:numValid
            d2RawCol(jj) = compute_d2_from_pop_trace_smoothed(popMat(jj, :)', pOrder, critType);
        end

        if ~normalizeD2Flag
            d2MatOut(kPos, validIx) = d2RawCol;
            continue;
        end

        d2ShuffledAll = nan(numValid, numShuffles);
        for shuffleIx = 1:numShuffles
            shuffledPopMatIter = nan(numValid, nBinsReference);
            for jj = 1:numValid
                wMatWin = windowMatCell{kPos, validIx(jj)};
                if isempty(wMatWin)
                    continue;
                end
                nBinsWin = size(wMatWin, 1);
                nNeuronsWin = size(wMatWin, 2);
                permutedMat = zeros(size(wMatWin));
                for unitIx = 1:nNeuronsWin
                    permutedMat(:, unitIx) = circshift(wMatWin(:, unitIx), randi(nBinsWin));
                end
                shuffledPopMatIter(jj, :) = mean(permutedMat, 2)';
            end
            if meanSubtractFlag
                shMeanPerBin = nanmean(shuffledPopMatIter, 1);
                shuffledPopMatIter = shuffledPopMatIter - shMeanPerBin;
            end
            for jj = 1:numValid
                d2ShuffledAll(jj, shuffleIx) = compute_d2_from_pop_trace_smoothed( ...
                    shuffledPopMatIter(jj, :)', pOrder, critType);
            end
        end

        meanShuffledReaches = nanmean(d2ShuffledAll, 2);
        d2NormColumn = nan(numValid, 1);
        for jj = 1:numValid
            if isfinite(d2RawCol(jj)) && isfinite(meanShuffledReaches(jj)) && ...
               meanShuffledReaches(jj) > 0
                d2NormColumn(jj) = d2RawCol(jj) / meanShuffledReaches(jj);
            end
        end
        d2MatOut(kPos, validIx) = d2NormColumn;
    end
end

function plot_bar_d2_raw_vs_residual(d2PopCol, d2ResidCol, eligibleMaskCol, yAxisLabel)
% plot_bar_d2_raw_vs_residual — grouped mean ± SEM for raw vs residual d2 across eligible reaches.

    eligLogical = eligibleMaskCol(:);
    categoriesList = {'raw pop', 'mean-smooth'};
    muEach = nan(1, 2);
    semEach = nan(1, 2);
    d2Stacks = [d2PopCol(:), d2ResidCol(:)];
    for cIx = 1:2
        valueCol = d2Stacks(:, cIx);
        reachUseRow = eligLogical & isfinite(valueCol);
        valueVecValid = valueCol(reachUseRow);
        muEach(cIx) = mean(valueVecValid, 'omitnan');
        numNn = numel(valueVecValid);
        if numNn > 1
            semEach(cIx) = std(valueVecValid, 0, 'omitnan') / sqrt(numNn);
        elseif numNn == 1
            semEach(cIx) = 0;
        end
    end

    barHandles = bar(1:2, muEach);
    barHandles.FaceColor = 'flat';
    barHandles.CData = [0.2 0.2 0.2; 0.2 0.5 0.75];
    hold on;
    errorbar(1:2, muEach, semEach, 'k.', 'LineWidth', 1.2, 'LineStyle', 'none');
    set(gca, 'XTick', 1:2, 'XTickLabel', categoriesList, 'XLim', [0.5 2.5]);
    ylabel(yAxisLabel);
    grid on;
end

function plot_pop_activity_raw_vs_residual(timeRelSec, popRawMean, popResidMean, areaTitleStr)
% plot_pop_activity_raw_vs_residual — overlay mean crossed-reach traces.

    hold on;
    legendEntries = {};
    if ~isempty(timeRelSec) && ~isempty(popRawMean)
        plot(timeRelSec(:), popRawMean(:), 'Color', [0.2 0.2 0.2], 'LineWidth', 1.35);
        legendEntries{end + 1} = 'raw pop mean'; %#ok<AGROW>
    end
    if ~isempty(timeRelSec) && ~isempty(popResidMean)
        plot(timeRelSec(:), popResidMean(:), 'Color', [0.15 0.45 0.75], 'LineWidth', 1.2);
        legendEntries{end + 1} = 'cross-reach avg residual'; %#ok<AGROW>
    end
    yline(0, 'k:', 'LineWidth', 0.7, 'HandleVisibility', 'off');
    xline(0, 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    xlabel('time from reach (s)');
    ylabel('mean spikes / bin (or residual)');
    if ~isempty(legendEntries)
        legend(legendEntries, 'Location', 'best');
    end
    grid on;
    title(areaTitleStr, 'Interpreter', 'none');
end

function yLimOut = compute_shared_ylim_smoothed(dataCell, padFrac)
% compute_shared_ylim_smoothed
% Goal: build one shared y-axis [min max] across vectors in dataCell.
% Variables: dataCell — cell array of numeric vectors; padFrac — fractional range padding.
% Returns: yLimOut — [] when no finite values.

    allVals = [];
    for iVec = 1:numel(dataCell)
        vecNow = dataCell{iVec};
        if isempty(vecNow)
            continue;
        end
        vecNow = vecNow(:);
        vecNow = vecNow(isfinite(vecNow));
        if isempty(vecNow)
            continue;
        end
        allVals = [allVals; vecNow]; %#ok<AGROW>
    end

    if isempty(allVals)
        yLimOut = [];
        return;
    end

    yMin = min(allVals);
    yMax = max(allVals);
    if yMin == yMax
        delta = max(1e-6, abs(yMin) * 0.1);
        yLimOut = [yMin - delta, yMax + delta];
        return;
    end

    padVal = (yMax - yMin) * padFrac;
    yLimOut = [yMin - padVal, yMax + padVal];
end

function mOutLog = apply_log10_safe_smoothed(mInLin)
    mOutLog = nan(size(mInLin));
    positiveFinite = isfinite(mInLin) & mInLin > 0;
    mOutLog(positiveFinite) = log10(mInLin(positiveFinite));
end
