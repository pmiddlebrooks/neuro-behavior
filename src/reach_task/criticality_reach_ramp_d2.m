%
% criticality_reach_ramp_d2
% Test whether peri-reach d2 is driven mainly by sharply ramping single units.
% One reach-centered window [reach - W/2, reach + W/2] per reach (no sliding).
%
% Depends on sessionType, sessionName (choose_task_and_session.m or workspace).
%
% Configuration (edit below):
%   reachWindowSec — total width (s) centered on each reach onset: ramp labeling, d2
%          population trace, and subsample rampiness (other reach onsets excluded).
%   slopeThreshold — slope z-score threshold for classifying ramp-up / ramp-down neurons.
%          z-score uses baseline slope stats from baselineSlopeWindowSec.
%   preSec, postSec — ramp slope test window relative to reach onset [−preSec, +postSec].
%   baselineSlopeWindowSec — baseline slope window (s) relative to reach (e.g., [-3 -2]).
%   slopeSmoothBins — movmean window (bins) applied before slope calculation.
%   prePostEpoch — epoch width (s) for Part 2 rampiness = mean(post epoch) - mean(pre epoch),
%          using epochs centered at reach-preSec and reach+postSec.
%   nSubNeurons, nSubIterations — random subsamples for correlation analysis (Part 2).
%          Part 2 runs on each area/group with neuron count > nSubNeurons only.
%   saveResults — if true, write results struct to .mat under saveDir (default false).
%   windowBuffer, binSizeManual — recording margin and bin width (s).
%   pOrder, critType, normalizeD2, nShuffles, meanSubtract, useLog10D2

%% ============================= Data loading =============================
fprintf('\n=== criticality_reach_ramp_d2: load reach spikes ===\n');

if ~exist('sessionType', 'var')
    error('sessionType must be defined. Run choose_task_and_session.m first or set sessionType in workspace.');
end
if ~exist('sessionName', 'var')
    error('sessionName must be defined. Run choose_task_and_session.m first or set sessionName in workspace.');
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
reachStart = dataR.R(:, 1) / 1000;  % ms -> s
nReach = numel(reachStart);

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
slopeThreshold = 5;
preSec = 0.3;
postSec = 0;
baselineSlopeWindowSec = [-3, -2];
slopeSmoothBins = 3;
prePostEpoch = 0.2;

nSubNeurons = 15;
nSubIterations = 500;
rng(42, 'twister');

windowBuffer = 0.5;
binSizeManual = 0.025;
makePlots = true;
saveResults = false;

binSize = zeros(1, numAreas);
binSize(:) = binSizeManual;

%% ============================= Part 0: rampy neuron labels =============================
% Slope-based ramp index:
%   ramp z = (max slope in [-preSec, +postSec] - mean baseline slope) / std baseline slope
% Baseline slopes come from baselineSlopeWindowSec before reach onset.

rampyMaskByArea = cell(1, numAreas);
rampUpMaskByArea = cell(1, numAreas);
rampDownMaskByArea = cell(1, numAreas);
nRampKeptByArea = zeros(1, numAreas);
nRampRejectedByArea = zeros(1, numAreas);
nRampUpByArea = zeros(1, numAreas);
nRampDownByArea = zeros(1, numAreas);
halfRampWin = reachWindowSec / 2;

for a = areasToTest
    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    nT = size(aDataMat, 1);
    nNeuronsHere = size(aDataMat, 2);

    rampUpMask = false(1, nNeuronsHere);
    rampDownMask = false(1, nNeuronsHere);
    rampyMask = false(1, nNeuronsHere);
    nValidSlopeByNeuron = zeros(1, nNeuronsHere);
    for j = 1:nNeuronsHere
        preSlopeAll = [];
        rampSlopeAll = [];
        for r = 1:nReach
            tReach = reachStart(r);
            reachStartBound = tReach - halfRampWin;
            reachEndBound = tReach + halfRampWin;
            baselineStart = tReach + baselineSlopeWindowSec(1);
            baselineEnd = tReach + baselineSlopeWindowSec(2);
            rampStart = tReach - preSec;
            rampEnd = tReach + postSec;

            if reachStartBound < tRecStart + windowBuffer || reachEndBound > tRecEnd - windowBuffer
                continue;
            end
            if baselineStart < tRecStart + windowBuffer || baselineEnd > tRecEnd - windowBuffer
                continue;
            end
            if rampStart < tRecStart + windowBuffer || rampEnd > tRecEnd - windowBuffer
                continue;
            end
            if window_contains_other_reaches_ramp(reachStartBound, reachEndBound, reachStart, r)
                continue;
            end
            if window_contains_any_reach_ramp(baselineStart, baselineEnd, reachStart)
                continue;
            end

            [b0, b1, bOk] = time_bounds_to_indices_ramp(baselineStart, baselineEnd, binSize(a), nT);
            [r0, r1, rOk] = time_bounds_to_indices_ramp(rampStart, rampEnd, binSize(a), nT);
            if ~bOk || ~rOk
                continue;
            end

            baselineTrace = aDataMat(b0:b1, j);
            rampTrace = aDataMat(r0:r1, j);
            baselineTrace = movmean(baselineTrace, slopeSmoothBins);
            rampTrace = movmean(rampTrace, slopeSmoothBins);
            if numel(baselineTrace) < 3 || numel(rampTrace) < 3
                continue;
            end

            preSlopeAll = [preSlopeAll; diff(baselineTrace) / binSize(a)]; %#ok<AGROW>
            rampSlopeAll = [rampSlopeAll; diff(rampTrace) / binSize(a)]; %#ok<AGROW>
        end

        nValidSlopeByNeuron(j) = numel(rampSlopeAll);
        if isempty(preSlopeAll) || isempty(rampSlopeAll)
            continue;
        end
        preMu = mean(preSlopeAll, 'omitnan');
        preSig = std(preSlopeAll, 0, 'omitnan');
        if ~isfinite(preSig) || preSig <= 0
            continue;
        end
        zSlopeUp = (max(rampSlopeAll, [], 'omitnan') - preMu) / preSig;
        zSlopeDown = (min(rampSlopeAll, [], 'omitnan') - preMu) / preSig;
        isRampUp = isfinite(zSlopeUp) && zSlopeUp > slopeThreshold;
        isRampDown = isfinite(zSlopeDown) && zSlopeDown < -slopeThreshold;

        % Enforce one-direction labels so up+down counts equal rejected count.
        if isRampUp && isRampDown
            if abs(zSlopeDown) > abs(zSlopeUp)
                isRampUp = false;
            else
                isRampDown = false;
            end
        end

        rampUpMask(j) = isRampUp;
        rampDownMask(j) = isRampDown;
        rampyMask(j) = isRampUp || isRampDown;
    end

    rampyMaskByArea{a} = rampyMask;
    rampUpMaskByArea{a} = rampUpMask;
    rampDownMaskByArea{a} = rampDownMask;
    nRampUpByArea(a) = sum(rampUpMask);
    nRampDownByArea(a) = sum(rampDownMask);
    nRampRejectedByArea(a) = sum(rampyMask);
    nRampKeptByArea(a) = nNeuronsHere - nRampRejectedByArea(a);

    fprintf(['Area %s | slope ramp test [-%.2f,+%.2f]s vs baseline [%.2f,%.2f]s | ' ...
        'slopeThreshold=%.1f | kept=%d rejected(total rampy)=%d (up=%d down=%d) of %d | ' ...
        'median valid slope samples/neuron=%.1f\n'], ...
        areas{a}, preSec, postSec, baselineSlopeWindowSec(1), baselineSlopeWindowSec(2), ...
        slopeThreshold, nRampKeptByArea(a), nRampRejectedByArea(a), nRampUpByArea(a), ...
        nRampDownByArea(a), nNeuronsHere, ...
        median(nValidSlopeByNeuron, 'omitnan'));
end

%% ============================= Part 1: d2 full / kept / rampy (one window) =============================
d2Full = cell(1, numAreas);
d2Kept = cell(1, numAreas);
d2Rampy = cell(1, numAreas);
eligibleReach = cell(1, numAreas);
% Mean population activity (mean across neurons, then mean across reaches) for plotting.
tRelPopSec = cell(1, numAreas);
popMeanTraceFull = cell(1, numAreas);
popMeanTraceKept = cell(1, numAreas);
popMeanTraceRampy = cell(1, numAreas);

for a = areasToTest
    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    nT = size(aDataMat, 1);
    nNeuronsHere = size(aDataMat, 2);
    rampyMask = rampyMaskByArea{a};
    if isempty(rampyMask) || numel(rampyMask) ~= nNeuronsHere
        rampyMask = false(1, nNeuronsHere);
    end
    keptCols = find(~rampyMask);
    rampyCols = find(rampyMask);

    fprintf('Area %s: reach-centered d2 (full / kept / rampy)...\n', areas{a});

    [trFull, winFull, elig] = collect_reach_centered_window_cells(aDataMat, nT, ...
        reachStart, reachWindowSec, binSize(a), tRecStart, tRecEnd, windowBuffer, ...
        1:nNeuronsHere);
    eligibleReach{a} = elig;
    [tRelPopSec{a}, popMeanTraceFull{a}] = mean_trace_across_reach_cells(trFull, ...
        reachWindowSec, binSize(a));
    d2Row = compute_d2_by_window_cells_ramp(trFull, winFull, pOrder, critType, ...
        normalizeD2, nShuffles, meanSubtract);
    d2Full{a} = d2Row(:);

    if isempty(keptCols)
        warning('Area %s: no kept neurons; d2 kept set to NaN.', areas{a});
        d2Kept{a} = nan(nReach, 1);
        popMeanTraceKept{a} = [];
    else
        [trK, winK, ~] = collect_reach_centered_window_cells(aDataMat, nT, ...
            reachStart, reachWindowSec, binSize(a), tRecStart, tRecEnd, windowBuffer, ...
            keptCols);
        [~, popMeanTraceKept{a}] = mean_trace_across_reach_cells(trK, reachWindowSec, binSize(a));
        d2RowK = compute_d2_by_window_cells_ramp(trK, winK, pOrder, critType, ...
            normalizeD2, nShuffles, meanSubtract);
        d2Kept{a} = d2RowK(:);
    end

    if isempty(rampyCols)
        warning('Area %s: no rampy neurons; d2 rampy set to NaN.', areas{a});
        d2Rampy{a} = nan(nReach, 1);
        popMeanTraceRampy{a} = [];
    else
        [trR, winR, ~] = collect_reach_centered_window_cells(aDataMat, nT, ...
            reachStart, reachWindowSec, binSize(a), tRecStart, tRecEnd, windowBuffer, ...
            rampyCols);
        [~, popMeanTraceRampy{a}] = mean_trace_across_reach_cells(trR, reachWindowSec, binSize(a));
        d2RowR = compute_d2_by_window_cells_ramp(trR, winR, pOrder, critType, ...
            normalizeD2, nShuffles, meanSubtract);
        d2Rampy{a} = d2RowR(:);
    end
end

if useLog10D2
    for a = areasToTest
        d2Full{a} = apply_log10_safe_matrix_ramp(d2Full{a});
        d2Kept{a} = apply_log10_safe_matrix_ramp(d2Kept{a});
        d2Rampy{a} = apply_log10_safe_matrix_ramp(d2Rampy{a});
    end
end

%% ============================= Part 2: subsample d2 vs rampiness =============================
% Areas with neuron count <= nSubNeurons are skipped (cannot draw fixed-size subsamples).
% Subsamples are drawn from the full area population (mixed neuron types).

d2SubVecByArea = cell(1, numAreas);
rampyIndexVecByArea = cell(1, numAreas);
rhoD2RampByArea = nan(1, numAreas);
subsampleAreasAnalyzed = [];

for ia = 1:numel(areasToTest)
    aSub = areasToTest(ia);
    nNeuSub = numel(dataStruct.idLabel{aSub});
    if nNeuSub <= nSubNeurons
        fprintf('Part 2: skip area %s (neurons=%d, need >%d)\n', areas{aSub}, nNeuSub, nSubNeurons);
        continue;
    end

    neuronIDsSub = dataStruct.idLabel{aSub};
    aDataMatSub = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDsSub, timeRange, binSize(aSub));
    nTsub = size(aDataMatSub, 1);
    d2SubVec = nan(nSubIterations, 1);
    rampyIndexVec = nan(nSubIterations, 1);

    for it = 1:nSubIterations
        cols = randperm(nNeuSub, nSubNeurons);
        [trS, winS, ~] = collect_reach_centered_window_cells(aDataMatSub, nTsub, ...
            reachStart, reachWindowSec, binSize(aSub), tRecStart, tRecEnd, windowBuffer, cols);
        d2RowS = compute_d2_by_window_cells_ramp(trS, winS, pOrder, critType, ...
            normalizeD2, nShuffles, meanSubtract);
        if useLog10D2
            d2RowS = apply_log10_safe_matrix_ramp(d2RowS);
        end
        d2SubVec(it) = mean(d2RowS(:), 'omitnan');

        rampyIndexVec(it) = compute_subsample_rampiness_index(aDataMatSub, nTsub, ...
            reachStart, reachWindowSec, binSize(aSub), tRecStart, tRecEnd, windowBuffer, ...
            cols, preSec, postSec, prePostEpoch);
    end

    d2SubVecByArea{aSub} = d2SubVec;
    rampyIndexVecByArea{aSub} = rampyIndexVec;
    rhoD2RampByArea(aSub) = corr(d2SubVec, rampyIndexVec, 'rows', 'pairwise');
    subsampleAreasAnalyzed(end + 1) = aSub; %#ok<AGROW>
    fprintf('Part 2: area %s | neurons=%d | iters=%d | corr(d2, rampy)=%.3f\n', ...
        areas{aSub}, nNeuSub, nSubIterations, rhoD2RampByArea(aSub));
end

% Defensive dedupe in case areasToTest contains repeated indices.
if ~isempty(subsampleAreasAnalyzed)
    subsampleAreasAnalyzed = unique(subsampleAreasAnalyzed, 'stable');
end

if isempty(subsampleAreasAnalyzed)
    warning('Part 2: no area had more than nSubNeurons=%d units; subsampling skipped.', nSubNeurons);
end

%% ============================= Plots =============================
if makePlots
    if useLog10D2
        yLab = 'mean log_{10}(d2)';
    else
        yLab = 'mean d2';
    end

    d2AxisData = cell(1, 3 * numel(areasToTest));
    popAxisData = cell(1, 3 * numel(areasToTest));
    idxAxis = 1;
    for ia = 1:numel(areasToTest)
        aAxis = areasToTest(ia);
        eligAxis = eligibleReach{aAxis}(:);
        d2ValsFull = d2Full{aAxis}(:);
        d2ValsKept = d2Kept{aAxis}(:);
        d2ValsRampy = d2Rampy{aAxis}(:);
        d2AxisData{idxAxis} = d2ValsFull(eligAxis & isfinite(d2ValsFull)); idxAxis = idxAxis + 1;
        d2AxisData{idxAxis} = d2ValsKept(eligAxis & isfinite(d2ValsKept)); idxAxis = idxAxis + 1;
        d2AxisData{idxAxis} = d2ValsRampy(eligAxis & isfinite(d2ValsRampy)); idxAxis = idxAxis + 1;
    end
    d2YLimShared = compute_shared_ylim_ramp(d2AxisData, 0.08);
    if isempty(d2YLimShared)
        d2YLimShared = [0, 1];
    end

    idxAxis = 1;
    for ia = 1:numel(areasToTest)
        aAxis = areasToTest(ia);
        popAxisData{idxAxis} = popMeanTraceFull{aAxis}; idxAxis = idxAxis + 1;
        popAxisData{idxAxis} = popMeanTraceKept{aAxis}; idxAxis = idxAxis + 1;
        popAxisData{idxAxis} = popMeanTraceRampy{aAxis}; idxAxis = idxAxis + 1;
    end
    popYLimShared = compute_shared_ylim_ramp(popAxisData, 0.08);
    if isempty(popYLimShared)
        popYLimShared = [0, 1];
    end

    figure('Color', 'w', 'Name', 'Ramp removal vs d2 (reach-centered)', 'NumberTitle', 'off');
    nPlotAreas = numel(areasToTest);
    for ia = 1:nPlotAreas
        a = areasToTest(ia);
        subplot(1, max(nPlotAreas, 1), ia);
        hold on;
        plot_bar_d2_three_way(d2Full{a}, d2Kept{a}, d2Rampy{a}, eligibleReach{a}, yLab);
        ylim(d2YLimShared);
        title(sprintf('%s', areas{a}), 'Interpreter', 'none');
    end
    sgtitle(sprintf('d2 (one window per reach): full vs kept vs rampy-only | %s', sessionName), ...
        'Interpreter', 'none');
    outPng1 = fullfile(saveDir, sprintf('criticality_reach_ramp_d2_bars_%s.png', sessionName));
    exportgraphics(gcf, outPng1, 'Resolution', 300);
    fprintf('Saved: %s\n', outPng1);

    figure('Color', 'w', 'Name', 'Pop activity full vs kept vs rampy', 'NumberTitle', 'off');
    for ia = 1:nPlotAreas
        a = areasToTest(ia);
        subplot(1, max(nPlotAreas, 1), ia);
        plot_pop_activity_three_way(tRelPopSec{a}, popMeanTraceFull{a}, popMeanTraceKept{a}, ...
            popMeanTraceRampy{a}, areas{a});
        ylim(popYLimShared);
    end
    sgtitle(sprintf(['Mean pop. activity (mean across neurons, then across reaches) | W=%.2fs | %s'], ...
        reachWindowSec, sessionName), 'Interpreter', 'none');
    outPngPop = fullfile(saveDir, sprintf('criticality_reach_ramp_d2_pop_traces_%s.png', sessionName));
    exportgraphics(gcf, outPngPop, 'Resolution', 300);
    fprintf('Saved: %s\n', outPngPop);

    nScatter = numel(subsampleAreasAnalyzed);
    if nScatter > 0
        scatterXData = cell(1, nScatter);
        scatterYData = cell(1, nScatter);
        for iscp = 1:nScatter
            aIx = subsampleAreasAnalyzed(iscp);
            rv = rampyIndexVecByArea{aIx};
            dv = d2SubVecByArea{aIx};
            useRow = isfinite(rv) & isfinite(dv);
            scatterXData{iscp} = rv(useRow);
            scatterYData{iscp} = dv(useRow);
        end
        scatterXLimShared = compute_shared_ylim_ramp(scatterXData, 0.08);
        if isempty(scatterXLimShared)
            scatterXLimShared = [0, 1];
        end
        scatterYLimShared = compute_shared_ylim_ramp(scatterYData, 0.08);
        if isempty(scatterYLimShared)
            scatterYLimShared = [0, 1];
        end

        figure('Color', 'w', 'Name', 'd2 vs rampiness subsample', 'NumberTitle', 'off');
        for iscp = 1:nScatter
            aIx = subsampleAreasAnalyzed(iscp);
            subplot(1, nScatter, iscp);
            rv = rampyIndexVecByArea{aIx};
            dv = d2SubVecByArea{aIx};
            scatter(rv, dv, 18, 'o', ...
                'MarkerFaceColor', 'none', ...
                'MarkerEdgeColor', [0.2 0.2 0.2], ...
                'LineWidth', 1.0);
            grid on;
            xlabel('rampiness index (post - pre)');
            ylabel(sprintf('mean %s (across reaches)', strrep(yLab, '_', '\_')));
            xlim(scatterXLimShared);
            ylim(scatterYLimShared);
            title(sprintf('%s | \\rho=%.3f', areas{aIx}, rhoD2RampByArea(aIx)), 'Interpreter', 'tex');
        end
        outPng2 = fullfile(saveDir, sprintf('criticality_reach_ramp_d2_scatter_%s.png', sessionName));
        exportgraphics(gcf, outPng2, 'Resolution', 300);
        fprintf('Saved: %s\n', outPng2);
    end
end

%% Save
results = struct();
results.sessionName = sessionName;
results.areas = areas;
results.areasToTest = areasToTest;
results.reachWindowSec = reachWindowSec;
results.slopeThreshold = slopeThreshold;
results.rampyMaskByArea = rampyMaskByArea;
results.rampUpMaskByArea = rampUpMaskByArea;
results.rampDownMaskByArea = rampDownMaskByArea;
results.nRampKeptByArea = nRampKeptByArea;
results.nRampRejectedByArea = nRampRejectedByArea;
results.nRampUpByArea = nRampUpByArea;
results.nRampDownByArea = nRampDownByArea;
results.d2Full = d2Full;
results.d2Kept = d2Kept;
results.d2Rampy = d2Rampy;
results.eligibleReach = eligibleReach;
results.tRelPopSec = tRelPopSec;
results.popMeanTraceFull = popMeanTraceFull;
results.popMeanTraceKept = popMeanTraceKept;
results.popMeanTraceRampy = popMeanTraceRampy;
results.d2SubVecByArea = d2SubVecByArea;
results.rampyIndexVecByArea = rampyIndexVecByArea;
results.rhoD2RampByArea = rhoD2RampByArea;
results.subsampleAreasAnalyzed = subsampleAreasAnalyzed;
results.params = struct('reachWindowSec', reachWindowSec, 'windowBuffer', windowBuffer, ...
    'pOrder', pOrder, 'critType', critType, 'useLog10D2', useLog10D2, ...
    'nSubNeurons', nSubNeurons, 'nSubIterations', nSubIterations, ...
    'preSec', preSec, 'postSec', postSec, 'baselineSlopeWindowSec', baselineSlopeWindowSec, ...
    'slopeSmoothBins', slopeSmoothBins, 'prePostEpoch', prePostEpoch, ...
    'saveResults', saveResults);

if saveResults
    resultsPath = fullfile(saveDir, sprintf('criticality_reach_ramp_d2_%s.mat', sessionName));
    save(resultsPath, 'results');
    fprintf('Saved results: %s\n', resultsPath);
end

%% ============================= Local functions =============================
function tf = window_contains_other_reaches_ramp(winStart, winEnd, reachStartAll, idxExcept)
% window_contains_other_reaches_ramp — other reach onsets inside [winStart, winEnd].
    reachCol = reachStartAll(:);
    nR = numel(reachCol);
    mask = true(nR, 1);
    if idxExcept >= 1 && idxExcept <= nR
        mask(idxExcept) = false;
    end
    rs = reachCol(mask);
    tf = any(rs >= winStart & rs <= winEnd);
end

function tf = window_contains_any_reach_ramp(winStart, winEnd, reachStartAll)
% window_contains_any_reach_ramp
% Goal: true if any reach onset lies inside [winStart, winEnd].
% Variables: winStart, winEnd (s); reachStartAll — vector of reach onset times (s).

    tf = any(reachStartAll >= winStart & reachStartAll <= winEnd);
end

function [startIdx, endIdx, ok] = window_indices_strict_ramp(centerTime, windowSec, binSize, numTimePoints)
% window_indices_strict_ramp — bin indices for symmetric windowSec around centerTime.
    centerIdx = round(centerTime / binSize) + 1;
    winSamples = round(windowSec / binSize);
    if winSamples < 1
        winSamples = 1;
    end
    halfWin = round(winSamples / 2);
    startIdx = centerIdx - halfWin + 1;
    endIdx = startIdx + winSamples - 1;
    ok = startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx;
end

function [startIdx, endIdx, ok] = time_bounds_to_indices_ramp(tStart, tEnd, binSize, numTimePoints)
% time_bounds_to_indices_ramp
% Goal: convert absolute time bounds (s) to binned indices with strict in-range check.
% Variables: tStart, tEnd — window bounds in seconds; binSize — seconds/bin.
% Returns: startIdx, endIdx (1-based), ok — true when bounds map inside data.

    startIdx = round(tStart / binSize) + 1;
    endIdx = round(tEnd / binSize) + 1;
    ok = startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx;
end

function [popTraceCell, windowMatCell, eligibleMask] = collect_reach_centered_window_cells( ...
    aDataMat, nT, reachStart, reachWindowSec, binSize, tRecStart, tRecEnd, windowBuffer, neuronCols)
% collect_reach_centered_window_cells
% Goal: one [bins x nNeu] window per reach, centered on reach onset; same gates as ramp labeling.
% Variables:
%   aDataMat — [timeBins x neurons]; neuronCols — column indices (subset allowed).
% Returns:
%   popTraceCell, windowMatCell — each 1 x nReach; eligibleMask — nReach x 1 logical.

    halfRampWin = reachWindowSec / 2;
    nReachLocal = numel(reachStart);
    popTraceCell = cell(1, nReachLocal);
    windowMatCell = cell(1, nReachLocal);
    eligibleMask = false(nReachLocal, 1);
    subMat = aDataMat(:, neuronCols);

    for r = 1:nReachLocal
        tReach = reachStart(r);
        ws = tReach - halfRampWin;
        we = tReach + halfRampWin;
        if ws < tRecStart + windowBuffer || we > tRecEnd - windowBuffer
            continue;
        end
        if window_contains_other_reaches_ramp(ws, we, reachStart, r)
            continue;
        end
        [i0, i1, idxOk] = window_indices_strict_ramp(tReach, reachWindowSec, binSize, nT);
        if ~idxOk
            continue;
        end
        wMat = subMat(i0:i1, :);
        popTraceCell{1, r} = mean(wMat, 2);
        windowMatCell{1, r} = wMat;
        eligibleMask(r) = true;
    end
end

function [tRelSec, meanPopTrace] = mean_trace_across_reach_cells(popTraceCell, reachWindowSec, binSize)
% mean_trace_across_reach_cells
% Goal: average population-mean traces across reaches (same bins as collect_reach_centered_window_cells).
% Variables:
%   popTraceCell — 1 x nReach, each nonempty entry [nBins x 1] mean firing in window for that reach.
%   reachWindowSec, binSize — used only to build time axis (s) relative to reach at row halfWin.
% Returns:
%   tRelSec — [nBins x 1] seconds relative to reach onset (0 at binned reach center row).
%   meanPopTrace — [nBins x 1] mean across reaches of per-reach pop means.

    nReachLocal = size(popTraceCell, 2);
    sumTrace = [];
    nAcc = 0;

    for r = 1:nReachLocal
        tr = popTraceCell{1, r};
        if isempty(tr)
            continue;
        end
        if isempty(sumTrace)
            sumTrace = zeros(size(tr));
        elseif ~isequal(size(sumTrace), size(tr))
            continue;
        end
        sumTrace = sumTrace + tr;
        nAcc = nAcc + 1;
    end

    if nAcc < 1
        tRelSec = [];
        meanPopTrace = [];
        return;
    end

    meanPopTrace = sumTrace / nAcc;
    nBins = numel(meanPopTrace);
    winSamples = round(reachWindowSec / binSize);
    halfWin = round(winSamples / 2);
    rr = (1:nBins)';
    tRelSec = (rr - halfWin) * binSize;
end

function d2Mat = compute_d2_by_window_cells_ramp(popTraceCell, windowMatCell, pOrder, critType, ...
    normalizeD2, nShuffles, meanSubtract)
% compute_d2_by_window_cells_ramp
% Goal: d2 per reach from one row of window cells (typically nPos=1 reach-centered windows).
% Variables: popTraceCell, windowMatCell — [nPos x nReach] (here nPos=1).
% Returns: d2Mat — [nPos x nReach].

    [nPos, nReachLocal] = size(popTraceCell);
    d2Mat = nan(nPos, nReachLocal);

    for k = 1:nPos
        validMask = false(1, nReachLocal);
        nBinsRef = nan;
        for r = 1:nReachLocal
            tr = popTraceCell{k, r};
            if ~isempty(tr)
                if isnan(nBinsRef)
                    nBinsRef = numel(tr);
                end
                if numel(tr) == nBinsRef
                    validMask(r) = true;
                end
            end
        end
        if ~any(validMask)
            continue;
        end

        validIdx = find(validMask);
        nValid = numel(validIdx);
        popMat = nan(nValid, nBinsRef);
        for ii = 1:nValid
            popMat(ii, :) = popTraceCell{k, validIdx(ii)}(:)';
        end
        if meanSubtract
            meanPerBin = nanmean(popMat, 1);
            popMat = popMat - meanPerBin;
        end

        d2Raw = nan(nValid, 1);
        for ii = 1:nValid
            d2Raw(ii) = compute_d2_from_pop_trace_local_ramp(popMat(ii, :)', pOrder, critType);
        end

        if ~normalizeD2
            d2Mat(k, validIdx) = d2Raw;
            continue;
        end

        d2Shuffled = nan(nValid, nShuffles);
        for sh = 1:nShuffles
            shuffledPopMat = nan(nValid, nBinsRef);
            for ii = 1:nValid
                wMat = windowMatCell{k, validIdx(ii)};
                if isempty(wMat)
                    continue;
                end
                nBins = size(wMat, 1);
                nNeu = size(wMat, 2);
                permutedMat = zeros(size(wMat));
                for n = 1:nNeu
                    permutedMat(:, n) = circshift(wMat(:, n), randi(nBins));
                end
                shuffledPopMat(ii, :) = mean(permutedMat, 2)';
            end
            if meanSubtract
                shMeanPerBin = nanmean(shuffledPopMat, 1);
                shuffledPopMat = shuffledPopMat - shMeanPerBin;
            end
            for ii = 1:nValid
                shuffledTrace = shuffledPopMat(ii, :)';
                d2Shuffled(ii, sh) = compute_d2_from_pop_trace_local_ramp(shuffledTrace, pOrder, critType);
            end
        end

        meanShuffled = nanmean(d2Shuffled, 2);
        d2Norm = nan(nValid, 1);
        for ii = 1:nValid
            if isfinite(d2Raw(ii)) && isfinite(meanShuffled(ii)) && meanShuffled(ii) > 0
                d2Norm(ii) = d2Raw(ii) / meanShuffled(ii);
            end
        end
        d2Mat(k, validIdx) = d2Norm;
    end
end

function d2Val = compute_d2_from_pop_trace_local_ramp(popTrace, pOrder, critType)
% compute_d2_from_pop_trace_local_ramp
% Goal: scalar d2 from one population-mean trace (one window, one reach).
% Variables: popTrace — column vector [timeBins x 1]; pOrder, critType — AR / d2 params.
% Returns: d2Val — scalar or NaN if too short or estimation fails.

    d2Val = nan;
    if isempty(popTrace) || numel(popTrace) < pOrder + 2
        return;
    end
    try
        [varphi, ~] = myYuleWalker3(double(popTrace), pOrder);
        d2Val = getFixedPointDistance2(pOrder, critType, varphi);
    catch
        d2Val = nan;
    end
end

function Mout = apply_log10_safe_matrix_ramp(Min)
    Mout = nan(size(Min));
    ok = isfinite(Min) & Min > 0;
    Mout(ok) = log10(Min(ok));
end

function rampyIdx = compute_subsample_rampiness_index(aDataMat, nT, reachStart, reachWindowSec, ...
    binSize, tRecStart, tRecEnd, windowBuffer, neuronCols, preSec, postSec, prePostEpoch)
% compute_subsample_rampiness_index
% Goal: mean over reaches of post-vs-pre population activity around ramp points.
% Variables: neuronCols — subsampled neuron column indices in aDataMat.
% Returns: rampyIdx — scalar summary (NaN if no valid reaches).

    halfRampWin = reachWindowSec / 2;
    nReachLocal = numel(reachStart);
    rampinessByReach = [];

    subMat = aDataMat(:, neuronCols);
    halfEpoch = prePostEpoch / 2;

    for r = 1:nReachLocal
        tReach = reachStart(r);
        reachStartBound = tReach - halfRampWin;
        reachEndBound = tReach + halfRampWin;
        preCenter = tReach - preSec;
        postCenter = tReach + postSec;
        preStart = preCenter - halfEpoch;
        preEnd = preCenter + halfEpoch;
        postStart = postCenter - halfEpoch;
        postEnd = postCenter + halfEpoch;
        if reachStartBound < tRecStart + windowBuffer || reachEndBound > tRecEnd - windowBuffer
            continue;
        end
        if preStart < tRecStart + windowBuffer || preEnd > tRecEnd - windowBuffer
            continue;
        end
        if postStart < tRecStart + windowBuffer || postEnd > tRecEnd - windowBuffer
            continue;
        end
        if window_contains_other_reaches_ramp(reachStartBound, reachEndBound, reachStart, r)
            continue;
        end

        [p0, p1, pOk] = time_bounds_to_indices_ramp(preStart, preEnd, binSize, nT);
        [q0, q1, qOk] = time_bounds_to_indices_ramp(postStart, postEnd, binSize, nT);
        if ~pOk || ~qOk
            continue;
        end

        preTrace = mean(subMat(p0:p1, :), 2);
        postTrace = mean(subMat(q0:q1, :), 2);
        if isempty(preTrace) || isempty(postTrace)
            continue;
        end
        preVal = mean(preTrace, 'omitnan');
        postVal = mean(postTrace, 'omitnan');
        rampinessByReach(end + 1) = postVal - preVal; %#ok<AGROW>
    end

    if isempty(rampinessByReach)
        rampyIdx = nan;
    else
        rampyIdx = mean(rampinessByReach, 'omitnan');
    end
end

function plot_bar_d2_three_way(d2FullCol, d2KeptCol, d2RampyCol, eligibleMaskCol, yLab)
% plot_bar_d2_three_way
% Goal: grouped mean ± SEM of d2 across reaches with valid windows (per condition, finite d2).
% Variables: d2*Col — nReach x 1; eligibleMaskCol — nReach x 1 logical (window valid).

    elig = eligibleMaskCol(:);
    xCat = {'full', 'no-ramp', 'rampy-only'};
    mu = nan(1, 3);
    se = nan(1, 3);
    for c = 1:3
        if c == 1
            v = d2FullCol(:);
        elseif c == 2
            v = d2KeptCol(:);
        else
            v = d2RampyCol(:);
        end
        useRow = elig & isfinite(v);
        vv = v(useRow);
        mu(c) = mean(vv, 'omitnan');
        nn = numel(vv);
        if nn > 1
            se(c) = std(vv, 0, 'omitnan') / sqrt(nn);
        elseif nn == 1
            se(c) = 0;
        end
    end

    b = bar(1:3, mu);
    b.FaceColor = 'flat';
    b.CData = [0.15 0.15 0.15; 0.15 0.45 0.75; 0.75 0.25 0.2];
    hold on;
    errorbar(1:3, mu, se, 'k.', 'LineWidth', 1.2, 'LineStyle', 'none');
    % Lock ticks to bar centers so labels do not repeat (default auto ticks can misalign).
    set(gca, 'XTick', 1:3, 'XTickLabel', xCat, 'XLim', [0.5 3.5]);
    ylabel(yLab);
    grid on;
end

function plot_pop_activity_three_way(tRelSec, popFull, popKept, popRampy, areaTitleStr)
% plot_pop_activity_three_way
% Goal: overlay mean population traces (full / kept / rampy-only) vs time from reach (s).
% Variables:
%   tRelSec — time axis (s); pop* — mean spikes/bin (mean over neurons, then over reaches).
%   areaTitleStr — subplot title (area name).

    hold on;
    leg = {};
    if ~isempty(tRelSec) && ~isempty(popFull)
        plot(tRelSec(:), popFull(:), 'Color', [0.15 0.15 0.15], 'LineWidth', 1.35);
        leg{end + 1} = 'full'; %#ok<AGROW>
    end
    if ~isempty(tRelSec) && ~isempty(popKept)
        plot(tRelSec(:), popKept(:), 'Color', [0.15 0.45 0.75], 'LineWidth', 1.2);
        leg{end + 1} = 'no-ramp'; %#ok<AGROW>
    end
    if ~isempty(tRelSec) && ~isempty(popRampy)
        plot(tRelSec(:), popRampy(:), 'Color', [0.75 0.25 0.2], 'LineWidth', 1.2);
        leg{end + 1} = 'rampy-only'; %#ok<AGROW>
    end
    xline(0, 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    xlabel('time from reach (s)');
    ylabel('mean spikes / bin');
    if ~isempty(leg)
        legend(leg, 'Location', 'best');
    end
    grid on;
    title(areaTitleStr, 'Interpreter', 'none');
end

function yLimOut = compute_shared_ylim_ramp(dataCell, padFrac)
% compute_shared_ylim_ramp
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
