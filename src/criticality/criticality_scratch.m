%%
% CRITICALITY_SCRATCH
%
% Scratch workspace for paired spike data in a MAT file:
%   Variables starting with n carry neuron ID per spike (e.g. n50_02_1_2_10)
%   Variables starting with t carry spike time in ms with the same substring after t
% Binning uses binSizeSec (default 25 ms); t* is converted to seconds before bin_spikes.
%
% Optional overrides (set before running):
%   dataFile, dataFileMultiPaul, binSizeSec, makePlots, slidingWindowSizeSec, pOrder,
%   critType, windowSizes, d2CenterSec
%
% Multi-set file (paired n*, t* columns; same substring after leading n vs t): default
%   dataFileMultiPaul next to crit4paul prefers crit4TestPaul.mat then critTestPaul.mat.

%% =============================    Configuration    =============================

    fPrefer = fullfile(paths.dropPath, 'temp_data', 'crit4TestPaul.mat');
    fAlt = fullfile(paths.dropPath, 'temp_data', 'critTestPaul.mat');
dataFileMultiPaul = fAlt;

    binSizeSec = 0.025;   % 25 ms bins for popActivity (file ti is still in ms)
    slidingWindowSizeSec = 6;
    pOrder = 10;
    critType = 2;
    makePlots = true;
    d2CenterSec = 10;
    windowSizes = 2:0.25:20;

msPerSec = 1000;   % spike times in t* vectors are ms; convert before bin_spikes

%% =============================    Load data    =============================
fprintf('\n=== criticality_scratch: load %s ===\n', dataFile);
if ~isfile(dataFile)
    error('criticality_scratch:MissingFile', 'Data file not found: %s', dataFile);
end

raw = load(dataFile);
fnRaw = fieldnames(raw);
[nVarsPrimary, tVarsPrimary] = scratch_pair_nt_variables(fnRaw);
numDatasets = numel(nVarsPrimary);
if numDatasets < 1
    error('criticality_scratch:LoadFile', ...
        'No paired (n*, t*) variable sets found in MAT file (need matching suffix after n vs t).');
end
stepSizeSec = binSizeSec;   % sliding-window step matches analysis bin size

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'sliding_window_prep', 'utils'));
fprintf('  Spike times: ms in file -> s for binning; binSize = %.3g s, step = %.3g s\n', ...
    binSizeSec, stepSizeSec);

datasets = struct('neuronVarTitle', '', 'neuronId', [], 'spikeTimeMs', [], 'spikeTimeSec', [], ...
    'spikeTimes', [], 'spikeClusters', [], 'popActivity', [], 'popRateHz', [], ...
    'timeAxisSec', [], 'timeRangeSec', [0, 0]);

for d = 1:numDatasets
    varN = nVarsPrimary{d};
    varT = tVarsPrimary{d};
    neuronId = raw.(varN)(:);
    spikeTimeMs = double(raw.(varT)(:));
    spikeTimeSec = spikeTimeMs / msPerSec;
    spikeTimes = spikeTimeSec;
    spikeClusters = double(neuronId(:));
    neuronIds = sort(unique(spikeClusters), 'ascend');

    tMin = min(spikeTimes);
    tMax = max(spikeTimes);
    timeRangeDataset = [min(0, tMin), tMax + binSizeSec];
    if ~(timeRangeDataset(2) > timeRangeDataset(1))
        error('criticality_scratch:BadTimeRange', ...
            'Dataset %s (%s): invalid time range.', varN, varT);
    end

    dataMat = bin_spikes(spikeTimes, spikeClusters, neuronIds, timeRangeDataset, binSizeSec);
    popActivity = sum(dataMat, 2);
    popRateHz = popActivity / binSizeSec;
    numBins = size(dataMat, 1);
    timeAxisSec = ((0:numBins - 1)' + 0.5) * binSizeSec;

    datasets(d).neuronVarTitle = varN;
    datasets(d).neuronId = neuronId;
    datasets(d).spikeTimeMs = spikeTimeMs;
    datasets(d).spikeTimeSec = spikeTimeSec;
    datasets(d).spikeTimes = spikeTimes;
    datasets(d).spikeClusters = spikeClusters;
    datasets(d).dataMat = dataMat;
    datasets(d).popActivity = popActivity;
    datasets(d).popRateHz = popRateHz;
    datasets(d).timeAxisSec = timeAxisSec;
    datasets(d).timeRangeSec = timeRangeDataset;

    fprintf('  %s: %d spikes, %d neurons, t = [%.4g, %.4g] s, mean pop rate = %.2f Hz\n', ...
        varN, numel(spikeTimes), numel(neuronIds), min(spikeTimes), max(spikeTimes), ...
        nanmean(popRateHz));
end

%% =============================    Sliding-window d2    =============================
winSamples = round(slidingWindowSizeSec / binSizeSec);
stepSamples = round(stepSizeSec / binSizeSec);
if winSamples < 1
    winSamples = 1;
end
if stepSamples < 1
    stepSamples = 1;
end

fprintf('\n--- Sliding d2: window = %.3g s, step = %.3g s (%d bins / %d bins) ---\n', ...
    slidingWindowSizeSec, stepSizeSec, winSamples, stepSamples);

for d = 1:numDatasets
    popActivity = datasets(d).popActivity;
    [d2Sliding, timeCenterSec, numWindows] = scratch_compute_sliding_d2( ...
        popActivity, binSizeSec, slidingWindowSizeSec, stepSizeSec, pOrder, critType);
    datasets(d).d2Sliding = d2Sliding;
    datasets(d).d2TimeCenterSec = timeCenterSec;
    nValidD2 = sum(isfinite(d2Sliding));
    fprintf('  %s: %d windows, %d finite d2\n', datasets(d).neuronVarTitle, numWindows, nValidD2);
end

%% =============================    d2 vs window (fixed center)    =============================
numWinSizes = numel(windowSizes);
fprintf('\n--- d2 vs window size (symmetric about t = %.3g s): %d sizes [%.3g .. %.3g] s ---\n', ...
    d2CenterSec, numWinSizes, windowSizes(1), windowSizes(end));

for d = 1:numDatasets
    popActivity = datasets(d).popActivity;
    numTimePoints = numel(popActivity);
    timeRangeDataset = datasets(d).timeRangeSec;
    d2ByWinSize = nan(1, numWinSizes);

    for iW = 1:numWinSizes
        winSize = windowSizes(iW);
        winStart = d2CenterSec - winSize / 2;
        winEnd = d2CenterSec + winSize / 2;
        if winStart < timeRangeDataset(1) || winEnd > timeRangeDataset(2)
            continue;
        end
        [startIdx, endIdx] = window_bin_indices_from_center_time( ...
            d2CenterSec, winSize, binSizeSec, numTimePoints);
        if isempty(startIdx)
            continue;
        end
        wPopActivity = popActivity(startIdx:endIdx);
        d2ByWinSize(iW) = compute_d2_from_pop_activity(wPopActivity, pOrder, critType);
    end

    datasets(d).d2ByWinSize = d2ByWinSize;
    datasets(d).windowSizesCentered = windowSizes;
    fprintf('  %s: %d / %d finite d2\n', datasets(d).neuronVarTitle, sum(isfinite(d2ByWinSize)), numWinSizes);
end

%% =============================    Plots    =============================
if makePlots
    [~, primaryBase, primaryExt] = fileparts(dataFile);
    figPopRate = figure(3201); clf(figPopRate);
    set(figPopRate, 'Units', 'normalized', 'Position', [0.12 0.1 0.72 0.78]);
    axPopList = tight_subplot(numDatasets, 1, [0.038 0]);
    for d = 1:numDatasets
        axThis = axPopList(d);
        hold(axThis, 'on');
        rateHz = datasets(d).popRateHz;
        plot(axThis, datasets(d).timeAxisSec, rateHz, 'Color', [0.2 0.45 0.72], 'LineWidth', 0.8);
        grid(axThis, 'on');
        xlim(axThis, datasets(d).timeRangeSec);
        ylabel(axThis, 'Population rate (Hz)');
        title(axThis, datasets(d).neuronVarTitle, 'Interpreter', 'none');
        if d < numDatasets
            set(axThis, 'XTickLabel', []);
        else
            xlabel(axThis, 'Time (s)');
        end
    end
    sgtitle(figPopRate, sprintf('%s%s: population rate (%.3g s bins)', ...
        primaryBase, primaryExt, binSizeSec), ...
        'Interpreter', 'none', 'FontSize', 12);

    figSlideD2 = figure(3202); clf(figSlideD2);
    set(figSlideD2, 'Units', 'normalized', 'Position', [0.12 0.1 0.72 0.78]);
    axD2List = tight_subplot(numDatasets, 1, [0.038 0]);
    for d = 1:numDatasets
        axThis = axD2List(d);
        hold(axThis, 'on');
        d2Trace = datasets(d).d2Sliding;
        plot(axThis, datasets(d).d2TimeCenterSec, d2Trace, 'Color', [0.2 0.45 0.72], 'LineWidth', 0.8);
        grid(axThis, 'on');
        xlim(axThis, datasets(d).timeRangeSec);
        ylabel(axThis, 'd2');
        title(axThis, datasets(d).neuronVarTitle, 'Interpreter', 'none');
        if d < numDatasets
            set(axThis, 'XTickLabel', []);
        else
            xlabel(axThis, 'Time (s)');
        end
    end
    sgtitle(figSlideD2, sprintf('%s%s: sliding d2 (%.3g s window, %.3g s step)', ...
        primaryBase, primaryExt, slidingWindowSizeSec, stepSizeSec), ...
        'Interpreter', 'none', 'FontSize', 12);

    figure(3203); clf;
    set(gcf, 'Units', 'normalized', 'Position', [0.12 0.1 0.72 0.78]);
    tlWin = tiledlayout(numDatasets, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for d = 1:numDatasets
        nexttile;
        hold on;
        d2VsWin = datasets(d).d2ByWinSize;
        winVec = datasets(d).windowSizesCentered;
        plot(winVec, d2VsWin, '-o', 'LineWidth', 1.2, 'Color', [0.2 0.45 0.72], ...
            'MarkerFaceColor', [0.65 0.8 1]);
        grid on;
        xlabel('Window size (s)');
        ylabel('d2');
        title(datasets(d).neuronVarTitle, 'Interpreter', 'none');
    end
    sgtitle(tlWin, sprintf('%s%s: d2 vs window (symmetric, center t = %.3g s)', ...
        primaryBase, primaryExt, d2CenterSec), 'Interpreter', 'none', 'FontSize', 12);
end

%% =============================    Multi-set Paul (n*, t* pairs)    =============================
if isfile(dataFileMultiPaul)
    fprintf('\n=== criticality_scratch: multi-set %s ===\n', dataFileMultiPaul);
    rawMultiPaul = load(dataFileMultiPaul);
    [nVarsMulti, tVarsMulti] = scratch_pair_nt_variables(fieldnames(rawMultiPaul));
    numMulti = numel(nVarsMulti);
    if numMulti ~= 9
        warning('criticality_scratch:MultiPaulCount', ...
            'Expected 9 paired n/t variable sets; found %d.', numMulti);
    end
    scratchMultiPaul = repmat(struct( ...
        'neuronVarTitle', '', 'popActivity', [], 'd2Sliding', [], 'd2TimeCenterSec', [], ...
        'timeAxisSec', [], 'timeRangeSec', [0, 0]), min(9, numMulti), 1);

    for k = 1:min(9, numMulti)
        varN = nVarsMulti{k};
        varT = tVarsMulti{k};
        neuronIdVec = double(rawMultiPaul.(varN)(:));
        spikeTimeMsVec = double(rawMultiPaul.(varT)(:));
        spikeTimeSecVec = spikeTimeMsVec / msPerSec;
        neuronIdsMulti = sort(unique(neuronIdVec), 'ascend');
        tMin = min(spikeTimeSecVec);
        tMax = max(spikeTimeSecVec);
        timeRangeMulti = [min(0, tMin), tMax + binSizeSec];
        if ~(timeRangeMulti(2) > timeRangeMulti(1))
            error('criticality_scratch:BadTimeRange', ...
                'Set %s (%s): invalid time range.', varN, varT);
        end

        dataMatM = bin_spikes(spikeTimeSecVec, neuronIdVec, neuronIdsMulti, timeRangeMulti, binSizeSec);
        popActM = sum(dataMatM, 2);
        numBinsM = size(dataMatM, 1);
        timeAxisM = ((0:numBinsM - 1)' + 0.5) * binSizeSec;

        [d2SlM, tCentM, ~] = scratch_compute_sliding_d2( ...
            popActM, binSizeSec, slidingWindowSizeSec, stepSizeSec, pOrder, critType);

        scratchMultiPaul(k).neuronVarTitle = varN;
        scratchMultiPaul(k).popActivity = popActM;
        scratchMultiPaul(k).d2Sliding = d2SlM;
        scratchMultiPaul(k).d2TimeCenterSec = tCentM;
        scratchMultiPaul(k).timeAxisSec = timeAxisM;
        scratchMultiPaul(k).timeRangeSec = timeRangeMulti;
        fprintf('  %s: %d spikes, %d neurons, %d windows, %d finite d2\n', ...
            varN, numel(spikeTimeSecVec), numel(neuronIdsMulti), ...
            numel(d2SlM), sum(isfinite(d2SlM)));
    end

    if makePlots
        figure(3204); clf;
        set(gcf, 'Units', 'normalized', 'Position', [0.08 0.06 0.88 0.84]);
        tlGrid = tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        for k = 1:min(9, numMulti)
            nexttile;
            hold on;
            plot(scratchMultiPaul(k).d2TimeCenterSec, scratchMultiPaul(k).d2Sliding, ...
                'Color', [0.2 0.45 0.72], 'LineWidth', 0.8);
            grid on;
            xlim(scratchMultiPaul(k).timeRangeSec);
            xlabel('Time (s)');
            ylabel('d2');
            title(scratchMultiPaul(k).neuronVarTitle, 'Interpreter', 'none');
        end
        [~, multiFileBase, multiFileExt] = fileparts(dataFileMultiPaul);
        sgtitle(tlGrid, sprintf('Sliding d2 (%.3g s window, %.3g s step) — %s%s', ...
            slidingWindowSizeSec, stepSizeSec, multiFileBase, multiFileExt), ...
            'Interpreter', 'none', 'FontSize', 11);
    end
else
    fprintf('\ncriticality_scratch: skip multi-set (not found): %s\n', dataFileMultiPaul);
end

fprintf('\n=== criticality_scratch: done ===\n');

%% -------------------------------------------------------------------------
function [d2Sliding, timeCenterSec, numWindows] = scratch_compute_sliding_d2( ...
    popActivity, binSizeSec, slidingWindowSizeSec, stepSizeSec, pOrder, critType)
% SCRATCH_COMPUTE_SLIDING_D2 Population d2 along a sliding window (column outputs).

    winSamples = round(slidingWindowSizeSec / binSizeSec);
    stepSamples = round(stepSizeSec / binSizeSec);
    if winSamples < 1
        winSamples = 1;
    end
    if stepSamples < 1
        stepSamples = 1;
    end
    numTimePoints = numel(popActivity);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    if numWindows < 1
        error('criticality_scratch:WindowTooLong', ...
            'Sliding window (%d bins) longer than trace (%d bins).', winSamples, numTimePoints);
    end
    d2Sliding = nan(numWindows, 1);
    timeCenterSec = nan(numWindows, 1);
    for w = 1:numWindows
        startIdx = (w - 1) * stepSamples + 1;
        endIdx = startIdx + winSamples - 1;
        centerIdx = startIdx + floor(winSamples / 2);
        timeCenterSec(w) = (centerIdx - 0.5) * binSizeSec;
        wPopActivity = popActivity(startIdx:endIdx);
        d2Sliding(w) = compute_d2_from_pop_activity(wPopActivity, pOrder, critType);
    end
end

function [nVars, tVars] = scratch_pair_nt_variables(fieldNames)
% SCRATCH_PAIR_NT_VARIABLES Pair n* neuron-ID vectors with t* spike-time vectors (same suffix).

    nVars = {};
    tVars = {};
    sortedNames = sort(fieldNames(:));
    for ii = 1:numel(sortedNames)
        vn = sortedNames{ii};
        if numel(vn) < 2 || vn(1) ~= 'n'
            continue;
        end
        suf = vn(2:end);
        tn = ['t', suf];
        if any(strcmp(tn, sortedNames))
            nVars{end+1} = vn; %#ok<AGROW>
            tVars{end+1} = tn;
        end
    end
end

function [startIdx, endIdx] = window_bin_indices_from_center_time( ...
    centerTimeSec, windowSizeSec, binSizeSec, numTimePoints)
% WINDOW_BIN_INDICES_FROM_CENTER_TIME
%
% Variables:
%   centerTimeSec   - window center (s)
%   windowSizeSec   - full window width (s)
%   binSizeSec      - bin width (s)
%   numTimePoints   - number of time bins in popActivity
%
% Goal:
%   Map a centered time window to inclusive 1-based row indices; [] if invalid.

    centerIdx = round(centerTimeSec / binSizeSec) + 1;
    winSamples = round(windowSizeSec / binSizeSec);
    if winSamples < 1
        winSamples = 1;
    end
    halfWin = round(winSamples / 2);
    startIdx = centerIdx - halfWin + 1;
    endIdx = startIdx + winSamples - 1;
    if startIdx < 1 || endIdx > numTimePoints || endIdx <= startIdx
        startIdx = [];
        endIdx = [];
    end
end

function d2Val = compute_d2_from_pop_activity(wPopActivity, pOrder, critType)
% COMPUTE_D2_FROM_POP_ACTIVITY
%
% Variables:
%   wPopActivity - population sum firing vector [timeBins x 1]
%   pOrder       - AR order for myYuleWalker3
%   critType     - criticality type for getFixedPointDistance2
%
% Goal:
%   Return scalar d2 for one window; NaN on failure.

    d2Val = nan;
    if isempty(wPopActivity)
        return;
    end
    v = double(wPopActivity(:));
    if numel(v) <= pOrder
        return;
    end
    try
        [varphi, ~] = myYuleWalker3(v, pOrder);
        d2Val = getFixedPointDistance2(pOrder, critType, varphi);
    catch
        d2Val = nan;
    end
end
