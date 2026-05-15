%%
% Criticality: d2 vs reach-aligned window size
%
% For each window duration in windowsToTest, extracts a single window per
% reach centered on that reach onset, computes population-mean d2 (same
% pipeline as criticality_reach_intertrial_d2.m), and averages across all
% reaches that pass validity for that specific window size.
%
% Validity (per reach r, per window size W):
%   Window [t_r - W/2, t_r + W/2] must not overlap the prior or next reach
%   onset (treated as point events): t_{r-1} < t_r - W/2 and t_{r+1} > t_r + W/2
%   when those neighbors exist. The window must also lie fully inside the
%   spike collection time range and yield in-bounds bin indices (no clamping).
%
% Workspace:
%   sessionType, sessionName - same as choose_task_and_session / load_sliding_window_data
%
% Optional overrides (set before running):
%   windowsToTest, binSizeManual, pOrder, critType, meanSubtract, useLog10D2,
%   makePlots, areasToTest, cvThreshold, windowMin, windowSizeStepSec,
%   halfSessionReachSweepMode, halfSessionWinMaxSec
%
% Half-session reach sweep (section before local functions):
%   halfSessionReachSweepMode:
%     'midSessionCenter' (default) — center on the reach closest to mid-session;
%       symmetric window [tCenter - W/2, tCenter + W/2]; marks when each reach
%       onset first enters that window.
%     'firstReachGrowRight' — anchor at first reach onset; window [tFirst, tFirst+W];
%       grow W into the session; marks when each reach onset first enters the window.
%   Window cap: if halfSessionWinMaxSec is unset or empty, use the session-derived
%   maximum for the chosen mode (symmetric margin around mid-session center, or
%   timeRange(2) - tFirst). If halfSessionWinMaxSec is set, the sweep uses
%   min(halfSessionWinMaxSec, that session maximum).
%
% Per window size, the script reports how many individual reach-level popActivity
% CV values exceed cvThreshold (default 5). With makePlots true, a second figure
% shows CV-per-reach histograms for the smallest and largest entries in windowsToTest.
%
% Coefficient of variation (CV): for each popActivity trace,
% CV = nanstd(x) / |nanmean(x)| (dimensionless proportion). Mean CV at each window
% size is nanmean across traces. A vertical line marks the smallest window size
% where mean CV <= cvThreshold (default 5; heuristic for high-CV / artifact
% dominated traces at short windows).

%% =============================    Configuration    =============================
if ~exist('windowsToTest', 'var') || isempty(windowsToTest)
    windowsToTest = 2:0.5:20;   % seconds; evaluated independently per size
end

if ~exist('cvThreshold', 'var') || isempty(cvThreshold)
    cvThreshold = 5;   % mean CV (proportion) at or below this -> xline marker (artifact screen)
end

binSizeManual = 0.025;      % seconds (fixed across window sizes)
pOrder = 10;
critType = 2;
meanSubtract = false;       % if true, subtract mean pop activity within each window before d2
useLog10D2 = false;          % if true, use log10 scale for y-axis when all plotted means are positive
makePlots = true;
%% =============================    Data loading    =============================
fprintf('\n=== criticality_reach_d2_vs_windowSize: load data ===\n');

if ~exist('sessionType', 'var')
    error('sessionType must be defined (e.g. run choose_task_and_session.m first).');
end
if ~exist('sessionName', 'var')
    error('sessionName must be defined.');
end

opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

if ~isfield(dataStruct, 'dataR')
    error('dataStruct.dataR is required for reach times.');
end

reachStart = dataStruct.dataR.R(:,1) / 1000;  % ms -> s
numReaches = numel(reachStart);
fprintf('Loaded %d reach onsets\n', numReaches);

areas = dataStruct.areas;
numAreas = length(areas);
if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'sliding_window_prep', 'utils'));

binSize = zeros(1, numAreas);
binSize(:) = binSizeManual;

%% =============================    Analysis    =============================
numW = numel(windowsToTest);
meanD2ByArea = cell(1, numAreas);
semD2ByArea = cell(1, numAreas);
nValidByArea = cell(1, numAreas);
meanCvByArea = cell(1, numAreas);
semCvByArea = cell(1, numAreas);
nCvValidByArea = cell(1, numAreas);
nCvAboveThresholdByArea = cell(1, numAreas);
cvPerSampleSmallestWinByArea = cell(1, numAreas);
cvPerSampleLargestWinByArea = cell(1, numAreas);
binnedSpikeMatByArea = cell(1, numAreas);
smallestWinCvBelowThresholdByArea = nan(1, numAreas);

for a = areasToTest
    fprintf('\n--- Area %s ---\n', areas{a});
    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);
    binSizeSec = binSize(a);

    meanD2 = nan(1, numW);
    semD2 = nan(1, numW);
    nValid = zeros(1, numW);
    meanCv = nan(1, numW);
    semCv = nan(1, numW);
    nCvValid = zeros(1, numW);
    nCvAboveThreshold = zeros(1, numW);

    for iW = 1:numW
        winSize = windowsToTest(iW);
        d2PerReach = nan(1, numReaches);
        cvPerReach = nan(1, numReaches);

        for r = 1:numReaches
            tR = reachStart(r);
            winStart = tR - winSize / 2;
            winEnd = tR + winSize / 2;

            if r > 1 && winStart <= reachStart(r - 1)
                continue;
            end
            if r < numReaches && winEnd >= reachStart(r + 1)
                continue;
            end
            if winStart < timeRange(1) || winEnd > timeRange(2)
                continue;
            end

            [startIdx, endIdx] = window_bin_indices_from_center_time( ...
                tR, winSize, binSizeSec, numTimePoints);
            if isempty(startIdx) || startIdx < 1 || endIdx > numTimePoints || endIdx <= startIdx
                continue;
            end

            wDataMat = aDataMat(startIdx:endIdx, :);
            wPopActivity = sum(wDataMat, 2);
            if meanSubtract
                wPopActivity = wPopActivity - nanmean(wPopActivity);
            end

            cvPerReach(r) = population_trace_cv(wPopActivity);
            d2PerReach(r) = compute_d2_from_pop_activity(wPopActivity, pOrder, critType);
        end

        validMask = ~isnan(d2PerReach);
        nValid(iW) = sum(validMask);
        if nValid(iW) > 0
            meanD2(iW) = nanmean(d2PerReach(validMask));
            if nValid(iW) > 1
                semD2(iW) = nanstd(d2PerReach(validMask)) / sqrt(nValid(iW));
            else
                semD2(iW) = nan;
            end
        end

        cvMask = ~isnan(cvPerReach);
        nCvValid(iW) = sum(cvMask);
        nCvAboveThreshold(iW) = sum(cvPerReach > cvThreshold & cvMask);
        if iW == 1
            cvPerSampleSmallestWinByArea{a} = cvPerReach(cvMask).';
        end
        if iW == numW
            cvPerSampleLargestWinByArea{a} = cvPerReach(cvMask).';
        end
        if nCvValid(iW) > 0
            meanCv(iW) = nanmean(cvPerReach(cvMask));
            if nCvValid(iW) > 1
                semCv(iW) = nanstd(cvPerReach(cvMask)) / sqrt(nCvValid(iW));
            else
                semCv(iW) = nan;
            end
        end

        fprintf(['  W = %5.1f s: nValid = %4d / %d, mean d2 = %.4g, mean CV = %.3g ', ...
            '(n_CV = %d, n_CV > %.3g = %d)\n'], ...
            winSize, nValid(iW), numReaches, meanD2(iW), meanCv(iW), nCvValid(iW), ...
            cvThreshold, nCvAboveThreshold(iW));
    end

    meanD2ByArea{a} = meanD2;
    semD2ByArea{a} = semD2;
    nValidByArea{a} = nValid;
    meanCvByArea{a} = meanCv;
    semCvByArea{a} = semCv;
    nCvValidByArea{a} = nCvValid;
    nCvAboveThresholdByArea{a} = nCvAboveThreshold;
    binnedSpikeMatByArea{a} = aDataMat;

    idxCross = find(meanCv <= cvThreshold & ~isnan(meanCv), 1);
    if ~isempty(idxCross)
        smallestWinCvBelowThresholdByArea(a) = windowsToTest(idxCross);
        fprintf('  Smallest W with mean CV <= %.3g: %.2f s\n', cvThreshold, ...
            smallestWinCvBelowThresholdByArea(a));
    end
end

%% =============================    Pack results    =============================
results = struct();
results.windowsToTest = windowsToTest;
results.reachStart = reachStart;
results.timeRange = timeRange;
results.binSize = binSize;
results.pOrder = pOrder;
results.critType = critType;
results.meanSubtract = meanSubtract;
results.areas = areas;
results.areasToTest = areasToTest;
results.meanD2ByArea = meanD2ByArea;
results.semD2ByArea = semD2ByArea;
results.nValidByArea = nValidByArea;
results.meanCvByArea = meanCvByArea;
results.semCvByArea = semCvByArea;
results.nCvValidByArea = nCvValidByArea;
results.nCvAboveThresholdByArea = nCvAboveThresholdByArea;
results.cvPerSampleSmallestWinByArea = cvPerSampleSmallestWinByArea;
results.cvPerSampleLargestWinByArea = cvPerSampleLargestWinByArea;
results.cvThreshold = cvThreshold;
results.smallestWinCvBelowThresholdByArea = smallestWinCvBelowThresholdByArea;
results.sessionName = sessionName;

% outMat = fullfile(saveDir, 'criticality_reach_d2_vs_windowSize_results.mat');
% save(outMat, '-struct', 'results');
% fprintf('\nSaved results: %s\n', outMat);

%% =============================    Plot    =============================
if makePlots
    nPlot = numel(areasToTest);
    figure(3101); clf;
    set(gcf, 'WindowState', 'maximized');
    tl = tiledlayout(nPlot, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:nPlot
        a = areasToTest(k);
        nexttile;
        hold on;
        meanD2 = meanD2ByArea{a};
        semD2 = semD2ByArea{a};
        meanCv = meanCvByArea{a};
        semCvPlot = semCvByArea{a};
        semCvPlot(isnan(semCvPlot)) = 0;

        yyaxis left
        semD2Plot = semD2;
        semD2Plot(isnan(semD2Plot)) = 0;
        errorbar(windowsToTest, meanD2, semD2Plot, '-o', 'LineWidth', 1.5, 'CapSize', 4, 'Color', [0 0.45 0.74]);
        if useLog10D2 && all(meanD2(~isnan(meanD2)) > 0)
            set(gca, 'YScale', 'log');
        end
        ylabel('mean d2');

        yyaxis right
        errorbar(windowsToTest, meanCv, semCvPlot, '-s', 'LineWidth', 1.5, 'CapSize', 4, 'Color', [0.85 0.33 0.1]);
        ylabel('mean CV');
        set(gca, 'YColor', [0.85 0.33 0.1]);

        winStar = smallestWinCvBelowThresholdByArea(a);
        if ~isnan(winStar)
            xline(winStar, '--', 'Color', [0.15 0.15 0.15], 'LineWidth', 1.25, ...
                'Label', sprintf('mean CV <= %.g', cvThreshold), ...
                'LabelHorizontalAlignment', 'left', 'FontSize', 9);
        end

        yyaxis left
        set(gca, 'YColor', [0 0.45 0.74]);
        grid on;
        xlabel('Window size (s)');
        title(sprintf('%s (bin %.3f s)', areas{a}, binSize(a)), 'Interpreter', 'none');
    end
    sgtitle(tl, sprintf('%s: d2 vs window size (reach-centered, non-overlap neighbors)', ...
        sessionName), 'Interpreter', 'none', 'FontSize', 12);
    outPng = fullfile(saveDir, 'criticality_reach_d2_vs_windowSize.png');
    exportgraphics(gcf, outPng, 'Resolution', 300);
    fprintf('Saved figure: %s\n', outPng);

    % CV per reach: histograms at minimum vs maximum window in windowsToTest
    winSmallestSec = windowsToTest(1);
    winLargestSec = windowsToTest(end);
    figure(3102); clf;
    set(gcf, 'WindowState', 'maximized');
    tlCv = tiledlayout(nPlot, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:nPlot
        a = areasToTest(k);
        cvSmall = cvPerSampleSmallestWinByArea{a}(:);
        cvLarge = cvPerSampleLargestWinByArea{a}(:);
        cvBoth = [cvSmall; cvLarge];
        if ~isempty(cvBoth)
            xLo = min(cvBoth);
            xHi = max(cvBoth);
            if ~(xHi > xLo)
                xHi = xLo + max(1e-6, abs(xLo) * 1e-4 + 1e-9);
            end
            xLimRow = [xLo, xHi];
        else
            xLimRow = [0, 1];
        end

        nexttile;
        hold on;
        if ~isempty(cvSmall)
            histogram(cvSmall, 30, 'FaceColor', [0.55 0.65 0.85], 'EdgeColor', [0.2 0.2 0.2]);
            xline(cvThreshold, '--', 'Color', [0.72 0.15 0.15], 'LineWidth', 1.1);
        end
        xlim(xLimRow);
        xlabel('CV');
        ylabel('Count');
        title(sprintf('%s: W = %.2f s (smallest)', areas{a}, winSmallestSec), 'Interpreter', 'none');
        grid on;

        nexttile;
        hold on;
        if ~isempty(cvLarge)
            histogram(cvLarge, 30, 'FaceColor', [0.85 0.72 0.55], 'EdgeColor', [0.2 0.2 0.2]);
            xline(cvThreshold, '--', 'Color', [0.72 0.15 0.15], 'LineWidth', 1.1);
        end
        xlim(xLimRow);
        xlabel('CV');
        ylabel('Count');
        title(sprintf('%s: W = %.2f s (largest)', areas{a}, winLargestSec), 'Interpreter', 'none');
        grid on;
    end
    sgtitle(tlCv, sprintf('%s: CV per reach (smallest vs largest window)', sessionName), ...
        'Interpreter', 'none', 'FontSize', 12);
    outPngCv = fullfile(saveDir, 'criticality_reach_d2_vs_windowSize_cv_hist_min_max_win.png');
    exportgraphics(gcf, outPngCv, 'Resolution', 300);
    fprintf('Saved figure: %s\n', outPngCv);
end

%% ==========    Half-session reach: d2 vs window (growing inclusion)    ==========
% Optional: halfSessionReachSweepMode ('midSessionCenter' | 'firstReachGrowRight'),
% halfSessionWinMaxSec (empty = use session max for the mode), windowMin,
% windowSizeStepSec.

    windowMin = windowsToTest(1);
    windowSizeStepSec = 0.5;


halfSessionReachSweepMode = 'firstReachGrowRight';
halfSessionTime = (timeRange(1) + timeRange(2)) / 2;
[~, reachIdxHalfSession] = min(abs(reachStart - halfSessionTime));
tCenterHalfReach = reachStart(reachIdxHalfSession);
tFirstReach = reachStart(1);

% Session-derived maximum window width for each sweep geometry
winMaxSessionSecMidCenter = 2 * min(tCenterHalfReach - timeRange(1), timeRange(2) - tCenterHalfReach);
winMaxSessionSecFirstReach = timeRange(2) - tFirstReach;

if strcmpi(halfSessionReachSweepMode, 'firstReachGrowRight')
    sweepModeLabel = 'firstReachGrowRight';
    winMaxFromSessionSec = winMaxSessionSecFirstReach;
    tCenterForBins = @(winSize) tFirstReach + winSize / 2;
    inclusionFun = @(winSize) reachStart >= tFirstReach & reachStart <= (tFirstReach + winSize);
    fprintf(['\n--- Half-session reach sweep (first reach -> grow right): ', ...
        'tFirst = %.4g s; session max W = %.4g s ---\n'], tFirstReach, winMaxFromSessionSec);
elseif strcmpi(halfSessionReachSweepMode, 'midSessionCenter')
    sweepModeLabel = 'midSessionCenter';
    winMaxFromSessionSec = winMaxSessionSecMidCenter;
    tCenterForBins = @(winSize) tCenterHalfReach;
    inclusionFun = @(winSize) reachStart >= (tCenterHalfReach - winSize / 2) & ...
        reachStart <= (tCenterHalfReach + winSize / 2);
    fprintf(['\n--- Half-session reach sweep (mid-session center): center reach idx %d at %.4g s ', ...
        '(mid-session %.4g s); session max W = %.4g s ---\n'], ...
        reachIdxHalfSession, tCenterHalfReach, halfSessionTime, winMaxFromSessionSec);
else
    error(['halfSessionReachSweepMode must be ''midSessionCenter'' or ''firstReachGrowRight''; ', ...
        'got ''%s''.'], char(halfSessionReachSweepMode));
end

halfSessionWinMaxSec = 800; %30*60;
winMaxHalfSweepSec = min(halfSessionWinMaxSec, winMaxFromSessionSec);

if winMaxHalfSweepSec < windowMin
    warning('criticality_reach_d2_vs_windowSize:halfSessionWinMaxTooSmall', ...
        'Half-session sweep: winMax (%.4g s) < windowMin (%.4g s); no sweep steps.', ...
        winMaxHalfSweepSec, windowMin);
end

d2HalfReachByArea = cell(1, numAreas);
reachInclusionWinSecByArea = cell(1, numAreas);
reachInclusionReachIdxByArea = cell(1, numAreas);
windowsHalfReachSweep = [];

if winMaxHalfSweepSec >= windowMin
    windowsHalfReachSweep = windowMin:windowSizeStepSec:winMaxHalfSweepSec;
else
    windowsHalfReachSweep = [];
end
numWinHalf = numel(windowsHalfReachSweep);
fprintf(['  Sweep W = %.4g:%.4g:%.4g s (%d steps); (session max %.4g s) ---\n'], ...
    windowMin, windowSizeStepSec, winMaxHalfSweepSec, numWinHalf, ...
    winMaxFromSessionSec);

for a = areasToTest
    aDataMat = binnedSpikeMatByArea{a};
    numTimePoints = size(aDataMat, 1);
    binSizeSec = binSize(a);
    d2HalfReachTrace = nan(1, numWinHalf);
    prevIncludedMask = false(1, numReaches);
    reachInclusionWinSecList = zeros(0, 1);
    reachInclusionIdxList = zeros(0, 1);

    for jW = 1:numWinHalf
        winSize = windowsHalfReachSweep(jW);
        includedNow = inclusionFun(winSize);
        newlyInMask = includedNow & ~prevIncludedMask;
        if any(newlyInMask)
            newIdx = find(newlyInMask);
            reachInclusionWinSecList = [reachInclusionWinSecList; repmat(winSize, numel(newIdx), 1)]; %#ok<AGROW>
            reachInclusionIdxList = [reachInclusionIdxList; newIdx(:)]; %#ok<AGROW>
        end
        prevIncludedMask = includedNow;

        tCenterBins = tCenterForBins(winSize);
        [startIdx, endIdx] = window_bin_indices_from_center_time( ...
            tCenterBins, winSize, binSizeSec, numTimePoints);
        if isempty(startIdx) || startIdx < 1 || endIdx > numTimePoints || endIdx <= startIdx
            continue;
        end
        wDataMat = aDataMat(startIdx:endIdx, :);
        wPopActivity = sum(wDataMat, 2);
        if meanSubtract
            wPopActivity = wPopActivity - nanmean(wPopActivity);
        end
        d2HalfReachTrace(jW) = compute_d2_from_pop_activity(wPopActivity, pOrder, critType);
    end

    d2HalfReachByArea{a} = d2HalfReachTrace;
    reachInclusionWinSecByArea{a} = reachInclusionWinSecList;
    reachInclusionReachIdxByArea{a} = reachInclusionIdxList;
end

results.windowMin = windowMin;
results.windowSizeStepSec = windowSizeStepSec;
results.halfSessionTime = halfSessionTime;
results.reachIdxHalfSession = reachIdxHalfSession;
results.tCenterHalfReach = tCenterHalfReach;
results.tFirstReach = tFirstReach;
results.winMaxSessionSecMidCenter = winMaxSessionSecMidCenter;
results.winMaxSessionSecFirstReach = winMaxSessionSecFirstReach;
results.winMaxSessionSec = winMaxFromSessionSec;   % session max for the active sweep mode
results.halfSessionReachSweepMode = sweepModeLabel;
results.halfSessionWinMaxSec = halfSessionWinMaxSec;
results.winMaxHalfSessionFromSessionSec = winMaxFromSessionSec;
results.winMaxHalfSweepSec = winMaxHalfSweepSec;
results.windowsHalfReachSweep = windowsHalfReachSweep;
results.d2HalfReachByArea = d2HalfReachByArea;
results.reachInclusionWinSecByArea = reachInclusionWinSecByArea;
results.reachInclusionReachIdxByArea = reachInclusionReachIdxByArea;

if makePlots && ~isempty(windowsHalfReachSweep)
    nPlotHalf = numel(areasToTest);
    figure(3103); clf;
    set(gcf, 'WindowState', 'maximized');
    tlHalf = tiledlayout(nPlotHalf, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:nPlotHalf
        a = areasToTest(k);
        nexttile;
        hold on;
        d2Trace = d2HalfReachByArea{a};
        plot(windowsHalfReachSweep, d2Trace, '-o', 'LineWidth', 1.5, 'Color', [0 0.45 0.74], ...
            'MarkerFaceColor', [0.65 0.8 1]);
        if useLog10D2
            d2Finite = d2Trace(~isnan(d2Trace));
            if ~isempty(d2Finite) && all(d2Finite > 0)
                set(gca, 'YScale', 'log');
            end
        end
        winInc = reachInclusionWinSecByArea{a};
        for e = 1:numel(winInc)
            wE = winInc(e);
            xline(wE, ':', 'Color', [0.82 0.25 0.2], 'LineWidth', 0.9);
            tolW = max(1e-9, 1e-9 * max(1, abs(wE)));
            jMatch = find(abs(windowsHalfReachSweep - wE) < tolW, 1);
            if isempty(jMatch)
                [~, jMatch] = min(abs(windowsHalfReachSweep - wE));
            end
            yE = d2Trace(jMatch);
            if ~isnan(yE)
                scatter(wE, yE, 28, [0.82 0.25 0.2], 'filled');
            end
        end
        grid on;
        xlabel('Window size (s)');
        ylabel('d2');
        title(sprintf('%s (bin %.3f s)', areas{a}, binSize(a)), 'Interpreter', 'none');
    end
    if strcmpi(sweepModeLabel, 'firstReachGrowRight')
        sgTitleStr = sprintf(['%s: d2 vs window (first reach at %.3g s, grow right; ', ...
            'red = reach onset newly inside window)'], sessionName, tFirstReach);
        outPngHalf = fullfile(saveDir, 'criticality_reach_d2_halfsession_first_reach_grow_right.png');
    else
        sgTitleStr = sprintf(['%s: d2 vs window (center = reach nearest mid-session; ', ...
            'red = reach onset newly inside window)'], sessionName);
        outPngHalf = fullfile(saveDir, 'criticality_reach_d2_halfsession_center_win_growth.png');
    end
    sgtitle(tlHalf, sgTitleStr, 'Interpreter', 'none', 'FontSize', 11);
    exportgraphics(gcf, outPngHalf, 'Resolution', 300);
    fprintf('Saved figure: %s\n', outPngHalf);
end


fprintf('\n=== criticality_reach_d2_vs_windowSize: done ===\n');

%% -------------------------------------------------------------------------
function [startIdx, endIdx] = window_bin_indices_from_center_time( ...
    centerTimeSec, windowSizeSec, binSizeSec, numTimePoints)
% WINDOW_BIN_INDICES_FROM_CENTER_TIME
%
% Variables:
%   centerTimeSec   - window center (s), aligned to reach onset here
%   windowSizeSec   - full window width (s)
%   binSizeSec      - bin width (s)
%   numTimePoints   - number of time bins in the binned matrix
%
% Goal:
%   Map a centered time window to 1-based row indices into binned spike data,
%   matching calculate_window_indices_from_center.m without clamping. If the
%   centered window would extend outside [1, numTimePoints], return empty
%   startIdx.
%
% Returns:
%   startIdx, endIdx - inclusive row indices, or startIdx = [] if invalid

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
%   wPopActivity - population mean firing vector [timeBins x 1] (or column)
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

function cvProp = population_trace_cv(wPopActivity)
% POPULATION_TRACE_CV Coefficient of variation (CV) for a popActivity trace
%
% Variables:
%   wPopActivity - population mean firing vector over time bins (column vector)
%
% Goal:
%   Return CV = nanstd(x) / |nanmean(x)| (dimensionless proportion). NaN if the
%   mean is undefined or too close to zero (|mean| < 1e-12).
%
% Returns:
%   cvProp - scalar CV (proportion)

    cvProp = nan;
    if isempty(wPopActivity)
        return;
    end
    x = double(wPopActivity(:));
    meanVal = nanmean(x);
    if isnan(meanVal) || abs(meanVal) < 1e-12
        return;
    end
    cvProp = nanstd(x) / abs(meanVal);
end
