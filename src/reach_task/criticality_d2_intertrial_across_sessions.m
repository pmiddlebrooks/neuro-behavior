%% criticality_d2_intertrial_across_sessions.m
% Across-session analysis: d2 from population activity in windows centered on
% each inter-trial midpoint vs. ITI duration (time between consecutive reaches).
%
% Sessions are taken from reach_session_list.m. For each session, loads spikes
% via load_sliding_window_data (same pattern as criticality_reach_intertrial_d2.m).
%
% ITI window modes (itiWindowMode):
%   'fixed'     — use fixedWindowSizeSeconds for every kept ITI; ITIs whose
%                 maximum admissible window (given windowBuffer) is smaller are dropped.
%   'maximize'  — for each ITI, use the largest window centered on the midpoint
%                 that respects buffer distance from the flanking reaches.
%
% Variables (main configuration block):
%   sessionType          — passed to load_sliding_window_data (default 'reach')
%   itiWindowMode        — 'fixed' or 'maximize'
%   fixedWindowSizeSeconds — window length when itiWindowMode is 'fixed'
%   windowBuffer         — min time from window edge to adjacent reach (s)
%   maxItiSeconds        — drop ITIs strictly greater than this for analysis and plots (Inf = off)
%   plotAreaIdx          — index into areas list for d2 vs ITI (must be in areasToTest)
%   pOrder, critType     — d2 / AR settings (same as template)
%   normalizeD2, nShuffles — optional shuffle normalization
%   useOptimalBinWindowFunction, binSizeManual — bin sizing (same as template)
%   figureId             — integer figure Number; same ID reuses one window each run

%% Configuration
sessionType = 'reach';
itiWindowMode = 'fixed';  % 'fixed' | 'maximize'
fixedWindowSizeSeconds = 6;
windowBuffer = 1;
maxItiSeconds = 20;  % e.g. 30 to cap ITI; Inf keeps all ITIs

pOrder = 10;
critType = 2;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;
nShuffles = 8;
normalizeD2 = false;

useOptimalBinWindowFunction = false;
binSizeManual = 0.025;

pcaFlag = 0;

plotAreaIdx = [2];  % empty = first entry of areasToTest after load

optsLoad = neuro_behavior_options;
optsLoad.frameSize = .001;
optsLoad.firingRateCheckTime = 5 * 60;
optsLoad.collectStart = 0;
optsLoad.collectEnd = [];
optsLoad.minFiringRate = .1;
optsLoad.maxFiringRate = 100;

nScatterCols = 4;
figureId = 901773;  % change if this clashes with another script's dedicated figure

% Paths
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, '..', 'sliding_window_prep', 'utils'));

%% Session list
sessionList = reach_session_list();
nSessions = numel(sessionList);

itiCell = cell(1, nSessions);
d2Cell = cell(1, nSessions);
sessionLabelCell = cell(1, nSessions);
regressionStats = struct();
regressionStats.slope = nan(1, nSessions);
regressionStats.intercept = nan(1, nSessions);
regressionStats.pSlope = nan(1, nSessions);
regressionStats.rSquared = nan(1, nSessions);
regressionStats.nPoints = zeros(1, nSessions);
regressionStats.errorMessage = cell(1, nSessions);

fprintf('\n=== criticality_d2_intertrial_across_sessions ===\n');
fprintf('Sessions: %d | mode: %s | max ITI: %s\n', nSessions, itiWindowMode, ...
    ternary_string(isfinite(maxItiSeconds), sprintf('%.3g s', maxItiSeconds), 'none (Inf)'));

for sessionIdx = 1:nSessions
    sessionName = sessionList{sessionIdx};
    regressionStats.errorMessage{sessionIdx} = '';
    fprintf('\n--- Session %d/%d: %s ---\n', sessionIdx, nSessions, sessionName);

    try
        dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
            'sessionName', sessionName, 'opts', optsLoad);

        if ~isfield(dataStruct, 'dataR')
            error('dataStruct.dataR missing for session %s', sessionName);
        end

        dataR = dataStruct.dataR;
        reachStart = dataR.R(:, 1) / 1000;
        nReaches = numel(reachStart);

        areas = dataStruct.areas;
        numAreas = numel(areas);
        if isfield(dataStruct, 'areasToTest')
            areasToTest = dataStruct.areasToTest;
        else
            areasToTest = 1:numAreas;
        end

        if isempty(plotAreaIdx)
            areaIdx = areasToTest(1);
        else
            if ~ismember(plotAreaIdx, areasToTest)
                error('plotAreaIdx %d not in areasToTest for %s', plotAreaIdx, sessionName);
            end
            areaIdx = plotAreaIdx;
        end

        if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
            timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
        else
            timeRange = [0, max(dataStruct.spikeTimes)];
        end

        binSize = zeros(1, numAreas);
        if useOptimalBinWindowFunction
            slidingWindowSizeOptimal = nan(1, numAreas);
            for a = areasToTest
                neuronIDs = dataStruct.idLabel{a};
                thisFiringRate = calculate_firing_rate_from_spikes( ...
                    dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                    neuronIDs, timeRange);
                [binSize(a), slidingWindowSizeOptimal(a)] = ...
                    find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, minBinsPerWindow);
            end
            validOptimalWindows = slidingWindowSizeOptimal(areasToTest);
            validOptimalWindows = validOptimalWindows(~isnan(validOptimalWindows) & validOptimalWindows > 0);
            if ~isempty(validOptimalWindows)
                refWindowForMode = min(validOptimalWindows);
            else
                refWindowForMode = fixedWindowSizeSeconds;
            end
        else
            if isempty(binSizeManual) || ~isscalar(binSizeManual) || binSizeManual <= 0
                error('binSizeManual must be a positive scalar.');
            end
            binSize(:) = binSizeManual;
            refWindowForMode = fixedWindowSizeSeconds;
        end

        neuronIDs = dataStruct.idLabel{areaIdx};
        aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
            neuronIDs, timeRange, binSize(areaIdx));
        numTimePoints = size(aDataMat, 1);

        if pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, min(6, forDim));
            nDimUse = 1:forDim;
            aDataMat = score(:, nDimUse) * coeff(:, nDimUse)' + mu;
        end

        itiList = [];
        d2List = [];

        for itiIdx = 1:nReaches - 1
            prevReach = reachStart(itiIdx);
            nextReach = reachStart(itiIdx + 1);
            midpointTime = (prevReach + nextReach) / 2;
            itiDuration = nextReach - prevReach;

            maxWin = intertrial_max_window_seconds(midpointTime, prevReach, nextReach, windowBuffer);
            if ~(maxWin > 0 && isfinite(maxWin))
                continue;
            end

            if strcmpi(itiWindowMode, 'fixed')
                windowSizeSeconds = fixedWindowSizeSeconds;
                if maxWin < windowSizeSeconds - 1e-9
                    continue;
                end
            elseif strcmpi(itiWindowMode, 'maximize')
                windowSizeSeconds = maxWin;
            else
                error('Unknown itiWindowMode: %s', itiWindowMode);
            end

            winStart = midpointTime - windowSizeSeconds / 2;
            winEnd = midpointTime + windowSizeSeconds / 2;
            if winStart < prevReach + windowBuffer - 1e-9 || winEnd > nextReach - windowBuffer + 1e-9
                continue;
            end

            [startIdx, endIdx] = calculate_window_indices_from_center(midpointTime, ...
                windowSizeSeconds, binSize(areaIdx), numTimePoints);
            if startIdx < 1 || endIdx > numTimePoints || endIdx <= startIdx
                continue;
            end

            wDataMat = aDataMat(startIdx:endIdx, :);
            d2Val = compute_d2_for_window_data_mat(wDataMat, pOrder, critType, normalizeD2, nShuffles);
            if isnan(d2Val)
                continue;
            end

            itiList(end + 1) = itiDuration; %#ok<AGROW>
            d2List(end + 1) = d2Val; %#ok<AGROW>
        end

        if isfinite(maxItiSeconds)
            keepItiMask = itiList(:) <= maxItiSeconds;
            itiList = itiList(keepItiMask);
            d2List = d2List(keepItiMask);
        end

        itiCell{sessionIdx} = itiList(:);
        d2Cell{sessionIdx} = d2List(:);
        sessionLabelCell{sessionIdx} = sprintf('%s (%s)', sessionName, areas{areaIdx});

        [slope, intercept, pSlope, rSquared, nPts] = linear_regression_inference(itiList, d2List);
        regressionStats.slope(sessionIdx) = slope;
        regressionStats.intercept(sessionIdx) = intercept;
        regressionStats.pSlope(sessionIdx) = pSlope;
        regressionStats.rSquared(sessionIdx) = rSquared;
        regressionStats.nPoints(sessionIdx) = nPts;
        fprintf('  Kept ITIs: %d | area %s | refWindow (optimal path): %.3f s\n', ...
            numel(itiList), areas{areaIdx}, refWindowForMode);

    catch ME
        warning('Session %s failed: %s', sessionName, ME.message);
        regressionStats.errorMessage{sessionIdx} = ME.message;
        itiCell{sessionIdx} = [];
        d2Cell{sessionIdx} = [];
        sessionLabelCell{sessionIdx} = sessionName;
    end
end

%%  Figure:
% Shared axis limits (all subplots)
allItiForLim = [];
allD2ForLim = [];
for limIdx = 1:nSessions
    allItiForLim = [allItiForLim; itiCell{limIdx}(:)]; %#ok<AGROW>
    allD2ForLim = [allD2ForLim; d2Cell{limIdx}(:)]; %#ok<AGROW>
end
if isempty(allItiForLim)
    if isfinite(maxItiSeconds)
        xLimShared = [fixedWindowSizeSeconds, maxItiSeconds];
    else
        xLimShared = [0, 1];
    end
    yLimShared = [0, 1];
else
    if isfinite(maxItiSeconds)
        xLimShared = [fixedWindowSizeSeconds, maxItiSeconds];
    else
        spanX = max(allItiForLim) - min(allItiForLim);
        padX = 0.05 * (spanX + eps);
        xLimShared = [min(allItiForLim) - padX, max(allItiForLim) + padX];
    end
    yCandidates = allD2ForLim(:);
    for limIdx = 1:nSessions
        nPtsLim = regressionStats.nPoints(limIdx);
        if nPtsLim >= 3 && all(isfinite([regressionStats.slope(limIdx), regressionStats.intercept(limIdx)]))
            yAtEnds = regressionStats.intercept(limIdx) + regressionStats.slope(limIdx) * xLimShared(:);
            yCandidates = [yCandidates; yAtEnds(:)]; %#ok<AGROW>
        end
    end
    spanY = max(yCandidates) - min(yCandidates);
    if spanY < 1e-12
        yMid = mean(yCandidates);
        yLimShared = [yMid - 1, yMid + 1];
    else
        padY = 0.05 * spanY;
        yLimShared = [min(yCandidates) - padY, max(yCandidates) + padY];
    end
end

% Figure: one scatter per session (4 columns)
nRows = ceil(nSessions / nScatterCols);
figHandle = figure(figureId);
clf(figHandle);
set(figHandle, 'Name', 'd2 vs ITI (intertrial midpoint windows)', 'Color', 'w', ...
    'NumberTitle', 'off');
apply_figure_full_screen(figHandle);
nPanels = nRows * nScatterCols;
useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(nRows, nScatterCols, [0.08 0.04], [0.1 0.1], [0.08 0.04]);
else
    warning('tight_subplot not on path; using subplot.');
    ha = gobjects(nPanels, 1);
    for panelIdx = 1:nPanels
        ha(panelIdx) = subplot(nRows, nScatterCols, panelIdx);
    end
end

for sessionIdx = 1:nSessions
    axPanel = ha(sessionIdx);
    axes(axPanel);
    hold on;

    itiList = itiCell{sessionIdx};
    d2List = d2Cell{sessionIdx};
    if isempty(itiList)
        title(sessionLabelCell{sessionIdx}, 'Interpreter', 'none');
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center', 'Units', 'normalized');
        axis on;
        xlim(xLimShared);
        ylim(yLimShared);
        xlabel('ITI (s)');
        ylabel('d2');
        grid on;
        make_axis_tick_labels_visible(axPanel);
        hold off;
        continue;
    end

    scatter(itiList, d2List, 22, 'o', 'MarkerEdgeColor', [0 0.45 0.74], ...
        'MarkerFaceColor', 'none', 'LineWidth', 0.75);

    nPts = regressionStats.nPoints(sessionIdx);
    if nPts >= 3 && all(isfinite([regressionStats.slope(sessionIdx), regressionStats.intercept(sessionIdx)]))
        xFit = linspace(xLimShared(1), xLimShared(2), 100);
        yFit = regressionStats.intercept(sessionIdx) + regressionStats.slope(sessionIdx) * xFit;
        plot(xFit, yFit, 'r-', 'LineWidth', 1.2);
    end

    xlabel('ITI (s)');
    ylabel('d2');
    grid on;
    xlim(xLimShared);
    ylim(yLimShared);
    pStr = 'n/a';
    if nPts >= 3 && ~isnan(regressionStats.pSlope(sessionIdx))
        if regressionStats.pSlope(sessionIdx) < 0.001
            pStr = 'p < 0.001';
        else
            pStr = sprintf('p = %.3g', regressionStats.pSlope(sessionIdx));
        end
    end
    statStr = sprintf('n=%d, R^2=%.2f, %s', nPts, regressionStats.rSquared(sessionIdx), pStr);
    title({sessionLabelCell{sessionIdx}, statStr}, 'Interpreter', 'none', 'FontSize', 8);
    make_axis_tick_labels_visible(axPanel);
    hold off;
end

if isfinite(maxItiSeconds)
    sgtitle(sprintf('d2 vs ITI | mode: %s | buffer: %.2f s | ITI \\leq %.3g s', ...
        itiWindowMode, windowBuffer, maxItiSeconds));
else
    sgtitle(sprintf('d2 vs ITI | mode: %s | buffer: %.2f s', itiWindowMode, windowBuffer));
end

%% Console summary
fprintf('\n=== Linear regression (d2 ~ ITI), per session ===\n');
for sessionIdx = 1:nSessions
    if ~isempty(regressionStats.errorMessage{sessionIdx})
        fprintf('%s: ERROR %s\n', sessionList{sessionIdx}, regressionStats.errorMessage{sessionIdx});
        continue;
    end
    fprintf(['%s: n=%d, slope=%.4g, intercept=%.4g, R^2=%.3f, p(slope)=%.4g %s\n'], ...
        sessionList{sessionIdx}, regressionStats.nPoints(sessionIdx), ...
        regressionStats.slope(sessionIdx), regressionStats.intercept(sessionIdx), ...
        regressionStats.rSquared(sessionIdx), regressionStats.pSlope(sessionIdx), ...
        ternary_string(~isnan(regressionStats.pSlope(sessionIdx)) && regressionStats.pSlope(sessionIdx) < 0.05, '*', ''));
end

%% Local functions
function maxWin = intertrial_max_window_seconds(midpointTime, prevReach, nextReach, windowBuffer)
% intertrial_max_window_seconds
% Variables:
%   midpointTime - time (s) halfway between the two reaches
%   prevReach, nextReach - reach times (s) bounding this ITI
%   windowBuffer - required clearance (s) from window edge to each reach
% Goal:
%   Largest duration (s) of a window centered at midpointTime that stays at
%   least windowBuffer away from prevReach and nextReach.

    maxFromPrev = 2 * (midpointTime - prevReach - windowBuffer);
    maxFromNext = 2 * (nextReach - midpointTime - windowBuffer);
    maxWin = min(maxFromPrev, maxFromNext);
end

function d2Val = compute_d2_for_window_data_mat(wDataMat, pOrder, critType, normalizeD2, nShuffles)
% compute_d2_for_window_data_mat
% Variables:
%   wDataMat - [timeBins x neurons] spike counts or activity in one window
%   pOrder, critType - passed to myYuleWalker3 / getFixedPointDistance2
%   normalizeD2 - if true, divide d2 by mean of shuffled-surrogate d2
%   nShuffles - number of circular-shift surrogates per neuron
% Goal:
%   Scalar d2 (optionally normalized) for mean population activity in the window.

    wPopActivity = mean(wDataMat, 2);
    d2Val = nan;
    if isempty(wPopActivity)
        return;
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
        d2Val = nan;
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
        permutedPopActivity = mean(permutedDataMat, 2);
        try
            [varphiPerm, ~] = myYuleWalker3(double(permutedPopActivity), pOrder);
            d2Shuffled(shuffleIdx) = getFixedPointDistance2(pOrder, critType, varphiPerm);
        catch
            d2Shuffled(shuffleIdx) = nan;
        end
    end
    meanShuffled = nanmean(d2Shuffled);
    if ~isnan(meanShuffled) && meanShuffled > 0
        d2Val = d2Raw / meanShuffled;
    else
        d2Val = nan;
    end
end

function [slope, intercept, pSlope, rSquared, nPts] = linear_regression_inference(xVec, yVec)
% linear_regression_inference
% Variables:
%   xVec, yVec - paired observations (ITI and d2)
% Goal:
%   Ordinary least squares of y ~ 1 + x; two-sided t-test on slope; R^2.
% Returns:
%   slope, intercept, pSlope, rSquared, nPts (finite pairs)

    slope = nan;
    intercept = nan;
    pSlope = nan;
    rSquared = nan;
    nPts = 0;

    xAll = xVec(:);
    yAll = yVec(:);
    validMask = ~isnan(xAll) & ~isnan(yAll);
    xCol = xAll(validMask);
    yCol = yAll(validMask);
    nPts = numel(xCol);
    if nPts < 3
        return;
    end

    XDesign = [ones(nPts, 1), xCol];
    coeffs = XDesign \ yCol;
    intercept = coeffs(1);
    slope = coeffs(2);

    yHat = XDesign * coeffs;
    residuals = yCol - yHat;
    ssRes = sum(residuals .^ 2);
    ssTot = sum((yCol - mean(yCol)) .^ 2);
    if ssTot > 0
        rSquared = 1 - ssRes / ssTot;
    else
        rSquared = nan;
    end

    mse = ssRes / (nPts - 2);
    sxx = sum((xCol - mean(xCol)) .^ 2);
    if sxx <= 0 || mse < 0
        pSlope = nan;
        return;
    end
    seSlope = sqrt(mse / sxx);
    tStat = slope / seSlope;
    pSlope = 2 * (1 - tcdf(abs(tStat), nPts - 2));
end

function outStr = ternary_string(condition, trueStr, falseStr)
% ternary_string — small helper for fprintf suffix (significant slope marker).

    if condition
        outStr = trueStr;
    else
        outStr = falseStr;
    end
end

function make_axis_tick_labels_visible(axPanel)
% make_axis_tick_labels_visible
% Variables:
%   axPanel - axes handle for one subplot (e.g. from tight_subplot)
% Goal:
%   Restore x/y tick labels; tight_subplot often clears inner-panel labels.

    set(axPanel, 'XTickLabelMode', 'auto', 'YTickLabelMode', 'auto');
    if isprop(axPanel, 'XTickLabelVisible')
        set(axPanel, 'XTickLabelVisible', 'on', 'YTickLabelVisible', 'on');
    end
end

function apply_figure_full_screen(figHandle)
% apply_figure_full_screen
% Variables:
%   figHandle - figure handle
% Goal:
%   Maximize to fill the screen (WindowState if available; else pixel ScreenSize).

    set(figHandle, 'Visible', 'on');
    if isprop(figHandle, 'WindowState')
        set(figHandle, 'WindowState', 'maximized');
    else
        oldUnits = get(figHandle, 'Units');
        set(figHandle, 'Units', 'pixels');
        screenSize = get(0, 'ScreenSize');
        set(figHandle, 'Position', screenSize);
        set(figHandle, 'Units', oldUnits);
    end
end
