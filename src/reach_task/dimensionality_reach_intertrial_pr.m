%%
% Dimensionality Reach vs Intertrial Participation Ratio Analysis
% Compares participation ratio (PR) between reaches and intertrial intervals.
% Performs sliding window PR analysis around reach starts and intertrial midpoints.
%
% Variables:
%   reachStart - Reach start times in seconds
%   intertrialMidpoints - Midpoint times between consecutive reaches
%   windowSizeNeuronMultiple - Per-area window size = this * nNeurons * binSize (seconds)
%   windowBuffer - Minimum distance from window edge to event/midpoint
%   beforeAlign - Start of sliding window range (seconds before alignment point)
%   afterAlign - End of sliding window range (seconds after alignment point)
%   stepSize - Step size for sliding window (seconds)
%   nShuffles - Number of circular permutations for PR normalization
%   normalizePR - Set to true to normalize PR by shuffled PR values (default: true)
%
% This script uses load_sliding_window_data() to load data (as in choose_task_and_session.m)
% and finds optimal bin sizes per area (as in participation_ratio_analysis.m).
%
% Window sizing: Per-area window = windowSizeNeuronMultiple * nNeurons * binSize
% (ensures enough time bins for reliable covariance estimation).
%
% Normalization: PR values are normalized by circularly permuting each neuron's
% activity independently within each window, then computing PR on the permuted
% data. The real PR is divided by the mean of shuffled PR values.

%% =============================    Data Loading    =============================
fprintf('\n=== Loading Reach Data ===\n');

if ~exist('sessionType', 'var')
    error('sessionType must be defined. Run choose_task_and_session.m first or set sessionType in workspace.');
end

if ~exist('sessionName', 'var')
    error('sessionName must be defined. Run choose_task_and_session.m first or set sessionName in workspace.');
end

opts = neuro_behavior_options;
opts.frameSize = .001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = .1;
opts.maxFiringRate = 100;

% Load data using load_sliding_window_data
dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

if ~isfield(dataStruct, 'dataR')
    error('dataR must be available in dataStruct for reach data. Check load_reach_data() implementation.');
end

dataR = dataStruct.dataR;
reachStart = dataR.R(:,1) / 1000;  % Convert from ms to seconds
totalReaches = length(reachStart);

fprintf('Loaded %d reaches\n', totalReaches);

areas = dataStruct.areas;
idMatIdx = dataStruct.idMatIdx;
numAreas = length(areas);

if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

% Define engagement segments for this session
paths = get_paths;
reachDataFile = fullfile(paths.reachDataPath, [sessionName, '.mat']);
if ~exist(reachDataFile, 'file')
    reachDataFile = fullfile(paths.reachDataPath, sessionName);
end

if exist(reachDataFile, 'file')
    segmentOpts = struct();
    segmentWindowsEng = reach_task_engagement(reachDataFile, segmentOpts);
    segmentNames = {'Block1 Eng', 'Block1 Not', 'Block2 Eng', 'Block2 Not'};
    segmentWindowsList = {
        segmentWindowsEng.block1EngagedWindow;
        segmentWindowsEng.block1NotEngagedWindow;
        segmentWindowsEng.block2EngagedWindow;
        segmentWindowsEng.block2NotEngagedWindow;
    };
    nSegments = numel(segmentNames);
else
    warning('Reach data file not found for engagement analysis. Skipping segment analysis.');
    segmentNames = {};
    segmentWindowsList = {};
    nSegments = 0;
end

% Calculate intertrial midpoints
intertrialMidpoints = nan(1, totalReaches - 1);
for i = 1:totalReaches - 1
    intertrialMidpoints(i) = (reachStart(i) + reachStart(i+1)) / 2;
end
fprintf('Calculated %d intertrial midpoints\n', length(intertrialMidpoints));

% =============================    Configuration    =============================
beforeAlign = -2;
afterAlign  =  2;
stepSize = .1;
windowBuffer = .5;

% PR analysis parameters (per-area window sizing like participation_ratio_analysis)
windowSizeNeuronMultiple = 10;  % Per-area window = this * nNeurons * binSize
minSpikesPerBin = 3;
minBinsPerWindow = 1000;
nShuffles = 3;
normalizePR = true;

% Plotting options
loadResultsForPlotting = false;
resultsFileForPlotting = '';
makePlots = true;

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end

% =============================    Find Optimal Bin Sizes and Per-Area Window Sizes    =============================
fprintf('\n=== Finding Optimal Bin Sizes and Window Sizes Per Area ===\n');

binSize = zeros(1, numAreas);
slidingWindowSize = zeros(1, numAreas);

for a = areasToTest
    neuronIDs = dataStruct.idLabel{a};
    nNeurons = length(dataStruct.idMatIdx{a});
    thisFiringRate = calculate_firing_rate_from_spikes(...
        dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange);
    [binSize(a), ~] = find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, minBinsPerWindow);
    slidingWindowSize(a) = windowSizeNeuronMultiple * nNeurons * binSize(a);
    fprintf('Area %s: bin size = %.3f s, win = %.1f s (n=%d, multiple=%d)\n', ...
        areas{a}, binSize(a), slidingWindowSize(a), nNeurons, windowSizeNeuronMultiple);
end

minWindowSize = min(slidingWindowSize(areasToTest));

% =============================    Find Valid Reaches and Intertrial Midpoints    =============================
fprintf('\n=== Finding Valid Reaches and Intertrial Midpoints ===\n');

validReachIndices = [];
maxWindowPerReach = nan(1, totalReaches);

for r = 1:totalReaches
    reachTime = reachStart(r);
    maxWindowForThisReach = inf;

    if r > 1
        prevMidpoint = intertrialMidpoints(r-1);
        maxWindowFromPrev = 2 * (reachTime - prevMidpoint - windowBuffer);
        if maxWindowFromPrev < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromPrev;
        end
    end

    if r <= length(intertrialMidpoints)
        nextMidpoint = intertrialMidpoints(r);
        maxWindowFromNext = 2 * (nextMidpoint - reachTime - windowBuffer);
        if maxWindowFromNext < maxWindowForThisReach
            maxWindowForThisReach = maxWindowFromNext;
        end
    end

    maxWindowPerReach(r) = maxWindowForThisReach;
    if maxWindowForThisReach >= minWindowSize
        validReachIndices = [validReachIndices, r];  %#ok<AGROW>
    end
end

validIntertrialIndices = [];
maxWindowPerIntertrial = nan(1, length(intertrialMidpoints));

for i = 1:length(intertrialMidpoints)
    midpointTime = intertrialMidpoints(i);
    prevReach = reachStart(i);
    nextReach = reachStart(i+1);

    maxWindowFromPrev = 2 * (midpointTime - prevReach - windowBuffer);
    maxWindowFromNext = 2 * (nextReach - midpointTime - windowBuffer);
    maxWindowForThisIntertrial = min(maxWindowFromPrev, maxWindowFromNext);
    maxWindowPerIntertrial(i) = maxWindowForThisIntertrial;

    if maxWindowForThisIntertrial >= minWindowSize
        validIntertrialIndices = [validIntertrialIndices, i];  %#ok<AGROW>
    end
end

if isempty(validReachIndices) && isempty(validIntertrialIndices)
    error('No valid reaches or intertrial midpoints found. Try reducing windowBuffer or windowSizeNeuronMultiple.');
end

reachStartOriginal = reachStart;
intertrialMidpointsOriginal = intertrialMidpoints;

reachStart = reachStart(validReachIndices);
totalReaches = length(reachStart);

% Recalculate intertrial midpoints from filtered reachStart
intertrialMidpointsFiltered = nan(1, totalReaches - 1);
intertrialMidpointValid = false(1, totalReaches - 1);

for i = 1:totalReaches - 1
    midpointTime = (reachStart(i) + reachStart(i+1)) / 2;
    origReachIdx1 = validReachIndices(i);
    origReachIdx2 = validReachIndices(i+1);

    if origReachIdx2 == origReachIdx1 + 1 && ismember(origReachIdx1, validIntertrialIndices)
        intertrialMidpointsFiltered(i) = midpointTime;
        intertrialMidpointValid(i) = true;
    end
end

intertrialMidpoints = intertrialMidpointsFiltered(intertrialMidpointValid);
validIntertrialIndicesFiltered = find(intertrialMidpointValid);

fprintf('Min window size: %.2f s\n', minWindowSize);
fprintf('Valid reaches: %d/%d\n', length(validReachIndices), length(maxWindowPerReach));
fprintf('Valid intertrial midpoints: %d/%d\n', length(intertrialMidpoints), length(maxWindowPerIntertrial));
fprintf('Window buffer: %.2f s\n', windowBuffer);

%% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

prMetrics = struct();
prMetrics.reach = cell(1, numAreas);
prMetrics.intertrial = cell(1, numAreas);
prMetrics.reachNormalized = cell(1, numAreas);
prMetrics.intertrialNormalized = cell(1, numAreas);

for a = areasToTest
    prMetrics.reach{a} = nan(1, numSlidingPositions);
    prMetrics.intertrial{a} = nan(1, numSlidingPositions);
    prMetrics.reachNormalized{a} = nan(1, numSlidingPositions);
    prMetrics.intertrialNormalized{a} = nan(1, numSlidingPositions);
end

% Initialize storage for collected window data
collectedReachWindowData = cell(numAreas, numSlidingPositions);
collectedIntertrialWindowData = cell(numAreas, numSlidingPositions);
collectedReachWindowCenters = cell(numAreas, numSlidingPositions);
collectedIntertrialWindowCenters = cell(numAreas, numSlidingPositions);

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    tic;

    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);
    winSizeA = slidingWindowSize(a);

    % Collect reach-aligned windows
    fprintf('  Collecting reach-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);

        for r = 1:totalReaches
            reachTime = reachStart(r);
            origReachIdx = validReachIndices(r);

            prevMidpoint = [];
            nextMidpoint = [];
            if origReachIdx > 1
                prevOrigMidpointIdx = origReachIdx - 1;
                if prevOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                    prevMidpoint = intertrialMidpointsOriginal(prevOrigMidpointIdx);
                end
            end
            if origReachIdx <= length(intertrialMidpointsOriginal)
                nextOrigMidpointIdx = origReachIdx;
                if nextOrigMidpointIdx <= length(intertrialMidpointsOriginal)
                    nextMidpoint = intertrialMidpointsOriginal(nextOrigMidpointIdx);
                end
            end

            windowCenter = reachTime + offset;
            winStart = windowCenter - winSizeA / 2;
            winEnd = windowCenter + winSizeA / 2;

            constraintViolated = false;
            if ~isempty(prevMidpoint) && winStart < prevMidpoint + windowBuffer
                constraintViolated = true;
            end
            if ~isempty(nextMidpoint) && winEnd > nextMidpoint - windowBuffer
                constraintViolated = true;
            end

            if constraintViolated
                continue;
            end

            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, winSizeA, binSize(a), numTimePoints);

            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);

                if isempty(collectedReachWindowData{a, posIdx})
                    collectedReachWindowData{a, posIdx} = {wDataMat};
                    collectedReachWindowCenters{a, posIdx} = windowCenter;
                else
                    collectedReachWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedReachWindowCenters{a, posIdx} = [collectedReachWindowCenters{a, posIdx}, windowCenter];
                end
            end
        end
    end

    % Collect intertrial-aligned windows
    fprintf('  Collecting intertrial-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);

        for idx = 1:length(intertrialMidpoints)
            midpointTime = intertrialMidpoints(idx);
            i = validIntertrialIndicesFiltered(idx);
            prevReach = reachStart(i);
            nextReach = reachStart(i+1);

            windowCenter = midpointTime + offset;
            winStart = windowCenter - winSizeA / 2;
            winEnd = windowCenter + winSizeA / 2;

            if winStart < prevReach + windowBuffer || winEnd > nextReach - windowBuffer
                continue;
            end

            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, winSizeA, binSize(a), numTimePoints);

            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);

                if isempty(collectedIntertrialWindowData{a, posIdx})
                    collectedIntertrialWindowData{a, posIdx} = {wDataMat};
                    collectedIntertrialWindowCenters{a, posIdx} = windowCenter;
                else
                    collectedIntertrialWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedIntertrialWindowCenters{a, posIdx} = [collectedIntertrialWindowCenters{a, posIdx}, windowCenter];
                end
            end
        end
    end

    fprintf('  Area %s completed in %.1f min\n', areas{a}, toc/60);
end

% =============================    Per-Sliding-Position PR Analysis    =============================
fprintf('\n=== Computing PR Per Sliding Position (Reach vs Intertrial) ===\n');

perWindowPR = struct();
perWindowPR.reach = cell(1, numAreas);
perWindowPR.intertrial = cell(1, numAreas);
perWindowPR.reachNormalized = cell(1, numAreas);
perWindowPR.intertrialNormalized = cell(1, numAreas);
perWindowPR.reachCenters = cell(1, numAreas);
perWindowPR.intertrialCenters = cell(1, numAreas);

for a = areasToTest
    perWindowPR.reach{a} = cell(1, numSlidingPositions);
    perWindowPR.intertrial{a} = cell(1, numSlidingPositions);
    perWindowPR.reachNormalized{a} = cell(1, numSlidingPositions);
    perWindowPR.intertrialNormalized{a} = cell(1, numSlidingPositions);
    perWindowPR.reachCenters{a} = cell(1, numSlidingPositions);
    perWindowPR.intertrialCenters{a} = cell(1, numSlidingPositions);
end

for a = areasToTest
    fprintf('\nComputing PR for area %s...\n', areas{a});

    for posIdx = 1:numSlidingPositions
        % Reach windows
        if ~isempty(collectedReachWindowData{a, posIdx})
            windowDataList = collectedReachWindowData{a, posIdx};
            windowCenters = collectedReachWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);

            prPerWindow = nan(1, numWindows);
            prShuffledPerWindow = nan(numWindows, nShuffles);

            for w = 1:numWindows
                wDataMat = windowDataList{w};
                prPerWindow(w) = compute_participation_ratio(wDataMat);

                if normalizePR
                    numNeurons = size(wDataMat, 2);
                    numTimeBins = size(wDataMat, 1);
                    for s = 1:nShuffles
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        prShuffledPerWindow(w, s) = compute_participation_ratio(permutedDataMat);
                    end
                end
            end

            prMetrics.reach{a}(posIdx) = nanmean(prPerWindow);
            perWindowPR.reach{a}{posIdx} = prPerWindow;
            perWindowPR.reachCenters{a}{posIdx} = windowCenters;

            if normalizePR
                meanShuffledPerWindow = nanmean(prShuffledPerWindow, 2);
                prNormalizedPerWindow = nan(1, numWindows);
                for w = 1:numWindows
                    if ~isnan(prPerWindow(w)) && ~isnan(meanShuffledPerWindow(w)) && meanShuffledPerWindow(w) > 0
                        prNormalizedPerWindow(w) = prPerWindow(w) / meanShuffledPerWindow(w);
                    end
                end
                prMetrics.reachNormalized{a}(posIdx) = nanmean(prNormalizedPerWindow);
                perWindowPR.reachNormalized{a}{posIdx} = prNormalizedPerWindow;
            else
                prMetrics.reachNormalized{a}(posIdx) = nan;
            end
        end

        % Intertrial windows
        if ~isempty(collectedIntertrialWindowData{a, posIdx})
            windowDataList = collectedIntertrialWindowData{a, posIdx};
            windowCenters = collectedIntertrialWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);

            prPerWindow = nan(1, numWindows);
            prShuffledPerWindow = nan(numWindows, nShuffles);

            for w = 1:numWindows
                wDataMat = windowDataList{w};
                prPerWindow(w) = compute_participation_ratio(wDataMat);

                if normalizePR
                    numNeurons = size(wDataMat, 2);
                    numTimeBins = size(wDataMat, 1);
                    for s = 1:nShuffles
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        prShuffledPerWindow(w, s) = compute_participation_ratio(permutedDataMat);
                    end
                end
            end

            prMetrics.intertrial{a}(posIdx) = nanmean(prPerWindow);
            perWindowPR.intertrial{a}{posIdx} = prPerWindow;
            perWindowPR.intertrialCenters{a}{posIdx} = windowCenters;

            if normalizePR
                meanShuffledPerWindow = nanmean(prShuffledPerWindow, 2);
                prNormalizedPerWindow = nan(1, numWindows);
                for w = 1:numWindows
                    if ~isnan(prPerWindow(w)) && ~isnan(meanShuffledPerWindow(w)) && meanShuffledPerWindow(w) > 0
                        prNormalizedPerWindow(w) = prPerWindow(w) / meanShuffledPerWindow(w);
                    end
                end
                prMetrics.intertrialNormalized{a}(posIdx) = nanmean(prNormalizedPerWindow);
                perWindowPR.intertrialNormalized{a}{posIdx} = prNormalizedPerWindow;
            else
                prMetrics.intertrialNormalized{a}(posIdx) = nan;
            end
        end
    end
end

% =============================    Engagement Segment-level PR Analysis    =============================
segmentMetrics = struct();
segmentMetrics.reach = cell(numAreas, nSegments);
segmentMetrics.intertrial = cell(numAreas, nSegments);
segmentMetrics.reachNormalized = cell(numAreas, nSegments);
segmentMetrics.intertrialNormalized = cell(numAreas, nSegments);

if nSegments > 0
    fprintf('\n=== Segment-level PR Analysis ===\n');
    alignPosIdx = find(abs(slidingPositions) < stepSize/2, 1);
    if isempty(alignPosIdx)
        alignPosIdx = 1;
    end

    for a = areasToTest
        for s = 1:nSegments
            if isempty(segmentWindowsList) || s > length(segmentWindowsList) || isempty(segmentWindowsList{s}) || any(isnan(segmentWindowsList{s}))
                segmentMetrics.reach{a, s} = nan;
                segmentMetrics.intertrial{a, s} = nan;
                segmentMetrics.reachNormalized{a, s} = nan;
                segmentMetrics.intertrialNormalized{a, s} = nan;
                continue;
            end
            segWin = segmentWindowsList{s};
            tStart = segWin(1);
            tEnd = segWin(2);

            reachPRInSegment = [];
            intertrialPRInSegment = [];
            reachPRNormInSegment = [];
            intertrialPRNormInSegment = [];

            if ~isempty(perWindowPR.reachCenters{a}{alignPosIdx})
                d2Vals = perWindowPR.reach{a}{alignPosIdx};
                d2NormVals = perWindowPR.reachNormalized{a}{alignPosIdx};
                windowCenters = perWindowPR.reachCenters{a}{alignPosIdx};
                if isscalar(windowCenters)
                    windowCenters = windowCenters(:)';
                end
                windowCenters = windowCenters(:)';
                if isscalar(d2Vals)
                    d2Vals = d2Vals(:)';
                end
                d2Vals = d2Vals(:)';
                if isscalar(d2NormVals)
                    d2NormVals = d2NormVals(:)';
                end
                d2NormVals = d2NormVals(:)';

                inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
                if any(inSegment)
                    reachPRInSegment = d2Vals(inSegment);
                    reachPRNormInSegment = d2NormVals(inSegment);
                end
            end

            if ~isempty(perWindowPR.intertrialCenters{a}{alignPosIdx})
                d2Vals = perWindowPR.intertrial{a}{alignPosIdx};
                d2NormVals = perWindowPR.intertrialNormalized{a}{alignPosIdx};
                windowCenters = perWindowPR.intertrialCenters{a}{alignPosIdx};
                if isscalar(windowCenters)
                    windowCenters = windowCenters(:)';
                end
                windowCenters = windowCenters(:)';
                if isscalar(d2Vals)
                    d2Vals = d2Vals(:)';
                end
                d2Vals = d2Vals(:)';
                if isscalar(d2NormVals)
                    d2NormVals = d2NormVals(:)';
                end
                d2NormVals = d2NormVals(:)';

                inSegment = (windowCenters >= tStart) & (windowCenters <= tEnd);
                if any(inSegment)
                    intertrialPRInSegment = d2Vals(inSegment);
                    intertrialPRNormInSegment = d2NormVals(inSegment);
                end
            end

            if ~isempty(reachPRInSegment)
                segmentMetrics.reach{a, s} = nanmean(reachPRInSegment);
            else
                segmentMetrics.reach{a, s} = nan;
            end
            if ~isempty(intertrialPRInSegment)
                segmentMetrics.intertrial{a, s} = nanmean(intertrialPRInSegment);
            else
                segmentMetrics.intertrial{a, s} = nan;
            end
            if ~isempty(reachPRNormInSegment)
                segmentMetrics.reachNormalized{a, s} = nanmean(reachPRNormInSegment);
            else
                segmentMetrics.reachNormalized{a, s} = nan;
            end
            if ~isempty(intertrialPRNormInSegment)
                segmentMetrics.intertrialNormalized{a, s} = nanmean(intertrialPRNormInSegment);
            else
                segmentMetrics.intertrialNormalized{a, s} = nan;
            end
        end
    end
end

%% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.reachStart = reachStart;
results.intertrialMidpoints = intertrialMidpoints;
results.slidingWindowSize = slidingWindowSize;  % Per-area
results.windowBuffer = windowBuffer;
results.beforeAlign = beforeAlign;
results.afterAlign = afterAlign;
results.stepSize = stepSize;
results.slidingPositions = slidingPositions;
results.prMetrics = prMetrics;
results.segmentMetrics = segmentMetrics;
results.segmentNames = segmentNames;
if exist('sessionName', 'var')
    results.sessionName = sessionName;
end
if exist('idMatIdx', 'var')
    results.idMatIdx = idMatIdx;
end
if exist('segmentWindowsEng', 'var')
    results.segmentWindows = segmentWindowsEng;
end
results.binSize = binSize;
results.params.windowSizeNeuronMultiple = windowSizeNeuronMultiple;
results.params.minSpikesPerBin = minSpikesPerBin;
results.params.minBinsPerWindow = minBinsPerWindow;
results.params.nShuffles = nShuffles;
results.params.normalizePR = normalizePR;

winMax = max(slidingWindowSize(areasToTest));
resultsPath = fullfile(saveDir, sprintf('dimensionality_reach_intertrial_pr_win%.1f_step%.2f.mat', winMax, stepSize));
save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);

%% =============================    Plotting    =============================
if ~makePlots
    fprintf('\n=== Skipping plots (makePlots = false) ===\n');
    return;
end

fprintf('\n=== Creating Summary Plots ===\n');

if loadResultsForPlotting
    fprintf('Loading saved results for plotting...\n');
    if isempty(resultsFileForPlotting)
        if ~exist('saveDir', 'var') || isempty(saveDir)
            error('saveDir not defined. Specify resultsFileForPlotting or run data loading first.');
        end
        resultsFiles = dir(fullfile(saveDir, 'dimensionality_reach_intertrial_pr_win*.mat'));
        if isempty(resultsFiles)
            error('No results files found in %s. Run analysis first.', saveDir);
        end
        [~, idx] = sort([resultsFiles.datenum], 'descend');
        resultsFileForPlotting = fullfile(saveDir, resultsFiles(idx(1)).name);
    end
    if ~exist(resultsFileForPlotting, 'file')
        error('Results file not found: %s', resultsFileForPlotting);
    end
    loadedResults = load(resultsFileForPlotting);
    results = loadedResults.results;
    areas = results.areas;
    reachStart = results.reachStart;
    intertrialMidpoints = results.intertrialMidpoints;
    slidingWindowSize = results.slidingWindowSize;
    windowBuffer = results.windowBuffer;
    beforeAlign = results.beforeAlign;
    afterAlign = results.afterAlign;
    stepSize = results.stepSize;
    slidingPositions = results.slidingPositions;
    prMetrics = results.prMetrics;
    segmentMetrics = results.segmentMetrics;
    segmentNames = results.segmentNames;
    binSize = results.binSize;
    if isfield(results, 'sessionName')
        sessionName = results.sessionName;
    end
    if isfield(results, 'idMatIdx')
        idMatIdx = results.idMatIdx;
    end
    if ~exist('saveDir', 'var') || isempty(saveDir)
        [saveDir, ~, ~] = fileparts(resultsFileForPlotting);
    end
    areasToTest = 1:length(areas);
    nSegments = numel(segmentNames);
    if isfield(results.params, 'normalizePR')
        normalizePR = results.params.normalizePR;
    end
else
    if ~exist('prMetrics', 'var')
        error('prMetrics not found. Run analysis or set loadResultsForPlotting = true.');
    end
    if ~exist('segmentMetrics', 'var')
        error('segmentMetrics not found. Run analysis or set loadResultsForPlotting = true.');
    end
    if ~exist('areasToTest', 'var') || isempty(areasToTest)
        areasToTest = 1:length(areas);
    end
end

% Y-axis limits: raw PR and normalized PR (different scales)
allYValsRaw = [];
allYValsNorm = [];
for a = areasToTest
    reachRaw = prMetrics.reach{a};
    intertrialRaw = prMetrics.intertrial{a};
    reachNorm = prMetrics.reachNormalized{a};
    intertrialNorm = prMetrics.intertrialNormalized{a};
    allYValsRaw = [allYValsRaw, reachRaw(~isnan(reachRaw)), intertrialRaw(~isnan(intertrialRaw))];
    allYValsNorm = [allYValsNorm, reachNorm(~isnan(reachNorm)), intertrialNorm(~isnan(intertrialNorm))];
end
if ~isempty(allYValsRaw)
    yMin = min(allYValsRaw);
    yMax = max(allYValsRaw);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 1;
    end
    yLimitsRaw = [yMin - 0.05*yRange, yMax + 0.05*yRange];
else
    yLimitsRaw = [0, 1];
end
if ~isempty(allYValsNorm)
    yMin = min(allYValsNorm);
    yMax = max(allYValsNorm);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 1;
    end
    yLimitsNorm = [yMin - 0.05*yRange, yMax + 0.05*yRange];
else
    yLimitsNorm = [0, 1];
end

numAreasToPlot = length(areasToTest);
numCols = numAreasToPlot;
if nSegments > 0
    numRows = 3;  % raw sliding, norm sliding, segments
else
    numRows = 2;  % raw sliding, norm sliding
end

monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
    targetPos = monitorPositions(size(monitorPositions, 1), :);
else
    targetPos = monitorPositions(1, :);
end

figure(1003); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', targetPos);

useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.1 0.1], [0.08 0.04]);
else
    ha = zeros(numRows * numCols, 1);
    for i = 1:numRows * numCols
        ha(i) = subplot(numRows, numCols, i);
    end
end

% Row 1: Raw PR sliding
plotIdx = 0;
for a = areasToTest
    plotIdx = plotIdx + 1;
    axes(ha(plotIdx));
    hold on;

    if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        numNeurons = length(idMatIdx{a});
        neuronStr = sprintf(' (n=%d)', numNeurons);
    else
        neuronStr = '';
    end

    reachVals = prMetrics.reach{a};
    intertrialVals = prMetrics.intertrial{a};

    plot(slidingPositions, reachVals, '-o', 'Color', [0 0 1], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Reach');
    plot(slidingPositions, intertrialVals, '-s', 'Color', [1 0 0], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Intertrial');

    xlabel('Sliding Position (s)', 'FontSize', 10);
    ylabel('Participation ratio', 'FontSize', 10);
    title(sprintf('%s%s - Raw PR', areas{a}, neuronStr), 'FontSize', 11);
    grid on;
    if plotIdx == 1
        legend('Location', 'best', 'FontSize', 9);
    end
    ylim(yLimitsRaw);
    set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');
    set(gca, 'XTickMode', 'auto', 'XTickLabelMode', 'auto');
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% Row 2: Normalized PR sliding
for a = areasToTest
    plotIdx = plotIdx + 1;
    axes(ha(plotIdx));
    hold on;

    if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        numNeurons = length(idMatIdx{a});
        neuronStr = sprintf(' (n=%d)', numNeurons);
    else
        neuronStr = '';
    end

    reachVals = prMetrics.reachNormalized{a};
    intertrialVals = prMetrics.intertrialNormalized{a};

    plot(slidingPositions, reachVals, '-o', 'Color', [0 0 1], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Reach');
    plot(slidingPositions, intertrialVals, '-s', 'Color', [1 0 0], 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Intertrial');

    xlabel('Sliding Position (s)', 'FontSize', 10);
    ylabel('PR (normalized)', 'FontSize', 10);
    title(sprintf('%s%s - PR (normalized)', areas{a}, neuronStr), 'FontSize', 11);
    grid on;
    if plotIdx == numCols + 1
        legend('Location', 'best', 'FontSize', 9);
    end
    ylim(yLimitsNorm);
    set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');
    set(gca, 'XTickMode', 'auto', 'XTickLabelMode', 'auto');
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
end

% Row 3: Segment bar plots
if nSegments > 0
    segmentYVals = [];
    for a = areasToTest
        for s = 1:nSegments
            if ~isempty(segmentMetrics.reachNormalized{a, s}) && ~isnan(segmentMetrics.reachNormalized{a, s})
                segmentYVals = [segmentYVals, segmentMetrics.reachNormalized{a, s}];
            end
            if ~isempty(segmentMetrics.intertrialNormalized{a, s}) && ~isnan(segmentMetrics.intertrialNormalized{a, s})
                segmentYVals = [segmentYVals, segmentMetrics.intertrialNormalized{a, s}];
            end
        end
    end
    if ~isempty(segmentYVals)
        segYMin = min(segmentYVals);
        segYMax = max(segmentYVals);
        segYRange = segYMax - segYMin;
        if segYRange == 0
            segYRange = 1;
        end
        segmentYLimits = [segYMin - 0.05*segYRange, segYMax + 0.05*segYRange];
    else
        segmentYLimits = [0, 1];
    end

    for a = areasToTest
        plotIdx = plotIdx + 1;
        axes(ha(plotIdx));
        hold on;

        nSeg = numel(segmentNames);
        reachVals = nan(1, nSeg);
        intertrialVals = nan(1, nSeg);
        for s = 1:nSeg
            reachVals(s) = segmentMetrics.reachNormalized{a, s};
            intertrialVals(s) = segmentMetrics.intertrialNormalized{a, s};
        end

        X = 1:nSeg;
        barWidth = 0.35;
        bar(X - barWidth/2, reachVals, barWidth, 'FaceColor', [0 0 1], 'DisplayName', 'Reach');
        bar(X + barWidth/2, intertrialVals, barWidth, 'FaceColor', [1 0 0], 'DisplayName', 'Intertrial');

        set(gca, 'XTick', 1:nSeg, 'XTickLabel', segmentNames);
        xtickangle(45);
        xlabel('Segment', 'FontSize', 10);
        if plotIdx == 2*numCols + 1
            ylabel('PR (normalized)', 'FontSize', 10);
        end
        if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
            neuronStr = sprintf(' (n=%d)', length(idMatIdx{a}));
        else
            neuronStr = '';
        end
        title(sprintf('%s%s - Segments', areas{a}, neuronStr), 'FontSize', 11);
        grid on;
        ylim(segmentYLimits);
        set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');
        yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    end
end

winMax = max(slidingWindowSize(areasToTest));
if exist('sessionName', 'var') && ~isempty(sessionName)
    titlePrefix = [sessionName(1:min(10, length(sessionName))), ' - '];
else
    titlePrefix = '';
end
sgtitle(sprintf('%sPR: Reach vs Intertrial - Raw, Normalized, Segments (Win~%.1fs, Buffer %.1fs)', titlePrefix, winMax, windowBuffer), 'FontSize', 14, 'interpreter', 'none');

saveFile = fullfile(saveDir, sprintf('dimensionality_reach_intertrial_pr_win%.1f_step%.2f.png', winMax, stepSize));
exportgraphics(gcf, saveFile, 'Resolution', 300);
fprintf('Saved plot to: %s\n', saveFile);

fprintf('\n=== Analysis Complete ===\n');

%% =============================    Local Function    =============================
function pr = compute_participation_ratio(X)
% X is [time x neurons]. PR = (trace(C))^2 / trace(C^2) on mean-centered data.
    if size(X, 1) < 2 || size(X, 2) < 1
        pr = nan;
        return;
    end
    X = X - mean(X, 1);
    C = cov(X);
    if any(isnan(C(:))) || any(isinf(C(:)))
        pr = nan;
        return;
    end
    trC = trace(C);
    trC2 = trace(C * C);
    if trC2 <= 0 || ~isfinite(trC) || ~isfinite(trC2)
        pr = nan;
        return;
    end
    pr = (trC ^ 2) / trC2;
end
