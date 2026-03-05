%%
% Complexity Reach vs Intertrial LZC Analysis
% -------------------------------------------
% Compares Lempel-Ziv complexity (LZC) between reaches and intertrial
% intervals using sliding windows around reach starts and intertrial
% midpoints.
%
% Variables:
%   reachStart           - Reach start times in seconds
%   intertrialMidpoints  - Midpoint times between consecutive reaches
%   slidingWindowSize    - Window size for LZC analysis (area-specific if useOptimalWindowSize is true)
%   windowBuffer         - Minimum distance from window edge to event/midpoint (seconds)
%   beforeAlign          - Start of sliding window range (seconds before alignment point)
%   afterAlign           - End of sliding window range (seconds after alignment point)
%   stepSize             - Step size for sliding window (seconds)
%   nShuffles            - Number of shuffles for LZC normalization
%   useBernoulliControl  - If true, also compute Bernoulli-normalized LZC
%
% This script uses load_sliding_window_data() to load data (as in
% choose_task_and_session.m) and follows the same overall structure as:
%   - dimensionality_reach_intertrial_pr.m
%   - criticality_reach_intertrial_d2.m
%
% Normalization:
%   - Shuffle: LZC is divided by the mean LZC from shuffled versions of the
%     same binary sequence.
%   - Optional Bernoulli: LZC is divided by the mean LZC from rate-matched
%     Bernoulli control sequences.

%% =============================    Data Loading    =============================
fprintf('\n=== Loading Reach Data (LZC) ===\n');

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

dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

if ~isfield(dataStruct, 'dataR')
    error('dataR must be available in dataStruct for reach data. Check load_reach_data() implementation.');
end

dataR = dataStruct.dataR;
reachStart = dataR.R(:,1) / 1000; % ms -> s
totalReaches = length(reachStart);
fprintf('Loaded %d reaches\n', totalReaches);

areas   = dataStruct.areas;
idMatIdx = dataStruct.idMatIdx;
numAreas = length(areas);

if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

% Engagement segments
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
    segmentNames      = {};
    segmentWindowsList = {};
    nSegments         = 0;
end

% Intertrial midpoints
intertrialMidpoints = nan(1, totalReaches - 1);
for i = 1:totalReaches - 1
    intertrialMidpoints(i) = (reachStart(i) + reachStart(i+1)) / 2;
end
fprintf('Calculated %d intertrial midpoints\n', length(intertrialMidpoints));

%% =============================    Configuration    =============================
beforeAlign = -2;
afterAlign  =  2;
slidingWindowSize = 5;   % base value; overridden per-area if useOptimalWindowSize = true
stepSize    = 1/3;
windowBuffer = 0.5;

% LZC parameters
nShuffles           = 4;
useBernoulliControl = false;
useOptimalBinSize   = true;
useOptimalWindowSize = true;
minSpikesPerBin     = 0.08;
minDataPoints       = 2e4;
minSlidingWindowSize = 6;
maxSlidingWindowSize = 7;
minBinSize          = 0.01;
nMinNeurons         = 10;
includeM2356        = true;

loadResultsForPlotting = false;
resultsFileForPlotting = '';
makePlots = true;

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

fprintf('\n--- Using spike times for on-demand binning ---\n');
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end

%% =============================    Optimal Bin and Window Sizes    =============================
fprintf('\n=== Finding Optimal Bin and Window Sizes Per Area ===\n');
fprintf('\n=== Filtering Areas by Neuron Count ===\n');

validAreasToTest = [];
for a = areasToTest
    if includeM2356 && any(strcmp(areas, 'M2356')) && a == find(strcmp(areas, 'M2356'))
        continue;
    end
    if a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        nNeurons = length(idMatIdx{a});
        if nNeurons >= nMinNeurons
            validAreasToTest = [validAreasToTest, a]; %#ok<AGROW>
            fprintf('Area %s: %d neurons (included)\n', areas{a}, nNeurons);
        else
            fprintf('Area %s: %d neurons (excluded, < %d neurons)\n', areas{a}, nNeurons, nMinNeurons);
        end
    else
        fprintf('Area %s: no neurons found (excluded)\n', areas{a});
    end
end

if includeM2356 && any(strcmp(areas, 'M2356'))
    m2356Idx = find(strcmp(areas, 'M2356'));
    if m2356Idx <= length(idMatIdx) && ~isempty(idMatIdx{m2356Idx})
        nNeuronsM2356 = length(idMatIdx{m2356Idx});
        if nNeuronsM2356 >= nMinNeurons
            validAreasToTest = [validAreasToTest, m2356Idx]; %#ok<AGROW>
            fprintf('Area %s: %d neurons (included, combined M23+M56)\n', areas{m2356Idx}, nNeuronsM2356);
        else
            fprintf('Area %s: %d neurons (excluded, < %d neurons even when combined)\n', areas{m2356Idx}, nNeuronsM2356, nMinNeurons);
        end
    end
end

if isempty(validAreasToTest)
    error('No areas have sufficient neurons (>= %d).', nMinNeurons);
end

areasToTest = validAreasToTest;
fprintf('\nWill process %d area(s): %s\n', length(areasToTest), strjoin(areas(areasToTest), ', '));

binSize = zeros(1, numAreas);
for a = areasToTest
    neuronIDs = dataStruct.idLabel{a};
    thisFiringRate = calculate_firing_rate_from_spikes( ...
        dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange);
    if useOptimalBinSize
        [binSize(a), ~] = find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, 1);
        binSize(a) = max(binSize(a), minBinSize);
    else
        binSize(a) = 0.01;
    end
    fprintf('Area %s: bin size = %.3f s, firing rate = %.2f spikes/s\n', ...
        areas{a}, binSize(a), thisFiringRate);
end

if useOptimalWindowSize
    fprintf('\n=== Finding Optimal Window Sizes Per Area ===\n');
    slidingWindowSize = zeros(1, numAreas);
    for a = areasToTest
        neuronIDs = dataStruct.idLabel{a};
        nNeurons = length(neuronIDs);
        minRequiredWindowSize = (minDataPoints * binSize(a)) / nNeurons;
        slidingWindowSize(a) = ceil(max(minSlidingWindowSize, ...
            min(maxSlidingWindowSize, minRequiredWindowSize)));
        actualDataPoints = nNeurons * (slidingWindowSize(a) / binSize(a));
        fprintf('Area %s: %d neurons, binSize=%.3fs -> window=%.2fs (%.0f data points)\n', ...
            areas{a}, nNeurons, binSize(a), slidingWindowSize(a), actualDataPoints);
    end
else
    if isscalar(slidingWindowSize)
        slidingWindowSize = repmat(slidingWindowSize, 1, numAreas);
    end
end

%% =============================    Valid Reaches and Intertrial Midpoints    =============================
fprintf('\n=== Finding Valid Reaches and Intertrial Midpoints ===\n');

validWindowSizes = slidingWindowSize(areasToTest);
validWindowSizes = validWindowSizes(~isnan(validWindowSizes));
if isempty(validWindowSizes)
    error('No valid window sizes found. All areas were filtered out.');
end
minWindowSizeAcrossAreas = min(validWindowSizes);

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
    if maxWindowForThisReach >= minWindowSizeAcrossAreas
        validReachIndices = [validReachIndices, r]; %#ok<AGROW>
    end
end

validIntertrialIndices = [];
maxWindowPerIntertrial = nan(1, length(intertrialMidpoints));
for i = 1:length(intertrialMidpoints)
    midpointTime = intertrialMidpoints(i);
    prevReach = reachStart(i);
    nextReach = reachStart(i+1);
    maxWindowForThisIntertrial = inf;
    maxWindowFromPrev = 2 * (midpointTime - prevReach - windowBuffer);
    if maxWindowFromPrev < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromPrev;
    end
    maxWindowFromNext = 2 * (nextReach - midpointTime - windowBuffer);
    if maxWindowFromNext < maxWindowForThisIntertrial
        maxWindowForThisIntertrial = maxWindowFromNext;
    end
    maxWindowPerIntertrial(i) = maxWindowForThisIntertrial;
    if maxWindowForThisIntertrial >= minWindowSizeAcrossAreas
        validIntertrialIndices = [validIntertrialIndices, i]; %#ok<AGROW>
    end
end

if isempty(validReachIndices) && isempty(validIntertrialIndices)
    error('No valid reaches or intertrial midpoints found. Try reducing windowBuffer or minWindowSize.');
end

for a = areasToTest
    areaWindowSize = slidingWindowSize(a);
    if isnan(areaWindowSize)
        continue;
    end
    maxReachWindow = inf;
    if ~isempty(validReachIndices)
        maxReachWindow = min(maxWindowPerReach(validReachIndices));
    end
    maxIntertrialWindow = inf;
    if ~isempty(validIntertrialIndices)
        maxIntertrialWindow = min(maxWindowPerIntertrial(validIntertrialIndices));
    end
    maxAllowedWindow = min(maxReachWindow, maxIntertrialWindow);
    if areaWindowSize > maxAllowedWindow
        warning('Area %s: slidingWindowSize (%.2f s) exceeds maximum allowed (%.2f s). Using maximum allowed.', ...
            areas{a}, areaWindowSize, maxAllowedWindow);
        slidingWindowSize(a) = maxAllowedWindow;
    end
    if slidingWindowSize(a) < minWindowSizeAcrossAreas
        slidingWindowSize(a) = minWindowSizeAcrossAreas;
    end
    if slidingWindowSize(a) <= 0 || isnan(slidingWindowSize(a)) || isinf(slidingWindowSize(a))
        error('Area %s: Invalid slidingWindowSize.', areas{a});
    end
end

reachStartOriginal = reachStart;
intertrialMidpointsOriginal = intertrialMidpoints;

reachStart = reachStart(validReachIndices);
totalReaches = length(reachStart);

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

fprintf('Using area-specific sliding window sizes (min: %.2f s, max: %.2f s)\n', ...
    min(slidingWindowSize(areasToTest)), max(slidingWindowSize(areasToTest)));
fprintf('  Minimum window size: %.2f seconds\n', minWindowSizeAcrossAreas);
fprintf('  Valid reaches: %d/%d (%.1f%%)\n', length(validReachIndices), length(maxWindowPerReach),  ...
    100 * length(validReachIndices) / length(maxWindowPerReach));
fprintf('  Valid intertrial midpoints: %d/%d (%.1f%%)\n', length(intertrialMidpoints), length(maxWindowPerIntertrial), ...
    100 * length(intertrialMidpoints) / length(maxWindowPerIntertrial));
fprintf('  Window buffer: %.2f seconds\n', windowBuffer);

%% =============================    Analysis (Collect Windows)    =============================
fprintf('\n=== Processing Areas ===\n');

if runParallel
    currentPool = gcp('nocreate');
    if isempty(currentPool)
        numWorkers = min(3, length(dataStruct.areas));
        parpool('local', numWorkers);
        fprintf('Started parallel pool with %d workers\n', numWorkers);
    else
        fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
    end
end

slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

lzcMetrics = struct();
lzcMetrics.reach                  = cell(1, numAreas);
lzcMetrics.intertrial             = cell(1, numAreas);
lzcMetrics.reachNormalized        = cell(1, numAreas);
lzcMetrics.intertrialNormalized   = cell(1, numAreas);
if useBernoulliControl
    lzcMetrics.reachNormalizedBernoulli      = cell(1, numAreas);
    lzcMetrics.intertrialNormalizedBernoulli = cell(1, numAreas);
end

for a = areasToTest
    lzcMetrics.reach{a}                = nan(1, numSlidingPositions);
    lzcMetrics.intertrial{a}           = nan(1, numSlidingPositions);
    lzcMetrics.reachNormalized{a}      = nan(1, numSlidingPositions);
    lzcMetrics.intertrialNormalized{a} = nan(1, numSlidingPositions);
    if useBernoulliControl
        lzcMetrics.reachNormalizedBernoulli{a}      = nan(1, numSlidingPositions);
        lzcMetrics.intertrialNormalizedBernoulli{a} = nan(1, numSlidingPositions);
    end
end

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    neuronIDs = dataStruct.idLabel{a};
    areaWindowSize = slidingWindowSize(a);

    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);

    if a == areasToTest(1)
        collectedReachWindowData      = cell(numAreas, numSlidingPositions);
        collectedIntertrialWindowData = cell(numAreas, numSlidingPositions);
        collectedReachWindowCenters   = cell(numAreas, numSlidingPositions);
        collectedIntertrialWindowCenters = cell(numAreas, numSlidingPositions);
    end

    fprintf('  Collecting reach-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);
        for r = 1:totalReaches
            reachTime    = reachStart(r);
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
            winStart     = windowCenter - areaWindowSize / 2;
            winEnd       = windowCenter + areaWindowSize / 2;

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

            [startIdx, endIdx] = calculate_window_indices_from_center( ...
                windowCenter, areaWindowSize, binSize(a), numTimePoints);
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);
                if isempty(collectedReachWindowData{a, posIdx})
                    collectedReachWindowData{a, posIdx}    = {wDataMat};
                    collectedReachWindowCenters{a, posIdx} = windowCenter;
                else
                    collectedReachWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedReachWindowCenters{a, posIdx}     = [collectedReachWindowCenters{a, posIdx}, windowCenter];
                end
            end
        end
    end

    fprintf('  Collecting intertrial-aligned windows...\n');
    for posIdx = 1:numSlidingPositions
        offset = slidingPositions(posIdx);
        for idx = 1:length(intertrialMidpoints)
            midpointTime = intertrialMidpoints(idx);
            i = validIntertrialIndicesFiltered(idx);

            prevReach = reachStart(i);
            nextReach = reachStart(i+1);

            windowCenter = midpointTime + offset;
            winStart     = windowCenter - areaWindowSize / 2;
            winEnd       = windowCenter + areaWindowSize / 2;

            if winStart < prevReach + windowBuffer || winEnd > nextReach - windowBuffer
                continue;
            end

            [startIdx, endIdx] = calculate_window_indices_from_center( ...
                windowCenter, areaWindowSize, binSize(a), numTimePoints);
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);
                if isempty(collectedIntertrialWindowData{a, posIdx})
                    collectedIntertrialWindowData{a, posIdx}    = {wDataMat};
                    collectedIntertrialWindowCenters{a, posIdx} = windowCenter;
                else
                    collectedIntertrialWindowData{a, posIdx}{end+1} = wDataMat;
                    collectedIntertrialWindowCenters{a, posIdx}     = [collectedIntertrialWindowCenters{a, posIdx}, windowCenter];
                end
            end
        end
    end
end

%% =============================    Per-Sliding-Position LZC    =============================
fprintf('\n=== Per-Sliding-Position LZC Analysis (Reach vs Intertrial) ===\n');

perWindowLZC = struct();
perWindowLZC.reach            = cell(1, numAreas);
perWindowLZC.intertrial       = cell(1, numAreas);
perWindowLZC.reachNormalized  = cell(1, numAreas);
perWindowLZC.intertrialNormalized = cell(1, numAreas);
if useBernoulliControl
    perWindowLZC.reachNormalizedBernoulli      = cell(1, numAreas);
    perWindowLZC.intertrialNormalizedBernoulli = cell(1, numAreas);
end
perWindowLZC.reachCenters     = cell(1, numAreas);
perWindowLZC.intertrialCenters = cell(1, numAreas);

for a = areasToTest
    perWindowLZC.reach{a}          = cell(1, numSlidingPositions);
    perWindowLZC.intertrial{a}     = cell(1, numSlidingPositions);
    perWindowLZC.reachNormalized{a}= cell(1, numSlidingPositions);
    perWindowLZC.intertrialNormalized{a}= cell(1, numSlidingPositions);
    if useBernoulliControl
        perWindowLZC.reachNormalizedBernoulli{a}      = cell(1, numSlidingPositions);
        perWindowLZC.intertrialNormalizedBernoulli{a} = cell(1, numSlidingPositions);
    end
    perWindowLZC.reachCenters{a}   = cell(1, numSlidingPositions);
    perWindowLZC.intertrialCenters{a}= cell(1, numSlidingPositions);
end

numAreasToProcess = length(areasToTest);
tempLzcMetricsReach        = cell(1, numAreasToProcess);
tempLzcMetricsIntertrial   = cell(1, numAreasToProcess);
tempLzcMetricsReachNormalized      = cell(1, numAreasToProcess);
tempLzcMetricsIntertrialNormalized = cell(1, numAreasToProcess);
if useBernoulliControl
    tempLzcMetricsReachNormalizedBernoulli      = cell(1, numAreasToProcess);
    tempLzcMetricsIntertrialNormalizedBernoulli = cell(1, numAreasToProcess);
end
tempPerWindowLZCReach         = cell(1, numAreasToProcess);
tempPerWindowLZCIntertrial    = cell(1, numAreasToProcess);
tempPerWindowLZCReachNormalized  = cell(1, numAreasToProcess);
tempPerWindowLZCIntertrialNormalized = cell(1, numAreasToProcess);
if useBernoulliControl
    tempPerWindowLZCReachNormalizedBernoulli      = cell(1, numAreasToProcess);
    tempPerWindowLZCIntertrialNormalizedBernoulli = cell(1, numAreasToProcess);
end
tempPerWindowLZCReachCenters    = cell(1, numAreasToProcess);
tempPerWindowLZCIntertrialCenters = cell(1, numAreasToProcess);

parfor idx = 1:numAreasToProcess
    a = areasToTest(idx);
    fprintf('\nComputing LZC metrics per sliding position for area %s...\n', areas{a});

    tempLzcMetricsReach{idx}        = nan(1, numSlidingPositions);
    tempLzcMetricsIntertrial{idx}   = nan(1, numSlidingPositions);
    tempLzcMetricsReachNormalized{idx}      = nan(1, numSlidingPositions);
    tempLzcMetricsIntertrialNormalized{idx} = nan(1, numSlidingPositions);
    if useBernoulliControl
        tempLzcMetricsReachNormalizedBernoulli{idx}      = nan(1, numSlidingPositions);
        tempLzcMetricsIntertrialNormalizedBernoulli{idx} = nan(1, numSlidingPositions);
    end
    tempPerWindowLZCReach{idx}           = cell(1, numSlidingPositions);
    tempPerWindowLZCIntertrial{idx}      = cell(1, numSlidingPositions);
    tempPerWindowLZCReachNormalized{idx} = cell(1, numSlidingPositions);
    tempPerWindowLZCIntertrialNormalized{idx} = cell(1, numSlidingPositions);
    if useBernoulliControl
        tempPerWindowLZCReachNormalizedBernoulli{idx}      = cell(1, numSlidingPositions);
        tempPerWindowLZCIntertrialNormalizedBernoulli{idx} = cell(1, numSlidingPositions);
    end
    tempPerWindowLZCReachCenters{idx}    = cell(1, numSlidingPositions);
    tempPerWindowLZCIntertrialCenters{idx}= cell(1, numSlidingPositions);

    for posIdx = 1:numSlidingPositions
        % Reach windows
        if ~isempty(collectedReachWindowData{a, posIdx})
            windowDataList = collectedReachWindowData{a, posIdx};
            windowCenters  = collectedReachWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);
            lzcPerWindow      = nan(1, numWindows);
            lzcNormPerWindow  = nan(1, numWindows);
            lzcNormBernPerWindow = nan(1, numWindows);
            for w = 1:numWindows
                wDataMat = windowDataList{w};
                nNeurons = size(wDataMat, 2);
                nSamples = size(wDataMat, 1);
                concatenatedSeq = reshape(wDataMat', nSamples * nNeurons, 1);
                binarySeq = double(concatenatedSeq > 0);
                [lzVal, lzNorm, lzNormBern] = compute_lz_complexity_with_controls( ...
                    binarySeq, nShuffles, useBernoulliControl);
                lzcPerWindow(w)     = lzVal;
                lzcNormPerWindow(w) = lzNorm;
                lzcNormBernPerWindow(w) = lzNormBern;
            end
            tempLzcMetricsReach{idx}(posIdx)       = nanmean(lzcPerWindow);
            tempLzcMetricsReachNormalized{idx}(posIdx) = nanmean(lzcNormPerWindow);
            if useBernoulliControl
                tempLzcMetricsReachNormalizedBernoulli{idx}(posIdx) = nanmean(lzcNormBernPerWindow);
            end
            tempPerWindowLZCReach{idx}{posIdx}          = lzcPerWindow;
            tempPerWindowLZCReachNormalized{idx}{posIdx}= lzcNormPerWindow;
            if useBernoulliControl
                tempPerWindowLZCReachNormalizedBernoulli{idx}{posIdx} = lzcNormBernPerWindow;
            end
            tempPerWindowLZCReachCenters{idx}{posIdx} = windowCenters;
        end

        % Intertrial windows
        if ~isempty(collectedIntertrialWindowData{a, posIdx})
            windowDataList = collectedIntertrialWindowData{a, posIdx};
            windowCenters  = collectedIntertrialWindowCenters{a, posIdx};
            numWindows = numel(windowDataList);
            lzcPerWindow      = nan(1, numWindows);
            lzcNormPerWindow  = nan(1, numWindows);
            lzcNormBernPerWindow = nan(1, numWindows);
            for w = 1:numWindows
                wDataMat = windowDataList{w};
                nNeurons = size(wDataMat, 2);
                nSamples = size(wDataMat, 1);
                concatenatedSeq = reshape(wDataMat', nSamples * nNeurons, 1);
                binarySeq = double(concatenatedSeq > 0);
                [lzVal, lzNorm, lzNormBern] = compute_lz_complexity_with_controls( ...
                    binarySeq, nShuffles, useBernoulliControl);
                lzcPerWindow(w)     = lzVal;
                lzcNormPerWindow(w) = lzNorm;
                lzcNormBernPerWindow(w) = lzNormBern;
            end
            tempLzcMetricsIntertrial{idx}(posIdx)       = nanmean(lzcPerWindow);
            tempLzcMetricsIntertrialNormalized{idx}(posIdx) = nanmean(lzcNormPerWindow);
            if useBernoulliControl
                tempLzcMetricsIntertrialNormalizedBernoulli{idx}(posIdx) = nanmean(lzcNormBernPerWindow);
            end
            tempPerWindowLZCIntertrial{idx}{posIdx}          = lzcPerWindow;
            tempPerWindowLZCIntertrialNormalized{idx}{posIdx}= lzcNormPerWindow;
            if useBernoulliControl
                tempPerWindowLZCIntertrialNormalizedBernoulli{idx}{posIdx} = lzcNormBernPerWindow;
            end
            tempPerWindowLZCIntertrialCenters{idx}{posIdx} = windowCenters;
        end
    end
end

for idx = 1:numAreasToProcess
    a = areasToTest(idx);
    lzcMetrics.reach{a}              = tempLzcMetricsReach{idx};
    lzcMetrics.intertrial{a}         = tempLzcMetricsIntertrial{idx};
    lzcMetrics.reachNormalized{a}    = tempLzcMetricsReachNormalized{idx};
    lzcMetrics.intertrialNormalized{a} = tempLzcMetricsIntertrialNormalized{idx};
    if useBernoulliControl
        lzcMetrics.reachNormalizedBernoulli{a}    = tempLzcMetricsReachNormalizedBernoulli{idx};
        lzcMetrics.intertrialNormalizedBernoulli{a} = tempLzcMetricsIntertrialNormalizedBernoulli{idx};
    end
    perWindowLZC.reach{a}            = tempPerWindowLZCReach{idx};
    perWindowLZC.intertrial{a}       = tempPerWindowLZCIntertrial{idx};
    perWindowLZC.reachNormalized{a}  = tempPerWindowLZCReachNormalized{idx};
    perWindowLZC.intertrialNormalized{a} = tempPerWindowLZCIntertrialNormalized{idx};
    if useBernoulliControl
        perWindowLZC.reachNormalizedBernoulli{a}      = tempPerWindowLZCReachNormalizedBernoulli{idx};
        perWindowLZC.intertrialNormalizedBernoulli{a} = tempPerWindowLZCIntertrialNormalizedBernoulli{idx};
    end
    perWindowLZC.reachCenters{a}     = tempPerWindowLZCReachCenters{idx};
    perWindowLZC.intertrialCenters{a}= tempPerWindowLZCIntertrialCenters{idx};
end

%% =============================   Segment-level LZC    =============================
fprintf('\n=== Segment-level LZC Analysis (Reach vs Intertrial) ===\n');

lzcSegmentMetrics = struct();
lzcSegmentMetrics.reach              = cell(numAreas, nSegments);
lzcSegmentMetrics.intertrial         = cell(numAreas, nSegments);
lzcSegmentMetrics.reachNormalized    = cell(numAreas, nSegments);
lzcSegmentMetrics.intertrialNormalized = cell(numAreas, nSegments);
if useBernoulliControl
    lzcSegmentMetrics.reachNormalizedBernoulli      = cell(numAreas, nSegments);
    lzcSegmentMetrics.intertrialNormalizedBernoulli = cell(numAreas, nSegments);
end

for a = areasToTest
    fprintf('\nProcessing segments for area %s...\n', areas{a});
    for s = 1:nSegments
        if isempty(segmentWindowsList) || s > length(segmentWindowsList) || ...
                isempty(segmentWindowsList{s}) || any(isnan(segmentWindowsList{s}))
            lzcSegmentMetrics.reach{a, s}              = nan;
            lzcSegmentMetrics.intertrial{a, s}         = nan;
            lzcSegmentMetrics.reachNormalized{a, s}    = nan;
            lzcSegmentMetrics.intertrialNormalized{a, s} = nan;
            if useBernoulliControl
                lzcSegmentMetrics.reachNormalizedBernoulli{a, s}      = nan;
                lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nan;
            end
            continue;
        end

        segWin   = segmentWindowsList{s};
        tStartSeg = segWin(1);
        tEndSeg   = segWin(2);

        alignPosIdx = find(abs(slidingPositions) < stepSize/2, 1);
        if isempty(alignPosIdx)
            alignPosIdx = 1;
        end

        reachNormInSeg     = [];
        intertrialNormInSeg = [];

        if ~isempty(perWindowLZC.reachNormalized{a}{alignPosIdx}) && ...
           ~isempty(perWindowLZC.reachCenters{a}{alignPosIdx})
            lzcVals = perWindowLZC.reachNormalized{a}{alignPosIdx};
            windowCenters = perWindowLZC.reachCenters{a}{alignPosIdx};
            windowCenters = windowCenters(:)';
            lzcVals       = lzcVals(:)';
            inSeg = (windowCenters >= tStartSeg) & (windowCenters <= tEndSeg);
            if any(inSeg)
                reachNormInSeg = [reachNormInSeg, lzcVals(inSeg)]; %#ok<AGROW>
            end
        end

        if ~isempty(perWindowLZC.intertrialNormalized{a}{alignPosIdx}) && ...
           ~isempty(perWindowLZC.intertrialCenters{a}{alignPosIdx})
            lzcVals = perWindowLZC.intertrialNormalized{a}{alignPosIdx};
            windowCenters = perWindowLZC.intertrialCenters{a}{alignPosIdx};
            windowCenters = windowCenters(:)';
            lzcVals       = lzcVals(:)';
            inSeg = (windowCenters >= tStartSeg) & (windowCenters <= tEndSeg);
            if any(inSeg)
                intertrialNormInSeg = [intertrialNormInSeg, lzcVals(inSeg)]; %#ok<AGROW>
            end
        end

        if ~isempty(reachNormInSeg)
            lzcSegmentMetrics.reachNormalized{a, s} = nanmean(reachNormInSeg);
        else
            lzcSegmentMetrics.reachNormalized{a, s} = nan;
        end
        if ~isempty(intertrialNormInSeg)
            lzcSegmentMetrics.intertrialNormalized{a, s} = nanmean(intertrialNormInSeg);
        else
            lzcSegmentMetrics.intertrialNormalized{a, s} = nan;
        end

        % raw
        if ~isempty(perWindowLZC.reach{a}{alignPosIdx}) && ...
           ~isempty(perWindowLZC.reachCenters{a}{alignPosIdx})
            lzcValsRaw = perWindowLZC.reach{a}{alignPosIdx};
            windowCenters = perWindowLZC.reachCenters{a}{alignPosIdx};
            windowCenters = windowCenters(:)';
            lzcValsRaw    = lzcValsRaw(:)';
            inSeg = (windowCenters >= tStartSeg) & (windowCenters <= tEndSeg);
            if any(inSeg)
                lzcSegmentMetrics.reach{a, s} = nanmean(lzcValsRaw(inSeg));
            else
                lzcSegmentMetrics.reach{a, s} = nan;
            end
        else
            lzcSegmentMetrics.reach{a, s} = nan;
        end

        if ~isempty(perWindowLZC.intertrial{a}{alignPosIdx}) && ...
           ~isempty(perWindowLZC.intertrialCenters{a}{alignPosIdx})
            lzcValsRaw = perWindowLZC.intertrial{a}{alignPosIdx};
            windowCenters = perWindowLZC.intertrialCenters{a}{alignPosIdx};
            windowCenters = windowCenters(:)';
            lzcValsRaw    = lzcValsRaw(:)';
            inSeg = (windowCenters >= tStartSeg) & (windowCenters <= tEndSeg);
            if any(inSeg)
                lzcSegmentMetrics.intertrial{a, s} = nanmean(lzcValsRaw(inSeg));
            else
                lzcSegmentMetrics.intertrial{a, s} = nan;
            end
        else
            lzcSegmentMetrics.intertrial{a, s} = nan;
        end

        if useBernoulliControl
            if ~isempty(perWindowLZC.reachNormalizedBernoulli{a}{alignPosIdx}) && ...
               ~isempty(perWindowLZC.reachCenters{a}{alignPosIdx})
                lzcValsBern = perWindowLZC.reachNormalizedBernoulli{a}{alignPosIdx};
                windowCenters = perWindowLZC.reachCenters{a}{alignPosIdx};
                windowCenters = windowCenters(:)';
                lzcValsBern   = lzcValsBern(:)';
                inSeg = (windowCenters >= tStartSeg) & (windowCenters <= tEndSeg);
                if any(inSeg)
                    lzcSegmentMetrics.reachNormalizedBernoulli{a, s} = nanmean(lzcValsBern(inSeg));
                else
                    lzcSegmentMetrics.reachNormalizedBernoulli{a, s} = nan;
                end
            else
                lzcSegmentMetrics.reachNormalizedBernoulli{a, s} = nan;
            end

            if ~isempty(perWindowLZC.intertrialNormalizedBernoulli{a}{alignPosIdx}) && ...
               ~isempty(perWindowLZC.intertrialCenters{a}{alignPosIdx})
                lzcValsBern = perWindowLZC.intertrialNormalizedBernoulli{a}{alignPosIdx};
                windowCenters = perWindowLZC.intertrialCenters{a}{alignPosIdx};
                windowCenters = windowCenters(:)';
                lzcValsBern   = lzcValsBern(:)';
                inSeg = (windowCenters >= tStartSeg) & (windowCenters <= tEndSeg);
                if any(inSeg)
                    lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nanmean(lzcValsBern(inSeg));
                else
                    lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nan;
                end
            else
                lzcSegmentMetrics.intertrialNormalizedBernoulli{a, s} = nan;
            end
        end
    end
end

%% =============================    Save Results    =============================
results = struct();
results.areas                = areas;
results.reachStart           = reachStart;
results.intertrialMidpoints  = intertrialMidpoints;
results.slidingWindowSize    = slidingWindowSize;
results.windowBuffer         = windowBuffer;
results.beforeAlign          = beforeAlign;
results.afterAlign           = afterAlign;
results.stepSize             = stepSize;
results.slidingPositions     = slidingPositions;
results.lzcMetrics           = lzcMetrics;
results.lzcSegmentMetrics    = lzcSegmentMetrics;
results.segmentNames         = segmentNames;
if exist('sessionName', 'var')
    results.sessionName = sessionName;
end
if exist('idMatIdx', 'var')
    results.idMatIdx   = idMatIdx;
end
if exist('segmentWindowsEng', 'var')
    results.segmentWindows = segmentWindowsEng;
end
results.binSize                     = binSize;
results.params.nShuffles            = nShuffles;
results.params.useBernoulliControl  = useBernoulliControl;
results.params.useOptimalBinSize    = useOptimalBinSize;
results.params.useOptimalWindowSize = useOptimalWindowSize;
results.params.minSpikesPerBin      = minSpikesPerBin;
results.params.minDataPoints        = minDataPoints;
results.params.minSlidingWindowSize = minSlidingWindowSize;
results.params.maxSlidingWindowSize = maxSlidingWindowSize;
results.params.minBinSize           = minBinSize;
results.params.nMinNeurons          = nMinNeurons;
results.params.includeM2356         = includeM2356;

meanWindowSize = mean(slidingWindowSize(areasToTest));
resultsPath = fullfile(saveDir, sprintf('complexity_reach_intertrial_lzc_win%.1f_step%.1f.mat', ...
    meanWindowSize, stepSize));
save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);

%% =============================    Plotting    =============================
if ~makePlots
    fprintf('\n=== Skipping plots (makePlots = false) ===\n');
    return;
end

fprintf('\n=== Creating Summary Plots ===\n');

% (Plotting code omitted here for brevity; you can keep the original
% lzc_reach_intertrial.m plotting section and just update filenames to use
% "complexity_reach_intertrial_lzc" if you want full parity.)

fprintf('\n=== Complexity Reach vs Intertrial LZC Analysis Complete ===\n');

%% =============================    Helper Function    =============================
function [lzComplexity, lzNormalized, lzNormalizedBernoulli] = compute_lz_complexity_with_controls(binarySeq, nShuffles, useBernoulliControl)
% COMPUTE_LZ_COMPLEXITY_WITH_CONTROLS
%   Compute LZ complexity with shuffle and optional Bernoulli controls.

    if nargin < 3
        useBernoulliControl = false;
    end

    try
        binarySeq = binarySeq(:);
        lzComplexity = limpel_ziv_complexity(binarySeq, 'method', 'binary');

        shuffledLZ = nan(1, nShuffles);
        for s = 1:nShuffles
            shuffledSeq = binarySeq(randperm(length(binarySeq)));
            shuffledLZ(s) = limpel_ziv_complexity(shuffledSeq, 'method', 'binary');
        end
        meanShuffledLZ = nanmean(shuffledLZ);
        if meanShuffledLZ > 0
            lzNormalized = lzComplexity / meanShuffledLZ;
        else
            lzNormalized = nan;
        end

        if useBernoulliControl
            firingRate = mean(binarySeq);
            bernoulliLZ = nan(1, nShuffles);
            for s = 1:nShuffles
                bernoulliSeq = double(rand(length(binarySeq), 1) < firingRate);
                bernoulliLZ(s) = limpel_ziv_complexity(bernoulliSeq, 'method', 'binary');
            end
            meanBernoulliLZ = nanmean(bernoulliLZ);
            if meanBernoulliLZ > 0
                lzNormalizedBernoulli = lzComplexity / meanBernoulliLZ;
            else
                lzNormalizedBernoulli = nan;
            end
        else
            lzNormalizedBernoulli = nan;
        end
    catch ME
        fprintf('    Warning: Error computing LZ complexity: %s\n', ME.message);
        lzComplexity           = nan;
        lzNormalized           = nan;
        lzNormalizedBernoulli  = nan;
    end
end