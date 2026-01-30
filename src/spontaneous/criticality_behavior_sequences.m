%%
% Criticality Behavior Sequences d2 Analysis
% Assesses d2 criticality in sliding windows around two alignment event
% types (e.g. onset and offset of behavior sequences). Takes the mean of d2
% values across windows at each sliding position for each alignment type.
%
% Variables:
%   sessionName   - Session name (e.g. 'ag112321_1' or 'ag/ag112321/recording1')
%   align1Times   - Event times for first alignment type (seconds)
%   align2Times   - Event times for second alignment type (seconds)
%   align1Name     - String name for first alignment (e.g. 'Onset') for legends, printing, saving
%   align2Name     - String name for second alignment (e.g. 'Offset') for legends, printing, saving
%   slidingWindowSize - Window size for d2 analysis (seconds)
%   windowBuffer  - Minimum distance from window edge to recording bounds (seconds)
%   beforeAlign   - Start of sliding range (seconds before alignment point)
%   afterAlign    - End of sliding range (seconds after alignment point)
%   stepSize      - Step size for sliding window (seconds)
%   nShuffles     - Number of circular permutations for d2 normalization
%   normalizeD2   - If true, normalize d2 by mean shuffled d2
%
% Uses load_sliding_window_data('spontaneous', 'spikes', ...) and
% spontaneous_behavior_sequences() for event times. align1Name and align2Name
% are used in plot legends, fprintf messages, and saved plot/file names.

%% =============================    Data Loading    =============================
fprintf('\n=== Loading Spontaneous Data ===\n');

if ~exist('sessionName', 'var') || isempty(sessionName)
    error('sessionName must be defined. Set sessionName in workspace (e.g. ''ag112321_1'').');
end

sessionType = 'spontaneous';
opts = neuro_behavior_options;
opts.frameSize = 0.001;
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = 0;
opts.collectEnd = [];
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;
if ~isfield(opts, 'fsBhv') || isempty(opts.fsBhv)
    opts.fsBhv = 30;  % Default behavior sampling rate
end

% Load neural data
dataStruct = load_sliding_window_data(sessionType, 'spikes', ...
    'sessionName', sessionName, 'opts', opts);

areas = dataStruct.areas;
idMatIdx = dataStruct.idMatIdx;
numAreas = length(areas);

if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

% Paths and options for behavior sequences
paths = get_paths;
pathParts = strsplit(sessionName, filesep);
subDir = pathParts{1}(1:min(2, numel(pathParts{1})));
optsBhv = opts;
optsBhv.frameSize = opts.fsBhv;
optsBhv.collectStart = 0;
optsBhv.collectEnd = [];
optsBhv.sessionName = sessionName;
optsBhv.dataPath = fullfile(paths.spontaneousDataPath, subDir);
if ~exist('optsBehaviorSequences', 'var') || isempty(optsBehaviorSequences)
    optsBhv.minDur = 5;
    optsBhv.propThreshold = 0.99;
    optsBhv.bufferSec = 5;
    optsBhv.behaviorIds = 5:10;
else
    optsBhv.minDur = optsBehaviorSequences.minDur;
    optsBhv.propThreshold = optsBehaviorSequences.propThreshold;
    optsBhv.bufferSec = optsBehaviorSequences.bufferSec;
    optsBhv.behaviorIds = optsBehaviorSequences.behaviorIds;
end

% % Get behavior sequence onset and offset times (used as align1 and align2 by default)
% [sequences, times] = spontaneous_behavior_sequences(sessionName, optsBhv);
% if isempty(times)
%     error('No behavior sequences found for session %s. Check optsBehaviorSequences (minDur, propThreshold, bufferSec, behaviorIds).', sessionName);
% end
% align1Times = cellfun(@(t) t(1), times);
% align2Times = cellfun(@(t) t(end), times);
% numSequences = length(align1Times);



fprintf('Found %d and %d behavior sequences (align1/align2 times in seconds)\n', length(align1Times), length(align2Times));

%% =============================    Configuration    =============================
beforeAlign = -2;
afterAlign  = 2;
slidingWindowSize = 7;
stepSize = 0.1;
windowBuffer = 0.5;
minWindowSize = slidingWindowSize;

pOrder = 10;
critType = 2;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;
nShuffles = 3;
normalizeD2 = true;

pcaFlag = 0;
pcaFirstFlag = 1;
nDim = 4;

loadResultsForPlotting = false;
resultsFileForPlotting = '';
makePlots = true;

% Alignment names for plot legends, printing, and saved file names
if ~exist('align1Name', 'var') || isempty(align1Name)
    align1Name = 'Onset';
end
if ~exist('align2Name', 'var') || isempty(align2Name)
    align2Name = 'Offset';
end

saveDir = dataStruct.saveDir;
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% Utilities and criticality
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'sliding_window_prep', 'utils'));
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'criticality'));

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end

%% =============================    Optimal Bin Sizes Per Area    =============================
fprintf('\n=== Finding Optimal Bin Sizes Per Area ===\n');

if pcaFlag
    reconstructedDataMat = cell(1, numAreas);
    tempBinSize = 0.001;
    for a = areasToTest
        neuronIDs = dataStruct.idLabel{a};
        thisDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
            neuronIDs, timeRange, tempBinSize);
        [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim;
        reconstructedDataMat{a} = score(:, nDim) * coeff(:, nDim)' + mu;
    end
else
    reconstructedDataMat = [];
end

binSize = zeros(1, numAreas);
for a = areasToTest
    neuronIDs = dataStruct.idLabel{a};
    thisFiringRate = calculate_firing_rate_from_spikes(...
        dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange);
    [binSize(a), ~] = find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, minBinsPerWindow);
    fprintf('Area %s: bin size = %.3f s, firing rate = %.2f spikes/s\n', ...
        areas{a}, binSize(a), thisFiringRate);
end

%% =============================    Valid Align1 and Align2 Times    =============================
fprintf('\n=== Finding Valid %s and %s Times ===\n', align1Name, align2Name);

tStart = timeRange(1);
tEnd = timeRange(2);
halfWin = slidingWindowSize / 2;
% For every sliding position the window must stay in [tStart+windowBuffer, tEnd-windowBuffer]
% Worst case: center at event+beforeAlign (leftmost) and event+afterAlign (rightmost)
winMinCenter = tStart + windowBuffer + halfWin;
winMaxCenter = tEnd - windowBuffer - halfWin;
minEventTime = winMinCenter - afterAlign;   % so event+afterAlign - halfWin >= tStart+buffer
maxEventTime = winMaxCenter - beforeAlign;  % so event+beforeAlign + halfWin <= tEnd-buffer

validAlign1Indices = [];
validAlign2Indices = [];
for s = 1:length(align1Times)
    if align1Times(s) >= minEventTime && align1Times(s) <= maxEventTime
        validAlign1Indices = [validAlign1Indices, s]; %#ok<AGROW>
    end
end
for s = 1:length(align2Times)
    if align2Times(s) >= minEventTime && align2Times(s) <= maxEventTime
        validAlign2Indices = [validAlign2Indices, s]; %#ok<AGROW>
    end
end

align1TimesValid = align1Times(validAlign1Indices);
align2TimesValid = align2Times(validAlign2Indices);
numAlign1 = length(align1TimesValid);
numAlign2 = length(align2TimesValid);

fprintf('Sliding window size: %.2f s, buffer: %.2f s\n', slidingWindowSize, windowBuffer);
fprintf('Valid %s: %d/%d, valid %s: %d/%d\n', ...
    align1Name, numAlign1, length(align1Times), align2Name, numAlign2, length(align2Times));

if numAlign1 == 0 && numAlign2 == 0
    error('No valid %s or %s times. Increase recording time or reduce slidingWindowSize/windowBuffer.', align1Name, align2Name);
end

%% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

d2Metrics = struct();
d2Metrics.align1 = cell(1, numAreas);
d2Metrics.align2 = cell(1, numAreas);
d2Metrics.align1Normalized = cell(1, numAreas);
d2Metrics.align2Normalized = cell(1, numAreas);

for a = areasToTest
    d2Metrics.align1{a} = nan(1, numSlidingPositions);
    d2Metrics.align2{a} = nan(1, numSlidingPositions);
    d2Metrics.align1Normalized{a} = nan(1, numSlidingPositions);
    d2Metrics.align2Normalized{a} = nan(1, numSlidingPositions);
end

for a = areasToTest
    fprintf('\nProcessing area %s...\n', areas{a});
    tic;

    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);

    if pcaFlag
        [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
        forDim = find(cumsum(explained) > 30, 1);
        forDim = max(3, min(6, forDim));
        nDim = 1:forDim;
        aDataMat = score(:, nDim) * coeff(:, nDim)' + mu;
    end

    if a == areasToTest(1)
        collectedAlign1Windows = cell(numAreas, numSlidingPositions);
        collectedAlign2Windows = cell(numAreas, numSlidingPositions);
        collectedAlign1WindowData = cell(numAreas, numSlidingPositions);
        collectedAlign2WindowData = cell(numAreas, numSlidingPositions);
        collectedAlign1WindowCenters = cell(numAreas, numSlidingPositions);
        collectedAlign2WindowCenters = cell(numAreas, numSlidingPositions);
    end

    % Collect align1-aligned windows
    for posIdx = 1:numSlidingPositions
        posSec = slidingPositions(posIdx);
        for s = 1:numAlign1
            eventTime = align1TimesValid(s);
            windowCenter = eventTime + posSec;
            winStart = windowCenter - slidingWindowSize / 2;
            winEnd = windowCenter + slidingWindowSize / 2;
            if winStart < tStart + windowBuffer || winEnd > tEnd - windowBuffer
                continue;
            end
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, slidingWindowSize, binSize(a), numTimePoints);
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);
                wPopActivity = sum(wDataMat, 2);
                if isempty(collectedAlign1Windows{a, posIdx})
                    collectedAlign1Windows{a, posIdx} = wPopActivity(:)';
                    collectedAlign1WindowData{a, posIdx} = {wDataMat};
                    collectedAlign1WindowCenters{a, posIdx} = windowCenter;
                else
                    collectedAlign1Windows{a, posIdx} = [collectedAlign1Windows{a, posIdx}; wPopActivity(:)'];
                    collectedAlign1WindowData{a, posIdx}{end+1} = wDataMat;
                    collectedAlign1WindowCenters{a, posIdx} = [collectedAlign1WindowCenters{a, posIdx}, windowCenter];
                end
            end
        end
    end

    % Collect align2-aligned windows
    for posIdx = 1:numSlidingPositions
        posSec = slidingPositions(posIdx);
        for s = 1:numAlign2
            eventTime = align2TimesValid(s);
            windowCenter = eventTime + posSec;
            winStart = windowCenter - slidingWindowSize / 2;
            winEnd = windowCenter + slidingWindowSize / 2;
            if winStart < tStart + windowBuffer || winEnd > tEnd - windowBuffer
                continue;
            end
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, slidingWindowSize, binSize(a), numTimePoints);
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);
                wPopActivity = sum(wDataMat, 2);
                if isempty(collectedAlign2Windows{a, posIdx})
                    collectedAlign2Windows{a, posIdx} = wPopActivity(:)';
                    collectedAlign2WindowData{a, posIdx} = {wDataMat};
                    collectedAlign2WindowCenters{a, posIdx} = windowCenter;
                else
                    collectedAlign2Windows{a, posIdx} = [collectedAlign2Windows{a, posIdx}; wPopActivity(:)'];
                    collectedAlign2WindowData{a, posIdx}{end+1} = wDataMat;
                    collectedAlign2WindowCenters{a, posIdx} = [collectedAlign2WindowCenters{a, posIdx}, windowCenter];
                end
            end
        end
    end

    fprintf('  Area %s completed in %.1f min\n', areas{a}, toc / 60);
end

%% =============================    Per-Sliding-Position d2    =============================
fprintf('\n=== Computing d2 Per Sliding Position (%s vs %s) ===\n', align1Name, align2Name);

% Mean population activity per sliding position (for right y-axis in plots)
meanPopActivity = struct();
meanPopActivity.align1 = cell(1, numAreas);
meanPopActivity.align2 = cell(1, numAreas);
for aa = areasToTest
    meanPopActivity.align1{aa} = nan(1, numSlidingPositions);
    meanPopActivity.align2{aa} = nan(1, numSlidingPositions);
end

for a = areasToTest
    fprintf('\nComputing d2 for area %s...\n', areas{a});

    for posIdx = 1:numSlidingPositions
        % Align1 windows: use precomputed population activity from collectedAlign1Windows
        if ~isempty(collectedAlign1Windows{a, posIdx})
            windowPopMat = collectedAlign1Windows{a, posIdx};  % [numWindows x numBins]
            % Mean pop activity = mean over windows of (mean over time in each window)
            meanPopActivity.align1{a}(posIdx) = mean(mean(windowPopMat, 2));
            numWindows = size(windowPopMat, 1);
            d2PerWindow = nan(1, numWindows);
            d2ShuffledPerWindow = nan(numWindows, nShuffles);
            windowDataList = collectedAlign1WindowData{a, posIdx};  % only needed for shuffles

            for w = 1:numWindows
                wPopActivity = windowPopMat(w, :)';  % column vector for myYuleWalker3
                if ~isempty(wPopActivity)
                    try
                        [varphi, ~] = myYuleWalker3(double(wPopActivity), pOrder);
                        d2PerWindow(w) = getFixedPointDistance2(pOrder, critType, varphi);
                    catch
                        d2PerWindow(w) = nan;
                    end
                end
                if normalizeD2
                    wDataMat = windowDataList{w};  % [timeBins x neurons] for per-neuron permutation
                    numNeurons = size(wDataMat, 2);
                    numTimeBins = size(wDataMat, 1);
                    for sh = 1:nShuffles
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        permutedPopActivity = sum(permutedDataMat, 2);
                        if ~isempty(permutedPopActivity)
                            try
                                [varphiPerm, ~] = myYuleWalker3(double(permutedPopActivity), pOrder);
                                d2ShuffledPerWindow(w, sh) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                            catch
                                d2ShuffledPerWindow(w, sh) = nan;
                            end
                        end
                    end
                end
            end

            d2Metrics.align1{a}(posIdx) = nanmean(d2PerWindow);
            if normalizeD2
                meanShuffledPerWindow = nanmean(d2ShuffledPerWindow, 2);
                d2NormalizedPerWindow = nan(1, numWindows);
                for w = 1:numWindows
                    if ~isnan(d2PerWindow(w)) && ~isnan(meanShuffledPerWindow(w)) && meanShuffledPerWindow(w) > 0
                        d2NormalizedPerWindow(w) = d2PerWindow(w) / meanShuffledPerWindow(w);
                    end
                end
                d2Metrics.align1Normalized{a}(posIdx) = nanmean(d2NormalizedPerWindow);
            end
        end

        % Align2 windows: use precomputed population activity from collectedAlign2Windows
        if ~isempty(collectedAlign2Windows{a, posIdx})
            windowPopMat = collectedAlign2Windows{a, posIdx};  % [numWindows x numBins]
            % Mean pop activity = mean over windows of (mean over time in each window)
            meanPopActivity.align2{a}(posIdx) = mean(mean(windowPopMat, 2));
            numWindows = size(windowPopMat, 1);
            d2PerWindow = nan(1, numWindows);
            d2ShuffledPerWindow = nan(numWindows, nShuffles);
            windowDataList = collectedAlign2WindowData{a, posIdx};  % only needed for shuffles

            for w = 1:numWindows
                wPopActivity = windowPopMat(w, :)';  % column vector for myYuleWalker3
                if ~isempty(wPopActivity)
                    try
                        [varphi, ~] = myYuleWalker3(double(wPopActivity), pOrder);
                        d2PerWindow(w) = getFixedPointDistance2(pOrder, critType, varphi);
                    catch
                        d2PerWindow(w) = nan;
                    end
                end
                if normalizeD2
                    wDataMat = windowDataList{w};  % [timeBins x neurons] for per-neuron permutation
                    numNeurons = size(wDataMat, 2);
                    numTimeBins = size(wDataMat, 1);
                    for sh = 1:nShuffles
                        permutedDataMat = zeros(size(wDataMat));
                        for n = 1:numNeurons
                            shiftAmount = randi(numTimeBins);
                            permutedDataMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
                        end
                        permutedPopActivity = sum(permutedDataMat, 2);
                        if ~isempty(permutedPopActivity)
                            try
                                [varphiPerm, ~] = myYuleWalker3(double(permutedPopActivity), pOrder);
                                d2ShuffledPerWindow(w, sh) = getFixedPointDistance2(pOrder, critType, varphiPerm);
                            catch
                                d2ShuffledPerWindow(w, sh) = nan;
                            end
                        end
                    end
                end
            end

            d2Metrics.align2{a}(posIdx) = nanmean(d2PerWindow);
            if normalizeD2
                meanShuffledPerWindow = nanmean(d2ShuffledPerWindow, 2);
                d2NormalizedPerWindow = nan(1, numWindows);
                for w = 1:numWindows
                    if ~isnan(d2PerWindow(w)) && ~isnan(meanShuffledPerWindow(w)) && meanShuffledPerWindow(w) > 0
                        d2NormalizedPerWindow(w) = d2PerWindow(w) / meanShuffledPerWindow(w);
                    end
                end
                d2Metrics.align2Normalized{a}(posIdx) = nanmean(d2NormalizedPerWindow);
            end
        end
    end
end

%% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.align1Times = align1TimesValid;
results.align2Times = align2TimesValid;
results.align1Name = align1Name;
results.align2Name = align2Name;
results.slidingWindowSize = slidingWindowSize;
results.windowBuffer = windowBuffer;
results.beforeAlign = beforeAlign;
results.afterAlign = afterAlign;
results.stepSize = stepSize;
results.slidingPositions = slidingPositions;
results.d2Metrics = d2Metrics;
results.meanPopActivity = meanPopActivity;
results.sessionName = sessionName;
results.idMatIdx = idMatIdx;
results.binSize = binSize;
results.params.pOrder = pOrder;
results.params.critType = critType;
results.params.minSpikesPerBin = minSpikesPerBin;
results.params.minBinsPerWindow = minBinsPerWindow;
results.params.nShuffles = nShuffles;
results.params.normalizeD2 = normalizeD2;

align1NameFile = strrep(align1Name, ' ', '_');
align2NameFile = strrep(align2Name, ' ', '_');
resultsPath = fullfile(saveDir, sprintf('criticality_behavior_sequences_d2_win%.1f_step%.2f_%s_vs_%s.mat', slidingWindowSize, stepSize, align1NameFile, align2NameFile));
save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);

%% =============================    Plotting    =============================
if ~makePlots
    fprintf('\nSkipping plots (makePlots = false)\n');
    return;
end

fprintf('\n=== Creating Summary Plots ===\n');

if loadResultsForPlotting
    if isempty(resultsFileForPlotting)
        if ~exist('saveDir', 'var') || isempty(saveDir)
            error('saveDir not defined. Specify resultsFileForPlotting or run data loading first.');
        end
        resultsFiles = dir(fullfile(saveDir, 'criticality_behavior_sequences_d2_win*.mat'));
        if isempty(resultsFiles)
            error('No results files in %s. Run analysis first or set resultsFileForPlotting.', saveDir);
        end
        [~, idx] = sort([resultsFiles.datenum], 'descend');
        resultsFileForPlotting = fullfile(saveDir, resultsFiles(idx(1)).name);
    end
    loadedResults = load(resultsFileForPlotting);
    results = loadedResults.results;
    areas = results.areas;
    slidingPositions = results.slidingPositions;
    d2Metrics = results.d2Metrics;
    % Backward compatibility: map old onset/offset to align1/align2
    if isfield(d2Metrics, 'onset') && ~isfield(d2Metrics, 'align1')
        d2Metrics.align1 = d2Metrics.onset;
        d2Metrics.align2 = d2Metrics.offset;
        d2Metrics.align1Normalized = d2Metrics.onsetNormalized;
        d2Metrics.align2Normalized = d2Metrics.offsetNormalized;
    end
    if isfield(results, 'align1Name') && ~isempty(results.align1Name)
        align1Name = results.align1Name;
    else
        align1Name = 'Onset';
    end
    if isfield(results, 'align2Name') && ~isempty(results.align2Name)
        align2Name = results.align2Name;
    else
        align2Name = 'Offset';
    end
    if isfield(results, 'meanPopActivity')
        meanPopActivity = results.meanPopActivity;
        if isfield(meanPopActivity, 'onset') && ~isfield(meanPopActivity, 'align1')
            meanPopActivity.align1 = meanPopActivity.onset;
            meanPopActivity.align2 = meanPopActivity.offset;
        end
    else
        meanPopActivity = [];
    end
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
    if isfield(results, 'slidingWindowSize')
        slidingWindowSize = results.slidingWindowSize;
    end
    if isfield(results, 'stepSize')
        stepSize = results.stepSize;
    end
    if isfield(results, 'params') && isfield(results.params, 'normalizeD2')
        normalizeD2 = results.params.normalizeD2;
    end
end

if ~exist('d2Metrics', 'var')
    error('d2Metrics not found. Run analysis or set loadResultsForPlotting = true.');
end
if ~exist('areasToTest', 'var')
    areasToTest = 1:length(areas);
end
if ~exist('meanPopActivity', 'var')
    meanPopActivity = [];
end

allYVals = [];
for a = areasToTest
    if normalizeD2
        oVals = d2Metrics.align1Normalized{a};
        fVals = d2Metrics.align2Normalized{a};
    else
        oVals = d2Metrics.align1{a};
        fVals = d2Metrics.align2{a};
    end
    allYVals = [allYVals, oVals(~isnan(oVals)), fVals(~isnan(fVals))];
end
if ~isempty(allYVals)
    yMin = min(allYVals);
    yMax = max(allYVals);
    yRange = yMax - yMin;
    if yRange == 0
        yRange = 1;
    end
    yLimits = [yMin - 0.05*yRange, yMax + 0.05*yRange];
else
    yLimits = [0, 1];
end

numCols = length(areasToTest);
figure(1002); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
targetPos = monitorPositions(size(monitorPositions, 1), :);
set(gcf, 'Position', targetPos);

useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(1, numCols, [0.08 0.04], [0.12 0.08], [0.08 0.04]);
else
    ha = zeros(numCols, 1);
    for i = 1:numCols
        ha(i) = subplot(1, numCols, i);
    end
end

% Colors for align1 (green) and align2 (magenta); same for d2 and mean pop
colorAlign1 = [0 0.6 0];
colorAlign2 = [0.8 0 0.8];

for plotIdx = 1:length(areasToTest)
    a = areasToTest(plotIdx);
    axes(ha(plotIdx));
    hold on;

    if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        numNeurons = length(idMatIdx{a});
        neuronStr = sprintf(' (n=%d)', numNeurons);
    else
        neuronStr = '';
    end

    if normalizeD2
        align1Vals = d2Metrics.align1Normalized{a};
        align2Vals = d2Metrics.align2Normalized{a};
        yLabelStr = 'd2 (normalized)';
    else
        align1Vals = d2Metrics.align1{a};
        align2Vals = d2Metrics.align2{a};
        yLabelStr = 'd2';
    end

    % Left y-axis: d2 (solid)
    plot(slidingPositions, align1Vals, '-o', 'Color', colorAlign1, 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', align1Name);
    plot(slidingPositions, align2Vals, '-s', 'Color', colorAlign2, 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', align2Name);

    xlabel('Sliding Position (s)', 'FontSize', 10);
    if plotIdx == 1
        ylabel(yLabelStr, 'FontSize', 10);
    end
    title(sprintf('%s%s', areas{a}, neuronStr), 'FontSize', 11);
    grid on;
    ylim(yLimits);
    set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    if normalizeD2
        yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    end

    % Right y-axis: mean population activity per window (dashed, same colors)
    if ~isempty(meanPopActivity) && isfield(meanPopActivity, 'align1') && a <= length(meanPopActivity.align1) && ...
            ~isempty(meanPopActivity.align1{a}) && isfield(meanPopActivity, 'align2') && a <= length(meanPopActivity.align2)
        yyaxis right;
        plot(slidingPositions, meanPopActivity.align1{a}, '--o', 'Color', colorAlign1, 'LineWidth', 1.5, 'MarkerSize', 4, 'HandleVisibility', 'off');
        plot(slidingPositions, meanPopActivity.align2{a}, '--s', 'Color', colorAlign2, 'LineWidth', 1.5, 'MarkerSize', 4, 'HandleVisibility', 'off');
        ylabel('Mean pop. activity', 'FontSize', 10);
        set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');
    end

    if plotIdx == 1
        legend('Location', 'best', 'FontSize', 9);
    end
end

if exist('sessionName', 'var') && ~isempty(sessionName)
    titlePrefix = [sessionName(1:min(10, length(sessionName))), ' - '];
else
    titlePrefix = '';
end
if normalizeD2
    sgtitle(sprintf('%sd2 (normalized): %s vs %s (Win %.1fs)', titlePrefix, align1Name, align2Name, slidingWindowSize), 'FontSize', 14, 'interpreter', 'none');
else
    sgtitle(sprintf('%sd2: %s vs %s (Win %.1fs)', titlePrefix, align1Name, align2Name, slidingWindowSize), 'FontSize', 14, 'interpreter', 'none');
end

align1NameFile = strrep(align1Name, ' ', '_');
align2NameFile = strrep(align2Name, ' ', '_');
saveFile = fullfile(saveDir, sprintf('criticality_behavior_sequences_d2_win%.1f_step%.2f_%s_vs_%s.png', slidingWindowSize, stepSize, align1NameFile, align2NameFile));
exportgraphics(gcf, saveFile, 'Resolution', 300);
fprintf('Saved plot to: %s\n', saveFile);

fprintf('\n=== Analysis Complete ===\n');
