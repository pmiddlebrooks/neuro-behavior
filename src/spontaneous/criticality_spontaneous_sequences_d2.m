%%
% Criticality Behavior Sequences d2 Analysis
% Assesses d2 criticality in sliding windows around two alignment event
% types (e.g. onset and offset of behavior sequences). Takes the mean of d2
% values across windows at each sliding position for each alignment type.
%
% Run spontaneous_behavior_sequences.m to get alignTimes, sequences,
% sequenceNames, and alignOnIdx
%
% Variables:
%   sessionName   - Session name (e.g. 'ag112321_1' or 'ag/ag112321/recording1')
%   alignTimes    - Cell array of behavior sequences. Typically produced by
%                   `spontaneous_behavior_sequences()`; see that script for details.
%                   alignTimes{k}{occ}(i) is the i'th behavior-start time within the
%                   run-length-compressed sequence occurrence (seconds).
%   sequenceNames - Cell array of human-readable names for each sequence k.
%   alignOnIdx    - Index (into the compressed sequence) specifying which
%                   behavior element to align on.
%   slidingWindowSize - Window size for d2 analysis (seconds)
%   windowBuffer  - Minimum distance from window edge to recording bounds (seconds)
%   beforeAlign   - Start of sliding range (seconds before alignment point)
%   afterAlign    - End of sliding range (seconds after alignment point)
%   stepSize      - Step size for sliding window (seconds)
%   nShuffles     - Number of circular permutations for d2 normalization
%   normalizeD2   - If true, normalize d2 by mean shuffled d2
%   includeM2356   - If true, add combined M23+M56 area and include in analyses/plots (default: false)
%
% Uses load_sliding_window_data('spontaneous', 'spikes', ...) and
% `spontaneous_behavior_sequences.m` for event times.

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



if exist('alignTimes', 'var') && ~isempty(alignTimes)
    if exist('alignOnIdx', 'var') && ~isempty(alignOnIdx)
        fprintf('Found %d behavior sequences (alignTimes), aligning on index %d\n', numel(alignTimes), alignOnIdx);
    else
        fprintf('Found %d behavior sequences (alignTimes)\n', numel(alignTimes));
    end
else
    fprintf('Found %d and %d behavior sequences (align1/align2 times in seconds)\n', length(align1Times), length(align2Times));
end

%% =============================    Configuration    =============================
beforeAlign = -2;
afterAlign  = 2;
slidingWindowSize = 6;
stepSize = 0.1;
windowBuffer = 0.5;
minWindowSize = slidingWindowSize;

% Bin/window selection mode (manual vs. optimal)
useOptimalBinWindowFunction = false;  % If true, choose binSize + slidingWindowSize from spike rates
binSizeManual = 0.025;                % Manual bin size (seconds) when useOptimalBinWindowFunction is false

pOrder = 10;
critType = 2;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;
nShuffles = 12;
normalizeD2 = false;

% Optional neural subsampling configuration for windowed d2
useSubsampling = true;        % If true, subsample neurons within each area
nSubsamples = 30;              % Number of independent subsampling iterations
nNeuronsSubsample = 20;        % Number of neurons per subsample
minNeuronsMultiple = 1.2;      % Minimum neurons required = round(nNeuronsSubsample * minNeuronsMultiple)

pcaFlag = 0;
pcaFirstFlag = 1;
nDim = 4;

loadResultsForPlotting = false;
resultsFileForPlotting = '';
makePlots = true;
includeM2356 = true;  % set true to add combined M23+M56 area (like criticality_ar_analysis)

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

% Optional: add combined M23+M56 area (like criticality_ar_analysis)
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
        numAreas = length(areas);
        idMatIdx = dataStruct.idMatIdx;
        fprintf('\n=== Added combined M2356 area ===\n');
        fprintf('M2356: %d neurons (M23: %d, M56: %d)\n', ...
            length(dataStruct.idMatIdx{end}), ...
            length(dataStruct.idMatIdx{idxM23}), ...
            length(dataStruct.idMatIdx{idxM56}));
        m2356Idx = find(strcmp(areas, 'M2356'));
        if ~ismember(m2356Idx, areasToTest)
            areasToTest = [areasToTest, m2356Idx];
            fprintf('Added M2356 (index %d) to areasToTest\n', m2356Idx);
        end
    elseif isempty(idxM23) || isempty(idxM56)
        fprintf('\n=== Warning: includeM2356 is true but M23 or M56 not found. Skipping M2356. ===\n');
    end
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

if useOptimalBinWindowFunction
    slidingWindowSizeOptimal = nan(1, numAreas);
    for a = areasToTest
        neuronIDs = dataStruct.idLabel{a};
        thisFiringRate = calculate_firing_rate_from_spikes(...
            dataStruct.spikeTimes, dataStruct.spikeClusters, ...
            neuronIDs, timeRange);
        [binSize(a), slidingWindowSizeOptimal(a)] = ...
            find_optimal_bin_and_window(thisFiringRate, minSpikesPerBin, minBinsPerWindow);
        fprintf('Area %s: bin size = %.3f s, optimal window = %.1f s, firing rate = %.2f spikes/s\n', ...
            areas{a}, binSize(a), slidingWindowSizeOptimal(a), thisFiringRate);
    end

    % Use a single slidingWindowSize across areas: minimum of optimal sizes
    validOptimalWindows = slidingWindowSizeOptimal(areasToTest);
    validOptimalWindows = validOptimalWindows(~isnan(validOptimalWindows) & validOptimalWindows > 0);
    if ~isempty(validOptimalWindows)
        slidingWindowSize = min(validOptimalWindows);
        fprintf('Using slidingWindowSize = %.1f s (min optimal window across areas)\n', slidingWindowSize);
    else
        warning('No valid optimal window sizes found; keeping user-defined slidingWindowSize = %.1f s', slidingWindowSize);
    end
else
    if isempty(binSizeManual) || ~isscalar(binSizeManual) || binSizeManual <= 0
        error('When useOptimalBinWindowFunction is false, binSizeManual must be a positive scalar (got %.3f).', binSizeManual);
    end
    binSize(:) = binSizeManual;
    fprintf('Using manual binSize = %.3f s and slidingWindowSize = %.1f s for all areas\n', ...
        binSizeManual, slidingWindowSize);
end

% Ensure minimum window size matches (possibly updated) slidingWindowSize
minWindowSize = slidingWindowSize;

%% =============================    Valid Sequence Event Times    =============================
% Generalizes the previous 2-align implementation to an arbitrary number of sequences.
% Event times are extracted from `alignTimes{seqIdx}{occIdx}` using `alignOnIdx`.

tStart = timeRange(1);
tEnd = timeRange(2);

halfWin = slidingWindowSize / 2;
% For every sliding position the window must stay in [tStart+windowBuffer, tEnd-windowBuffer]
% Worst case: center at event+beforeAlign (leftmost) and event+afterAlign (rightmost)
winMinCenter = tStart + windowBuffer + halfWin;
winMaxCenter = tEnd - windowBuffer - halfWin;
minEventTime = winMinCenter - afterAlign;   % so event+afterAlign - halfWin >= tStart+buffer
maxEventTime = winMaxCenter - beforeAlign;  % so event+beforeAlign + halfWin <= tEnd-buffer

% Build event times per sequence (pre-filter)
if exist('alignTimes', 'var') && ~isempty(alignTimes)
    if ~exist('alignOnIdx', 'var') || isempty(alignOnIdx) || ~isscalar(alignOnIdx)
        error('alignOnIdx must be defined as a scalar when using alignTimes.');
    end
    numSequences = numel(alignTimes);

    if exist('sequenceNames', 'var') && ~isempty(sequenceNames)
        sequenceNamesPlot = sequenceNames;
    else
        sequenceNamesPlot = cell(1, numSequences);
        for seqIdx = 1:numSequences
            if exist('sequences', 'var') && numel(sequences) >= seqIdx && ~isempty(sequences{seqIdx})
                sequenceNamesPlot{seqIdx} = mat2str(sequences{seqIdx});
            else
                sequenceNamesPlot{seqIdx} = sprintf('Seq%d', seqIdx);
            end
        end
    end

    alignEventTimesPerSeq = cell(1, numSequences);
    for seqIdx = 1:numSequences
        seqOcc = alignTimes{seqIdx};
        if isempty(seqOcc)
            alignEventTimesPerSeq{seqIdx} = [];
            continue;
        end

        if iscell(seqOcc)
            nOcc = numel(seqOcc);
            evTimes = nan(1, nOcc);
            for occIdx = 1:nOcc
                tVec = seqOcc{occIdx};
                if ~isempty(tVec) && alignOnIdx <= numel(tVec)
                    evTimes(occIdx) = tVec(alignOnIdx);
                end
            end
            alignEventTimesPerSeq{seqIdx} = evTimes(~isnan(evTimes));
        else
            alignEventTimesPerSeq{seqIdx} = seqOcc(:)';
        end
    end
else
    % Backward compatibility: old two-align workflow
    if ~exist('align1Times', 'var') || ~exist('align2Times', 'var')
        error('Expected alignTimes (and alignOnIdx) or fallback align1Times/align2Times to be defined.');
    end
    numSequences = 2;
    if ~exist('align1Name', 'var') || isempty(align1Name); align1Name = 'Align1'; end
    if ~exist('align2Name', 'var') || isempty(align2Name); align2Name = 'Align2'; end
    sequenceNamesPlot = {align1Name, align2Name};
    alignEventTimesPerSeq = {align1Times(:)', align2Times(:)'};
    alignOnIdx = nan;
end

% Filter event times for validity
eventTimesValidPerSeq = cell(1, numSequences);
numEventsValid = nan(1, numSequences);
numEventsRaw = nan(1, numSequences);
for seqIdx = 1:numSequences
    ev = alignEventTimesPerSeq{seqIdx};
    numEventsRaw(seqIdx) = numel(ev);
    if isempty(ev)
        eventTimesValidPerSeq{seqIdx} = [];
        numEventsValid(seqIdx) = 0;
        continue;
    end
    keepMask = (ev >= minEventTime) & (ev <= maxEventTime);
    eventTimesValidPerSeq{seqIdx} = ev(keepMask);
    numEventsValid(seqIdx) = numel(eventTimesValidPerSeq{seqIdx});
end

fprintf('Sliding window size: %.2f s, buffer: %.2f s\n', slidingWindowSize, windowBuffer);
for seqIdx = 1:numSequences
    fprintf('  %s: %d/%d valid events\n', sequenceNamesPlot{seqIdx}, numEventsValid(seqIdx), numEventsRaw(seqIdx));
end

if all(numEventsValid == 0)
    error('No valid events for any sequence. Increase recording time or reduce slidingWindowSize/windowBuffer.');
end

% Provide align1/align2 variables so the remaining (older) code path still works.
% These will be superseded once the analysis/plotting sections are generalized.
if numSequences >= 1
    align1TimesValid = eventTimesValidPerSeq{1};
    align1Name = sequenceNamesPlot{1};
    numAlign1 = numel(align1TimesValid);
else
    align1TimesValid = [];
    align1Name = 'Align1';
    numAlign1 = 0;
end
if numSequences >= 2
    align2TimesValid = eventTimesValidPerSeq{2};
    align2Name = sequenceNamesPlot{2};
    numAlign2 = numel(align2TimesValid);
else
    align2TimesValid = [];
    align2Name = 'Align2';
    numAlign2 = 0;
end

% Safety: avoid silent empty alignments
if numAlign1 == 0 && numAlign2 == 0
    error('No valid alignments for the first two sequences. Check alignOnIdx and/or windows.');
end

%% =============================    Analysis    =============================
fprintf('\n=== Processing Areas ===\n');

slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

d2Metrics = struct();
% d2Metrics.d2{seqIdx}{areaIdx}(posIdx)
d2Metrics.d2 = cell(1, numSequences);
d2Metrics.d2Normalized = cell(1, numSequences);
for seqIdx = 1:numSequences
    d2Metrics.d2{seqIdx} = cell(1, numAreas);
    d2Metrics.d2Normalized{seqIdx} = cell(1, numAreas);
    for a = areasToTest
        d2Metrics.d2{seqIdx}{a} = nan(1, numSlidingPositions);
        d2Metrics.d2Normalized{seqIdx}{a} = nan(1, numSlidingPositions);
    end
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
        collectedSeqWindows = cell(1, numSequences);
        collectedSeqWindowData = cell(1, numSequences);
        for seqIdx = 1:numSequences
            collectedSeqWindows{seqIdx} = cell(numAreas, numSlidingPositions);
            collectedSeqWindowData{seqIdx} = cell(numAreas, numSlidingPositions);
        end
    end

    % Collect sequence-aligned windows
    for posIdx = 1:numSlidingPositions
        posSec = slidingPositions(posIdx);

        for seqIdx = 1:numSequences
            evTimes = eventTimesValidPerSeq{seqIdx};
            if isempty(evTimes)
                continue;
            end

            for s = 1:numel(evTimes)
                eventTime = evTimes(s);
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

                    % Optionally subsample neurons within this window
                    if useSubsampling
                        nNeuronsTotal = size(wDataMat, 2);
                        minNeuronsRequired = round(nNeuronsSubsample * minNeuronsMultiple);
                        if nNeuronsTotal < minNeuronsRequired
                            continue;
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
                            wPopMatrix(ss, :) = sum(subMat, 2)';
                        end
                        wPopActivity = mean(wPopMatrix, 1)';  % average across subsamples
                    else
                        wPopActivity = sum(wDataMat, 2);
                    end

                    if isempty(collectedSeqWindows{seqIdx}{a, posIdx})
                        collectedSeqWindows{seqIdx}{a, posIdx} = wPopActivity(:)';
                        collectedSeqWindowData{seqIdx}{a, posIdx} = {wDataMat};
                    else
                        collectedSeqWindows{seqIdx}{a, posIdx} = [collectedSeqWindows{seqIdx}{a, posIdx}; wPopActivity(:)'];
                        collectedSeqWindowData{seqIdx}{a, posIdx}{end+1} = wDataMat;
                    end
                end
            end
        end
    end

    fprintf('  Area %s completed in %.1f min\n', areas{a}, toc / 60);
end

%% =============================    Per-Sliding-Position d2    =============================
fprintf('\n=== Computing d2 Per Sliding Position (multi-sequence) ===\n');

% Mean population activity per sliding position (for right y-axis in plots)
meanPopActivity = struct();
meanPopActivity.meanPop = cell(1, numSequences);
for seqIdx = 1:numSequences
    meanPopActivity.meanPop{seqIdx} = cell(1, numAreas);
    for aa = areasToTest
        meanPopActivity.meanPop{seqIdx}{aa} = nan(1, numSlidingPositions);
    end
end

for a = areasToTest
    fprintf('\nComputing d2 for area %s...\n', areas{a});

    for posIdx = 1:numSlidingPositions
        for seqIdx = 1:numSequences
            if isempty(collectedSeqWindows{seqIdx}{a, posIdx})
                continue;
            end

            windowPopMat = collectedSeqWindows{seqIdx}{a, posIdx};  % [numWindows x numBins]
            % Mean pop activity = mean over windows of (mean over time in each window)
            meanPopActivity.meanPop{seqIdx}{a}(posIdx) = mean(mean(windowPopMat, 2));

            numWindows = size(windowPopMat, 1);
            d2PerWindow = nan(1, numWindows);
            d2ShuffledPerWindow = nan(numWindows, nShuffles);
            windowDataList = collectedSeqWindowData{seqIdx}{a, posIdx};  % only needed for shuffles

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
                    wDataMat = windowDataList{w};  % [timeBins x neurons]
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

            d2Metrics.d2{seqIdx}{a}(posIdx) = nanmean(d2PerWindow);
            if normalizeD2
                meanShuffledPerWindow = nanmean(d2ShuffledPerWindow, 2);
                d2NormalizedPerWindow = nan(1, numWindows);
                for w = 1:numWindows
                    if ~isnan(d2PerWindow(w)) && ~isnan(meanShuffledPerWindow(w)) && meanShuffledPerWindow(w) > 0
                        d2NormalizedPerWindow(w) = d2PerWindow(w) / meanShuffledPerWindow(w);
                    end
                end
                d2Metrics.d2Normalized{seqIdx}{a}(posIdx) = nanmean(d2NormalizedPerWindow);
            end
        end
    end
end


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
        resultsFiles = dir(fullfile(saveDir, 'criticality_spontaneous_sequences_d2_win*.mat'));
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
    if isfield(results, 'sequenceNames')
        sequenceNamesPlot = results.sequenceNames;
    else
        sequenceNamesPlot = {};
    end
    if isfield(results, 'alignOnIdx')
        alignOnIdx = results.alignOnIdx;
    end
    % Backward compatibility: map old onset/offset to align1/align2
    if isfield(d2Metrics, 'onset') && ~isfield(d2Metrics, 'align1')
        d2Metrics.align1 = d2Metrics.onset;
        d2Metrics.align2 = d2Metrics.offset;
        d2Metrics.align1Normalized = d2Metrics.onsetNormalized;
        d2Metrics.align2Normalized = d2Metrics.offsetNormalized;
    end

    % Backward compatibility: convert align1/align2 style to d2{seqIdx} style
    if ~isfield(d2Metrics, 'd2') && isfield(d2Metrics, 'align1') && isfield(d2Metrics, 'align2')
        d2Metrics.d2 = {d2Metrics.align1, d2Metrics.align2};
        if isfield(d2Metrics, 'align1Normalized') && isfield(d2Metrics, 'align2Normalized')
            d2Metrics.d2Normalized = {d2Metrics.align1Normalized, d2Metrics.align2Normalized};
        else
            d2Metrics.d2Normalized = {d2Metrics.align1, d2Metrics.align2};
        end
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

    if isempty(sequenceNamesPlot)
        sequenceNamesPlot = {align1Name, align2Name};
    end
    if isfield(results, 'meanPopActivity')
        meanPopActivity = results.meanPopActivity;
        if isfield(meanPopActivity, 'onset') && ~isfield(meanPopActivity, 'align1')
            meanPopActivity.align1 = meanPopActivity.onset;
            meanPopActivity.align2 = meanPopActivity.offset;
        end

        % Backward compatibility: convert mean pop activity to meanPop{seqIdx}{area}
        if ~isfield(meanPopActivity, 'meanPop') && isfield(meanPopActivity, 'align1') && isfield(meanPopActivity, 'align2')
            meanPopActivity.meanPop = {meanPopActivity.align1, meanPopActivity.align2};
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

numSequencesPlot = numel(d2Metrics.d2);
allYVals = [];
for a = areasToTest
    for seqIdx = 1:numSequencesPlot
        if normalizeD2
            vals = d2Metrics.d2Normalized{seqIdx}{a};
        else
            vals = d2Metrics.d2{seqIdx}{a};
        end
        allYVals = [allYVals, vals(~isnan(vals))]; %#ok<AGROW>
    end
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
% Precompute d2/popActivity ratios across sliding positions
ratioMetrics = struct();
ratioMetrics.ratio = cell(1, numSequencesPlot);
for seqIdx = 1:numSequencesPlot
    ratioMetrics.ratio{seqIdx} = cell(1, numAreas);
end

allRatioYVals = [];
for a = areasToTest
    for seqIdx = 1:numSequencesPlot
        if normalizeD2
            d2Vals = d2Metrics.d2Normalized{seqIdx}{a};
        else
            d2Vals = d2Metrics.d2{seqIdx}{a};
        end
        popVals = meanPopActivity.meanPop{seqIdx}{a};

        ratioVals = nan(size(slidingPositions));
        validMask = ~isnan(d2Vals) & ~isnan(popVals) & (popVals ~= 0);
        ratioVals(validMask) = d2Vals(validMask) ./ popVals(validMask);

        ratioMetrics.ratio{seqIdx}{a} = ratioVals;
        allRatioYVals = [allRatioYVals, ratioVals(~isnan(ratioVals))]; %#ok<AGROW>
    end
end

if ~isempty(allRatioYVals)
    rMin = min(allRatioYVals);
    rMax = max(allRatioYVals);
    rRange = rMax - rMin;
    if rRange == 0
        rRange = 1;
    end
    yLimitsRatio = [rMin - 0.05*rRange, rMax + 0.05*rRange];
else
    yLimitsRatio = [0, 1];
end

% Layout:
% - Row 1: d2 sliding-position traces with mean population activity (right axis)
% - Row 2: d2/popActivity sliding-position traces
numCols = length(areasToTest);
numRows = 2;

figure(1002); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
targetPos = monitorPositions(size(monitorPositions, 1), :);
set(gcf, 'Position', targetPos);

useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.12 0.08], [0.08 0.04]);
else
    ha = zeros(numRows * numCols, 1);
    for i = 1:(numRows * numCols)
        ha(i) = subplot(numRows, numCols, i);
    end
end

colors = lines(numSequencesPlot);

% Row 1: d2 traces across sliding positions (with mean population activity on right axis)
plotIdx = 0;
for colIdx = 1:length(areasToTest)
    a = areasToTest(colIdx);
    plotIdx = plotIdx + 1;
    axes(ha(plotIdx));
    hold on;

    if exist('idMatIdx', 'var') && ~isempty(idMatIdx) && a <= length(idMatIdx) && ~isempty(idMatIdx{a})
        numNeurons = length(idMatIdx{a});
        neuronStr = sprintf(' (n=%d)', numNeurons);
    else
        neuronStr = '';
    end

    if normalizeD2
        yLabelStr = 'd2 (normalized)';
    else
        yLabelStr = 'd2';
    end

    % Left y-axis: d2 (solid) for each sequence
    for seqIdx = 1:numSequencesPlot
        if normalizeD2
            yVals = d2Metrics.d2Normalized{seqIdx}{a};
        else
            yVals = d2Metrics.d2{seqIdx}{a};
        end
        plot(slidingPositions, yVals, '-o', 'Color', colors(seqIdx, :), ...
            'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', sequenceNamesPlot{seqIdx});
    end

    xlabel('Sliding Position (s)', 'FontSize', 10);
    if colIdx == 1
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
    yyaxis right;
    if ~isempty(meanPopActivity) && isfield(meanPopActivity, 'meanPop')
        for seqIdx = 1:numSequencesPlot
            plot(slidingPositions, meanPopActivity.meanPop{seqIdx}{a}, '--o', ...
                'Color', colors(seqIdx, :), 'LineWidth', 1.5, 'MarkerSize', 4, ...
                'HandleVisibility', 'off');
        end
        ylabel('Mean pop. activity', 'FontSize', 10);
        set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');
    end

    if colIdx == 1
        legend('Location', 'best', 'FontSize', 9);
    end
end

% Row 2: d2/popActivity traces across sliding positions
for colIdx = 1:length(areasToTest)
    a = areasToTest(colIdx);
    plotIdx = plotIdx + 1;
    axes(ha(plotIdx));
    hold on;

    for seqIdx = 1:numSequencesPlot
        ratioVals = ratioMetrics.ratio{seqIdx}{a};
        if ~isempty(ratioVals) && any(~isnan(ratioVals))
            plot(slidingPositions, ratioVals, '--o', 'Color', colors(seqIdx, :), ...
                'LineWidth', 1.5, 'MarkerSize', 4, ...
                'DisplayName', sequenceNamesPlot{seqIdx});
        end
    end

    if colIdx == 1
        ylabel('d2 / popActivity', 'FontSize', 10);
    end
    xlabel('Sliding Position (s)', 'FontSize', 10);

    grid on;
    if colIdx == 1
        legend('Location', 'best', 'FontSize', 9);
    end
    set(gca, 'XTickLabelMode', 'auto');
    set(gca, 'YTickLabelMode', 'auto');
    ylim(yLimitsRatio);

    % Add vertical line at 0 (alignment point)
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
end

if exist('sessionName', 'var') && ~isempty(sessionName)
    titlePrefix = [sessionName(1:min(10, length(sessionName))), ' - '];
else
    titlePrefix = '';
end
if exist('alignOnIdx', 'var') && ~isempty(alignOnIdx) && ~isnan(alignOnIdx)
    alignPart = sprintf('alignOnIdx=%d', alignOnIdx);
else
    alignPart = 'alignOnIdx=NA';
end

if normalizeD2
    sgtitle(sprintf('%sd2 (normalized) and d2/popActivity (%s, Win %.1fs)', ...
        titlePrefix, alignPart, slidingWindowSize), 'FontSize', 14, 'interpreter', 'none');
else
    sgtitle(sprintf('%sd2 and d2/popActivity (%s, Win %.1fs)', ...
        titlePrefix, alignPart, slidingWindowSize), 'FontSize', 14, 'interpreter', 'none');
end

saveFile = fullfile(saveDir, sprintf('criticality_spontaneous_sequences_d2_allseqs_win%.1f_step%.2f_%s.png', ...
    slidingWindowSize, stepSize, alignPart));
exportgraphics(gcf, saveFile, 'Resolution', 300);
fprintf('Saved plot to: %s\n', saveFile);

fprintf('\n=== Analysis Complete ===\n');



%% =============================    Save Results    =============================
results = struct();
results.areas = areas;
results.sequenceNames = sequenceNamesPlot;
results.alignOnIdx = alignOnIdx;
results.eventTimesPerSeqValid = eventTimesValidPerSeq;
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
results.params.useOptimalBinWindowFunction = useOptimalBinWindowFunction;
results.params.binSizeManual = binSizeManual;

if exist('alignOnIdx', 'var') && ~isempty(alignOnIdx) && ~isnan(alignOnIdx)
    alignOnIdxPart = sprintf('alignOnIdx%d', alignOnIdx);
else
    alignOnIdxPart = 'alignOnIdxNA';
end
resultsPath = fullfile(saveDir, sprintf('criticality_spontaneous_sequences_d2_win%.1f_step%.2f_%s_seqCount%d.mat', ...
    slidingWindowSize, stepSize, alignOnIdxPart, numSequences));
save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);
