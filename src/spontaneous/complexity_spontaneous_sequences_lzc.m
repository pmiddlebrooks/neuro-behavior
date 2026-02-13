%%
% Complexity Spontaneous Behavior Sequences LZC Analysis
% ------------------------------------------------------
% Assesses Lempel-Ziv complexity (LZC) in sliding windows around two
% alignment event types (e.g. onset and offset of behavior sequences).
% Takes the mean of LZC values across windows at each sliding position
% for each alignment type.
%
% Modeled after:
%   - `criticality_behavior_sequences.m` (d2)
%   - `dimensionality_spontaneous_sequences_pr.m` (PR)
%
% Variables:
%   sessionName   - Session name (e.g. 'ag112321_1' or 'ag/ag112321/recording1')
%   align1Times   - Event times for first alignment type (seconds)
%   align2Times   - Event times for second alignment type (seconds)
%   align1Name    - String name for first alignment (e.g. 'Onset')
%   align2Name    - String name for second alignment (e.g. 'Offset')
%   windowSizeNeuronMultiple - Per-area window = this * nNeurons * binSize (seconds)
%   windowBuffer  - Minimum distance from window edge to recording bounds
%   beforeAlign   - Start of sliding range (seconds before alignment point)
%   afterAlign    - End of sliding range (seconds after alignment point)
%   stepSize      - Step size for sliding window (seconds)
%   nShuffles     - Number of shuffles for LZC normalization
%   useBernoulliControl - If true, also compute Bernoulli-normalized LZC
%   includeM2356  - If true, add combined M23+M56 area (default: false)
%
% LZC computation:
%   - Spikes are binned per area.
%   - For each window, [time x neurons] spike counts are concatenated
%     across neurons, binarized (count > 0 => 1), and LZC is computed.
%   - Shuffle normalization: LZC is divided by the mean LZC of shuffled
%     versions of the same binary sequence.
%   - Optional Bernoulli normalization: LZC is divided by the mean LZC of
%     rate-matched Bernoulli sequences.
%
% Requires:
%   - `load_sliding_window_data`
%   - `calculate_firing_rate_from_spikes`
%   - `find_optimal_bin_and_window`
%   - `calculate_window_indices_from_center`
%   - `spontaneous_behavior_sequences` (if you want to derive align1/align2)
%   - `limpel_ziv_complexity` (complexity/ implementation)

%% =============================    Data Loading    =============================
fprintf('\n=== Loading Spontaneous Data (LZC) ===\n');

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

% Paths and options for behavior sequences (if needed to derive align times)
paths = get_paths;
pathParts = strsplit(sessionName, filesep);
subDir = pathParts{1}(1:min(2, numel(pathParts{1})));
sessionFolder = fullfile(paths.spontaneousDataPath, subDir, sessionName);

% If align1Times/align2Times are not already defined, derive them from behavior
if ~exist('align1Times', 'var') || isempty(align1Times) || ...
   ~exist('align2Times', 'var') || isempty(align2Times)
    csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
    if isempty(csvFiles)
        error('No behavior_labels CSV found in %s', sessionFolder);
    end
    if length(csvFiles) > 1
        warning('Multiple behavior_labels CSV files. Using: %s', csvFiles(1).name);
    end
    dataFull = readtable(fullfile(sessionFolder, csvFiles(1).name));

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
        if ~isfield(optsBhv, 'nMinUniqueBhv') || isempty(optsBhv.nMinUniqueBhv)
            optsBhv.nMinUniqueBhv = 2;
        end
    else
        optsBhv.minDur = optsBehaviorSequences.minDur;
        optsBhv.propThreshold = optsBehaviorSequences.propThreshold;
        optsBhv.bufferSec = optsBehaviorSequences.bufferSec;
        optsBhv.behaviorIds = optsBehaviorSequences.behaviorIds;
        if isfield(optsBehaviorSequences, 'nMinUniqueBhv')
            optsBhv.nMinUniqueBhv = optsBehaviorSequences.nMinUniqueBhv;
        else
            optsBhv.nMinUniqueBhv = 2;
        end
    end

    [sequences, times] = spontaneous_behavior_sequences(dataFull, optsBhv); %#ok<ASGLU>
    if isempty(times)
        error('No behavior sequences found for session %s. Check optsBehaviorSequences.', sessionName);
    end
    align1Times = cellfun(@(t) t(1), times);
    align2Times = cellfun(@(t) t(end), times);
end

fprintf('Using %d and %d behavior sequences (%s/%s times in seconds)\n', ...
    length(align1Times), length(align2Times), 'align1', 'align2');

%% =============================    Configuration    =============================
beforeAlign = -2;
afterAlign  =  2;
stepSize = 0.1;
windowBuffer = 0.5;

% LZC analysis parameters (per-area window sizing like PR script)
windowSizeNeuronMultiple = 10;
minSpikesPerBin = 3;
minBinsPerWindow = 1000;
nShuffles = 4;
useBernoulliControl = false;

loadResultsForPlotting = false;
resultsFileForPlotting = '';
makePlots = true;
includeM2356 = false;

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

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end

% Optional: add combined M23+M56 area
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
        m2356Idx = find(strcmp(areas, 'M2356'));
        if ~ismember(m2356Idx, areasToTest)
            areasToTest = [areasToTest, m2356Idx];
        end
    end
end

%% =============================    Optimal Bin Sizes and Window Sizes    =============================
fprintf('\n=== Finding Optimal Bin Sizes and Window Sizes Per Area (LZC) ===\n');

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
    fprintf('Area %s: bin size = %.3f s, win = %.1f s (n=%d)\n', ...
        areas{a}, binSize(a), slidingWindowSize(a), nNeurons);
end

minWindowSize = min(slidingWindowSize(areasToTest));

%% =============================    Valid Align1 and Align2 Times    =============================
fprintf('\n=== Finding Valid %s and %s Times (LZC) ===\n', align1Name, align2Name);

tStart = timeRange(1);
tEnd = timeRange(2);
halfWin = minWindowSize / 2;
winMinCenter = tStart + windowBuffer + halfWin;
winMaxCenter = tEnd - windowBuffer - halfWin;
minEventTime = winMinCenter - afterAlign;
maxEventTime = winMaxCenter - beforeAlign;

validAlign1Indices = [];
validAlign2Indices = [];
for s = 1:length(align1Times)
    if align1Times(s) >= minEventTime && align1Times(s) <= maxEventTime
        validAlign1Indices = [validAlign1Indices, s];  %#ok<AGROW>
    end
end
for s = 1:length(align2Times)
    if align2Times(s) >= minEventTime && align2Times(s) <= maxEventTime
        validAlign2Indices = [validAlign2Indices, s];  %#ok<AGROW>
    end
end

align1TimesValid = align1Times(validAlign1Indices);
align2TimesValid = align2Times(validAlign2Indices);
numAlign1 = length(align1TimesValid);
numAlign2 = length(align2TimesValid);

fprintf('Min window size: %.2f s, buffer: %.2f s\n', minWindowSize, windowBuffer);
fprintf('Valid %s: %d/%d, valid %s: %d/%d\n', ...
    align1Name, numAlign1, length(align1Times), align2Name, numAlign2, length(align2Times));

if numAlign1 == 0 && numAlign2 == 0
    error('No valid %s or %s times. Increase recording time or reduce windowSizeNeuronMultiple/windowBuffer.', align1Name, align2Name);
end

%% =============================    Analysis    =============================
fprintf('\n=== Processing Areas (LZC) ===\n');

slidingPositions = beforeAlign:stepSize:afterAlign;
numSlidingPositions = length(slidingPositions);

lzcMetrics = struct();
lzcMetrics.align1 = cell(1, numAreas);
lzcMetrics.align2 = cell(1, numAreas);
lzcMetrics.align1Normalized = cell(1, numAreas);
lzcMetrics.align2Normalized = cell(1, numAreas);
if useBernoulliControl
    lzcMetrics.align1NormalizedBernoulli = cell(1, numAreas);
    lzcMetrics.align2NormalizedBernoulli = cell(1, numAreas);
end

for a = areasToTest
    lzcMetrics.align1{a} = nan(1, numSlidingPositions);
    lzcMetrics.align2{a} = nan(1, numSlidingPositions);
    lzcMetrics.align1Normalized{a} = nan(1, numSlidingPositions);
    lzcMetrics.align2Normalized{a} = nan(1, numSlidingPositions);
    if useBernoulliControl
        lzcMetrics.align1NormalizedBernoulli{a} = nan(1, numSlidingPositions);
        lzcMetrics.align2NormalizedBernoulli{a} = nan(1, numSlidingPositions);
    end
end

collectedAlign1WindowData = cell(numAreas, numSlidingPositions);
collectedAlign2WindowData = cell(numAreas, numSlidingPositions);
meanPopActivity = struct();
meanPopActivity.align1 = cell(1, numAreas);
meanPopActivity.align2 = cell(1, numAreas);
for aa = areasToTest
    meanPopActivity.align1{aa} = nan(1, numSlidingPositions);
    meanPopActivity.align2{aa} = nan(1, numSlidingPositions);
end

for a = areasToTest
    fprintf('\nCollecting windows for area %s...\n', areas{a});
    tic;

    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);
    winSizeA = slidingWindowSize(a);

    % Collect align1-aligned windows
    for posIdx = 1:numSlidingPositions
        posSec = slidingPositions(posIdx);
        for s = 1:numAlign1
            eventTime = align1TimesValid(s);
            windowCenter = eventTime + posSec;
            winStart = windowCenter - winSizeA / 2;
            winEnd = windowCenter + winSizeA / 2;
            if winStart < tStart + windowBuffer || winEnd > tEnd - windowBuffer
                continue;
            end
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, winSizeA, binSize(a), numTimePoints);
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);
                if isempty(collectedAlign1WindowData{a, posIdx})
                    collectedAlign1WindowData{a, posIdx} = {wDataMat};
                else
                    collectedAlign1WindowData{a, posIdx}{end+1} = wDataMat;
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
            winStart = windowCenter - winSizeA / 2;
            winEnd = windowCenter + winSizeA / 2;
            if winStart < tStart + windowBuffer || winEnd > tEnd - windowBuffer
                continue;
            end
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                windowCenter, winSizeA, binSize(a), numTimePoints);
            if startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx
                wDataMat = aDataMat(startIdx:endIdx, :);
                if isempty(collectedAlign2WindowData{a, posIdx})
                    collectedAlign2WindowData{a, posIdx} = {wDataMat};
                else
                    collectedAlign2WindowData{a, posIdx}{end+1} = wDataMat;
                end
            end
        end
    end

    fprintf('  Area %s window collection completed in %.1f min\n', areas{a}, toc/60);
end

%% =============================    Per-Sliding-Position LZC    =============================
fprintf('\n=== Computing LZC Per Sliding Position (%s vs %s) ===\n', align1Name, align2Name);

for a = areasToTest
    fprintf('\nComputing LZC for area %s...\n', areas{a});

    for posIdx = 1:numSlidingPositions
        % Align1 windows
        if ~isempty(collectedAlign1WindowData{a, posIdx})
            windowDataList = collectedAlign1WindowData{a, posIdx};
            numWindows = numel(windowDataList);
            meanPopActivity.align1{a}(posIdx) = mean(cellfun(@(w) mean(mean(w, 2)), windowDataList));

            lzcPerWindow = nan(1, numWindows);
            lzcNormPerWindow = nan(1, numWindows);
            lzcNormBernPerWindow = nan(1, numWindows);

            for w = 1:numWindows
                wDataMat = windowDataList{w};

                % Concatenate spikes across neurons over time and binarize
                nNeurons = size(wDataMat, 2);
                nSamples = size(wDataMat, 1);
                concatenatedSeq = reshape(wDataMat', nSamples * nNeurons, 1);
                binarySeq = double(concatenatedSeq > 0);

                [lzVal, lzNorm, lzNormBern] = compute_lz_complexity_with_controls( ...
                    binarySeq, nShuffles, useBernoulliControl);

                lzcPerWindow(w) = lzVal;
                lzcNormPerWindow(w) = lzNorm;
                lzcNormBernPerWindow(w) = lzNormBern;
            end

            lzcMetrics.align1{a}(posIdx) = nanmean(lzcPerWindow);
            lzcMetrics.align1Normalized{a}(posIdx) = nanmean(lzcNormPerWindow);
            if useBernoulliControl
                lzcMetrics.align1NormalizedBernoulli{a}(posIdx) = nanmean(lzcNormBernPerWindow);
            end
        end

        % Align2 windows
        if ~isempty(collectedAlign2WindowData{a, posIdx})
            windowDataList = collectedAlign2WindowData{a, posIdx};
            numWindows = numel(windowDataList);
            meanPopActivity.align2{a}(posIdx) = mean(cellfun(@(w) mean(mean(w, 2)), windowDataList));

            lzcPerWindow = nan(1, numWindows);
            lzcNormPerWindow = nan(1, numWindows);
            lzcNormBernPerWindow = nan(1, numWindows);

            for w = 1:numWindows
                wDataMat = windowDataList{w};

                nNeurons = size(wDataMat, 2);
                nSamples = size(wDataMat, 1);
                concatenatedSeq = reshape(wDataMat', nSamples * nNeurons, 1);
                binarySeq = double(concatenatedSeq > 0);

                [lzVal, lzNorm, lzNormBern] = compute_lz_complexity_with_controls( ...
                    binarySeq, nShuffles, useBernoulliControl);

                lzcPerWindow(w) = lzVal;
                lzcNormPerWindow(w) = lzNorm;
                lzcNormBernPerWindow(w) = lzNormBern;
            end

            lzcMetrics.align2{a}(posIdx) = nanmean(lzcPerWindow);
            lzcMetrics.align2Normalized{a}(posIdx) = nanmean(lzcNormPerWindow);
            if useBernoulliControl
                lzcMetrics.align2NormalizedBernoulli{a}(posIdx) = nanmean(lzcNormBernPerWindow);
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
results.lzcMetrics = lzcMetrics;
results.meanPopActivity = meanPopActivity;
results.sessionName = sessionName;
results.idMatIdx = idMatIdx;
results.binSize = binSize;
results.params.windowSizeNeuronMultiple = windowSizeNeuronMultiple;
results.params.minSpikesPerBin = minSpikesPerBin;
results.params.minBinsPerWindow = minBinsPerWindow;
results.params.nShuffles = nShuffles;
results.params.useBernoulliControl = useBernoulliControl;

align1NameFile = strrep(align1Name, ' ', '_');
align2NameFile = strrep(align2Name, ' ', '_');
winMax = max(slidingWindowSize(areasToTest));
resultsPath = fullfile(saveDir, sprintf('complexity_spontaneous_sequences_lzc_win%.1f_step%.2f_%s_vs_%s.mat', ...
    winMax, stepSize, align1NameFile, align2NameFile));
save(resultsPath, 'results');
fprintf('\nSaved results to: %s\n', resultsPath);

%% =============================    Plotting    =============================
if ~makePlots
    fprintf('\nSkipping plots (makePlots = false)\n');
    return;
end

fprintf('\n=== Creating Summary Plots (LZC) ===\n');

if loadResultsForPlotting
    if isempty(resultsFileForPlotting)
        if ~exist('saveDir', 'var') || isempty(saveDir)
            error('saveDir not defined. Specify resultsFileForPlotting or run data loading first.');
        end
        resultsFiles = dir(fullfile(saveDir, 'complexity_spontaneous_sequences_lzc_win*.mat'));
        if isempty(resultsFiles)
            error('No results files in %s. Run analysis first.', saveDir);
        end
        [~, idx] = sort([resultsFiles.datenum], 'descend');
        resultsFileForPlotting = fullfile(saveDir, resultsFiles(idx(1)).name);
    end
    loadedResults = load(resultsFileForPlotting);
    results = loadedResults.results;
    areas = results.areas;
    slidingPositions = results.slidingPositions;
    lzcMetrics = results.lzcMetrics;
    align1Name = results.align1Name;
    align2Name = results.align2Name;
    meanPopActivity = results.meanPopActivity;
    sessionName = results.sessionName;
    idMatIdx = results.idMatIdx;
    slidingWindowSize = results.slidingWindowSize;
    if ~exist('saveDir', 'var') || isempty(saveDir)
        [saveDir, ~, ~] = fileparts(resultsFileForPlotting);
    end
    areasToTest = 1:length(areas);
    if isfield(results.params, 'useBernoulliControl')
        useBernoulliControl = results.params.useBernoulliControl;
    end
end

if ~exist('lzcMetrics', 'var')
    error('lzcMetrics not found. Run analysis or set loadResultsForPlotting = true.');
end
if ~exist('areasToTest', 'var')
    areasToTest = 1:length(areas);
end
if ~exist('meanPopActivity', 'var')
    meanPopActivity = [];
end

% Choose which metric to plot (normalized or Bernoulli-normalized)
plotFieldAlign1 = 'align1Normalized';
plotFieldAlign2 = 'align2Normalized';
yLabelStr = 'LZC (normalized)';
if useBernoulliControl && isfield(lzcMetrics, 'align1NormalizedBernoulli')
    plotFieldAlign1 = 'align1NormalizedBernoulli';
    plotFieldAlign2 = 'align2NormalizedBernoulli';
    yLabelStr = 'LZC (Bernoulli normalized)';
end

% Y-axis limits
allYVals = [];
for a = areasToTest
    vals1 = lzcMetrics.(plotFieldAlign1){a};
    vals2 = lzcMetrics.(plotFieldAlign2){a};
    allYVals = [allYVals, vals1(~isnan(vals1)), vals2(~isnan(vals2))]; %#ok<AGROW>
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
numRows = 1;
figure(1005); clf;
set(gcf, 'Units', 'pixels');
monitorPositions = get(0, 'MonitorPositions');
targetPos = monitorPositions(size(monitorPositions, 1), :);
set(gcf, 'Position', targetPos);

useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(numRows, numCols, [0.08 0.04], [0.12 0.08], [0.08 0.04]);
else
    ha = zeros(numRows * numCols, 1);
    for i = 1:numRows * numCols
        ha(i) = subplot(numRows, numCols, i);
    end
end

colorAlign1 = [0 0.6 0];
colorAlign2 = [0.8 0 0.8];

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

    align1Vals = lzcMetrics.(plotFieldAlign1){a};
    align2Vals = lzcMetrics.(plotFieldAlign2){a};

    plot(slidingPositions, align1Vals, '-o', 'Color', colorAlign1, ...
        'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', align1Name);
    plot(slidingPositions, align2Vals, '-s', 'Color', colorAlign2, ...
        'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', align2Name);

    xlabel('Sliding Position (s)', 'FontSize', 10);
    if plotIdx == 1
        ylabel(yLabelStr, 'FontSize', 10);
    end
    title(sprintf('%s%s', areas{a}, neuronStr), 'FontSize', 11);
    grid on;
    ylim(yLimits);
    set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto');
    set(gca, 'XTickMode', 'auto', 'XTickLabelMode', 'auto');
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');
    yline(1, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'HandleVisibility', 'off');

    % Right-axis mean population activity
    if ~isempty(meanPopActivity) && isfield(meanPopActivity, 'align1') && ...
            a <= length(meanPopActivity.align1) && ~isempty(meanPopActivity.align1{a}) && ...
            isfield(meanPopActivity, 'align2') && a <= length(meanPopActivity.align2)
        yyaxis right;
        plot(slidingPositions, meanPopActivity.align1{a}, '--o', 'Color', colorAlign1, ...
            'LineWidth', 1.5, 'MarkerSize', 4, 'HandleVisibility', 'off');
        plot(slidingPositions, meanPopActivity.align2{a}, '--s', 'Color', colorAlign2, ...
            'LineWidth', 1.5, 'MarkerSize', 4, 'HandleVisibility', 'off');
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
sgtitle(sprintf('%sLZC: %s vs %s (Win~%.1fs)', titlePrefix, align1Name, align2Name, winMax), ...
    'FontSize', 14, 'interpreter', 'none');

align1NameFile = strrep(align1Name, ' ', '_');
align2NameFile = strrep(align2Name, ' ', '_');
saveFile = fullfile(saveDir, sprintf('complexity_spontaneous_sequences_lzc_win%.1f_step%.2f_%s_vs_%s.png', ...
    winMax, stepSize, align1NameFile, align2NameFile));
exportgraphics(gcf, saveFile, 'Resolution', 300);
fprintf('Saved plot to: %s\n', saveFile);

fprintf('\n=== LZC Behavior Sequences Analysis Complete ===\n');

%% =============================    Local Function    =============================
function [lzComplexity, lzNormalized, lzNormalizedBernoulli] = compute_lz_complexity_with_controls(binarySeq, nShuffles, useBernoulliControl)
% COMPUTE_LZ_COMPLEXITY_WITH_CONTROLS
%   Compute Lempel-Ziv complexity with shuffle and optional Bernoulli controls.
%
% Inputs:
%   binarySeq          - Binary sequence (column or row vector)
%   nShuffles          - Number of shuffles for normalization
%   useBernoulliControl - If true, also compute Bernoulli-normalized LZC
%
% Outputs:
%   lzComplexity            - Raw LZC
%   lzNormalized            - LZC / mean(shuffled LZC)
%   lzNormalizedBernoulli   - LZC / mean(Bernoulli LZC) or NaN if disabled

    if nargin < 3
        useBernoulliControl = false;
    end

    try
        % Ensure column vector
        binarySeq = binarySeq(:);

        % Raw LZC
        lzComplexity = limpel_ziv_complexity(binarySeq, 'method', 'binary');

        % Shuffled control
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

        % Bernoulli control
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
        fprintf('Warning: Error computing LZ complexity: %s\n', ME.message);
        lzComplexity = nan;
        lzNormalized = nan;
        lzNormalizedBernoulli = nan;
    end
end

