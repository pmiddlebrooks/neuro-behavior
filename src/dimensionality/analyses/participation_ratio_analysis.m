function results = participation_ratio_analysis(dataStruct, config)
% PARTICIPATION_RATIO_ANALYSIS Sliding window participation ratio from spike data
% See: RecanatesiS_etal_2022_Patterns: A scale-dependent measure of system dimensionality
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config     - Configuration struct with fields:
%     .stepSize          - Step size in seconds (required)
%     .useOptimalBinWindowFunction - Find optimal bin/window (default: true)
%     .binSize           - Bin size in seconds (used if useOptimalBinWindowFunction false)
%     .minSpikesPerBin   - For optimal bin (default: 3)
%     .minBinsPerWindow  - For optimal bin finding (default: 1000)
%     .windowSizeNeuronMultiple - Sliding window size per area = this multiple * nNeurons * binSize
%                                (seconds). Ensures enough time bins for reliable PR (default: 10).
%     .nShuffles         - Number of circular permutations for PR normalization (default: 3).
%     .normalizePR       - If true, normalize PR by mean shuffled PR for comparison across datasets
%                          (default: true). Raw PR is always stored; normalized is plotted/stored.
%     .makePlots         - Create plots (default: true)
%     .saveData          - Save results (default: true)
%     .saveDir           - Save directory (optional, uses dataStruct.saveDir)
%     .nMinNeurons       - Minimum neurons per area (default: 10)
%     .includeM2356      - Include combined M23+M56 area (default: false)
%     .runParallel       - If true, use parfor for area loop (default: false)
%
% Goal:
%   Compute participation ratio PR = (trace(C))^2 / trace(C^2) in sliding windows,
%   where C is the covariance of the mean-centered binned data [time x neurons].
%   Optionally normalize PR by mean shuffled PR (circular permutation per neuron).
%
% Returns:
%   results - Structure with participationRatio (raw), participationRatioNormalized,
%             participationRatioOverNeurons (PR/nNeurons per area), startS,
%             popActivity, popActivityWindows, params

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));

validate_workspace_vars({'spikeTimes', 'spikeClusters', 'areas', 'idMatIdx'}, dataStruct, ...
    'errorMsg', 'Required field', 'source', 'load_sliding_window_data');

if nargin < 2 || isempty(config) || ~isstruct(config)
    config = struct();
end
config = set_participation_ratio_config_defaults(config);

sessionType = dataStruct.sessionType;
areas = dataStruct.areas;
numAreas = length(areas);

if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

% Add combined M2356 area if requested (like criticality_ar_analysis)
if config.includeM2356
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
        fprintf('\n=== Added combined M2356 area ===\n');
        fprintf('M2356: %d neurons (M23: %d, M56: %d)\n', ...
            length(dataStruct.idMatIdx{end}), ...
            length(dataStruct.idMatIdx{idxM23}), ...
            length(dataStruct.idMatIdx{idxM56}));
    elseif isempty(idxM23) || isempty(idxM56)
        fprintf('\n=== Warning: includeM2356 is true but M23 or M56 not found. Skipping M2356 creation. ===\n');
    end
end
if config.includeM2356 && any(strcmp(areas, 'M2356'))
    m2356Idx = find(strcmp(areas, 'M2356'));
    if ~ismember(m2356Idx, areasToTest)
        areasToTest = [areasToTest, m2356Idx];
        fprintf('Added M2356 (index %d) to areasToTest\n', m2356Idx);
    end
end

fprintf('\n=== Participation Ratio Analysis ===\n');
fprintf('Data type: %s, areas: %d\n', sessionType, numAreas);
fprintf('Window size: per area (multiple * nNeurons * binSize), multiple = %d, step: %.3f s\n', ...
    config.windowSizeNeuronMultiple, config.stepSize);
fprintf('nShuffles: %d, normalizePR: %d\n', config.nShuffles, config.normalizePR);

if ~isfield(config, 'saveDir') || isempty(config.saveDir)
    config.saveDir = dataStruct.saveDir;
end

sessionNameForPath = '';
if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
    sessionNameForPath = dataStruct.sessionName;
end

resultsPath = create_results_path('participation_ratio', sessionType, ...
    sessionNameForPath, config.saveDir, 'createDir', true);

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
    timeRange = [0, max(dataStruct.spikeTimes)];
end

% Bin and window sizes
fprintf('\n--- Bin and window sizes ---\n');
[binSize, slidingWindowSize] = find_bin_window_participation_ratio(...
    dataStruct, config, areasToTest, timeRange);

if ~isfield(config, 'stepSize') || isempty(config.stepSize)
    error('stepSize must be provided in config');
end

totalTime = timeRange(2) - timeRange(1);
maxWindowSize = max(slidingWindowSize(areasToTest));
firstCenterTime = maxWindowSize / 2;
lastCenterTime = totalTime - maxWindowSize / 2;
commonCenterTimes = firstCenterTime:config.stepSize:lastCenterTime;

if isempty(commonCenterTimes)
    error('No valid windows. Check windowSizeNeuronMultiple and stepSize.');
end

numWindows = length(commonCenterTimes);
fprintf('Common window centers: %d (step %.3f s)\n', numWindows, config.stepSize);

[participationRatio, participationRatioNormalized, participationRatioOverNeurons, startS, popActivity, popActivityWindows, popActivityFull] = ...
    deal(cell(1, numAreas));

areasToProcess = [];
for a = areasToTest
    nNeurons = length(dataStruct.idMatIdx{a});
    if nNeurons < config.nMinNeurons || isnan(binSize(a))
        participationRatio{a} = [];
        participationRatioNormalized{a} = [];
        participationRatioOverNeurons{a} = [];
        startS{a} = [];
        popActivity{a} = [];
        popActivityWindows{a} = [];
        popActivityFull{a} = [];
        fprintf('  Skip %s: n=%d or invalid bin\n', areas{a}, nNeurons);
    else
        areasToProcess = [areasToProcess, a];
    end
end

if isempty(areasToProcess)
    error('No areas to process.');
end

fprintf('\n--- Processing areas ---\n');
if config.runParallel
    parfor a = areasToProcess
        [participationRatio{a}, participationRatioNormalized{a}, participationRatioOverNeurons{a}, startS{a}, ...
            popActivity{a}, popActivityWindows{a}, popActivityFull{a}] = ...
            process_area_pr(a, areas, dataStruct, binSize, slidingWindowSize, ...
            commonCenterTimes, numWindows, timeRange, config);
    end
else
    for a = areasToProcess
        [participationRatio{a}, participationRatioNormalized{a}, participationRatioOverNeurons{a}, startS{a}, ...
            popActivity{a}, popActivityWindows{a}, popActivityFull{a}] = ...
            process_area_pr(a, areas, dataStruct, binSize, slidingWindowSize, ...
            commonCenterTimes, numWindows, timeRange, config);
    end
end

% Build results
results = struct();
results.sessionType = sessionType;
results.areas = areas;
results.participationRatio = participationRatio;
results.participationRatioNormalized = participationRatioNormalized;
results.participationRatioOverNeurons = participationRatioOverNeurons;
results.startS = startS;
results.popActivity = popActivity;
results.popActivityWindows = popActivityWindows;
results.popActivityFull = popActivityFull;
results.binSize = binSize;
results.slidingWindowSize = slidingWindowSize;  % per-area window (s) = windowSizeNeuronMultiple * nNeurons * binSize
results.params.stepSize = config.stepSize;
results.params.windowSizeNeuronMultiple = config.windowSizeNeuronMultiple;
results.params.windowSizeMax = max(slidingWindowSize(areasToProcess));  % for plot labels
results.params.nShuffles = config.nShuffles;
results.params.normalizePR = config.normalizePR;

if config.saveData
    save(resultsPath, 'results');
    fprintf('Saved to %s\n', resultsPath);
end

if config.makePlots
    plotArgs = {};
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
    end
    if isfield(dataStruct, 'dataBaseName') && ~isempty(dataStruct.dataBaseName)
        plotArgs = [plotArgs, {'dataBaseName', dataStruct.dataBaseName}];
    end
    plotConfig = setup_plotting(config.saveDir, plotArgs{:});
    participation_ratio_plot(results, plotConfig, config, dataStruct);
end
end

function [prOut, prNormOut, prOverNOut, startSOut, popActOut, popActWinOut, popActFullOut] = ...
    process_area_pr(a, areas, dataStruct, binSize, slidingWindowSize, ...
    commonCenterTimes, numWindows, timeRange, config)
    fprintf('Area %s...\n', areas{a});
    tic;
    neuronIDs = dataStruct.idLabel{a};
    aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize(a));
    numTimePoints = size(aDataMat, 1);
    winSamples = round(slidingWindowSize(a) / binSize(a));

    popActOut = mean(aDataMat, 2);
    [startSOut, prOut, prNormOut, prOverNOut, popActWinOut, popActFullOut] = ...
        deal(nan(1, numWindows));

    numNeurons = size(aDataMat, 2);
    for w = 1:numWindows
        centerTime = commonCenterTimes(w);
        startSOut(w) = centerTime;
        [startIdx, endIdx] = calculate_window_indices_from_center(...
            centerTime, slidingWindowSize(a), binSize(a), numTimePoints);

        if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
            continue;
        end

        wDataMat = aDataMat(startIdx:endIdx, :);  % [time x neurons]
        wPopActivity = mean(wDataMat, 2);
        popActWinOut(w) = mean(wPopActivity);
        popActFullOut(w) = popActOut(startIdx + round(winSamples/2) - 1);

        prVal = compute_participation_ratio(wDataMat);
        prOut(w) = prVal;

        % Shuffled PR: circular permutation per neuron, then normalize
        prShuffled = nan(1, config.nShuffles);
        for sh = 1:config.nShuffles
            permutedMat = zeros(size(wDataMat));
            for n = 1:numNeurons
                shiftAmount = randi(size(wDataMat, 1));
                permutedMat(:, n) = circshift(wDataMat(:, n), shiftAmount);
            end
            prShuffled(sh) = compute_participation_ratio(permutedMat);
        end
        meanShuffled = nanmean(prShuffled);
        if config.normalizePR && ~isnan(prVal) && ~isnan(meanShuffled) && meanShuffled > 0
            prNormOut(w) = prVal / meanShuffled;
        else
            prNormOut(w) = nan;
        end
        % PR normalized by neuron count (proportion of used dimensions)
        if ~isnan(prVal) && numNeurons > 0
            prOverNOut(w) = prVal / numNeurons;
        else
            prOverNOut(w) = nan;
        end
    end

    fprintf('  %s done in %.1f min\n', areas{a}, toc/60);
end

function pr = compute_participation_ratio(X)
% X is [time x neurons]. PR is computed on the data matrix with each neuron
% (column) mean-centered. C = covariance of mean-centered X [neurons x neurons].
% PR = (trace(C))^2 / trace(C^2) = (sum(lambda))^2 / sum(lambda.^2).
if size(X, 1) < 2 || size(X, 2) < 1
    pr = nan;
    return;
end
% Mean-center each neuron (column) so covariance reflects shared variance
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

function config = set_participation_ratio_config_defaults(config)
defaults = struct();
defaults.stepSize = 0.1;
defaults.useOptimalBinWindowFunction = true;
defaults.minSpikesPerBin = 3;
defaults.minBinsPerWindow = 1000;
defaults.windowSizeNeuronMultiple = 10;  % per-area window (s) = this * nNeurons * binSize
defaults.nShuffles = 3;
defaults.normalizePR = true;
defaults.makePlots = true;
defaults.saveData = true;
defaults.nMinNeurons = 10;
defaults.includeM2356 = false;
defaults.runParallel = false;
fn = fieldnames(defaults);
for i = 1:length(fn)
    if ~isfield(config, fn{i})
        config.(fn{i}) = defaults.(fn{i});
    end
end
end

function [binSize, slidingWindowSize] = find_bin_window_participation_ratio(...
    dataStruct, config, areasToTest, timeRange)
numAreas = length(dataStruct.areas);
binSize = zeros(1, numAreas);
slidingWindowSize = zeros(1, numAreas);
multiple = config.windowSizeNeuronMultiple;

if config.useOptimalBinWindowFunction
    for a = areasToTest
        neuronIDs = dataStruct.idLabel{a};
        nNeurons = length(dataStruct.idMatIdx{a});
        fr = calculate_firing_rate_from_spikes(...
            dataStruct.spikeTimes, dataStruct.spikeClusters, ...
            neuronIDs, timeRange);
        [binSize(a), ~] = find_optimal_bin_and_window(fr, config.minSpikesPerBin, config.minBinsPerWindow);
        % Window length so that time bins per window = multiple * nNeurons (reliable covariance)
        slidingWindowSize(a) = multiple * nNeurons * binSize(a);
    end
else
    if isfield(config, 'binSize')
        binSize = repmat(config.binSize, 1, numAreas);
    else
        error('binSize required when useOptimalBinWindowFunction is false');
    end
    for a = areasToTest
        nNeurons = length(dataStruct.idMatIdx{a});
        slidingWindowSize(a) = multiple * nNeurons * binSize(a);
    end
end

for a = areasToTest
    nNeurons = length(dataStruct.idMatIdx{a});
    nBins = round(slidingWindowSize(a) / binSize(a));
    fprintf('  %s: bin = %.3f s, win = %.1f s (%d bins, %d neurons, multiple %d)\n', ...
        dataStruct.areas{a}, binSize(a), slidingWindowSize(a), nBins, nNeurons, multiple);
end
end
