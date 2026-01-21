function results = rqa_sliding_analysis(dataStruct, config)
% RQA_SLIDING_ANALYSIS Perform Recurrence Quantitative Analysis (RQA) in sliding windows
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config - Configuration structure with fields:
%     .slidingWindowSize - Window size in seconds (optional, auto-calculated from binSize if not provided)
%     .stepSize - Step size in seconds (optional, calculated if not provided)
%     .nShuffles - Number of shuffles for normalization (default: 3)
%     .binSize - Bin size for spikes (required if dataSource == 'spikes')
%     .minTimeBins - Minimum number of time bins per window (default: 10000, used to auto-calculate slidingWindowSize)
%     .nPCADim - Number of PCA dimensions to use (default: 3)
%     .recurrenceThreshold - Target recurrence rate (default: 0.02 = 2%, or numeric value between 0.01-0.05)
%     .distanceMetric - Distance metric: 'euclidean' or 'cosine' (default: 'euclidean')
%     .nMinNeurons - Minimum number of neurons required (default: 10)
%     .makePlots - Whether to create plots (default: true)
%     .useBernoulliControl - Whether to compute Bernoulli normalized metric (default: true)
%     .saveRecurrencePlots - Whether to compute and store recurrence plots (default: false, saves memory)
%     .includeM2356 - Include combined M23+M56 area (default: false)
%     Note: saveDir is taken from dataStruct.saveDir (set by data loading functions)
%
% Goal:
%   Compute RQA metrics (recurrence rate, determinism, laminarity, trapping time)
%   in sliding windows for spike data. Projects data into PCA space and includes
%   normalization by shuffled and rate-matched Bernoulli controls.
%
% Returns:
%   results - Structure with recurrenceRate, determinism, laminarity, trappingTime,
%             and their normalized versions, plus recurrence plots

% Add sliding_window_prep to path if needed
utilsPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils');
if exist(utilsPath, 'dir')
    addpath(utilsPath);
end

% Add criticality path for find_optimal_bin_and_window function
criticalityPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'criticality');
if exist(criticalityPath, 'dir')
    addpath(criticalityPath);
end

% Add data_prep path for spike times functions
dataPrepPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data_prep');
if exist(dataPrepPath, 'dir')
    addpath(dataPrepPath);
end

% Validate inputs
validate_workspace_vars({'sessionType', 'dataSource', 'areas'}, dataStruct, ...
    'errorMsg', 'Required field', 'source', 'load_sliding_window_data');


dataSource = dataStruct.dataSource;
sessionType = dataStruct.sessionType;
areas = dataStruct.areas;

% Set default includeM2356 if not provided
if ~isfield(config, 'includeM2356')
    config.includeM2356 = true;  % Default to false (opt-in)
end

% Add combined M2356 area if requested and M23 and M56 exist
if config.includeM2356
    idxM23 = find(strcmp(areas, 'M23'));
    idxM56 = find(strcmp(areas, 'M56'));
    if ~isempty(idxM23) && ~isempty(idxM56) && ~any(strcmp(areas, 'M2356'))
        % Create combined M2356 area
        areas{end+1} = 'M2356';
        dataStruct.areas = areas;  % Update dataStruct.areas
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

numAreas = length(areas);
% Get areasToTest
if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:numAreas;
end

% If M2356 was created, ensure it's included in areasToTest
if config.includeM2356 && any(strcmp(areas, 'M2356'))
    m2356Idx = find(strcmp(areas, 'M2356'));
    if ~ismember(m2356Idx, areasToTest)
        areasToTest = [areasToTest, m2356Idx];
        fprintf('Added M2356 (index %d) to areasToTest\n', m2356Idx);
    end
end

fprintf('\n=== RQA Sliding Window Analysis Setup ===\n');
fprintf('Data source: %s\n', dataSource);
fprintf('Number of areas: %d\n', numAreas);
fprintf('PCA dimensions: %d\n', config.nPCADim);

% Validate data source specific requirements
if strcmp(dataSource, 'spikes')
    validate_workspace_vars({'spikeTimes', 'spikeClusters', 'idMatIdx'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
    
    % Calculate time range from spike data
    if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
        timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
    else
        timeRange = [0, max(dataStruct.spikeTimes)];
    end

    % Set default useOptimalBinSize if not provided
    if ~isfield(config, 'useOptimalBinSize')
        config.useOptimalBinSize = true;  % Default to true
    end

    % Set default minSpikesPerBin if not provided
    if ~isfield(config, 'minSpikesPerBin')
        config.minSpikesPerBin = 2.5;  % Default: 2.5 spikes per bin
    end

    % Set default minTimeBins if not provided
    if ~isfield(config, 'minTimeBins') || isempty(config.minTimeBins)
        config.minTimeBins = 10000;
    end

    % Calculate bin size per area (either optimal or user-specified)
    % Always store as binSize vector (length numAreas)
    if config.useOptimalBinSize
        fprintf('Calculating optimal bin size per area (minSpikesPerBin=%.1f)...\n', config.minSpikesPerBin);
        config.binSize = zeros(1, numAreas);

        for a = areasToTest
            aID = dataStruct.idMatIdx{a};
            nNeurons = length(aID);

            if nNeurons < config.nMinNeurons
                % Skip this area, but still need to set a bin size
                config.binSize(a) = nan;
                continue;
            end

            % Calculate overall spike rate (spikes per second) across all neurons
            % Get neuron IDs for this area
            neuronIDs = dataStruct.idLabel{a};
            spikeRate = calculate_firing_rate_from_spikes(dataStruct.spikeTimes, ...
                dataStruct.spikeClusters, neuronIDs, timeRange);

            % Calculate optimal bin size using find_optimal_bin_and_window
            % We only need bin size, so we'll use a dummy minBinsPerWindow
            % The actual window size will be calculated from minTimeBins
            [config.binSize(a), ~] = find_optimal_bin_and_window(spikeRate, config.minSpikesPerBin, 1);

            fprintf('  Area %s: spike rate = %.2f spikes/s, optimal bin size = %.3f s (%.1f ms)\n', ...
                areas{a}, spikeRate, config.binSize(a), config.binSize(a) * 1000);
        end
    else
        % Use user-specified bin size
        if ~isfield(config, 'binSize') || isempty(config.binSize)
            error('binSize must be provided in config for spike data analysis (or set useOptimalBinSize = true)');
        end
        % Convert scalar to vector if needed
        if isscalar(config.binSize)
            config.binSize = repmat(config.binSize, 1, numAreas);
        end
        fprintf('Using user-specified bin size: %.3f s (%.1f ms)\n', ...
            config.binSize(1), config.binSize(1) * 1000);
    end

    % Calculate sliding window size per area from binSize and minTimeBins
    % Always store as slidingWindowSize vector (length numAreas)
    % Convert scalar to vector if needed
    if isfield(config, 'slidingWindowSize') && ~isempty(config.slidingWindowSize)
        if isscalar(config.slidingWindowSize)
            config.slidingWindowSize = repmat(config.slidingWindowSize, 1, numAreas);
        end
        fprintf('Using user-specified window size: %.2f s\n', config.slidingWindowSize(1));
    else
        % Auto-calculate from binSize and minTimeBins
        config.slidingWindowSize = zeros(1, numAreas);
        for a = areasToTest
            if ~isnan(config.binSize(a))
                config.slidingWindowSize(a) = max(config.minWindowSize, config.binSize(a) * config.minTimeBins);
                fprintf('  Area %s: auto-calculated window size: %.2f s (from binSize=%.3f s, minTimeBins=%d)\n', ...
                    areas{a}, config.slidingWindowSize(a), config.binSize(a), config.minTimeBins);
            else
                % Use a default if area was skipped
                config.slidingWindowSize(a) = 60;
            end
        end
    end

    % Calculate step size if not provided
    if ~isfield(config, 'stepSize') || isempty(config.stepSize)
        % Use minimum bin size across areas (excluding NaN) for step size
        validBinSizes = config.binSize(~isnan(config.binSize));
        if ~isempty(validBinSizes)
            minBinSize = min(validBinSizes);
            config.stepSize = minBinSize * 2;  % Default: 2x bin size
        else
            error('No valid bin sizes found. Check nMinNeurons threshold.');
        end
    end

else
    error('RQA analysis currently only supports spike data');
end

% Calculate common centerTime values based on slidingWindowSize and stepSize
% This ensures all areas have aligned windows regardless of their bin sizes
if strcmp(dataSource, 'spikes')
    % Total time from spike data
    totalTime = timeRange(2) - timeRange(1);
end

% Generate common centerTime values
% Use minimum window size for common alignment (before area-specific values)
% Convert scalar to vector if needed for calculation
if isscalar(config.slidingWindowSize)
    minWindowSize = config.slidingWindowSize;
else
    minWindowSize = min(config.slidingWindowSize(~isnan(config.slidingWindowSize)));
end
firstCenterTime = minWindowSize / 2;
lastCenterTime = totalTime - minWindowSize / 2;
commonCenterTimes = firstCenterTime:config.stepSize:lastCenterTime;

if isempty(commonCenterTimes)
    error('No valid windows found. Check slidingWindowSize and stepSize relative to total time.');
end

numWindows = length(commonCenterTimes);
fprintf('\nCommon window centers: %d windows from %.2f s to %.2f s\n', ...
    numWindows, firstCenterTime, lastCenterTime);

% Set default useBernoulliControl if not provided
if ~isfield(config, 'useBernoulliControl')
    config.useBernoulliControl = true;  % Default to true for backward compatibility
end

% Set defaults for other config fields
if ~isfield(config, 'makePlots')
    config.makePlots = true;
end
if ~isfield(config, 'nShuffles')
    config.nShuffles = 3;
end
if ~isfield(config, 'nPCADim')
    config.nPCADim = 3;
end
if ~isfield(config, 'recurrenceThreshold')
    config.recurrenceThreshold = 0.02;  % Default: 2% recurrence rate
end
if ~isfield(config, 'distanceMetric')
    config.distanceMetric = 'euclidean';  % Default: euclidean
end
if ~isfield(config, 'nMinNeurons')
    config.nMinNeurons = 10;  % Default: minimum 10 neurons required
end
if ~isfield(config, 'saveRecurrencePlots')
    config.saveRecurrencePlots = false;  % Default: don't save recurrence plots (saves memory)
end
if ~isfield(config, 'usePerWindowPCA')
    config.usePerWindowPCA = false;  % Default: perform PCA on entire session (original behavior)
end
if ~isfield(config, 'saveData')
    config.saveData = true;  % Default to true (save results)
end

% Validate distance metric
if ~ismember(lower(config.distanceMetric), {'euclidean', 'cosine'})
    error('distanceMetric must be ''euclidean'' or ''cosine''');
end

% Validate recurrence threshold (should be between 0.01 and 0.05 for recurrence rate)
if isnumeric(config.recurrenceThreshold)
    if config.recurrenceThreshold < 0.01 || config.recurrenceThreshold > 0.05
        warning('Recurrence threshold %.4f is outside recommended range (0.01-0.05)', ...
            config.recurrenceThreshold);
    end
end

fprintf('Distance metric: %s\n', config.distanceMetric);
fprintf('Target recurrence rate: %.2f%%\n', config.recurrenceThreshold * 100);

% Initialize results
recurrenceRate = cell(1, numAreas);
determinism = cell(1, numAreas);
laminarity = cell(1, numAreas);
trappingTime = cell(1, numAreas);
recurrenceRateNormalized = cell(1, numAreas);
determinismNormalized = cell(1, numAreas);
laminarityNormalized = cell(1, numAreas);
trappingTimeNormalized = cell(1, numAreas);
recurrenceRateNormalizedBernoulli = cell(1, numAreas);
determinismNormalizedBernoulli = cell(1, numAreas);
laminarityNormalizedBernoulli = cell(1, numAreas);
trappingTimeNormalizedBernoulli = cell(1, numAreas);
recurrencePlots = cell(1, numAreas);  % Store recurrence plots for each area
startS = cell(1, numAreas);

% Initialize behavior proportion if enabled for spontaneous sessions
if strcmp(sessionType, 'spontaneous') && isfield(config, 'behaviorNumeratorIDs') && ...
        isfield(config, 'behaviorDenominatorIDs') && ...
        ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
    behaviorProportion = cell(1, numAreas);
else
    behaviorProportion = cell(1, numAreas);
    for a = 1:numAreas
        behaviorProportion{a} = [];
    end
end

% Filter areas to process: exclude areas with too few neurons or invalid bin sizes
fprintf('\n=== Filtering Areas to Process ===\n');
areasToProcess = [];
areasToSkip = [];

for a = areasToTest
    shouldSkip = false;
    skipReason = '';
    
    if strcmp(dataSource, 'spikes')
        aID = dataStruct.idMatIdx{a};
        nNeurons = length(aID);
        
        % Check minimum number of neurons
        if nNeurons < config.nMinNeurons
            shouldSkip = true;
            skipReason = sprintf('Only %d neurons (minimum required: %d)', nNeurons, config.nMinNeurons);
        end
        
        % Check if bin size is valid
        if ~shouldSkip && isnan(config.binSize(a))
            shouldSkip = true;
            skipReason = 'Invalid bin size (area skipped earlier)';
        end
    end
    
    if shouldSkip
        areasToSkip = [areasToSkip, a];
        fprintf('  Will skip area %s: %s\n', areas{a}, skipReason);
        % Initialize with empty arrays
        recurrenceRate{a} = [];
        determinism{a} = [];
        laminarity{a} = [];
        trappingTime{a} = [];
        recurrenceRateNormalized{a} = [];
        determinismNormalized{a} = [];
        laminarityNormalized{a} = [];
        trappingTimeNormalized{a} = [];
        recurrenceRateNormalizedBernoulli{a} = [];
        determinismNormalizedBernoulli{a} = [];
        laminarityNormalizedBernoulli{a} = [];
        trappingTimeNormalizedBernoulli{a} = [];
        recurrencePlots{a} = {};
        startS{a} = [];
    else
        areasToProcess = [areasToProcess, a];
    end
end

if isempty(areasToProcess)
    error('No valid areas to process. All areas were skipped due to insufficient neurons or invalid bin sizes.');
end

fprintf('  Will process %d area(s): %s\n', length(areasToProcess), strjoin(areas(areasToProcess), ', '));

% Analysis loop
fprintf('\n=== Processing Areas ===\n');


parfor a = areasToProcess
% for a = areasToProcess
    fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataSource);

    tic;

    if strcmp(dataSource, 'spikes')
        % ========== Spike Data Analysis ==========
        aID = dataStruct.idMatIdx{a};
        nNeurons = length(aID);

        % Determine number of PCA dimensions to use
        actualPCADim = min(config.nPCADim, nNeurons);
        fprintf('  Using %d PCA dimensions (requested: %d, neurons: %d)\n', ...
            actualPCADim, config.nPCADim, nNeurons);

        % Get neuron IDs for this area
        neuronIDs = dataStruct.idLabel{a};
        
        % Bin data using area-specific bin size from spike times
        aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
            neuronIDs, timeRange, config.binSize(a));
        % Ensure single precision for memory efficiency
        aDataMat = single(aDataMat);
        numTimePoints = size(aDataMat, 1);

        % Use area-specific window size
        areaWindowSize = config.slidingWindowSize(a);

        % % Calculate window size in samples
        % winSamples = round(areaWindowSize / config.binSize(a));
        % if winSamples < 1
        %     winSamples = 1;
        % end

        % fprintf('  Time points: %d, Window size: %.2fs, Window samples: %d, Bin size: %.3f s\n', ...
        %     numTimePoints, areaWindowSize, winSamples, config.binSize(a));

        % Perform PCA based on config option
        if config.usePerWindowPCA
            fprintf('  Using per-window PCA (drift correction mode)...\n');
            pcaDataFull = [];  % Not used in per-window mode
        else
            % Perform PCA on entire session data for this area
            fprintf('  Performing PCA on entire session data...\n');
            [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
            % Use first actualPCADim dimensions, ensure single precision
            pcaDataFull = single(score(:, 1:actualPCADim));
            fprintf('  PCA completed %s: using %d dimensions (explained variance: %.1f%%)\n', ...
                areas{a}, actualPCADim, sum(explained(1:actualPCADim)));
        end

        % Initialize arrays
        recurrenceRate{a} = nan(1, numWindows);
        determinism{a} = nan(1, numWindows);
        laminarity{a} = nan(1, numWindows);
        trappingTime{a} = nan(1, numWindows);
        recurrenceRateNormalized{a} = nan(1, numWindows);
        determinismNormalized{a} = nan(1, numWindows);
        laminarityNormalized{a} = nan(1, numWindows);
        trappingTimeNormalized{a} = nan(1, numWindows);
        recurrenceRateNormalizedBernoulli{a} = nan(1, numWindows);
        determinismNormalizedBernoulli{a} = nan(1, numWindows);
        laminarityNormalizedBernoulli{a} = nan(1, numWindows);
        trappingTimeNormalizedBernoulli{a} = nan(1, numWindows);
        if config.saveRecurrencePlots
            recurrencePlots{a} = cell(1, numWindows);  % Store recurrence plot for each window
        else
            recurrencePlots{a} = {};  % Empty if not saving
        end
        startS{a} = nan(1, numWindows);

        % Initialize behavior proportion array if enabled
        if strcmp(sessionType, 'spontaneous') && isfield(config, 'behaviorNumeratorIDs') && ...
                isfield(config, 'behaviorDenominatorIDs') && ...
                ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
            behaviorProportion{a} = nan(1, numWindows);
        else
            behaviorProportion{a} = [];
        end

        % Process each window using common centerTime
        for w = 1:numWindows
            centerTime = commonCenterTimes(w);
            startS{a}(w) = centerTime;

            % Convert centerTime to indices for this area's binning
            % Use area-specific window size
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                centerTime, areaWindowSize, config.binSize(a), numTimePoints);

            % Check if window is valid (within bounds)
            if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                % Window is out of bounds for this area, skip
                continue;
            end

            % Extract window data and perform PCA based on config option
            if config.usePerWindowPCA
                % Perform PCA on this window's data only (drift correction)
                windowData = aDataMat(startIdx:endIdx, :);
                [~, scoreWindow, ~, ~, explainedWindow, ~] = pca(windowData);
                % Use first actualPCADim dimensions, ensure single precision
                pcaData = single(scoreWindow(:, 1:actualPCADim));
            else
                % Extract window data from pre-computed PCA space
                % Use the same PCA space computed for the entire session
                pcaData = single(pcaDataFull(startIdx:endIdx, :));
            end

            % Calculate RQA metrics
            if config.saveRecurrencePlots
                [recurrenceRate{a}(w), determinism{a}(w), laminarity{a}(w), ...
                    trappingTime{a}(w), recurrencePlots{a}{w}] = ...
                    compute_rqa_metrics(pcaData, config.recurrenceThreshold, config.distanceMetric);
            else
                [recurrenceRate{a}(w), determinism{a}(w), laminarity{a}(w), ...
                    trappingTime{a}(w), ~] = ...
                    compute_rqa_metrics(pcaData, config.recurrenceThreshold, config.distanceMetric);
            end

            % Normalize by shuffled version
            % Shuffle spiking data using neuron-by-neuron circular permutation before PCA
            shuffledRR = nan(1, config.nShuffles);
            shuffledDET = nan(1, config.nShuffles);
            shuffledLAM = nan(1, config.nShuffles);
            shuffledTT = nan(1, config.nShuffles);

            % Pre-allocate permutedDataMat in single precision and reuse across shuffles
            permutedDataMat = zeros(size(aDataMat), 'single');

            for s = 1:config.nShuffles
                % Circularly shift each neuron's activity independently
                % Reuse permutedDataMat (in-place operations)
                for n = 1:nNeurons
                    shiftAmount = randi(size(aDataMat, 1));
                    permutedDataMat(:, n) = circshift(aDataMat(:, n), shiftAmount);
                end

                % OPTIMIZATION: Always do PCA only on window data (not full session)
                % This dramatically reduces memory usage, especially for large windows
                shuffledWindowData = permutedDataMat(startIdx:endIdx, :);
                [~, scoreShuffled, ~, ~, ~, ~] = pca(shuffledWindowData);
                % Use first actualPCADim dimensions, ensure single precision
                shuffledPCA = single(scoreShuffled(:, 1:actualPCADim));
                
                % Explicitly clear intermediate matrices to free memory immediately
                % clear shuffledWindowData scoreShuffled;

                [shuffledRR(s), shuffledDET(s), shuffledLAM(s), shuffledTT(s)] = ...
                    compute_rqa_metrics(shuffledPCA, config.recurrenceThreshold, config.distanceMetric);
                
                % Explicitly clear shuffledPCA to free memory immediately
                % clear shuffledPCA;
            end
            
            % Clear permutedDataMat after all shuffles are done
            % clear permutedDataMat;

            meanShuffledRR = nanmean(shuffledRR);
            meanShuffledDET = nanmean(shuffledDET);
            meanShuffledLAM = nanmean(shuffledLAM);
            meanShuffledTT = nanmean(shuffledTT);

            if meanShuffledRR > 0
                recurrenceRateNormalized{a}(w) = recurrenceRate{a}(w) / meanShuffledRR;
            else
                recurrenceRateNormalized{a}(w) = nan;
            end

            if meanShuffledDET > 0
                determinismNormalized{a}(w) = determinism{a}(w) / meanShuffledDET;
            else
                determinismNormalized{a}(w) = nan;
            end

            if meanShuffledLAM > 0
                laminarityNormalized{a}(w) = laminarity{a}(w) / meanShuffledLAM;
            else
                laminarityNormalized{a}(w) = nan;
            end

            if meanShuffledTT > 0
                trappingTimeNormalized{a}(w) = trappingTime{a}(w) / meanShuffledTT;
            else
                trappingTimeNormalized{a}(w) = nan;
            end

            % Normalize by rate-matched Bernoulli control (optional)
            if config.useBernoulliControl
                % Generate random PCA data with same statistics
                bernoulliRR = nan(1, config.nShuffles);
                bernoulliDET = nan(1, config.nShuffles);
                bernoulliLAM = nan(1, config.nShuffles);
                bernoulliTT = nan(1, config.nShuffles);

                for s = 1:config.nShuffles
                    if config.usePerWindowPCA
                        % For per-window PCA, generate random window data and perform PCA
                        % Generate random data matching the window's statistics
                        windowData = aDataMat(startIdx:endIdx, :);
                        bernoulliWindowData = zeros(size(windowData), 'single');
                        for n = 1:nNeurons
                            firingRate = mean(windowData(:, n));
                            bernoulliWindowData(:, n) = single(rand(size(windowData, 1), 1) < firingRate);
                        end
                        % Perform PCA on Bernoulli window data
                        [~, bernoulliScore, ~, ~, ~, ~] = pca(bernoulliWindowData);
                        bernoulliPCA = single(bernoulliScore(:, 1:actualPCADim));
                        % clear bernoulliWindowData bernoulliScore;  % Free memory
                    else
                        % Generate random data with same mean and std as original PCA space
                        bernoulliPCA = zeros(size(pcaData), 'single');
                        for dim = 1:actualPCADim
                            bernoulliPCA(:, dim) = single(randn(size(pcaData, 1), 1)) * std(pcaData(:, dim)) + mean(pcaData(:, dim));
                        end
                    end

                    [bernoulliRR(s), bernoulliDET(s), bernoulliLAM(s), bernoulliTT(s)] = ...
                        compute_rqa_metrics(bernoulliPCA, config.recurrenceThreshold, config.distanceMetric);
                    
                    % Explicitly clear bernoulliPCA to free memory
                    % clear bernoulliPCA;
                end

                meanBernoulliRR = nanmean(bernoulliRR);
                meanBernoulliDET = nanmean(bernoulliDET);
                meanBernoulliLAM = nanmean(bernoulliLAM);
                meanBernoulliTT = nanmean(bernoulliTT);

                if meanBernoulliRR > 0
                    recurrenceRateNormalizedBernoulli{a}(w) = recurrenceRate{a}(w) / meanBernoulliRR;
                else
                    recurrenceRateNormalizedBernoulli{a}(w) = nan;
                end

                if meanBernoulliDET > 0
                    determinismNormalizedBernoulli{a}(w) = determinism{a}(w) / meanBernoulliDET;
                else
                    determinismNormalizedBernoulli{a}(w) = nan;
                end

                if meanBernoulliLAM > 0
                    laminarityNormalizedBernoulli{a}(w) = laminarity{a}(w) / meanBernoulliLAM;
                else
                    laminarityNormalizedBernoulli{a}(w) = nan;
                end

                if meanBernoulliTT > 0
                    trappingTimeNormalizedBernoulli{a}(w) = trappingTime{a}(w) / meanBernoulliTT;
                else
                    trappingTimeNormalizedBernoulli{a}(w) = nan;
                end
            else
                recurrenceRateNormalizedBernoulli{a}(w) = nan;  % Not computed
                determinismNormalizedBernoulli{a}(w) = nan;
                laminarityNormalizedBernoulli{a}(w) = nan;
                trappingTimeNormalizedBernoulli{a}(w) = nan;
            end

            % Calculate behavior proportion if enabled for spontaneous sessions
            if strcmp(sessionType, 'spontaneous') && isfield(dataStruct, 'bhvID') && ...
                    ~isempty(dataStruct.bhvID) && isfield(config, 'behaviorNumeratorIDs') && ...
                    isfield(config, 'behaviorDenominatorIDs') && ...
                    ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
                % Convert window time range to bhvID indices (bhvID is at fsBhv sampling rate)
                if isfield(dataStruct, 'fsBhv') && ~isempty(dataStruct.fsBhv)
                    fsBhv = dataStruct.fsBhv;
                    bhvBinSize = 1 / fsBhv;  % Behavior bin size in seconds
                    winStartTime = centerTime - areaWindowSize / 2;
                    winEndTime = centerTime + areaWindowSize / 2;
                    bhvStartIdx = round(winStartTime / bhvBinSize) + 1;
                    bhvEndIdx = round(winEndTime / bhvBinSize);
                    bhvStartIdx = max(1, bhvStartIdx);
                    bhvEndIdx = min(length(dataStruct.bhvID), bhvEndIdx);

                    if bhvStartIdx <= bhvEndIdx
                        % Calculate proportion inline
                        windowBhvID = dataStruct.bhvID(bhvStartIdx:bhvEndIdx);
                        numeratorCount = sum(ismember(windowBhvID, config.behaviorNumeratorIDs));
                        denominatorCount = sum(ismember(windowBhvID, config.behaviorDenominatorIDs));
                        if denominatorCount > 0
                            behaviorProportion{a}(w) = numeratorCount / denominatorCount;
                        else
                            behaviorProportion{a}(w) = nan;
                        end
                    end
                end
            end
        end

    end

    fprintf('  Area %s completed in %.1f minutes\n', areas{a}, toc/60);
end


% Build results structure
results = struct();
results.dataSource = dataSource;
results.areas = areas;
results.startS = startS;
results.recurrenceRate = recurrenceRate;
results.determinism = determinism;
results.laminarity = laminarity;
results.trappingTime = trappingTime;
results.recurrenceRateNormalized = recurrenceRateNormalized;
results.determinismNormalized = determinismNormalized;
results.laminarityNormalized = laminarityNormalized;
results.trappingTimeNormalized = trappingTimeNormalized;
results.recurrenceRateNormalizedBernoulli = recurrenceRateNormalizedBernoulli;
results.determinismNormalizedBernoulli = determinismNormalizedBernoulli;
results.laminarityNormalizedBernoulli = laminarityNormalizedBernoulli;
results.trappingTimeNormalizedBernoulli = trappingTimeNormalizedBernoulli;
if config.saveRecurrencePlots
    results.recurrencePlots = recurrencePlots;
end

% Store behavior proportion if calculated
if strcmp(sessionType, 'spontaneous') && isfield(config, 'behaviorNumeratorIDs') && ...
        isfield(config, 'behaviorDenominatorIDs') && ...
        ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
    results.behaviorProportion = behaviorProportion;
    results.params.behaviorNumeratorIDs = config.behaviorNumeratorIDs;
    results.params.behaviorDenominatorIDs = config.behaviorDenominatorIDs;
else
    results.behaviorProportion = cell(1, numAreas);
    for a = 1:numAreas
        results.behaviorProportion{a} = [];
    end
end

results.params.stepSize = config.stepSize;
results.params.nShuffles = config.nShuffles;
results.params.nPCADim = config.nPCADim;
results.params.recurrenceThreshold = config.recurrenceThreshold;
results.params.useBernoulliControl = config.useBernoulliControl;
results.params.binSize = config.binSize;  % Area-specific bin sizes (vector)
results.params.slidingWindowSize = config.slidingWindowSize;  % Area-specific window sizes (vector)
results.params.distanceMetric = config.distanceMetric;
results.params.minTimeBins = config.minTimeBins;
results.params.nMinNeurons = config.nMinNeurons;
results.params.saveRecurrencePlots = config.saveRecurrencePlots;
results.params.useOptimalBinSize = config.useOptimalBinSize;
results.params.minSpikesPerBin = config.minSpikesPerBin;
results.params.usePerWindowPCA = config.usePerWindowPCA;
results.sessionType = sessionType;

% Setup results path (always create for potential plotting use)
% Get saveDir from dataStruct (set by data loading functions)
if ~isfield(dataStruct, 'saveDir') || isempty(dataStruct.saveDir)
    error('dataStruct.saveDir must be set by data loading function');
end

sessionNameForPath = '';
if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
    sessionNameForPath = dataStruct.sessionName;
end

% Handle sessionName with subdirectories (e.g., "kw/kw092821")
% For spontaneous sessions, saveDir already includes the full path (subjectID/recording),
% so we don't need to create additional subdirectories
actualSaveDir = dataStruct.saveDir;
if ~strcmp(sessionType, 'spontaneous') && ~isempty(sessionNameForPath) && contains(sessionNameForPath, filesep)
    % Session name contains path separators, extract first directory
    % Only do this for non-spontaneous sessions (spontaneous already has full path in saveDir)
    pathParts = strsplit(sessionNameForPath, filesep);
    if length(pathParts) > 1
        % Create subdirectory in saveDir
        actualSaveDir = fullfile(dataStruct.saveDir, pathParts{1});
    end
end

% Add filename suffix with PCA dimension and drift flag
% Always include PCA dimensions in RQA filenames
filenameSuffix = sprintf('_pca%d', config.nPCADim);
if config.usePerWindowPCA
    filenameSuffix = [filenameSuffix, '_drift'];
end
resultsPath = create_results_path('rqa', sessionType, ...
    sessionNameForPath, actualSaveDir, 'dataSource', dataSource, ...
    'filenameSuffix', filenameSuffix);

% Save results if requested
if config.saveData
    % Ensure directory exists before saving (including all parent directories)
    resultsDir = fileparts(resultsPath);
    if ~isempty(resultsDir)
        % mkdir creates all parent directories automatically
        [status, msg] = mkdir(resultsDir);
        if ~status
            error('Failed to create results directory %s: %s', resultsDir, msg);
        end
        % Double-check it was created
        if ~exist(resultsDir, 'dir')
            error('Results directory %s still does not exist after mkdir', resultsDir);
        end
    end
    
    % Remove recurrence plots from saved results (too large)
    % Keep them in memory for plotting only if saveRecurrencePlots is true
    resultsToSave = results;
    if isfield(resultsToSave, 'recurrencePlots')
        resultsToSave = rmfield(resultsToSave, 'recurrencePlots');
    end
    save(resultsPath, 'resultsToSave');
    % Remove recurrence plots from memory too (to save space)
    results = resultsToSave;
    clear resultsToSave;
    
    fprintf('\nSaved results to: %s\n', resultsPath);
    if config.saveRecurrencePlots
        fprintf('Note: Recurrence plots computed but excluded from saved file (too large)\n');
    end
else
    fprintf('\nSkipping save (config.saveData = false)\n');
end

% Store resultsPath in results for plotting function
results.resultsPath = resultsPath;

% Plotting
if config.makePlots
    plotArgs = {};
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
    end
    if isfield(dataStruct, 'dataBaseName') && ~isempty(dataStruct.dataBaseName)
        plotArgs = [plotArgs, {'dataBaseName', dataStruct.dataBaseName}];
    end
    plotConfig = setup_plotting(dataStruct.saveDir, plotArgs{:});
    rqa_sliding_plot(results, plotConfig, config, dataStruct);
end
end

function [RR, DET, LAM, TT, recurrencePlot] = compute_rqa_metrics(data, targetRR, distanceMetric)
% COMPUTE_RQA_METRICS Compute Recurrence Quantitative Analysis metrics
%
% Variables:
%   data - [nTimePoints x nDimensions] matrix (PCA projected data)
%   targetRR - Target recurrence rate (e.g., 0.02 for 2%, between 0.01-0.05)
%   distanceMetric - Distance metric: 'euclidean' or 'cosine'
%
% Returns:
%   RR - Recurrence Rate (density of recurrence points)
%   DET - Determinism (proportion of recurrence points forming diagonal lines)
%   LAM - Laminarity (proportion of recurrence points forming vertical lines)
%   TT - Trapping Time (average length of vertical lines)
%   recurrencePlot - Binary recurrence matrix

nPoints = size(data, 1);

if nPoints < 2
    RR = 0;
    DET = 0;
    LAM = 0;
    TT = 0;
    recurrencePlot = false(nPoints, nPoints);  % Use logical instead of zeros
    return;
end

% Ensure data is single precision for memory efficiency
if ~isa(data, 'single')
    data = single(data);
end

% Compute distance matrix based on specified metric
if strcmpi(distanceMetric, 'cosine')
    % Cosine distance: 1 - cosine similarity
    % Normalize data to unit vectors for cosine distance
    dataNorm = data ./ (sqrt(sum(data.^2, 2)) + eps);  % Add eps to avoid division by zero
    distMatrix = 1 - (dataNorm * dataNorm');  % Cosine distance (will be single if data is single)
    % Ensure non-negative (numerical precision issues)
    distMatrix = max(distMatrix, 0);
    clear dataNorm;  % Free memory
else
    % Euclidean distance (default) - pdist2 will return single if input is single
    distMatrix = pdist2(data, data);
end

% Fix recurrence rate and solve for threshold (ε)
% We want: RR = (number of recurrence points) / (total possible points excluding diagonal)
% Total possible = nPoints * (nPoints - 1)
% Target number of recurrence points = targetRR * nPoints * (nPoints - 1)
totalPossible = nPoints * (nPoints - 1);
targetRecurrencePoints = targetRR * totalPossible;

% Get all off-diagonal distances
offDiagMask = ~eye(nPoints);
offDiagDistances = distMatrix(offDiagMask);

% Sort distances to find threshold that gives target recurrence rate
sortedDistances = sort(offDiagDistances);

% Find threshold: we want targetRecurrencePoints smallest distances
if targetRecurrencePoints >= length(sortedDistances)
    % If target is >= all possible points, use max distance
    thresholdValue = max(sortedDistances);
elseif targetRecurrencePoints <= 1
    % If target is very small, use min distance
    thresholdValue = sortedDistances(1);
else
    % Use the distance at the targetRecurrencePoints-th position
    thresholdValue = sortedDistances(round(targetRecurrencePoints));
end

% Clear intermediate arrays to free memory (but keep distMatrix for recurrence plot)
clear offDiagDistances sortedDistances;

% Create recurrence plot (binary matrix) using solved threshold
% Use logical array for memory efficiency (1 byte per element vs 8 bytes for double)
recurrencePlot = logical(distMatrix <= thresholdValue);

% Clear distMatrix after creating recurrence plot to free memory
clear distMatrix;

% Remove main diagonal (self-recurrence)
recurrencePlot = recurrencePlot & ~eye(nPoints);  % Keep as logical

% Recurrence Rate: density of recurrence points (excluding diagonal)
totalPossible = nPoints * (nPoints - 1);
RR = sum(recurrencePlot(:)) / totalPossible;

% Find diagonal lines (for determinism)
% Look for sequences of 1s along diagonals
minLineLength = 2;  % Minimum line length to count
diagonalLines = [];

% Check all diagonals (excluding main diagonal which is all zeros)
for offset = [-(nPoints-1):-1, 1:(nPoints-1)]
    diagSeq = diag(recurrencePlot, offset);
    if length(diagSeq) >= minLineLength
        % Find consecutive sequences of 1s
        % Convert to row vector for processing
        if iscolumn(diagSeq)
            diagSeq = diagSeq';
        end
        % Find runs of consecutive 1s
        diffSeq = diff([0, diagSeq, 0]);
        starts = find(diffSeq == 1);
        ends = find(diffSeq == -1) - 1;
        for i = 1:length(starts)
            lineLength = ends(i) - starts(i) + 1;
            if lineLength >= minLineLength
                diagonalLines = [diagonalLines, lineLength];
            end
        end
    end
end

% Determinism: proportion of recurrence points in diagonal lines
totalRecurrencePoints = sum(recurrencePlot(:));
if ~isempty(diagonalLines) && totalRecurrencePoints > 0
    totalDiagonalPoints = sum(diagonalLines);
    DET = totalDiagonalPoints / totalRecurrencePoints;
else
    DET = 0;
end

% Find vertical lines (for laminarity and trapping time)
% Vertical lines = horizontal lines in the recurrence plot
verticalLines = [];
for col = 1:nPoints
    colSeq = recurrencePlot(:, col);
    % Find consecutive sequences of 1s
    diffSeq = diff([0; colSeq; 0]);
    starts = find(diffSeq == 1);
    ends = find(diffSeq == -1) - 1;
    for i = 1:length(starts)
        lineLength = ends(i) - starts(i) + 1;
        if lineLength >= minLineLength
            verticalLines = [verticalLines, lineLength];
        end
    end
end

% Laminarity: proportion of recurrence points in vertical lines
if ~isempty(verticalLines) && totalRecurrencePoints > 0
    totalVerticalPoints = sum(verticalLines);
    LAM = totalVerticalPoints / totalRecurrencePoints;
else
    LAM = 0;
end

% Trapping Time: average length of vertical lines
if ~isempty(verticalLines)
    TT = mean(verticalLines);
else
    TT = 0;
end
end

