function results = lzc_sliding_analysis(dataStruct, config)
% LZC_SLIDING_ANALYSIS Perform Lempel-Ziv complexity sliding window analysis
% See: https://www.nature.com/articles/srep46421#Sec2
%      Increased spontaneous MEG signal diversity for psychoactive doses of ketamine, LSD and psilocybin
%      Schartner et al 2017 Sci Reports
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config - Configuration structure with fields:
%     .slidingWindowSize - Window size in seconds
%     .stepSize - Step size in seconds (optional, calculated if not provided)
%     .nShuffles - Number of shuffles for normalization (default: 3)
%     .lfpLowpassFreq - Low-pass filter frequency for LFP (default: 80)
%     .binSize - Bin size for spikes (required if dataSource == 'spikes')
%     .makePlots - Whether to create plots (default: true)
%     .minDataPoints - Minimum data points per window for optimization (default: 100000)
%     .useBernoulliControl - Whether to compute Bernoulli normalized metric (default: true)
%     Note: saveDir is taken from dataStruct.saveDir (set by data loading functions)
%
% Goal:
%   Compute Lempel-Ziv complexity in sliding windows for spike or LFP data.
%   Includes normalization by shuffled and rate-matched Bernoulli controls.
%
% Returns:
%   results - Structure with lzComplexity, lzComplexityNormalized,
%             lzComplexityNormalizedBernoulli, startS, and params

    % Add sliding_window_prep to path if needed
    utilsPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils');
    if exist(utilsPath, 'dir')
        addpath(utilsPath);
    end
    
    % Validate inputs
    validate_workspace_vars({'sessionType', 'dataSource', 'areas'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
    

    dataSource = dataStruct.dataSource;
    sessionType = dataStruct.sessionType;
    areas = dataStruct.areas;
    numAreas = length(areas);
    
    % Get areasToTest
    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:numAreas;
    end
    
    fprintf('\n=== Complexity Sliding Window Analysis Setup ===\n');
    fprintf('Data source: %s\n', dataSource);
    fprintf('Number of areas: %d\n', numAreas);
    % Handle scalar or vector window size
    if isscalar(config.slidingWindowSize)
        fprintf('Window size: %.2f s\n', config.slidingWindowSize);
    else
        fprintf('Window size: %.2f s (area-specific)\n', config.slidingWindowSize(1));
    end
    
    % Validate data source specific requirements
    if strcmp(dataSource, 'spikes')
        validate_workspace_vars({'dataMat', 'idMatIdx'}, dataStruct, ...
            'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
        
        % Calculate bin size per area (either optimal or user-specified)
        % Always store as binSize vector (length numAreas)
        if config.useOptimalBinSize
            fprintf('Calculating optimal bin size per area...\n');
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
                areaData = dataStruct.dataMat(:, aID);
                totalSpikes = sum(areaData(:));
                totalTime = size(areaData, 2) * size(areaData, 1) / 1000;  % Convert ms to seconds
                spikeRate = totalSpikes / totalTime;  % Spikes per second
                
                % Calculate optimal bin size: binSize = minSpikesPerBin / spikeRate
                % Round to nearest millisecond (0.001 s)
                optimalBinSize = config.minSpikesPerBin / spikeRate;
                config.binSize(a) = ceil(optimalBinSize / 0.001) * 0.001;  % Round to nearest ms
                
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
        
        % Calculate step size if not provided
        if ~isfield(config, 'stepSize') || isempty(config.stepSize)
            % Use minimum bin size across areas (excluding NaN)
            validBinSizes = config.binSize(~isnan(config.binSize));
            if ~isempty(validBinSizes)
                minBinSize = min(validBinSizes);
                config.stepSize = minBinSize * 2;  % Default: 2x bin size
            else
                error('No valid bin sizes found. Check nMinNeurons threshold.');
            end
        end
        
    elseif strcmp(dataSource, 'lfp')
        validate_workspace_vars({'lfpPerArea', 'opts'}, dataStruct, ...
            'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
        
        if ~isfield(dataStruct.opts, 'fsLfp')
            error('opts.fsLfp must be defined in dataStruct for LFP analysis');
        end
        
        fprintf('LFP sampling rate: %.1f Hz\n', dataStruct.opts.fsLfp);
        fprintf('Low-pass filter: %.1f Hz\n', config.lfpLowpassFreq);
        
        % Calculate step size if not provided
        if ~isfield(config, 'stepSize') || isempty(config.stepSize)
            % Use minimum window size for step size calculation
            if isscalar(config.slidingWindowSize)
                config.stepSize = config.slidingWindowSize / 10;  % Default: 10 steps per window
            else
                config.stepSize = min(config.slidingWindowSize) / 10;
            end
        end
        fprintf('Step size: %.3f s\n', config.stepSize);
    else
        error('Invalid dataSource. Must be ''spikes'' or ''lfp''.');
    end
    
    % Calculate common centerTime values based on slidingWindowSize and stepSize
    % This ensures all areas have aligned windows regardless of their bin sizes
    if strcmp(dataSource, 'spikes')
        % Total time from original data (in seconds, assuming 1000 Hz)
        totalTime = size(dataStruct.dataMat, 1) / 1000;
    elseif strcmp(dataSource, 'lfp')
        % For LFP, we'll calculate from the first area's signal length
        % (all areas should have same length)
        rawSignal = dataStruct.lfpPerArea(:, areasToTest(1));
        fsRaw = dataStruct.opts.fsLfp;
        totalTime = length(rawSignal) / fsRaw;
    end
    
    % Generate common centerTime values
    % Use minimum window size for common alignment (before area-specific optimization)
    % Convert scalar to vector if needed for calculation
    if isscalar(config.slidingWindowSize)
        minWindowSize = config.slidingWindowSize;
    else
        minWindowSize = min(config.slidingWindowSize);
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
    
    % Set default minDataPoints if not provided
    if ~isfield(config, 'minDataPoints') || isempty(config.minDataPoints)
        config.minDataPoints = 2*10^5;
    end
    
    % Set default useBernoulliControl if not provided
    if ~isfield(config, 'useBernoulliControl')
        config.useBernoulliControl = true;  % Default to true for backward compatibility
    end
    
    % Set default useOptimalWindowSize if not provided
    if ~isfield(config, 'useOptimalWindowSize')
        config.useOptimalWindowSize = false;  % Default to false (use user-specified window size)
    end
    
    % Set default minSlidingWindowSize and maxSlidingWindowSize if useOptimalWindowSize is true
    if config.useOptimalWindowSize
        if ~isfield(config, 'minSlidingWindowSize') || isempty(config.minSlidingWindowSize)
            config.minSlidingWindowSize = 5;  % Default minimum window size (seconds)
        end
        if ~isfield(config, 'maxSlidingWindowSize') || isempty(config.maxSlidingWindowSize)
            config.maxSlidingWindowSize = 60;  % Default maximum window size (seconds)
        end
    end
    
    % Calculate window size per area (either optimal or user-specified)
    % Always store as slidingWindowSize vector (length numAreas)
    % Window size is optimized to have at least minDataPoints data points if useOptimalWindowSize is true
    % Constrained to be between 5s and 20s
    fprintf('\n=== Calculating Window Sizes ===\n');
    % Initialize all areas with original window size
    % Convert scalar to vector if needed
    if isscalar(config.slidingWindowSize)
        config.slidingWindowSize = repmat(config.slidingWindowSize, 1, numAreas);
    end
    slidingWindowSize = config.slidingWindowSize;
    
     for a = areasToTest
        if strcmp(dataSource, 'spikes')
            aID = dataStruct.idMatIdx{a};
            nNeurons = length(aID);
            
            if nNeurons < config.nMinNeurons || isnan(config.binSize(a))
                slidingWindowSize(a) = config.slidingWindowSize(a);  % Use original if area is skipped
                continue;
            end
            
            if config.useOptimalWindowSize
                % Calculate minimum window size needed: totalDataPoints = nNeurons × (windowSize / binSize)
                % We need: nNeurons × (windowSize / binSize) >= minDataPoints
                % So: windowSize >= (minDataPoints × binSize) / nNeurons
                minRequiredWindowSize = (config.minDataPoints * config.binSize(a)) / nNeurons;
                
                % Constrain to 5-20s range and use minimum that satisfies requirement
                slidingWindowSize(a) = ceil(max(config.minSlidingWindowSize, min(config.maxSlidingWindowSize, minRequiredWindowSize)));
                
                % Calculate actual data points with this window size
                actualDataPoints = nNeurons * (slidingWindowSize(a) / config.binSize(a));
                
                fprintf('  Area %s: %d neurons, binSize=%.3fs -> window=%.2fs (%.0f data points)\n', ...
                    areas{a}, nNeurons, config.binSize(a), slidingWindowSize(a), actualDataPoints);
            else
                % Use user-specified window size (already set)
                fprintf('  Area %s: using user-specified window size: %.2fs\n', ...
                    areas{a}, slidingWindowSize(a));
            end
            
        elseif strcmp(dataSource, 'lfp')
            % For LFP, calculate window size based on sampling rate if optimizing
            if config.useOptimalWindowSize
                % For LFP, data points = windowSize × samplingRate
                % We need: windowSize × fsRaw >= minDataPoints
                % So: windowSize >= minDataPoints / fsRaw
                fsRaw = dataStruct.opts.fsLfp;
                minRequiredWindowSize = config.minDataPoints / fsRaw;
                
                % Constrain to 5-20s range
                slidingWindowSize(a) = ceil(max(config.minSlidingWindowSize, min(config.maxSlidingWindowSize, minRequiredWindowSize)));
                
                % Calculate actual data points
                actualDataPoints = slidingWindowSize(a) * fsRaw;
                
                fprintf('  Area %s: fs=%.1fHz -> window=%.2fs (%.0f data points)\n', ...
                    areas{a}, fsRaw, slidingWindowSize(a), actualDataPoints);
            else
                % Use user-specified window size (already set)
                fprintf('  Area %s: using user-specified window size: %.2fs\n', ...
                    areas{a}, slidingWindowSize(a));
            end
        end
    end
    
    % Initialize results
    lzComplexity = cell(1, numAreas);
    lzComplexityNormalized = cell(1, numAreas);
    lzComplexityNormalizedBernoulli = cell(1, numAreas);
    startS = cell(1, numAreas);
    
    % Initialize behavior proportion if enabled for naturalistic sessions
    if strcmp(sessionType, 'naturalistic') && isfield(config, 'behaviorNumeratorIDs') && ...
            isfield(config, 'behaviorDenominatorIDs') && ...
            ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
        behaviorProportion = cell(1, numAreas);
    else
        behaviorProportion = cell(1, numAreas);
        for a = 1:numAreas
            behaviorProportion{a} = [];
        end
    end
    
    % Analysis loop
    fprintf('\n=== Processing Areas ===\n');
        
    parfor a = areasToTest
    % for a = areasToTest
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataSource);
        
        % Check minimum number of neurons for spike data
        if strcmp(dataSource, 'spikes')
            aID = dataStruct.idMatIdx{a};
            nNeurons = length(aID);
            if nNeurons < config.nMinNeurons
                fprintf('  Skipping area %s: Only %d neurons (minimum required: %d)\n', ...
                    areas{a}, nNeurons, config.nMinNeurons);
                % Initialize with NaN values
                lzComplexity{a} = [];
                lzComplexityNormalized{a} = [];
                lzComplexityNormalizedBernoulli{a} = [];
                startS{a} = [];
                behaviorProportion{a} = [];
                continue;
            end
        end
        
        tic;
        
        if strcmp(dataSource, 'spikes')
            % ========== Spike Data Analysis ==========
            aID = dataStruct.idMatIdx{a};
            
            % Check if bin size is valid for this area
            if isnan(config.binSize(a))
                fprintf('  Skipping area %s: Invalid bin size (area skipped earlier)\n', areas{a});
                continue;
            end
            
            % Bin data using bin size for this area
            aDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), config.binSize(a));
            numTimePoints = size(aDataMat, 1);
            
            % Use area-specific window size
            areaWindowSize = slidingWindowSize(a);
            
            % Calculate window size in samples for this area
            winSamples = round(areaWindowSize / config.binSize(a));
            if winSamples < 1
                winSamples = 1;
            end
            
            fprintf('  Time points: %d, Window size: %.2fs, Window samples: %d, Bin size: %.3f s\n', ...
                numTimePoints, areaWindowSize, winSamples, config.binSize(a));
            
            % Initialize arrays
            lzComplexity{a} = nan(1, numWindows);
            lzComplexityNormalized{a} = nan(1, numWindows);
            lzComplexityNormalizedBernoulli{a} = nan(1, numWindows);
            startS{a} = nan(1, numWindows);
            
            % Initialize behavior proportion array if enabled
            if strcmp(sessionType, 'naturalistic') && isfield(config, 'behaviorNumeratorIDs') && ...
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
                    centerTime, slidingWindowSize(a), config.binSize(a), numTimePoints);
                
                % Check if window is valid (within bounds)
                if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                    % Window is out of bounds for this area, skip
                    continue;
                end
                
                % Extract window data [nSamples x nNeurons]
                wData = aDataMat(startIdx:endIdx, :);
                
                % Concatenate across neurons over time
                nNeurons = size(wData, 2);
                nSamples = size(wData, 1);
                concatenatedSeq = reshape(wData', nSamples * nNeurons, 1);
                
                % Binarize: any value > 0 becomes 1, 0 stays 0
                binarySeq = double(concatenatedSeq > 0);
                
                % Calculate Lempel-Ziv complexity with controls
                [lzComplexity{a}(w), lzComplexityNormalized{a}(w), ...
                    lzComplexityNormalizedBernoulli{a}(w)] = ...
                    compute_lz_complexity_with_controls(binarySeq, config.nShuffles, config.useBernoulliControl);
                
                % Calculate behavior proportion if enabled for naturalistic sessions
                if strcmp(sessionType, 'naturalistic') && isfield(dataStruct, 'bhvID') && ...
                        ~isempty(dataStruct.bhvID) && isfield(config, 'behaviorNumeratorIDs') && ...
                        isfield(config, 'behaviorDenominatorIDs') && ...
                        ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
                    % Convert window time range to bhvID indices (bhvID is at fsBhv sampling rate)
                    if isfield(dataStruct, 'fsBhv') && ~isempty(dataStruct.fsBhv)
                        fsBhv = dataStruct.fsBhv;
                        bhvBinSize = 1 / fsBhv;  % Behavior bin size in seconds
                        winStartTime = centerTime - slidingWindowSize(a) / 2;
                        winEndTime = centerTime + slidingWindowSize(a) / 2;
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
            
        elseif strcmp(dataSource, 'lfp')
            % ========== LFP Data Analysis ==========
            rawSignal = dataStruct.lfpPerArea(:, a);
            fsRaw = dataStruct.opts.fsLfp;
            
            % Low-pass filter at specified frequency
            if config.lfpLowpassFreq < fsRaw / 2
                filteredSignal = lowpass(rawSignal, config.lfpLowpassFreq, fsRaw);
            else
                warning('Low-pass frequency (%.1f Hz) >= Nyquist (%.1f Hz). Skipping filter.', ...
                    config.lfpLowpassFreq, fsRaw / 2);
                filteredSignal = rawSignal;
            end
            
            numTimePoints = length(filteredSignal);
            
            % Use area-specific window size
            areaWindowSize = slidingWindowSize(a);
            
            % Calculate window size in samples
            winSamples = round(areaWindowSize * fsRaw);
            if winSamples < 1
                winSamples = 1;
            end
            
            fprintf('  Time points: %d, Window size: %.2fs, Window samples: %d, Sampling rate: %.1f Hz\n', ...
                numTimePoints, areaWindowSize, winSamples, fsRaw);
            
            % Initialize arrays
            lzComplexity{a} = nan(1, numWindows);
            lzComplexityNormalized{a} = nan(1, numWindows);
            lzComplexityNormalizedBernoulli{a} = nan(1, numWindows);
            startS{a} = nan(1, numWindows);
            
            % Initialize behavior proportion array if enabled
            if strcmp(sessionType, 'naturalistic') && isfield(config, 'behaviorNumeratorIDs') && ...
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
                
                % Convert centerTime to indices for LFP (sampling rate = fsRaw)
                % Use area-specific window size
                [startIdx, endIdx] = calculate_window_indices_from_center(...
                    centerTime, slidingWindowSize(a), 1/fsRaw, numTimePoints);
                
                % Check if window is valid (within bounds)
                if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                    % Window is out of bounds for this area, skip
                    continue;
                end
                
                % Extract window signal
                wSignal = filteredSignal(startIdx:endIdx);
                
                % Binarize: compare each sample to window mean
                windowMean = mean(wSignal);
                binarySeq = double(wSignal > windowMean);
                
                % Calculate Lempel-Ziv complexity with controls
                [lzComplexity{a}(w), lzComplexityNormalized{a}(w), ...
                    lzComplexityNormalizedBernoulli{a}(w)] = ...
                    compute_lz_complexity_with_controls(binarySeq, config.nShuffles, config.useBernoulliControl);
                
                % Calculate behavior proportion if enabled for naturalistic sessions
                if strcmp(sessionType, 'naturalistic') && isfield(dataStruct, 'bhvID') && ...
                        ~isempty(dataStruct.bhvID) && isfield(config, 'behaviorNumeratorIDs') && ...
                        isfield(config, 'behaviorDenominatorIDs') && ...
                        ~isempty(config.behaviorNumeratorIDs) && ~isempty(config.behaviorDenominatorIDs)
                    % Convert window time range to bhvID indices (bhvID is at fsBhv sampling rate)
                    if isfield(dataStruct, 'fsBhv') && ~isempty(dataStruct.fsBhv)
                        fsBhv = dataStruct.fsBhv;
                        bhvBinSize = 1 / fsBhv;  % Behavior bin size in seconds
                        winStartTime = centerTime - slidingWindowSize(a) / 2;
                        winEndTime = centerTime + slidingWindowSize(a) / 2;
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
    results.lzComplexity = lzComplexity;
    results.lzComplexityNormalized = lzComplexityNormalized;
    results.lzComplexityNormalizedBernoulli = lzComplexityNormalizedBernoulli;
    
    % Store behavior proportion if calculated
    if strcmp(sessionType, 'naturalistic') && isfield(config, 'behaviorNumeratorIDs') && ...
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
    results.params.minDataPoints = config.minDataPoints;
    results.params.useBernoulliControl = config.useBernoulliControl;
    
    if strcmp(dataSource, 'spikes')
        results.params.binSize = config.binSize;  % Area-specific bin sizes (vector)
        results.params.slidingWindowSize = slidingWindowSize;  % Area-specific window sizes (vector)
        results.sessionType = sessionType;
    elseif strcmp(dataSource, 'lfp')
        results.params.lfpLowpassFreq = config.lfpLowpassFreq;
        results.params.fsLfp = dataStruct.opts.fsLfp;
    end
    
    % Save results
    % Get saveDir from dataStruct (set by data loading functions)
    if ~isfield(dataStruct, 'saveDir') || isempty(dataStruct.saveDir)
        error('dataStruct.saveDir must be set by data loading function');
    end
    
    sessionNameForPath = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionNameForPath = dataStruct.sessionName;
    end
    
    % Handle sessionName with subdirectories (e.g., "kw/kw092821")
    % For naturalistic sessions, saveDir already includes the full path (subjectID/recording),
    % so we don't need to create additional subdirectories
    actualSaveDir = dataStruct.saveDir;
    if ~strcmp(sessionType, 'naturalistic') && ~isempty(sessionNameForPath) && contains(sessionNameForPath, filesep)
        % Session name contains path separators, extract first directory
        % Only do this for non-naturalistic sessions (naturalistic already has full path in saveDir)
        pathParts = strsplit(sessionNameForPath, filesep);
        if length(pathParts) > 1
            % Create subdirectory in saveDir
            actualSaveDir = fullfile(dataStruct.saveDir, pathParts{1});
        end
    end
    
    resultsPath = create_results_path('lzc', sessionType, ...
        sessionNameForPath, actualSaveDir, 'dataSource', dataSource);
    
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
    
    save(resultsPath, 'results');
    fprintf('\nSaved results to: %s\n', resultsPath);
    
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
        lzc_sliding_plot(results, plotConfig, config, dataStruct);
    end
end

function [lzComplexity, lzNormalized, lzNormalizedBernoulli] = compute_lz_complexity_with_controls(binarySeq, nShuffles, useBernoulliControl)
% COMPUTE_LZ_COMPLEXITY_WITH_CONTROLS Compute LZ complexity with shuffle and optional Bernoulli controls
%
% Variables:
%   binarySeq - Binary sequence
%   nShuffles - Number of shuffles for normalization
%   useBernoulliControl - Whether to compute Bernoulli normalized metric (default: true)
    
    if nargin < 3
        useBernoulliControl = true;  % Default to true for backward compatibility
    end
    
    try
        % Calculate original LZ complexity
        lzComplexity = limpel_ziv_complexity(binarySeq, 'method', 'binary');
        
        % Normalize by shuffled version
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
        
        % Normalize by rate-matched Bernoulli control (optional)
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
            lzNormalizedBernoulli = nan;  % Not computed
        end
    catch ME
        fprintf('    Warning: Error computing LZ complexity: %s\n', ME.message);
        lzComplexity = nan;
        lzNormalized = nan;
        lzNormalizedBernoulli = nan;
    end
end


