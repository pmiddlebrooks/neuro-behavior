function results = criticality_ar_analysis(dataStruct, config)
% CRITICALITY_AR_ANALYSIS Perform d2 and mrBr criticality sliding window analysis
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config - Configuration structure with fields:
%     .slidingWindowSize - Window size in seconds
%     .binSize - Bin size in seconds (optional, calculated if useOptimalBinWindowFunction)
%     .analyzeD2 - Compute d2 (default: true)
%     .analyzeMrBr - Compute mrBr (default: false)
%     .pcaFlag - Use PCA (default: 0)
%     .pcaFirstFlag - Use first nDim if 1, last nDim if 0 (default: 1)
%     .nDim - Number of PCA dimensions (default: 4)
%     .useOptimalBinWindowFunction - Find optimal bin/window (default: true)
%     .enablePermutations - Perform circular permutations (default: true)
%     .nShuffles - Number of permutations (default: 3)
%     .analyzeModulation - Split into modulated/unmodulated (default: false)
%     .makePlots - Create plots (default: true)
%     .saveDir - Save directory (optional, uses dataStruct.saveDir)
%     .includeM2356 - Include combined M23+M56 area (default: false)
%
% Goal:
%   Compute d2 and/or mrBr criticality measures in sliding windows for spike data.
%   Supports PCA, modulation analysis, and permutation testing.
%
% Returns:
%   results - Structure with d2, mrBr, startS, popActivity, and params

    % Add paths
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data_prep'));
    
    % Validate inputs
    validate_workspace_vars({'sessionType', 'spikeTimes', 'spikeClusters', 'areas', 'idMatIdx'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
    
    % Set defaults only if config is not supplied or is empty
    if nargin < 2 || isempty(config) || ~isstruct(config)
    config = set_config_defaults(config);
    end
    
    sessionType = dataStruct.sessionType;
    areas = dataStruct.areas;
    numAreas = length(areas);
    
    % Set default includeM2356 if not provided
    if ~isfield(config, 'includeM2356')
        config.includeM2356 = false;  % Default to false (opt-in)
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
    
    fprintf('\n=== Criticality AR Analysis Setup ===\n');
    fprintf('Data type: %s\n', sessionType);
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('Window size: %.2f s\n', config.slidingWindowSize);
    fprintf('Analyze d2: %d, Analyze mrBr: %d\n', config.analyzeD2, config.analyzeMrBr);
    
    % Create filename suffix based on PCA flag
    if config.pcaFlag
        filenameSuffix = '_pca';
    else
        filenameSuffix = '';
    end
    
    % Setup results path (always create for potential plotting use)
    if ~isfield(config, 'saveData')
        config.saveData = true;  % Default to true if not set
    end
    
    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end
    
    sessionNameForPath = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionNameForPath = dataStruct.sessionName;
    end
    
    resultsPath = create_results_path('criticality_ar', sessionType, ...
        sessionNameForPath, config.saveDir, 'filenameSuffix', filenameSuffix);
    
    % Load spike data for modulation analysis if needed
    if config.analyzeModulation
        if ~isfield(dataStruct, 'spikeData') || isempty(dataStruct.spikeData)
            fprintf('\n=== Loading spike data for modulation analysis ===\n');
            if strcmp(sessionType, 'reach')
                if ~isfield(dataStruct, 'dataR')
                    error('dataR must be available for reach data modulation analysis');
                end
                dataStruct.spikeData = dataStruct.dataR.CSV(:,1:2);
            else
                error('spikeData must be loaded by load_sliding_window_data() for modulation analysis');
            end
        end
    end
    
    % Perform modulation analysis if requested
    if config.analyzeModulation
        modulationResults = perform_modulation_analysis(dataStruct, config);
    else
        modulationResults = cell(1, numAreas);
    end
    
    % Calculate time range from spike data
    fprintf('\n--- Using spike times for on-demand binning ---\n');
    if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
        timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
    else
        timeRange = [0, max(dataStruct.spikeTimes)];
    end
    
    % For PCA with spike times, we'll bin at a temporary bin size first
    % Use a small bin size (1ms) for PCA calculation
    if config.pcaFlag
        fprintf('\n--- Step 1-2: PCA on original data (binned at 1ms for PCA) ---\n');
        reconstructedDataMat = cell(1, numAreas);
        tempBinSize = 0.001;  % 1ms for PCA calculation
        for a = areasToTest
            neuronIDs = dataStruct.idLabel{a};
            % Bin at 1ms for PCA
            thisDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIDs, timeRange, tempBinSize);
            [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, min(6, forDim));
            nDim = 1:forDim;
            reconstructedDataMat{a} = score(:,nDim) * coeff(:,nDim)' + mu;
        end
    else
        reconstructedDataMat = [];  % Not needed if no PCA
        tempBinSize = [];  % Not used if no PCA
    end
    
    % Find bin and window sizes (either optimal or user-specified)
    % Always store as binSize and slidingWindowSize vectors (length numAreas)
    fprintf('\n--- Step 3: Finding bin and window sizes ---\n');
    if config.pcaFlag
        % Use PCA-reconstructed data for parameter optimization
        [binSize, slidingWindowSize, ...
            binSizeModulated, binSizeUnmodulated, ...
            slidingWindowSizeModulated, slidingWindowSizeUnmodulated] = ...
            find_optimal_parameters_from_spikes_pca(reconstructedDataMat, config, modulationResults, areasToTest, timeRange, tempBinSize, areas, dataStruct);
    else
        % Use original spike times for parameter optimization
        [binSize, slidingWindowSize, ...
            binSizeModulated, binSizeUnmodulated, ...
            slidingWindowSizeModulated, slidingWindowSizeUnmodulated] = ...
            find_optimal_parameters_from_spikes(dataStruct, config, modulationResults, areasToTest, timeRange);
    end
    slidingWindowSize(slidingWindowSize < 10) = 10;
    % Initialize modulation parameters if not already set
    if ~config.analyzeModulation
        if ~exist('binSizeModulated', 'var') || isempty(binSizeModulated)
            binSizeModulated = nan(1, numAreas);
        end
        if ~exist('binSizeUnmodulated', 'var') || isempty(binSizeUnmodulated)
            binSizeUnmodulated = nan(1, numAreas);
        end
        if ~exist('slidingWindowSizeModulated', 'var') || isempty(slidingWindowSizeModulated)
            slidingWindowSizeModulated = nan(1, numAreas);
        end
        if ~exist('slidingWindowSizeUnmodulated', 'var') || isempty(slidingWindowSizeUnmodulated)
            slidingWindowSizeUnmodulated = nan(1, numAreas);
        end
    end
    
    % Validate stepSize is provided
    if ~isfield(config, 'stepSize') || isempty(config.stepSize)
        error('stepSize must be provided in config');
    end
    
    % Calculate common centerTime values based on slidingWindowSize and stepSize
    % This ensures all areas have aligned windows regardless of their optimized window sizes
    % Total time from spike data
    totalTime = timeRange(2) - timeRange(1);
    
    % Generate common centerTime values
    % Start from slidingWindowSize/2, end at totalTime - slidingWindowSize/2
    firstCenterTime = config.slidingWindowSize / 2;
    lastCenterTime = totalTime - config.slidingWindowSize / 2;
    commonCenterTimes = firstCenterTime:config.stepSize:lastCenterTime;
    
    if isempty(commonCenterTimes)
        error('No valid windows found. Check slidingWindowSize and stepSize relative to total time.');
    end
    
    numWindows = length(commonCenterTimes);
    fprintf('\nCommon window centers: %d windows from %.2f s to %.2f s (stepSize=%.3f s)\n', ...
        numWindows, firstCenterTime, lastCenterTime, config.stepSize);
    
    % Initialize results
    [popActivity, mrBr, d2, d2Normalized, startS, popActivityWindows, popActivityFull] = ...
        deal(cell(1, numAreas));
    
    if config.enablePermutations
        d2Permuted = cell(1, numAreas);
        mrBrPermuted = cell(1, numAreas);
        for a = 1:numAreas
            d2Permuted{a} = [];
            mrBrPermuted{a} = [];
        end
    end
    
    if config.analyzeModulation
        [popActivityModulated, mrBrModulated, d2Modulated, startSModulated, ...
            popActivityWindowsModulated, popActivityFullModulated] = deal(cell(1, numAreas));
        [popActivityUnmodulated, mrBrUnmodulated, d2Unmodulated, startSUnmodulated, ...
            popActivityWindowsUnmodulated, popActivityFullUnmodulated] = deal(cell(1, numAreas));
    else
        % Initialize empty even if not analyzing modulation
        [popActivityModulated, mrBrModulated, d2Modulated, startSModulated, ...
            popActivityWindowsModulated, popActivityFullModulated] = deal(cell(1, numAreas));
        [popActivityUnmodulated, mrBrUnmodulated, d2Unmodulated, startSUnmodulated, ...
            popActivityWindowsUnmodulated, popActivityFullUnmodulated] = deal(cell(1, numAreas));
        binSizeModulated = nan(1, numAreas);
        binSizeUnmodulated = nan(1, numAreas);
        slidingWindowSizeModulated = nan(1, numAreas);
        slidingWindowSizeUnmodulated = nan(1, numAreas);
    end
    
    % Filter areas to process: exclude areas with too few neurons or invalid bin sizes
    fprintf('\n=== Filtering Areas to Process ===\n');
    areasToProcess = [];
    areasToSkip = [];
    
    for a = areasToTest
        shouldSkip = false;
        skipReason = '';
        
        aID = dataStruct.idMatIdx{a};
        nNeurons = length(aID);
        
        % Check minimum number of neurons
        if nNeurons < config.nMinNeurons
            shouldSkip = true;
            skipReason = sprintf('Only %d neurons (minimum required: %d)', nNeurons, config.nMinNeurons);
        end
        
        % Check if bin size is valid
        if ~shouldSkip && isnan(binSize(a))
            shouldSkip = true;
            skipReason = 'Invalid bin size (area skipped earlier)';
        end
        
        if shouldSkip
            areasToSkip = [areasToSkip, a];
            fprintf('  Will skip area %s: %s\n', areas{a}, skipReason);
            % Initialize with empty arrays
            popActivity{a} = [];
            mrBr{a} = [];
            d2{a} = [];
            d2Normalized{a} = [];
            startS{a} = [];
            popActivityWindows{a} = [];
            popActivityFull{a} = [];
            if config.enablePermutations
                d2Permuted{a} = [];
                mrBrPermuted{a} = [];
            end
        else
            areasToProcess = [areasToProcess, a];
        end
    end
    
    if isempty(areasToProcess)
        error('No valid areas to process. All areas were skipped due to insufficient neurons or invalid bin sizes.');
    end
    
    fprintf('  Will process %d area(s): %s\n', length(areasToProcess), strjoin(areas(areasToProcess), ', '));
    
    % Main analysis loop
    fprintf('\n=== Processing Areas ===\n');
    for a = areasToProcess
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, sessionType);
        tic;
        
        aID = dataStruct.idMatIdx{a};
        
        % Calculate window size in samples for this area
        winSamples = round(slidingWindowSize(a) / binSize(a));
        
        % Get neuron IDs for this area
        neuronIDs = dataStruct.idLabel{a};
        
        % Bin data at area-specific bin size
        if config.pcaFlag
            % Use PCA-reconstructed data: rebin from 1ms to optimal bin size
            % reconstructedDataMat{a} is [timeBins_1ms x neurons]
            % We need to downsample to binSize(a)
            reconstructedMat_1ms = reconstructedDataMat{a};  % [timeBins_1ms x neurons]
            
            % Calculate number of bins at optimal bin size
            totalTime = timeRange(2) - timeRange(1);
            numBins_1ms = size(reconstructedMat_1ms, 1);
            numBins_optimal = round(totalTime / binSize(a));
            
            % Downsample reconstructed data to optimal bin size
            % Average across bins within each optimal bin
            binsPerOptimalBin = binSize(a) / tempBinSize;
            aDataMat = zeros(numBins_optimal, size(reconstructedMat_1ms, 2));
            for b = 1:numBins_optimal
                startIdx_1ms = round((b-1) * binsPerOptimalBin) + 1;
                endIdx_1ms = min(round(b * binsPerOptimalBin), numBins_1ms);
                if startIdx_1ms <= numBins_1ms
                    aDataMat(b, :) = mean(reconstructedMat_1ms(startIdx_1ms:endIdx_1ms, :), 1);
                end
            end
            numTimePoints = size(aDataMat, 1);
        else
            % Bin spikes on-demand at area-specific bin size
            aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIDs, timeRange, binSize(a));
            numTimePoints = size(aDataMat, 1);
        end
        
        popActivity{a} = mean(aDataMat, 2);
        [startS{a}, mrBr{a}, d2{a}, d2Normalized{a}, popActivityWindows{a}, popActivityFull{a}] = ...
            deal(nan(1, numWindows));
        
        % Process each window using common centerTime
        for w = 1:numWindows
            centerTime = commonCenterTimes(w);
            startS{a}(w) = centerTime;
            
            % Convert centerTime to indices for this area's binning
            % Use area-specific optimal window size
                [startIdx, endIdx] = calculate_window_indices_from_center(...
                    centerTime, slidingWindowSize(a), binSize(a), numTimePoints);
            
            % Check if window is valid (within bounds)
            if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                % Window is out of bounds for this area, skip
                continue;
            end
            
            wPopActivity = popActivity{a}(startIdx:endIdx);
            popActivityWindows{a}(w) = mean(wPopActivity);
            popActivityFull{a}(w) = popActivity{a}(startIdx + round(winSamples/2) - 1);
            
            if config.analyzeMrBr
                result = branching_ratio_mr_estimation(wPopActivity);
                mrBr{a}(w) = result.branching_ratio;
            else
                mrBr{a}(w) = nan;
            end
            
            if config.analyzeD2
                [varphi, ~] = myYuleWalker3(double(wPopActivity), config.pOrder);
                d2{a}(w) = getFixedPointDistance2(config.pOrder, config.critType, varphi);
            else
                d2{a}(w) = nan;
            end
        end
        
        % Perform circular permutations if enabled
        if config.enablePermutations
            if config.pcaFlag
                % Use PCA-reconstructed data for permutations
                [d2Permuted{a}, mrBrPermuted{a}] = perform_circular_permutations_pca(...
                    reconstructedDataMat{a}, a, commonCenterTimes, slidingWindowSize(a), binSize(a), numTimePoints, config, timeRange, tempBinSize);
            else
                % Use original binned data for permutations
                [d2Permuted{a}, mrBrPermuted{a}] = perform_circular_permutations(...
                    aDataMat, commonCenterTimes, slidingWindowSize(a), binSize(a), numTimePoints, config);
            end
            
            % Normalize d2 by shuffled d2 values if requested
            if config.normalizeD2 && config.analyzeD2 && ~isempty(d2Permuted{a})
                % Calculate mean shuffled d2 for each window
                d2PermutedMean = nanmean(d2Permuted{a}, 2);
                % Normalize: d2Normalized = d2 / mean(shuffled_d2)
                for w = 1:numWindows
                    if ~isnan(d2{a}(w)) && ~isnan(d2PermutedMean(w)) && d2PermutedMean(w) > 0
                        d2Normalized{a}(w) = d2{a}(w) / d2PermutedMean(w);
                    else
                        d2Normalized{a}(w) = nan;
                    end
                end
            else
                % If normalization disabled or no permutations, set to NaN
                d2Normalized{a}(:) = nan;
            end
        else
            % Initialize empty if permutations disabled
            if isempty(d2Permuted{a})
                d2Permuted{a} = [];
            end
            if isempty(mrBrPermuted{a})
                mrBrPermuted{a} = [];
            end
            % If no permutations, normalization not possible
            d2Normalized{a}(:) = nan;
        end
        
        fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
    end
    
    % Build results structure
    results = build_results_structure(dataStruct, config, areas, areasToTest, ...
        popActivity, mrBr, d2, d2Normalized, startS, popActivityWindows, popActivityFull, ...
        binSize, slidingWindowSize, ...
        d2Permuted, mrBrPermuted, ...
        modulationResults, ...
        popActivityModulated, mrBrModulated, d2Modulated, startSModulated, ...
        popActivityWindowsModulated, popActivityFullModulated, ...
        popActivityUnmodulated, mrBrUnmodulated, d2Unmodulated, startSUnmodulated, ...
        popActivityWindowsUnmodulated, popActivityFullUnmodulated, ...
        binSizeModulated, binSizeUnmodulated, ...
        slidingWindowSizeModulated, slidingWindowSizeUnmodulated);
    
    % Save results if requested
    if config.saveData
        save(resultsPath, 'results');
        fprintf('Saved %s d2/mrBr to %s\n', sessionType, resultsPath);
    else
        fprintf('Skipping save (config.saveData = false)\n');
    end
    
    % Plotting
    if config.makePlots
        plotArgs = {};
        if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
            plotArgs = [plotArgs, {'sessionName', dataStruct.sessionName}];
        end
        if isfield(dataStruct, 'dataBaseName') && ~isempty(dataStruct.dataBaseName)
            plotArgs = [plotArgs, {'dataBaseName', dataStruct.dataBaseName}];
        end
        plotConfig = setup_plotting(config.saveDir, plotArgs{:});
        criticality_ar_plot(results, plotConfig, config, dataStruct, filenameSuffix);
    end
end

function config = set_config_defaults(config)
% SET_CONFIG_DEFAULTS Set default values for configuration structure
    
    defaults = struct();
    defaults.analyzeD2 = true;
    defaults.analyzeMrBr = false;
    defaults.pcaFlag = 0;
    defaults.pcaFirstFlag = 1;
    defaults.nDim = 4;
    defaults.useOptimalBinWindowFunction = true;
    defaults.enablePermutations = true;
    defaults.nShuffles = 3;
    defaults.analyzeModulation = false;
    defaults.makePlots = true;
    defaults.saveData = true;  % Set to false to skip saving results
    defaults.normalizeD2 = true;  % Normalize d2 by shuffled d2 values
    defaults.minSpikesPerBin = 3;
    defaults.maxSpikesPerBin = 50;
    defaults.minBinsPerWindow = 1000;
    defaults.pOrder = 10;
    defaults.critType = 2;
    defaults.modulationThreshold = 2;
    defaults.modulationBinSize = nan;
    defaults.modulationBaseWindow = [-3, -2];
    defaults.modulationEventWindow = [-0.2, 0.6];
    defaults.modulationPlotFlag = false;
    defaults.includeM2356 = false;  % Include combined M23+M56 area (optional)
    defaults.nMinNeurons = 10;  % Minimum number of neurons required
    
    % Apply defaults
    fields = fieldnames(defaults);
    for i = 1:length(fields)
        if ~isfield(config, fields{i})
            config.(fields{i}) = defaults.(fields{i});
        end
    end
end

function modulationResults = perform_modulation_analysis(dataStruct, config)
% PERFORM_MODULATION_ANALYSIS Perform modulation analysis
    
    % This is a placeholder - the actual function should be called here
    % For now, return empty cell array
    modulationResults = cell(1, length(dataStruct.areas));
    
    % Uncomment when perform_modulation_analysis function is available:
    % modulationResults = perform_modulation_analysis(dataStruct.spikeData, ...
    %     dataStruct.areas, dataStruct.idLabel, dataStruct.areasToTest, ...
    %     dataStruct.sessionType, dataStruct.dataR, dataStruct.opts, ...
    %     config.modulationBinSize, config.modulationBaseWindow, ...
    %     config.modulationEventWindow, config.modulationThreshold, ...
    %     config.modulationPlotFlag);
end

function [binSize, slidingWindowSize, ...
    binSizeModulated, binSizeUnmodulated, ...
    slidingWindowSizeModulated, slidingWindowSizeUnmodulated] = ...
    find_optimal_parameters_from_spikes(dataStruct, config, modulationResults, areasToTest, timeRange)
% FIND_OPTIMAL_PARAMETERS_FROM_SPIKES Find bin and window sizes from spike times
% Returns binSize and slidingWindowSize as vectors (length numAreas)
% Uses spike times directly (new approach)
    
    numAreas = length(dataStruct.areas);
    binSize = zeros(1, numAreas);
    slidingWindowSize = zeros(1, numAreas);
    
    if config.useOptimalBinWindowFunction
        for a = areasToTest
            % Get neuron IDs for this area
            neuronIDs = dataStruct.idLabel{a};
            
            % Calculate firing rate from spike times
            thisFiringRate = calculate_firing_rate_from_spikes(...
                dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIDs, timeRange);
            
            [binSize(a), slidingWindowSize(a)] = ...
                find_optimal_bin_and_window(thisFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
        end
    else
        % Use manually defined values
        if isfield(config, 'binSize')
            % Convert scalar to vector if needed
            if isscalar(config.binSize)
                binSize = repmat(config.binSize, 1, numAreas);
            else
                binSize = config.binSize;
            end
        else
            error('binSize must be provided if useOptimalBinWindowFunction is false');
        end
        % Convert scalar to vector if needed
        if isscalar(config.slidingWindowSize)
            slidingWindowSize = repmat(config.slidingWindowSize, 1, numAreas);
        else
            slidingWindowSize = config.slidingWindowSize;
        end
    end
    
    % Initialize modulation parameters
    binSizeModulated = nan(1, numAreas);
    binSizeUnmodulated = nan(1, numAreas);
    slidingWindowSizeModulated = nan(1, numAreas);
    slidingWindowSizeUnmodulated = nan(1, numAreas);
    
    if config.analyzeModulation && config.useOptimalBinWindowFunction
        for a = areasToTest
            if ~isempty(modulationResults{a})
                % Get modulated and unmodulated neuron IDs
                modulatedNeurons = modulationResults{a}.neuronIds(modulationResults{a}.isModulated);
                unmodulatedNeurons = modulationResults{a}.neuronIds(~modulationResults{a}.isModulated);
                
                if length(modulatedNeurons) >= 5
                    mFiringRate = calculate_firing_rate_from_spikes(...
                        dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                        modulatedNeurons, timeRange);
                    [binSizeModulated(a), slidingWindowSizeModulated(a)] = ...
                        find_optimal_bin_and_window(mFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
                end
                
                if length(unmodulatedNeurons) >= 5
                    uFiringRate = calculate_firing_rate_from_spikes(...
                        dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                        unmodulatedNeurons, timeRange);
                    [binSizeUnmodulated(a), slidingWindowSizeUnmodulated(a)] = ...
                        find_optimal_bin_and_window(uFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
                end
            end
        end
    elseif config.analyzeModulation
        binSizeModulated = binSize;
        binSizeUnmodulated = binSize;
        slidingWindowSizeModulated = slidingWindowSize;
        slidingWindowSizeUnmodulated = slidingWindowSize;
    end
    
    for a = areasToTest
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s\n', ...
            dataStruct.areas{a}, binSize(a), slidingWindowSize(a));
    end
end

function [binSize, slidingWindowSize, ...
    binSizeModulated, binSizeUnmodulated, ...
    slidingWindowSizeModulated, slidingWindowSizeUnmodulated] = ...
    find_optimal_parameters_from_spikes_pca(reconstructedDataMat, config, modulationResults, areasToTest, timeRange, tempBinSize, areas, dataStruct)
% FIND_OPTIMAL_PARAMETERS_FROM_SPIKES_PCA Find bin and window sizes from PCA-reconstructed data
% Returns binSize and slidingWindowSize as vectors (length numAreas)
% Uses PCA-reconstructed data matrices (for when pcaFlag = 1)
%
% Variables:
%   reconstructedDataMat - Cell array of PCA-reconstructed data matrices [timeBins x neurons] at tempBinSize
%   config - Configuration structure
%   modulationResults - Modulation analysis results (cell array)
%   areasToTest - Areas to process
%   timeRange - [startTime, endTime] in seconds
%   tempBinSize - Bin size used for PCA reconstruction (typically 0.001s = 1ms)
%   areas - Cell array of area names
%   dataStruct - Data structure (for modulation analysis if needed)
%
% Returns:
%   binSize - Optimal bin sizes per area
%   slidingWindowSize - Optimal window sizes per area
%   binSizeModulated - Optimal bin sizes for modulated neurons (if modulation analysis enabled)
%   binSizeUnmodulated - Optimal bin sizes for unmodulated neurons (if modulation analysis enabled)
%   slidingWindowSizeModulated - Optimal window sizes for modulated neurons (if modulation analysis enabled)
%   slidingWindowSizeUnmodulated - Optimal window sizes for unmodulated neurons (if modulation analysis enabled)
    
    numAreas = length(reconstructedDataMat);
    binSize = zeros(1, numAreas);
    slidingWindowSize = zeros(1, numAreas);
    
    totalTime = timeRange(2) - timeRange(1);
    
    if config.useOptimalBinWindowFunction
        for a = areasToTest
            % Get PCA-reconstructed data for this area
            % reconstructedDataMat{a} is [timeBins_1ms x neurons]
            reconstructedMat = reconstructedDataMat{a};
            
            % Calculate firing rate from reconstructed data
            % Sum across neurons and time, then divide by total time
            totalSpikes = sum(reconstructedMat(:));  % Sum across all neurons and time bins
            thisFiringRate = totalSpikes / totalTime;  % Firing rate in spikes per second
            
            [binSize(a), slidingWindowSize(a)] = ...
                find_optimal_bin_and_window(thisFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
        end
    else
        % Use manually defined values
        if isfield(config, 'binSize')
            % Convert scalar to vector if needed
            if isscalar(config.binSize)
                binSize = repmat(config.binSize, 1, numAreas);
            else
                binSize = config.binSize;
            end
        else
            error('binSize must be provided if useOptimalBinWindowFunction is false');
        end
        % Convert scalar to vector if needed
        if isscalar(config.slidingWindowSize)
            slidingWindowSize = repmat(config.slidingWindowSize, 1, numAreas);
        else
            slidingWindowSize = config.slidingWindowSize;
        end
    end
    
    % Initialize modulation parameters
    binSizeModulated = nan(1, numAreas);
    binSizeUnmodulated = nan(1, numAreas);
    slidingWindowSizeModulated = nan(1, numAreas);
    slidingWindowSizeUnmodulated = nan(1, numAreas);
    
    if config.analyzeModulation && config.useOptimalBinWindowFunction
        for a = areasToTest
            if ~isempty(modulationResults{a})
                % Get modulated and unmodulated neuron IDs
                modulatedNeurons = modulationResults{a}.neuronIds(modulationResults{a}.isModulated);
                unmodulatedNeurons = modulationResults{a}.neuronIds(~modulationResults{a}.isModulated);
                
                % For PCA, we need to extract the relevant neurons from the reconstructed data
                % reconstructedDataMat{a} is [timeBins x all_neurons]
                % We need to identify which columns correspond to modulated/unmodulated neurons
                % This requires matching neuronIDs from modulationResults to the original neuron order
                % For now, we'll use the full reconstructed data and calculate firing rates
                % (This is a simplification - ideally we'd extract specific neurons)
                
                if length(modulatedNeurons) >= 5
                    % Use full reconstructed data as approximation
                    reconstructedMat = reconstructedDataMat{a};
                    totalSpikes = sum(reconstructedMat(:));
                    mFiringRate = totalSpikes / totalTime;
                    [binSizeModulated(a), slidingWindowSizeModulated(a)] = ...
                        find_optimal_bin_and_window(mFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
                end
                
                if length(unmodulatedNeurons) >= 5
                    % Use full reconstructed data as approximation
                    reconstructedMat = reconstructedDataMat{a};
                    totalSpikes = sum(reconstructedMat(:));
                    uFiringRate = totalSpikes / totalTime;
                    [binSizeUnmodulated(a), slidingWindowSizeUnmodulated(a)] = ...
                        find_optimal_bin_and_window(uFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
                end
            end
        end
    elseif config.analyzeModulation
        binSizeModulated = binSize;
        binSizeUnmodulated = binSize;
        slidingWindowSizeModulated = slidingWindowSize;
        slidingWindowSizeUnmodulated = slidingWindowSize;
    end
    
    for a = areasToTest
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s (from PCA-reconstructed data)\n', ...
            areas{a}, binSize(a), slidingWindowSize(a));
    end
end

function [d2Permuted, mrBrPermuted] = perform_circular_permutations(aDataMat, commonCenterTimes, slidingWindowSize, binSize, numTimePoints, config)
% PERFORM_CIRCULAR_PERMUTATIONS Perform circular permutation testing
%   Shuffles each neuron's activity independently using circular shifts,
%   then computes population activity from the shuffled data.
%
% Variables:
%   aDataMat - Binned spike matrix [time bins x neurons]
%   commonCenterTimes - Vector of window center times
%   slidingWindowSize - Window size in seconds
%   binSize - Bin size in seconds
%   numTimePoints - Number of time points in aDataMat
%   config - Configuration structure
%
% Goal:
%   For each shuffle, circularly shift each neuron's activity independently,
%   then compute population activity and analyze d2/mrBr in sliding windows.
    
    numWindows = length(commonCenterTimes);
    numNeurons = size(aDataMat, 2);
    
    d2Permuted = nan(numWindows, config.nShuffles);
    mrBrPermuted = nan(numWindows, config.nShuffles);
    
    for s = 1:config.nShuffles
        % Circularly shift each neuron's activity independently
        % aDataMat is [time bins x neurons]
        permutedDataMat = zeros(size(aDataMat));
        for n = 1:numNeurons
            shiftAmount = randi(size(aDataMat, 1));
            permutedDataMat(:, n) = circshift(aDataMat(:, n), shiftAmount);
        end
        
        % Compute population activity from shuffled data
        permutedPopActivity = mean(permutedDataMat, 2);
        
        for w = 1:numWindows
            centerTime = commonCenterTimes(w);
            
            % Convert centerTime to indices for this area's binning
            % Use area-specific window size
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                centerTime, slidingWindowSize, binSize, numTimePoints);
            
            % Check if window is valid (within bounds)
            if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                % Window is out of bounds, skip
                continue;
            end
            
            wPopActivity = permutedPopActivity(startIdx:endIdx);
            
            if config.analyzeMrBr
                result = branching_ratio_mr_estimation(wPopActivity);
                mrBrPermuted(w, s) = result.branching_ratio;
            end
            
            if config.analyzeD2
                [varphi, ~] = myYuleWalker3(double(wPopActivity), config.pOrder);
                d2Permuted(w, s) = getFixedPointDistance2(config.pOrder, config.critType, varphi);
            end
        end
    end
end

function [d2Permuted, mrBrPermuted] = perform_circular_permutations_pca(reconstructedMat_1ms, a, commonCenterTimes, slidingWindowSize, binSize, numTimePoints, config, timeRange, tempBinSize)
% PERFORM_CIRCULAR_PERMUTATIONS_PCA Perform circular permutation testing for AR analysis
%   Uses PCA-reconstructed data (for when pcaFlag = 1)
%   Shuffles each neuron's activity independently using circular shifts,
%   then computes population activity from the shuffled data.
%
% Variables:
%   reconstructedMat_1ms - PCA-reconstructed data matrix [timeBins_1ms x neurons] at tempBinSize
%   a - Area index
%   commonCenterTimes - Vector of window center times
%   slidingWindowSize - Window size in seconds
%   binSize - Bin size in seconds
%   numTimePoints - Number of time points at optimal bin size
%   config - Configuration structure
%   timeRange - [startTime, endTime] in seconds
%   tempBinSize - Bin size used for PCA (typically 0.001s = 1ms)
%
% Goal:
%   For each shuffle, circularly shift each neuron's activity independently,
%   then compute population activity and analyze d2/mrBr in sliding windows.
    
    numWindows = length(commonCenterTimes);
    
    d2Permuted = nan(numWindows, config.nShuffles);
    mrBrPermuted = nan(numWindows, config.nShuffles);
    
    % Downsample reconstructed data from 1ms to optimal bin size for full time range
    totalTime = timeRange(2) - timeRange(1);
    numBins_1ms = size(reconstructedMat_1ms, 1);
    numBins_optimal = round(totalTime / binSize);
    binsPerOptimalBin = binSize / tempBinSize;
    
    % Create downsampled data matrix at optimal bin size
    originalDataMat = zeros(numBins_optimal, size(reconstructedMat_1ms, 2));
    for b = 1:numBins_optimal
        startIdx_1ms = round((b-1) * binsPerOptimalBin) + 1;
        endIdx_1ms = min(round(b * binsPerOptimalBin), numBins_1ms);
        if startIdx_1ms <= numBins_1ms
            originalDataMat(b, :) = mean(reconstructedMat_1ms(startIdx_1ms:endIdx_1ms, :), 1);
        end
    end
    numNeurons = size(originalDataMat, 2);
    
    for s = 1:config.nShuffles
        % Circularly shift each neuron's activity independently
        % originalDataMat is [time bins x neurons]
        permutedDataMat = zeros(size(originalDataMat));
        for n = 1:numNeurons
            shiftAmount = randi(size(originalDataMat, 1));
            permutedDataMat(:, n) = circshift(originalDataMat(:, n), shiftAmount);
        end
        
        % No need to re-apply PCA - data is already PCA-reconstructed
        % Just use the permuted data directly
        
        % Compute population activity from shuffled data
        permutedPopActivity = mean(permutedDataMat, 2);
        
        for w = 1:numWindows
            centerTime = commonCenterTimes(w);
            
            % Convert centerTime to indices for this area's binning
            % Use area-specific window size
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                centerTime, slidingWindowSize, binSize, numTimePoints);
            
            % Check if window is valid (within bounds)
            if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
                % Window is out of bounds, skip
                continue;
            end
            
            wPopActivity = permutedPopActivity(startIdx:endIdx);
            
            if config.analyzeMrBr
                result = branching_ratio_mr_estimation(wPopActivity);
                mrBrPermuted(w, s) = result.branching_ratio;
            end
            
            if config.analyzeD2
                [varphi, ~] = myYuleWalker3(double(wPopActivity), config.pOrder);
                d2Permuted(w, s) = getFixedPointDistance2(config.pOrder, config.critType, varphi);
            end
        end
    end
end

function results = build_results_structure(dataStruct, config, areas, areasToTest, ...
    popActivity, mrBr, d2, d2Normalized, startS, popActivityWindows, popActivityFull, ...
    binSize, slidingWindowSize, ...
    d2Permuted, mrBrPermuted, ...
    modulationResults, ...
    popActivityModulated, mrBrModulated, d2Modulated, startSModulated, ...
    popActivityWindowsModulated, popActivityFullModulated, ...
    popActivityUnmodulated, mrBrUnmodulated, d2Unmodulated, startSUnmodulated, ...
    popActivityWindowsUnmodulated, popActivityFullUnmodulated, ...
    binSizeModulated, binSizeUnmodulated, ...
    slidingWindowSizeModulated, slidingWindowSizeUnmodulated)
% BUILD_RESULTS_STRUCTURE Build results structure
    
    results = struct();
    results.sessionType = dataStruct.sessionType;
    results.areas = areas;
    results.mrBr = mrBr;
    results.d2 = d2;  % Raw d2 values
    results.d2Normalized = d2Normalized;  % Normalized d2 values (d2 / mean(shuffled_d2))
    results.startS = startS;
    results.popActivity = popActivity;
    results.popActivityWindows = popActivityWindows;
    results.popActivityFull = popActivityFull;
    results.binSize = binSize;
    results.slidingWindowSize = slidingWindowSize;
    results.d2WindowSize = slidingWindowSize;
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.stepSize = config.stepSize;
    results.params.analyzeD2 = config.analyzeD2;
    results.params.analyzeMrBr = config.analyzeMrBr;
    results.params.pcaFlag = config.pcaFlag;
    results.params.pcaFirstFlag = config.pcaFirstFlag;
    results.params.nDim = config.nDim;
    results.params.pOrder = config.pOrder;
    results.params.critType = config.critType;
    results.params.normalizeD2 = config.normalizeD2;
    
    if config.enablePermutations
        results.enablePermutations = true;
        results.nShuffles = config.nShuffles;
        results.d2Permuted = d2Permuted;
        results.mrBrPermuted = mrBrPermuted;
        
        % Calculate mean and SEM
        d2PermutedMean = cell(1, length(areas));
        d2PermutedSEM = cell(1, length(areas));
        mrBrPermutedMean = cell(1, length(areas));
        mrBrPermutedSEM = cell(1, length(areas));
        
        for a = 1:length(areas)
            if ~isempty(d2Permuted{a})
                d2PermutedMean{a} = nanmean(d2Permuted{a}, 2);
                d2PermutedSEM{a} = nanstd(d2Permuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                d2PermutedMean{a} = [];
                d2PermutedSEM{a} = [];
            end
            if ~isempty(mrBrPermuted{a})
                mrBrPermutedMean{a} = nanmean(mrBrPermuted{a}, 2);
                mrBrPermutedSEM{a} = nanstd(mrBrPermuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                mrBrPermutedMean{a} = [];
                mrBrPermutedSEM{a} = [];
            end
        end
        
        results.d2PermutedMean = d2PermutedMean;
        results.d2PermutedSEM = d2PermutedSEM;
        results.mrBrPermutedMean = mrBrPermutedMean;
        results.mrBrPermutedSEM = mrBrPermutedSEM;
    else
        results.enablePermutations = false;
        results.nShuffles = 0;
    end
    
    if config.analyzeModulation
        results.analyzeModulation = true;
        results.modulationResults = modulationResults;
        results.modulationThreshold = config.modulationThreshold;
        results.modulationBinSize = config.modulationBinSize;
        results.modulationBaseWindow = config.modulationBaseWindow;
        results.modulationEventWindow = config.modulationEventWindow;
        results.modulationPlotFlag = config.modulationPlotFlag;
        results.popActivityModulated = popActivityModulated;
        results.mrBrModulated = mrBrModulated;
        results.d2Modulated = d2Modulated;
        results.startSModulated = startSModulated;
        results.popActivityFullModulated = popActivityFullModulated;
        results.popActivityUnmodulated = popActivityUnmodulated;
        results.mrBrUnmodulated = mrBrUnmodulated;
        results.d2Unmodulated = d2Unmodulated;
        results.startSUnmodulated = startSUnmodulated;
        results.popActivityFullUnmodulated = popActivityFullUnmodulated;
        results.binSizeModulated = binSizeModulated;
        results.binSizeUnmodulated = binSizeUnmodulated;
        results.slidingWindowSizeModulated = slidingWindowSizeModulated;
        results.slidingWindowSizeUnmodulated = slidingWindowSizeUnmodulated;
    else
        results.analyzeModulation = false;
    end
end


