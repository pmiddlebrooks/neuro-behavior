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
%
% Goal:
%   Compute d2 and/or mrBr criticality measures in sliding windows for spike data.
%   Supports PCA, modulation analysis, and permutation testing.
%
% Returns:
%   results - Structure with d2, mrBr, startS, popActivity, and params

    % Add paths
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils'));
    
    % Validate inputs
    validate_workspace_vars({'dataType', 'dataMat', 'areas', 'idMatIdx'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
    
    % Set defaults
    config = set_config_defaults(config);
    
    dataType = dataStruct.dataType;
    areas = dataStruct.areas;
    numAreas = length(areas);
    
    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:numAreas;
    end
    
    fprintf('\n=== Criticality AR Analysis Setup ===\n');
    fprintf('Data type: %s\n', dataType);
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('Window size: %.2f s\n', config.slidingWindowSize);
    fprintf('Analyze d2: %d, Analyze mrBr: %d\n', config.analyzeD2, config.analyzeMrBr);
    
    % Create filename suffix based on PCA flag
    if config.pcaFlag
        filenameSuffix = '_pca';
    else
        filenameSuffix = '';
    end
    
    % Setup results path
    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end
    
    sessionNameForPath = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionNameForPath = dataStruct.sessionName;
    end
    
    resultsPath = create_results_path('criticality_ar', dataType, config.slidingWindowSize, ...
        sessionNameForPath, config.saveDir, 'filenameSuffix', filenameSuffix);
    
    % Load spike data for modulation analysis if needed
    if config.analyzeModulation
        if ~isfield(dataStruct, 'spikeData') || isempty(dataStruct.spikeData)
            fprintf('\n=== Loading spike data for modulation analysis ===\n');
            if strcmp(dataType, 'reach')
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
    
    % Apply PCA to original data if requested
    fprintf('\n--- Step 1-2: PCA on original data if requested ---\n');
    reconstructedDataMat = cell(1, numAreas);
    for a = areasToTest
        aID = dataStruct.idMatIdx{a};
        thisDataMat = dataStruct.dataMat(:, aID);
        if config.pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(thisDataMat);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, min(6, forDim));
            nDim = 1:forDim;
            reconstructedDataMat{a} = score(:,nDim) * coeff(:,nDim)' + mu;
        else
            reconstructedDataMat{a} = thisDataMat;
        end
    end
    
    % Find optimal parameters
    fprintf('\n--- Step 3: Finding optimal parameters ---\n');
    [optimalBinSize, optimalWindowSize, d2StepSize, ...
        optimalBinSizeModulated, optimalBinSizeUnmodulated, ...
        optimalWindowSizeModulated, optimalWindowSizeUnmodulated] = ...
        find_optimal_parameters(reconstructedDataMat, dataStruct, config, modulationResults, areasToTest);
    
    % Initialize modulation parameters if not already set
    if ~config.analyzeModulation
        if ~exist('optimalBinSizeModulated', 'var') || isempty(optimalBinSizeModulated)
            optimalBinSizeModulated = nan(1, numAreas);
        end
        if ~exist('optimalBinSizeUnmodulated', 'var') || isempty(optimalBinSizeUnmodulated)
            optimalBinSizeUnmodulated = nan(1, numAreas);
        end
        if ~exist('optimalWindowSizeModulated', 'var') || isempty(optimalWindowSizeModulated)
            optimalWindowSizeModulated = nan(1, numAreas);
        end
        if ~exist('optimalWindowSizeUnmodulated', 'var') || isempty(optimalWindowSizeUnmodulated)
            optimalWindowSizeUnmodulated = nan(1, numAreas);
        end
    end
    
    % Initialize results
    [popActivity, mrBr, d2, startS, popActivityWindows, popActivityFull] = ...
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
        optimalBinSizeModulated = nan(1, numAreas);
        optimalBinSizeUnmodulated = nan(1, numAreas);
        optimalWindowSizeModulated = nan(1, numAreas);
        optimalWindowSizeUnmodulated = nan(1, numAreas);
    end
    
    % Main analysis loop
    fprintf('\n=== Processing Areas ===\n');
    for a = areasToTest
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataType);
        tic;
        
        aID = dataStruct.idMatIdx{a};
        stepSamples = round(d2StepSize(a) / optimalBinSize(a));
        winSamples = round(optimalWindowSize(a) / optimalBinSize(a));
        
        if winSamples < config.minSegmentLength
            fprintf('Skipping: Not enough data in %s (%s)...\n', areas{a}, dataType);
            continue
        end
        
        aDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), optimalBinSize(a));
        numTimePoints = size(aDataMat, 1);
        numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
        
        if config.pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, min(6, forDim));
            nDim = 1:forDim;
            aDataMat = score(:,nDim) * coeff(:,nDim)' + mu;
        end
        
        popActivity{a} = mean(aDataMat, 2);
        [startS{a}, mrBr{a}, d2{a}, popActivityWindows{a}, popActivityFull{a}] = ...
            deal(nan(1, numWindows));
        
        % Process each window
        for w = 1:numWindows
            [startIdx, endIdx, centerTime, ~] = calculate_window_indices(...
                w, numTimePoints, winSamples, stepSamples, optimalBinSize(a));
            startS{a}(w) = centerTime;
            
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
                [varphi, ~] = myYuleWalker3(wPopActivity, config.pOrder);
                d2{a}(w) = getFixedPointDistance2(config.pOrder, config.critType, varphi);
            else
                d2{a}(w) = nan;
            end
        end
        
        % Perform circular permutations if enabled
        if config.enablePermutations
            [d2Permuted{a}, mrBrPermuted{a}] = perform_circular_permutations(...
                popActivity{a}, winSamples, stepSamples, optimalBinSize(a), config);
        else
            % Initialize empty if permutations disabled
            if isempty(d2Permuted{a})
                d2Permuted{a} = [];
            end
            if isempty(mrBrPermuted{a})
                mrBrPermuted{a} = [];
            end
        end
        
        fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
    end
    
    % Build results structure
    results = build_results_structure(dataStruct, config, areas, areasToTest, ...
        popActivity, mrBr, d2, startS, popActivityWindows, popActivityFull, ...
        optimalBinSize, optimalWindowSize, d2StepSize, ...
        d2Permuted, mrBrPermuted, ...
        modulationResults, ...
        popActivityModulated, mrBrModulated, d2Modulated, startSModulated, ...
        popActivityWindowsModulated, popActivityFullModulated, ...
        popActivityUnmodulated, mrBrUnmodulated, d2Unmodulated, startSUnmodulated, ...
        popActivityWindowsUnmodulated, popActivityFullUnmodulated, ...
        optimalBinSizeModulated, optimalBinSizeUnmodulated, ...
        optimalWindowSizeModulated, optimalWindowSizeUnmodulated);
    
    % Save results
    save(resultsPath, 'results');
    fprintf('Saved %s d2/mrBr to %s\n', dataType, resultsPath);
    
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
    defaults.minSegmentLength = 50;
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
    %     dataStruct.dataType, dataStruct.dataR, dataStruct.opts, ...
    %     config.modulationBinSize, config.modulationBaseWindow, ...
    %     config.modulationEventWindow, config.modulationThreshold, ...
    %     config.modulationPlotFlag);
end

function [optimalBinSize, optimalWindowSize, d2StepSize, ...
    optimalBinSizeModulated, optimalBinSizeUnmodulated, ...
    optimalWindowSizeModulated, optimalWindowSizeUnmodulated] = ...
    find_optimal_parameters(reconstructedDataMat, dataStruct, config, modulationResults, areasToTest)
% FIND_OPTIMAL_PARAMETERS Find optimal bin and window sizes
    
    numAreas = length(dataStruct.areas);
    optimalBinSize = zeros(1, numAreas);
    optimalWindowSize = zeros(1, numAreas);
    
    if config.useOptimalBinWindowFunction
        for a = areasToTest
            thisDataMat = reconstructedDataMat{a};
            thisFiringRate = sum(thisDataMat(:)) / (size(thisDataMat, 1)/1000);
            [optimalBinSize(a), optimalWindowSize(a)] = ...
                find_optimal_bin_and_window(thisFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
        end
    else
        % Use manually defined values
        if isfield(config, 'binSize')
            optimalBinSize = repmat(config.binSize, 1, numAreas);
        else
            error('binSize must be provided if useOptimalBinWindowFunction is false');
        end
        optimalWindowSize = repmat(config.slidingWindowSize, 1, numAreas);
    end
    
    d2StepSize = optimalBinSize * 2;
    
    % Initialize modulation parameters
    optimalBinSizeModulated = nan(1, numAreas);
    optimalBinSizeUnmodulated = nan(1, numAreas);
    optimalWindowSizeModulated = nan(1, numAreas);
    optimalWindowSizeUnmodulated = nan(1, numAreas);
    
    if config.analyzeModulation && config.useOptimalBinWindowFunction
        for a = areasToTest
            if ~isempty(modulationResults{a})
                % Get modulated and unmodulated neuron indices
                modulatedNeurons = modulationResults{a}.neuronIds(modulationResults{a}.isModulated);
                unmodulatedNeurons = modulationResults{a}.neuronIds(~modulationResults{a}.isModulated);
                
                modulatedIndices = ismember(dataStruct.idLabel{a}, modulatedNeurons);
                unmodulatedIndices = ismember(dataStruct.idLabel{a}, unmodulatedNeurons);
                
                if sum(modulatedIndices) >= 5
                    modulatedDataMat = reconstructedDataMat{a}(:, modulatedIndices);
                    mFiringRate = sum(modulatedDataMat(:)) / (size(modulatedDataMat, 1)/1000);
                    [optimalBinSizeModulated(a), optimalWindowSizeModulated(a)] = ...
                        find_optimal_bin_and_window(mFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
                end
                
                if sum(unmodulatedIndices) >= 5
                    unmodulatedDataMat = reconstructedDataMat{a}(:, unmodulatedIndices);
                    uFiringRate = sum(unmodulatedDataMat(:)) / (size(unmodulatedDataMat, 1)/1000);
                    [optimalBinSizeUnmodulated(a), optimalWindowSizeUnmodulated(a)] = ...
                        find_optimal_bin_and_window(uFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
                end
            end
        end
    elseif config.analyzeModulation
        optimalBinSizeModulated = optimalBinSize;
        optimalBinSizeUnmodulated = optimalBinSize;
        optimalWindowSizeModulated = optimalWindowSize;
        optimalWindowSizeUnmodulated = optimalWindowSize;
    end
    
    for a = areasToTest
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s, step size = %.3f\n', ...
            dataStruct.areas{a}, optimalBinSize(a), optimalWindowSize(a), d2StepSize(a));
    end
end

function [d2Permuted, mrBrPermuted] = perform_circular_permutations(popActivity, winSamples, stepSamples, binSize, config)
% PERFORM_CIRCULAR_PERMUTATIONS Perform circular permutation testing
    
    numTimePoints = length(popActivity);
    numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
    
    d2Permuted = nan(numWindows, config.nShuffles);
    mrBrPermuted = nan(numWindows, config.nShuffles);
    
    for s = 1:config.nShuffles
        % Circular shift
        shiftAmount = randi(numTimePoints);
        permutedActivity = circshift(popActivity, shiftAmount);
        
        for w = 1:numWindows
            startIdx = (w - 1) * stepSamples + 1;
            endIdx = startIdx + winSamples - 1;
            wPopActivity = permutedActivity(startIdx:endIdx);
            
            if config.analyzeMrBr
                result = branching_ratio_mr_estimation(wPopActivity);
                mrBrPermuted(w, s) = result.branching_ratio;
            end
            
            if config.analyzeD2
                [varphi, ~] = myYuleWalker3(wPopActivity, config.pOrder);
                d2Permuted(w, s) = getFixedPointDistance2(config.pOrder, config.critType, varphi);
            end
        end
    end
end

function results = build_results_structure(dataStruct, config, areas, areasToTest, ...
    popActivity, mrBr, d2, startS, popActivityWindows, popActivityFull, ...
    optimalBinSize, optimalWindowSize, d2StepSize, ...
    d2Permuted, mrBrPermuted, ...
    modulationResults, ...
    popActivityModulated, mrBrModulated, d2Modulated, startSModulated, ...
    popActivityWindowsModulated, popActivityFullModulated, ...
    popActivityUnmodulated, mrBrUnmodulated, d2Unmodulated, startSUnmodulated, ...
    popActivityWindowsUnmodulated, popActivityFullUnmodulated, ...
    optimalBinSizeModulated, optimalBinSizeUnmodulated, ...
    optimalWindowSizeModulated, optimalWindowSizeUnmodulated)
% BUILD_RESULTS_STRUCTURE Build results structure
    
    results = struct();
    results.dataType = dataStruct.dataType;
    results.areas = areas;
    results.mrBr = mrBr;
    results.d2 = d2;
    results.startS = startS;
    results.popActivity = popActivity;
    results.popActivityWindows = popActivityWindows;
    results.popActivityFull = popActivityFull;
    results.optimalBinSize = optimalBinSize;
    results.optimalWindowSize = optimalWindowSize;
    results.d2StepSize = d2StepSize;
    results.d2WindowSize = optimalWindowSize;
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.analyzeD2 = config.analyzeD2;
    results.params.analyzeMrBr = config.analyzeMrBr;
    results.params.pcaFlag = config.pcaFlag;
    results.params.pcaFirstFlag = config.pcaFirstFlag;
    results.params.nDim = config.nDim;
    results.params.pOrder = config.pOrder;
    results.params.critType = config.critType;
    
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
        results.optimalBinSizeModulated = optimalBinSizeModulated;
        results.optimalBinSizeUnmodulated = optimalBinSizeUnmodulated;
        results.optimalWindowSizeModulated = optimalWindowSizeModulated;
        results.optimalWindowSizeUnmodulated = optimalWindowSizeUnmodulated;
    else
        results.analyzeModulation = false;
    end
end


