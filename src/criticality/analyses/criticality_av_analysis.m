function results = criticality_av_analysis(dataStruct, config)
% CRITICALITY_AV_ANALYSIS Perform avalanche analysis (dcc + kappa) in sliding windows
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config - Configuration structure with fields:
%     .slidingWindowSize - Window size in seconds
%     .avStepSize - Step size in seconds
%     .analyzeDcc - Compute dcc (default: true)
%     .analyzeKappa - Compute kappa (default: true)
%     .pcaFlag - Use PCA (default: 0)
%     .pcaFirstFlag - Use first nDim if 1, last nDim if 0 (default: 1)
%     .nDim - Number of PCA dimensions (default: 4)
%     .enablePermutations - Perform circular permutations (default: true)
%     .nShuffles - Number of permutations (default: 3)
%     .thresholdFlag - Use threshold method (default: 1)
%     .thresholdPct - Threshold as percentage of median (default: 1)
%     .makePlots - Create plots (default: true)
%     .saveDir - Save directory (optional, uses dataStruct.saveDir)
%     .includeM2356 - Include combined M23+M56 area (default: false)
%
% Goal:
%   Compute avalanche-based criticality measures (dcc, kappa) in sliding windows.
%   Supports PCA and permutation testing.
%
% Returns:
%   results - Structure with dcc, kappa, decades, tau, alpha, paramSD, startS, and params

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
    
    % Add combined M2356 area if requested and M23 and M56 exist
    if config.includeM2356
        idxM23 = find(strcmp(areas, 'M23'));
        idxM56 = find(strcmp(areas, 'M56'));
        if ~isempty(idxM23) && ~isempty(idxM56) && ~any(strcmp(areas, 'M2356'))
            % Create combined M2356 area
            areas{end+1} = 'M2356';
            dataStruct.areas = areas;  % Update dataStruct.areas
            dataStruct.idMatIdx{end+1} = [dataStruct.idMatIdx{idxM23}, dataStruct.idMatIdx{idxM56}];
            if isfield(dataStruct, 'idLabel')
                dataStruct.idLabel{end+1} = [dataStruct.idLabel{idxM23}; dataStruct.idLabel{idxM56}];
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
    
    fprintf('\n=== Criticality Avalanche Analysis Setup ===\n');
    fprintf('Data type: %s\n', dataType);
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('Window size: %.2f s, Step size: %.2f s\n', config.slidingWindowSize, config.avStepSize);
    
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
    
    resultsPath = create_results_path('criticality_av', dataType, ...
        sessionNameForPath, config.saveDir, 'filenameSuffix', filenameSuffix);
    
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
    [binSize, slidingWindowSize] = find_optimal_parameters_av(...
        reconstructedDataMat, dataStruct, config, areasToTest);
    
    % Filter areas based on valid optimal bin sizes
    if ~isfield(config, 'minNeurons')
        config.minNeurons = 10;
    end
    if ~isfield(config, 'candidateFrameSizes')
        config.candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15];
    end
    validMask = arrayfun(@(a) length(dataStruct.idMatIdx{a}) >= config.minNeurons & ...
        binSize(a) <= max(config.candidateFrameSizes), areasToTest);
    areasToTest = areasToTest(validMask);
    
    % Validate avStepSize is provided
    if ~isfield(config, 'avStepSize') || isempty(config.avStepSize)
        error('avStepSize must be provided in config');
    end
    
    % Calculate common centerTime values based on slidingWindowSize and avStepSize
    % This ensures all areas have aligned windows regardless of their optimized window sizes
    % Total time from original data (in seconds, assuming 1000 Hz)
    totalTime = size(dataStruct.dataMat, 1) / 1000;
    
    % Generate common centerTime values
    % Start from slidingWindowSize/2, end at totalTime - slidingWindowSize/2
    firstCenterTime = config.slidingWindowSize / 2;
    lastCenterTime = totalTime - config.slidingWindowSize / 2;
    commonCenterTimes = firstCenterTime:config.avStepSize:lastCenterTime;
    
    if isempty(commonCenterTimes)
        error('No valid windows found. Check slidingWindowSize and avStepSize relative to total time.');
    end
    
    numWindows = length(commonCenterTimes);
    fprintf('\nCommon window centers: %d windows from %.2f s to %.2f s (stepSize=%.3f s)\n', ...
        numWindows, firstCenterTime, lastCenterTime, config.avStepSize);
    
    % Initialize results
    [dcc, kappa, decades, startS, tau, alpha, paramSD] = deal(cell(1, numAreas));
    
    if config.enablePermutations
        dccPermuted = cell(1, numAreas);
        kappaPermuted = cell(1, numAreas);
        decadesPermuted = cell(1, numAreas);
        tauPermuted = cell(1, numAreas);
        alphaPermuted = cell(1, numAreas);
        paramSDPermuted = cell(1, numAreas);
        for a = 1:numAreas
            dccPermuted{a} = [];
            kappaPermuted{a} = [];
            decadesPermuted{a} = [];
            tauPermuted{a} = [];
            alphaPermuted{a} = [];
            paramSDPermuted{a} = [];
        end
    end
    
    % Main analysis loop
    fprintf('\n=== Processing Areas ===\n');
    for a = areasToTest
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataType);
        tic;
        
        aID = dataStruct.idMatIdx{a};
        
        % Bin the original data for dcc/kappa analysis
        aDataMat_dcc = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), binSize(a));
        numTimePoints_dcc = size(aDataMat_dcc, 1);
        
        % Apply PCA to binned data if needed
        if config.pcaFlag
            [coeff, score, ~, ~, explained, mu] = pca(aDataMat_dcc);
            forDim = find(cumsum(explained) > 30, 1);
            forDim = max(3, min(6, forDim));
            nDim = 1:forDim;
            aDataMat_dcc = score(:,nDim) * coeff(:,nDim)' + mu;
        end
        
        % Calculate population activity
        aDataMat_dcc = mean(aDataMat_dcc, 2);
        
        % Initialize arrays
        dcc{a} = nan(1, numWindows);
        kappa{a} = nan(1, numWindows);
        decades{a} = nan(1, numWindows);
        tau{a} = nan(1, numWindows);
        alpha{a} = nan(1, numWindows);
        paramSD{a} = nan(1, numWindows);
        startS{a} = nan(1, numWindows);
        
        % Process each window using common centerTime
        for w = 1:numWindows
            centerTime = commonCenterTimes(w);
            startS{a}(w) = centerTime;
            
            % Convert centerTime to indices for this area's binning
            % Use area-specific optimal window size
                [startIdx, endIdx] = calculate_window_indices_from_center(...
                    centerTime, slidingWindowSize(a), binSize(a), numTimePoints_dcc);
            
            % Check if window is valid (within bounds)
            if startIdx < 1 || endIdx > numTimePoints_dcc || startIdx > endIdx
                % Window is out of bounds for this area, skip
                continue;
            end
            
            % Calculate population activity for this window
            wPopActivity = aDataMat_dcc(startIdx:endIdx);
            
            % Apply thresholding using median of this window
            threshSpikes = median(wPopActivity);
            wPopActivity(wPopActivity < threshSpikes) = 0;
            
            % Avalanche analysis
            zeroBins = find(wPopActivity == 0);
            if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                [sizes, durs] = getAvalanches(wPopActivity', .5, 1);
                gof = .8;
                plotAv = 0;
                [tauVal, plrS, minavS, maxavS, ~, ~, ~] = plfit2023(sizes, gof, plotAv, 0);
                [alphaVal, plrD, minavD, maxavD, ~, ~, ~] = plfit2023(durs, gof, plotAv, 0);
                [paramSDVal, sigmaNuZInvStd, logCoeff] = size_given_duration(sizes, durs, ...
                    'durmin', minavD, 'durmax', maxavD);
                
                % dcc (distance to criticality from avalanche analysis)
                dcc{a}(w) = distance_to_criticality(tauVal, alphaVal, paramSDVal);
                
                % kappa (avalanche shape parameter)
                kappa{a}(w) = compute_kappa(sizes);
                
                % decades (log10 of avalanche size range)
                decades{a}(w) = plrS;
                
                % Store tau, alpha, and paramSD
                tau{a}(w) = tauVal;
                alpha{a}(w) = alphaVal;
                paramSD{a}(w) = paramSDVal;
            end
        end
        
        fprintf('Area %s completed in %.1f minutes\n', areas{a}, toc/60);
        
        % Perform circular permutations if enabled
        if config.enablePermutations
            [dccPermuted{a}, kappaPermuted{a}, decadesPermuted{a}, ...
                tauPermuted{a}, alphaPermuted{a}, paramSDPermuted{a}] = ...
                perform_circular_permutations_av(dataStruct, a, aID, binSize(a), ...
                slidingWindowSize(a), config, commonCenterTimes, numTimePoints_dcc);
        end
    end
    
    % Build results structure
    results = build_results_structure_av(dataStruct, config, areas, areasToTest, ...
        dcc, kappa, decades, startS, tau, alpha, paramSD, ...
        binSize, slidingWindowSize, ...
        dccPermuted, kappaPermuted, decadesPermuted, ...
        tauPermuted, alphaPermuted, paramSDPermuted);
    
    % Save results
    save(resultsPath, 'results');
    fprintf('Saved %s dcc/kappa to %s\n', dataType, resultsPath);
    
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
        plot_criticality_av_results(results, plotConfig, config, dataStruct, filenameSuffix);
    end
end

function config = set_config_defaults(config)
% SET_CONFIG_DEFAULTS Set default values for configuration structure
    
    defaults = struct();
    defaults.analyzeDcc = true;
    defaults.analyzeKappa = true;
    defaults.pcaFlag = 0;
    defaults.pcaFirstFlag = 1;
    defaults.nDim = 4;
    defaults.enablePermutations = true;
    defaults.nShuffles = 3;
    defaults.makePlots = true;
    defaults.minNeurons = 10;
    defaults.minSpikesPerBin = 4;
    defaults.maxSpikesPerBin = 50;
    defaults.minBinsPerWindow = 1000;
    defaults.thresholdFlag = 1;
    defaults.thresholdPct = 1;
    defaults.candidateFrameSizes = [.02 .03 .04 0.05, .075, 0.1 .15];
    defaults.candidateWindowSizes = [30, 45, 60, 90, 120];
    defaults.includeM2356 = false;  % Include combined M23+M56 area (optional)
    
    % Apply defaults
    fields = fieldnames(defaults);
    for i = 1:length(fields)
        if ~isfield(config, fields{i})
            config.(fields{i}) = defaults.(fields{i});
        end
    end
end

function [binSize, slidingWindowSize] = find_optimal_parameters_av(...
    reconstructedDataMat, dataStruct, config, areasToTest)
% FIND_OPTIMAL_PARAMETERS_AV Find optimal bin and window sizes for avalanche analysis
    
    numAreas = length(dataStruct.areas);
    binSize = zeros(1, numAreas);
    slidingWindowSize = zeros(1, numAreas);
    
    for a = areasToTest
        thisDataMat = reconstructedDataMat{a};
        thisFiringRate = sum(thisDataMat(:)) / (size(thisDataMat, 1)/1000);
        [binSize(a), slidingWindowSize(a)] = ...
            find_optimal_bin_and_window(thisFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s\n', ...
            dataStruct.areas{a}, binSize(a), slidingWindowSize(a));
    end
end

function [dccPermuted, kappaPermuted, decadesPermuted, ...
    tauPermuted, alphaPermuted, paramSDPermuted] = ...
    perform_circular_permutations_av(dataStruct, a, aID, binSize, ...
    slidingWindowSize, config, commonCenterTimes, numTimePoints)
% PERFORM_CIRCULAR_PERMUTATIONS_AV Perform circular permutation testing for avalanche analysis
    
    fprintf('  Running %d circular permutations per window for area %s...\n', ...
        config.nShuffles, dataStruct.areas{a});
    ticPerm = tic;
    
    numWindows = length(commonCenterTimes);
    
    % Initialize storage
    dccPermuted = nan(numWindows, config.nShuffles);
    kappaPermuted = nan(numWindows, config.nShuffles);
    decadesPermuted = nan(numWindows, config.nShuffles);
    tauPermuted = nan(numWindows, config.nShuffles);
    alphaPermuted = nan(numWindows, config.nShuffles);
    paramSDPermuted = nan(numWindows, config.nShuffles);
    
    % Get original binned data matrix
    originalDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), binSize);
    nNeurons = size(originalDataMat, 2);
    
    % Permute each window independently
    for w = 1:numWindows
        centerTime = commonCenterTimes(w);
        
        % Convert centerTime to indices for this area's binning
        % Use area-specific optimal window size
            [startIdx, endIdx] = calculate_window_indices_from_center(...
                centerTime, slidingWindowSize, binSize, numTimePoints);
        
        % Check if window is valid (within bounds)
        if startIdx < 1 || endIdx > numTimePoints || startIdx > endIdx
            % Window is out of bounds, skip
            continue;
        end
        
        % Extract this window's data
        windowData = originalDataMat(startIdx:endIdx, :);
        winSamples = size(windowData, 1);
        
        % For each shuffle, permute this window's data independently
        for shuffle = 1:config.nShuffles
            % Circularly permute each neuron independently within this window
            permutedWindowData = windowData;
            for n = 1:nNeurons
                shiftAmount = randi([1, winSamples]);
                permutedWindowData(:, n) = circshift(permutedWindowData(:, n), shiftAmount);
            end
            
            % Apply PCA if needed
            if config.pcaFlag
                [coeffPerm, scorePerm, ~, ~, explainedPerm, muPerm] = pca(permutedWindowData);
                forDimPerm = find(cumsum(explainedPerm) > 30, 1);
                forDimPerm = max(3, min(6, forDimPerm));
                nDimPerm = 1:forDimPerm;
                permutedWindowData = scorePerm(:,nDimPerm) * coeffPerm(:,nDimPerm)' + muPerm;
            end
            
            % Calculate population activity for this permuted window
            wPopActivityPerm = mean(permutedWindowData, 2);
            
            % Apply thresholding using median of this window
            threshSpikes = median(wPopActivityPerm);
            wPopActivityPerm(wPopActivityPerm < threshSpikes) = 0;
            
            % Avalanche analysis for permuted data
            zeroBins = find(wPopActivityPerm == 0);
            if length(zeroBins) > 1 && any(diff(zeroBins)>1)
                [sizesPerm, dursPerm] = getAvalanches(wPopActivityPerm', .5, 1);
                gof = .8;
                plotAv = 0;
                [tauPerm, plrSPerm, minavSPerm, maxavSPerm, ~, ~, ~] = plfit2023(sizesPerm, gof, plotAv, 0);
                [alphaPerm, plrDPerm, minavDPerm, maxavDPerm, ~, ~, ~] = plfit2023(dursPerm, gof, plotAv, 0);
                [paramSDPerm, sigmaNuZInvStdPerm, logCoeffPerm] = size_given_duration(sizesPerm, dursPerm, ...
                    'durmin', minavDPerm, 'durmax', maxavDPerm);
                
                % Calculate metrics for permuted data
                dccPermuted(w, shuffle) = distance_to_criticality(tauPerm, alphaPerm, paramSDPerm);
                kappaPermuted(w, shuffle) = compute_kappa(sizesPerm);
                decadesPermuted(w, shuffle) = plrSPerm;
                tauPermuted(w, shuffle) = tauPerm;
                alphaPermuted(w, shuffle) = alphaPerm;
                paramSDPermuted(w, shuffle) = paramSDPerm;
            end
        end
        
        if mod(w, max(1, round(numWindows/10))) == 0
            fprintf('    Completed %d/%d windows (%.1f min elapsed)\n', w, numWindows, toc(ticPerm)/60);
        end
    end
    
    fprintf('  Permutations completed in %.1f minutes\n', toc(ticPerm)/60);
end

function results = build_results_structure_av(dataStruct, config, areas, areasToTest, ...
    dcc, kappa, decades, startS, tau, alpha, paramSD, ...
    binSize, slidingWindowSize, ...
    dccPermuted, kappaPermuted, decadesPermuted, ...
    tauPermuted, alphaPermuted, paramSDPermuted)
% BUILD_RESULTS_STRUCTURE_AV Build results structure for avalanche analysis
    
    results = struct();
    results.dataType = dataStruct.dataType;
    results.areas = areas;
    results.dcc = dcc;
    results.kappa = kappa;
    results.decades = decades;
    results.tau = tau;
    results.alpha = alpha;
    results.paramSD = paramSD;
    results.startS = startS;
    results.binSize = binSize;
    results.slidingWindowSize = slidingWindowSize;
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.avStepSize = config.avStepSize;
    results.params.pcaFlag = config.pcaFlag;
    results.params.pcaFirstFlag = config.pcaFirstFlag;
    results.params.nDim = config.nDim;
    results.params.thresholdFlag = config.thresholdFlag;
    results.params.thresholdPct = config.thresholdPct;
    
    if config.enablePermutations
        results.enablePermutations = true;
        results.nShuffles = config.nShuffles;
        results.dccPermuted = dccPermuted;
        results.kappaPermuted = kappaPermuted;
        results.decadesPermuted = decadesPermuted;
        results.tauPermuted = tauPermuted;
        results.alphaPermuted = alphaPermuted;
        results.paramSDPermuted = paramSDPermuted;
        
        % Calculate mean and SEM
        dccPermutedMean = cell(1, length(areas));
        dccPermutedSEM = cell(1, length(areas));
        kappaPermutedMean = cell(1, length(areas));
        kappaPermutedSEM = cell(1, length(areas));
        decadesPermutedMean = cell(1, length(areas));
        decadesPermutedSEM = cell(1, length(areas));
        tauPermutedMean = cell(1, length(areas));
        tauPermutedSEM = cell(1, length(areas));
        alphaPermutedMean = cell(1, length(areas));
        alphaPermutedSEM = cell(1, length(areas));
        paramSDPermutedMean = cell(1, length(areas));
        paramSDPermutedSEM = cell(1, length(areas));
        
        for a = 1:length(areas)
            if ~isempty(dccPermuted{a})
                dccPermutedMean{a} = nanmean(dccPermuted{a}, 2);
                dccPermutedSEM{a} = nanstd(dccPermuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                dccPermutedMean{a} = [];
                dccPermutedSEM{a} = [];
            end
            if ~isempty(kappaPermuted{a})
                kappaPermutedMean{a} = nanmean(kappaPermuted{a}, 2);
                kappaPermutedSEM{a} = nanstd(kappaPermuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                kappaPermutedMean{a} = [];
                kappaPermutedSEM{a} = [];
            end
            if ~isempty(decadesPermuted{a})
                decadesPermutedMean{a} = nanmean(decadesPermuted{a}, 2);
                decadesPermutedSEM{a} = nanstd(decadesPermuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                decadesPermutedMean{a} = [];
                decadesPermutedSEM{a} = [];
            end
            if ~isempty(tauPermuted{a})
                tauPermutedMean{a} = nanmean(tauPermuted{a}, 2);
                tauPermutedSEM{a} = nanstd(tauPermuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                tauPermutedMean{a} = [];
                tauPermutedSEM{a} = [];
            end
            if ~isempty(alphaPermuted{a})
                alphaPermutedMean{a} = nanmean(alphaPermuted{a}, 2);
                alphaPermutedSEM{a} = nanstd(alphaPermuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                alphaPermutedMean{a} = [];
                alphaPermutedSEM{a} = [];
            end
            if ~isempty(paramSDPermuted{a})
                paramSDPermutedMean{a} = nanmean(paramSDPermuted{a}, 2);
                paramSDPermutedSEM{a} = nanstd(paramSDPermuted{a}, 0, 2) / sqrt(config.nShuffles);
            else
                paramSDPermutedMean{a} = [];
                paramSDPermutedSEM{a} = [];
            end
        end
        
        results.dccPermutedMean = dccPermutedMean;
        results.dccPermutedSEM = dccPermutedSEM;
        results.kappaPermutedMean = kappaPermutedMean;
        results.kappaPermutedSEM = kappaPermutedSEM;
        results.decadesPermutedMean = decadesPermutedMean;
        results.decadesPermutedSEM = decadesPermutedSEM;
        results.tauPermutedMean = tauPermutedMean;
        results.tauPermutedSEM = tauPermutedSEM;
        results.alphaPermutedMean = alphaPermutedMean;
        results.alphaPermutedSEM = alphaPermutedSEM;
        results.paramSDPermutedMean = paramSDPermutedMean;
        results.paramSDPermutedSEM = paramSDPermutedSEM;
    else
        results.enablePermutations = false;
        results.nShuffles = 0;
    end
end

function plot_criticality_av_results(results, plotConfig, config, dataStruct, filenameSuffix)
% PLOT_CRITICALITY_AV_RESULTS Create plots for criticality avalanche analysis
    
    criticality_av_plot(results, plotConfig, config, dataStruct, filenameSuffix);
end

