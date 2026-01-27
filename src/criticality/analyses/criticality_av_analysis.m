function results = criticality_av_analysis(dataStruct, config)
% CRITICALITY_AV_ANALYSIS Perform avalanche analysis (dcc + kappa) in sliding windows
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data() with spikeTimes, spikeClusters
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
%   Uses spike times directly with on-demand binning. Supports PCA and permutation testing.
%
% Returns:
%   results - Structure with dcc, kappa, decades, tau, alpha, paramSD, startS, and params

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
    
    fprintf('\n=== Criticality Avalanche Analysis Setup ===\n');
    fprintf('Data type: %s\n', sessionType);
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('Window size: %.2f s, Step size: %.2f s\n', config.slidingWindowSize, config.avStepSize);
    
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
    
    resultsPath = create_results_path('criticality_av', sessionType, ...
        sessionNameForPath, config.saveDir, 'filenameSuffix', filenameSuffix);
    
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
    
    % Find optimal parameters
    fprintf('\n--- Step 3: Finding optimal parameters ---\n');
    if config.pcaFlag
        % Use PCA-reconstructed data for parameter optimization
        [binSize, slidingWindowSize] = find_optimal_parameters_av_pca(...
            reconstructedDataMat, config, areasToTest, timeRange, tempBinSize, areas);
    else
        % Use original spike times for parameter optimization
        [binSize, slidingWindowSize] = find_optimal_parameters_av(...
            dataStruct, config, areasToTest, timeRange);
    end
    
    % Filter areas based on valid optimal bin sizes and minimum neurons
    if ~isfield(config, 'nMinNeurons')
        config.nMinNeurons = 10;
    end
    
    % Pre-loop filtering: identify areas to skip
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
            dcc{a} = [];
            kappa{a} = [];
            decades{a} = [];
            tau{a} = [];
            alpha{a} = [];
            paramSD{a} = [];
            startS{a} = [];
            dccNormalized{a} = [];
            kappaNormalized{a} = [];
            decadesNormalized{a} = [];
            tauNormalized{a} = [];
            alphaNormalized{a} = [];
            paramSDNormalized{a} = [];
            if config.enablePermutations
                dccPermuted{a} = [];
                kappaPermuted{a} = [];
                decadesPermuted{a} = [];
                tauPermuted{a} = [];
                alphaPermuted{a} = [];
                paramSDPermuted{a} = [];
            end
        else
            areasToProcess = [areasToProcess, a];
        end
    end
    
    if isempty(areasToProcess)
        error('No valid areas to process. All areas were skipped due to insufficient neurons or invalid bin sizes.');
    end
    
    fprintf('  Will process %d area(s): %s\n', length(areasToProcess), strjoin(areas(areasToProcess), ', '));
    
    % Update areasToTest to only include areas to process
    areasToTest = areasToProcess;
    
    % Validate avStepSize is provided
    if ~isfield(config, 'avStepSize') || isempty(config.avStepSize)
        error('avStepSize must be provided in config');
    end
    
    % Calculate common centerTime values based on slidingWindowSize and avStepSize
    % This ensures all areas have aligned windows regardless of their optimized window sizes
    % Total time from spike data
    totalTime = timeRange(2) - timeRange(1);
    
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
    [dccNormalized, kappaNormalized, decadesNormalized, tauNormalized, alphaNormalized, paramSDNormalized] = ...
        deal(cell(1, numAreas));
    
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
    for a = areasToProcess
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, sessionType);
        tic;
        
        aID = dataStruct.idMatIdx{a};
        
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
            aDataMat_dcc = zeros(numBins_optimal, size(reconstructedMat_1ms, 2));
            for b = 1:numBins_optimal
                startIdx_1ms = round((b-1) * binsPerOptimalBin) + 1;
                endIdx_1ms = min(round(b * binsPerOptimalBin), numBins_1ms);
                if startIdx_1ms <= numBins_1ms
                    aDataMat_dcc(b, :) = mean(reconstructedMat_1ms(startIdx_1ms:endIdx_1ms, :), 1);
                end
            end
            numTimePoints_dcc = size(aDataMat_dcc, 1);
        else
            % Bin spikes on-demand at area-specific bin size
            aDataMat_dcc = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                neuronIDs, timeRange, binSize(a));
            numTimePoints_dcc = size(aDataMat_dcc, 1);
        end
        
        % Calculate population activity
        aDataMat_dcc = sum(aDataMat_dcc, 2);
        
        % Initialize arrays
        dcc{a} = nan(1, numWindows);
        kappa{a} = nan(1, numWindows);
        decades{a} = nan(1, numWindows);
        tau{a} = nan(1, numWindows);
        alpha{a} = nan(1, numWindows);
        paramSD{a} = nan(1, numWindows);
        startS{a} = nan(1, numWindows);
        dccNormalized{a} = nan(1, numWindows);
        kappaNormalized{a} = nan(1, numWindows);
        decadesNormalized{a} = nan(1, numWindows);
        tauNormalized{a} = nan(1, numWindows);
        alphaNormalized{a} = nan(1, numWindows);
        paramSDNormalized{a} = nan(1, numWindows);
        
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
            if config.pcaFlag
                % Use PCA-reconstructed data for permutations
                [dccPermuted{a}, kappaPermuted{a}, decadesPermuted{a}, ...
                    tauPermuted{a}, alphaPermuted{a}, paramSDPermuted{a}] = ...
                    perform_circular_permutations_av_pca(reconstructedDataMat{a}, a, binSize(a), ...
                    slidingWindowSize(a), config, commonCenterTimes, numTimePoints_dcc, timeRange, tempBinSize);
            else
                % Use original spike times for permutations
                [dccPermuted{a}, kappaPermuted{a}, decadesPermuted{a}, ...
                    tauPermuted{a}, alphaPermuted{a}, paramSDPermuted{a}] = ...
                    perform_circular_permutations_av(dataStruct, a, neuronIDs, binSize(a), ...
                    slidingWindowSize(a), config, commonCenterTimes, numTimePoints_dcc, timeRange);
            end
            
            % Normalize metrics by shuffled metric values if requested
            if config.normalizeMetrics && ~isempty(dccPermuted{a})
                % Calculate mean shuffled metrics for each window
                dccPermutedMean = nanmean(dccPermuted{a}, 2);
                kappaPermutedMean = nanmean(kappaPermuted{a}, 2);
                decadesPermutedMean = nanmean(decadesPermuted{a}, 2);
                tauPermutedMean = nanmean(tauPermuted{a}, 2);
                alphaPermutedMean = nanmean(alphaPermuted{a}, 2);
                paramSDPermutedMean = nanmean(paramSDPermuted{a}, 2);
                
                % Normalize: metricNormalized = metric / mean(shuffled_metric)
                for w = 1:numWindows
                    if ~isnan(dcc{a}(w)) && ~isnan(dccPermutedMean(w)) && dccPermutedMean(w) > 0
                        dccNormalized{a}(w) = dcc{a}(w) / dccPermutedMean(w);
                    else
                        dccNormalized{a}(w) = nan;
                    end
                    
                    if ~isnan(kappa{a}(w)) && ~isnan(kappaPermutedMean(w)) && kappaPermutedMean(w) > 0
                        kappaNormalized{a}(w) = kappa{a}(w) / kappaPermutedMean(w);
                    else
                        kappaNormalized{a}(w) = nan;
                    end
                    
                    if ~isnan(decades{a}(w)) && ~isnan(decadesPermutedMean(w)) && decadesPermutedMean(w) > 0
                        decadesNormalized{a}(w) = decades{a}(w) / decadesPermutedMean(w);
                    else
                        decadesNormalized{a}(w) = nan;
                    end
                    
                    if ~isnan(tau{a}(w)) && ~isnan(tauPermutedMean(w)) && tauPermutedMean(w) > 0
                        tauNormalized{a}(w) = tau{a}(w) / tauPermutedMean(w);
                    else
                        tauNormalized{a}(w) = nan;
                    end
                    
                    if ~isnan(alpha{a}(w)) && ~isnan(alphaPermutedMean(w)) && alphaPermutedMean(w) > 0
                        alphaNormalized{a}(w) = alpha{a}(w) / alphaPermutedMean(w);
                    else
                        alphaNormalized{a}(w) = nan;
                    end
                    
                    if ~isnan(paramSD{a}(w)) && ~isnan(paramSDPermutedMean(w)) && paramSDPermutedMean(w) > 0
                        paramSDNormalized{a}(w) = paramSD{a}(w) / paramSDPermutedMean(w);
                    else
                        paramSDNormalized{a}(w) = nan;
                    end
                end
            else
                % If normalization disabled or no permutations, set to NaN
                dccNormalized{a}(:) = nan;
                kappaNormalized{a}(:) = nan;
                decadesNormalized{a}(:) = nan;
                tauNormalized{a}(:) = nan;
                alphaNormalized{a}(:) = nan;
                paramSDNormalized{a}(:) = nan;
            end
        else
            % Initialize empty if permutations disabled
            if isempty(dccPermuted{a})
                dccPermuted{a} = [];
            end
            if isempty(kappaPermuted{a})
                kappaPermuted{a} = [];
            end
            if isempty(decadesPermuted{a})
                decadesPermuted{a} = [];
            end
            if isempty(tauPermuted{a})
                tauPermuted{a} = [];
            end
            if isempty(alphaPermuted{a})
                alphaPermuted{a} = [];
            end
            if isempty(paramSDPermuted{a})
                paramSDPermuted{a} = [];
            end
            % If no permutations, normalization not possible
            dccNormalized{a}(:) = nan;
            kappaNormalized{a}(:) = nan;
            decadesNormalized{a}(:) = nan;
            tauNormalized{a}(:) = nan;
            alphaNormalized{a}(:) = nan;
            paramSDNormalized{a}(:) = nan;
        end
    end
    
    % Build results structure
    results = build_results_structure_av(dataStruct, config, areas, areasToProcess, ...
        dcc, kappa, decades, startS, tau, alpha, paramSD, ...
        dccNormalized, kappaNormalized, decadesNormalized, tauNormalized, alphaNormalized, paramSDNormalized, ...
        binSize, slidingWindowSize, ...
        dccPermuted, kappaPermuted, decadesPermuted, ...
        tauPermuted, alphaPermuted, paramSDPermuted);
    
    % Save results if requested
    if config.saveData
        save(resultsPath, 'results');
        fprintf('Saved %s dcc/kappa to %s\n', sessionType, resultsPath);
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
    defaults.saveData = true;  % Set to false to skip saving results
    defaults.useOptimalBinWindowFunction = true;  % Find optimal bin/window sizes
    defaults.nMinNeurons = 10;  % Minimum number of neurons required
    defaults.minSpikesPerBin = 4;
    defaults.maxSpikesPerBin = 50;
    defaults.minBinsPerWindow = 1000;
    defaults.thresholdFlag = 1;
    defaults.thresholdPct = 1;
    defaults.normalizeMetrics = true;  % Normalize metrics by shuffled metric values
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
    dataStruct, config, areasToTest, timeRange)
% FIND_OPTIMAL_PARAMETERS_AV Find optimal bin and window sizes for avalanche analysis
% Uses spike times directly (new approach)
    
    numAreas = length(dataStruct.areas);
    binSize = zeros(1, numAreas);
    slidingWindowSize = zeros(1, numAreas);
    
    for a = areasToTest
        % Get neuron IDs for this area
        neuronIDs = dataStruct.idLabel{a};
        
        % Calculate firing rate from spike times
        thisFiringRate = calculate_firing_rate_from_spikes(...
            dataStruct.spikeTimes, dataStruct.spikeClusters, ...
            neuronIDs, timeRange);
        
        [binSize(a), slidingWindowSize(a)] = ...
            find_optimal_bin_and_window(thisFiringRate, config.minSpikesPerBin, config.minBinsPerWindow);
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s\n', ...
            dataStruct.areas{a}, binSize(a), slidingWindowSize(a));
    end
end

function [binSize, slidingWindowSize] = find_optimal_parameters_av_pca(...
    reconstructedDataMat, config, areasToTest, timeRange, tempBinSize, areas)
% FIND_OPTIMAL_PARAMETERS_AV_PCA Find optimal bin and window sizes for avalanche analysis
% Uses PCA-reconstructed data matrices (for when pcaFlag = 1)
%
% Variables:
%   reconstructedDataMat - Cell array of PCA-reconstructed data matrices [timeBins x neurons] at tempBinSize
%   config - Configuration structure
%   areasToTest - Areas to process
%   timeRange - [startTime, endTime] in seconds
%   tempBinSize - Bin size used for PCA reconstruction (typically 0.001s = 1ms)
%   areas - Cell array of area names
%
% Returns:
%   binSize - Optimal bin sizes per area
%   slidingWindowSize - Optimal window sizes per area
    
    numAreas = length(reconstructedDataMat);
    binSize = zeros(1, numAreas);
    slidingWindowSize = zeros(1, numAreas);
    
    totalTime = timeRange(2) - timeRange(1);
    
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
        fprintf('Area %s: bin size = %.3f s, window size = %.1f s (from PCA-reconstructed data)\n', ...
            areas{a}, binSize(a), slidingWindowSize(a));
    end
end

function [dccPermuted, kappaPermuted, decadesPermuted, ...
    tauPermuted, alphaPermuted, paramSDPermuted] = ...
    perform_circular_permutations_av(dataStruct, a, neuronIDs, binSize, ...
    slidingWindowSize, config, commonCenterTimes, numTimePoints, timeRange)
% PERFORM_CIRCULAR_PERMUTATIONS_AV Perform circular permutation testing for avalanche analysis
%   Shuffles each neuron's activity independently using circular shifts,
%   then computes population activity from the shuffled data.
    
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
    
    % Bin spikes on-demand at area-specific bin size for the full time range
    originalDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
        neuronIDs, timeRange, binSize);
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
            wPopActivityPerm = sum(permutedWindowData, 2);
            
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

function [dccPermuted, kappaPermuted, decadesPermuted, ...
    tauPermuted, alphaPermuted, paramSDPermuted] = ...
    perform_circular_permutations_av_pca(reconstructedMat_1ms, a, binSize, ...
    slidingWindowSize, config, commonCenterTimes, numTimePoints, timeRange, tempBinSize)
% PERFORM_CIRCULAR_PERMUTATIONS_AV_PCA Perform circular permutation testing for avalanche analysis
%   Uses PCA-reconstructed data (for when pcaFlag = 1)
%   Shuffles each neuron's activity independently using circular shifts,
%   then computes population activity from the shuffled data.
%
% Variables:
%   reconstructedMat_1ms - PCA-reconstructed data matrix [timeBins_1ms x neurons] at tempBinSize
%   a - Area index
%   binSize - Optimal bin size for this area
%   slidingWindowSize - Optimal window size for this area
%   config - Configuration structure
%   commonCenterTimes - Window center times
%   numTimePoints - Number of time points at optimal bin size
%   timeRange - [startTime, endTime] in seconds
%   tempBinSize - Bin size used for PCA (typically 0.001s = 1ms)
    
    fprintf('  Running %d circular permutations per window for area %d (using PCA-reconstructed data)...\n', ...
        config.nShuffles, a);
    ticPerm = tic;
    
    numWindows = length(commonCenterTimes);
    
    % Initialize storage
    dccPermuted = nan(numWindows, config.nShuffles);
    kappaPermuted = nan(numWindows, config.nShuffles);
    decadesPermuted = nan(numWindows, config.nShuffles);
    tauPermuted = nan(numWindows, config.nShuffles);
    alphaPermuted = nan(numWindows, config.nShuffles);
    paramSDPermuted = nan(numWindows, config.nShuffles);
    
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
            
            % No need to re-apply PCA - data is already PCA-reconstructed
            % Just use the permuted window data directly
            
            % Calculate population activity for this permuted window
            wPopActivityPerm = sum(permutedWindowData, 2);
            
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

function results = build_results_structure_av(dataStruct, config, areas, areasToProcess, ...
    dcc, kappa, decades, startS, tau, alpha, paramSD, ...
    dccNormalized, kappaNormalized, decadesNormalized, tauNormalized, alphaNormalized, paramSDNormalized, ...
    binSize, slidingWindowSize, ...
    dccPermuted, kappaPermuted, decadesPermuted, ...
    tauPermuted, alphaPermuted, paramSDPermuted)
% BUILD_RESULTS_STRUCTURE_AV Build results structure for avalanche analysis
    
    results = struct();
    results.sessionType = dataStruct.sessionType;
    results.areas = areas;
    results.dcc = dcc;  % Raw dcc values
    results.dccNormalized = dccNormalized;  % Normalized dcc values (dcc / mean(shuffled_dcc))
    results.kappa = kappa;  % Raw kappa values
    results.kappaNormalized = kappaNormalized;  % Normalized kappa values
    results.decades = decades;  % Raw decades values
    results.decadesNormalized = decadesNormalized;  % Normalized decades values
    results.tau = tau;  % Raw tau values
    results.tauNormalized = tauNormalized;  % Normalized tau values
    results.alpha = alpha;  % Raw alpha values
    results.alphaNormalized = alphaNormalized;  % Normalized alpha values
    results.paramSD = paramSD;  % Raw paramSD values
    results.paramSDNormalized = paramSDNormalized;  % Normalized paramSD values
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
    results.params.normalizeMetrics = config.normalizeMetrics;
    
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

