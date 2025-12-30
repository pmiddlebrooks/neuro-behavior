function results = rqa_sliding_analysis(dataStruct, config)
% RQA_SLIDING_ANALYSIS Perform Recurrence Quantitative Analysis (RQA) in sliding windows
%
% Variables:
%   dataStruct - Data structure from load_sliding_window_data()
%   config - Configuration structure with fields:
%     .slidingWindowSize - Window size in seconds
%     .stepSize - Step size in seconds (optional, calculated if not provided)
%     .binSize - Bin size for spikes (required if dataSource == 'spikes')
%     .nPCADim - Number of PCA dimensions to use (default: 3)
%     .recurrenceThreshold - Threshold for recurrence (default: 'mean', or numeric value)
%     .nShuffles - Number of shuffles for normalization (default: 3)
%     .makePlots - Whether to create plots (default: true)
%     .saveDir - Save directory (optional, uses dataStruct.saveDir if not provided)
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
    
    % Validate inputs
    validate_workspace_vars({'sessionType', 'dataSource', 'areas'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
    
    % Set defaults
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
        config.recurrenceThreshold = 'mean';  % Can be 'mean', 'median', or numeric
    end
    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end
    
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
    
    fprintf('\n=== RQA Sliding Window Analysis Setup ===\n');
    fprintf('Data source: %s\n', dataSource);
    fprintf('Number of areas: %d\n', numAreas);
    fprintf('Window size: %.2f s\n', config.slidingWindowSize);
    fprintf('PCA dimensions: %d\n', config.nPCADim);
    
    % Validate data source specific requirements
    if strcmp(dataSource, 'spikes')
        validate_workspace_vars({'dataMat', 'idMatIdx'}, dataStruct, ...
            'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
        
        if ~isfield(config, 'binSize') || isempty(config.binSize)
            error('binSize must be provided in config for spike data analysis');
        end
        config.optimalBinSize = repmat(config.binSize, 1, numAreas);
        
        % Calculate step size if not provided
        if ~isfield(config, 'stepSize') || isempty(config.stepSize)
            config.stepSize = config.binSize * 2;  % Default: 2x bin size
        end
    else
        error('RQA analysis currently only supports spike data');
    end
    
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
    
    % Analysis loop
    fprintf('\n=== Processing Areas ===\n');
    
    for a = areasToTest
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataSource);
        tic;
        
        aID = dataStruct.idMatIdx{a};
        nNeurons = length(aID);
        
        % Determine number of PCA dimensions to use
        actualPCADim = min(config.nPCADim, nNeurons);
        fprintf('  Using %d PCA dimensions (requested: %d, neurons: %d)\n', ...
            actualPCADim, config.nPCADim, nNeurons);
        
        % Bin data using optimal bin size for this area
        aDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), config.optimalBinSize(a));
        numTimePoints = size(aDataMat, 1);
        
        % Calculate window and step sizes in samples
        winSamples = round(config.slidingWindowSize / config.optimalBinSize(a));
        stepSamples = round(config.stepSize / config.optimalBinSize(a));
        if stepSamples < 1
            stepSamples = 1;
        end
        
        % Calculate number of windows
        numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
        fprintf('  Time points: %d, Window samples: %d, Step samples: %d, Windows: %d\n', ...
            numTimePoints, winSamples, stepSamples, numWindows);
        fprintf('  Bins per window: %d (stability indicator)\n', winSamples);
        
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
        recurrencePlots{a} = cell(1, numWindows);  % Store recurrence plot for each window
        startS{a} = nan(1, numWindows);
        
        % Process each window
        for w = 1:numWindows
            [startIdx, endIdx, centerTime, ~] = calculate_window_indices(...
                w, numTimePoints, winSamples, stepSamples, config.optimalBinSize(a));
            startS{a}(w) = centerTime;
            
            % Extract window data [nSamples x nNeurons]
            wData = aDataMat(startIdx:endIdx, :);
            
            % Project to PCA space
            [coeff, score, ~, ~, explained, mu] = pca(wData);
            % Use first actualPCADim dimensions
            pcaData = score(:, 1:actualPCADim);
            
            % Calculate RQA metrics
            [recurrenceRate{a}(w), determinism{a}(w), laminarity{a}(w), ...
                trappingTime{a}(w), recurrencePlots{a}{w}] = ...
                compute_rqa_metrics(pcaData, config.recurrenceThreshold);
            
            % Normalize by shuffled version
            shuffledRR = nan(1, config.nShuffles);
            shuffledDET = nan(1, config.nShuffles);
            shuffledLAM = nan(1, config.nShuffles);
            shuffledTT = nan(1, config.nShuffles);
            
            for s = 1:config.nShuffles
                % Shuffle each dimension independently
                shuffledPCA = pcaData;
                for dim = 1:actualPCADim
                    shuffledPCA(:, dim) = pcaData(randperm(size(pcaData, 1)), dim);
                end
                
                [shuffledRR(s), shuffledDET(s), shuffledLAM(s), shuffledTT(s)] = ...
                    compute_rqa_metrics(shuffledPCA, config.recurrenceThreshold);
            end
            
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
            
            % Normalize by rate-matched Bernoulli control
            % Generate random PCA data with same statistics
            bernoulliRR = nan(1, config.nShuffles);
            bernoulliDET = nan(1, config.nShuffles);
            bernoulliLAM = nan(1, config.nShuffles);
            bernoulliTT = nan(1, config.nShuffles);
            
            for s = 1:config.nShuffles
                % Generate random data with same mean and std as original
                bernoulliPCA = zeros(size(pcaData));
                for dim = 1:actualPCADim
                    bernoulliPCA(:, dim) = randn(size(pcaData, 1), 1) * std(pcaData(:, dim)) + mean(pcaData(:, dim));
                end
                
                [bernoulliRR(s), bernoulliDET(s), bernoulliLAM(s), bernoulliTT(s)] = ...
                    compute_rqa_metrics(bernoulliPCA, config.recurrenceThreshold);
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
    results.recurrencePlots = recurrencePlots;
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.stepSize = config.stepSize;
    results.params.nShuffles = config.nShuffles;
    results.params.nPCADim = config.nPCADim;
    results.params.recurrenceThreshold = config.recurrenceThreshold;
    results.params.optimalBinSize = config.optimalBinSize;
    results.sessionType = sessionType;
    
    % Save results
    sessionNameForPath = '';
    if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
        sessionNameForPath = dataStruct.sessionName;
    end
    
    % Handle sessionName with subdirectories
    actualSaveDir = config.saveDir;
    if ~isempty(sessionNameForPath) && contains(sessionNameForPath, filesep)
        pathParts = strsplit(sessionNameForPath, filesep);
        if length(pathParts) > 1
            actualSaveDir = fullfile(config.saveDir, pathParts{1});
        end
    end
    
    % create_results_path expects dataType, but we have sessionType
    % For RQA, use sessionType as dataType
    resultsPath = create_results_path('rqa', sessionType, config.slidingWindowSize, ...
        sessionNameForPath, actualSaveDir, 'dataSource', dataSource);
    
    % Ensure directory exists before saving
    resultsDir = fileparts(resultsPath);
    if ~isempty(resultsDir)
        [status, msg] = mkdir(resultsDir);
        if ~status
            error('Failed to create results directory %s: %s', resultsDir, msg);
        end
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
        plotConfig = setup_plotting(config.saveDir, plotArgs{:});
        rqa_sliding_plot(results, plotConfig, config, dataStruct);
    end
end

function [RR, DET, LAM, TT, recurrencePlot] = compute_rqa_metrics(data, threshold)
% COMPUTE_RQA_METRICS Compute Recurrence Quantitative Analysis metrics
%
% Variables:
%   data - [nTimePoints x nDimensions] matrix (PCA projected data)
%   threshold - Threshold for recurrence ('mean', 'median', or numeric value)
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
        recurrencePlot = zeros(nPoints, nPoints);
        return;
    end
    
    % Compute distance matrix (Euclidean distance in PCA space)
    distMatrix = pdist2(data, data);
    
    % Determine threshold
    if ischar(threshold) || isstring(threshold)
        if strcmpi(threshold, 'mean')
            % Use mean of off-diagonal distances
            offDiagMask = ~eye(nPoints);
            thresholdValue = mean(distMatrix(offDiagMask));
        elseif strcmpi(threshold, 'median')
            offDiagMask = ~eye(nPoints);
            thresholdValue = median(distMatrix(offDiagMask));
        else
            error('Unknown threshold method: %s. Use ''mean'', ''median'', or numeric value', threshold);
        end
    else
        thresholdValue = threshold;
    end
    
    % Create recurrence plot (binary matrix)
    recurrencePlot = double(distMatrix <= thresholdValue);
    
    % Remove main diagonal (self-recurrence)
    recurrencePlot = recurrencePlot - eye(nPoints);
    recurrencePlot(recurrencePlot < 0) = 0;
    
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

