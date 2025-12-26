function results = complexity_analysis(dataStruct, config)
% COMPLEXITY_ANALYSIS Perform Lempel-Ziv complexity sliding window analysis
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
%     .saveDir - Save directory (optional, uses dataStruct.saveDir if not provided)
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
    validate_workspace_vars({'dataType', 'dataSource', 'areas'}, dataStruct, ...
        'errorMsg', 'Required field', 'source', 'load_sliding_window_data');
    
    % Set defaults
    if ~isfield(config, 'makePlots')
        config.makePlots = true;
    end
    if ~isfield(config, 'nShuffles')
        config.nShuffles = 3;
    end
    if ~isfield(config, 'lfpLowpassFreq')
        config.lfpLowpassFreq = 80;
    end
    if ~isfield(config, 'saveDir') || isempty(config.saveDir)
        config.saveDir = dataStruct.saveDir;
    end
    
    dataSource = dataStruct.dataSource;
    dataType = dataStruct.dataType;
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
    fprintf('Window size: %.2f s\n', config.slidingWindowSize);
    
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
            config.stepSize = config.slidingWindowSize / 10;  % Default: 10 steps per window
        end
        fprintf('Step size: %.3f s\n', config.stepSize);
    else
        error('Invalid dataSource. Must be ''spikes'' or ''lfp''.');
    end
    
    % Initialize results
    lzComplexity = cell(1, numAreas);
    lzComplexityNormalized = cell(1, numAreas);
    lzComplexityNormalizedBernoulli = cell(1, numAreas);
    startS = cell(1, numAreas);
    
    % Analysis loop
    fprintf('\n=== Processing Areas ===\n');
    
    for a = areasToTest
        fprintf('\nProcessing area %s (%s)...\n', areas{a}, dataSource);
        tic;
        
        if strcmp(dataSource, 'spikes')
            % ========== Spike Data Analysis ==========
            aID = dataStruct.idMatIdx{a};
            
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
            
            % Initialize arrays
            lzComplexity{a} = nan(1, numWindows);
            lzComplexityNormalized{a} = nan(1, numWindows);
            lzComplexityNormalizedBernoulli{a} = nan(1, numWindows);
            startS{a} = nan(1, numWindows);
            
            % Process each window
            for w = 1:numWindows
                [startIdx, endIdx, centerTime, ~] = calculate_window_indices(...
                    w, numTimePoints, winSamples, stepSamples, config.optimalBinSize(a));
                startS{a}(w) = centerTime;
                
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
                    compute_lz_complexity_with_controls(binarySeq, config.nShuffles);
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
            
            % Calculate window and step sizes in samples
            winSamples = round(config.slidingWindowSize * fsRaw);
            stepSamples = round(config.stepSize * fsRaw);
            if stepSamples < 1
                stepSamples = 1;
            end
            
            numTimePoints = length(filteredSignal);
            
            % Calculate number of windows
            numWindows = floor((numTimePoints - winSamples) / stepSamples) + 1;
            fprintf('  Time points: %d, Window samples: %d, Step samples: %d, Windows: %d\n', ...
                numTimePoints, winSamples, stepSamples, numWindows);
            
            % Initialize arrays
            lzComplexity{a} = nan(1, numWindows);
            lzComplexityNormalized{a} = nan(1, numWindows);
            lzComplexityNormalizedBernoulli{a} = nan(1, numWindows);
            startS{a} = nan(1, numWindows);
            
            % Process each window
            for w = 1:numWindows
                [startIdx, endIdx, centerTime, ~] = calculate_window_indices(...
                    w, numTimePoints, winSamples, stepSamples, 1/fsRaw);
                startS{a}(w) = centerTime;
                
                % Extract window signal
                wSignal = filteredSignal(startIdx:endIdx);
                
                % Binarize: compare each sample to window mean
                windowMean = mean(wSignal);
                binarySeq = double(wSignal > windowMean);
                
                % Calculate Lempel-Ziv complexity with controls
                [lzComplexity{a}(w), lzComplexityNormalized{a}(w), ...
                    lzComplexityNormalizedBernoulli{a}(w)] = ...
                    compute_lz_complexity_with_controls(binarySeq, config.nShuffles);
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
    results.params.slidingWindowSize = config.slidingWindowSize;
    results.params.stepSize = config.stepSize;
    results.params.nShuffles = config.nShuffles;
    
    if strcmp(dataSource, 'spikes')
        results.params.optimalBinSize = config.optimalBinSize;
        results.dataType = dataType;
    elseif strcmp(dataSource, 'lfp')
        results.params.lfpLowpassFreq = config.lfpLowpassFreq;
        results.params.fsLfp = dataStruct.opts.fsLfp;
    end
    
    % Save results
    resultsPath = create_results_path('complexity', dataType, config.slidingWindowSize, ...
        dataStruct.sessionName, config.saveDir, 'dataSource', dataSource);
    save(resultsPath, 'results');
    fprintf('\nSaved results to: %s\n', resultsPath);
    
    % Plotting
    if config.makePlots
        plotConfig = setup_plotting(config.saveDir, 'sessionName', dataStruct.sessionName, ...
            'dataBaseName', dataStruct.dataBaseName);
        plot_complexity_results(results, plotConfig, config, dataStruct);
    end
end

function [lzComplexity, lzNormalized, lzNormalizedBernoulli] = compute_lz_complexity_with_controls(binarySeq, nShuffles)
% COMPUTE_LZ_COMPLEXITY_WITH_CONTROLS Compute LZ complexity with shuffle and Bernoulli controls
    
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
        
        % Normalize by rate-matched Bernoulli control
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
    catch ME
        fprintf('    Warning: Error computing LZ complexity: %s\n', ME.message);
        lzComplexity = nan;
        lzNormalized = nan;
        lzNormalizedBernoulli = nan;
    end
end

function plot_complexity_results(results, plotConfig, config, dataStruct)
% PLOT_COMPLEXITY_RESULTS Create plots for complexity analysis results
    
    % Collect data for axis limits
    allStartS = [];
    allLZNorm = [];
    
    for a = 1:length(results.areas)
        if ~isempty(results.startS{a})
            allStartS = [allStartS, results.startS{a}];
        end
        if ~isempty(results.lzComplexityNormalized{a})
            allLZNorm = [allLZNorm, results.lzComplexityNormalized{a}(~isnan(results.lzComplexityNormalized{a}))];
        end
    end
    
    % Determine axis limits
    if ~isempty(allStartS)
        xMin = min(allStartS);
        xMax = max(allStartS);
    else
        xMin = 0;
        xMax = 100;
    end
    
    if ~isempty(allLZNorm)
        yMinLZNorm = min(allLZNorm);
        yMaxLZNorm = max(allLZNorm);
        yRangeLZNorm = yMaxLZNorm - yMinLZNorm;
        yMinLZNorm = max(0, yMinLZNorm - 0.05 * yRangeLZNorm);
        yMaxLZNorm = yMaxLZNorm + 0.05 * yRangeLZNorm;
    else
        yMinLZNorm = 0;
        yMaxLZNorm = 2;
    end
    
    % Create plot
    figure(914); clf;
    set(gcf, 'Position', plotConfig.targetPos);
    numRows = length(results.areas);
    ha = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
    
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};
    
    for idx = 1:length(results.areas)
        a = idx;
        axes(ha(idx)); hold on;
        
        if ~isempty(results.lzComplexityNormalized{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.lzComplexityNormalized{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.lzComplexityNormalized{a}(validIdx), ...
                    '-', 'Color', areaColors{min(a, length(areaColors))}, 'LineWidth', 2, ...
                    'DisplayName', results.areas{a});
            end
        end
        
        yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Shuffled mean');
        
        % Add reach onsets if applicable
        if strcmp(results.dataSource, 'spikes') && isfield(dataStruct, 'dataType') && ...
                strcmp(dataStruct.dataType, 'reach') && isfield(dataStruct, 'reachStart')
            if ~isempty(results.startS{a}) && ~isempty(dataStruct.reachStart)
                plotTimeRange = [results.startS{a}(1), results.startS{a}(end)];
                reachOnsetsInRange = dataStruct.reachStart(...
                    dataStruct.reachStart >= plotTimeRange(1) & dataStruct.reachStart <= plotTimeRange(2));
                
                if ~isempty(reachOnsetsInRange)
                    for i = 1:length(reachOnsetsInRange)
                        xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
                            'LineStyle', '--', 'Alpha', 0.7);
                    end
                    if isfield(dataStruct, 'startBlock2') && ~isempty(dataStruct.startBlock2)
                        xline(dataStruct.startBlock2, 'Color', [1 0 0], 'LineWidth', 3);
                    end
                end
            end
        end
        
        title(sprintf('%s - Normalized Lempel-Ziv Complexity', results.areas{a}));
        xlabel('Time (s)');
        ylabel('Normalized LZ Complexity');
        if ~isempty(results.startS{a})
            xlim([results.startS{a}(1), results.startS{a}(end)]);
        else
            xlim([xMin, xMax]);
        end
        ylim([yMinLZNorm, yMaxLZNorm]);
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        grid on;
    end
    
    % Create title
    if strcmp(results.dataSource, 'spikes') && isfield(results, 'dataType')
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] %s Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                plotConfig.filePrefix, results.dataType, results.dataSource, ...
                config.slidingWindowSize, config.stepSize, config.nShuffles));
        else
            sgtitle(sprintf('%s Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                results.dataType, results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles));
        end
    else
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                plotConfig.filePrefix, results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles));
        else
            sgtitle(sprintf('Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d', ...
                results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles));
        end
    end
    
    % Save figure
    if ~isempty(plotConfig.filePrefix)
        exportgraphics(gcf, fullfile(config.saveDir, sprintf('%s_complexity_%s_win%.1f.png', ...
            plotConfig.filePrefix, results.dataSource, config.slidingWindowSize)), 'Resolution', 300);
    else
        exportgraphics(gcf, fullfile(config.saveDir, sprintf('complexity_%s_win%.1f.png', ...
            results.dataSource, config.slidingWindowSize)), 'Resolution', 300);
    end
end

