function lzc_sliding_plot(results, plotConfig, config, dataStruct)
% lzc_sliding_plot - Create plots for Lempel-Ziv complexity analysis results
%
% Variables:
%   results - Results structure from lzc_sliding_analysis()
%   plotConfig - Plotting configuration from setup_plotting()
%   config - Configuration structure
%   dataStruct - Data structure from load_sliding_window_data()
%
% Goal:
%   Create time series plots of normalized Lempel-Ziv complexity with
%   reference line at shuffled mean (1.0) and optional reach onset markers.

    % Ensure dataStruct has all areas from results (including M2356 if it exists)
    % This is important when loading results in loadAndPlot mode
    if length(results.areas) > length(dataStruct.areas)
        % Check if M2356 exists in results but not in dataStruct
        m2356IdxResults = find(strcmp(results.areas, 'M2356'));
        if ~isempty(m2356IdxResults)
            idxM23 = find(strcmp(dataStruct.areas, 'M23'));
            idxM56 = find(strcmp(dataStruct.areas, 'M56'));
            if ~isempty(idxM23) && ~isempty(idxM56) && ~any(strcmp(dataStruct.areas, 'M2356'))
                % Add M2356 to dataStruct to match results
                dataStruct.areas{end+1} = 'M2356';
                dataStruct.idMatIdx{end+1} = [dataStruct.idMatIdx{idxM23}(:); dataStruct.idMatIdx{idxM56}(:)];
                if isfield(dataStruct, 'idLabel') && ~isempty(dataStruct.idLabel{idxM23}) && ~isempty(dataStruct.idLabel{idxM56})
                    dataStruct.idLabel{end+1} = [dataStruct.idLabel{idxM23}(:); dataStruct.idLabel{idxM56}(:)];
                end
                fprintf('Added M2356 to dataStruct for plotting (to match results.areas)\n');
            end
        end
    end
    
    % Collect data for axis limits and calculate metric means
    allStartS = [];
    allLZNorm = [];
    
    % Calculate mean of normalized LZ complexity per area (for centering behavior proportion)
    metricMeans = cell(1, length(results.areas));
    for a = 1:length(results.areas)
        if ~isempty(results.startS{a})
            allStartS = [allStartS, results.startS{a}];
        end
        if ~isempty(results.lzComplexityNormalized{a})
            allLZNorm = [allLZNorm, results.lzComplexityNormalized{a}(~isnan(results.lzComplexityNormalized{a}))];
        end
        % Also include Bernoulli normalized values for axis limits
        if ~isempty(results.lzComplexityNormalizedBernoulli{a})
            allLZNorm = [allLZNorm, results.lzComplexityNormalizedBernoulli{a}(~isnan(results.lzComplexityNormalizedBernoulli{a}))];
        end
        
        % Calculate mean of normalized LZ complexity for this area
        allMetricsForArea = [];
        if ~isempty(results.lzComplexityNormalized{a})
            allMetricsForArea = [allMetricsForArea, results.lzComplexityNormalized{a}(~isnan(results.lzComplexityNormalized{a}))];
        end
        if ~isempty(results.lzComplexityNormalizedBernoulli{a})
            allMetricsForArea = [allMetricsForArea, results.lzComplexityNormalizedBernoulli{a}(~isnan(results.lzComplexityNormalizedBernoulli{a}))];
        end
        if ~isempty(allMetricsForArea)
            metricMeans{a} = mean(allMetricsForArea);
        else
            metricMeans{a} = nan;
        end
    end
    
    % Extract behavior proportion if available and center on metric means
    if isfield(results, 'behaviorProportion') && strcmp(results.sessionType, 'spontaneous')
        behaviorProportion = results.behaviorProportion;
        % Center behavior proportion on the mean of normalized LZ complexity
        behaviorProportionCentered = cell(1, length(results.areas));
        for a = 1:length(results.areas)
            if ~isempty(behaviorProportion{a}) && ~isnan(metricMeans{a})
                validBhvVals = behaviorProportion{a}(~isnan(behaviorProportion{a}));
                if ~isempty(validBhvVals)
                    meanBhvVal = mean(validBhvVals);
                    % Center: subtract behavior mean, add metric mean
                    behaviorProportionCentered{a} = behaviorProportion{a} - meanBhvVal + metricMeans{a};
                else
                    behaviorProportionCentered{a} = [];
                end
            else
                behaviorProportionCentered{a} = [];
            end
        end
        plotBehaviorProportion = true;
    else
        behaviorProportionCentered = cell(1, length(results.areas));
        for a = 1:length(results.areas)
            behaviorProportionCentered{a} = [];
        end
        plotBehaviorProportion = false;
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
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', plotConfig.targetPos);
    numRows = max(length(results.areas), 4);
    
    % Use tight_subplot if available, otherwise use subplot
    useTightSubplot = exist('tight_subplot', 'file');
    if useTightSubplot
        ha = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
    else
        ha = zeros(numRows, 1);
        for i = 1:numRows
            ha(i) = subplot(numRows, 1, i);
        end
    end
    
    % Colors: M23 (pink), M56 (green), DS (blue), VS (magenta), M2356 (orange)
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1], [1 0.5 0]};
    
    % Calculate windowed mean activity for each area (for right y-axis)
    summedActivityWindowed = cell(1, length(results.areas));
    if strcmp(results.dataSource, 'spikes') && isfield(dataStruct, 'spikeTimes') && isfield(dataStruct, 'spikeClusters') && isfield(dataStruct, 'idMatIdx')
        % Add path to data_prep for bin_spikes function
        dataPrepPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data_prep');
        if exist(dataPrepPath, 'dir')
            addpath(dataPrepPath);
        end
        
        % Calculate time range from spike data
        if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
            timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
        else
            timeRange = [0, max(dataStruct.spikeTimes)];
        end
        
        % Use binSize and slidingWindowSize (area-specific vectors)
        if isfield(results.params, 'binSize')
            binSize = results.params.binSize;
        else
            error('binSize not found in results.params');
        end
        if isfield(results.params, 'slidingWindowSize')
            slidingWindowSize = results.params.slidingWindowSize;
        else
            error('slidingWindowSize not found in results.params');
        end
        for a = 1:length(results.areas)
            % Check if this area index exists in dataStruct
            if a > length(dataStruct.idMatIdx) || isempty(dataStruct.idMatIdx{a})
                summedActivityWindowed{a} = [];
                continue;
            end
            aID = dataStruct.idMatIdx{a};
            if ~isempty(aID) && ~isempty(results.startS{a}) && ~isnan(binSize(a))
                % Get neuron IDs for this area
                if isfield(dataStruct, 'idLabel') && a <= length(dataStruct.idLabel) && ~isempty(dataStruct.idLabel{a})
                    neuronIDs = dataStruct.idLabel{a};
                else
                    % Fallback: use idMatIdx if idLabel not available
                    neuronIDs = dataStruct.idMatIdx{a};
                end
                
                % Bin data using bin_spikes (on-demand binning)
                aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
                    neuronIDs, timeRange, binSize(a));
                
                % Sum across neurons
                summedActivity = sum(aDataMat, 2);
                % Calculate time bins (center of each bin)
                numBins = size(aDataMat, 1);
                activityTimeBins = ((0:numBins-1) + 0.5) * binSize(a);
                
                % Calculate windowed mean activity for each window center
                numWindows = length(results.startS{a});
                summedActivityWindowed{a} = nan(1, numWindows);
                for w = 1:numWindows
                    centerTime = results.startS{a}(w);
                    % Use area-specific window size
                    winStart = centerTime - slidingWindowSize(a) / 2;
                    winEnd = centerTime + slidingWindowSize(a) / 2;
                    % Find bins within this window
                    binMask = activityTimeBins >= winStart & activityTimeBins < winEnd;
                    if any(binMask)
                        summedActivityWindowed{a}(w) = mean(summedActivity(binMask));
                    end
                end
            else
                summedActivityWindowed{a} = [];
            end
        end
    else
        for a = 1:length(results.areas)
            summedActivityWindowed{a} = [];
        end
    end
    
    % Find first non-empty area for event markers
    firstNonEmptyArea = find(~cellfun(@isempty, results.startS), 1);
    if isempty(firstNonEmptyArea)
        firstNonEmptyArea = [];
    end
    
    % Add path to utility functions
    utilsPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils');
    if exist(utilsPath, 'dir')
        addpath(utilsPath);
    end
    
    for idx = 1:length(results.areas)
        a = idx;
        if useTightSubplot
            axes(ha(idx));
        else
            subplot(numRows, 1, idx);
        end
        hold on;
        
        % Add event markers first (so they appear behind the data)
        add_event_markers(dataStruct, results.startS, ...
            'firstNonEmptyArea', firstNonEmptyArea, ...
            'areaIdx', a, ...
            'dataSource', results.dataSource, ...
            'numAreas', length(results.areas));
        
        % Plot summed neural activity first (behind metrics) on right y-axis
        areaColor = areaColors{min(a, length(areaColors))};
        if strcmp(results.dataSource, 'spikes') && ~isempty(summedActivityWindowed{a}) && ~isempty(results.startS{a})
            yyaxis right;
            validIdx = ~isnan(summedActivityWindowed{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), summedActivityWindowed{a}(validIdx), '-', ...
                    'Color', [0.2 0.2 0.2], 'LineWidth', 2, ...
                    'HandleVisibility', 'off');
            end
            ylabel('Summed Activity', 'Color', [0.5 0.5 0.5]);
            set(gca, 'YTickLabelMode', 'auto');
            yyaxis left;
        end
        
        yline(1);

        % Plot metrics on top
        if ~isempty(results.lzComplexityNormalized{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.lzComplexityNormalized{a});
            if any(validIdx)
                % Plot shuffle normalized LZ complexity (solid line)
                plot(results.startS{a}(validIdx), results.lzComplexityNormalized{a}(validIdx), ...
                    '-', 'Color', areaColor, 'LineWidth', 3, ...
                    'DisplayName', sprintf('%s (shuffle norm)', results.areas{a}));
            end
        end
        
        % Plot Bernoulli normalized LZ complexity (dashed line, same color)
        if ~isempty(results.lzComplexityNormalizedBernoulli{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.lzComplexityNormalizedBernoulli{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.lzComplexityNormalizedBernoulli{a}(validIdx), ...
                    '--', 'Color', areaColor, 'LineWidth', 3, ...
                    'DisplayName', sprintf('%s (Bernoulli norm)', results.areas{a}));
            end
        end
        
        % Plot behavior proportion (centered on metric mean) if available
        if plotBehaviorProportion && ~isempty(behaviorProportion{a}) && ~isempty(results.startS{a})
            % Use metric mean for this area
            if ~isnan(metricMeans{a})
                validBhvVals = behaviorProportion{a}(~isnan(behaviorProportion{a}));
                if ~isempty(validBhvVals)
                    meanBhvVal = mean(validBhvVals);
                    behaviorProportionCentered = .2*(behaviorProportion{a} - meanBhvVal) + metricMeans{a};
                    validIdx = ~isnan(behaviorProportionCentered);
                    if any(validIdx)
                        plot(results.startS{a}(validIdx), behaviorProportionCentered(validIdx), ':', ...
                            'Color', [0 0 0], 'LineWidth', 2, ...
                            'DisplayName', sprintf('%s (bhv prop)', results.areas{a}));
                    end
                end
            end
        end
                
        
        % Get neuron count for spike data
        if strcmp(results.dataSource, 'spikes') && isfield(dataStruct, 'idMatIdx') && ...
                a <= length(dataStruct.idMatIdx) && ~isempty(dataStruct.idMatIdx{a})
            nNeurons = length(dataStruct.idMatIdx{a});
            title(sprintf('%s - Normalized Lempel-Ziv Complexity (n=%d)', results.areas{a}, nNeurons));
        else
            title(sprintf('%s - Normalized Lempel-Ziv Complexity', results.areas{a}));
        end
        if idx == length(results.areas)
        xlabel('Time (s)');
        end
        ylabel('Normalized LZ Complexity');
        if ~isempty(results.startS{a})
            xlim([results.startS{a}(1), results.startS{a}(end)]);
        else
            xlim([xMin, xMax]);
        end
        ylim([max(.8, yMinLZNorm), min(1.2, yMaxLZNorm)]);
        % ylim([.85, 1.05]);
        set(gca, 'XTickLabelMode', 'auto');
        set(gca, 'YTickLabelMode', 'auto');
        grid on;
    end
    
    % Check if Bernoulli control was computed
    useBernoulliControl = true;  % Default
    if isfield(results.params, 'useBernoulliControl')
        useBernoulliControl = results.params.useBernoulliControl;
    end
    
    % Create title with conditional Bernoulli mention
    if useBernoulliControl
        normText = '(Solid: shuffle norm, Dashed: Bernoulli norm)';
    else
        normText = '(Shuffle normalized)';
    end
    
            sgtitle(sprintf('[%s] %s Normalized Lempel-Ziv Complexity - %s %s, nShuffles=%d', ...
                plotConfig.filePrefix, results.sessionType, results.dataSource, results.dataSource, ...
                config.nShuffles), 'interpreter', 'none');
       
    % Use the same directory as the results file (session-specific if applicable)
    % Extract directory from resultsPath if available, otherwise use config.saveDir
    if isfield(results, 'resultsPath') && ~isempty(results.resultsPath)
        plotSaveDir = fileparts(results.resultsPath);
    else
        plotSaveDir = config.saveDir;
    end
    
    % Build full plot path first
    % Only add dataSource if it's 'lfp' (not 'spikes')
    if strcmp(results.dataSource, 'lfp')
        dataSourceStr = ['_', results.dataSource];
    else
        dataSourceStr = '';
    end
    if ~isempty(plotConfig.filePrefix)
        plotPath = fullfile(plotSaveDir, sprintf('%s_lzc_sliding_window%s.png', ...
            plotConfig.filePrefix, dataSourceStr));
    else
        plotPath = fullfile(plotSaveDir, sprintf('lzc_sliding_window%s.png', ...
            dataSourceStr));
    end
    
    % Extract directory from plot path and create it (including all parent directories)
    plotDir = fileparts(plotPath);
    if ~isempty(plotDir) && ~exist(plotDir, 'dir')
        % mkdir creates all parent directories automatically
        [status, msg] = mkdir(plotDir);
        if ~status
            error('Failed to create directory %s: %s', plotDir, msg);
        end
        % Double-check it was created
        if ~exist(plotDir, 'dir')
            error('Directory %s still does not exist after mkdir', plotDir);
        end
    end
    
    % Now save the figure
    try
                    drawnow
        exportgraphics(gcf, plotPath, 'Resolution', 300);
        fprintf('Saved LZC plot to: %s\n', plotPath);
    catch ME
        error('Failed to save plot to %s: %s\nDirectory exists: %d', plotPath, ME.message, exist(plotDir, 'dir'));
    end
end
