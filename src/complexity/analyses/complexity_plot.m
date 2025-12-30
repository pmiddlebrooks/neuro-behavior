function complexity_plot(results, plotConfig, config, dataStruct)
% complexity_plot - Create plots for complexity analysis results
%
% Variables:
%   results - Results structure from complexity_analysis()
%   plotConfig - Plotting configuration from setup_plotting()
%   config - Configuration structure
%   dataStruct - Data structure from load_sliding_window_data()
%
% Goal:
%   Create time series plots of normalized Lempel-Ziv complexity with
%   reference line at shuffled mean (1.0) and optional reach onset markers.

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
        % Also include Bernoulli normalized values for axis limits
        if ~isempty(results.lzComplexityNormalizedBernoulli{a})
            allLZNorm = [allLZNorm, results.lzComplexityNormalizedBernoulli{a}(~isnan(results.lzComplexityNormalizedBernoulli{a}))];
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
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', plotConfig.targetPos);
    numRows = length(results.areas);
    
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
    
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};
    
    for idx = 1:length(results.areas)
        a = idx;
        if useTightSubplot
            axes(ha(idx));
        else
            subplot(numRows, 1, idx);
        end
        hold on;
        
        if ~isempty(results.lzComplexityNormalized{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.lzComplexityNormalized{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                % Plot shuffle normalized LZ complexity (solid line)
                plot(results.startS{a}(validIdx), results.lzComplexityNormalized{a}(validIdx), ...
                    '-', 'Color', areaColor, 'LineWidth', 2, ...
                    'DisplayName', sprintf('%s (shuffle norm)', results.areas{a}));
            end
        end
        
        % Plot Bernoulli normalized LZ complexity (dashed line, same color)
        if ~isempty(results.lzComplexityNormalizedBernoulli{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.lzComplexityNormalizedBernoulli{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                plot(results.startS{a}(validIdx), results.lzComplexityNormalizedBernoulli{a}(validIdx), ...
                    '--', 'Color', areaColor, 'LineWidth', 2, ...
                    'DisplayName', sprintf('%s (Bernoulli norm)', results.areas{a}));
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
        % ylim([yMinLZNorm, yMaxLZNorm]);
        ylim([.85, 1.05]);
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
    
    if strcmp(results.dataSource, 'spikes') && isfield(results, 'dataType')
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] %s Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d\n%s', ...
                plotConfig.filePrefix, results.dataType, results.dataSource, ...
                config.slidingWindowSize, config.stepSize, config.nShuffles, normText));
        else
            sgtitle(sprintf('%s Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d\n%s', ...
                results.dataType, results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles, normText));
        end
    else
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d\n%s', ...
                plotConfig.filePrefix, results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles, normText));
        else
            sgtitle(sprintf('Normalized Lempel-Ziv Complexity - %s, win=%.2fs, step=%.3fs, nShuffles=%d\n%s', ...
                results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles, normText));
        end
    end
    
    % Use the same directory as the results file (session-specific if applicable)
    % Extract directory from resultsPath if available, otherwise use config.saveDir
    if isfield(results, 'resultsPath') && ~isempty(results.resultsPath)
        plotSaveDir = fileparts(results.resultsPath);
    else
        plotSaveDir = config.saveDir;
    end
    
    % Build full plot path first
    if ~isempty(plotConfig.filePrefix)
        plotPath = fullfile(plotSaveDir, sprintf('%s_complexity_%s_win%.1f.png', ...
            plotConfig.filePrefix, results.dataSource, config.slidingWindowSize));
    else
        plotPath = fullfile(plotSaveDir, sprintf('complexity_%s_win%.1f.png', ...
            results.dataSource, config.slidingWindowSize));
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
        exportgraphics(gcf, plotPath, 'Resolution', 300);
        fprintf('Saved complexity plot to: %s\n', plotPath);
    catch ME
        error('Failed to save plot to %s: %s\nDirectory exists: %d', plotPath, ME.message, exist(plotDir, 'dir'));
    end
end
