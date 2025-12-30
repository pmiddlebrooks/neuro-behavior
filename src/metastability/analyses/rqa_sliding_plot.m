function rqa_sliding_plot(results, plotConfig, config, dataStruct)
% RQA_SLIDING_PLOT Create plots for RQA analysis results
%
% Variables:
%   results - Results structure from rqa_sliding_analysis()
%   plotConfig - Plotting configuration from setup_plotting()
%   config - Configuration structure
%   dataStruct - Data structure from load_sliding_window_data()
%
% Goal:
%   Create time series plots of RQA metrics with reference lines and recurrence plots.

    % Collect data for axis limits
    allStartS = [];
    allRR = [];
    allDET = [];
    allLAM = [];
    allTT = [];
    
    for a = 1:length(results.areas)
        if ~isempty(results.startS{a})
            allStartS = [allStartS, results.startS{a}];
        end
        if ~isempty(results.recurrenceRateNormalized{a})
            allRR = [allRR, results.recurrenceRateNormalized{a}(~isnan(results.recurrenceRateNormalized{a}))];
        end
        if ~isempty(results.determinismNormalized{a})
            allDET = [allDET, results.determinismNormalized{a}(~isnan(results.determinismNormalized{a}))];
        end
        if ~isempty(results.laminarityNormalized{a})
            allLAM = [allLAM, results.laminarityNormalized{a}(~isnan(results.laminarityNormalized{a}))];
        end
        if ~isempty(results.trappingTimeNormalized{a})
            allTT = [allTT, results.trappingTimeNormalized{a}(~isnan(results.trappingTimeNormalized{a}))];
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
    
    % Y-axis limits for each metric
    if ~isempty(allRR)
        yMinRR = max(0, min(allRR) - 0.05 * (max(allRR) - min(allRR)));
        yMaxRR = max(allRR) + 0.05 * (max(allRR) - min(allRR));
    else
        yMinRR = 0;
        yMaxRR = 2;
    end
    
    if ~isempty(allDET)
        yMinDET = max(0, min(allDET) - 0.05 * (max(allDET) - min(allDET)));
        yMaxDET = max(allDET) + 0.05 * (max(allDET) - min(allDET));
    else
        yMinDET = 0;
        yMaxDET = 2;
    end
    
    if ~isempty(allLAM)
        yMinLAM = max(0, min(allLAM) - 0.05 * (max(allLAM) - min(allLAM)));
        yMaxLAM = max(allLAM) + 0.05 * (max(allLAM) - min(allLAM));
    else
        yMinLAM = 0;
        yMaxLAM = 2;
    end
    
    if ~isempty(allTT)
        yMinTT = max(0, min(allTT) - 0.05 * (max(allTT) - min(allTT)));
        yMaxTT = max(allTT) + 0.05 * (max(allTT) - min(allTT));
    else
        yMinTT = 0;
        yMaxTT = 2;
    end
    
    % Create plot with subplots for each metric
    figure(915); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', plotConfig.targetPos);
    
    % 4 metrics + recurrence plots = 5 rows per area
    numRows = length(results.areas) * 5;
    
    % Use tight_subplot if available
    useTightSubplot = exist('tight_subplot', 'file');
    if useTightSubplot
        ha = tight_subplot(numRows, 1, [0.02 0.03], [0.03 0.06], [0.08 0.04]);
    else
        ha = zeros(numRows, 1);
        for i = 1:numRows
            ha(i) = subplot(numRows, 1, i);
        end
    end
    
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};
    
    plotIdx = 1;
    for idx = 1:length(results.areas)
        a = idx;
        areaColor = areaColors{min(a, length(areaColors))};
        
        % Plot 1: Recurrence Rate
        if useTightSubplot
            axes(ha(plotIdx));
        else
            subplot(numRows, 1, plotIdx);
        end
        hold on;
        
        if ~isempty(results.recurrenceRateNormalized{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.recurrenceRateNormalized{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.recurrenceRateNormalized{a}(validIdx), ...
                    '-', 'Color', areaColor, 'LineWidth', 2, 'DisplayName', sprintf('%s (shuffle norm)', results.areas{a}));
            end
        end
        
        if ~isempty(results.recurrenceRateNormalizedBernoulli{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.recurrenceRateNormalizedBernoulli{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.recurrenceRateNormalizedBernoulli{a}(validIdx), ...
                    '--', 'Color', areaColor, 'LineWidth', 2, 'DisplayName', sprintf('%s (Bernoulli norm)', results.areas{a}));
            end
        end
        
        yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
        title(sprintf('%s - Recurrence Rate (Normalized)', results.areas{a}));
        ylabel('RR (norm)');
        if ~isempty(results.startS{a})
            xlim([results.startS{a}(1), results.startS{a}(end)]);
        end
        ylim([yMinRR, yMaxRR]);
        grid on;
        plotIdx = plotIdx + 1;
        
        % Plot 2: Determinism
        if useTightSubplot
            axes(ha(plotIdx));
        else
            subplot(numRows, 1, plotIdx);
        end
        hold on;
        
        if ~isempty(results.determinismNormalized{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.determinismNormalized{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.determinismNormalized{a}(validIdx), ...
                    '-', 'Color', areaColor, 'LineWidth', 2);
            end
        end
        
        if ~isempty(results.determinismNormalizedBernoulli{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.determinismNormalizedBernoulli{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.determinismNormalizedBernoulli{a}(validIdx), ...
                    '--', 'Color', areaColor, 'LineWidth', 2);
            end
        end
        
        yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
        title(sprintf('%s - Determinism (Normalized)', results.areas{a}));
        ylabel('DET (norm)');
        if ~isempty(results.startS{a})
            xlim([results.startS{a}(1), results.startS{a}(end)]);
        end
        ylim([yMinDET, yMaxDET]);
        grid on;
        plotIdx = plotIdx + 1;
        
        % Plot 3: Laminarity
        if useTightSubplot
            axes(ha(plotIdx));
        else
            subplot(numRows, 1, plotIdx);
        end
        hold on;
        
        if ~isempty(results.laminarityNormalized{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.laminarityNormalized{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.laminarityNormalized{a}(validIdx), ...
                    '-', 'Color', areaColor, 'LineWidth', 2);
            end
        end
        
        if ~isempty(results.laminarityNormalizedBernoulli{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.laminarityNormalizedBernoulli{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.laminarityNormalizedBernoulli{a}(validIdx), ...
                    '--', 'Color', areaColor, 'LineWidth', 2);
            end
        end
        
        yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
        title(sprintf('%s - Laminarity (Normalized)', results.areas{a}));
        ylabel('LAM (norm)');
        if ~isempty(results.startS{a})
            xlim([results.startS{a}(1), results.startS{a}(end)]);
        end
        ylim([yMinLAM, yMaxLAM]);
        grid on;
        plotIdx = plotIdx + 1;
        
        % Plot 4: Trapping Time
        if useTightSubplot
            axes(ha(plotIdx));
        else
            subplot(numRows, 1, plotIdx);
        end
        hold on;
        
        if ~isempty(results.trappingTimeNormalized{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.trappingTimeNormalized{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.trappingTimeNormalized{a}(validIdx), ...
                    '-', 'Color', areaColor, 'LineWidth', 2);
            end
        end
        
        if ~isempty(results.trappingTimeNormalizedBernoulli{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(results.trappingTimeNormalizedBernoulli{a});
            if any(validIdx)
                plot(results.startS{a}(validIdx), results.trappingTimeNormalizedBernoulli{a}(validIdx), ...
                    '--', 'Color', areaColor, 'LineWidth', 2);
            end
        end
        
        yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
        title(sprintf('%s - Trapping Time (Normalized)', results.areas{a}));
        ylabel('TT (norm)');
        xlabel('Time (s)');
        if ~isempty(results.startS{a})
            xlim([results.startS{a}(1), results.startS{a}(end)]);
        end
        ylim([yMinTT, yMaxTT]);
        grid on;
        plotIdx = plotIdx + 1;
        
        % Plot 5: Recurrence Plot (show one example from middle of session)
        if useTightSubplot
            axes(ha(plotIdx));
        else
            subplot(numRows, 1, plotIdx);
        end
        
        % Find a window with a valid recurrence plot (prefer middle of session)
        midWindow = round(length(results.recurrencePlots{a}) / 2);
        validPlotIdx = [];
        for w = 1:length(results.recurrencePlots{a})
            if ~isempty(results.recurrencePlots{a}{w})
                validPlotIdx = [validPlotIdx, w];
            end
        end
        
        if ~isempty(validPlotIdx)
            % Use middle window if available, otherwise first valid
            if any(validPlotIdx == midWindow)
                plotWindow = midWindow;
            else
                plotWindow = validPlotIdx(1);
            end
            
            imagesc(results.recurrencePlots{a}{plotWindow});
            colormap(gca, [1 1 1; 0 0 0]);  % White for 0, black for 1
            axis square;
            title(sprintf('%s - Recurrence Plot (window %d, t=%.1f s)', ...
                results.areas{a}, plotWindow, results.startS{a}(plotWindow)));
            xlabel('Time point');
            ylabel('Time point');
            colorbar;
        else
            text(0.5, 0.5, 'No recurrence plot available', 'HorizontalAlignment', 'center');
            title(sprintf('%s - Recurrence Plot', results.areas{a}));
        end
        plotIdx = plotIdx + 1;
    end
    
    % Create overall title
    if strcmp(results.dataSource, 'spikes') && isfield(results, 'sessionType')
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] %s RQA Analysis - %s, win=%.2fs, step=%.3fs, nShuffles=%d, nPCA=%d\n(Solid: shuffle norm, Dashed: Bernoulli norm)', ...
                plotConfig.filePrefix, results.sessionType, results.dataSource, ...
                config.slidingWindowSize, config.stepSize, config.nShuffles, config.nPCADim));
        else
            sgtitle(sprintf('%s RQA Analysis - %s, win=%.2fs, step=%.3fs, nShuffles=%d, nPCA=%d\n(Solid: shuffle norm, Dashed: Bernoulli norm)', ...
                results.sessionType, results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles, config.nPCADim));
        end
    else
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] RQA Analysis - %s, win=%.2fs, step=%.3fs, nShuffles=%d, nPCA=%d\n(Solid: shuffle norm, Dashed: Bernoulli norm)', ...
                plotConfig.filePrefix, results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles, config.nPCADim));
        else
            sgtitle(sprintf('RQA Analysis - %s, win=%.2fs, step=%.3fs, nShuffles=%d, nPCA=%d\n(Solid: shuffle norm, Dashed: Bernoulli norm)', ...
                results.dataSource, config.slidingWindowSize, config.stepSize, config.nShuffles, config.nPCADim));
        end
    end
    
    % Use the same directory as the results file (session-specific if applicable)
    if isfield(results, 'resultsPath') && ~isempty(results.resultsPath)
        plotSaveDir = fileparts(results.resultsPath);
    else
        plotSaveDir = config.saveDir;
    end
    
    % Build full plot path first
    if ~isempty(plotConfig.filePrefix)
        plotPath = fullfile(plotSaveDir, sprintf('%s_rqa_%s_win%.1f.png', ...
            plotConfig.filePrefix, results.dataSource, config.slidingWindowSize));
    else
        plotPath = fullfile(plotSaveDir, sprintf('rqa_%s_win%.1f.png', ...
            results.dataSource, config.slidingWindowSize));
    end
    
    % Extract directory from plot path and create it (including all parent directories)
    plotDir = fileparts(plotPath);
    if ~isempty(plotDir) && ~exist(plotDir, 'dir')
        [status, msg] = mkdir(plotDir);
        if ~status
            error('Failed to create directory %s: %s', plotDir, msg);
        end
        if ~exist(plotDir, 'dir')
            error('Directory %s still does not exist after mkdir', plotDir);
        end
    end
    
    % Now save the figure
    try
        exportgraphics(gcf, plotPath, 'Resolution', 300);
        fprintf('Saved RQA plot to: %s\n', plotPath);
    catch ME
        error('Failed to save plot to %s: %s\nDirectory exists: %d', plotPath, ME.message, exist(plotDir, 'dir'));
    end
end

