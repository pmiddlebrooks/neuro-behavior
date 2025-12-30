function criticality_lfp_plot(results, plotConfig, config, dataStruct)
% criticality_lfp_plot - Create plots for criticality LFP analysis
%
% Variables:
%   results - Results structure from criticality_lfp_analysis()
%   plotConfig - Plotting configuration from setup_plotting()
%   config - Configuration structure
%   dataStruct - Data structure from load_sliding_window_data()
%
% Goal:
%   Create time series plots of d2 and DFA metrics for LFP data, with options
%   to plot binned envelopes and/or raw LFP data.

    % Extract data from results
    areas = results.areas;
    startS = results.startS;
    d2 = results.d2;
    dfa = results.dfa;
    bands = results.bands;
    bandBinSizes = results.bandBinSizes;
    d2WindowSize = results.d2WindowSize;
    stepSize = results.stepSize;
    
    % Get areas to plot
    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:length(areas);
    end
    
    % Get plotting flags from config
    plotD2 = config.plotD2;
    plotDFA = config.plotDFA;
    plotBinnedEnvelopes = config.plotBinnedEnvelopes;
    plotRawLfp = config.plotRawLfp;
    
    % Check if raw LFP data exists
    hasRawLfp = isfield(results, 'd2Lfp') && isfield(results, 'dfaLfp');
    if hasRawLfp
        d2Lfp = results.d2Lfp;
        dfaLfp = results.dfaLfp;
        lfpBinSize = results.lfpBinSize;
    end
    
    numBands = size(bands, 1);
    
    % Collect data for axis limits
    allStartS = startS(:);
    allD2 = [];
    allDFA = [];
    
    for a = areasToTest
        if plotD2 && plotBinnedEnvelopes
            for b = 1:numBands
                if ~isempty(d2{a}{b})
                    allD2 = [allD2(:); d2{a}{b}(~isnan(d2{a}{b}))'];
                end
            end
        end
        if plotD2 && plotRawLfp && hasRawLfp
            numLfpBins = length(lfpBinSize);
            for lb = 1:numLfpBins
                if ~isempty(d2Lfp{a}{lb})
                    allD2 = [allD2(:); d2Lfp{a}{lb}(~isnan(d2Lfp{a}{lb}))'];
                end
            end
        end
        if plotDFA && plotBinnedEnvelopes
            for b = 1:numBands
                if ~isempty(dfa{a}{b})
                    allDFA = [allDFA(:); dfa{a}{b}(~isnan(dfa{a}{b}))'];
                end
            end
        end
        if plotDFA && plotRawLfp && hasRawLfp
            numLfpBins = length(lfpBinSize);
            for lb = 1:numLfpBins
                if ~isempty(dfaLfp{a}{lb})
                    allDFA = [allDFA(:); dfaLfp{a}{lb}(~isnan(dfaLfp{a}{lb}))'];
                end
            end
        end
    end
    
    % Determine axis limits
    xMin = min(allStartS);
    xMax = max(allStartS);
    
    if ~isempty(allD2)
        yMinD2 = min(allD2);
        yMaxD2 = max(allD2);
        yRangeD2 = yMaxD2 - yMinD2;
        yMinD2 = yMinD2 - 0.05 * yRangeD2;
        yMaxD2 = yMaxD2 + 0.05 * yRangeD2;
    else
        yMinD2 = 0;
        yMaxD2 = 0.5;
    end
    
    if ~isempty(allDFA)
        yMinDFA = min(allDFA);
        yMaxDFA = max(allDFA);
        yRangeDFA = yMaxDFA - yMinDFA;
        yMinDFA = max(0.3, yMinDFA - 0.05 * yRangeDFA);
        yMaxDFA = min(1.7, yMaxDFA + 0.05 * yRangeDFA);
    else
        yMinDFA = 0.3;
        yMaxDFA = 1.7;
    end
    
    % ========== d2 Plot ==========
    if plotD2
        figure(910); clf;
        set(gcf, 'Units', 'pixels');
        set(gcf, 'Position', plotConfig.targetPos);
        numRows = length(areasToTest);
        
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
        
        bandColors = lines(numBands);
        
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            if useTightSubplot
                axes(ha(idx));
            else
                subplot(numRows, 1, idx);
            end
            hold on;
            
            % Plot raw LFP d2
            if plotRawLfp && hasRawLfp
                numLfpBins = length(lfpBinSize);
                grayColors = repmat(linspace(0.8, 0, numLfpBins)', 1, 3);
                for lb = 1:numLfpBins
                    if ~isempty(d2Lfp{a}{lb})
                        validIdx = ~isnan(d2Lfp{a}{lb});
                        if any(validIdx)
                            plot(startS(validIdx), d2Lfp{a}{lb}(validIdx), '-', ...
                                'Color', grayColors(lb, :), 'LineWidth', 2, ...
                                'DisplayName', sprintf('Raw LFP (%.3fs bin)', lfpBinSize(lb)));
                        end
                    end
                end
            end
            
            % Plot binned envelope d2
            if plotBinnedEnvelopes
                for b = 1:numBands
                    if ~isempty(d2{a}{b})
                        validIdx = ~isnan(d2{a}{b});
                        if any(validIdx)
                            plot(startS(validIdx), d2{a}{b}(validIdx), '-', ...
                                'Color', bandColors(b, :), 'LineWidth', 2, ...
                                'DisplayName', sprintf('%s (%.3fs bin)', bands{b, 1}, bandBinSizes(b)));
                        end
                    end
                end
            end
            
            title(sprintf('%s - d2 Analysis', areas{a}));
            xlabel('Time (s)');
            ylabel('d2');
            xlim([xMin, xMax]);
            ylim([yMinD2, yMaxD2]);
            set(gca, 'XTickLabelMode', 'auto');
            set(gca, 'YTickLabelMode', 'auto');
            if idx == 1
                legend('Location', 'best');
            end
            grid on;
        end
        
        % Create super title
        if isfield(dataStruct, 'dataType') && strcmp(dataStruct.dataType, 'reach')
            if ~isempty(plotConfig.filePrefix)
                sgtitle(sprintf('[%s] LFP d2 Analysis - win=%.2fs, step=%.3fs', ...
                    plotConfig.filePrefix, d2WindowSize, stepSize));
            else
                sgtitle(sprintf('LFP d2 Analysis - win=%.2fs, step=%.3fs', d2WindowSize, stepSize));
            end
        else
            if ~isempty(plotConfig.filePrefix)
                sgtitle(sprintf('[%s] LFP d2 Analysis - win=%.2fs, step=%.3fs', ...
                    plotConfig.filePrefix, d2WindowSize, stepSize));
            else
                sgtitle(sprintf('LFP d2 Analysis - win=%.2fs, step=%.3fs', d2WindowSize, stepSize));
            end
        end
        
        % Ensure save directory exists (including any subdirectories)
        if ~exist(config.saveDir, 'dir')
            mkdir(config.saveDir);
        end
        
        % Save figure
        if ~isempty(plotConfig.filePrefix)
            plotPath = fullfile(config.saveDir, ...
                sprintf('%s_criticality_lfp_d2_win%.1f.png', ...
                plotConfig.filePrefix, config.slidingWindowSize));
            exportgraphics(gcf, plotPath, 'Resolution', 300);
            fprintf('Saved plot to: %s\n', plotPath);
        else
            plotPath = fullfile(config.saveDir, ...
                sprintf('criticality_lfp_d2_win%.1f.png', config.slidingWindowSize));
            exportgraphics(gcf, plotPath, 'Resolution', 300);
            fprintf('Saved plot to: %s\n', plotPath);
        end
    end
    
    % ========== DFA Plot ==========
    if plotDFA
        figure(911); clf;
        set(gcf, 'Units', 'pixels');
        set(gcf, 'Position', plotConfig.targetPos);
        numRows = length(areasToTest);
        
        % Use tight_subplot if available, otherwise use subplot
        useTightSubplot = exist('tight_subplot', 'file');
        if useTightSubplot
            haDFA = tight_subplot(numRows, 1, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
        else
            haDFA = zeros(numRows, 1);
            for i = 1:numRows
                haDFA(i) = subplot(numRows, 1, i);
            end
        end
        
        bandColors = lines(numBands);
        
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            if useTightSubplot
                axes(haDFA(idx));
            else
                subplot(numRows, 1, idx);
            end
            hold on;
            
            % Plot raw LFP DFA
            if plotRawLfp && hasRawLfp
                numLfpBins = length(lfpBinSize);
                grayColors = repmat(linspace(0.8, 0, numLfpBins)', 1, 3);
                for lb = 1:numLfpBins
                    if ~isempty(dfaLfp{a}{lb})
                        validIdx = ~isnan(dfaLfp{a}{lb});
                        if any(validIdx)
                            plot(startS(validIdx), dfaLfp{a}{lb}(validIdx), '-', ...
                                'Color', grayColors(lb, :), 'LineWidth', 1.5, ...
                                'DisplayName', sprintf('DFA Raw LFP (%.3fs bin)', lfpBinSize(lb)));
                        end
                    end
                end
            end
            
            % Plot binned envelope DFA
            if plotBinnedEnvelopes
                for b = 1:numBands
                    if ~isempty(dfa{a}{b})
                        validIdx = ~isnan(dfa{a}{b});
                        if any(validIdx)
                            plot(startS(validIdx), dfa{a}{b}(validIdx), '-', ...
                                'Color', bandColors(b, :), 'LineWidth', 1.5, ...
                                'DisplayName', sprintf('DFA %s', bands{b, 1}));
                        end
                    end
                end
            end
            
            title(sprintf('%s - DFA Analysis', areas{a}));
            xlabel('Time (s)');
            ylabel('DFA \\alpha');
            xlim([xMin, xMax]);
            ylim([yMinDFA, yMaxDFA]);
            set(gca, 'XTickLabelMode', 'auto');
            set(gca, 'YTickLabelMode', 'auto');
            if idx == 1
                legend('Location', 'best');
            end
            grid on;
        end
        
        % Create super title
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] LFP DFA Analysis - step=%.3fs', plotConfig.filePrefix, stepSize));
        else
            sgtitle(sprintf('LFP DFA Analysis - step=%.3fs', stepSize));
        end
        
        % Ensure save directory exists (including any subdirectories)
        if ~exist(config.saveDir, 'dir')
            mkdir(config.saveDir);
        end
        
        % Save figure
        if ~isempty(plotConfig.filePrefix)
            plotPath = fullfile(config.saveDir, ...
                sprintf('%s_criticality_lfp_dfa_win%.1f.png', ...
                plotConfig.filePrefix, config.slidingWindowSize));
            exportgraphics(gcf, plotPath, 'Resolution', 300);
            fprintf('Saved plot to: %s\n', plotPath);
        else
            plotPath = fullfile(config.saveDir, ...
                sprintf('criticality_lfp_dfa_win%.1f.png', config.slidingWindowSize));
            exportgraphics(gcf, plotPath, 'Resolution', 300);
            fprintf('Saved plot to: %s\n', plotPath);
        end
    end
    
    fprintf('Saved criticality LFP plots to: %s\n', config.saveDir);
end
