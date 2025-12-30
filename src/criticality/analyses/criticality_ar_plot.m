function criticality_ar_plot(results, plotConfig, config, dataStruct, filenameSuffix)
% criticality_ar_plot - Create plots for criticality AR analysis
%
% Variables:
%   results - Results structure from criticality_ar_analysis()
%   plotConfig - Plotting configuration from setup_plotting()
%   config - Configuration structure
%   dataStruct - Data structure from load_sliding_window_data()
%   filenameSuffix - Filename suffix (e.g., '_pca')
%
% Goal:
%   Create time series plots of d2 and population activity with optional
%   permutation shading and reach onset markers.

    % Extract data from results
    areas = results.areas;
    d2 = results.d2;
    mrBr = results.mrBr;
    startS = results.startS;
    if isfield(results, 'popActivityWindows')
        popActivityWindows = results.popActivityWindows;
    else
        popActivityWindows = cell(1, length(areas));
    end
    
    % Get areas to plot
    if isfield(dataStruct, 'areasToTest')
        areasToTest = dataStruct.areasToTest;
    else
        areasToTest = 1:length(areas);
    end
    
    dataType = results.dataType;
    slidingWindowSize = results.params.slidingWindowSize;
    analyzeD2 = results.params.analyzeD2;
    analyzeMrBr = results.params.analyzeMrBr;
    if isfield(results, 'enablePermutations')
        enablePermutations = results.enablePermutations;
    else
        enablePermutations = false;
    end
    
    % Create figure
    figure(909); clf;
    set(gcf, 'Units', 'pixels');
    set(gcf, 'Position', plotConfig.targetPos);
    
    numRows = length(areasToTest) + 1;  % Add one row for combined d2 plot
    
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
    
    % Define colors for each area
    areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};  % Red, Green, Blue, Magenta for M23, M56, DS, VS
    
    % First row: Plot all d2 traces together
    if useTightSubplot
        axes(ha(1));
    else
        subplot(numRows, 1, 1);
    end
    hold on;
    if analyzeD2
        % Find common time range across all areas
        allStartS = [];
        for a = areasToTest
            if ~isempty(startS{a})
                allStartS = [allStartS, startS{a}];
            end
        end
        if ~isempty(allStartS)
            xLimitsCombined = [min(allStartS), max(allStartS)];
            
            % Collect all d2 values to find maximum
            allD2Values = [];
            for idx = 1:length(areasToTest)
                a = areasToTest(idx);
                if ~isempty(d2{a})
                    allD2Values = [allD2Values; d2{a}(:)];
                end
            end
            
            % Plot d2 for each area with different colors
            for idx = 1:length(areasToTest)
                a = areasToTest(idx);
                if ~isempty(d2{a}) && ~isempty(startS{a})
                    validIdx = ~isnan(d2{a});
                    if any(validIdx)
                        plot(startS{a}(validIdx), d2{a}(validIdx), '-', ...
                            'Color', areaColors{min(a, length(areaColors))}, ...
                            'LineWidth', 2, 'DisplayName', areas{a});
                    end
                end
            end
            
            % Add block 2 marker for reach data
            if strcmp(dataType, 'reach') && isfield(dataStruct, 'startBlock2') && ~isempty(dataStruct.startBlock2)
                xline(dataStruct.startBlock2, 'Color', [.8 0 0], 'LineWidth', 3);
            end
                     
            xlim(xLimitsCombined);
            % Set y-axis from min to max of all d2 signals
            if ~isempty(allD2Values)
                maxD2 = max(allD2Values(~isnan(allD2Values)));
                minD2 = min(allD2Values(~isnan(allD2Values)));
                ylim([minD2 maxD2]);
            end
            ylabel('d2');
            title('All Areas - d2');
            if length(areasToTest) > 1
                legend('Location', 'best');
            end
            grid on;
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
        end
    end
    
    % Subsequent rows: Individual area plots
    for idx = 1:length(areasToTest)
        a = areasToTest(idx);
        if useTightSubplot
            axes(ha(idx + 1));
        else
            subplot(numRows, 1, idx + 1);
        end
        hold on;  % +1 because first row is the combined plot
        
        if analyzeD2
            yyaxis left; 
            % Plot real data first (so permutation line appears on top)
            if ~isempty(d2{a}) && ~isempty(startS{a})
                validIdx = ~isnan(d2{a});
                if any(validIdx)
                    plot(startS{a}(validIdx), d2{a}(validIdx), '-', ...
                        'Color', [0 0 1], 'LineWidth', 2, 'DisplayName', 'Real data');
                end
            end
            ylabel('d2', 'Color', [0 0 1]); 
            ylim([0 0.5]);
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
            
            % Plot permutation mean ± std per window if available
            if enablePermutations && isfield(results, 'd2Permuted') && ...
                    ~isempty(results.d2Permuted{a})
                % Calculate mean and std for each window across shuffles
                permutedMean = nanmean(results.d2Permuted{a}, 2);
                permutedStd = nanstd(results.d2Permuted{a}, 0, 2);
                
                % Find valid indices
                validIdx = ~isnan(permutedMean) & ~isnan(permutedStd) & ...
                    ~isnan(startS{a}(:));
                if any(validIdx)
                    xFill = startS{a}(validIdx);
                    yMean = permutedMean(validIdx);
                    yStd = permutedStd(validIdx);
                    
                    % Ensure row vectors for fill
                    if iscolumn(xFill); xFill = xFill'; end
                    if iscolumn(yMean); yMean = yMean'; end
                    if iscolumn(yStd); yStd = yStd'; end
                    
                    % Plot shaded region (mean ± std)
                    fill([xFill, fliplr(xFill)], ...
                         [yMean + yStd, fliplr(yMean - yStd)], ...
                         [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
                         'DisplayName', 'Permuted mean ± std');
                    
                    % Plot mean line
                    plot(startS{a}(validIdx), permutedMean(validIdx), '-', ...
                        'Color', [0.5 0.5 1], 'LineWidth', 1.5, 'LineStyle', '--', ...
                        'DisplayName', 'Permuted mean');
                end
            end
        end
        
        % Plot population activity on right y-axis
        if isfield(results, 'popActivityWindows') && ...
                ~isempty(popActivityWindows{a}) && ...
                any(~isnan(popActivityWindows{a}))
            yyaxis right;
            validIdx = ~isnan(popActivityWindows{a});
            if any(validIdx)
                plot(startS{a}(validIdx), popActivityWindows{a}(validIdx), '-', ...
                    'Color', [0 0 0], 'LineWidth', 2, 'DisplayName', 'PopActivity Windows');
            end
            ylabel('PopActivity Windows', 'Color', [0 0 0]); 
            ylim('auto');
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
        end
        
        % Add vertical lines at reach onsets (only for reach data)
        if strcmp(dataType, 'reach')
            yyaxis left;
            if ~isempty(startS{a}) && isfield(dataStruct, 'reachStart') && ...
                    ~isempty(dataStruct.reachStart)
                plotTimeRange = [startS{a}(1), startS{a}(end)];
                reachOnsetsInRange = dataStruct.reachStart(...
                    dataStruct.reachStart >= plotTimeRange(1) & ...
                    dataStruct.reachStart <= plotTimeRange(2));
                
                if ~isempty(reachOnsetsInRange)
                    for i = 1:length(reachOnsetsInRange)
                        xline(reachOnsetsInRange(i), 'Color', [0.5 0.5 0.5], ...
                            'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                    end
                    if isfield(dataStruct, 'startBlock2') && ~isempty(dataStruct.startBlock2)
                        xline(dataStruct.startBlock2, 'Color', [1 0 0], 'LineWidth', 3);
                    end
                end
            end
        end
        
        % Add vertical lines at response onsets (only for schall data)
        if strcmp(dataType, 'schall')
            yyaxis left;
            if ~isempty(startS{a}) && isfield(dataStruct, 'responseOnset') && ...
                    ~isempty(dataStruct.responseOnset)
                plotTimeRange = [startS{a}(1), startS{a}(end)];
                responseOnsetsInRange = dataStruct.responseOnset(...
                    dataStruct.responseOnset >= plotTimeRange(1) & ...
                    dataStruct.responseOnset <= plotTimeRange(2));
                
                if ~isempty(responseOnsetsInRange)
                    for i = 1:length(responseOnsetsInRange)
                        xline(responseOnsetsInRange(i), 'Color', [0.5 0.5 0.5], ...
                            'LineWidth', 0.8, 'LineStyle', '--', 'Alpha', 0.7);
                    end
                end
            end
        end
        
        if ~isempty(startS{a})
            xlim([startS{a}(1) startS{a}(end)]);
        end
        title(sprintf('%s - d2 (blue, left) and PopActivity Windows (black, right)', areas{a})); 
        xlabel('Time (s)'); 
        grid on; 
        set(gca, 'XTickLabelMode', 'auto'); 
        % Ensure both left and right y-axes have visible tick labels
        yyaxis left;
        set(gca, 'YTickLabelMode', 'auto');
        set(gca, 'YTickMode', 'auto');
        if isfield(results, 'popActivityWindows') && ...
                ~isempty(popActivityWindows{a}) && ...
                any(~isnan(popActivityWindows{a}))
            yyaxis right;
            set(gca, 'YTickLabelMode', 'auto');
            set(gca, 'YTickMode', 'auto');
        end
    end
    
    % Create super title
    if strcmp(dataType, 'reach')
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] %s d2 (blue, left) and PopActivity Windows (black, right) with reach onsets (gray dashed) - win=%gs', ...
                plotConfig.filePrefix, dataType, slidingWindowSize));
        else
            sgtitle(sprintf('%s d2 (blue, left) and PopActivity Windows (black, right) with reach onsets (gray dashed) - win=%gs', ...
                dataType, slidingWindowSize));
        end
    else
        if ~isempty(plotConfig.filePrefix)
            sgtitle(sprintf('[%s] %s d2 (blue, left) and PopActivity Windows (black, right) - win=%gs', ...
                plotConfig.filePrefix, dataType, slidingWindowSize));
        else
            sgtitle(sprintf('%s d2 (blue, left) and PopActivity Windows (black, right) - win=%gs', ...
                dataType, slidingWindowSize));
        end
    end
    
    % Ensure save directory exists (including any subdirectories)
    if ~exist(config.saveDir, 'dir')
        mkdir(config.saveDir);
    end
    
    % Save figure
    if ~isempty(plotConfig.filePrefix)
        plotPath = fullfile(config.saveDir, ...
            sprintf('%s_criticality_%s_ar%s_win%d.png', ...
            plotConfig.filePrefix, dataType, filenameSuffix, slidingWindowSize));
        exportgraphics(gcf, plotPath, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotPath);
    else
        plotPath = fullfile(config.saveDir, ...
            sprintf('criticality_%s_ar%s_win%d.png', ...
            dataType, filenameSuffix, slidingWindowSize));
        exportgraphics(gcf, plotPath, 'Resolution', 300);
        fprintf('Saved plot to: %s\n', plotPath);
    end
    
    fprintf('Saved criticality AR plot to: %s\n', config.saveDir);
end
