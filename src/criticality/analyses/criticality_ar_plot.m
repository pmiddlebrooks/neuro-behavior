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

% Add path to utility functions
utilsPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils');
if exist(utilsPath, 'dir')
    addpath(utilsPath);
end

% Extract data from results
areas = results.areas;
d2 = results.d2;  % Raw d2 values
mrBr = results.mrBr;
startS = results.startS;
if isfield(results, 'popActivityWindows')
    popActivityWindows = results.popActivityWindows;
else
    popActivityWindows = cell(1, length(areas));
end

% Check if normalization is enabled and use normalized values for plotting
normalizeD2 = false;
if isfield(results.params, 'normalizeD2')
    normalizeD2 = results.params.normalizeD2;
end

% Use normalized d2 values if normalization is enabled
if normalizeD2 && isfield(results, 'd2Normalized')
    d2ToPlot = results.d2Normalized;
    d2Label = 'd2 (normalized)';
else
    d2ToPlot = d2;
    d2Label = 'd2';
end

% Get areas to plot
if isfield(dataStruct, 'areasToTest')
    areasToTest = dataStruct.areasToTest;
else
    areasToTest = 1:length(areas);
end

sessionType = results.sessionType;
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


% Calculate windowed mean activity for each area (for right y-axis)
summedActivityWindowed = cell(1, length(areas));
if strcmp(sessionType, 'reach') && isfield(dataStruct, 'dataMat') && isfield(dataStruct, 'idMatIdx')
    % Use binSize and slidingWindowSize (area-specific vectors)
    if isfield(results, 'binSize')
        binSize = results.binSize;
    elseif isfield(results.params, 'binSize')
        binSize = results.params.binSize;
    else
        error('binSize not found in results');
    end
    if isfield(results, 'slidingWindowSize')
        slidingWindowSize = results.slidingWindowSize;
    elseif isfield(results.params, 'slidingWindowSize')
        slidingWindowSize = results.params.slidingWindowSize;
    else
        error('slidingWindowSize not found in results');
    end
    for a = 1:length(areas)
        aID = dataStruct.idMatIdx{a};
        if ~isempty(aID) && ~isempty(startS{a}) && ~isnan(binSize(a))
            % Bin data using the area-specific binSize
            aDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), binSize(a));
            % Sum across neurons
            summedActivity = sum(aDataMat, 2);
            % Calculate time bins (center of each bin)
            numBins = size(aDataMat, 1);
            activityTimeBins = ((0:numBins-1) + 0.5) * binSize(a);
            
            % Calculate windowed mean activity for each window center
            numWindows = length(startS{a});
            summedActivityWindowed{a} = nan(1, numWindows);
            for w = 1:numWindows
                centerTime = startS{a}(w);
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
    for a = 1:length(areas)
        summedActivityWindowed{a} = [];
    end
end

% First row: Plot all d2 traces together
if useTightSubplot
    axes(ha(1));
else
    subplot(numRows, 1, 1);
end
hold on;

% Find first non-empty startS for plotting reference
firstNonEmptyArea = [];
for a = areasToTest
    if ~isempty(startS{a})
        firstNonEmptyArea = a;
        break;
    end
end
if isempty(firstNonEmptyArea)
    % No areas have data, cannot plot
    warning('No areas with valid startS data found. Skipping plot.');
    return;
end

% Add event markers first (so they appear behind the data)
add_event_markers(dataStruct, startS, 'firstNonEmptyArea', firstNonEmptyArea);

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
            if ~isempty(d2ToPlot{a})
                allD2Values = [allD2Values; d2ToPlot{a}(:)];
            end
        end


        % % Plot summed neural activity first (behind metrics) on right y-axis
        % if strcmp(sessionType, 'reach') && ~isempty(summedActivityWindowed)
        %     yyaxis right;
        %     for idx = 1:length(areasToTest)
        %         a = areasToTest(idx);
        %         if ~isempty(summedActivityWindowed{a}) && ~isempty(startS{a})
        %             validIdx = ~isnan(summedActivityWindowed{a});
        %             if any(validIdx)
        %                 areaColor = areaColors{min(a, length(areaColors))};
        %                 plot(startS{a}(validIdx), summedActivityWindowed{a}(validIdx), '-', ...
        %                     'Color', areaColor, 'LineWidth', 2, ...
        %                     'HandleVisibility', 'off');
        %             end
        %         end
        %     end
        %     ylabel('Summed Activity', 'Color', [0.5 0.5 0.5]);
        %     set(gca, 'YTickLabelMode', 'auto');
        %     yyaxis left;
        % end

        % Plot d2 metrics on top
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            if ~isempty(d2ToPlot{a}) && ~isempty(startS{a})
                validIdx = ~isnan(d2ToPlot{a});
                if any(validIdx)
                    plot(startS{a}(validIdx), d2ToPlot{a}(validIdx), '-', ...
                        'Color', areaColors{min(a, length(areaColors))}, ...
                        'LineWidth', 3, 'DisplayName', areas{a});
                end
            end
        end

        % Add reference line at 1.0 for normalized values
        if normalizeD2
            yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Shuffled mean');
        end

        
        xlim(xLimitsCombined);
        % Set y-axis from min to max of all d2 signals
        if ~isempty(allD2Values)
            maxD2 = max(allD2Values(~isnan(allD2Values)));
            minD2 = min(allD2Values(~isnan(allD2Values)));
            if normalizeD2
                % Ensure reference line at 1.0 is visible
                yRange = maxD2 - minD2;
                if yRange == 0
                    yRange = 0.1;
                end
                yMin = min(1.0, minD2) - 0.05 * yRange;
                yMax = max(1.0, maxD2) + 0.05 * yRange;
                yMin = max(0, yMin);
                ylim([yMin yMax]);
            else
                ylim([minD2 maxD2]);
            end
        end
        ylabel(d2Label);
        if normalizeD2
            title('All Areas - d2 (Normalized)');
        else
            title('All Areas - d2');
        end
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

    % Add event markers first (so they appear behind the data)
    % Map idx to actual area index
    actualAreaIdx = areasToTest(idx);
    add_event_markers(dataStruct, startS, 'areaIdx', actualAreaIdx);

    % Plot popActivityWindows on right y-axis (color-coded by area)
    if isfield(results, 'popActivityWindows') && ~isempty(popActivityWindows{a}) && ...
            ~isempty(startS{a}) && any(~isnan(popActivityWindows{a}))
        yyaxis right;
        validIdx = ~isnan(popActivityWindows{a});
        if any(validIdx)
            areaColor = areaColors{min(a, length(areaColors))};
            plot(startS{a}(validIdx), popActivityWindows{a}(validIdx), '-', ...
                'Color', [.7 .7 .7], 'LineWidth', 2, ...
                'HandleVisibility', 'off');
        end
        ylabel('Pop Activity', 'Color', [0.5 0.5 0.5]);
        ylim('auto');
        set(gca, 'YTickLabelMode', 'auto');
        set(gca, 'YTickMode', 'auto');
        yyaxis left;
    end

    if analyzeD2
        yyaxis left;
        % Plot real data first (so permutation line appears on top)
        if ~isempty(d2ToPlot{a}) && ~isempty(startS{a})
            validIdx = ~isnan(d2ToPlot{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                plot(startS{a}(validIdx), d2ToPlot{a}(validIdx), '-', ...
                    'Color', areaColor, 'LineWidth', 3, 'DisplayName', 'Real data');
            end
        end
        areaColor = areaColors{min(a, length(areaColors))};
        ylabel(d2Label, 'Color', areaColor);

        % Set y-axis limits based on whether normalized
        if normalizeD2
            % For normalized, ensure reference line at 1.0 is visible
            if ~isempty(d2ToPlot{a})
                validD2 = d2ToPlot{a}(~isnan(d2ToPlot{a}));
                if ~isempty(validD2)
                    yRange = max(validD2) - min(validD2);
                    if yRange == 0
                        yRange = 0.1;
                    end
                    yMin = min(1.0, min(validD2)) - 0.05 * yRange;
                    yMax = max(1.0, max(validD2)) + 0.05 * yRange;
                    yMin = max(0, yMin);
                    ylim([yMin yMax]);
                else
                    ylim([0 2]);
                end
            else
                ylim([0 2]);
            end
            % Add reference line at 1.0
            yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Shuffled mean');
        else
            ylim([0 0.5]);
        end
        set(gca, 'YTickLabelMode', 'auto');
        set(gca, 'YTickMode', 'auto');

        % Plot permutation mean ± std per window if available (only for raw d2)
        if ~normalizeD2 && enablePermutations && isfield(results, 'd2Permuted') && ...
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

    if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)]);
    end
    if normalizeD2
        title(sprintf('%s - d2 (normalized, left) and PopActivity Windows (right)', areas{a}));
    else
        title(sprintf('%s - d2 (left) and PopActivity Windows (right)', areas{a}));
    end
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
        yyaxis left;
    end
end

% Create super title
normText = '';
if normalizeD2
    normText = ' (normalized)';
end
if strcmp(sessionType, 'reach')
    if ~isempty(plotConfig.filePrefix)
        sgtitle(sprintf('[%s] %s d2%s (left) and PopActivity Windows (right) with reach onsets (gray dashed) - win=%gs', ...
            plotConfig.filePrefix, sessionType, normText, slidingWindowSize));
    else
        sgtitle(sprintf('%s d2%s (left) and PopActivity Windows (right) with reach onsets (gray dashed) - win=%gs', ...
            sessionType, normText, slidingWindowSize));
    end
else
    if ~isempty(plotConfig.filePrefix)
        sgtitle(sprintf('[%s] %s d2%s (left) and PopActivity Windows (right) - win=%gs', ...
            plotConfig.filePrefix, sessionType, normText, slidingWindowSize));
    else
        sgtitle(sprintf('%s d2%s (left) and PopActivity Windows (right) - win=%gs', ...
            sessionType, normText, slidingWindowSize));
    end
end

% Ensure save directory exists (including any subdirectories)
if ~exist(config.saveDir, 'dir')
    mkdir(config.saveDir);
end

% Save figure
if ~isempty(plotConfig.filePrefix)
    plotPath = fullfile(config.saveDir, ...
        sprintf('%s_criticality_%s_ar%s.png', ...
        plotConfig.filePrefix, sessionType, filenameSuffix));
else
    plotPath = fullfile(config.saveDir, ...
        sprintf('criticality_%s_ar%s.png', ...
        sessionType, filenameSuffix));
end

drawnow
    exportgraphics(gcf, plotPath, 'Resolution', 300);
    fprintf('Saved plot to: %s\n', plotPath);

fprintf('Saved criticality AR plot to: %s\n', config.saveDir);



end