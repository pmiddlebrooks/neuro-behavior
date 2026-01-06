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
%
% Note on Normalization:
%   All metrics (RR, DET, LAM, TT) are normalized by dividing by the mean of shuffled
%   controls. This allows comparison against null models:
%   - Recurrence Rate (RR): Since we fix target RR (e.g., 2%), raw RR is approximately
%     constant. Normalization shows deviations from shuffled control.
%   - Determinism (DET): Proportion of recurrence points in diagonal lines. Normalized to
%     compare against shuffled (which has no structure).
%   - Laminarity (LAM): Proportion of recurrence points in vertical lines. Normalized to
%     compare against shuffled.
%   - Trapping Time (TT): Average length of vertical lines. Normalized to compare against
%     shuffled.
%   All normalized metrics have a reference line at 1.0 (shuffled mean). Values > 1.0
%   indicate more structure than shuffled controls.

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

% Extract behavior proportion if available (will be centered per subplot)
if isfield(results, 'behaviorProportion') && strcmp(results.sessionType, 'naturalistic')
    behaviorProportion = results.behaviorProportion;
    plotBehaviorProportion = true;
else
    behaviorProportion = cell(1, length(results.areas));
    for a = 1:length(results.areas)
        behaviorProportion{a} = [];
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

% Y-axis limits for each metric
% Ensure reference line at 1.0 is always visible
referenceLine = 1.0;

if ~isempty(allRR)
    yRange = max(allRR) - min(allRR);
    if yRange == 0
        yRange = 0.1;  % Default range if all values are the same
    end
    yMinRR = min(referenceLine, min(allRR)) - 0.05 * yRange;
    yMaxRR = max(referenceLine, max(allRR)) + 0.05 * yRange;
    yMinRR = max(0, yMinRR);  % Don't go below 0
else
    yMinRR = 0;
    yMaxRR = 2;
end

if ~isempty(allDET)
    yRange = max(allDET) - min(allDET);
    if yRange == 0
        yRange = 0.1;
    end
    yMinDET = min(referenceLine, min(allDET)) - 0.05 * yRange;
    yMaxDET = max(referenceLine, max(allDET)) + 0.05 * yRange;
    yMinDET = max(0, yMinDET);
else
    yMinDET = 0;
    yMaxDET = 2;
end

if ~isempty(allLAM)
    yRange = max(allLAM) - min(allLAM);
    if yRange == 0
        yRange = 0.1;
    end
    yMinLAM = min(referenceLine, min(allLAM)) - 0.05 * yRange;
    yMaxLAM = max(referenceLine, max(allLAM)) + 0.05 * yRange;
    yMinLAM = max(0, yMinLAM);
else
    yMinLAM = 0;
    yMaxLAM = 2;
end

if ~isempty(allTT)
    yRange = max(allTT) - min(allTT);
    if yRange == 0
        yRange = 0.1;
    end
    yMinTT = min(referenceLine, min(allTT)) - 0.05 * yRange;
    yMaxTT = max(referenceLine, max(allTT)) + 0.05 * yRange;
    yMinTT = max(0, yMinTT);
else
    yMinTT = 0;
    yMaxTT = 2;
end

% Create plot with subplots for each metric
figure(918); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', plotConfig.targetPos);

% 4x1 grid (one subplot per metric, showing all areas)
numAreas = length(results.areas);
numRows = 4;
numCols = 1;

% Use tight_subplot if available
useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(numRows, numCols, [0.035 0.04], [0.03 0.08], [0.08 0.04]);
else
    ha = zeros(numRows, 1);
    for i = 1:numRows
        ha(i) = subplot(numRows, numCols, i);
    end
end

areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1]};

% Calculate summed neural activity for each area (for right y-axis)
% Calculate windowed mean activity for smoothing
summedActivityWindowed = cell(1, numAreas);
if strcmp(results.dataSource, 'spikes') && isfield(dataStruct, 'dataMat') && isfield(dataStruct, 'idMatIdx')
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
    for a = 1:numAreas
        aID = dataStruct.idMatIdx{a};
        if ~isempty(aID) && ~isempty(results.startS{a}) && ~isnan(binSize(a))
            % Bin data using the area-specific binSize
            aDataMat = neural_matrix_ms_to_frames(dataStruct.dataMat(:, aID), binSize(a));
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
    for a = 1:numAreas
        summedActivityWindowed{a} = [];
    end
end

% Find first non-empty startS for plotting reference
firstNonEmptyArea = find(~cellfun(@isempty, results.startS), 1);
if isempty(firstNonEmptyArea)
    % No areas have data, cannot plot
    warning('No areas with valid startS data found. Skipping plot.');
    return;
end

% Add path to utility functions
utilsPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'sliding_window_prep', 'utils');
if exist(utilsPath, 'dir')
    addpath(utilsPath);
end

% Plot 1: Recurrence Rate (row 1) - all areas
if useTightSubplot
    axes(ha(1));
else
    subplot(numRows, numCols, 1);
end
hold on;

% Plot summed neural activity first (behind metrics) on right y-axis
if strcmp(results.dataSource, 'spikes') && ~isempty(summedActivityWindowed)
    yyaxis right;
    for a = 1:numAreas
        if ~isempty(summedActivityWindowed{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(summedActivityWindowed{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                plot(results.startS{a}(validIdx), summedActivityWindowed{a}(validIdx), '--', ...
                    'Color', areaColor, 'LineWidth', 2, ...
                    'HandleVisibility', 'off');
            end
        end
    end
    ylabel('Summed Activity', 'Color', [0.5 0.5 0.5]);
    set(gca, 'YTickLabelMode', 'auto');
    yyaxis left;
end

% Plot metrics on top
for a = 1:numAreas
    areaColor = areaColors{min(a, length(areaColors))};

    if ~isempty(results.recurrenceRateNormalized{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.recurrenceRateNormalized{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.recurrenceRateNormalized{a}(validIdx), ...
                '-', 'Color', areaColor, 'LineWidth', 3, 'DisplayName', sprintf('%s (shuffle norm)', results.areas{a}));
        end
    end

    if ~isempty(results.recurrenceRateNormalizedBernoulli{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.recurrenceRateNormalizedBernoulli{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.recurrenceRateNormalizedBernoulli{a}(validIdx), ...
                '--', 'Color', areaColor, 'LineWidth', 3, 'DisplayName', sprintf('%s (Bernoulli norm)', results.areas{a}));
        end
    end
    
    % Plot behavior proportion (centered on metric mean) if available
    if plotBehaviorProportion && ~isempty(behaviorProportion{a}) && ~isempty(results.startS{a})
        % Calculate mean of RR for this subplot (across all areas)
        rrMeanForSubplot = mean(allRR);
        validBhvVals = behaviorProportion{a}(~isnan(behaviorProportion{a}));
        if ~isempty(validBhvVals) && ~isnan(rrMeanForSubplot)
            meanBhvVal = mean(validBhvVals);
            behaviorProportionCentered = behaviorProportion{a} - meanBhvVal + rrMeanForSubplot;
            validIdx = ~isnan(behaviorProportionCentered);
            if any(validIdx)
                plot(results.startS{a}(validIdx), behaviorProportionCentered(validIdx), ':', ...
                    'Color', areaColor, 'LineWidth', 2, ...
                    'DisplayName', sprintf('%s (bhv prop)', results.areas{a}));
            end
        end
    end
end

yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Shuffled mean');
if plotBehaviorProportion && ~isempty(allRR)
    rrMeanForSubplot = mean(allRR);
    yline(rrMeanForSubplot, 'k:', 'LineWidth', 1, 'Alpha', 0.3, 'HandleVisibility', 'off');
end

title('Recurrence Rate (Normalized)');
ylabel('RR (norm)');
xlabel('Time (s)');
if numAreas > 0 && ~isempty(firstNonEmptyArea) && ~isempty(results.startS{firstNonEmptyArea})
    xlim([results.startS{firstNonEmptyArea}(1), results.startS{firstNonEmptyArea}(end)]);
end
ylim([yMinRR, yMaxRR]);
set(gca, 'YTickLabelMode', 'auto');
set(gca, 'XTickLabelMode', 'auto');
grid on;
legend('Location', 'best');

% Plot 2: Determinism (row 2) - all areas
if useTightSubplot
    axes(ha(2));
else
    subplot(numRows, numCols, 2);
end
hold on;

% Add event markers first (so they appear behind the data)
add_event_markers(dataStruct, results.startS, ...
    'firstNonEmptyArea', firstNonEmptyArea, ...
    'dataSource', results.dataSource, ...
    'numAreas', numAreas);

% Plot summed neural activity first (behind metrics) on right y-axis
if strcmp(results.dataSource, 'spikes') && ~isempty(summedActivityWindowed)
    yyaxis right;
    for a = 1:numAreas
        if ~isempty(summedActivityWindowed{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(summedActivityWindowed{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                plot(results.startS{a}(validIdx), summedActivityWindowed{a}(validIdx), '--', ...
                    'Color', areaColor, 'LineWidth', 2, ...
                    'HandleVisibility', 'off');
            end
        end
    end
    ylabel('Summed Activity', 'Color', [0.5 0.5 0.5]);
    set(gca, 'YTickLabelMode', 'auto');
    yyaxis left;
end

% Plot metrics on top
for a = 1:numAreas
    areaColor = areaColors{min(a, length(areaColors))};

    if ~isempty(results.determinismNormalized{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.determinismNormalized{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.determinismNormalized{a}(validIdx), ...
                '-', 'Color', areaColor, 'LineWidth', 3);
        end
    end

    if ~isempty(results.determinismNormalizedBernoulli{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.determinismNormalizedBernoulli{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.determinismNormalizedBernoulli{a}(validIdx), ...
                '--', 'Color', areaColor, 'LineWidth', 3);
        end
    end
    
    % Plot behavior proportion (centered on metric mean) if available
    if plotBehaviorProportion && ~isempty(behaviorProportion{a}) && ~isempty(results.startS{a})
        % Calculate mean of DET for this subplot (across all areas)
        detMeanForSubplot = mean(allDET);
        validBhvVals = behaviorProportion{a}(~isnan(behaviorProportion{a}));
        if ~isempty(validBhvVals) && ~isnan(detMeanForSubplot)
            meanBhvVal = mean(validBhvVals);
            behaviorProportionCentered = behaviorProportion{a} - meanBhvVal + detMeanForSubplot;
            validIdx = ~isnan(behaviorProportionCentered);
            if any(validIdx)
                plot(results.startS{a}(validIdx), behaviorProportionCentered(validIdx), ':', ...
                    'Color', areaColor, 'LineWidth', 2, 'HandleVisibility', 'off');
            end
        end
    end
end

yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
if plotBehaviorProportion && ~isempty(allDET)
    detMeanForSubplot = mean(allDET);
    yline(detMeanForSubplot, 'k:', 'LineWidth', 1, 'Alpha', 0.3, 'HandleVisibility', 'off');
end

title('Determinism (Normalized)');
ylabel('DET (norm)');
xlabel('Time (s)');
if numAreas > 0 && ~isempty(firstNonEmptyArea) && ~isempty(results.startS{firstNonEmptyArea})
    xlim([results.startS{firstNonEmptyArea}(1), results.startS{firstNonEmptyArea}(end)]);
end
ylim([yMinDET, yMaxDET]);
set(gca, 'YTickLabelMode', 'auto');
set(gca, 'XTickLabelMode', 'auto');
grid on;

% Plot 3: Laminarity (row 3) - all areas
if useTightSubplot
    axes(ha(3));
else
    subplot(numRows, numCols, 3);
end
hold on;

% Add event markers first (so they appear behind the data)
add_event_markers(dataStruct, results.startS, ...
    'firstNonEmptyArea', firstNonEmptyArea, ...
    'dataSource', results.dataSource, ...
    'numAreas', numAreas);

% Plot summed neural activity first (behind metrics) on right y-axis
if strcmp(results.dataSource, 'spikes') && ~isempty(summedActivityWindowed)
    yyaxis right;
    for a = 1:numAreas
        if ~isempty(summedActivityWindowed{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(summedActivityWindowed{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                plot(results.startS{a}(validIdx), summedActivityWindowed{a}(validIdx), '--', ...
                    'Color', areaColor, 'LineWidth', 2, ...
                    'HandleVisibility', 'off');
            end
        end
    end
    ylabel('Summed Activity', 'Color', [0.5 0.5 0.5]);
    set(gca, 'YTickLabelMode', 'auto');
    yyaxis left;
end

% Plot metrics on top
for a = 1:numAreas
    areaColor = areaColors{min(a, length(areaColors))};

    if ~isempty(results.laminarityNormalized{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.laminarityNormalized{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.laminarityNormalized{a}(validIdx), ...
                '-', 'Color', areaColor, 'LineWidth', 3);
        end
    end

    if ~isempty(results.laminarityNormalizedBernoulli{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.laminarityNormalizedBernoulli{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.laminarityNormalizedBernoulli{a}(validIdx), ...
                '--', 'Color', areaColor, 'LineWidth', 3);
        end
    end
    
    % Plot behavior proportion (centered on metric mean) if available
    if plotBehaviorProportion && ~isempty(behaviorProportion{a}) && ~isempty(results.startS{a})
        % Calculate mean of LAM for this subplot (across all areas)
        lamMeanForSubplot = mean(allLAM);
        validBhvVals = behaviorProportion{a}(~isnan(behaviorProportion{a}));
        if ~isempty(validBhvVals) && ~isnan(lamMeanForSubplot)
            meanBhvVal = mean(validBhvVals);
            behaviorProportionCentered = behaviorProportion{a} - meanBhvVal + lamMeanForSubplot;
            validIdx = ~isnan(behaviorProportionCentered);
            if any(validIdx)
                plot(results.startS{a}(validIdx), behaviorProportionCentered(validIdx), ':', ...
                    'Color', areaColor, 'LineWidth', 2, 'HandleVisibility', 'off');
            end
        end
    end
end

yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
if plotBehaviorProportion && ~isempty(allLAM)
    lamMeanForSubplot = mean(allLAM);
    yline(lamMeanForSubplot, 'k:', 'LineWidth', 1, 'Alpha', 0.3, 'HandleVisibility', 'off');
end

title('Laminarity (Normalized)');
ylabel('LAM (norm)');
xlabel('Time (s)');
if numAreas > 0 && ~isempty(firstNonEmptyArea) && ~isempty(results.startS{firstNonEmptyArea})
    xlim([results.startS{firstNonEmptyArea}(1), results.startS{firstNonEmptyArea}(end)]);
end
ylim([yMinLAM, yMaxLAM]);
set(gca, 'YTickLabelMode', 'auto');
set(gca, 'XTickLabelMode', 'auto');
grid on;

% Plot 4: Trapping Time (row 4) - all areas
if useTightSubplot
    axes(ha(4));
else
    subplot(numRows, numCols, 4);
end
hold on;

% Add event markers first (so they appear behind the data)
add_event_markers(dataStruct, results.startS, ...
    'firstNonEmptyArea', firstNonEmptyArea, ...
    'dataSource', results.dataSource, ...
    'numAreas', numAreas);

% Plot summed neural activity first (behind metrics) on right y-axis
if strcmp(results.dataSource, 'spikes') && ~isempty(summedActivityWindowed)
    yyaxis right;
    for a = 1:numAreas
        if ~isempty(summedActivityWindowed{a}) && ~isempty(results.startS{a})
            validIdx = ~isnan(summedActivityWindowed{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};
                plot(results.startS{a}(validIdx), summedActivityWindowed{a}(validIdx), '--', ...
                    'Color', areaColor, 'LineWidth', 2, ...
                    'HandleVisibility', 'off');
            end
        end
    end
    ylabel('Summed Activity', 'Color', [0.5 0.5 0.5]);
    set(gca, 'YTickLabelMode', 'auto');
    yyaxis left;
end

% Plot metrics on top
for a = 1:numAreas
    areaColor = areaColors{min(a, length(areaColors))};

    if ~isempty(results.trappingTimeNormalized{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.trappingTimeNormalized{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.trappingTimeNormalized{a}(validIdx), ...
                '-', 'Color', areaColor, 'LineWidth', 3);
        end
    end

    if ~isempty(results.trappingTimeNormalizedBernoulli{a}) && ~isempty(results.startS{a})
        validIdx = ~isnan(results.trappingTimeNormalizedBernoulli{a});
        if any(validIdx)
            plot(results.startS{a}(validIdx), results.trappingTimeNormalizedBernoulli{a}(validIdx), ...
                '--', 'Color', areaColor, 'LineWidth', 3);
        end
    end
    
    % Plot behavior proportion (centered on metric mean) if available
    if plotBehaviorProportion && ~isempty(behaviorProportion{a}) && ~isempty(results.startS{a})
        % Calculate mean of TT for this subplot (across all areas)
        ttMeanForSubplot = mean(allTT);
        validBhvVals = behaviorProportion{a}(~isnan(behaviorProportion{a}));
        if ~isempty(validBhvVals) && ~isnan(ttMeanForSubplot)
            meanBhvVal = mean(validBhvVals);
            behaviorProportionCentered = behaviorProportion{a} - meanBhvVal + ttMeanForSubplot;
            validIdx = ~isnan(behaviorProportionCentered);
            if any(validIdx)
                plot(results.startS{a}(validIdx), behaviorProportionCentered(validIdx), ':', ...
                    'Color', areaColor, 'LineWidth', 2, 'HandleVisibility', 'off');
            end
        end
    end
end

yline(1.0, 'k--', 'LineWidth', 1, 'Alpha', 0.5);
if plotBehaviorProportion && ~isempty(allTT)
    ttMeanForSubplot = mean(allTT);
    yline(ttMeanForSubplot, 'k:', 'LineWidth', 1, 'Alpha', 0.3, 'HandleVisibility', 'off');
end

title('Trapping Time (Normalized)');
ylabel('TT (norm)');
xlabel('Time (s)');
if numAreas > 0 && ~isempty(results.startS{firstNonEmptyArea})
    xlim([results.startS{firstNonEmptyArea}(1), results.startS{firstNonEmptyArea}(end)]);
end
ylim([yMinTT, yMaxTT]);
set(gca, 'YTickLabelMode', 'auto');
set(gca, 'XTickLabelMode', 'auto');
grid on;

% Check if Bernoulli control was computed
useBernoulliControl = true;  % Default
if isfield(results.params, 'useBernoulliControl')
    useBernoulliControl = results.params.useBernoulliControl;
end

% Create overall title with conditional Bernoulli mention
if useBernoulliControl
    normText = '(Solid: shuffle norm, Dashed: Bernoulli norm)';
else
    normText = '(Shuffle normalized)';
end

% Add drift indicator to title if using per-window PCA
driftText = '';
if isfield(results.params, 'usePerWindowPCA') && results.params.usePerWindowPCA
    driftText = ' [per-window PCA]';
end

if strcmp(results.dataSource, 'spikes') && isfield(results, 'sessionType')

        sgtitle(sprintf('[%s] %s RQA Analysis - %s, nShuffles=%d, nPCA=%d%s\n%s', ...
            plotConfig.filePrefix, results.sessionType, results.dataSource, config.nShuffles, config.nPCADim, driftText, normText), 'interpreter', 'none');
else
        sgtitle(sprintf('[%s] RQA Analysis - %s, nShuffles=%d, nPCA=%d%s\n%s', ...
            plotConfig.filePrefix, results.dataSource, config.nShuffles, config.nPCADim, driftText, normText), 'interpreter', 'none');
end

% Use the same directory as the results file (session-specific if applicable)
% Extract directory from resultsPath if available, otherwise use config.saveDir
if isfield(results, 'resultsPath') && ~isempty(results.resultsPath)
    plotSaveDir = fileparts(results.resultsPath);
else
    plotSaveDir = config.saveDir;
end

% Build full plot path first (append _drift if using per-window PCA)
% Always include PCA dimensions in RQA plot filenames
% Only add dataSource if it's 'lfp' (not 'spikes')
driftSuffix = '';
if isfield(results.params, 'usePerWindowPCA') && results.params.usePerWindowPCA
    driftSuffix = '_drift';
end
pcaSuffix = sprintf('_pca%d', config.nPCADim);
if strcmp(results.dataSource, 'lfp')
    dataSourceStr = ['_', results.dataSource];
else
    dataSourceStr = '';
end
if ~isempty(plotConfig.filePrefix)
    plotPath = fullfile(plotSaveDir, sprintf('%s_rqa%s%s%s.png', ...
        plotConfig.filePrefix, dataSourceStr, pcaSuffix, driftSuffix));
else
    plotPath = fullfile(plotSaveDir, sprintf('rqa%s%s%s.png', ...
        dataSourceStr, pcaSuffix, driftSuffix));
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
    fprintf('Saved RQA plot to: %s\n', plotPath);
catch ME
    error('Failed to save plot to %s: %s\nDirectory exists: %d', plotPath, ME.message, exist(plotDir, 'dir'));
end

% Create separate figure for recurrence plots (all areas together) - only if available
saveRecurrencePlots = false;
if isfield(results.params, 'saveRecurrencePlots')
    saveRecurrencePlots = results.params.saveRecurrencePlots;
end

if saveRecurrencePlots && isfield(results, 'recurrencePlots')
    % Check if any recurrence plots exist
    hasRecurrencePlots = false;
    for a = 1:length(results.areas)
        if ~isempty(results.recurrencePlots{a})
            hasRecurrencePlots = true;
            break;
        end
    end

    if hasRecurrencePlots
        figure(916); clf;
        set(gcf, 'Units', 'pixels');
        set(gcf, 'Position', plotConfig.targetPos);

        numAreas = length(results.areas);

        % Use tight_subplot if available - 1 row, 4 columns
        useTightSubplot = exist('tight_subplot', 'file');
        if useTightSubplot
            ha = tight_subplot(1, 4, [0.08 0.04], [0.1 0.1], [0.08 0.04]);
        else
            ha = zeros(1, 4);
            for i = 1:4
                ha(i) = subplot(1, 4, i);
            end
        end

        for a = 1:min(numAreas, 4)  % Show up to 4 areas
            if useTightSubplot
                axes(ha(a));
            else
                subplot(1, 4, a);
            end

            % Find a window with a valid recurrence plot (prefer middle of session)
            if ~isempty(results.recurrencePlots{a})
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

                    % Get neuron count for spike data and time info
                    timeStr = '';
                    if ~isempty(results.startS{a}) && plotWindow <= length(results.startS{a}) && ...
                            ~isnan(results.startS{a}(plotWindow))
                        timeStr = sprintf(', t=%.1f s', results.startS{a}(plotWindow));
                    end

                    if strcmp(results.dataSource, 'spikes') && isfield(dataStruct, 'idMatIdx') && ...
                            ~isempty(dataStruct.idMatIdx{a})
                        nNeurons = length(dataStruct.idMatIdx{a});
                        title(sprintf('%s - Recurrence Plot (window %d%s, n=%d)', ...
                            results.areas{a}, plotWindow, timeStr, nNeurons));
                    else
                        title(sprintf('%s - Recurrence Plot (window %d%s)', ...
                            results.areas{a}, plotWindow, timeStr));
                    end
                    xlabel('Time point');
                    ylabel('Time point');
                    colorbar;
                else
                    text(0.5, 0.5, 'No recurrence plot available', 'HorizontalAlignment', 'center');
                    title(sprintf('%s - Recurrence Plot', results.areas{a}));
                end
            else
                text(0.5, 0.5, 'No recurrence plot available', 'HorizontalAlignment', 'center');
                title(sprintf('%s - Recurrence Plot', results.areas{a}));
            end
        end

        % Create title for recurrence plots figure
        if strcmp(results.dataSource, 'spikes') && isfield(results, 'sessionType')
            if ~isempty(plotConfig.filePrefix)
                sgtitle(sprintf('[%s] %s RQA Recurrence Plots - %s, win=%.2fs, step=%.3fs, nPCA=%d', ...
                    plotConfig.filePrefix, results.sessionType, results.dataSource, ...
                    config.slidingWindowSize, config.stepSize, config.nPCADim));
            else
                sgtitle(sprintf('%s RQA Recurrence Plots - %s, win=%.2fs, step=%.3fs, nPCA=%d', ...
                    results.sessionType, results.dataSource, config.slidingWindowSize, config.stepSize, config.nPCADim));
            end
        else
            if ~isempty(plotConfig.filePrefix)
                sgtitle(sprintf('[%s] RQA Recurrence Plots - %s, win=%.2fs, step=%.3fs, nPCA=%d', ...
                    plotConfig.filePrefix, results.dataSource, config.slidingWindowSize, config.stepSize, config.nPCADim));
            else
                sgtitle(sprintf('RQA Recurrence Plots - %s, win=%.2fs, step=%.3fs, nPCA=%d', ...
                    results.dataSource, config.slidingWindowSize, config.stepSize, config.nPCADim));
            end
        end

        % Save recurrence plots figure (append _drift if using per-window PCA)
        % Always include PCA dimensions, only add dataSource if it's 'lfp'
        driftSuffix = '';
        if isfield(results.params, 'usePerWindowPCA') && results.params.usePerWindowPCA
            driftSuffix = '_drift';
        end
        if strcmp(results.dataSource, 'lfp')
            dataSourceStr = ['_', results.dataSource];
        else
            dataSourceStr = '';
        end
        if ~isempty(plotConfig.filePrefix)
            recurrencePlotPath = fullfile(plotSaveDir, sprintf('%s_rqa_recurrence_plots%s_pca%d%s.png', ...
                plotConfig.filePrefix, dataSourceStr, config.nPCADim, driftSuffix));
        else
            recurrencePlotPath = fullfile(plotSaveDir, sprintf('rqa_recurrence_plots%s_pca%d%s.png', ...
                dataSourceStr, config.nPCADim, driftSuffix));
        end

        % Extract directory from plot path and create it if needed
        recurrencePlotDir = fileparts(recurrencePlotPath);
        if ~isempty(recurrencePlotDir) && ~exist(recurrencePlotDir, 'dir')
            [status, msg] = mkdir(recurrencePlotDir);
            if ~status
                error('Failed to create directory %s: %s', recurrencePlotDir, msg);
            end
            if ~exist(recurrencePlotDir, 'dir')
                error('Directory %s still does not exist after mkdir', recurrencePlotDir);
            end
        end

        try
drawnow
exportgraphics(gcf, recurrencePlotPath, 'Resolution', 300);
            fprintf('Saved RQA recurrence plots to: %s\n', recurrencePlotPath);
        catch ME
            error('Failed to save recurrence plots to %s: %s\nDirectory exists: %d', recurrencePlotPath, ME.message, exist(recurrencePlotDir, 'dir'));
        end
    end
end
end

