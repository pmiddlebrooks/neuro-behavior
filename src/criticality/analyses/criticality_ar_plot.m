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

% Light-plot options to avoid renderer crashes and heavy export (many windows = many vertices)
plotResolution = 300;
if isfield(config, 'plotResolution') && ~isempty(config.plotResolution)
    plotResolution = config.plotResolution;
end
% No downsampling: keep all vertices (Inf). Set config.maxPlotPoints to e.g. 500 to re-enable light plot.
maxPlotPoints = Inf;
if isfield(config, 'maxPlotPoints') && ~isempty(config.maxPlotPoints)
    maxPlotPoints = config.maxPlotPoints;
end
useSoftwareRenderer = true;
if isfield(config, 'useSoftwareRenderer') && ~isempty(config.useSoftwareRenderer)
    useSoftwareRenderer = config.useSoftwareRenderer;
end
saveEps = true;
if isfield(config, 'saveEps') && ~isempty(config.saveEps)
    saveEps = config.saveEps;
end
% Downsample long series to this many points for all fill/plot (fewer vertices = less crash-prone)
downsample_series = @(x, varargin) downsample_plot_series(x, maxPlotPoints, varargin{:});

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

% Detect subsampling usage (for optional error ribbons)
useSubsampling = false;
if isfield(results, 'useSubsampling')
    useSubsampling = results.useSubsampling;
elseif isfield(results.params, 'useSubsampling')
    useSubsampling = results.params.useSubsampling;
end

% Per-window subsample matrices (raw and normalized), used for error ribbons
d2SubsamplesAll = {};
d2NormalizedSubsamples = {};
if useSubsampling
    if isfield(results, 'd2Subsamples')
        d2SubsamplesAll = results.d2Subsamples;
    end
    if normalizeD2 && isfield(results, 'd2NormalizedSubsamples')
        d2NormalizedSubsamples = results.d2NormalizedSubsamples;
    end
end

% Use normalized d2 values if normalization is enabled
if normalizeD2 && isfield(results, 'd2Normalized')
    d2ToPlot = results.d2Normalized;
    d2Label = 'd2 (normalized)';
else
    d2ToPlot = d2;
    d2Label = 'd2';
end

% Pre-compute mean metric per area (for centering behavior proportion)
metricMeans = cell(1, length(areas));
for a = 1:length(areas)
    if ~isempty(d2ToPlot{a})
        validVals = d2ToPlot{a}(~isnan(d2ToPlot{a}));
        if ~isempty(validVals)
            metricMeans{a} = mean(validVals);
        else
            metricMeans{a} = nan;
        end
    else
        metricMeans{a} = nan;
    end
end

% Extract behavior proportion if available (spontaneous sessions)
if isfield(results, 'behaviorProportion') && strcmp(results.sessionType, 'spontaneous')
    behaviorProportion = results.behaviorProportion;
    plotBehaviorProportion = true;
else
    behaviorProportion = cell(1, length(areas));
    for a = 1:length(areas)
        behaviorProportion{a} = [];
    end
    plotBehaviorProportion = false;
end

% Get areas to plot
areasToTest = 1:length(areas);

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
% Use software renderer to avoid OpenGL crashes
if useSoftwareRenderer
    set(gcf, 'Renderer', 'zbuffer');
end

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
% Colors: M23 (pink), M56 (green), DS (blue), VS (magenta), M2356 (orange)
areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1], [1 0.5 0]};  % Red, Green, Blue, Magenta, Orange

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
if ~strcmp(dataStruct.sessionType, 'schall')
add_event_markers(dataStruct, startS, 'firstNonEmptyArea', firstNonEmptyArea);
end

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


        % Plot d2 metrics
        for idx = 1:length(areasToTest)
            a = areasToTest(idx);
            if ~isempty(d2ToPlot{a}) && ~isempty(startS{a})
                validIdx = ~isnan(d2ToPlot{a});
                if any(validIdx)
                    [xLine, yLine] = downsample_series(startS{a}(validIdx), d2ToPlot{a}(validIdx));
                    plot(xLine, yLine, '-', ...
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
            [xLine, yLine] = downsample_series(startS{a}(validIdx), popActivityWindows{a}(validIdx));
            plot(xLine, yLine, '-', ...
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
        % Plot real data first (so error ribbons and permutation lines appear correctly)
        if ~isempty(d2ToPlot{a}) && ~isempty(startS{a})
            validIdx = ~isnan(d2ToPlot{a});
            if any(validIdx)
                areaColor = areaColors{min(a, length(areaColors))};

                % If using subsampling, draw error ribbons (std across subsamples)
                if useSubsampling
                    subMat = [];
                    if normalizeD2 && ~isempty(d2NormalizedSubsamples) && ...
                            a <= numel(d2NormalizedSubsamples) && ~isempty(d2NormalizedSubsamples{a})
                        % Use normalized subsample matrix
                        subMat = d2NormalizedSubsamples{a};  % [numWindows x nSubsamples]
                    elseif ~normalizeD2 && ~isempty(d2SubsamplesAll) && ...
                            a <= numel(d2SubsamplesAll) && ~isempty(d2SubsamplesAll{a})
                        % Use raw d2 subsample matrix
                        subMat = d2SubsamplesAll{a};  % [numWindows x nSubsamples]
                    end

                    if ~isempty(subMat) && size(subMat, 1) == numel(d2ToPlot{a})
                        subMean = nanmean(subMat, 2);
                        subStd = nanstd(subMat, 0, 2);  % Std across subsamples
                        ribbonValid = ~isnan(subMean) & ~isnan(subStd) & ~isnan(startS{a}(:));
                        if any(ribbonValid)
                            xFill = startS{a}(ribbonValid);
                            yMean = subMean(ribbonValid);
                            yStd = subStd(ribbonValid);
                            [xFill, yMean, yStd] = downsample_series(xFill, yMean, yStd);
                            fill([xFill, fliplr(xFill)], ...
                                 [yMean + yStd, fliplr(yMean - yStd)], ...
                                 areaColor, 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
                                 'HandleVisibility', 'off');
                        end
                    end
                end

                % Plot mean trace on top (downsampled for light export)
                [xLine, yLine] = downsample_series(startS{a}(validIdx), d2ToPlot{a}(validIdx));
                plot(xLine, yLine, '-', ...
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
                [xFill, yMean, yStd] = downsample_series(xFill, yMean, yStd);

                % Plot shaded region (mean ± std)
                fill([xFill, fliplr(xFill)], ...
                    [yMean + yStd, fliplr(yMean - yStd)], ...
                    [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
                    'DisplayName', 'Permuted mean ± std');

                % Plot mean line (downsampled)
                [xLine, yLine] = downsample_series(startS{a}(validIdx), permutedMean(validIdx));
                plot(xLine, yLine, '-', ...
                    'Color', [0.5 0.5 1], 'LineWidth', 1.5, 'LineStyle', '--', ...
                    'DisplayName', 'Permuted mean');
            end
        end
    end
    
    % Plot behavior proportion (centered on metric mean) if available
    if plotBehaviorProportion && ~isempty(behaviorProportion{a}) && ~isempty(startS{a}) && ...
            ~isnan(metricMeans{a})
        validBhvVals = behaviorProportion{a}(~isnan(behaviorProportion{a}));
        if ~isempty(validBhvVals)
            meanBhvVal = mean(validBhvVals);
            % Center: subtract behavior mean, add metric mean; scale by 0.2 for visibility
            behaviorProportionCentered = 0.2 * (behaviorProportion{a} - meanBhvVal) + metricMeans{a};
            validIdxBhv = ~isnan(behaviorProportionCentered);
            if any(validIdxBhv)
                [xLineBhv, yLineBhv] = downsample_series(startS{a}(validIdxBhv), behaviorProportionCentered(validIdxBhv));
                plot(xLineBhv, yLineBhv, ':', ...
                    'Color', [0 0 0], 'LineWidth', 2, ...
                    'DisplayName', sprintf('%s (bhv prop)', areas{a}));
            end
        end
    end

    if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)]);
    end
    
    % Get number of neurons for this area
    nNeurons = 0;
    if isfield(dataStruct, 'idMatIdx') && a <= length(dataStruct.idMatIdx) && ~isempty(dataStruct.idMatIdx{a})
        nNeurons = length(dataStruct.idMatIdx{a});
    end
    
    if normalizeD2
        title(sprintf('%s (n=%d) - d2 (normalized, left) and PopActivity Windows (right)', areas{a}, nNeurons));
    else
        title(sprintf('%s (n=%d) - d2 (left) and PopActivity Windows (right)', areas{a}, nNeurons));
    end
    if idx == length(areasToTest)
    xlabel('Time (s)');
    end
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
    exportgraphics(gcf, plotPath, 'Resolution', plotResolution);
    fprintf('Saved plot to: %s\n', plotPath);

% Optional EPS (off by default; vector export often triggers renderer crash)
if saveEps
    if ~isempty(plotConfig.filePrefix)
        plotPathEps = fullfile(config.saveDir, ...
            sprintf('%s_criticality_%s_ar%s.eps', ...
            plotConfig.filePrefix, sessionType, filenameSuffix));
    else
        plotPathEps = fullfile(config.saveDir, ...
            sprintf('criticality_%s_ar%s.eps', ...
            sessionType, filenameSuffix));
    end
    try
        set(gcf, 'Renderer', 'painters');
        exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
        fprintf('Saved plot to: %s\n', plotPathEps);
    catch me
        fprintf('Skipping EPS save (renderer failed): %s\n', me.message);
    end
end
end

function [xOut, varargout] = downsample_plot_series(x, maxPts, varargin)
% Downsample long series to maxPts points for lighter plotting (avoids renderer crash).
% Preserves first and last point. All outputs as row vectors.
    n = numel(x);
    if n <= maxPts
        if iscolumn(x), x = x'; end
        xOut = x;
        varargout = cell(1, numel(varargin));
        for i = 1:numel(varargin)
            y = varargin{i};
            if iscolumn(y), y = y'; end
            varargout{i} = y;
        end
        return;
    end
    idx = round(linspace(1, n, maxPts));
    xOut = x(idx);
    if iscolumn(xOut), xOut = xOut'; end
    varargout = cell(1, numel(varargin));
    for i = 1:numel(varargin)
        y = varargin{i};
        yOut = y(idx);
        if iscolumn(yOut), yOut = yOut'; end
        varargout{i} = yOut;
    end
end