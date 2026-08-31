function criticality_ar_compare_plot(results, plotConfig, config, dataStruct, filenameSuffix)
% CRITICALITY_AR_COMPARE_PLOT Overlay Euclidean d2 and KL-rate d2 (+ error)
%
% Variables:
%   results        - Results from criticality_ar_compare_analysis()
%   plotConfig     - Plotting configuration from setup_plotting()
%   config         - Configuration structure
%   dataStruct     - Data structure from load_sliding_window_data()
%   filenameSuffix - Filename suffix (includes _klcompare)
%
% Goal:
%   Per-area time series with Euclidean d2 on the left y-axis and KL-rate
%   d2 on the right y-axis, plus a shaded S2.5 error ribbon. A second
%   figure scatters the two metrics window-by-window.

srcRoot = fullfile(fileparts(mfilename('fullpath')), '..', '..');
addpath(srcRoot);
add_figure_tools_path();
utilsPath = fullfile(srcRoot, 'sliding_window_prep', 'utils');
if exist(utilsPath, 'dir')
    addpath(utilsPath);
end
dataPrepPath = fullfile(srcRoot, 'data_prep');
if exist(dataPrepPath, 'dir')
    addpath(dataPrepPath);
end

plotResolution = 300;
if isfield(config, 'plotResolution') && ~isempty(config.plotResolution)
    plotResolution = config.plotResolution;
end
maxPlotPoints = Inf;
if isfield(config, 'maxPlotPoints') && ~isempty(config.maxPlotPoints)
    maxPlotPoints = config.maxPlotPoints;
end
useSoftwareRenderer = true;
if isfield(config, 'useSoftwareRenderer') && ~isempty(config.useSoftwareRenderer)
    useSoftwareRenderer = config.useSoftwareRenderer;
end
saveEps = false;
if isfield(config, 'saveEps') && ~isempty(config.saveEps)
    saveEps = config.saveEps;
end
downsample_series = @(x, varargin) downsample_plot_series(x, maxPlotPoints, varargin{:});

areas = results.areas;
d2 = results.d2;
d2Kl = results.d2Kl;
d2KlErr = results.d2KlErr;
startS = results.startS;

useRelativeTime = false;
if isfield(config, 'useRelativeTime') && ~isempty(config.useRelativeTime)
    useRelativeTime = logical(config.useRelativeTime);
end
if useRelativeTime
    t0 = 0;
    if nargin >= 4 && ~isempty(dataStruct)
        t0 = session_time_origin(dataStruct);
    elseif isfield(results, 'params') && isfield(results.params, 'timeOrigin') ...
            && ~isempty(results.params.timeOrigin)
        t0 = results.params.timeOrigin;
    end
    for aRel = 1:numel(startS)
        if ~isempty(startS{aRel})
            startS{aRel} = startS{aRel} - t0;
        end
    end
end

useLog10D2 = false;
if isfield(config, 'useLog10D2') && ~isempty(config.useLog10D2)
    useLog10D2 = config.useLog10D2;
elseif isfield(results, 'params') && isfield(results.params, 'useLog10D2')
    useLog10D2 = results.params.useLog10D2;
end

klErrBars = true;
if isfield(config, 'klErrBars') && ~isempty(config.klErrBars)
    klErrBars = logical(config.klErrBars);
elseif isfield(results, 'params') && isfield(results.params, 'klErrBars')
    klErrBars = logical(results.params.klErrBars);
end

d2ToPlot = d2;
d2KlToPlot = d2Kl;
d2KlLo = cell(size(d2Kl));
d2KlHi = cell(size(d2Kl));
for a = 1:numel(d2Kl)
    if isempty(d2Kl{a})
        d2KlLo{a} = [];
        d2KlHi{a} = [];
        continue;
    end
    errA = nan(size(d2Kl{a}));
    if a <= numel(d2KlErr) && ~isempty(d2KlErr{a})
        errA = d2KlErr{a};
    end
    d2KlLo{a} = d2Kl{a} - errA;
    d2KlHi{a} = d2Kl{a} + errA;
end

if useLog10D2
    d2ToPlot = log10_cell_numeric(d2ToPlot);
    d2KlToPlot = log10_cell_numeric(d2KlToPlot);
    d2KlLo = log10_cell_numeric(d2KlLo);
    d2KlHi = log10_cell_numeric(d2KlHi);
    d2LabelOld = 'log_{10}(d2 Euclidean)';
    d2LabelKl = 'log_{10}(d2 KL bits/s)';
else
    d2LabelOld = 'd2 (Euclidean)';
    d2LabelKl = 'd2 (KL bits/s)';
end

if isfield(results, 'behaviorProportion') && strcmp(results.sessionType, 'spontaneous')
    behaviorProportion = results.behaviorProportion;
    plotBehaviorProportion = true;
else
    behaviorProportion = cell(1, length(areas));
    plotBehaviorProportion = false;
end

areasToTest = resolve_compare_plot_area_indices(results, config, dataStruct, areas, startS);
if isempty(areasToTest)
    warning('No requested brain areas with valid startS data found. Skipping plot.');
    return;
end
fprintf('Plotting brain areas: %s\n', strjoin(areas(areasToTest), ', '));

sessionType = results.sessionType;
slidingWindowSize = results.params.slidingWindowSize;

yLimOld = compute_shared_ylim(d2ToPlot, areasToTest, useLog10D2);
yLimKl = compute_shared_ylim_with_bounds(d2KlToPlot, d2KlLo, d2KlHi, areasToTest, useLog10D2);

figure(921); clf;
set(gcf, 'Units', 'pixels');
set(gcf, 'Position', plotConfig.targetPos);
if useSoftwareRenderer
    set(gcf, 'Renderer', 'zbuffer');
end

numRows = numel(areasToTest);
useTightSubplot = exist('tight_subplot', 'file');
if useTightSubplot
    ha = tight_subplot(numRows, 1, [0.035 0.04], [0.06 0.08], [0.08 0.08]);
else
    ha = zeros(numRows, 1);
    for i = 1:numRows
        ha(i) = subplot(numRows, 1, i);
    end
end

areaColors = {[1 0.6 0.6], [0 .8 0], [0 0 1], [1 .4 1], [1 0.5 0]};
klColor = [0.15 0.15 0.15];
klRibbonColor = [0.45 0.45 0.45];

for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    if useTightSubplot
        axes(ha(idx));
    else
        subplot(numRows, 1, idx);
    end
    hold on;

    add_event_markers(dataStruct, startS, 'areaIdx', a);

    yyaxis left;
    hold on;
    if ~isempty(d2ToPlot{a}) && ~isempty(startS{a})
        validIdx = ~isnan(d2ToPlot{a});
        if any(validIdx)
            areaColor = areaColors{min(a, length(areaColors))};
            [xLine, yLine] = downsample_series(startS{a}(validIdx), d2ToPlot{a}(validIdx));
            plot(xLine, yLine, '-', 'Color', areaColor, 'LineWidth', 2.5, ...
                'DisplayName', 'd2 Euclidean');
        end
    end
    ylabel(d2LabelOld, 'Color', areaColors{min(a, length(areaColors))});
    if ~isempty(yLimOld)
        ylim(yLimOld);
    end
    set(gca, 'YColor', areaColors{min(a, length(areaColors))});

    yyaxis right;
    hold on;
    if ~isempty(d2KlToPlot{a}) && ~isempty(startS{a})
        validKl = ~isnan(d2KlToPlot{a});
        if klErrBars && ~isempty(d2KlLo{a}) && ~isempty(d2KlHi{a})
            ribbonValid = validKl & ~isnan(d2KlLo{a}) & ~isnan(d2KlHi{a});
            if any(ribbonValid)
                [xFill, yLo, yHi] = downsample_series(startS{a}(ribbonValid), ...
                    d2KlLo{a}(ribbonValid), d2KlHi{a}(ribbonValid));
                fill([xFill, fliplr(xFill)], [yHi, fliplr(yLo)], ...
                    klRibbonColor, 'FaceAlpha', 0.28, 'EdgeColor', 'none', ...
                    'DisplayName', '\Delta d2 (KL)');
            end
        end
        if any(validKl)
            [xLineKl, yLineKl] = downsample_series(startS{a}(validKl), d2KlToPlot{a}(validKl));
            plot(xLineKl, yLineKl, '-', 'Color', klColor, 'LineWidth', 2, ...
                'DisplayName', 'd2 KL');
        end
    end
    ylabel(d2LabelKl, 'Color', klColor);
    if ~isempty(yLimKl)
        ylim(yLimKl);
    end
    set(gca, 'YColor', klColor);

    if plotBehaviorProportion && idx <= numel(behaviorProportion) && ...
            ~isempty(behaviorProportion{a}) && ~isempty(startS{a}) && ...
            ~isempty(d2ToPlot{a})
        validD2 = d2ToPlot{a}(~isnan(d2ToPlot{a}));
        validBhv = behaviorProportion{a}(~isnan(behaviorProportion{a}));
        if ~isempty(validD2) && ~isempty(validBhv)
            yyaxis left;
            meanD2 = mean(validD2);
            meanBhv = mean(validBhv);
            behaviorCentered = 0.2 * (behaviorProportion{a} - meanBhv) + meanD2;
            validIdxBhv = ~isnan(behaviorCentered);
            if any(validIdxBhv)
                [xLineBhv, yLineBhv] = downsample_series(startS{a}(validIdxBhv), ...
                    behaviorCentered(validIdxBhv));
                plot(xLineBhv, yLineBhv, ':', 'Color', [0 0 0], 'LineWidth', 1.5, ...
                    'DisplayName', 'bhv prop');
            end
        end
    end

    if ~isempty(startS{a})
        xlim([startS{a}(1) startS{a}(end)]);
    end

    nNeurons = 0;
    if isfield(dataStruct, 'idMatIdx') && a <= length(dataStruct.idMatIdx) && ...
            ~isempty(dataStruct.idMatIdx{a})
        nNeurons = length(dataStruct.idMatIdx{a});
    end
    title(sprintf('%s (n=%d) - Euclidean d2 (left) vs KL d2 (right)', areas{a}, nNeurons));
    if idx == length(areasToTest)
        xlabel('Time (s)');
    end
    grid on;
    legend('Location', 'best');
    set(gca, 'XTickLabelMode', 'auto');
end

if ~isempty(plotConfig.filePrefix)
    sgtitle(sprintf('[%s] %s Euclidean vs KL d2 - win=%gs', ...
        plotConfig.filePrefix, sessionType, slidingWindowSize));
else
    sgtitle(sprintf('%s Euclidean vs KL d2 - win=%gs', sessionType, slidingWindowSize));
end

if ~exist(config.saveDir, 'dir')
    mkdir(config.saveDir);
end

if ~isempty(plotConfig.filePrefix)
    plotPath = fullfile(config.saveDir, ...
        sprintf('%s_criticality_%s_ar_compare%s.png', ...
        plotConfig.filePrefix, sessionType, filenameSuffix));
else
    plotPath = fullfile(config.saveDir, ...
        sprintf('criticality_%s_ar_compare%s.png', sessionType, filenameSuffix));
end
drawnow;
exportgraphics(gcf, plotPath, 'Resolution', plotResolution);
fprintf('Saved plot to: %s\n', plotPath);

if saveEps
    [plotDir, plotName] = fileparts(plotPath);
    plotPathEps = fullfile(plotDir, [plotName, '.eps']);
    try
        set(gcf, 'Renderer', 'painters');
        exportgraphics(gcf, plotPathEps, 'ContentType', 'vector');
        fprintf('Saved plot to: %s\n', plotPathEps);
    catch me
        fprintf('Skipping EPS save (renderer failed): %s\n', me.message);
    end
end

plot_d2_scatter(d2, d2Kl, d2KlErr, areas, areasToTest, config, plotConfig, ...
    sessionType, filenameSuffix, useLog10D2, plotResolution);
end

function plot_d2_scatter(d2, d2Kl, d2KlErr, areas, areasToTest, config, plotConfig, ...
    sessionType, filenameSuffix, useLog10D2, plotResolution)
% PLOT_D2_SCATTER Window-by-window Euclidean vs KL d2 with vertical error bars

nAreas = numel(areasToTest);
figure(922); clf;
set(gcf, 'Units', 'pixels');
pos = plotConfig.targetPos;
pos(3) = max(pos(3), 420 * min(nAreas, 3));
set(gcf, 'Position', pos);

for idx = 1:nAreas
    a = areasToTest(idx);
    subplot(1, nAreas, idx);
    hold on;

    x = d2{a}(:);
    y = d2Kl{a}(:);
    if a <= numel(d2KlErr) && ~isempty(d2KlErr{a})
        yErr = d2KlErr{a}(:);
    else
        yErr = nan(size(y));
    end
    valid = isfinite(x) & isfinite(y);
    if ~any(valid)
        title(sprintf('%s (no valid pairs)', areas{a}));
        continue;
    end
    x = x(valid);
    y = y(valid);
    yErr = yErr(valid);
    yErr(~isfinite(yErr)) = 0;

    if useLog10D2
        xPlot = log10_safe_numeric(x);
        yPlot = log10_safe_numeric(y);
        yLo = log10_safe_numeric(max(y - yErr, realmin));
        yHi = log10_safe_numeric(y + yErr);
        negErr = yPlot - yLo;
        posErr = yHi - yPlot;
        negErr(~isfinite(negErr) | negErr < 0) = 0;
        posErr(~isfinite(posErr) | posErr < 0) = 0;
        errorbar(xPlot, yPlot, negErr, posErr, 'o', 'Color', [0.3 0.3 0.3], ...
            'MarkerFaceColor', [0.2 0.2 0.8], 'MarkerSize', 4, 'LineWidth', 0.8);
        xUse = xPlot;
        yUse = yPlot;
        xlabel('log_{10}(d2 Euclidean)');
        ylabel('log_{10}(d2 KL bits/s)');
    else
        errorbar(x, y, yErr, 'o', 'Color', [0.3 0.3 0.3], ...
            'MarkerFaceColor', [0.2 0.2 0.8], 'MarkerSize', 4, 'LineWidth', 0.8);
        xUse = x;
        yUse = y;
        xlabel('d2 (Euclidean)');
        ylabel('d2 (KL bits/s)');
    end

    corrMask = isfinite(xUse) & isfinite(yUse);
    if nnz(corrMask) >= 3
        rVal = corr(xUse(corrMask), yUse(corrMask), 'rows', 'complete');
        title(sprintf('%s  r = %.3f  n = %d', areas{a}, rVal, nnz(corrMask)));
    else
        title(sprintf('%s  n = %d', areas{a}, nnz(corrMask)));
    end
    grid on;
    axis square;
end

if ~isempty(plotConfig.filePrefix)
    sgtitle(sprintf('[%s] %s Euclidean vs KL d2 (window scatter)', ...
        plotConfig.filePrefix, sessionType));
else
    sgtitle(sprintf('%s Euclidean vs KL d2 (window scatter)', sessionType));
end

if ~isempty(plotConfig.filePrefix)
    scatterPath = fullfile(config.saveDir, ...
        sprintf('%s_criticality_%s_ar_compare_scatter%s.png', ...
        plotConfig.filePrefix, sessionType, filenameSuffix));
else
    scatterPath = fullfile(config.saveDir, ...
        sprintf('criticality_%s_ar_compare_scatter%s.png', sessionType, filenameSuffix));
end
drawnow;
exportgraphics(gcf, scatterPath, 'Resolution', plotResolution);
fprintf('Saved scatter to: %s\n', scatterPath);
end

function yLimShared = compute_shared_ylim(valsCell, areasToTest, useLog10D2)
% COMPUTE_SHARED_YLIM Shared y-limits across area traces

allY = [];
for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    if a <= numel(valsCell) && ~isempty(valsCell{a})
        allY = [allY; valsCell{a}(:)]; %#ok<AGROW>
    end
end
yLimShared = ylimits_from_values(allY, useLog10D2);
end

function yLimShared = compute_shared_ylim_with_bounds(midCell, loCell, hiCell, areasToTest, useLog10D2)
% COMPUTE_SHARED_YLIM_WITH_BOUNDS Shared y-limits including error ribbon

allY = [];
for idx = 1:length(areasToTest)
    a = areasToTest(idx);
    if a <= numel(midCell) && ~isempty(midCell{a})
        allY = [allY; midCell{a}(:)]; %#ok<AGROW>
    end
    if a <= numel(loCell) && ~isempty(loCell{a})
        allY = [allY; loCell{a}(:)]; %#ok<AGROW>
    end
    if a <= numel(hiCell) && ~isempty(hiCell{a})
        allY = [allY; hiCell{a}(:)]; %#ok<AGROW>
    end
end
yLimShared = ylimits_from_values(allY, useLog10D2);
end

function yLimShared = ylimits_from_values(allY, useLog10D2)
allY = allY(isfinite(allY));
if isempty(allY)
    yLimShared = [];
    return;
end
minY = min(allY);
maxY = max(allY);
yRange = maxY - minY;
if yRange == 0
    yRange = max(0.1, 0.05 * max(abs(maxY), abs(minY)));
end
pad = 0.05 * yRange;
yMin = minY - pad;
yMax = maxY + pad;
if ~useLog10D2
    yMin = min(yMin, 0);
    if minY >= 0
        yMin = max(0, yMin);
    end
end
yLimShared = [yMin, yMax];
end

function cellOut = log10_cell_numeric(cellIn)
cellOut = cellIn;
for i = 1:numel(cellIn)
    if isempty(cellIn{i}) || ~isnumeric(cellIn{i})
        continue;
    end
    cellOut{i} = log10_safe_numeric(cellIn{i});
end
end

function y = log10_safe_numeric(x)
validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function [xOut, varargout] = downsample_plot_series(x, maxPts, varargin)
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

function areasToTest = resolve_compare_plot_area_indices(results, config, dataStruct, areas, startS)
areasToTest = [];
requestedNames = {};
if isfield(config, 'brainAreas') && ~isempty(config.brainAreas)
    requestedNames = config.brainAreas;
elseif isfield(results, 'params') && isfield(results.params, 'brainAreas') ...
        && ~isempty(results.params.brainAreas)
    requestedNames = results.params.brainAreas;
end
if ~isempty(requestedNames)
    areasToTest = area_names_to_indices(areas, requestedNames);
end
if isempty(areasToTest) && isfield(results, 'params') ...
        && isfield(results.params, 'areasToTest') && ~isempty(results.params.areasToTest)
    areasToTest = results.params.areasToTest(:)';
end
if isempty(areasToTest) && ~isempty(dataStruct) && isstruct(dataStruct) ...
        && isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
    areasToTest = dataStruct.areasToTest(:)';
end
if isempty(areasToTest)
    areasToTest = 1:numel(areas);
end
areasToTest = unique(areasToTest, 'stable');
areasToTest = areasToTest(areasToTest >= 1 & areasToTest <= numel(areas));
keepMask = false(size(areasToTest));
for i = 1:numel(areasToTest)
    a = areasToTest(i);
    keepMask(i) = a <= numel(startS) && ~isempty(startS{a});
end
if any(keepMask)
    areasToTest = areasToTest(keepMask);
end
end

function idx = area_names_to_indices(areas, areaNames)
if ischar(areaNames) || isstring(areaNames)
    areaNames = cellstr(areaNames);
end
idx = [];
for i = 1:numel(areaNames)
    thisName = char(areaNames{i});
    matchIdx = find(strcmp(areas, thisName));
    if ~isempty(matchIdx)
        idx = [idx, matchIdx(:)']; %#ok<AGROW>
    end
end
end
