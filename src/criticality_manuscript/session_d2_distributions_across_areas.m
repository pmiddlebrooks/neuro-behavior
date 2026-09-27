%%
% Session d2 across brain areas (Manuscript)
%
% Same windowed AR/d2 pipeline as session_d2_distributions.m. Correlates
% window-wise d2 between user-specified brain areas and plots:
%   1) pairwise d2 scatter (one point per shared window)
%   2) mean popActivity and d2 vs time, areas overlaid in area colors
%
% Variables (configure in this section):
%   sessionType, sessionName, subjectName - Set in the workspace before running
%   areasToCorrelate - Cell of area names to correlate (at least two),
%                      e.g. {'M23','VS'} or {'M23','M56','DS','VS'}
%   dataSource, collectStart, collectEnd, d2Window, d2WindowAlign, stepSize
%   useLog10D2, useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple
%   enablePermutations, nPermutations, binSize, d2Method, kl* , runParallel, nWorkers
%   useRelativeTime, saveFigure, plotConfig
%
% Goal:
%   Show whether window-wise d2 moves together across brain areas, and show
%   those areas' pop activity and d2 on one shared time axis.

%% Configuration
% Prefer session identity already set in the workspace (e.g. scratch.m batch).

if ~exist('sessionType', 'var') || isempty(sessionType) ...
        || ~exist('sessionName', 'var') || isempty(sessionName)
    error(['Set sessionType and sessionName in the workspace before running ', ...
        '(subjectName as well for spontaneous, interval, and semicircle).']);
end

areasToCorrelate = {'M23M56', 'DS', 'VS'};

dataSource = 'spikes';
collectStart = 0;
collectEnd = [];
d2Window = 45;  % seconds
stepSize = 5;   % same window grid as session_d2_distributions.m
d2WindowAlign = 'center';  % 'center' | 'leadingEdge'
useLog10D2 = false;
useSubsampling = true;
nSubsamples = 20;
nNeuronsSubsample = 60;
minNeuronsMultiple = 1.1;
enablePermutations = false;
nPermutations = 3;
useRelativeTime = false;
binSize = 0.025;
binSize = 0.04;
d2Method = 'kl'; % euclidean or kl
klFitMethod = 'MaxLikelihood';
klErrBars = false;
runParallel = true;
nWorkers = 3;
klParallel = runParallel;
saveFigure = false;

plotConfig = fill_manuscript_plot_config();

opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.25;
opts.maxFiringRate = 200;
opts.loadBehaviorLabels = true;

analysisConfig = struct();
analysisConfig.slidingWindowSize = d2Window;
analysisConfig.stepSize = stepSize;
analysisConfig.binSize = binSize;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 5;
analysisConfig.enablePermutations = enablePermutations;
analysisConfig.nShuffles = nPermutations;
analysisConfig.normalizeD2 = enablePermutations;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.d2Method = d2Method;
analysisConfig.klFitMethod = klFitMethod;
analysisConfig.klErrBars = klErrBars;
analysisConfig.klParallel = klParallel;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = 20;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;

setup_criticality_manuscript_paths('session_d2_distributions_across_areas');
paths = get_paths();

areasToCorrelate = normalize_area_name_list(areasToCorrelate);
if numel(areasToCorrelate) < 2
    error('areasToCorrelate needs at least two brain areas.');
end

[d2Method, klFitMethod, klErrBars, klParallel] = normalize_kl_d2_options( ...
    d2Method, klFitMethod, klErrBars, klParallel);
analysisConfig.d2Method = d2Method;
analysisConfig.klFitMethod = klFitMethod;
analysisConfig.klErrBars = klErrBars;
analysisConfig.klParallel = klParallel;
d2PlotTag = format_d2_method_file_tag(d2Method, klFitMethod, klErrBars);
plotConfig.d2Method = d2Method;
d2WindowAlign = normalize_d2_window_align(d2WindowAlign);

fprintf('\n=== Session d2 across areas ===\n');
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Areas: %s\n', strjoin(areasToCorrelate, ', '));
fprintf('d2 windows: %.1f s, step %.1f s (%s); binSize: %.3f s\n', ...
    d2Window, stepSize, d2WindowAlign, binSize);
fprintf('d2Method: %s\n', d2Method);
maybe_start_kl_d2_parallel_pool(d2Method, klErrBars, klParallel, nWorkers);

subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
    subjectNameForLoad = subjectName;
end
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

loadedStart = scalar_time(session_time_origin(dataStruct), 0, @min);
collectStart = scalar_time([collectStart(:); loadedStart], 0, @max);
opts.collectStart = collectStart;

if ~isempty(collectEnd)
    sessionEndAbs = loaded_session_end(dataStruct);
    if isscalar(sessionEndAbs) && isfinite(sessionEndAbs)
        collectEnd = clamp_collect_end_to_session(collectEnd, sessionEndAbs, collectStart);
        opts.collectEnd = collectEnd;
    end
end

if isempty(collectEnd)
    fprintf('Collect window: [%.1f, full session] s\n', collectStart);
else
    fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
        collectStart, collectEnd, (collectEnd - collectStart) / 60);
end

brainAreaCombinations = default_manuscript_brain_area_combinations();
for iArea = 1:numel(areasToCorrelate)
    areaName = areasToCorrelate{iArea};
    if any(strcmp(dataStruct.areas, areaName))
        continue;
    end
    [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
        dataStruct, areaName, brainAreaCombinations, true);
    if ~areaOk || ~any(strcmp(dataStruct.areas, areaName))
        error('Brain area "%s" is not available in this session.', areaName);
    end
end
analysisConfig.brainAreas = areasToCorrelate;

results = criticality_ar_analysis(dataStruct, analysisConfig);
[results, keptAreas] = keep_correlate_areas(results, areasToCorrelate);
if numel(keptAreas) < 2
    error('Need at least two areas with d2. Kept: %s', strjoin(keptAreas, ', '));
end

[d2AxisLabel, d2TitleLabel, d2LabelInterpreter] = get_d2_plot_labels(useLog10D2, d2Method);

figCorr = plot_d2_area_correlation_scatters(results, keptAreas, useLog10D2, ...
    d2AxisLabel, d2LabelInterpreter, plotConfig, sessionName, d2Window);
print_d2_area_correlations(results, keptAreas, useLog10D2);

figTime = plot_overlaid_pop_d2_timeline(results, keptAreas, collectStart, collectEnd, ...
    d2Window, binSize, useLog10D2, plotConfig, sessionName, useRelativeTime, ...
    d2WindowAlign, d2AxisLabel, d2TitleLabel, d2LabelInterpreter, dataStruct);

if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    areaTag = strjoin(keptAreas, '-');
    if isempty(collectEnd)
        collectTag = sprintf('%.0f-full', collectStart);
    else
        collectTag = sprintf('%.0f-%.0f', collectStart, collectEnd);
    end
    logTag = '';
    if useLog10D2
        logTag = '_log10';
    end
    corrBase = sprintf('session_d2_across_areas_corr_%s_%s_win%.0fs_%ss%s%s', ...
        sessionName, areaTag, d2Window, collectTag, logTag, d2PlotTag);
    timeBase = sprintf('session_d2_across_areas_timeline_%s_%s_win%.0fs_%ss_%s%s%s', ...
        sessionName, areaTag, d2Window, collectTag, d2WindowAlign, logTag, d2PlotTag);
    exportgraphics(figCorr, fullfile(saveDir, [corrBase, '.png']), 'Resolution', 300);
    exportgraphics(figCorr, fullfile(saveDir, [corrBase, '.eps']), 'ContentType', 'vector');
    exportgraphics(figTime, fullfile(saveDir, [timeBase, '.png']), 'Resolution', 300);
    exportgraphics(figTime, fullfile(saveDir, [timeBase, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figures:\n  %s\n  %s\n', ...
        fullfile(saveDir, corrBase), fullfile(saveDir, timeBase));
end

function areaNames = normalize_area_name_list(areaNames)
% NORMALIZE_AREA_NAME_LIST - Cell column of area-name chars
%
% Variables:
%   areaNames - Char, string, or cell of area names
%
% Goal:
%   Return a cell array of char area names in the order given.

if ischar(areaNames)
    areaNames = {areaNames};
elseif isstring(areaNames)
    areaNames = cellstr(areaNames);
end
areaNames = areaNames(:);
for iName = 1:numel(areaNames)
    areaNames{iName} = char(strtrim(string(areaNames{iName})));
end
areaNames = areaNames(~cellfun(@isempty, areaNames));
end

function [resultsOut, keptAreas] = keep_correlate_areas(results, areasToCorrelate)
% KEEP_CORRELATE_AREAS - Drop requested areas that have no d2
%
% Variables:
%   results           - criticality_ar_analysis output
%   areasToCorrelate  - Requested area names, in plot order
%
% Goal:
%   Keep only areas that have a non-empty d2 vector, preserving request order.

keptAreas = {};
keepIdx = [];
for iArea = 1:numel(areasToCorrelate)
    areaName = areasToCorrelate{iArea};
    areaIdx = find(strcmp(results.areas, areaName), 1);
    if isempty(areaIdx) || areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
        warning('session_d2_distributions_across_areas:MissingArea', ...
            'No d2 for "%s"; omitting it from correlation plots.', areaName);
        continue;
    end
    keepIdx(end + 1) = areaIdx; %#ok<AGROW>
    keptAreas{end + 1} = areaName; %#ok<AGROW>
end
resultsOut = results;
resultsOut.areas = keptAreas;
cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'd2PermutedMean', ...
    'd2PermutedSEM', 'popActivityWindows', 'popActivityFull', ...
    'd2Subsamples', 'd2NormalizedSubsamples'};
for iField = 1:numel(cellFields)
    fieldName = cellFields{iField};
    if ~isfield(results, fieldName)
        continue;
    end
    fieldVals = results.(fieldName);
    keptVals = cell(1, numel(keepIdx));
    for iKeep = 1:numel(keepIdx)
        srcIdx = keepIdx(iKeep);
        if numel(fieldVals) >= srcIdx
            keptVals{iKeep} = fieldVals{srcIdx};
        else
            keptVals{iKeep} = [];
        end
    end
    resultsOut.(fieldName) = keptVals;
end
end

function fig = plot_d2_area_correlation_scatters(results, keptAreas, useLog10D2, ...
    d2AxisLabel, d2LabelInterpreter, plotConfig, sessionName, d2Window)
% PLOT_D2_AREA_CORRELATION_SCATTERS - Pairwise window d2 scatter
%
% Variables:
%   results, keptAreas - Areas in plot order, already filtered
%   useLog10D2         - Apply log10 before correlating
%   d2AxisLabel        - Shared axis label text
%   plotConfig         - Fonts and marker sizes
%
% Goal:
%   One scatter per unique pair. Each point is one window present in both areas.
%   Every panel uses the same x and y limits, spanning all plotted d2 values.

nAreas = numel(keptAreas);
pairIdx = nchoosek(1:nAreas, 2);
nPairs = size(pairIdx, 1);
nCols = ceil(sqrt(nPairs));
nRows = ceil(nPairs / nCols);
sharedLim = shared_d2_axis_limits(results, nAreas, useLog10D2);

fig = figure('Color', 'w', 'Name', sprintf('d2 area correlation — %s', sessionName), ...
    'Position', [80 80 max(520, 380 * nCols) max(420, 340 * nRows)]);

for iPair = 1:nPairs
    idxX = pairIdx(iPair, 1);
    idxY = pairIdx(iPair, 2);
    [xVals, yVals] = paired_window_d2(results, idxX, idxY, useLog10D2);
    ax = subplot(nRows, nCols, iPair, 'Parent', fig);
    hold(ax, 'on');
    colorX = area_plot_color(keptAreas{idxX});
    colorY = area_plot_color(keptAreas{idxY});
    if isempty(xVals)
        text(ax, 0.5, 0.5, 'no shared windows', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'Color', [0.45 0.45 0.45]);
        rVal = nan;
        nWin = 0;
    else
        scatter(ax, xVals, yVals, plotConfig.scatterMarkerSize, [0.2 0.2 0.2], ...
            'filled', 'MarkerFaceAlpha', plotConfig.markerFaceAlpha, ...
            'MarkerEdgeColor', 'none');
        add_regression_line(ax, xVals, yVals, plotConfig);
        rVal = pearson_r(xVals, yVals);
        nWin = numel(xVals);
    end
    xlabel(ax, sprintf('%s %s', keptAreas{idxX}, d2AxisLabel), ...
        'Color', colorX, 'FontSize', plotConfig.axisLabelFontSize, ...
        'Interpreter', d2LabelInterpreter);
    ylabel(ax, sprintf('%s %s', keptAreas{idxY}, d2AxisLabel), ...
        'Color', colorY, 'FontSize', plotConfig.axisLabelFontSize, ...
        'Interpreter', d2LabelInterpreter);
    title(ax, sprintf('r = %.3f   n = %d', rVal, nWin), ...
        'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
    if ~isempty(sharedLim)
        xlim(ax, sharedLim);
        ylim(ax, sharedLim);
        axis(ax, 'square');
    end
    set(ax, 'Box', 'off', 'TickDir', 'out', ...
        'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
    hold(ax, 'off');
end

sgtitle(fig, sprintf('%s | d2 across areas (%.0f s windows)', sessionName, d2Window), ...
    'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
end

function sharedLim = shared_d2_axis_limits(results, nAreas, useLog10D2)
% SHARED_D2_AXIS_LIMITS - One [min max] span for every correlation axis
%
% Variables:
%   results    - AR results with per-area d2
%   nAreas     - Number of areas being correlated
%   useLog10D2 - Match the values drawn on the scatters
%
% Goal:
%   Cover every finite d2 from the plotted areas, with a small pad, so x and y
%   and every pair panel share one scale.

allD2 = [];
for iArea = 1:nAreas
    d2Vec = area_d2_vector(results, iArea, useLog10D2);
    allD2 = [allD2; d2Vec(isfinite(d2Vec))]; %#ok<AGROW>
end
if isempty(allD2)
    sharedLim = [];
    return;
end
d2Min = min(allD2);
d2Max = max(allD2);
if d2Min == d2Max
    pad = max(abs(d2Min) * 0.05, 1e-3);
else
    pad = 0.05 * (d2Max - d2Min);
end
sharedLim = [d2Min - pad, d2Max + pad];
end

function print_d2_area_correlations(results, keptAreas, useLog10D2)
% PRINT_D2_AREA_CORRELATIONS - Pearson r for each area pair

fprintf('\n=== Window-wise d2 correlations ===\n');
nAreas = numel(keptAreas);
for idxX = 1:nAreas
    for idxY = idxX + 1:nAreas
        [xVals, yVals] = paired_window_d2(results, idxX, idxY, useLog10D2);
        rVal = pearson_r(xVals, yVals);
        fprintf('  %s vs %s: r = %.3f, n = %d\n', ...
            keptAreas{idxX}, keptAreas{idxY}, rVal, numel(xVals));
    end
end
end

function [xVals, yVals] = paired_window_d2(results, idxX, idxY, useLog10D2)
% PAIRED_WINDOW_D2 - Finite d2 pairs aligned on window center time
%
% Variables:
%   results    - AR results with startS and d2 per area
%   idxX, idxY - Area indices
%   useLog10D2 - log10 before pairing
%
% Goal:
%   Match windows by startS so each pair is the same analysis window.

xVals = [];
yVals = [];
d2X = area_d2_vector(results, idxX, useLog10D2);
d2Y = area_d2_vector(results, idxY, useLog10D2);
tX = area_start_times(results, idxX);
tY = area_start_times(results, idxY);
if isempty(d2X) || isempty(d2Y) || isempty(tX) || isempty(tY)
    return;
end

nX = min(numel(d2X), numel(tX));
nY = min(numel(d2Y), numel(tY));
d2X = d2X(1:nX);
tX = tX(1:nX);
d2Y = d2Y(1:nY);
tY = tY(1:nY);

if nX == nY && max(abs(tX - tY)) < 1e-6
    validMask = isfinite(d2X) & isfinite(d2Y);
    xVals = d2X(validMask);
    yVals = d2Y(validMask);
    return;
end

matchTol = 1e-3;
[tCommon, idxInX, idxInY] = intersect_window_times(tX, tY, matchTol);
if isempty(tCommon)
    return;
end
d2X = d2X(idxInX);
d2Y = d2Y(idxInY);
validMask = isfinite(d2X) & isfinite(d2Y);
xVals = d2X(validMask);
yVals = d2Y(validMask);
end

function [tCommon, idxInX, idxInY] = intersect_window_times(tX, tY, matchTol)
% INTERSECT_WINDOW_TIMES - Indices of window times shared by two areas

tCommon = [];
idxInX = [];
idxInY = [];
if isempty(tX) || isempty(tY)
    return;
end
usedY = false(numel(tY), 1);
for iX = 1:numel(tX)
    timeDelta = abs(tY - tX(iX));
    timeDelta(usedY) = inf;
    [minDelta, iY] = min(timeDelta);
    if minDelta <= matchTol
        tCommon(end + 1, 1) = tX(iX); %#ok<AGROW>
        idxInX(end + 1, 1) = iX; %#ok<AGROW>
        idxInY(end + 1, 1) = iY; %#ok<AGROW>
        usedY(iY) = true;
    end
end
end

function fig = plot_overlaid_pop_d2_timeline(results, keptAreas, collectStart, collectEnd, ...
    d2Window, binSize, useLog10D2, plotConfig, sessionName, useRelativeTime, ...
    d2WindowAlign, d2AxisLabel, d2TitleLabel, d2LabelInterpreter, dataStruct)
% PLOT_OVERLAID_POP_D2_TIMELINE - popActivity and d2 vs time, areas overlaid
%
% Variables:
%   results, keptAreas - Filtered areas
%   d2WindowAlign      - 'center' or 'leadingEdge'
%   useRelativeTime    - If true, t = 0 at collectStart
%
% Goal:
%   Two stacked axes. Each axis draws every requested area in its own color.

nAreas = numel(keptAreas);
tMinAbs = collectStart;
if isempty(tMinAbs) || ~isfinite(tMinAbs)
    tMinAbs = session_time_origin(dataStruct);
end
tMaxAbs = resolve_timeline_tmax(results, collectStart, collectEnd, d2Window, dataStruct);
if useRelativeTime
    tMin = 0;
    tMax = tMaxAbs - tMinAbs;
    timeShift = tMinAbs;
    xLabelText = 'Time from collectStart (s)';
else
    tMin = tMinAbs;
    tMax = tMaxAbs;
    timeShift = 0;
    xLabelText = 'Time (s)';
end

fig = figure('Color', 'w', 'Name', sprintf('%s across areas — %s', d2TitleLabel, sessionName), ...
    'Position', [140 60 980 640]);
axPop = subplot(2, 1, 1, 'Parent', fig);
axD2 = subplot(2, 1, 2, 'Parent', fig);
hold(axPop, 'on');
hold(axD2, 'on');

for iArea = 1:nAreas
    areaName = keptAreas{iArea};
    areaColor = area_plot_color(areaName);
    tWin = [];
    if iArea <= numel(results.startS) && ~isempty(results.startS{iArea})
        tWin = d2_window_align_times(results.startS{iArea}(:), d2Window, d2WindowAlign) - timeShift;
    end
    popVec = [];
    if isfield(results, 'popActivityWindows') && iArea <= numel(results.popActivityWindows)
        popVec = results.popActivityWindows{iArea}(:);
    end
    if ~isempty(popVec) && ~isempty(tWin)
        nPlot = min(numel(popVec), numel(tWin));
        plot(axPop, tWin(1:nPlot), popVec(1:nPlot), '-o', ...
            'Color', areaColor, 'MarkerFaceColor', areaColor, ...
            'MarkerSize', 4, 'LineWidth', plotConfig.axesLineWidth, ...
            'DisplayName', areaName);
    end

    d2Vec = area_d2_vector(results, iArea, useLog10D2);
    if ~isempty(d2Vec) && ~isempty(tWin)
        nPlot = min(numel(d2Vec), numel(tWin));
        tD2 = tWin(1:nPlot);
        d2Vec = d2Vec(1:nPlot);
        d2Sem = subsample_sem_d2(results, iArea, useLog10D2);
        add_sem_errorbars(axD2, tD2, d2Vec, d2Sem, isfinite(d2Vec) & isfinite(tD2), ...
            areaColor, plotConfig);
        plot(axD2, tD2, d2Vec, '-o', ...
            'Color', areaColor, 'MarkerFaceColor', areaColor, ...
            'MarkerSize', 4, 'LineWidth', plotConfig.axesLineWidth, ...
            'DisplayName', areaName);
    end
end

xlim(axPop, [tMin, tMax]);
xlim(axD2, [tMin, tMax]);
ylabel(axPop, 'mean pop', 'FontSize', plotConfig.axisLabelFontSize);
ylabel(axD2, d2AxisLabel, 'FontSize', plotConfig.axisLabelFontSize, ...
    'Interpreter', d2LabelInterpreter);
xlabel(axD2, xLabelText, 'FontSize', plotConfig.axisLabelFontSize);
set(axPop, 'XTickLabel', [], 'Box', 'off', 'TickDir', 'out', ...
    'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
set(axD2, 'Box', 'off', 'TickDir', 'out', ...
    'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
legend(axPop, 'Location', 'best', 'FontSize', plotConfig.legendFontSize, ...
    'Interpreter', 'none');
hold(axPop, 'off');
hold(axD2, 'off');
linkaxes([axPop, axD2], 'x');

sgtitle(fig, sprintf('%s | mean pop / %s (%.0fs windows, %s, bin=%.0f ms)', ...
    sessionName, d2TitleLabel, d2Window, d2WindowAlign, binSize * 1000), ...
    'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
fprintf('Plotted overlaid popActivity and d2 (%d areas, t=[%.1f, %.1f] s).\n', ...
    nAreas, tMin, tMax);
end

function rgb = area_plot_color(areaName)
% AREA_PLOT_COLOR - Stable RGB for a brain-area name
%
% Goal:
%   M23 and M2356 blue, M56 red-orange, DS yellow, VS green. Other names
%   get a distinct color from the lines palette.

knownNames = {'M23', 'M56', 'DS', 'VS', 'M23M56', 'M2356'};
knownColors = [
    0.0000, 0.4470, 0.7410
    0.8500, 0.3250, 0.0980
    0.9290, 0.6940, 0.1250
    0.4660, 0.6740, 0.1880
    0.4940, 0.1840, 0.5560
    0.0000, 0.4470, 0.7410
    ];
matchIdx = find(strcmpi(knownNames, areaName), 1);
if ~isempty(matchIdx)
    rgb = knownColors(matchIdx, :);
    return;
end
palette = lines(12);
nameKey = sum(double(char(areaName)));
rgb = palette(mod(nameKey, size(palette, 1)) + 1, :);
end

function d2Vec = area_d2_vector(results, areaIdx, useLog10D2)
% AREA_D2_VECTOR - d2 per window, optional log10

d2Vec = [];
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
    return;
end
d2Vec = results.d2{areaIdx}(:);
if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
end
end

function tStart = area_start_times(results, areaIdx)
% AREA_START_TIMES - Window center times (s) for one area

tStart = [];
if ~isfield(results, 'startS') || areaIdx > numel(results.startS)
    return;
end
tStart = results.startS{areaIdx}(:);
end

function d2Sem = subsample_sem_d2(results, areaIdx, useLog10D2)
% SUBSAMPLE_SEM_D2 - SEM of d2 across neuron subsamples, per window

d2Sem = [];
if ~isfield(results, 'd2Subsamples') || areaIdx > numel(results.d2Subsamples) ...
        || isempty(results.d2Subsamples{areaIdx})
    return;
end
subMat = results.d2Subsamples{areaIdx};
if useLog10D2
    subMat = log10_safe_numeric(subMat);
end
nSub = sum(isfinite(subMat), 2);
d2Sem = nanstd(subMat, 0, 2) ./ sqrt(max(nSub, 1));
d2Sem(nSub <= 1) = 0;
end

function add_sem_errorbars(ax, xVals, yVals, ySem, validMask, barColor, plotConfig)
% ADD_SEM_ERRORBARS - Vertical subsample SEM whiskers

if isempty(ySem) || isempty(xVals) || isempty(yVals)
    return;
end
nPlot = min([numel(xVals), numel(yVals), numel(ySem)]);
validMask = validMask(:);
validMask = validMask(1:nPlot);
xVals = xVals(1:nPlot);
yVals = yVals(1:nPlot);
ySem = ySem(1:nPlot);
semMask = validMask & isfinite(xVals) & isfinite(yVals) & isfinite(ySem);
if ~any(semMask)
    return;
end
errorbar(ax, xVals(semMask), yVals(semMask), ySem(semMask), ...
    'LineStyle', 'none', 'Marker', 'none', 'Color', barColor, ...
    'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
    'HandleVisibility', 'off');
end

function add_regression_line(ax, xVals, yVals, plotConfig)
% ADD_REGRESSION_LINE - Least-squares line through a d2 pair

if numel(xVals) < 2
    return;
end
slope = polyfit(xVals(:), yVals(:), 1);
xLine = [min(xVals), max(xVals)];
yLine = polyval(slope, xLine);
plot(ax, xLine, yLine, '-', 'Color', [0 0 0], ...
    'LineWidth', plotConfig.lineWidth, 'HandleVisibility', 'off');
end

function rVal = pearson_r(x, y)
% PEARSON_R - Pearson correlation, NaN when undefined

rVal = nan;
if numel(x) < 2 || numel(y) < 2
    return;
end
cMat = corrcoef(x(:), y(:));
rVal = cMat(1, 2);
end

function [axisLabel, titleLabel, labelInterpreter] = get_d2_plot_labels(useLog10D2, d2Method)
% GET_D2_PLOT_LABELS - Axis and title text for Euclidean vs KL d2

if nargin < 2 || isempty(d2Method)
    d2Method = 'euclidean';
end
d2Method = lower(strtrim(char(d2Method)));
if strcmp(d2Method, 'kl')
    methodParen = 'kl; bits/sec';
else
    methodParen = 'euclidean';
end
if useLog10D2
    axisLabel = sprintf('log_{10}(d2) (%s)', methodParen);
    titleLabel = sprintf('log10(d2) (%s)', methodParen);
    labelInterpreter = 'tex';
else
    axisLabel = sprintf('d2 (%s)', methodParen);
    titleLabel = axisLabel;
    labelInterpreter = 'none';
end
end

function alignMode = normalize_d2_window_align(d2WindowAlign)
% NORMALIZE_D2_WINDOW_ALIGN - Canonical 'center' or 'leadingEdge'

if nargin < 1 || isempty(d2WindowAlign)
    alignMode = 'center';
    return;
end
key = lower(strtrim(char(d2WindowAlign)));
key = strrep(key, '-', '');
key = strrep(key, '_', '');
key = strrep(key, ' ', '');
switch key
    case {'center', 'centre', 'mid', 'middle'}
        alignMode = 'center';
    case {'leadingedge', 'leading', 'lead', 'end', 'trailing'}
        alignMode = 'leadingEdge';
    otherwise
        error('session_d2_distributions_across_areas:BadD2WindowAlign', ...
            'd2WindowAlign must be ''center'' or ''leadingEdge'' (got %s).', d2WindowAlign);
end
end

function tAlign = d2_window_align_times(startS, d2Window, d2WindowAlign)
% D2_WINDOW_ALIGN_TIMES - Map window-center times to plot times

d2WindowAlign = normalize_d2_window_align(d2WindowAlign);
tAlign = startS(:);
if strcmp(d2WindowAlign, 'leadingEdge')
    tAlign = tAlign + d2Window / 2;
end
end

function tMax = resolve_timeline_tmax(results, collectStart, collectEnd, d2Window, dataStruct)
% RESOLVE_TIMELINE_TMAX - Right edge of the shared time axis

sessionEndAbs = loaded_session_end(dataStruct);
collectEnd = scalar_time(collectEnd, nan, @max);
sessionEndAbs = scalar_time(sessionEndAbs, nan, @max);
collectStart = scalar_time(collectStart, 0, @min);
if isfinite(collectEnd)
    tMax = collectEnd;
    if isfinite(sessionEndAbs)
        tMax = min(tMax, sessionEndAbs);
    end
elseif isfinite(sessionEndAbs)
    tMax = sessionEndAbs;
else
    tMax = nan;
    if isfield(results, 'startS')
        for iArea = 1:numel(results.startS)
            if ~isempty(results.startS{iArea})
                tMax = max(tMax, max(results.startS{iArea}(:)) + d2Window / 2);
            end
        end
    end
end
if ~isfinite(tMax) || tMax <= collectStart
    tMax = collectStart + 1;
end
end

function t = scalar_time(t, emptyDefault, reduceFcn)
% SCALAR_TIME - One finite time, or emptyDefault

if isempty(t)
    t = emptyDefault;
    return;
end
t = t(isfinite(t));
if isempty(t)
    t = emptyDefault;
else
    t = reduceFcn(t(:));
end
end

function sessionEnd = loaded_session_end(dataStruct)
% LOADED_SESSION_END - Absolute end time (s) of loaded spikes

sessionEnd = nan;
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
        && ~isempty(dataStruct.spikeData.collectEnd)
    sessionEnd = dataStruct.spikeData.collectEnd;
    return;
end
if isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
    sessionEnd = max(dataStruct.spikeTimes);
    return;
end
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
        && ~isempty(dataStruct.opts.collectEnd)
    sessionEnd = dataStruct.opts.collectEnd;
end
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive values

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end
