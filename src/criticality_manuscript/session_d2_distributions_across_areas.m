%%
% Session d2 across brain areas (Manuscript)
%
% Same windowed AR/d2 pipeline as session_d2_distributions.m. Correlates
% window-wise d2 between user-specified brain areas and plots:
%   1) pairwise d2 scatter (one point per shared window)
%   2) mean popActivity and d2 vs time, areas overlaid in area colors
% restrictToAwake drops spontaneous sleep windows (sessions without lfp.mat
% keep the whole recording). Other session types ignore that flag.
%
% Variables (configure in this section):
%   sessionType, sessionName, subjectName - Set in the workspace before running
%   areasToCorrelate - Cell of area names to correlate (at least two),
%                      e.g. {'M23','VS'} or {'M23','M56','DS','VS'}
%   dataSource, collectStart, collectEnd, d2Window, d2WindowAlign, stepSize
%   useLog10D2, useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple
%   enablePermutations, nPermutations, binSize, d2Method, kl* , runParallel, nWorkers
%   permuteCorrelations - If true, circularly shift window-mean d2 to test
%                       each pair's correlation and each difference between
%                       pair correlations. Independent of enablePermutations.
%   nShuffles          - Number of circular shifts for those tests
%   restrictToAwake - Spontaneous only. If true, tile d2 windows inside awake
%                     epochs from lfp_sleep_detector. The whole window must sit
%                     in one awake bout. No lfp.mat means the whole recording.
%   useRelativeTime, saveFigure, plotConfig
%   splitByEngagement - If true (reach/interval/semicircle), center-pack d2
%                       windows inside engaged and non-engaged segments.
%                       Correlation and timeline plots use open markers for
%                       engaged windows and filled markers for non-engaged.
%   engagementBufferBefore, engagementBufferAfter, minNonEngagedWindow,
%   absorbSingleEvents, minTimeNonEngaged - Same meaning as
%                       criticality_spontaneous_vs_task.m. Non-engaged windows
%                       stay computed but are omitted from plots when their
%                       total time is below minTimeNonEngaged (0 = no filter).
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
stepSize = 45;   % same window grid as session_d2_distributions.m
d2WindowAlign = 'center';  % 'center' | 'leadingEdge'
restrictToAwake = true;  % spontaneous: true drops sleep windows
useLog10D2 = false;
useSubsampling = true;
nSubsamples = 25;
nNeuronsSubsample = 60;
minNeuronsMultiple = 1.1;
enablePermutations = false;
nPermutations = 3;
permuteCorrelations = true;  % circular-shift tests of window-mean d2 correlations
nShuffles = 1000;            % shifts per test; not the spike-shuffle count
useRelativeTime = false;
binSize = 0.025;
binSize = 0.04;
d2Method = 'euclidean'; % euclidean or kl
klFitMethod = 'MaxLikelihood';
klErrBars = false;
runParallel = true;
nWorkers = 3;
klParallel = runParallel;
saveFigure = false;

% Engaged vs non-engaged (reach / interval / semicircle only)
splitByEngagement = false;
engagementBufferBefore = 3;  % s before each engagement event = engaged
engagementBufferAfter = 1;   % s after each engagement event = engaged
minNonEngagedWindow = 30;    % min gap (s) for non-engaged segments
absorbSingleEvents = true;   % merge isolated single events into non-engaged gaps
minTimeNonEngaged = 180;     % min total non-engaged time (s) to plot; 0 = no filter

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
analysisConfig.restrictToAwake = logical(restrictToAwake);

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
fprintf('restrictToAwake: %d (spontaneous sleep windows dropped when true and lfp.mat exists)\n', ...
    analysisConfig.restrictToAwake);
fprintf('d2Method: %s\n', d2Method);
fprintf('permuteCorrelations: %d\n', permuteCorrelations);
if permuteCorrelations
    if isempty(nShuffles) || ~isscalar(nShuffles) || ~isfinite(nShuffles) || nShuffles < 1
        error('session_d2_distributions_across_areas:BadNShuffles', ...
            'nShuffles must be a positive integer when permuteCorrelations is true.');
    end
    nShuffles = round(nShuffles);
    fprintf('nShuffles: %d (circular shifts of window-mean d2)\n', nShuffles);
end
fprintf('splitByEngagement: %d\n', splitByEngagement);
if splitByEngagement
    if ~is_manuscript_engagement_session_type(sessionType)
        error('session_d2_distributions_across_areas:BadEngagementType', ...
            ['splitByEngagement requires sessionType interval, reach, or semicircle ', ...
            '(got %s).'], sessionType);
    end
    fprintf(['engagementBuffer: before=%.3g s, after=%.3g s; ', ...
        'minNonEngagedWindow: %.1f s; absorbSingleEvents: %d\n'], ...
        engagementBufferBefore, engagementBufferAfter, minNonEngagedWindow, absorbSingleEvents);
    fprintf('minTimeNonEngaged: %.1f s (blank non-engaged below this; 0 = off)\n', ...
        minTimeNonEngaged);
    if abs(stepSize - d2Window) > 1e-6
        fprintf(['splitByEngagement center-packs non-overlapping %.0f s windows ', ...
            '(stepSize %.0f s is not used).\n'], d2Window, stepSize);
    end
end
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

if splitByEngagement
    engagementCollectEnd = collectEnd;
    if isempty(engagementCollectEnd)
        engagementCollectEnd = loaded_session_end(dataStruct);
    end
    if ~(isscalar(engagementCollectEnd) && isfinite(engagementCollectEnd) ...
            && engagementCollectEnd > collectStart)
        error('session_d2_distributions_across_areas:NoCollectEnd', ...
            'Could not resolve a collect end for engagement segments.');
    end
    eventTimes = engagement_event_times(dataStruct, sessionType, subjectNameForLoad, ...
        collectStart, engagementCollectEnd);
    if isempty(eventTimes)
        error('session_d2_distributions_across_areas:NoEngagementEvents', ...
            'No engagement events in the collect window.');
    end
    [engagedSegs, nonEngagedSegs] = define_engagement_segments( ...
        collectStart, engagementCollectEnd, eventTimes, minNonEngagedWindow, ...
        engagementBufferBefore, engagementBufferAfter, absorbSingleEvents);
    windowCenters = [ ...
        center_pack_windows_in_segments(engagedSegs, d2Window), ...
        center_pack_windows_in_segments(nonEngagedSegs, d2Window)];
    windowCenters = unique(windowCenters, 'sorted');
    if isempty(windowCenters)
        error('session_d2_distributions_across_areas:NoEngagementWindows', ...
            'No %.1f s d2 windows fit inside engaged or non-engaged segments.', d2Window);
    end
    fprintf(['Center-packed d2 windows: %d (%.1f s), placed from the middle ', ...
        'of each engagement segment\n'], numel(windowCenters), d2Window);
    analysisConfig.windowCenterTimes = windowCenters;
    analysisConfig.stepSize = d2Window;
end

results = criticality_ar_analysis(dataStruct, analysisConfig);
[results, keptAreas] = keep_correlate_areas(results, areasToCorrelate);
if numel(keptAreas) < 2
    error('Need at least two areas with d2. Kept: %s', strjoin(keptAreas, ', '));
end
if splitByEngagement
    results = label_d2_windows_by_engagement(results, engagedSegs, nonEngagedSegs, ...
        d2Window, minTimeNonEngaged, sessionName);
end

[d2AxisLabel, d2TitleLabel, d2LabelInterpreter] = get_d2_plot_labels(useLog10D2, d2Method);

correlationTests = [];
if permuteCorrelations
    correlationTests = permute_d2_area_correlations( ...
        results, keptAreas, useLog10D2, nShuffles);
end

figCorr = plot_d2_area_correlation_scatters(results, keptAreas, useLog10D2, ...
    d2AxisLabel, d2LabelInterpreter, plotConfig, sessionName, d2Window, ...
    correlationTests);
print_d2_area_correlations(results, keptAreas, useLog10D2);
if permuteCorrelations
    print_d2_correlation_permutations(correlationTests);
end

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
    engTag = '';
    if splitByEngagement
        engTag = '_eng';
    end
    permTag = '';
    if permuteCorrelations
        permTag = '_perm';
    end
    awakeTag = '';
    if restrictToAwake
        awakeTag = '_awake';
    end
    corrBase = sprintf('session_d2_across_areas_corr_%s_%s_win%.0fs_%ss%s%s%s%s%s', ...
        sessionName, areaTag, d2Window, collectTag, engTag, permTag, awakeTag, logTag, d2PlotTag);
    timeBase = sprintf('session_d2_across_areas_timeline_%s_%s_win%.0fs_%ss_%s%s%s%s%s', ...
        sessionName, areaTag, d2Window, collectTag, d2WindowAlign, engTag, awakeTag, logTag, d2PlotTag);
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
    d2AxisLabel, d2LabelInterpreter, plotConfig, sessionName, d2Window, ...
    correlationTests)
% PLOT_D2_AREA_CORRELATION_SCATTERS - Pairwise window d2 scatter
%
% Variables:
%   results, keptAreas - Areas in plot order, already filtered
%   useLog10D2         - Apply log10 before correlating
%   d2AxisLabel        - Shared axis label text
%   plotConfig         - Fonts and marker sizes
%   correlationTests   - Output of permute_d2_area_correlations, or []
%
% Goal:
%   One scatter per unique pair. Each point is one window present in both areas.
%   Every panel uses the same x and y limits, spanning all plotted d2 values.
%   When splitByEngagement, engaged windows are open markers and non-engaged
%   windows are filled. Each class gets its own correlation.

nAreas = numel(keptAreas);
pairIdx = nchoosek(1:nAreas, 2);
nPairs = size(pairIdx, 1);
nCols = ceil(sqrt(nPairs));
nRows = ceil(nPairs / nCols);
sharedLim = shared_d2_axis_limits(results, nAreas, useLog10D2);
splitEngagement = results_split_by_engagement(results);
engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];

fig = figure('Color', 'w', 'Name', sprintf('d2 area correlation — %s', sessionName), ...
    'Position', [80 80 max(520, 380 * nCols) max(420, 340 * nRows)]);

for iPair = 1:nPairs
    idxX = pairIdx(iPair, 1);
    idxY = pairIdx(iPair, 2);
    [xVals, yVals, classLabels] = paired_window_d2(results, idxX, idxY, useLog10D2);
    ax = subplot(nRows, nCols, iPair, 'Parent', fig);
    hold(ax, 'on');
    colorX = area_plot_color(keptAreas{idxX});
    colorY = area_plot_color(keptAreas{idxY});
    if isempty(xVals)
        text(ax, 0.5, 0.5, 'no shared windows', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'Color', [0.45 0.45 0.45]);
        titleText = 'r = NaN   n = 0';
    elseif splitEngagement
        engagedMask = strcmp(classLabels, 'engaged');
        nonEngagedMask = strcmp(classLabels, 'nonEngaged');
        if any(nonEngagedMask)
            scatter(ax, xVals(nonEngagedMask), yVals(nonEngagedMask), ...
                plotConfig.scatterMarkerSize, nonEngagedColor, 'filled', ...
                'MarkerFaceAlpha', plotConfig.markerFaceAlpha, ...
                'MarkerEdgeColor', 'none', 'DisplayName', 'Non-engaged');
            add_regression_line(ax, xVals(nonEngagedMask), yVals(nonEngagedMask), ...
                plotConfig, nonEngagedColor);
        end
        if any(engagedMask)
            scatter(ax, xVals(engagedMask), yVals(engagedMask), ...
                plotConfig.scatterMarkerSize, 'MarkerFaceColor', 'none', ...
                'MarkerEdgeColor', engagedColor, 'LineWidth', plotConfig.lineWidth, ...
                'DisplayName', 'Engaged');
            add_regression_line(ax, xVals(engagedMask), yVals(engagedMask), ...
                plotConfig, engagedColor);
        end
        pEngaged = lookup_pair_permutation_p(correlationTests, 'engaged', ...
            keptAreas{idxX}, keptAreas{idxY});
        pNonEngaged = lookup_pair_permutation_p(correlationTests, 'nonEngaged', ...
            keptAreas{idxX}, keptAreas{idxY});
        titleText = sprintf('eng r = %.3f (n=%d%s)   non r = %.3f (n=%d%s)', ...
            pearson_r(xVals(engagedMask), yVals(engagedMask)), sum(engagedMask), ...
            format_p_clause(pEngaged), ...
            pearson_r(xVals(nonEngagedMask), yVals(nonEngagedMask)), sum(nonEngagedMask), ...
            format_p_clause(pNonEngaged));
        if iPair == 1
            legend(ax, 'Location', 'best', 'FontSize', plotConfig.legendFontSize, ...
                'Interpreter', 'none');
        end
    else
        scatter(ax, xVals, yVals, plotConfig.scatterMarkerSize, [0.2 0.2 0.2], ...
            'filled', 'MarkerFaceAlpha', plotConfig.markerFaceAlpha, ...
            'MarkerEdgeColor', 'none');
        add_regression_line(ax, xVals, yVals, plotConfig, [0 0 0]);
        pPair = lookup_pair_permutation_p(correlationTests, 'all', ...
            keptAreas{idxX}, keptAreas{idxY});
        titleText = sprintf('r = %.3f   n = %d%s', pearson_r(xVals, yVals), ...
            numel(xVals), format_p_clause(pPair));
    end
    xlabel(ax, sprintf('%s %s', keptAreas{idxX}, d2AxisLabel), ...
        'Color', colorX, 'FontSize', plotConfig.axisLabelFontSize, ...
        'Interpreter', d2LabelInterpreter);
    ylabel(ax, sprintf('%s %s', keptAreas{idxY}, d2AxisLabel), ...
        'Color', colorY, 'FontSize', plotConfig.axisLabelFontSize, ...
        'Interpreter', d2LabelInterpreter);
    title(ax, titleText, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
    if ~isempty(sharedLim)
        xlim(ax, sharedLim);
        ylim(ax, sharedLim);
        axis(ax, 'square');
    end
    set(ax, 'Box', 'off', 'TickDir', 'out', ...
        'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
    hold(ax, 'off');
end

titleStr = sprintf('%s | d2 across areas (%.0f s windows)', sessionName, d2Window);
if splitEngagement
    titleStr = sprintf('%s | open = engaged, filled = non-engaged', titleStr);
end
sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, ...
    'FontWeight', 'bold', 'Interpreter', 'none');
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
    includeMask = windows_included_in_plots(results, iArea, numel(d2Vec));
    d2Vec = d2Vec(includeMask);
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
%
% Goal:
%   With splitByEngagement, print engaged and non-engaged correlations separately.

fprintf('\n=== Window-wise d2 correlations ===\n');
nAreas = numel(keptAreas);
splitEngagement = results_split_by_engagement(results);
for idxX = 1:nAreas
    for idxY = idxX + 1:nAreas
        [xVals, yVals, classLabels] = paired_window_d2(results, idxX, idxY, useLog10D2);
        if splitEngagement
            engagedMask = strcmp(classLabels, 'engaged');
            nonEngagedMask = strcmp(classLabels, 'nonEngaged');
            fprintf('  %s vs %s engaged: r = %.3f, n = %d\n', ...
                keptAreas{idxX}, keptAreas{idxY}, ...
                pearson_r(xVals(engagedMask), yVals(engagedMask)), sum(engagedMask));
            fprintf('  %s vs %s non-engaged: r = %.3f, n = %d\n', ...
                keptAreas{idxX}, keptAreas{idxY}, ...
                pearson_r(xVals(nonEngagedMask), yVals(nonEngagedMask)), sum(nonEngagedMask));
        else
            fprintf('  %s vs %s: r = %.3f, n = %d\n', ...
                keptAreas{idxX}, keptAreas{idxY}, pearson_r(xVals, yVals), numel(xVals));
        end
    end
end
end

function [xVals, yVals, classLabels] = paired_window_d2(results, idxX, idxY, useLog10D2)
% PAIRED_WINDOW_D2 - Finite d2 pairs aligned on window center time
%
% Variables:
%   results    - AR results with startS and d2 per area
%   idxX, idxY - Area indices
%   useLog10D2 - log10 before pairing
%
% Goal:
%   Match windows by startS so each pair is the same analysis window.
%   classLabels is 'engaged', 'nonEngaged', or '' for each returned pair.
%   Windows blanked by minTimeNonEngaged are omitted.

xVals = [];
yVals = [];
classLabels = {};
d2X = area_d2_vector(results, idxX, useLog10D2);
d2Y = area_d2_vector(results, idxY, useLog10D2);
tX = area_start_times(results, idxX);
tY = area_start_times(results, idxY);
classX = window_class_labels(results, idxX);
if isempty(d2X) || isempty(d2Y) || isempty(tX) || isempty(tY)
    return;
end

nX = min(numel(d2X), numel(tX));
nY = min(numel(d2Y), numel(tY));
d2X = d2X(1:nX);
tX = tX(1:nX);
d2Y = d2Y(1:nY);
tY = tY(1:nY);
includeX = windows_included_in_plots(results, idxX, nX);
includeY = windows_included_in_plots(results, idxY, nY);
if numel(classX) < nX
    classX = [classX(:); repmat({''}, nX - numel(classX), 1)];
else
    classX = classX(1:nX);
end

if nX == nY && max(abs(tX - tY)) < 1e-6
    validMask = isfinite(d2X) & isfinite(d2Y) & includeX & includeY;
    xVals = d2X(validMask);
    yVals = d2Y(validMask);
    classLabels = classX(validMask);
    return;
end

matchTol = 1e-3;
[tCommon, idxInX, idxInY] = intersect_window_times(tX, tY, matchTol);
if isempty(tCommon)
    return;
end
d2X = d2X(idxInX);
d2Y = d2Y(idxInY);
includeX = includeX(idxInX);
includeY = includeY(idxInY);
classX = classX(idxInX);
validMask = isfinite(d2X) & isfinite(d2Y) & includeX & includeY;
xVals = d2X(validMask);
yVals = d2Y(validMask);
classLabels = classX(validMask);
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
%   When splitByEngagement, open markers are engaged windows and filled markers
%   are non-engaged. Segment spans are shaded behind both axes.

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
    splitEngagement = results_split_by_engagement(results);
    if ~isempty(popVec) && ~isempty(tWin)
        nPlot = min(numel(popVec), numel(tWin));
        tPop = tWin(1:nPlot);
        popPlot = popVec(1:nPlot);
        includeMask = windows_included_in_plots(results, iArea, nPlot);
        if splitEngagement
            plot(axPop, tPop(includeMask), popPlot(includeMask), '-', ...
                'Color', areaColor, 'LineWidth', plotConfig.axesLineWidth, ...
                'DisplayName', areaName);
            plot_engagement_time_markers(axPop, tPop, popPlot, ...
                window_class_labels(results, iArea), includeMask, areaColor, plotConfig);
        else
            plot(axPop, tPop, popPlot, '-o', ...
                'Color', areaColor, 'MarkerFaceColor', areaColor, ...
                'MarkerSize', 4, 'LineWidth', plotConfig.axesLineWidth, ...
                'DisplayName', areaName);
        end
    end

    d2Vec = area_d2_vector(results, iArea, useLog10D2);
    if ~isempty(d2Vec) && ~isempty(tWin)
        nPlot = min(numel(d2Vec), numel(tWin));
        tD2 = tWin(1:nPlot);
        d2Vec = d2Vec(1:nPlot);
        includeMask = windows_included_in_plots(results, iArea, nPlot);
        finiteMask = isfinite(d2Vec) & isfinite(tD2) & includeMask;
        d2Sem = subsample_sem_d2(results, iArea, useLog10D2);
        add_sem_errorbars(axD2, tD2, d2Vec, d2Sem, finiteMask, areaColor, plotConfig);
        if splitEngagement
            plot(axD2, tD2(includeMask), d2Vec(includeMask), '-', ...
                'Color', areaColor, 'LineWidth', plotConfig.axesLineWidth, ...
                'HandleVisibility', 'off');
            plot_engagement_time_markers(axD2, tD2, d2Vec, ...
                window_class_labels(results, iArea), includeMask, areaColor, plotConfig);
        else
            plot(axD2, tD2, d2Vec, '-o', ...
                'Color', areaColor, 'MarkerFaceColor', areaColor, ...
                'MarkerSize', 4, 'LineWidth', plotConfig.axesLineWidth, ...
                'DisplayName', areaName);
        end
    end
end

if results_split_by_engagement(results)
    shade_engagement_segments(axPop, results, timeShift);
    shade_engagement_segments(axD2, results, timeShift);
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

sgtitle(fig, sprintf('%s | mean pop / %s (%.0fs windows, %s, bin=%.0f ms)%s', ...
    sessionName, d2TitleLabel, d2Window, d2WindowAlign, binSize * 1000, ...
    engagement_timeline_title_suffix(results)), ...
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

function add_regression_line(ax, xVals, yVals, plotConfig, lineColor)
% ADD_REGRESSION_LINE - Least-squares line through a d2 pair
%
% Variables:
%   lineColor - RGB for the fit. Defaults to black.

if nargin < 5 || isempty(lineColor)
    lineColor = [0 0 0];
end
if numel(xVals) < 2
    return;
end
slope = polyfit(xVals(:), yVals(:), 1);
xLine = [min(xVals), max(xVals)];
yLine = polyval(slope, xLine);
plot(ax, xLine, yLine, '-', 'Color', lineColor, ...
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

function eventTimes = engagement_event_times(dataStruct, sessionType, subjectName, ...
    collectStart, collectEnd)
% ENGAGEMENT_EVENT_TIMES - Event times that define engaged neighborhoods
%
% Variables:
%   dataStruct   - Loaded session
%   sessionType  - 'reach', 'interval', or 'semicircle'
%   subjectName  - Subject folder for interval and semicircle logs
%   collectStart, collectEnd - Keep events inside this window (s)
%
% Goal:
%   Match the engagement modules: reach onsets, interval beam-break outcomes,
%   or semicircle TaskMatrix times.

sessionType = lower(strtrim(char(sessionType)));
if isempty(subjectName) && isfield(dataStruct, 'subjectName')
    subjectName = dataStruct.subjectName;
end
paths = get_paths();
switch sessionType
    case 'reach'
        if ~isfield(dataStruct, 'reachStart') || isempty(dataStruct.reachStart)
            error('session_d2_distributions_across_areas:NoReachStarts', ...
                'Reach session has no reachStart times.');
        end
        eventTimes = dataStruct.reachStart(:);
    case 'interval'
        if isempty(subjectName)
            error('session_d2_distributions_across_areas:NoSubject', ...
                'subjectName is required to load interval engagement events.');
        end
        eventTimes = load_interval_beam_break_events( ...
            paths, subjectName, dataStruct.sessionName, 0.1);
    case 'semicircle'
        if isempty(subjectName)
            error('session_d2_distributions_across_areas:NoSubject', ...
                'subjectName is required to load semicircle engagement events.');
        end
        eventTimes = load_semicircle_beam_break_events( ...
            paths, subjectName, dataStruct.sessionName);
    otherwise
        error('session_d2_distributions_across_areas:BadEngagementType', ...
            'splitByEngagement requires interval, reach, or semicircle (got %s).', ...
            sessionType);
end
eventTimes = eventTimes(isfinite(eventTimes));
eventTimes = eventTimes(eventTimes >= collectStart & eventTimes <= collectEnd);
eventTimes = unique(eventTimes(:))';
end

function results = label_d2_windows_by_engagement(results, engagedSegs, nonEngagedSegs, ...
    d2Window, minTimeNonEngaged, sessionName)
% LABEL_D2_WINDOWS_BY_ENGAGEMENT - Mark each window engaged or non-engaged
%
% Variables:
%   results            - AR results after area filtering
%   engagedSegs        - Struct array with .start and .end (s)
%   nonEngagedSegs     - Struct array with .start and .end (s)
%   d2Window           - Window length (s); startS is the window center
%   minTimeNonEngaged  - Blank non-engaged plots below this covered time (s)
%   sessionName        - Used in the blanking log line
%
% Goal:
%   A center-packed window is engaged or non-engaged only when it lies fully
%   inside one of those segments. Covered non-engaged time is nWindows * d2Window.

if nargin < 6 || isempty(sessionName)
    sessionName = '';
end

nAreas = numel(results.areas);
results.splitByEngagement = true;
results.engagedSegs = engagedSegs;
results.nonEngagedSegs = nonEngagedSegs;
results.engagementClass = cell(1, nAreas);
results.blankNonEngaged = false;

nEngaged = 0;
nNonEngaged = 0;
countedWindows = false;
halfWindow = d2Window / 2;
edgeTol = 1e-4;
for iArea = 1:nAreas
    tCenter = area_start_times(results, iArea);
    classLabels = repmat({''}, numel(tCenter), 1);
    if ~isempty(tCenter)
        winStart = tCenter - halfWindow + edgeTol;
        winEnd = tCenter + halfWindow - edgeTol;
        engagedMask = window_fully_inside_segments(winStart, winEnd, engagedSegs);
        nonEngagedMask = window_fully_inside_segments(winStart, winEnd, nonEngagedSegs) ...
            & ~engagedMask;
        classLabels(engagedMask) = {'engaged'};
        classLabels(nonEngagedMask) = {'nonEngaged'};
        if ~countedWindows
            nEngaged = sum(engagedMask);
            nNonEngaged = sum(nonEngagedMask);
            countedWindows = true;
        end
    end
    results.engagementClass{iArea} = classLabels;
end

results.engagedSec = nEngaged * d2Window;
results.nonEngagedSec = nNonEngaged * d2Window;
fprintf('  Engagement windows: engaged=%d (%.1f s), non-engaged=%d (%.1f s)\n', ...
    nEngaged, results.engagedSec, nNonEngaged, results.nonEngagedSec);

if isfinite(minTimeNonEngaged) && minTimeNonEngaged > 0 ...
        && results.nonEngagedSec < minTimeNonEngaged
    results.blankNonEngaged = true;
    fprintf(['  Blanking non-engaged %s: non-engaged time %.1f s ', ...
        '< minTimeNonEngaged %.1f s\n'], sessionName, results.nonEngagedSec, ...
        minTimeNonEngaged);
end
end

function isInside = window_fully_inside_segments(winStart, winEnd, segments)
% WINDOW_FULLY_INSIDE_SEGMENTS - True when [winStart, winEnd] lies in one segment

isInside = false(size(winStart));
if isempty(segments)
    return;
end
segStarts = [segments.start];
segEnds = [segments.end];
for iWin = 1:numel(winStart)
    isInside(iWin) = any(winStart(iWin) >= segStarts & winEnd(iWin) <= segEnds);
end
end

function tf = results_split_by_engagement(results)
% RESULTS_SPLIT_BY_ENGAGEMENT - True when windows were labeled by engagement

tf = isfield(results, 'splitByEngagement') && logical(results.splitByEngagement);
end

function classLabels = window_class_labels(results, areaIdx)
% WINDOW_CLASS_LABELS - 'engaged', 'nonEngaged', or '' per window

classLabels = {};
if ~isfield(results, 'engagementClass') || areaIdx > numel(results.engagementClass)
    return;
end
classLabels = results.engagementClass{areaIdx};
if isstring(classLabels)
    classLabels = cellstr(classLabels);
end
classLabels = classLabels(:);
end

function includeMask = windows_included_in_plots(results, areaIdx, nWindows)
% WINDOWS_INCLUDED_IN_PLOTS - Windows drawn after the non-engaged time filter
%
% Goal:
%   Without an engagement split, every window is included. When non-engaged
%   time is below minTimeNonEngaged, those windows stay in results but drop
%   out of correlations and the timeline.

includeMask = true(nWindows, 1);
if nWindows == 0 || ~results_split_by_engagement(results)
    return;
end
classLabels = window_class_labels(results, areaIdx);
blankNonEngaged = isfield(results, 'blankNonEngaged') && results.blankNonEngaged;
nUse = min(nWindows, numel(classLabels));
includeMask = false(nWindows, 1);
for iWin = 1:nUse
    label = classLabels{iWin};
    if strcmp(label, 'engaged') || (strcmp(label, 'nonEngaged') && ~blankNonEngaged)
        includeMask(iWin) = true;
    end
end
end

function plot_engagement_time_markers(ax, tVals, yVals, classLabels, includeMask, ...
    areaColor, plotConfig)
% PLOT_ENGAGEMENT_TIME_MARKERS - Open engaged and filled non-engaged markers
%
% Variables:
%   tVals, yVals   - Aligned window times and values
%   classLabels    - Per-window 'engaged' / 'nonEngaged'
%   includeMask    - Windows that pass the non-engaged time filter
%   areaColor      - Marker color for this brain area
%
% Goal:
%   Same open/filled convention as the across-task engagement plots, colored
%   by area. Markers are omitted from the legend; the area line carries the name.

if isempty(tVals) || isempty(yVals)
    return;
end
nPlot = min(numel(tVals), numel(yVals));
tVals = tVals(1:nPlot);
yVals = yVals(1:nPlot);
includeMask = includeMask(:);
includeMask = includeMask(1:min(nPlot, numel(includeMask)));
if numel(includeMask) < nPlot
    includeMask(end + 1:nPlot) = false;
end
if numel(classLabels) < nPlot
    classLabels = [classLabels(:); repmat({''}, nPlot - numel(classLabels), 1)];
else
    classLabels = classLabels(1:nPlot);
end
engagedMask = includeMask & strcmp(classLabels, 'engaged') & isfinite(tVals) & isfinite(yVals);
nonEngagedMask = includeMask & strcmp(classLabels, 'nonEngaged') & isfinite(tVals) & isfinite(yVals);
markerSize = 5;
if isfield(plotConfig, 'markerSize') && ~isempty(plotConfig.markerSize)
    markerSize = plotConfig.markerSize;
end
if any(nonEngagedMask)
    plot(ax, tVals(nonEngagedMask), yVals(nonEngagedMask), 'o', ...
        'Color', areaColor, 'MarkerFaceColor', areaColor, ...
        'MarkerSize', markerSize, 'LineWidth', plotConfig.axesLineWidth, ...
        'HandleVisibility', 'off');
end
if any(engagedMask)
    plot(ax, tVals(engagedMask), yVals(engagedMask), 'o', ...
        'Color', areaColor, 'MarkerFaceColor', 'none', ...
        'MarkerSize', markerSize, 'LineWidth', plotConfig.axesLineWidth, ...
        'HandleVisibility', 'off');
end
end

function shade_engagement_segments(ax, results, timeShift)
% SHADE_ENGAGEMENT_SEGMENTS - Light engaged and non-engaged spans behind data
%
% Variables:
%   ax        - Target axes
%   results   - Must contain engagedSegs and nonEngagedSegs
%   timeShift - Subtracted from absolute segment times (0 when plotting absolute time)
%
% Goal:
%   Show which parts of the session are engaged without covering the traces.

if ~isfield(results, 'engagedSegs')
    return;
end
engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];
yLim = ylim(ax);
blankNonEngaged = isfield(results, 'blankNonEngaged') && results.blankNonEngaged;
add_segment_patches(ax, results.engagedSegs, engagedColor, yLim, timeShift);
if ~blankNonEngaged
    add_segment_patches(ax, results.nonEngagedSegs, nonEngagedColor, yLim, timeShift);
end
ylim(ax, yLim);
end

function add_segment_patches(ax, segments, faceColor, yLim, timeShift)
% ADD_SEGMENT_PATCHES - Translucent [start, end] spans on one axes

if isempty(segments)
    return;
end
for iSeg = 1:numel(segments)
    t0 = segments(iSeg).start - timeShift;
    t1 = segments(iSeg).end - timeShift;
    if ~(isfinite(t0) && isfinite(t1)) || t1 <= t0
        continue;
    end
    hPatch = patch(ax, [t0, t1, t1, t0], [yLim(1), yLim(1), yLim(2), yLim(2)], faceColor, ...
        'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(hPatch, 'bottom');
end
end

function suffix = engagement_timeline_title_suffix(results)
% ENGAGEMENT_TIMELINE_TITLE_SUFFIX - Marker key appended to the timeline title

suffix = '';
if results_split_by_engagement(results)
    suffix = ' | open = engaged, filled = non-engaged';
end
end

function tests = permute_d2_area_correlations(results, areaNames, useLog10D2, nShuffles)
% PERMUTE_D2_AREA_CORRELATIONS - Circular-shift tests of window-mean d2 correlations
%
% Variables:
%   results     - AR results after area filtering (and engagement labels)
%   areaNames   - Area names in column order
%   useLog10D2  - Apply the same transform used in the scatters
%   nShuffles   - Number of circular shifts
%
% Goal:
%   Test each pair's Pearson r by shifting one area's window-mean d2, and
%   test each difference between pair correlations by shifting every area
%   independently. Engagement classes are tested separately. p-values are
%   two-sided and use the +1 correction, so the smallest p is 1/(nShuffles+1).
%
% Returns:
%   tests - tests.nShuffles and tests.classes, with .pairs and .differences

[d2Mat, rowClass] = aligned_area_d2_matrix(results, numel(areaNames), useLog10D2);
if results_split_by_engagement(results)
    classNames = {'engaged', 'nonEngaged'};
else
    classNames = {'all'};
end

classList = cell(1, numel(classNames));
for iClass = 1:numel(classNames)
    className = classNames{iClass};
    if strcmp(className, 'all')
        rowMask = any(isfinite(d2Mat), 2);
    else
        rowMask = strcmp(rowClass, className);
    end
    classList{iClass} = permute_one_d2_class( ...
        d2Mat(rowMask, :), areaNames, nShuffles, className);
end

tests = struct();
tests.nShuffles = nShuffles;
tests.classes = vertcat(classList{:});
end

function classResult = permute_one_d2_class(d2Mat, areaNames, nShuffles, className)
% PERMUTE_ONE_D2_CLASS - Within-pair and between-pair shift tests for one window set
%
% Variables:
%   d2Mat      - Window-mean d2, rows are time-ordered windows, columns are areas
%   areaNames  - Column names
%   nShuffles  - Number of circular shifts
%   className  - 'all', 'engaged', or 'nonEngaged'
%
% Goal:
%   Within a pair, shift only the second area by a nonzero offset. Between
%   pairs, shift every area by its own offset and record r1 - r2.

nWindows = size(d2Mat, 1);
nAreas = size(d2Mat, 2);
if nAreas >= 2
    pairIdx = nchoosek(1:nAreas, 2);
else
    pairIdx = zeros(0, 2);
end
nPairs = size(pairIdx, 1);
[rObs, nObserved] = pairwise_window_correlations(d2Mat, pairIdx);

pairLabels = cell(nPairs, 1);
pairs = repmat(struct('areaX', '', 'areaY', '', 'label', '', 'r', nan, 'n', 0, 'p', nan), ...
    nPairs, 1);
for iPair = 1:nPairs
    idxX = pairIdx(iPair, 1);
    idxY = pairIdx(iPair, 2);
    pairs(iPair).areaX = areaNames{idxX};
    pairs(iPair).areaY = areaNames{idxY};
    pairs(iPair).label = sprintf('%s vs %s', areaNames{idxX}, areaNames{idxY});
    pairs(iPair).r = rObs(iPair);
    pairs(iPair).n = nObserved(iPair);
    pairLabels{iPair} = pairs(iPair).label;
end

nullR = nan(nShuffles, nPairs);
canShift = nWindows >= 3 && nShuffles >= 1 && nPairs >= 1;
if canShift
    pairShifts = randi(nWindows - 1, nShuffles, nPairs);
    for iShuffle = 1:nShuffles
        for iPair = 1:nPairs
            yShift = circshift(d2Mat(:, pairIdx(iPair, 2)), pairShifts(iShuffle, iPair));
            nullR(iShuffle, iPair) = pearson_r(d2Mat(:, pairIdx(iPair, 1)), yShift);
        end
    end
end
pPair = permutation_p_values(rObs, nullR);
tooFew = nObserved < 3 | ~isfinite(rObs);
pPair(tooFew) = nan;
for iPair = 1:nPairs
    pairs(iPair).p = pPair(iPair);
end

if nPairs >= 2
    diffIdx = nchoosek(1:nPairs, 2);
else
    diffIdx = zeros(0, 2);
end
nDiff = size(diffIdx, 1);
deltaObs = nan(nDiff, 1);
if nDiff > 0
    deltaObs = rObs(diffIdx(:, 1)) - rObs(diffIdx(:, 2));
end
nullDeltaR = nan(nShuffles, nDiff);
if canShift && nDiff > 0
    for iShuffle = 1:nShuffles
        areaShifts = randi(nWindows, 1, nAreas) - 1;
        shifted = d2Mat;
        for iArea = 1:nAreas
            shifted(:, iArea) = circshift(d2Mat(:, iArea), areaShifts(iArea));
        end
        rPerm = pairwise_window_correlations(shifted, pairIdx);
        nullDeltaR(iShuffle, :) = (rPerm(diffIdx(:, 1)) - rPerm(diffIdx(:, 2)))';
    end
end
pDiff = permutation_p_values(deltaObs, nullDeltaR);
if nDiff > 0
    validDiff = nObserved(diffIdx(:, 1)) >= 3 & nObserved(diffIdx(:, 2)) >= 3 ...
        & isfinite(deltaObs);
    pDiff(~validDiff) = nan;
end

differences = repmat(struct( ...
    'pairLabel1', '', 'pairLabel2', '', 'deltaR', nan, 'p', nan), nDiff, 1);
for iDiff = 1:nDiff
    differences(iDiff).pairLabel1 = pairLabels{diffIdx(iDiff, 1)};
    differences(iDiff).pairLabel2 = pairLabels{diffIdx(iDiff, 2)};
    differences(iDiff).deltaR = deltaObs(iDiff);
    differences(iDiff).p = pDiff(iDiff);
end

classResult = struct();
classResult.name = className;
classResult.nWindows = nWindows;
classResult.pairs = pairs;
classResult.differences = differences;
classResult.nullR = nullR;
classResult.nullDeltaR = nullDeltaR;
end

function [rVals, nVals] = pairwise_window_correlations(d2Mat, pairIdx)
% PAIRWISE_WINDOW_CORRELATIONS - Pearson r and pairwise-complete n for each pair

nPairs = size(pairIdx, 1);
rVals = nan(nPairs, 1);
nVals = zeros(nPairs, 1);
for iPair = 1:nPairs
    xVals = d2Mat(:, pairIdx(iPair, 1));
    yVals = d2Mat(:, pairIdx(iPair, 2));
    nVals(iPair) = sum(isfinite(xVals) & isfinite(yVals));
    rVals(iPair) = pearson_r(xVals, yVals);
end
end

function [d2Mat, rowClass] = aligned_area_d2_matrix(results, nAreas, useLog10D2)
% ALIGNED_AREA_D2_MATRIX - Time-ordered window-mean d2 with one column per area
%
% Variables:
%   results    - AR results with startS and d2
%   nAreas     - Number of kept areas
%   useLog10D2 - Match the scatter values
%
% Goal:
%   Put every area on one window clock so a circular shift moves whole windows.
%   Rows dropped from the plots (including blanked non-engaged windows) stay NaN.

matchTol = 1e-3;
timeKeys = [];
for iArea = 1:nAreas
    tArea = area_start_times(results, iArea);
    tArea = tArea(isfinite(tArea));
    timeKeys = [timeKeys; round(tArea(:) / matchTol)]; %#ok<AGROW>
end
uniqueKeys = unique(timeKeys, 'sorted');
nWindows = numel(uniqueKeys);
d2Mat = nan(nWindows, nAreas);
rowClass = repmat({''}, nWindows, 1);
if nWindows == 0
    return;
end

for iArea = 1:nAreas
    tArea = area_start_times(results, iArea);
    d2Vec = area_d2_vector(results, iArea, useLog10D2);
    nUse = min(numel(tArea), numel(d2Vec));
    if nUse == 0
        continue;
    end
    tArea = tArea(1:nUse);
    d2Vec = d2Vec(1:nUse);
    includeMask = windows_included_in_plots(results, iArea, nUse);
    classLabels = window_class_labels(results, iArea);
    [found, loc] = ismember(round(tArea / matchTol), uniqueKeys);
    valid = found & includeMask & isfinite(d2Vec);
    d2Mat(loc(valid), iArea) = d2Vec(valid);
    if isempty(classLabels)
        continue;
    end
    nClass = min(nUse, numel(classLabels));
    for iWin = find(valid(1:nClass))'
        if isempty(rowClass{loc(iWin)})
            rowClass{loc(iWin)} = classLabels{iWin};
        end
    end
end
end

function pVals = permutation_p_values(observed, nullSamples)
% PERMUTATION_P_VALUES - Two-sided permutation p with the +1 correction
%
% Variables:
%   observed    - Column of observed statistics
%   nullSamples - nShuffles by nStats null values
%
% Goal:
%   p = (nExtreme + 1) / (nFiniteNull + 1), counting null draws at least as
%   far from zero as the observed statistic.

observed = observed(:);
nStats = numel(observed);
pVals = nan(nStats, 1);
if isempty(nullSamples) || nStats == 0
    return;
end
for iStat = 1:nStats
    obs = observed(iStat);
    if iStat > size(nullSamples, 2) || ~isfinite(obs)
        continue;
    end
    nullVals = nullSamples(:, iStat);
    nullVals = nullVals(isfinite(nullVals));
    if isempty(nullVals)
        continue;
    end
    nExtreme = sum(abs(nullVals) >= abs(obs));
    pVals(iStat) = (nExtreme + 1) / (numel(nullVals) + 1);
end
end

function print_d2_correlation_permutations(correlationTests)
% PRINT_D2_CORRELATION_PERMUTATIONS - Within-pair r and between-pair differences
%
% Goal:
%   Report the circular-shift tests. Within-pair p values ask whether that
%   pair's r is large. Between-pair p values ask whether r of pair 1 minus r
%   of pair 2 is large. Both are two-sided.

if isempty(correlationTests) || ~isfield(correlationTests, 'classes')
    return;
end
fprintf(['\n=== d2 correlation permutations (%d circular shifts) ===\n', ...
    'Within pair: shift the second area. Between pairs: shift every area.\n'], ...
    correlationTests.nShuffles);
for iClass = 1:numel(correlationTests.classes)
    classResult = correlationTests.classes(iClass);
    fprintf('\n%s (%d windows)\n', classResult.name, classResult.nWindows);
    if classResult.nWindows < 3
        fprintf('  Too few windows to permute (need at least 3).\n');
    end
    fprintf('  Within pair:\n');
    if isempty(classResult.pairs)
        fprintf('    (none)\n');
    end
    for iPair = 1:numel(classResult.pairs)
        pairResult = classResult.pairs(iPair);
        fprintf('    %s: r = %.3f, n = %d, p = %.3f\n', ...
            pairResult.label, pairResult.r, pairResult.n, pairResult.p);
    end
    fprintf('  Between pairs (r1 - r2):\n');
    if isempty(classResult.differences)
        fprintf('    Need at least three areas.\n');
    end
    for iDiff = 1:numel(classResult.differences)
        diffResult = classResult.differences(iDiff);
        fprintf('    (%s) - (%s): delta r = %.3f, p = %.3f\n', ...
            diffResult.pairLabel1, diffResult.pairLabel2, ...
            diffResult.deltaR, diffResult.p);
    end
end
end

function pVal = lookup_pair_permutation_p(correlationTests, className, areaX, areaY)
% LOOKUP_PAIR_PERMUTATION_P - Within-pair p for one area pair and window class

pVal = nan;
if isempty(correlationTests) || ~isfield(correlationTests, 'classes')
    return;
end
for iClass = 1:numel(correlationTests.classes)
    classResult = correlationTests.classes(iClass);
    if ~strcmp(classResult.name, className)
        continue;
    end
    for iPair = 1:numel(classResult.pairs)
        pairResult = classResult.pairs(iPair);
        sameOrder = strcmp(pairResult.areaX, areaX) && strcmp(pairResult.areaY, areaY);
        flipOrder = strcmp(pairResult.areaX, areaY) && strcmp(pairResult.areaY, areaX);
        if sameOrder || flipOrder
            pVal = pairResult.p;
            return;
        end
    end
end
end

function clause = format_p_clause(pVal)
% FORMAT_P_CLAUSE - Title suffix for a finite permutation p-value

clause = '';
if isempty(pVal) || ~isscalar(pVal) || ~isfinite(pVal)
    return;
end
clause = sprintf(', p = %.3f', pVal);
end
