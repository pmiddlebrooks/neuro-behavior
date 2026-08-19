%%
% Criticality Multiple Metrics Across Task Types (Manuscript)
%
% Runs d2 (AR), avalanche (AV), and PRG analyses (per-session cache under
% dropPath/criticality_manuscript), plots aligned multi-metric session
% a cross-session metric correlation matrix (pooled across task types).
%
% Variables:
%   sessionTypes, collectStart, collectEnd, d2Window, prgWindow, brainArea, areasToPlot
%   d2Window / prgWindow - Analysis window length (s); [] = one window over the
%                         full collect duration per session
%   avWindow             - Avalanche analysis tile (s); [] = full collect with
%                         one shared population threshold (total class). When
%                         set, each tile gets its own threshold from that
%                         tile's pop activity; avalanches are pooled and fit
%                         once (not averaged). With splitByEngagement, engaged
%                         and non-engaged each use a distinct cutoff from that
%                         class's pop activity (or per-tile when avWindow set).
%   binSizeD2 / binSizePrg / binSizeAv - Spike bin width (s) for d2, PRG, and
%                         avalanche analyses; overrides each pipeline default
%   engagementBufferBefore - Seconds before each reach/beam-break counted as engaged
%                            (reach: reachBufferBefore; interval: eventBufferBefore)
%   engagementBufferAfter  - Seconds after each reach/beam-break counted as engaged
%                            (reach: reachBufferAfter; interval: eventBufferAfter)
%   engagementBuffer       - Legacy symmetric alias; if before/after unset, sets both
%   minNonEngagedWindow - Min gap without events (s) for non-engaged avalanche
%                        segments (default 30)
%   absorbSingleEvents - If true, isolated single events flanked by qualifying
%                        non-engaged gaps are merged into non-engaged time
%   minTimeNonEngaged  - When splitByEngagement, min total non-engaged time (s)
%                        to keep non-engaged metrics; shorter sessions stay on
%                        the x-axis but plot blank (NaN) so plots stay aligned
%   thresholdMethod    - Avalanche population cutoff: 'median' or 'quantile10'
%   runArBatch, runAvBatch, runPrgBatch - Select which pipelines to run
%                            (any non-empty combination of d2 / avalanche / PRG).
%                            Unselected metrics stay blank in combined / separated
%                            / pair-scatter layouts; correlation matrix only
%                            includes selected families.
%   useSessionCache    - If true (default), load/save per-session pipeline files
%                        under dropPath/criticality_manuscript/<task>/[<subject>/]<session>/
%   forceRecompute     - If true, ignore per-session cache and overwrite
%   plotResults        - Create combined d2/tau/alpha figure(s)
%   plotMetricPairScatters - 2x2 figure: d2 vs tau, d2 vs alpha,
%                            paramSD (crackling 1/σνz) vs (α-1)/(τ-1),
%                            d2 vs paramSD
%   plotCorrelationMatrix - Pearson corr heatmap across sessions (all tasks);
%                            only metrics from selected pipelines
%   enablePermutations - If false, observed metrics only (no shuffles; faster)
%   useAnchorAffineMap - If true, non-anchor metrics LS-affine-map onto
%                        anchorMetric (minimize within-session differences).
%                        If false, markers still share one x-axis (slight
%                        offsets); secondary metrics use independent range
%                        maps onto the primary display ylim, with right-side
%                        axes showing native tau/alpha ticks.
%   plotSeparatedMetrics - 2x4 figure: d2/tau/alpha/paramSD (top) and
%                          decades/dcc/kurtosis/D_JS (bottom); consecutive sessions
%                          linked within each task type on each panel.
%                          When enablePermutations is true, shuffled/surrogate
%                          session summaries are overlaid in gray.
%   anchorMetric       - 'd2', 'tau', or 'alpha' (primary / left axis)
%   metricsToPlot      - Subset of {'d2','tau','alpha'} markers to draw
%   splitByEngagement  - If true, interval/reach use engaged vs non-engaged
%                        analyses (d2, AV including decades, PRG); make two
%                        plots (engaged and non-engaged), each including
%                        spontaneous alongside that class.
%                        d2/PRG: split windows from full-session cache when
%                        present (only event times needed). Avalanches: still
%                        detected on engaged vs non-engaged segments (cannot
%                        be split from a full-session fit).
%                        Paired plots share d2-aligned y-limits for comparison.
%                        Correlation matrix always uses full-session metrics.
%                        See minTimeNonEngaged for blanking short non-engaged.
%
% Goal:
%   One session-grouped plot per brain area with d2, tau, and alpha per session.
%   Optionally anchor non-anchor metrics onto the chosen metric's y-scale via
%   affine maps. Optional pair scatters, separated metric panels, and
%   correlation matrix across sessions.

%% Configuration
sessionTypes = default_manuscript_session_types();
sessionTypes = order_manuscript_session_types(sessionTypes);
collectStart = 10;
collectEnd = 90*60;
% collectEnd = [];  % [] = full session
d2Window = 30;
prgWindow = d2Window;
avWindow = 5*60;   % [] = full collect, shared threshold; e.g. 30 = per-window thresholds
% One d2/PRG estimate for the full collect window ([] when collectEnd is [])

binSizeD2 = 0.025;   % d2/AR spike bin width (s); overrides AR default
binSizePrg = 0.05;  % PRG spike bin width (s); overrides PRG default
binSizeAv = 0.05;   % avalanche spike bin width (s); overrides AV default

% Engagement timing (reach + interval + semicircle); defaults match engagement module fill_*_defaults
engagementBufferBefore = 3;  % s before each reach/beam-break = engaged
engagementBufferAfter = 1;   % s after each reach/beam-break = engaged
minNonEngagedWindow = 30;   % min gap (s) for non-engaged avalanche segments
absorbSingleEvents = true;  % merge isolated single events into non-engaged gaps
minTimeNonEngaged = 180;      % min total non-engaged time (s) to plot; 0 = no filter
% Sessions below minTimeNonEngaged stay in non-engaged plots but are blanked

% Paths first — needed by default_manuscript_brain_area_combinations / plotConfig
setup_criticality_manuscript_paths('criticality_multiple_metrics_across_tasks');
paths = get_paths();

brainArea = 'M23M56';
% brainArea = 'M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};

% Pipeline selection — any combination of d2 (AR), avalanche (AV), PRG
runArBatch = true;   % d2
runAvBatch = true;   % tau, alpha, paramSD, decades, dcc
runPrgBatch = true;  % kurtosis, JS distance
runEngagementBatch = true;
useSessionCache = true;   % per-session d2 / AV / PRG files; skip cached sessions
forceRecompute = false;   % true: reprocess and overwrite per-session cache
plotResults = true;
plotMetricPairScatters = true;
plotSeparatedMetrics = true;
plotCorrelationMatrix = true;
enablePermutations = false;
useAnchorAffineMap = false;  % false: native scales with independent right axes
anchorMetric = 'd2';  % 'd2', 'tau', or 'alpha' (primary / left axis)
metricsToPlot = {'d2', 'tau', 'alpha'};  % subset of markers; auto-narrowed to selected pipelines
% metricsToPlot = {'d2', 'tau'};  % any non-empty subset
splitByEngagement = false;  % true: engaged / non-engaged plots (spontaneous on both)

useLog10D2 = true;
useSubsampling = true;
nSubsamples = 40;
nNeuronsSubsample = 45;
minNeuronsMultiple = 1.1;

powerLawFitMethod = 'plfit2023';
avalancheDetectionMode = 'fixedBinMedian';
thresholdMethod = 'quantile10';  % 'median' or 'quantile10' (10th percentile cutoff)

finalCutoffDivisor = 16;
prgMethod = 'pca';

plotConfig = fill_manuscript_plot_config();

% Resolve which pipelines are active. Unselected → blank panels.
useAr = logical(runArBatch);
useAv = logical(runAvBatch);
usePrg = logical(runPrgBatch);
if ~(useAr || useAv || usePrg)
  error('Select at least one pipeline: set runArBatch / runAvBatch / runPrgBatch true.');
end

fprintf('\n=== Criticality Multiple Metrics Across Tasks ===\n');
fprintf('Pipelines: AR(d2)=%d  AV=%d  PRG=%d\n', useAr, useAv, usePrg);
fprintf('Session types: %s\n', strjoin(sessionTypes, ', '));
if isempty(collectEnd)
  fprintf('Collect window: [%.1f, full] s\n', collectStart);
else
  fprintf('Collect window: [%.1f, %.1f] s\n', collectStart, collectEnd);
end
if isempty(d2Window)
  fprintf('d2 windows: full collect duration (one window per session)\n');
else
  fprintf('d2 windows: %.0f s\n', d2Window);
end
if isempty(prgWindow)
  fprintf('PRG blocks: full collect duration (one block per session)\n');
else
  fprintf('PRG blocks: %.0f s\n', prgWindow);
end
if isempty(avWindow)
  if splitByEngagement
    fprintf(['AV window: full collect (distinct thresholds for total / ', ...
      'engaged / non-engaged)\n']);
  else
    fprintf('AV window: full collect (one shared threshold)\n');
  end
else
  fprintf('AV window: %.0f s (per-window thresholds; pool events, one fit)\n', avWindow);
end
fprintf('binSizeD2: %.3f s; binSizePrg: %.3f s; binSizeAv: %.3f s\n', ...
  binSizeD2, binSizePrg, binSizeAv);
fprintf('engagementBuffer: before=%.3g s, after=%.3g s; minNonEngagedWindow: %.1f s; absorbSingleEvents: %d\n', ...
  engagementBufferBefore, engagementBufferAfter, minNonEngagedWindow, absorbSingleEvents);
fprintf('minTimeNonEngaged: %.1f s (blank non-engaged below this; 0 = off)\n', ...
  minTimeNonEngaged);
fprintf('avalancheDetectionMode: %s; thresholdMethod: %s\n', ...
  avalancheDetectionMode, thresholdMethod);
fprintf('enablePermutations: %d (observed metrics only when false)\n', enablePermutations);
fprintf('useAnchorAffineMap: %d\n', useAnchorAffineMap);
fprintf('anchorMetric: %s\n', anchorMetric);
fprintf('metricsToPlot: %s\n', strjoin(metricsToPlot, ', '));
fprintf('splitByEngagement: %d\n', splitByEngagement);
fprintf('useSessionCache: %d; forceRecompute: %d\n', useSessionCache, forceRecompute);
fprintf('plotMetricPairScatters: %d\n', plotMetricPairScatters);
fprintf('plotSeparatedMetrics: %d\n', plotSeparatedMetrics);
fprintf('plotCorrelationMatrix: %d\n', plotCorrelationMatrix);
if isempty(d2Window)
  fprintf('Plot filenames will use tag "winfull" (d2Window=[]).\n');
end
if isempty(avWindow)
  fprintf('Plot filenames will use tag "avfull" (avWindow=[]).\n');
end
if isempty(prgWindow)
  fprintf('PRG batch uses full-session blocks when prgWindow=[].\n');
end
set_manuscript_av_window(avWindow);

% AR batch (d2) — full-session metrics across all requested session types
arOpts = struct( ...
  'sessionTypes', {sessionTypes}, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'd2Window', d2Window, ...
  'binSize', binSizeD2, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'areasToPlot', {areasToPlot}, ...
  'useLog10D2', useLog10D2, ...
  'useSubsampling', useSubsampling, ...
  'nSubsamples', nSubsamples, ...
  'nNeuronsSubsample', nNeuronsSubsample, ...
  'minNeuronsMultiple', minNeuronsMultiple, ...
  'enablePermutations', enablePermutations, ...
  'useSessionCache', useSessionCache, ...
  'forceRecompute', forceRecompute, ...
  'plotResults', false);

if useAr
  arOut = criticality_ar_across_tasks(arOpts);
else
  arOut = [];
  fprintf('Skipping AR (d2) batch.\n');
end

% AV batch (tau, alpha, paramSD, decades, dcc)
avOpts = struct( ...
  'sessionTypes', {sessionTypes}, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'avWindow', avWindow, ...
  'binSize', binSizeAv, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'areasToPlot', {areasToPlot}, ...
  'powerLawFitMethod', powerLawFitMethod, ...
  'avalancheDetectionMode', avalancheDetectionMode, ...
  'thresholdMethod', thresholdMethod, ...
  'useSubsampling', useSubsampling, ...
  'nSubsamples', nSubsamples, ...
  'nNeuronsSubsample', nNeuronsSubsample, ...
  'minNeuronsMultiple', minNeuronsMultiple, ...
  'enablePermutations', enablePermutations, ...
  'useSessionCache', useSessionCache, ...
  'forceRecompute', forceRecompute, ...
  'plotResults', false);

if useAv
  avOut = criticality_av_across_tasks(avOpts);
else
  avOut = [];
  fprintf('Skipping AV batch.\n');
end

% PRG batch (kurtosis / kappaMean, Jensen-Shannon / djsMean)
prgOpts = struct( ...
  'sessionTypes', {sessionTypes}, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'prgWindow', prgWindow, ...
  'binSize', binSizePrg, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'areasToPlot', {areasToPlot}, ...
  'useSubsampling', useSubsampling, ...
  'nSubsamples', nSubsamples, ...
  'nNeuronsSubsample', nNeuronsSubsample, ...
  'minNeuronsMultiple', minNeuronsMultiple, ...
  'enableSurrogates', enablePermutations, ...
  'finalCutoffDivisor', finalCutoffDivisor, ...
  'prgMethod', prgMethod, ...
  'useSessionCache', useSessionCache, ...
  'forceRecompute', forceRecompute, ...
  'plotResults', false);

if usePrg
  prgOut = criticality_prg_across_tasks(prgOpts);
else
  prgOut = [];
  fprintf('Skipping PRG batch.\n');
end

% Stub empty plotData for skipped pipelines (same areas so panels stay blank)
refAreas = resolve_pipeline_ref_areas(arOut, avOut, prgOut, areasToPlot, brainArea);
if isempty(arOut) || ~isfield(arOut, 'plotData')
  arOut = make_empty_pipeline_out(sessionTypes, refAreas, 'ar', useLog10D2);
end
if isempty(avOut) || ~isfield(avOut, 'plotData')
  avOut = make_empty_pipeline_out(sessionTypes, refAreas, 'av', false);
end
if isempty(prgOut) || ~isfield(prgOut, 'plotData')
  prgOut = make_empty_pipeline_out(sessionTypes, refAreas, 'prg', false);
end

% Narrow combined-plot markers to selected pipelines (may be empty if PRG-only)
metricsToPlot = filter_metrics_to_plot_by_pipelines(metricsToPlot, useAr, useAv);
if ~isempty(metricsToPlot) && useAnchorAffineMap && ~ismember(anchorMetric, metricsToPlot)
  fprintf('anchorMetric "%s" not in active metricsToPlot; using "%s".\n', ...
    anchorMetric, metricsToPlot{1});
  anchorMetric = metricsToPlot{1};
end

% Engagement batch (interval/reach engaged vs non-engaged for selected pipelines)
engOut = [];
if splitByEngagement
  engSessionTypes = intersect(sessionTypes, {'interval', 'reach', 'semicircle'}, 'stable');
  if isempty(engSessionTypes)
    error('splitByEngagement requires interval, reach, and/or semicircle in sessionTypes.');
  end
  engAnalyses = {};
  if useAr, engAnalyses{end + 1} = 'd2'; end %#ok<AGROW>
  if useAv, engAnalyses{end + 1} = 'avalanches'; end %#ok<AGROW>
  if usePrg, engAnalyses{end + 1} = 'kurtosis'; end %#ok<AGROW>
  if isempty(engAnalyses)
    error('splitByEngagement requires at least one active pipeline (AR/AV/PRG).');
  end
  engOpts = struct( ...
    'sessionTypes', {engSessionTypes}, ...
    'collectStart', collectStart, ...
    'collectEnd', collectEnd, ...
    'd2Window', d2Window, ...
    'prgWindow', prgWindow, ...
    'avWindow', avWindow, ...
    'binSizeD2', binSizeD2, ...
    'binSizePrg', binSizePrg, ...
    'binSizeAv', binSizeAv, ...
    'engagementBufferBefore', engagementBufferBefore, ...
    'engagementBufferAfter', engagementBufferAfter, ...
    'minNonEngagedWindow', minNonEngagedWindow, ...
    'absorbSingleEvents', absorbSingleEvents, ...
    'minTimeNonEngaged', minTimeNonEngaged, ...
    'brainArea', brainArea, ...
    'brainAreaCombinations', {brainAreaCombinations}, ...
    'useLog10D2', useLog10D2, ...
    'useSubsampling', useSubsampling, ...
    'nSubsamples', nSubsamples, ...
    'nNeuronsSubsample', nNeuronsSubsample, ...
    'minNeuronsMultiple', minNeuronsMultiple, ...
    'enablePermutations', enablePermutations, ...
    'powerLawFitMethod', powerLawFitMethod, ...
    'avalancheDetectionMode', avalancheDetectionMode, ...
    'thresholdMethod', thresholdMethod, ...
    'finalCutoffDivisor', finalCutoffDivisor, ...
    'prgMethod', prgMethod, ...
    'analyses', {engAnalyses}, ...
    'useSessionCache', useSessionCache, ...
    'forceRecompute', forceRecompute, ...
    'plotConfig', plotConfig);

  if ~runEngagementBatch
    error('splitByEngagement requires runEngagementBatch true (per-session cache skips already processed sessions).');
  end
  engOut = run_multimetric_engagement_batch(engOpts);
end

activeAnalyses = struct('ar', useAr, 'av', useAv, 'prg', usePrg);

% Combined plotting (same Editor section as batch load — run the full script)
if plotResults
  if ~exist('plotSeparatedMetrics', 'var') || isempty(plotSeparatedMetrics)
    plotSeparatedMetrics = true;
  end
  if ~exist('plotMetricPairScatters', 'var') || isempty(plotMetricPairScatters)
    plotMetricPairScatters = true;
  end
  if isempty(areasToPlot) && ~isempty(brainArea)
    areasToPlot = {brainArea};
  end
  if ~isempty(metricsToPlot)
    metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
    if useAnchorAffineMap && ~ismember(anchorMetric, metricsToPlot)
      error('anchorMetric "%s" must be included in metricsToPlot when useAnchorAffineMap is true.', ...
        anchorMetric);
    end
  end

  fprintf('\n=== Combined plotting ===\n');
  fprintf('Figures save to: %s\n', fullfile(paths.dropPath, 'criticality_manuscript'));
  if isempty(metricsToPlot)
    fprintf('No d2/tau/alpha markers active (PRG-only or empty metricsToPlot).\n');
  elseif useAnchorAffineMap
    fprintf('Anchor metric: %s (affine map onto this scale)\n', anchorMetric);
  else
    fprintf('Anchor affine map: off (native metric scales)\n');
  end
  if ~isempty(metricsToPlot)
    fprintf('Markers: %s\n', strjoin(metricsToPlot, ', '));
  end

  if ~splitByEngagement
    plotAreas = resolve_multimetric_plot_areas(arOut, avOut, prgOut, useAr, useAv, usePrg, ...
      areasToPlot, brainArea);
    if isempty(plotAreas)
      error('No brain areas available for plotting from selected pipelines.');
    end
    fprintf('Areas: %s\n', strjoin(plotAreas, ', '));
    if ~isempty(metricsToPlot)
      plot_multimetric_d2_tau_alpha_across_tasks(arOut.plotData, avOut.plotData, plotAreas, ...
        sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
        plotConfig, anchorMetric, '', metricsToPlot, struct(), useAnchorAffineMap, ...
        useSubsampling, nNeuronsSubsample, avWindow);
    end
    if plotSeparatedMetrics
      plot_multimetric_separated_axes_across_tasks(arOut.plotData, avOut.plotData, ...
        prgOut.plotData, plotAreas, sessionTypes, collectStart, collectEnd, d2Window, ...
        paths, brainArea, useLog10D2, plotConfig, '', metricsToPlot, avOut.plotData, ...
        finalCutoffDivisor, enablePermutations, useSubsampling, nNeuronsSubsample, ...
        activeAnalyses, avWindow, binSizeD2);
    end
    if plotMetricPairScatters
      plot_multimetric_pair_scatters_across_tasks(arOut.plotData, avOut.plotData, plotAreas, ...
        sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
        plotConfig, '', useSubsampling, nNeuronsSubsample, activeAnalyses, avWindow);
    end
  else
    engagementClasses = {'engaged', 'nonEngaged'};
    classViews = struct();
    plotAreas = {};
    prgPlotDataFull = prgOut.plotData;
    for iClass = 1:numel(engagementClasses)
      engClass = engagementClasses{iClass};
      [arView, avView, prgView] = build_engagement_class_metric_views( ...
        arOut.plotData, avOut.plotData, prgPlotDataFull, engOut.plotData, engClass, ...
        sessionTypes, minTimeNonEngaged);
      classViews.(engClass).ar = arView;
      classViews.(engClass).av = avView;
      classViews.(engClass).prg = prgView;
      classAreas = resolve_multimetric_plot_areas( ...
        struct('plotData', arView), struct('plotData', avView), struct('plotData', prgView), ...
        useAr, useAv, usePrg, {}, '');
      if isempty(plotAreas)
        plotAreas = classAreas;
      else
        plotAreas = intersect(plotAreas, classAreas, 'stable');
      end
    end
    if ~isempty(areasToPlot)
      plotAreas = intersect(plotAreas, cellstr(string(areasToPlot)), 'stable');
    end
    if isempty(plotAreas)
      error('No common brain areas for engagement paired plots.');
    end

    % Shared maps + y-limits across engaged/non-engaged so d2 axes match
    if ~isempty(metricsToPlot)
      sharedByArea = compute_shared_engagement_plot_scales(classViews, plotAreas, ...
        sessionTypes, metricsToPlot, anchorMetric, useAnchorAffineMap);
    else
      sharedByArea = struct();
    end

    for iClass = 1:numel(engagementClasses)
      engClass = engagementClasses{iClass};
      fprintf('Areas (%s): %s\n', engClass, strjoin(plotAreas, ', '));
      if ~isempty(metricsToPlot)
        plot_multimetric_d2_tau_alpha_across_tasks( ...
          classViews.(engClass).ar, classViews.(engClass).av, plotAreas, ...
          sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
          plotConfig, anchorMetric, engClass, metricsToPlot, sharedByArea, useAnchorAffineMap, ...
          useSubsampling, nNeuronsSubsample, avWindow);
      end
      if plotSeparatedMetrics
        plot_multimetric_separated_axes_across_tasks( ...
          classViews.(engClass).ar, classViews.(engClass).av, classViews.(engClass).prg, ...
          plotAreas, sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, ...
          useLog10D2, plotConfig, engClass, metricsToPlot, classViews.(engClass).av, ...
          finalCutoffDivisor, enablePermutations, useSubsampling, nNeuronsSubsample, ...
          activeAnalyses, avWindow, binSizeD2);
      end
      if plotMetricPairScatters
        plot_multimetric_pair_scatters_across_tasks( ...
          classViews.(engClass).ar, classViews.(engClass).av, plotAreas, ...
          sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
          plotConfig, engClass, useSubsampling, nNeuronsSubsample, activeAnalyses, avWindow);
      end
    end
  end
else
  fprintf('\nplotResults=false; skipping combined / separated / pair-scatter figures.\n');
end

% Cross-session metric correlation matrix (pooled across task types)
if plotCorrelationMatrix
  if isempty(areasToPlot) && ~isempty(brainArea)
    areasToPlot = {brainArea};
  end
  corrAreas = resolve_multimetric_plot_areas(arOut, avOut, prgOut, useAr, useAv, usePrg, ...
    areasToPlot, brainArea);
  if isempty(corrAreas)
    error('No brain areas available among selected pipelines for correlation matrix.');
  end
  fprintf('\n=== Metric correlation matrix (sessions pooled across tasks) ===\n');
  fprintf('Active pipelines: AR=%d AV=%d PRG=%d\n', useAr, useAv, usePrg);
  fprintf('Areas: %s\n', strjoin(corrAreas, ', '));
  plot_metric_correlation_matrix_across_sessions( ...
    arOut, avOut, prgOut, corrAreas, sessionTypes, collectStart, collectEnd, ...
    d2Window, paths, brainArea, useLog10D2, plotConfig, useSubsampling, nNeuronsSubsample, ...
    activeAnalyses, avWindow);
end

fprintf('\n=== Done ===\n');
set_manuscript_av_window([]);

%% Local functions

function plot_metric_correlation_matrix_across_sessions(arOut, avOut, prgOut, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, plotConfig, ...
    useSubsampling, nNeuronsSubsample, activeAnalyses, avWindow)
% PLOT_METRIC_CORRELATION_MATRIX_ACROSS_SESSIONS - Pearson corr heatmap per area
%
% Variables:
%   arOut/avOut/prgOut - Batch outputs with plotData (and AR batchResults fallback)
%   areasToPlot        - Brain areas to plot
%   sessionTypes       - Session types pooled into one correlation
%   activeAnalyses     - Struct with .ar/.av/.prg booleans; only those metrics plotted
%
% Goal:
%   Each matrix entry is corr(metric_i, metric_j) across sessions (all tasks).
sessionTypes = order_manuscript_session_types(sessionTypes);
%   Crackling (paramSD=1/σνz), decades, kurtosis, and JS distance are inverted
%   (1/x) before correlating so expected co-variation with criticality is positive.

if nargin < 12 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 13 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 14 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 15 || isempty(activeAnalyses)
  activeAnalyses = struct('ar', true, 'av', true, 'prg', true);
end
if nargin < 16
  avWindow = [];
end
activeAnalyses = fill_active_analyses(activeAnalyses);

allMetricKeys = {'d2', 'tau', 'alpha', 'paramSD', 'decades', 'dcc', 'kurtosis', 'djs', ...
  'meanSpikesPerBinPerNeuron'};
allMetricLabels = {'d2', 'tau', 'alpha', '1/\sigma\nu z', 'decades', 'dcc', 'kurtosis', ...
  'JS distance', 'spikes/bin/neuron'};
% Invert metrics expected to anti-correlate with criticality (so all should co-vary)
invertMetricKeys = {'paramSD', 'decades', 'kurtosis', 'djs'};
invertMetricLabels = {'\sigma\nu z', '1/decades', '1/kurtosis', '1/JS distance'};

keepMask = false(size(allMetricKeys));
for iMetric = 1:numel(allMetricKeys)
  key = allMetricKeys{iMetric};
  if ismember(key, {'d2', 'meanSpikesPerBinPerNeuron'})
    keepMask(iMetric) = activeAnalyses.ar;
  elseif ismember(key, {'tau', 'alpha', 'paramSD', 'decades', 'dcc'})
    keepMask(iMetric) = activeAnalyses.av;
  elseif ismember(key, {'kurtosis', 'djs'})
    keepMask(iMetric) = activeAnalyses.prg;
  end
end
metricKeys = allMetricKeys(keepMask);
metricLabels = allMetricLabels(keepMask);
if numel(metricKeys) < 2
  warning('Correlation matrix needs >=2 selected metrics; skipping.');
  return;
end

saveDir = fullfile(paths.dropPath, 'criticality_manuscript', 'figures');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for iArea = 1:numel(areasToPlot)
  areaName = areasToPlot{iArea};
  areaIdxAr = find(strcmp(arOut.plotData.areas, areaName), 1);
  areaIdxAv = find(strcmp(avOut.plotData.areas, areaName), 1);
  areaIdxPrg = find(strcmp(prgOut.plotData.areas, areaName), 1);
  if (activeAnalyses.ar && isempty(areaIdxAr)) ...
      || (activeAnalyses.av && isempty(areaIdxAv)) ...
      || (activeAnalyses.prg && isempty(areaIdxPrg))
    warning('Skipping correlation for area %s (missing in an active pipeline).', areaName);
    continue;
  end
  if isempty(areaIdxAr), areaIdxAr = 1; end
  if isempty(areaIdxAv), areaIdxAv = 1; end
  if isempty(areaIdxPrg), areaIdxPrg = 1; end

  sessionTable = build_correlation_metric_session_table( ...
    arOut, avOut, prgOut, sessionTypes, areaIdxAr, areaIdxAv, areaIdxPrg, areaName, ...
    activeAnalyses);
  if height(sessionTable) < 3
    warning('Skipping correlation for area %s: only %d sessions with metrics.', ...
      areaName, height(sessionTable));
    continue;
  end

  metricMat = nan(height(sessionTable), numel(metricKeys));
  plotLabels = metricLabels;
  for iMetric = 1:numel(metricKeys)
    metricMat(:, iMetric) = sessionTable.(metricKeys{iMetric});
    invertIdx = find(strcmp(invertMetricKeys, metricKeys{iMetric}), 1);
    if ~isempty(invertIdx)
      metricMat(:, iMetric) = safe_reciprocal_metric(metricMat(:, iMetric));
      plotLabels{iMetric} = invertMetricLabels{invertIdx};
    end
  end

  corrMat = corrcoef(metricMat, 'Rows', 'pairwise');
  nPair = count_pairwise_session_counts(metricMat);

  fprintf('  %s: %d sessions in correlation table\n', areaName, height(sessionTable));
  for iMetric = 1:numel(metricKeys)
    nValid = sum(isfinite(metricMat(:, iMetric)));
    fprintf('    %s: %d finite\n', plotLabels{iMetric}, nValid);
  end

  fig = figure('Color', 'w', 'Position', [100 100 780 700]);
  ax = axes(fig);
  imagesc(ax, corrMat);
  axis(ax, 'image');
  set(ax, 'YDir', 'normal');
  colormap(ax, correlation_blue_white_red_colormap(256));
  caxis(ax, [-1 1]); %#ok<CAXIS>
  cb = colorbar(ax);
  cb.Label.String = 'Pearson r (across sessions)';
  cb.Label.FontSize = plotConfig.axisLabelFontSize;

  nMetric = numel(plotLabels);
  set(ax, 'XTick', 1:nMetric, 'XTickLabel', plotLabels, ...
    'YTick', 1:nMetric, 'YTickLabel', plotLabels, ...
    'TickLabelInterpreter', 'tex', 'FontSize', plotConfig.tickLabelFontSize);
  xtickangle(ax, 45);
  xlabel(ax, 'Metric', 'FontSize', plotConfig.axisLabelFontSize);
  ylabel(ax, 'Metric', 'FontSize', plotConfig.axisLabelFontSize);

  for iRow = 1:nMetric
    for iCol = 1:nMetric
      rVal = corrMat(iRow, iCol);
      if ~isfinite(rVal)
        continue;
      end
      if abs(rVal) > 0.55
        textColor = [1 1 1];
      else
        textColor = [0.1 0.1 0.1];
      end
      text(ax, iCol, iRow, sprintf('%.2f\nn=%d', rVal, nPair(iRow, iCol)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', max(7, plotConfig.tickLabelFontSize - 2), 'Color', textColor);
    end
  end

  titleStr = sprintf('Metric correlations across sessions (%s) [%s]', ...
    areaName, format_d2_av_window_title_phrase(d2Window, avWindow));
  if useLog10D2 && activeAnalyses.ar
    titleStr = [titleStr, '; d2 = log10(d2)']; %#ok<AGROW>
  end
  invActive = intersect(invertMetricKeys, metricKeys, 'stable');
  if ~isempty(invActive)
    titleStr = [titleStr, '; inv: crackling, decades, kurtosis, JS']; %#ok<AGROW>
  end
  titleStr = append_subsamp_title_tag(titleStr, useSubsampling, nNeuronsSubsample);
  title(ax, titleStr, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');

  plotBase = make_correlation_matrix_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, useSubsampling, nNeuronsSubsample, avWindow);
  plotBase = [plotBase, '_invMetrics']; %#ok<AGROW>
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved correlation matrix: %s\n', fullfile(saveDir, plotBase));
end
end

function y = safe_reciprocal_metric(x)
% SAFE_RECIPROCAL_METRIC - Element-wise 1/x; non-finite or zero -> NaN
y = nan(size(x));
ok = isfinite(x) & (x ~= 0);
y(ok) = 1 ./ x(ok);
end

function sessionTable = build_correlation_metric_session_table(arOut, avOut, prgOut, ...
    sessionTypes, areaIdxAr, areaIdxAv, areaIdxPrg, areaName, activeAnalyses)
% BUILD_CORRELATION_METRIC_SESSION_TABLE - One row per session, metrics joined by name
%
% Variables:
%   arOut/avOut/prgOut - Pipeline outputs
%   sessionTypes       - Types to pool
%   areaIdx*           - Area indices in each plotData
%   areaName           - Area name (for AR batchResults rate fallback)
%   activeAnalyses     - Struct with .ar/.av/.prg; inactive fields stay NaN
%
% Goal:
%   Align available d2, AV exponents, PRG kurtosis/JS, and mean spikes/bin/neuron.

if nargin < 9 || isempty(activeAnalyses)
  activeAnalyses = struct('ar', true, 'av', true, 'prg', true);
end
activeAnalyses = fill_active_analyses(activeAnalyses);

arPlotData = arOut.plotData;
avPlotData = avOut.plotData;
prgPlotData = prgOut.plotData;

sessionTypeCol = {};
sessionNameCol = {};
d2Col = [];
tauCol = [];
alphaCol = [];
paramSDCol = [];
decadesCol = [];
dccCol = [];
kurtosisCol = [];
djsCol = [];
meanSpikesCol = [];

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  hasArType = isfield(arPlotData.byType, typeKey);
  hasAvType = isfield(avPlotData.byType, typeKey);
  hasPrgType = isfield(prgPlotData.byType, typeKey);
  if (activeAnalyses.ar && ~hasArType) && (activeAnalyses.av && ~hasAvType) ...
      && (activeAnalyses.prg && ~hasPrgType)
    continue;
  end
  if ~hasArType && ~hasAvType && ~hasPrgType
    continue;
  end

  arType = [];
  avType = [];
  prgType = [];
  if hasArType, arType = arPlotData.byType.(typeKey); end
  if hasAvType, avType = avPlotData.byType.(typeKey); end
  if hasPrgType, prgType = prgPlotData.byType.(typeKey); end

  % Driver session list: prefer AR, else AV, else PRG
  if activeAnalyses.ar && hasArType && isfield(arType, 'd2Mean') ...
      && areaIdxAr <= numel(arType.d2Mean) && ~isempty(arType.d2Mean{areaIdxAr})
    driverNames = get_type_session_names(arType);
    numSess = numel(arType.d2Mean{areaIdxAr});
    driver = 'ar';
  elseif activeAnalyses.av && hasAvType && isfield(avType, 'tau') ...
      && areaIdxAv <= numel(avType.tau) && ~isempty(avType.tau{areaIdxAv})
    driverNames = get_type_session_names(avType);
    numSess = numel(avType.tau{areaIdxAv});
    driver = 'av';
  elseif activeAnalyses.prg && hasPrgType && isfield(prgType, 'kappaMean') ...
      && areaIdxPrg <= numel(prgType.kappaMean) && ~isempty(prgType.kappaMean{areaIdxPrg})
    driverNames = get_type_session_names(prgType);
    numSess = numel(prgType.kappaMean{areaIdxPrg});
    driver = 'prg';
  else
    continue;
  end

  for i = 1:numSess
    sessionName = driverNames{min(i, numel(driverNames))};
    arIdx = i;
    avIdx = i;
    prgIdx = i;
    if ~strcmp(driver, 'ar') && hasArType
      arIdx = find_matching_session_index(arType, sessionName, i);
    end
    if ~strcmp(driver, 'av') && hasAvType
      avIdx = find_matching_session_index(avType, sessionName, i);
    elseif ~hasAvType
      avIdx = [];
    end
    if ~strcmp(driver, 'prg') && hasPrgType
      prgIdx = find_matching_session_index(prgType, sessionName, i);
    elseif ~hasPrgType
      prgIdx = [];
    end
    if strcmp(driver, 'ar') && ~hasArType
      arIdx = [];
    end

    d2Val = nan;
    tauVal = nan;
    alphaVal = nan;
    paramSDVal = nan;
    decadesVal = nan;
    dccVal = nan;
    kurtosisVal = nan;
    djsVal = nan;
    meanSpikesVal = nan;

    if activeAnalyses.ar && ~isempty(arIdx) && hasArType
      d2Val = get_metric_series_value(arType.d2Mean{areaIdxAr}, arIdx);
      meanSpikesVal = get_mean_spikes_per_bin_per_neuron_session( ...
        arType, arOut, sessionType, sessionName, areaIdxAr, arIdx, areaName);
    end
    if activeAnalyses.av && ~isempty(avIdx) && hasAvType
      tauVal = get_type_cell_metric(avType, 'tau', areaIdxAv, avIdx);
      alphaVal = get_type_cell_metric(avType, 'alpha', areaIdxAv, avIdx);
      paramSDVal = get_type_cell_metric(avType, 'paramSD', areaIdxAv, avIdx);
      decadesVal = get_type_cell_metric(avType, 'decades', areaIdxAv, avIdx);
      dccVal = get_type_cell_metric(avType, 'dcc', areaIdxAv, avIdx);
    end
    if activeAnalyses.prg && ~isempty(prgIdx) && hasPrgType
      kurtosisVal = get_type_cell_metric(prgType, 'kappaMean', areaIdxPrg, prgIdx);
      djsVal = get_type_cell_metric(prgType, 'djsMean', areaIdxPrg, prgIdx);
    end

    % Keep session if at least two selected metrics are finite
    metricVec = [];
    if activeAnalyses.ar
      metricVec = [metricVec, d2Val, meanSpikesVal]; %#ok<AGROW>
    end
    if activeAnalyses.av
      metricVec = [metricVec, tauVal, alphaVal, paramSDVal, decadesVal, dccVal]; %#ok<AGROW>
    end
    if activeAnalyses.prg
      metricVec = [metricVec, kurtosisVal, djsVal]; %#ok<AGROW>
    end
    if sum(isfinite(metricVec)) < 2
      continue;
    end

    sessionTypeCol{end + 1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = sessionName; %#ok<AGROW>
    d2Col(end + 1, 1) = d2Val; %#ok<AGROW>
    tauCol(end + 1, 1) = tauVal; %#ok<AGROW>
    alphaCol(end + 1, 1) = alphaVal; %#ok<AGROW>
    paramSDCol(end + 1, 1) = paramSDVal; %#ok<AGROW>
    decadesCol(end + 1, 1) = decadesVal; %#ok<AGROW>
    dccCol(end + 1, 1) = dccVal; %#ok<AGROW>
    kurtosisCol(end + 1, 1) = kurtosisVal; %#ok<AGROW>
    djsCol(end + 1, 1) = djsVal; %#ok<AGROW>
    meanSpikesCol(end + 1, 1) = meanSpikesVal; %#ok<AGROW>
  end
end

sessionTable = table(sessionTypeCol, sessionNameCol, d2Col, tauCol, alphaCol, ...
  paramSDCol, decadesCol, dccCol, kurtosisCol, djsCol, meanSpikesCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'd2', 'tau', 'alpha', ...
  'paramSD', 'decades', 'dcc', 'kurtosis', 'djs', 'meanSpikesPerBinPerNeuron'});
end

function val = get_type_cell_metric(typeData, fieldName, areaIdx, sessionIdx)
% GET_TYPE_CELL_METRIC - Scalar from plotData.byType cell series
val = nan;
if ~isfield(typeData, fieldName) || areaIdx > numel(typeData.(fieldName))
  return;
end
series = typeData.(fieldName){areaIdx};
val = get_metric_series_value(series, sessionIdx);
end

function rateVal = get_mean_spikes_per_bin_per_neuron_session(arType, arOut, sessionType, ...
    sessionName, areaIdxAr, sessionIdx, areaName)
% GET_MEAN_SPIKES_PER_BIN_PER_NEURON_SESSION - Prefer plotData; else batchResults
%
% Variables:
%   arType      - AR plotData.byType entry
%   arOut       - Full AR output (batchResults for fallback)
%   sessionType - Session type string
%   sessionName - Session name for batch lookup
%   areaIdxAr   - Area index in plotData
%   sessionIdx  - Index within type series
%   areaName    - Area name for batchResults lookup
%
% Goal:
%   Session mean of (pop spikes/bin) / nNeurons.

rateVal = nan;
if isfield(arType, 'meanSpikesPerBinPerNeuron') ...
    && areaIdxAr <= numel(arType.meanSpikesPerBinPerNeuron) ...
    && ~isempty(arType.meanSpikesPerBinPerNeuron{areaIdxAr})
  rateVal = get_metric_series_value(arType.meanSpikesPerBinPerNeuron{areaIdxAr}, sessionIdx);
  if isfinite(rateVal)
    return;
  end
end

if ~isfield(arOut, 'batchResults') || isempty(arOut.batchResults)
  return;
end
batchResults = arOut.batchResults;
for s = 1:numel(batchResults)
  if ~batchResults(s).success || isempty(batchResults(s).results)
    continue;
  end
  if ~strcmp(batchResults(s).sessionType, sessionType)
    continue;
  end
  if ~strcmp(char(batchResults(s).sessionName), char(sessionName))
    continue;
  end
  results = batchResults(s).results;
  areaIdxRes = find(strcmp(results.areas, areaName), 1);
  if isempty(areaIdxRes)
    return;
  end
  rateVal = summarize_mean_spikes_from_ar_results(results, areaIdxRes);
  return;
end
end

function rateVal = summarize_mean_spikes_from_ar_results(results, areaIdx)
% SUMMARIZE_MEAN_SPIKES_FROM_AR_RESULTS - mean(popActivityWindows) / nNeurons
rateVal = nan;
if ~isfield(results, 'popActivityWindows') || areaIdx > numel(results.popActivityWindows) ...
    || isempty(results.popActivityWindows{areaIdx})
  return;
end
popWin = results.popActivityWindows{areaIdx}(:);
popWin = popWin(isfinite(popWin));
if isempty(popWin)
  return;
end
nNeurons = nan;
if isfield(results, 'nNeurons') && numel(results.nNeurons) >= areaIdx
  nNeurons = results.nNeurons(areaIdx);
end
if ~(isfinite(nNeurons) && nNeurons > 0)
  return;
end
rateVal = mean(popWin) / nNeurons;
end

function nPair = count_pairwise_session_counts(metricMat)
% COUNT_PAIRWISE_SESSION_COUNTS - Finite-pair counts for each metric pair
nMetric = size(metricMat, 2);
nPair = zeros(nMetric, nMetric);
for i = 1:nMetric
  for j = 1:nMetric
    nPair(i, j) = sum(isfinite(metricMat(:, i)) & isfinite(metricMat(:, j)));
  end
end
end

function cmap = correlation_blue_white_red_colormap(nLevels)
% CORRELATION_BLUE_WHITE_RED_COLORMAP - Diverging map centered at 0
if nargin < 1 || isempty(nLevels)
  nLevels = 256;
end
halfN = floor(nLevels / 2);
blueToWhite = [linspace(0.15, 1, halfN)', linspace(0.35, 1, halfN)', linspace(0.75, 1, halfN)'];
whiteToRed = [linspace(1, 0.75, nLevels - halfN)', linspace(1, 0.15, nLevels - halfN)', ...
  linspace(1, 0.15, nLevels - halfN)'];
cmap = [blueToWhite; whiteToRed];
end

function plotBase = make_correlation_matrix_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, useSubsampling, nNeuronsSubsample, avWindow)
% MAKE_CORRELATION_MATRIX_PLOT_BASENAME - File stem for correlation heatmap
if nargin < 7 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 8 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 9
  avWindow = [];
end
if isempty(brainArea)
  areaTag = areaName;
else
  areaTag = brainArea;
end
winTag = format_window_sec_file_tag(d2Window);
avTag = format_window_sec_file_tag(avWindow);
if isempty(collectEnd)
  timeTag = sprintf('%.0f-full', collectStart);
else
  timeTag = sprintf('%.0f-%.0fs', collectStart, collectEnd);
end
logTag = '';
if useLog10D2
  logTag = '_log10d2';
end
plotBase = sprintf('metric_corr_across_sessions_%s_win%s_av%s_%s%s%s', ...
  areaTag, winTag, avTag, timeTag, logTag, format_subsamp_file_tag(useSubsampling, nNeuronsSubsample));
end

function plot_multimetric_d2_tau_alpha_across_tasks(arPlotData, avPlotData, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    plotConfig, anchorMetric, engagementTag, metricsToPlot, sharedByArea, useAnchorAffineMap, ...
    useSubsampling, nNeuronsSubsample, avWindow)
% PLOT_MULTIMETRIC_D2_TAU_ALPHA_ACROSS_TASKS - Aligned d2/tau/alpha session plot
%
% Variables:
%   useAnchorAffineMap - If true, affine-map non-anchor metrics onto anchorMetric
sessionTypes = order_manuscript_session_types(sessionTypes);

if nargin < 11 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 12 || isempty(anchorMetric)
  anchorMetric = 'd2';
end
if nargin < 13 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 14 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
end
if nargin < 15 || isempty(sharedByArea)
  sharedByArea = struct();
end
if nargin < 16 || isempty(useAnchorAffineMap)
  useAnchorAffineMap = true;
end
if nargin < 17 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 18 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 19
  avWindow = [];
end
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
anchorMetric = lower(char(anchorMetric));
validAnchors = {'d2', 'tau', 'alpha'};
if useAnchorAffineMap
  if ~ismember(anchorMetric, validAnchors)
    error('anchorMetric must be one of: %s', strjoin(validAnchors, ', '));
  end
  if ~ismember(anchorMetric, metricsToPlot)
    error('anchorMetric "%s" must be included in metricsToPlot.', anchorMetric);
  end
else
  % Primary ylabel uses first plotted metric when not anchoring
  if ~ismember(anchorMetric, metricsToPlot)
    anchorMetric = metricsToPlot{1};
  end
end
engagementTag = char(engagementTag);

if useLog10D2
  d2Label = 'log_{10}(d2)';
else
  d2Label = 'd2';
end
metricLabels = struct('d2', d2Label, 'tau', 'tau', 'alpha', 'alpha');
metricMarkers = struct('d2', 'o', 'tau', 's', 'alpha', 'd');
metricFill = struct('d2', true, 'tau', false, 'alpha', false);
anchorLabel = metricLabels.(anchorMetric);
nMetrics = numel(metricsToPlot);
xOffsets = linspace(-0.12, 0.12, max(nMetrics, 1));
if nMetrics == 1
  xOffsets = 0;
end

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:numel(areasToPlot)
  areaName = areasToPlot{a};
  areaIdxAr = find(strcmp(arPlotData.areas, areaName), 1);
  areaIdxAv = find(strcmp(avPlotData.areas, areaName), 1);
  if isempty(areaIdxAr) || isempty(areaIdxAv)
    continue;
  end

  sessionTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv, metricsToPlot, strcmpi(engagementTag, 'nonEngaged'));
  if isempty(sessionTable)
    fprintf('Skipping %s: no aligned sessions for metrics [%s].\n', ...
      areaName, strjoin(metricsToPlot, ', '));
    continue;
  end

  areaKey = matlab.lang.makeValidName(areaName);
  if isfield(sharedByArea, areaKey) && isfield(sharedByArea.(areaKey), 'maps')
    maps = sharedByArea.(areaKey).maps;
  else
    maps = compute_anchored_metric_maps(anchorMetric, sessionTable.d2Mean, ...
      sessionTable.tauMean, sessionTable.alphaMean, metricsToPlot, useAnchorAffineMap);
  end

  % Native metric values (always); display values use affine maps when anchoring
  nativeVals = struct( ...
    'd2', sessionTable.d2Mean, ...
    'tau', sessionTable.tauMean, ...
    'alpha', sessionTable.alphaMean);
  nativeSems = struct( ...
    'd2', sessionTable.d2Sem, ...
    'tau', sessionTable.tauSem, ...
    'alpha', sessionTable.alphaSem);
  yVals = struct();
  ySems = struct();
  yVals.d2 = apply_metric_affine_map(nativeVals.d2, maps.d2);
  yVals.tau = apply_metric_affine_map(nativeVals.tau, maps.tau);
  yVals.alpha = apply_metric_affine_map(nativeVals.alpha, maps.alpha);
  ySems.d2 = abs(maps.d2.gain) * nativeSems.d2;
  ySems.tau = abs(maps.tau.gain) * nativeSems.tau;
  ySems.alpha = abs(maps.alpha.gain) * nativeSems.alpha;

  fig = figure('Color', 'w', 'Name', sprintf('%s — %s', strjoin(metricsToPlot, ' '), areaName));
  position_figure_full_monitor(fig);
  axMain = axes(fig);
  hold(axMain, 'on');

  % When not affine-anchoring, still plot all markers on one axes so x positions
  % align (with slight per-metric offset). Secondary metrics are independently
  % range-mapped onto the primary display ylim; right axes show native ticks.
  if ~useAnchorAffineMap
    if isfield(sharedByArea, areaKey) && isfield(sharedByArea.(areaKey), 'maps') ...
        && isfield(sharedByArea.(areaKey), 'yLim') ...
        && numel(sharedByArea.(areaKey).yLim) == 2
      maps = sharedByArea.(areaKey).maps;
      yLimPrimary = sharedByArea.(areaKey).yLim;
    else
      yLimPrimary = compute_native_ylim_for_metric(nativeVals.(anchorMetric));
      if isempty(yLimPrimary)
        warning('criticality_multiple_metrics_across_tasks:EmptyPrimaryYLim', ...
          'Skipping %s: no finite values for anchor metric "%s".', areaName, anchorMetric);
        close(fig);
        continue;
      end
      maps = compute_independent_range_maps(anchorMetric, nativeVals, metricsToPlot, yLimPrimary);
    end
    yVals.d2 = apply_metric_affine_map(nativeVals.d2, maps.d2);
    yVals.tau = apply_metric_affine_map(nativeVals.tau, maps.tau);
    yVals.alpha = apply_metric_affine_map(nativeVals.alpha, maps.alpha);
    ySems.d2 = abs(maps.d2.gain) * nativeSems.d2;
    ySems.tau = abs(maps.tau.gain) * nativeSems.tau;
    ySems.alpha = abs(maps.alpha.gain) * nativeSems.alpha;
  end

  legendHandles = [];
  legendLabels = {};
  xCursor = 0;
  xticksCenters = [];
  xtickLabels = {};

  for t = 1:numel(sessionTypes)
    sessionType = sessionTypes{t};
    rowMask = strcmp(sessionTable.sessionType, sessionType);
    if ~any(rowMask)
      continue;
    end
    taskColor = colors_for_tasks(sessionType);
    rowIdx = find(rowMask);
    numSessions = numel(rowIdx);
    xPos = xCursor + (1:numSessions);

    % Collect per-metric positions to join across sessions within this task
    metricLineX = struct('d2', [], 'tau', [], 'alpha', []);
    metricLineY = struct('d2', [], 'tau', [], 'alpha', []);

    for iSess = 1:numSessions
      for m = 1:nMetrics
        metricName = metricsToPlot{m};
        xMetric = xPos(iSess) + xOffsets(m);
        yMetric = yVals.(metricName)(rowIdx(iSess));
        if isfinite(xMetric) && isfinite(yMetric)
          metricLineX.(metricName)(end + 1) = xMetric; %#ok<AGROW>
          metricLineY.(metricName)(end + 1) = yMetric; %#ok<AGROW>
        end
      end
    end

    % Draw across-session lines first so markers sit on top
    draw_across_session_metric_lines(axMain, metricLineX, metricLineY, metricsToPlot, ...
      taskColor, plotConfig);

    for iSess = 1:numSessions
      for m = 1:nMetrics
        metricName = metricsToPlot{m};
        faceColor = taskColor;
        if ~metricFill.(metricName)
          faceColor = 'none';
        end
        % Shared x center per session; slight metric offset for visibility
        xMetric = xPos(iSess) + xOffsets(m);
        yMetric = yVals.(metricName)(rowIdx(iSess));
        ySem = ySems.(metricName)(rowIdx(iSess));
        if ~(isfinite(xMetric) && isfinite(yMetric))
          continue;
        end
        hMetric = plot_metric_errorbar_group(axMain, xMetric, yMetric, ySem, ...
          metricMarkers.(metricName), taskColor, faceColor, plotConfig);
        if isempty(legendHandles) || ~ismember(metricLabels.(metricName), legendLabels)
          legendHandles(end + 1) = hMetric; %#ok<AGROW>
          legendLabels{end + 1} = metricLabels.(metricName); %#ok<AGROW>
        end
      end
    end

    for i = 1:numSessions
      xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
      xtickLabels{end + 1} = char(sessionTable.sessionLabel(rowIdx(i))); %#ok<AGROW>
    end
    xCursor = xPos(end) + 1.5;
  end

  if isempty(xticksCenters)
    warning('criticality_multiple_metrics_across_tasks:EmptyXTicks', ...
      'Skipping %s: no session x-ticks after layout (aligned table may lack matching sessionTypes).', ...
      areaName);
    close(fig);
    continue;
  end
  xLimPlot = [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8];

  if useAnchorAffineMap
    if isfield(sharedByArea, areaKey) && isfield(sharedByArea.(areaKey), 'yLim') ...
        && numel(sharedByArea.(areaKey).yLim) == 2
      yLimPlot = sharedByArea.(areaKey).yLim;
    else
      yLimPlot = compute_display_ylim_for_metrics(yVals, metricsToPlot, anchorMetric);
    end
  else
    if isfield(sharedByArea, areaKey) && isfield(sharedByArea.(areaKey), 'yLim') ...
        && numel(sharedByArea.(areaKey).yLim) == 2
      yLimPlot = sharedByArea.(areaKey).yLim;
    else
      yLimPlot = compute_native_ylim_for_metric(nativeVals.(anchorMetric));
    end
  end
  if isempty(yLimPlot) || ~all(isfinite(yLimPlot))
    warning('criticality_multiple_metrics_across_tasks:EmptyYLim', ...
      'Skipping %s: could not compute y-limits for anchor metric "%s".', ...
      areaName, anchorMetric);
    close(fig);
    continue;
  end
  ylim(axMain, yLimPlot);
  xlim(axMain, xLimPlot);
  set(axMain, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45);
  grid(axMain, 'off');
  xlabel(axMain, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
  ylabel(axMain, anchorLabel, 'FontSize', plotConfig.axisLabelFontSize, ...
    'Interpreter', ternary_metric_label_interpreter(anchorLabel));
  set(axMain, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
    'Box', 'off', 'TickDir', 'out');

  % Right-side native axes for non-primary metrics (affine or independent range maps)
  rightOffset = 1.0;
  for m = 1:nMetrics
    metricName = metricsToPlot{m};
    if strcmp(metricName, anchorMetric)
      continue;
    end
    add_affine_metric_yaxis(axMain, maps.(metricName), 'right', metricLabels.(metricName), ...
      plotConfig, rightOffset);
    rightOffset = rightOffset + 0.1;
  end

  if ~isempty(legendHandles)
    legend(axMain, legendHandles, legendLabels, 'Location', 'best', ...
      'FontSize', plotConfig.legendFontSize);
  end
  hold(axMain, 'off');

  if useAnchorAffineMap
    fprintf('  Anchor: %s | maps (display = gain * metric + offset):\n', anchorMetric);
  else
    fprintf('  Native scales (independent range maps onto %s display; right axes):\n', ...
      anchorMetric);
  end
  for m = 1:nMetrics
    metricName = metricsToPlot{m};
    fprintf('    %s: gain=%.4g, offset=%.4g\n', metricName, maps.(metricName).gain, ...
      maps.(metricName).offset);
  end

  collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
  winPhrase = format_d2_av_window_title_phrase(d2Window, avWindow);
  engTitle = format_engagement_title_tag(engagementTag);
  metricTitle = strjoin(metricsToPlot, ', ');
  if useAnchorAffineMap
    scaleTag = sprintf('anchor=%s', anchorMetric);
  else
    scaleTag = 'native scales';
  end
  if ~isempty(brainArea)
    titleStr = sprintf('%s (%s)%s — %s [%s, %s]', ...
      metricTitle, scaleTag, engTitle, brainArea, collectTag, winPhrase);
  else
    titleStr = sprintf('%s (%s)%s — %s [%s, %s]', ...
      metricTitle, scaleTag, engTitle, areaName, collectTag, winPhrase);
  end
  titleStr = append_subsamp_title_tag(titleStr, useSubsampling, nNeuronsSubsample);
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_multimetric_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, anchorMetric, engagementTag, metricsToPlot, ...
    useAnchorAffineMap, useSubsampling, nNeuronsSubsample, avWindow);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
end

fprintf('\nAll combined figures saved to %s\n', saveDir);
end

function plot_multimetric_separated_axes_across_tasks(arPlotData, avPlotData, prgPlotData, ...
    areasToPlot, sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, ...
    useLog10D2, plotConfig, engagementTag, metricsToPlot, avPlotDataDecades, ...
    finalCutoffDivisor, enablePermutations, useSubsampling, nNeuronsSubsample, activeAnalyses, ...
    avWindow, binSizeD2)
% PLOT_MULTIMETRIC_SEPARATED_AXES_ACROSS_TASKS - 2x4 panels of session metrics
%
% Layout:
%   Top:    D2 | Avalanche Sizes | Avalanche Durations | Crackling 1/σνz
%   Bottom: Scale Range | dcc | Renorm: Kurtosis | Renorm: JS-Distance
sessionTypes = order_manuscript_session_types(sessionTypes);
%
% Variables:
%   arPlotData / avPlotData - Sources for d2 / tau / alpha / paramSD / dcc
%                             (may be engagement views)
%   prgPlotData             - PRG plotData for kurtosis (kappaMean) and D_JS
%   avPlotDataDecades       - AV plotData used for decades (defaults to avPlotData;
%                             pass engagement AV view so scale-free range splits)
%   metricsToPlot           - Controls which of d2/tau/alpha appear in the top row
%   finalCutoffDivisor      - PRG kappa reported at N/finalCutoffDivisor (ylabel)
%   enablePermutations      - If true, overlay shuffled/surrogate means in gray
%   activeAnalyses          - Struct .ar/.av/.prg; inactive panels stay blank
%   binSizeD2               - d2 spike bin width (s); included in title / filename
%
% Goal:
%   Same session-level data as the combined plot, each metric on its own axis.
%   Within each task type, a horizontal line marks the within-task mean from
%   first to last session. Optional gray markers show shuffle/surrogate summaries.

if nargin < 12 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 13 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 14 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
end
if nargin < 15 || isempty(avPlotDataDecades)
  avPlotDataDecades = avPlotData;
end
if nargin < 16 || isempty(finalCutoffDivisor)
  if isfield(prgPlotData, 'finalCutoffDivisor') && ~isempty(prgPlotData.finalCutoffDivisor)
    finalCutoffDivisor = prgPlotData.finalCutoffDivisor;
  else
    finalCutoffDivisor = 4;
  end
end
if nargin < 17 || isempty(enablePermutations)
  enablePermutations = false;
end
if nargin < 18 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 19 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 20 || isempty(activeAnalyses)
  activeAnalyses = struct('ar', true, 'av', true, 'prg', true);
end
if nargin < 21
  avWindow = [];
end
if nargin < 22 || isempty(binSizeD2)
  binSizeD2 = [];
end
activeAnalyses = fill_active_analyses(activeAnalyses);
enablePermutations = logical(enablePermutations);
if isempty(metricsToPlot)
  metricsToPlot = {};
else
  metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
end
engagementTag = char(engagementTag);

activePanelKeys = {};
if activeAnalyses.ar
  activePanelKeys = [activePanelKeys, {'d2'}]; %#ok<AGROW>
end
if activeAnalyses.av
  activePanelKeys = [activePanelKeys, {'tau', 'alpha', 'paramSD', 'decades', 'dcc'}]; %#ok<AGROW>
end
if activeAnalyses.prg
  activePanelKeys = [activePanelKeys, {'kurtosis', 'djs'}]; %#ok<AGROW>
end
% Top-row d2/tau/alpha also respect metricsToPlot when provided
if ~isempty(metricsToPlot)
  for topKey = {'d2', 'tau', 'alpha'}
    if ~ismember(topKey{1}, metricsToPlot)
      activePanelKeys = setdiff(activePanelKeys, topKey, 'stable');
    end
  end
end

if useLog10D2
  d2Label = 'log_{10}(d2)';
else
  d2Label = 'd2';
end
kurtosisLabel = sprintf('kurtosis (N = %d)', finalCutoffDivisor);
djsLabel = sprintf('D_{JS} (N = %d)', finalCutoffDivisor);
paramSdLabel = '1/\sigma\nu z';

% Fixed 2x4 panel order (crackling next to exponents / decades)
panelMetricKeys = {'d2', 'tau', 'alpha', 'paramSD', 'decades', 'dcc', 'kurtosis', 'djs'};
labelByKey = struct('d2', d2Label, 'tau', 'tau', 'alpha', 'alpha', ...
  'paramSD', paramSdLabel, 'decades', 'decades', 'dcc', 'dcc', ...
  'kurtosis', kurtosisLabel, 'djs', djsLabel);
titleByKey = struct( ...
  'd2', 'D2', ...
  'tau', 'Avalanche Sizes', ...
  'alpha', 'Avalanche Durations', ...
  'paramSD', 'Crackling 1/\sigma\nu z', ...
  'decades', 'Scale-Free Range', ...
  'dcc', 'Distance to Criticality', ...
  'kurtosis', 'Renorm: Kurtosis', ...
  'djs', 'Renorm: JS-Distance');
fieldByKey = struct('d2', 'd2Mean', 'tau', 'tauMean', 'alpha', 'alphaMean', ...
  'paramSD', 'paramSD', 'decades', 'decades', 'dcc', 'dcc', ...
  'kurtosis', 'kurtosis', 'djs', 'djs');
semByKey = struct('d2', 'd2Sem', 'tau', 'tauSem', 'alpha', 'alphaSem', ...
  'paramSD', 'paramSDSem', 'decades', 'decadesSem', 'dcc', 'dccSem', ...
  'kurtosis', 'kurtosisSem', 'djs', 'djsSem');
shuffleFieldByKey = struct('d2', 'd2ShuffleMean', 'tau', 'tauShuffleMean', ...
  'alpha', 'alphaShuffleMean', 'paramSD', 'paramSDShuffleMean', ...
  'decades', 'decadesShuffleMean', 'dcc', 'dccShuffleMean', ...
  'kurtosis', 'kurtosisShuffleMean', 'djs', 'djsShuffleMean');
shuffleSemByKey = struct('d2', 'd2ShuffleSem', 'tau', 'tauShuffleSem', ...
  'alpha', 'alphaShuffleSem', 'paramSD', 'paramSDShuffleSem', ...
  'decades', 'decadesShuffleSem', 'dcc', 'dccShuffleSem', ...
  'kurtosis', 'kurtosisShuffleSem', 'djs', 'djsShuffleSem');
% d2 / paramSD / decades / dcc / kurtosis / Djs: filled circles; tau square; alpha diamond
markerByKey = struct('d2', 'o', 'tau', 's', 'alpha', 'd', ...
  'paramSD', 'o', 'decades', 'o', 'dcc', 'o', 'kurtosis', 'o', 'djs', 'o');
fillByKey = struct('d2', true, 'tau', false, 'alpha', false, ...
  'paramSD', true, 'decades', true, 'dcc', true, 'kurtosis', true, 'djs', true);
shuffleColor = [0.55, 0.55, 0.55];
shuffleXOffset = 0.22;

nPanels = numel(panelMetricKeys);
nCols = 4;
nRows = 2;
panelYFields = cell(1, nPanels);
panelSemFields = cell(1, nPanels);
panelLabels = cell(1, nPanels);
panelTitles = cell(1, nPanels);
for iPanel = 1:nPanels
  key = panelMetricKeys{iPanel};
  panelYFields{iPanel} = fieldByKey.(key);
  panelSemFields{iPanel} = semByKey.(key);
  panelLabels{iPanel} = labelByKey.(key);
  panelTitles{iPanel} = titleByKey.(key);
end

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:numel(areasToPlot)
  areaName = areasToPlot{a};
  areaIdxAr = find(strcmp(arPlotData.areas, areaName), 1);
  areaIdxAv = find(strcmp(avPlotData.areas, areaName), 1);
  areaIdxAvDec = find(strcmp(avPlotDataDecades.areas, areaName), 1);
  areaIdxPrg = find(strcmp(prgPlotData.areas, areaName), 1);
  hasArea = (activeAnalyses.ar && ~isempty(areaIdxAr)) ...
    || (activeAnalyses.av && ~isempty(areaIdxAv)) ...
    || (activeAnalyses.prg && ~isempty(areaIdxPrg));
  if ~hasArea
    continue;
  end
  if isempty(areaIdxAr), areaIdxAr = 1; end
  if isempty(areaIdxAv), areaIdxAv = 1; end
  if isempty(areaIdxAvDec)
    areaIdxAvDec = areaIdxAv;
    avPlotDataDecades = avPlotData;
  end
  if isempty(areaIdxPrg), areaIdxPrg = 1; end

  topMetrics = metricsToPlot;
  if isempty(topMetrics)
    if activeAnalyses.ar
      topMetrics = {'d2'};
    elseif activeAnalyses.av
      topMetrics = {'tau'};
    else
      topMetrics = {};  % PRG-only: session list from PRG
    end
  end
  sessionTable = build_separated_metrics_session_table( ...
    arPlotData, avPlotData, prgPlotData, avPlotDataDecades, sessionTypes, ...
    areaIdxAr, areaIdxAv, areaIdxPrg, areaIdxAvDec, topMetrics, ...
    strcmpi(engagementTag, 'nonEngaged') || numel(activePanelKeys) < 8);
  if isempty(sessionTable)
    fprintf('Skipping separated metrics for %s: no aligned sessions.\n', areaName);
    continue;
  end

  fig = figure('Color', 'w', 'Name', sprintf('Separated metrics — %s', areaName));
  position_figure_full_monitor(fig);

  for iPanel = 1:nPanels
    metricKey = panelMetricKeys{iPanel};
    ax = subplot(nRows, nCols, iPanel, 'Parent', fig);
    hold(ax, 'on');
    xCursor = 0;
    xticksCenters = [];
    xtickLabels = {};
    yField = panelYFields{iPanel};
    semField = panelSemFields{iPanel};
    shuffleField = shuffleFieldByKey.(metricKey);
    shuffleSemField = shuffleSemByKey.(metricKey);
    panelIsActive = ismember(metricKey, activePanelKeys);
    yLimVals = sessionTable.(yField)(:);
    if ~panelIsActive
      yLimVals = nan(size(yLimVals));
    end
    if panelIsActive && enablePermutations ...
        && ismember(shuffleField, sessionTable.Properties.VariableNames)
      yLimVals = [yLimVals; sessionTable.(shuffleField)(:)]; %#ok<AGROW>
    end

    for t = 1:numel(sessionTypes)
      sessionType = sessionTypes{t};
      rowMask = strcmp(sessionTable.sessionType, sessionType);
      if ~any(rowMask)
        continue;
      end
      taskColor = colors_for_tasks(sessionType);
      rowIdx = find(rowMask);
      numSessions = numel(rowIdx);
      xPos = xCursor + (1:numSessions);
      yPos = sessionTable.(yField)(rowIdx);
      ySem = sessionTable.(semField)(rowIdx);
      if ~panelIsActive
        yPos = nan(size(yPos));
        ySem = nan(size(ySem));
      end

      faceColor = taskColor;
      if ~fillByKey.(metricKey)
        faceColor = 'none';
      end

      % Horizontal mean across sessions within this task (first → last x)
      yMeanTask = mean(yPos(isfinite(yPos)));
      if isfinite(yMeanTask) && numSessions >= 1
        plot(ax, [xPos(1), xPos(end)], [yMeanTask, yMeanTask], '-', ...
          'Color', taskColor, ...
          'LineWidth', 2, ...
          'HandleVisibility', 'off');
      end

      if panelIsActive && enablePermutations ...
          && ismember(shuffleField, sessionTable.Properties.VariableNames)
        yShuffle = sessionTable.(shuffleField)(rowIdx);
        yShuffleSem = sessionTable.(shuffleSemField)(rowIdx);
        if any(isfinite(yShuffle))
          plot_metric_errorbar_group(ax, xPos + shuffleXOffset, yShuffle, yShuffleSem, ...
            markerByKey.(metricKey), shuffleColor, shuffleColor, plotConfig);
        end
      end

      plot_metric_errorbar_group(ax, xPos, yPos, ySem, ...
        markerByKey.(metricKey), taskColor, faceColor, plotConfig);

      for i = 1:numSessions
        xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
        xtickLabels{end + 1} = char(sessionTable.sessionLabel(rowIdx(i))); %#ok<AGROW>
      end
      xCursor = xPos(end) + 1.5;
    end

    yLimPlot = compute_native_ylim_for_metric(yLimVals);
    if ~isempty(yLimPlot)
      ylim(ax, yLimPlot);
    end
    if ~isempty(xticksCenters)
      xlim(ax, [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8]);
      set(ax, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45);
    end
    xlabel(ax, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
    ylabel(ax, panelLabels{iPanel}, 'FontSize', plotConfig.axisLabelFontSize, ...
      'Interpreter', ternary_metric_label_interpreter(panelLabels{iPanel}));
    title(ax, panelTitles{iPanel}, 'FontSize', plotConfig.titleFontSize, ...
      'Interpreter', ternary_metric_label_interpreter(panelTitles{iPanel}));
    set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
      'Box', 'off', 'TickDir', 'out');
    hold(ax, 'off');
  end

  collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
  winPhrase = format_d2_av_window_title_phrase(d2Window, avWindow);
  engTitle = format_engagement_title_tag(engagementTag);
  binPhrase = format_bin_size_title_phrase(binSizeD2);
  if ~isempty(brainArea)
    titleStr = sprintf('Separated metrics%s — %s [%s, %s%s]', ...
      engTitle, brainArea, collectTag, winPhrase, binPhrase);
  else
    titleStr = sprintf('Separated metrics%s — %s [%s, %s%s]', ...
      engTitle, areaName, collectTag, winPhrase, binPhrase);
  end
  if enablePermutations
    titleStr = sprintf('%s (gray = shuffled)', titleStr);
  end
  titleStr = append_subsamp_title_tag(titleStr, useSubsampling, nNeuronsSubsample);
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_separated_metrics_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, engagementTag, ...
    useSubsampling, nNeuronsSubsample, avWindow, binSizeD2);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved separated metrics: %s\n', fullfile(saveDir, plotBase));
end
end

function sessionTable = build_separated_metrics_session_table(arPlotData, avPlotData, ...
    prgPlotData, avPlotDataDecades, sessionTypes, areaIdxAr, areaIdxAv, areaIdxPrg, ...
    areaIdxAvDec, topMetrics, keepBlankSessions)
% BUILD_SEPARATED_METRICS_SESSION_TABLE - d2/tau/alpha + crackling + PRG metrics
%
% Variables:
%   avPlotDataDecades - AV source for decades (may differ from avPlotData)
%   topMetrics        - Which of d2/tau/alpha must be finite to keep a session
%                       (ignored when keepBlankSessions is true)
%   keepBlankSessions - If true, keep sessions with NaN top-row metrics (blank slots)
%
% Goal:
%   Align top-row metrics with paramSD/dcc (AV), decades (AV), and PRG kurtosis / D_JS.
%   Also collect shuffle/surrogate session summaries when present in plotData.

if nargin < 10 || isempty(topMetrics)
  topMetrics = {'d2', 'tau', 'alpha'};
end
if nargin < 11 || isempty(keepBlankSessions)
  keepBlankSessions = false;
end
if isempty(topMetrics)
  baseTable = table();
else
  topMetrics = normalize_metrics_to_plot(topMetrics);
  baseTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv, topMetrics, keepBlankSessions);
end
if isempty(baseTable)
  baseTable = build_prg_only_session_base_table(prgPlotData, sessionTypes, areaIdxPrg);
end
if isempty(baseTable)
  sessionTable = baseTable;
  return;
end

nRow = height(baseTable);
paramSDCol = nan(nRow, 1);
paramSDSemCol = zeros(nRow, 1);
dccCol = nan(nRow, 1);
dccSemCol = zeros(nRow, 1);
decadesCol = nan(nRow, 1);
decadesSemCol = zeros(nRow, 1);
kurtosisCol = nan(nRow, 1);
kurtosisSemCol = zeros(nRow, 1);
djsCol = nan(nRow, 1);
djsSemCol = zeros(nRow, 1);

d2ShuffleMeanCol = nan(nRow, 1);
d2ShuffleSemCol = nan(nRow, 1);
tauShuffleMeanCol = nan(nRow, 1);
tauShuffleSemCol = nan(nRow, 1);
alphaShuffleMeanCol = nan(nRow, 1);
alphaShuffleSemCol = nan(nRow, 1);
paramSDShuffleMeanCol = nan(nRow, 1);
paramSDShuffleSemCol = nan(nRow, 1);
dccShuffleMeanCol = nan(nRow, 1);
dccShuffleSemCol = nan(nRow, 1);
decadesShuffleMeanCol = nan(nRow, 1);
decadesShuffleSemCol = nan(nRow, 1);
kurtosisShuffleMeanCol = nan(nRow, 1);
kurtosisShuffleSemCol = nan(nRow, 1);
djsShuffleMeanCol = nan(nRow, 1);
djsShuffleSemCol = nan(nRow, 1);

for i = 1:nRow
  sessionType = baseTable.sessionType{i};
  sessionName = baseTable.sessionName{i};
  typeKey = matlab.lang.makeValidName(sessionType);
  % Within-type index for fallback (not the global table row)
  typeRows = find(strcmp(baseTable.sessionType, sessionType));
  withinTypeIdx = find(typeRows == i, 1);

  if isfield(arPlotData.byType, typeKey)
    arType = arPlotData.byType.(typeKey);
    arIdx = find_matching_session_index(arType, sessionName, withinTypeIdx);
    if ~isempty(arIdx)
      d2ShuffleMeanCol(i) = get_type_cell_metric(arType, 'd2ShuffleMean', areaIdxAr, arIdx);
      d2ShuffleSemCol(i) = get_type_cell_metric(arType, 'd2ShuffleSem', areaIdxAr, arIdx);
    end
  end

  if isfield(avPlotData.byType, typeKey)
    avType = avPlotData.byType.(typeKey);
    avIdx = find_matching_session_index(avType, sessionName, withinTypeIdx);
    if ~isempty(avIdx)
      paramSDCol(i) = get_type_cell_metric(avType, 'paramSD', areaIdxAv, avIdx);
      paramSDSemCol(i) = get_type_cell_metric(avType, 'paramSDSem', areaIdxAv, avIdx);
      dccCol(i) = get_type_cell_metric(avType, 'dcc', areaIdxAv, avIdx);
      dccSemCol(i) = get_type_cell_metric(avType, 'dccSem', areaIdxAv, avIdx);
      tauShuffleMeanCol(i) = get_type_cell_metric(avType, 'tauPermutedMean', areaIdxAv, avIdx);
      alphaShuffleMeanCol(i) = get_type_cell_metric(avType, 'alphaPermutedMean', areaIdxAv, avIdx);
      paramSDShuffleMeanCol(i) = get_type_cell_metric(avType, 'paramSDPermutedMean', areaIdxAv, avIdx);
      dccShuffleMeanCol(i) = get_type_cell_metric(avType, 'dccPermutedMean', areaIdxAv, avIdx);
      if ~isfinite(paramSDSemCol(i)), paramSDSemCol(i) = 0; end
      if ~isfinite(dccSemCol(i)), dccSemCol(i) = 0; end
    end
  end

  if isfield(avPlotDataDecades.byType, typeKey)
    avDecType = avPlotDataDecades.byType.(typeKey);
    avDecIdx = find_matching_session_index(avDecType, sessionName, withinTypeIdx);
    if ~isempty(avDecIdx)
      decadesCol(i) = get_type_cell_metric(avDecType, 'decades', areaIdxAvDec, avDecIdx);
      decadesSemCol(i) = get_type_cell_metric(avDecType, 'decadesSem', areaIdxAvDec, avDecIdx);
      decadesShuffleMeanCol(i) = get_type_cell_metric(avDecType, 'decadesPermutedMean', ...
        areaIdxAvDec, avDecIdx);
      if ~isfinite(decadesSemCol(i)), decadesSemCol(i) = 0; end
    end
  end

  if isfield(prgPlotData.byType, typeKey)
    prgType = prgPlotData.byType.(typeKey);
    prgIdx = find_matching_session_index(prgType, sessionName, withinTypeIdx);
    if ~isempty(prgIdx)
      kurtosisCol(i) = get_type_cell_metric(prgType, 'kappaMean', areaIdxPrg, prgIdx);
      kurtosisSemCol(i) = get_type_cell_metric(prgType, 'kappaSem', areaIdxPrg, prgIdx);
      djsCol(i) = get_type_cell_metric(prgType, 'djsMean', areaIdxPrg, prgIdx);
      djsSemCol(i) = get_type_cell_metric(prgType, 'djsSem', areaIdxPrg, prgIdx);
      kurtosisShuffleMeanCol(i) = get_type_cell_metric(prgType, 'kappaShuffleMean', ...
        areaIdxPrg, prgIdx);
      kurtosisShuffleSemCol(i) = get_type_cell_metric(prgType, 'kappaShuffleSem', ...
        areaIdxPrg, prgIdx);
      djsShuffleMeanCol(i) = get_type_cell_metric(prgType, 'djsShuffleMean', areaIdxPrg, prgIdx);
      djsShuffleSemCol(i) = get_type_cell_metric(prgType, 'djsShuffleSem', areaIdxPrg, prgIdx);
      if ~isfinite(kurtosisSemCol(i)), kurtosisSemCol(i) = 0; end
      if ~isfinite(djsSemCol(i)), djsSemCol(i) = 0; end
    end
  end
end

sessionTable = [baseTable, table(paramSDCol, paramSDSemCol, dccCol, dccSemCol, ...
  decadesCol, decadesSemCol, kurtosisCol, kurtosisSemCol, djsCol, djsSemCol, ...
  d2ShuffleMeanCol, d2ShuffleSemCol, tauShuffleMeanCol, tauShuffleSemCol, ...
  alphaShuffleMeanCol, alphaShuffleSemCol, paramSDShuffleMeanCol, paramSDShuffleSemCol, ...
  dccShuffleMeanCol, dccShuffleSemCol, decadesShuffleMeanCol, decadesShuffleSemCol, ...
  kurtosisShuffleMeanCol, kurtosisShuffleSemCol, djsShuffleMeanCol, djsShuffleSemCol, ...
  'VariableNames', {'paramSD', 'paramSDSem', 'dcc', 'dccSem', ...
  'decades', 'decadesSem', 'kurtosis', 'kurtosisSem', 'djs', 'djsSem', ...
  'd2ShuffleMean', 'd2ShuffleSem', 'tauShuffleMean', 'tauShuffleSem', ...
  'alphaShuffleMean', 'alphaShuffleSem', 'paramSDShuffleMean', 'paramSDShuffleSem', ...
  'dccShuffleMean', 'dccShuffleSem', 'decadesShuffleMean', 'decadesShuffleSem', ...
  'kurtosisShuffleMean', 'kurtosisShuffleSem', 'djsShuffleMean', 'djsShuffleSem'})];
end

function plot_multimetric_pair_scatters_across_tasks(arPlotData, avPlotData, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    plotConfig, engagementTag, useSubsampling, nNeuronsSubsample, activeAnalyses, avWindow)
% PLOT_MULTIMETRIC_PAIR_SCATTERS_ACROSS_TASKS - 2x2 session metric scatters
%
% Panels (titles are Y vs X):
%   (1,1) Avalanche Sizes vs D2
%   (1,2) Avalanche Durations vs D2
%   (2,1) Measured vs Predicted Crackling Exponent
%   (2,2) Crackling Exponent vs D2
sessionTypes = order_manuscript_session_types(sessionTypes);
%
% Crackling naming:
%   Measured / observed crackling exponent = paramSD = 1/σνz from WLS ⟨S⟩(T)
%     (avalanches restricted to the duration power-law fit range [minavD, maxavD]
%     used for α — see avalanche_power_law_metrics / size_given_duration)
%   Predicted crackling exponent = (α-1)/(τ-1) from the size/duration PDFs
%   dcc = |predicted - measured|
%
% Variables:
%   arPlotData / avPlotData - Aggregated plotData from AR / AV batches
%   areasToPlot             - Brain areas to plot (one figure each)
%   sessionTypes            - Session types (points colored by type)
%   engagementTag           - Optional engaged / nonEngaged suffix
%   activeAnalyses          - Struct .ar/.av; inactive metrics stay blank
%
% Goal:
%   Cross-session relationships among d2 and avalanche / crackling exponents.

if nargin < 11 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 12 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 13 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 14 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 15 || isempty(activeAnalyses)
  activeAnalyses = struct('ar', true, 'av', true, 'prg', false);
end
if nargin < 16
  avWindow = [];
end
activeAnalyses = fill_active_analyses(activeAnalyses);
engagementTag = char(engagementTag);
if ~activeAnalyses.ar && ~activeAnalyses.av
  fprintf('Skipping pair scatters (requires AR and/or AV).\n');
  return;
end

if useLog10D2
  d2Label = 'log_{10}(d2)';
else
  d2Label = 'd2';
end
% Measured γ from ⟨S⟩~T^γ (size_given_duration); predicted γ from τ, α
measuredCracklingLabel = 'Measured 1/\sigma\nu z';
predictedCracklingLabel = 'Predicted (\alpha-1)/(\tau-1)';

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:numel(areasToPlot)
  areaName = areasToPlot{a};
  areaIdxAr = find(strcmp(arPlotData.areas, areaName), 1);
  areaIdxAv = find(strcmp(avPlotData.areas, areaName), 1);
  hasArea = (activeAnalyses.ar && ~isempty(areaIdxAr)) ...
    || (activeAnalyses.av && ~isempty(areaIdxAv));
  if ~hasArea
    continue;
  end
  if isempty(areaIdxAr), areaIdxAr = 1; end
  if isempty(areaIdxAv), areaIdxAv = 1; end

  sessionTable = build_pair_scatter_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv, activeAnalyses);
  if isempty(sessionTable)
    fprintf('Skipping pair scatters for %s: no aligned sessions.\n', areaName);
    continue;
  end

  fig = figure('Color', 'w', 'Name', sprintf('Metric pair scatters — %s', areaName));
  position_figure_full_monitor(fig);
  legendHandles = gobjects(0);
  legendLabels = {};

  % Columns: tile, xField, yField, xLabel, yLabel, drawIdentity, title (Y vs X)
  panelSpecs = { ...
    1, 'd2Mean', 'tauMean', d2Label, 'tau', false, 'Avalanche Sizes vs D2'
    2, 'd2Mean', 'alphaMean', d2Label, 'alpha', false, 'Avalanche Durations vs D2'
    3, 'gammaPred', 'paramSD', predictedCracklingLabel, measuredCracklingLabel, true, ...
      'Measured vs Predicted Crackling Exponent'
    4, 'd2Mean', 'paramSD', d2Label, measuredCracklingLabel, false, ...
      'Crackling Exponent vs D2'
    };

  for iPair = 1:4
    ax = subplot(2, 2, iPair, 'Parent', fig);
    hold(ax, 'on');
    xField = panelSpecs{iPair, 2};
    yField = panelSpecs{iPair, 3};
    xLabel = panelSpecs{iPair, 4};
    yLabel = panelSpecs{iPair, 5};
    drawIdentity = panelSpecs{iPair, 6};
    panelTitle = panelSpecs{iPair, 7};

    for t = 1:numel(sessionTypes)
      sessionType = sessionTypes{t};
      rowMask = strcmp(sessionTable.sessionType, sessionType);
      if ~any(rowMask)
        continue;
      end
      xVals = sessionTable.(xField)(rowMask);
      yVals = sessionTable.(yField)(rowMask);
      valid = isfinite(xVals) & isfinite(yVals);
      if ~any(valid)
        continue;
      end
      taskColor = colors_for_tasks(sessionType);
      plotConfig.scatterMarkerSize = 180;
      plotConfig.markerFaceAlpha = 0.8;
      hSc = scatter_manuscript_filled(ax, xVals(valid), yVals(valid), plotConfig, ...
        taskColor, sessionType);
      if iPair == 1 && (isempty(legendLabels) || ~ismember(sessionType, legendLabels))
        legendHandles(end + 1) = hSc; %#ok<AGROW>
        legendLabels{end + 1} = sessionType; %#ok<AGROW>
      end
    end

    if drawIdentity
      add_identity_line_to_axes(ax);
    end

    xlabel(ax, xLabel, 'FontSize', plotConfig.axisLabelFontSize, ...
      'Interpreter', ternary_metric_label_interpreter(xLabel));
    ylabel(ax, yLabel, 'FontSize', plotConfig.axisLabelFontSize, ...
      'Interpreter', ternary_metric_label_interpreter(yLabel));
    title(ax, panelTitle, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
    set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
      'Box', 'off', 'TickDir', 'out');
    hold(ax, 'off');
  end

  if ~isempty(legendHandles)
    legend(legendHandles, legendLabels, 'Location', 'best', ...
      'FontSize', plotConfig.legendFontSize);
  end

  collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
  winPhrase = format_d2_av_window_title_phrase(d2Window, avWindow);
  engTitle = format_engagement_title_tag(engagementTag);
  if ~isempty(brainArea)
    titleStr = sprintf('Session metric pairs%s — %s [%s, %s]', ...
      engTitle, brainArea, collectTag, winPhrase);
  else
    titleStr = sprintf('Session metric pairs%s — %s [%s, %s]', ...
      engTitle, areaName, collectTag, winPhrase);
  end
  titleStr = append_subsamp_title_tag(titleStr, useSubsampling, nNeuronsSubsample);
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_pair_scatter_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, engagementTag, useSubsampling, nNeuronsSubsample, ...
    avWindow);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved pair scatters: %s\n', fullfile(saveDir, plotBase));
end
end

function sessionTable = build_pair_scatter_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv, activeAnalyses)
% BUILD_PAIR_SCATTER_SESSION_TABLE - d2/tau/alpha + paramSD and γ_pred
%
% Goal:
%   Align AR d2 with AV tau, alpha, and crackling paramSD (1/σνz). Also store
%   predicted crackling exponent (α-1)/(τ-1) for the identity-line panel.
%   Measured paramSD comes from the AV batch, where ⟨S⟩(T) was fit only on
%   durations inside the α power-law range (same range used for alpha).

if nargin < 6 || isempty(activeAnalyses)
  activeAnalyses = struct('ar', true, 'av', true, 'prg', false);
end
activeAnalyses = fill_active_analyses(activeAnalyses);
topMetrics = {};
if activeAnalyses.ar
  topMetrics{end + 1} = 'd2'; %#ok<AGROW>
end
if activeAnalyses.av
  topMetrics = [topMetrics, {'tau', 'alpha'}]; %#ok<AGROW>
end
if isempty(topMetrics)
  sessionTable = table();
  return;
end

baseTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
  areaIdxAr, areaIdxAv, topMetrics, true);
if isempty(baseTable)
  sessionTable = baseTable;
  return;
end

nRow = height(baseTable);
paramSDCol = nan(nRow, 1);
gammaPredCol = nan(nRow, 1);

for i = 1:nRow
  sessionType = baseTable.sessionType{i};
  sessionName = baseTable.sessionName{i};
  typeKey = matlab.lang.makeValidName(sessionType);
  typeRows = find(strcmp(baseTable.sessionType, sessionType));
  withinTypeIdx = find(typeRows == i, 1);

  if isfield(avPlotData.byType, typeKey)
    avType = avPlotData.byType.(typeKey);
    avIdx = find_matching_session_index(avType, sessionName, withinTypeIdx);
    if ~isempty(avIdx)
      paramSDCol(i) = get_type_cell_metric(avType, 'paramSD', areaIdxAv, avIdx);
    end
  end

  tauVal = baseTable.tauMean(i);
  alphaVal = baseTable.alphaMean(i);
  if isfinite(tauVal) && isfinite(alphaVal) && tauVal > 1
    gammaPredCol(i) = (alphaVal - 1) / (tauVal - 1);
  end
end

sessionTable = [baseTable, table(paramSDCol, gammaPredCol, ...
  'VariableNames', {'paramSD', 'gammaPred'})];
end

function add_identity_line_to_axes(ax)
% ADD_IDENTITY_LINE_TO_AXES - y=x reference over current finite data limits
hold(ax, 'on');
xLim = xlim(ax);
yLim = ylim(ax);
% Expand to include data children if limits are still default
lineChildren = findobj(ax, 'Type', 'Scatter');
if ~isempty(lineChildren)
  xAll = [];
  yAll = [];
  for i = 1:numel(lineChildren)
    xAll = [xAll; lineChildren(i).XData(:)]; %#ok<AGROW>
    yAll = [yAll; lineChildren(i).YData(:)]; %#ok<AGROW>
  end
  valid = isfinite(xAll) & isfinite(yAll);
  if any(valid)
    lo = min([xAll(valid); yAll(valid)]);
    hi = max([xAll(valid); yAll(valid)]);
    pad = 0.05 * max(hi - lo, eps);
    lim = [lo - pad, hi + pad];
    xlim(ax, lim);
    ylim(ax, lim);
    xLim = lim;
    yLim = lim;
  end
end
lo = max(xLim(1), yLim(1));
hi = min(xLim(2), yLim(2));
if isfinite(lo) && isfinite(hi) && hi > lo
  plot(ax, [lo, hi], [lo, hi], 'k--', 'LineWidth', 1.25, 'HandleVisibility', 'off');
end
end

function sessionTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv, metricsToPlot, keepBlankSessions)
% BUILD_MULTIMETRIC_SESSION_TABLE - Align d2/tau/alpha per session across pipelines
%
% Variables:
%   keepBlankSessions - If true, keep sessions even when requested metrics are NaN
%                       (used for non-engaged plots so x-axis slots stay aligned)

if nargin < 6 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
end
if nargin < 7 || isempty(keepBlankSessions)
  keepBlankSessions = false;
end
if isempty(metricsToPlot)
  needD2 = false;
  needTau = false;
  needAlpha = false;
else
  metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
  needD2 = ismember('d2', metricsToPlot);
  needTau = ismember('tau', metricsToPlot);
  needAlpha = ismember('alpha', metricsToPlot);
end
needAv = needTau || needAlpha;

sessionTypeCol = {};
sessionLabelCol = {};
sessionNameCol = {};
d2MeanCol = [];
d2SemCol = [];
tauMeanCol = [];
tauSemCol = [];
alphaMeanCol = [];
alphaSemCol = [];

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  hasArType = isfield(arPlotData.byType, typeKey);
  hasAvType = isfield(avPlotData.byType, typeKey);
  if needD2 && ~hasArType
    continue;
  end
  if needAv && ~hasAvType
    continue;
  end
  if ~hasArType && ~hasAvType
    continue;
  end

  arType = [];
  avType = [];
  if hasArType
    arType = arPlotData.byType.(typeKey);
  end
  if hasAvType
    avType = avPlotData.byType.(typeKey);
  end
  if needD2 && (areaIdxAr > numel(arType.d2Mean) || isempty(arType.d2Mean{areaIdxAr}))
    continue;
  end
  if needAv && (areaIdxAv > numel(avType.tau) || isempty(avType.tau{areaIdxAv}))
    continue;
  end

  % Drive session list from AR when d2 requested (or AV unavailable); else AV
  useArDriver = hasArType && (needD2 || ~needAv || ~hasAvType ...
    || (isfield(arType, 'd2Mean') && areaIdxAr <= numel(arType.d2Mean) ...
    && ~isempty(arType.d2Mean{areaIdxAr})));
  if useArDriver && needD2
    driverNames = get_type_session_names(arType);
    numSess = numel(arType.d2Mean{areaIdxAr});
    driver = 'ar';
  elseif hasAvType && needAv
    driverNames = get_type_session_names(avType);
    numSess = numel(avType.tau{areaIdxAv});
    driver = 'av';
  elseif hasArType
    driverNames = get_type_session_names(arType);
    if isfield(arType, 'd2Mean') && areaIdxAr <= numel(arType.d2Mean) ...
        && ~isempty(arType.d2Mean{areaIdxAr})
      numSess = numel(arType.d2Mean{areaIdxAr});
    else
      numSess = numel(driverNames);
    end
    driver = 'ar';
  elseif hasAvType
    driverNames = get_type_session_names(avType);
    if isfield(avType, 'tau') && areaIdxAv <= numel(avType.tau) ...
        && ~isempty(avType.tau{areaIdxAv})
      numSess = numel(avType.tau{areaIdxAv});
    else
      numSess = numel(driverNames);
    end
    driver = 'av';
  else
    continue;
  end

  for i = 1:numSess
    sessionName = driverNames{min(i, numel(driverNames))};
    arIdx = i;
    avIdx = i;
    if strcmp(driver, 'ar') && needAv
      avIdx = find_matching_session_index(avType, sessionName, i);
      if isempty(avIdx)
        continue;
      end
    elseif strcmp(driver, 'av') && needD2
      arIdx = find_matching_session_index(arType, sessionName, i);
      if isempty(arIdx)
        continue;
      end
    elseif strcmp(driver, 'av') && hasArType && ~needD2
      arIdx = find_matching_session_index(arType, sessionName, i);
    end

    d2Val = nan;
    tauVal = nan;
    alphaVal = nan;
    d2SemVal = nan;
    tauSemVal = 0;
    alphaSemVal = 0;
    if needD2 && ~isempty(arIdx) && hasArType
      d2Val = arType.d2Mean{areaIdxAr}(arIdx);
      d2SemVal = get_metric_series_value(arType.d2Sem{areaIdxAr}, arIdx);
    end
    if needTau && ~isempty(avIdx) && hasAvType
      tauVal = avType.tau{areaIdxAv}(avIdx);
      if isfield(avType, 'tauSem') && areaIdxAv <= numel(avType.tauSem) ...
          && ~isempty(avType.tauSem{areaIdxAv})
        tauSemVal = get_metric_series_value(avType.tauSem{areaIdxAv}, avIdx);
        if ~isfinite(tauSemVal)
          tauSemVal = 0;
        end
      end
    end
    if needAlpha && ~isempty(avIdx) && hasAvType
      alphaVal = avType.alpha{areaIdxAv}(avIdx);
      if isfield(avType, 'alphaSem') && areaIdxAv <= numel(avType.alphaSem) ...
          && ~isempty(avType.alphaSem{areaIdxAv})
        alphaSemVal = get_metric_series_value(avType.alphaSem{areaIdxAv}, avIdx);
        if ~isfinite(alphaSemVal)
          alphaSemVal = 0;
        end
      end
    end

    checkVals = [];
    if needD2, checkVals(end + 1) = d2Val; end %#ok<AGROW>
    if needTau, checkVals(end + 1) = tauVal; end %#ok<AGROW>
    if needAlpha, checkVals(end + 1) = alphaVal; end %#ok<AGROW>
    if ~keepBlankSessions && (isempty(checkVals) || ~all(isfinite(checkVals)))
      continue;
    end

    if strcmp(driver, 'ar')
      labelIdx = arIdx;
      labelType = arType;
    else
      labelIdx = avIdx;
      labelType = avType;
    end
    sessionTypeCol{end + 1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = sessionName; %#ok<AGROW>
    sessionLabelCol{end + 1, 1} = get_session_display_label(labelType, labelIdx, sessionType); %#ok<AGROW>
    d2MeanCol(end + 1, 1) = d2Val; %#ok<AGROW>
    d2SemCol(end + 1, 1) = d2SemVal; %#ok<AGROW>
    tauMeanCol(end + 1, 1) = tauVal; %#ok<AGROW>
    tauSemCol(end + 1, 1) = tauSemVal; %#ok<AGROW>
    alphaMeanCol(end + 1, 1) = alphaVal; %#ok<AGROW>
    alphaSemCol(end + 1, 1) = alphaSemVal; %#ok<AGROW>
  end
end

sessionTable = table(sessionTypeCol, sessionNameCol, sessionLabelCol, ...
  d2MeanCol, d2SemCol, tauMeanCol, tauSemCol, alphaMeanCol, alphaSemCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'sessionLabel', ...
  'd2Mean', 'd2Sem', 'tauMean', 'tauSem', 'alphaMean', 'alphaSem'});
end

function names = get_type_session_names(typeData)
if isfield(typeData, 'sessionNames') && ~isempty(typeData.sessionNames)
  names = cellfun(@char, typeData.sessionNames, 'UniformOutput', false);
else
  names = cellfun(@char, typeData.sessionLabels, 'UniformOutput', false);
end
end

function idx = find_matching_session_index(typeData, sessionName, fallbackIdx)
% FIND_MATCHING_SESSION_INDEX - Match by session name; optional within-type fallback
%
% Variables:
%   typeData     - plotData.byType.(sessionType) struct
%   sessionName  - Session name to find
%   fallbackIdx  - Optional 1-based index within this type's series (not a
%                  global table row). Used only if name match fails.
%
% Goal:
%   Always try name matching first. Do not gate name search on fallbackIdx,
%   otherwise later session-types (interval/reach) fail when the global row
%   index exceeds that type's session count.

idx = [];
if nargin < 3
  fallbackIdx = [];
end

names = {};
if isfield(typeData, 'sessionNames') && ~isempty(typeData.sessionNames)
  names = get_type_session_names(typeData);
elseif isfield(typeData, 'sessionLabels') && ~isempty(typeData.sessionLabels)
  names = cellfun(@char, typeData.sessionLabels, 'UniformOutput', false);
end

if ~isempty(names) && ~isempty(sessionName)
  idx = find(strcmp(names, char(sessionName)), 1);
end

if isempty(idx) && ~isempty(fallbackIdx) && isfinite(fallbackIdx) && fallbackIdx >= 1
  nSeries = numel(names);
  if nSeries == 0 && isfield(typeData, 'sessionLabels')
    nSeries = numel(typeData.sessionLabels);
  end
  if fallbackIdx <= nSeries
    idx = fallbackIdx;
  end
end
end

function label = get_session_display_label(typeData, sessionIdx, sessionType)
if isfield(typeData, 'sessionNames') && numel(typeData.sessionNames) >= sessionIdx
  label = char(typeData.sessionNames{sessionIdx});
elseif isfield(typeData, 'sessionLabels') && numel(typeData.sessionLabels) >= sessionIdx
  label = char(typeData.sessionLabels{sessionIdx});
else
  label = sessionType;
end
label = truncate_session_xtick_label(label, 7);
end

function label = truncate_session_xtick_label(label, maxChars)
% TRUNCATE_SESSION_XTICK_LABEL - Cap session-name tick text length
%
% Variables:
%   label    - Session name / label string
%   maxChars - Maximum characters to display (default 7)
%
% Goal:
%   Keep x-tick labels short so dense session plots remain readable.

if nargin < 2 || isempty(maxChars)
  maxChars = 7;
end
label = char(label);
if numel(label) > maxChars
  label = label(1:maxChars);
end
end

function val = get_metric_series_value(metricSeries, idx)
val = nan;
if isempty(metricSeries) || idx > numel(metricSeries)
  return;
end
val = metricSeries(idx);
end

function maps = compute_anchored_metric_maps(anchorMetric, d2Vals, tauVals, alphaVals, ...
    metricsToPlot, useAnchorAffineMap)
% COMPUTE_ANCHORED_METRIC_MAPS - Affine maps of non-anchor metrics into anchor space
%
% Variables:
%   useAnchorAffineMap - If false, all maps are identity (native scales)

if nargin < 5 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
end
if nargin < 6 || isempty(useAnchorAffineMap)
  useAnchorAffineMap = true;
end
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);

metricVals = struct('d2', d2Vals(:), 'tau', tauVals(:), 'alpha', alphaVals(:));
anchorVals = metricVals.(anchorMetric);

maps = struct();
metricNames = {'d2', 'tau', 'alpha'};
for i = 1:numel(metricNames)
  name = metricNames{i};
  if ~useAnchorAffineMap || ~ismember(name, metricsToPlot) || strcmp(name, anchorMetric)
    maps.(name) = struct('gain', 1, 'offset', 0);
  else
    maps.(name) = fit_metric_affine_map_to_anchor(metricVals.(name), anchorVals);
  end
end
end

function maps = compute_independent_range_maps(anchorMetric, nativeVals, metricsToPlot, displayYLim)
% COMPUTE_INDEPENDENT_RANGE_MAPS - Map each metric's native range onto displayYLim
%
% Variables:
%   anchorMetric  - Primary metric (identity map; displayYLim is its native ylim)
%   nativeVals    - Struct with .d2 / .tau / .alpha vectors
%   metricsToPlot - Metrics included in the figure
%   displayYLim   - [ymin ymax] display limits (usually primary native ylim)
%
% Goal:
%   Keep markers on one axes (aligned x) while right-side axes report native
%   units. Unlike session-wise LS anchoring, each metric uses its own min/max.

metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
maps = struct();
metricNames = {'d2', 'tau', 'alpha'};
for i = 1:numel(metricNames)
  name = metricNames{i};
  if ~ismember(name, metricsToPlot) || strcmp(name, anchorMetric)
    maps.(name) = struct('gain', 1, 'offset', 0);
  else
    maps.(name) = fit_metric_range_map_to_display(nativeVals.(name), displayYLim);
  end
end
end

function map = fit_metric_range_map_to_display(metricVals, displayYLim)
% FIT_METRIC_RANGE_MAP_TO_DISPLAY - Linear map native [min max] -> displayYLim
map = struct('gain', 1, 'offset', 0);
vals = metricVals(isfinite(metricVals));
if isempty(vals) || numel(displayYLim) ~= 2 || ~all(isfinite(displayYLim))
  return;
end
mMin = min(vals);
mMax = max(vals);
if mMax == mMin
  map.gain = 1;
  map.offset = mean(displayYLim) - mMin;
  return;
end
map.gain = (displayYLim(2) - displayYLim(1)) / (mMax - mMin);
map.offset = displayYLim(1) - map.gain * mMin;
end

function map = fit_metric_affine_map_to_anchor(metricVals, anchorVals)
% FIT_METRIC_AFFINE_MAP_TO_ANCHOR - Least-squares: anchor ≈ gain * metric + offset

map = struct('gain', 1, 'offset', 0);
metricVals = metricVals(:);
anchorVals = anchorVals(:);
valid = isfinite(metricVals) & isfinite(anchorVals);
metricVals = metricVals(valid);
anchorVals = anchorVals(valid);
if numel(metricVals) < 2
  if numel(metricVals) == 1
    map.offset = anchorVals(1) - metricVals(1);
  end
  return;
end

design = [metricVals, ones(size(metricVals))];
coeffs = design \ anchorVals;
if ~all(isfinite(coeffs)) || abs(coeffs(1)) < eps
  map.gain = 1;
  map.offset = mean(anchorVals) - mean(metricVals);
  return;
end
map.gain = coeffs(1);
map.offset = coeffs(2);
end

function y = apply_metric_affine_map(metricVals, map)
y = map.gain * metricVals + map.offset;
end

function h = plot_metric_errorbar_group(ax, xPos, yVals, ySem, markerSpec, color, faceColor, plotConfig)
semPlot = ySem;
semPlot(~isfinite(semPlot)) = 0;
h = errorbar(ax, xPos, yVals, semPlot, markerSpec, ...
  'Color', color, 'MarkerFaceColor', faceColor, ...
  'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
  'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
end

function add_affine_metric_yaxis(axRef, map, side, labelText, plotConfig, axisOffset)
% ADD_AFFINE_METRIC_YAXIS - Overlay axis with native metric ticks for affine map

if nargin < 6 || isempty(axisOffset)
  axisOffset = 1.0;
end
if abs(map.gain) < eps
  return;
end

axPos = axRef.Position;
if strcmp(side, 'right') && axisOffset > 1
  shiftFrac = min(0.08, max(0, axisOffset - 1));
  axPos = [axPos(1), axPos(2), axPos(3) * (1 + shiftFrac), axPos(4)];
end

axNative = axes('Parent', axRef.Parent, 'Position', axPos, ...
  'Color', 'none', 'XColor', 'none', 'XLim', axRef.XLim, 'YLim', axRef.YLim, ...
  'YAxisLocation', side, 'Box', 'off', 'HitTest', 'off');

displayTicks = axRef.YTick;
nativeTicks = (displayTicks - map.offset) ./ map.gain;
axNative.YTick = displayTicks;
tickLabels = arrayfun(@(v) sprintf('%.3g', v), nativeTicks, 'UniformOutput', false);
axNative.YTickLabel = tickLabels;
ylabel(axNative, labelText, 'FontSize', plotConfig.axisLabelFontSize, ...
  'Interpreter', ternary_metric_label_interpreter(labelText));
set(axNative, 'FontSize', plotConfig.tickLabelFontSize, 'TickDir', 'out');
uistack(axRef, 'top');
end

function axOverlay = create_native_overlay_yaxis(axRef, rightOffset, plotConfig)
% CREATE_NATIVE_OVERLAY_YAXIS - Transparent axes sharing x with independent y
%
% Variables:
%   axRef       - Primary axes (left metric)
%   rightOffset - Extra right margin fraction for stacked right axes
%   plotConfig  - Manuscript plot config
%
% Goal:
%   Plot a second metric in native units with its own right y-axis.

axPos = axRef.Position;
if rightOffset > 0
  shrink = min(0.12, rightOffset);
  axPos = [axPos(1), axPos(2), max(0.2, axPos(3) - shrink), axPos(4)];
  axRef.Position = axPos;
end
axOverlay = axes('Parent', axRef.Parent, 'Position', axRef.Position, ...
  'Color', 'none', 'XColor', 'none', 'YAxisLocation', 'right', ...
  'Box', 'off', 'HitTest', 'off', 'FontSize', plotConfig.tickLabelFontSize, ...
  'LineWidth', plotConfig.axesLineWidth, 'TickDir', 'out');
hold(axOverlay, 'on');
linkprop([axRef, axOverlay], 'Position');
end

function yLimPlot = compute_native_ylim_for_metric(metricVals)
% COMPUTE_NATIVE_YLIM_FOR_METRIC - Padded y-limits from one metric series
yLimPlot = [];
vals = metricVals(isfinite(metricVals));
if isempty(vals)
  return;
end
yPad = max(0.05 * max(range(vals), eps), 0.02 * max(abs(vals)));
if yPad == 0
  yPad = max(0.05, 0.05 * abs(vals(1)));
end
yLimPlot = [min(vals) - yPad, max(vals) + yPad];
end

function draw_across_session_metric_lines(ax, metricLineX, metricLineY, metricsToPlot, ...
    taskColor, plotConfig)
% DRAW_ACROSS_SESSION_METRIC_LINES - Join each metric across sessions within a task
%
% Variables:
%   ax             - Axes handle
%   metricLineX/Y  - Structs with .d2 / .tau / .alpha session x/y vectors
%   metricsToPlot  - Which metrics are present
%   taskColor      - RGB for this session type (used for d2 and tau)
%   plotConfig     - Line-width baseline
%
% Goal:
%   d2: thick solid task-colored line
%   tau: thinner dashed task-colored line
%   alpha: solid gray line

if nargin < 6 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

baseWidth = plotConfig.lineWidth;
lineStyleByMetric = struct( ...
  'd2', struct('LineStyle', '-', 'LineWidth', baseWidth + 1.25, 'Color', taskColor), ...
  'tau', struct('LineStyle', '--', 'LineWidth', max(0.75, baseWidth - 0.5), 'Color', taskColor), ...
  'alpha', struct('LineStyle', '-', 'LineWidth', baseWidth, 'Color', [0.55, 0.55, 0.55]));

hold(ax, 'on');
for m = 1:numel(metricsToPlot)
  metricName = metricsToPlot{m};
  if ~isfield(metricLineX, metricName) || ~isfield(lineStyleByMetric, metricName)
    continue;
  end
  xPts = metricLineX.(metricName)(:);
  yPts = metricLineY.(metricName)(:);
  valid = isfinite(xPts) & isfinite(yPts);
  if sum(valid) < 2
    continue;
  end
  style = lineStyleByMetric.(metricName);
  plot(ax, xPts(valid), yPts(valid), style.LineStyle, ...
    'Color', style.Color, 'LineWidth', style.LineWidth, 'HandleVisibility', 'off');
end
end

function plotBase = make_separated_metrics_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, engagementTag, ...
    useSubsampling, nNeuronsSubsample, avWindow, binSizeD2)
% MAKE_SEPARATED_METRICS_PLOT_BASENAME - File stem for separated metric panels
if nargin < 7 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 8 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 9 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 10
  avWindow = [];
end
if nargin < 11 || isempty(binSizeD2)
  binSizeD2 = [];
end
collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
winTag = format_window_sec_file_tag(d2Window);
avTag = format_window_sec_file_tag(avWindow);
binTag = format_bin_size_file_tag(binSizeD2);
if ~isempty(brainArea)
  areaTag = brainArea;
else
  areaTag = areaName;
end
plotBase = sprintf('criticality_separated_metrics_%s_win%s_bin%s_av%s_%s', ...
  areaTag, winTag, binTag, avTag, collectTag);
if ~isempty(engagementTag)
  plotBase = sprintf('%s_%s', plotBase, engagementTag);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
plotBase = [plotBase, format_subsamp_file_tag(useSubsampling, nNeuronsSubsample)];
end

function interp = ternary_metric_label_interpreter(labelText)
if contains(labelText, '_{') || contains(labelText, '\')
  interp = 'tex';
else
  interp = 'none';
end
end

function plotBase = make_multimetric_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, anchorMetric, engagementTag, metricsToPlot, ...
    useAnchorAffineMap, useSubsampling, nNeuronsSubsample, avWindow)
if nargin < 7 || isempty(anchorMetric)
  anchorMetric = 'd2';
end
if nargin < 8 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 9 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
end
if nargin < 10 || isempty(useAnchorAffineMap)
  useAnchorAffineMap = true;
end
if nargin < 11 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 12 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 13
  avWindow = [];
end
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
winTag = format_window_sec_file_tag(d2Window);
avTag = format_window_sec_file_tag(avWindow);
if ~isempty(brainArea)
  areaTag = brainArea;
else
  areaTag = areaName;
end
plotBase = sprintf('criticality_multiple_metrics_%s_win%s_av%s_%s', ...
  areaTag, winTag, avTag, collectTag);
if useAnchorAffineMap
  plotBase = sprintf('%s_anchor%s', plotBase, anchorMetric);
end
if ~isempty(engagementTag)
  plotBase = sprintf('%s_%s', plotBase, engagementTag);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
plotBase = [plotBase, format_subsamp_file_tag(useSubsampling, nNeuronsSubsample)];
end

function plotBase = make_pair_scatter_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, engagementTag, useSubsampling, nNeuronsSubsample, ...
    avWindow)
% MAKE_PAIR_SCATTER_PLOT_BASENAME - File stem for 1x3 metric pair scatters
if nargin < 7 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 8 || isempty(useSubsampling)
  useSubsampling = false;
end
if nargin < 9 || isempty(nNeuronsSubsample)
  nNeuronsSubsample = 0;
end
if nargin < 10
  avWindow = [];
end
collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
winTag = format_window_sec_file_tag(d2Window);
avTag = format_window_sec_file_tag(avWindow);
if ~isempty(brainArea)
  areaTag = brainArea;
else
  areaTag = areaName;
end
plotBase = sprintf('criticality_metric_pair_scatters_%s_win%s_av%s_%s', ...
  areaTag, winTag, avTag, collectTag);
if ~isempty(engagementTag)
  plotBase = sprintf('%s_%s', plotBase, engagementTag);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
plotBase = [plotBase, format_subsamp_file_tag(useSubsampling, nNeuronsSubsample)];
end

function tag = format_window_sec_file_tag(windowSec)
% FORMAT_WINDOW_SEC_FILE_TAG - 'full' if empty, else e.g. '45s'
if isempty(windowSec)
  tag = 'full';
else
  tag = sprintf('%.0fs', windowSec);
end
end

function tag = format_bin_size_file_tag(binSizeSec)
% FORMAT_BIN_SIZE_FILE_TAG - 'na' if empty, else e.g. '40ms'
if isempty(binSizeSec) || ~isfinite(binSizeSec)
  tag = 'na';
else
  tag = sprintf('%.0fms', binSizeSec * 1000);
end
end

function phrase = format_d2_av_window_title_phrase(d2Window, avWindow)
% FORMAT_D2_AV_WINDOW_TITLE_PHRASE - Title fragment for d2 and AV window lengths
phrase = sprintf('%s d2 windows, %s AV windows', ...
  format_window_sec_file_tag(d2Window), format_window_sec_file_tag(avWindow));
end

function phrase = format_bin_size_title_phrase(binSizeSec)
% FORMAT_BIN_SIZE_TITLE_PHRASE - Title fragment for d2 spike bin width
if isempty(binSizeSec) || ~isfinite(binSizeSec)
  phrase = '';
else
  phrase = sprintf(', %.0f ms d2 bins', binSizeSec * 1000);
end
end

function tag = format_subsamp_file_tag(useSubsampling, nNeuronsSubsample)
% FORMAT_SUBSAMP_FILE_TAG - Filename suffix when neuron subsampling is on
tag = '';
if useSubsampling
  tag = sprintf('_subsamp%d', nNeuronsSubsample);
end
end

function titleStr = append_subsamp_title_tag(titleStr, useSubsampling, nNeuronsSubsample)
% APPEND_SUBSAMP_TITLE_TAG - Append subsampN to figure title when subsampling
if useSubsampling
  titleStr = sprintf('%s; subsamp%d', titleStr, nNeuronsSubsample);
end
end

function tag = format_multimetric_collect_tag(collectStart, collectEnd)
if isempty(collectEnd)
  tag = sprintf('%.0f-full', collectStart);
else
  tag = sprintf('%.0f-%.0f', collectStart, collectEnd);
end
end

function titleTag = format_engagement_title_tag(engagementTag)
if isempty(engagementTag)
  titleTag = '';
elseif strcmpi(engagementTag, 'engaged')
  titleTag = ' | engaged';
elseif strcmpi(engagementTag, 'nonEngaged')
  titleTag = ' | non-engaged';
else
  titleTag = sprintf(' | %s', engagementTag);
end
end

function metricsToPlot = normalize_metrics_to_plot(metricsToPlot)
% NORMALIZE_METRICS_TO_PLOT - Validate and order metric marker names

if ischar(metricsToPlot) || isstring(metricsToPlot)
  metricsToPlot = cellstr(metricsToPlot);
end
metricsToPlot = lower(metricsToPlot(:)');
valid = {'d2', 'tau', 'alpha'};
unknown = setdiff(metricsToPlot, valid);
if ~isempty(unknown)
  error('metricsToPlot has unknown entries: %s', strjoin(unknown, ', '));
end
metricsToPlot = intersect(valid, metricsToPlot, 'stable');
if isempty(metricsToPlot)
  error('metricsToPlot must contain at least one of: d2, tau, alpha');
end
end

function metricsToPlot = filter_metrics_to_plot_by_pipelines(metricsToPlot, useAr, useAv)
% FILTER_METRICS_TO_PLOT_BY_PIPELINES - Drop markers for inactive pipelines
%
% May return {} when only PRG is selected (combined d2/tau/alpha plot skipped).

if ischar(metricsToPlot) || isstring(metricsToPlot)
  metricsToPlot = cellstr(metricsToPlot);
end
metricsToPlot = lower(metricsToPlot(:)');
valid = {'d2', 'tau', 'alpha'};
metricsToPlot = intersect(valid, metricsToPlot, 'stable');
if ~useAr
  metricsToPlot = setdiff(metricsToPlot, {'d2'}, 'stable');
end
if ~useAv
  metricsToPlot = setdiff(metricsToPlot, {'tau', 'alpha'}, 'stable');
end
end

function activeAnalyses = fill_active_analyses(activeAnalyses)
% FILL_ACTIVE_ANALYSES - Ensure .ar/.av/.prg logical fields exist
if ~isstruct(activeAnalyses)
  activeAnalyses = struct('ar', true, 'av', true, 'prg', true);
end
if ~isfield(activeAnalyses, 'ar') || isempty(activeAnalyses.ar)
  activeAnalyses.ar = true;
end
if ~isfield(activeAnalyses, 'av') || isempty(activeAnalyses.av)
  activeAnalyses.av = true;
end
if ~isfield(activeAnalyses, 'prg') || isempty(activeAnalyses.prg)
  activeAnalyses.prg = true;
end
activeAnalyses.ar = logical(activeAnalyses.ar);
activeAnalyses.av = logical(activeAnalyses.av);
activeAnalyses.prg = logical(activeAnalyses.prg);
end

function areas = resolve_pipeline_ref_areas(arOut, avOut, prgOut, areasToPlot, brainArea)
% RESOLVE_PIPELINE_REF_AREAS - Prefer loaded pipeline areas for stubs
areas = {};
outs = {arOut, avOut, prgOut};
for i = 1:numel(outs)
  out = outs{i};
  if isempty(out) || ~isfield(out, 'plotData') || ~isfield(out.plotData, 'areas')
    continue;
  end
  areas = [areas, cellstr(string(out.plotData.areas))]; %#ok<AGROW>
end
areas = unique(areas, 'stable');
if isempty(areas)
  if ~isempty(areasToPlot)
    areas = cellstr(string(areasToPlot));
  elseif ~isempty(brainArea)
    areas = {char(brainArea)};
  end
end
end

function areas = resolve_multimetric_plot_areas(arOut, avOut, prgOut, useAr, useAv, usePrg, ...
    areasToPlot, brainArea)
% RESOLVE_MULTIMETRIC_PLOT_AREAS - Areas from active pipelines (intersect when >1)
areaLists = {};
if useAr && ~isempty(arOut) && isfield(arOut, 'plotData') && isfield(arOut.plotData, 'areas') ...
    && ~isempty(arOut.plotData.areas)
  areaLists{end + 1} = cellstr(string(arOut.plotData.areas)); %#ok<AGROW>
end
if useAv && ~isempty(avOut) && isfield(avOut, 'plotData') && isfield(avOut.plotData, 'areas') ...
    && ~isempty(avOut.plotData.areas)
  areaLists{end + 1} = cellstr(string(avOut.plotData.areas)); %#ok<AGROW>
end
if usePrg && ~isempty(prgOut) && isfield(prgOut, 'plotData') && isfield(prgOut.plotData, 'areas') ...
    && ~isempty(prgOut.plotData.areas)
  areaLists{end + 1} = cellstr(string(prgOut.plotData.areas)); %#ok<AGROW>
end
if isempty(areaLists)
  areas = resolve_pipeline_ref_areas(arOut, avOut, prgOut, areasToPlot, brainArea);
else
  areas = areaLists{1};
  for i = 2:numel(areaLists)
    areas = intersect(areas, areaLists{i}, 'stable');
  end
end
if ~isempty(areasToPlot)
  areas = intersect(areas, cellstr(string(areasToPlot)), 'stable');
elseif ~isempty(brainArea)
  areas = intersect(areas, {char(brainArea)}, 'stable');
end
end

function out = make_empty_pipeline_out(sessionTypes, areas, kind, useLog10D2)
% MAKE_EMPTY_PIPELINE_OUT - Stub batch output so plotting layouts stay intact
%
% Variables:
%   sessionTypes - Task types for byType keys
%   areas        - Brain-area list (shared with active pipelines)
%   kind         - 'ar', 'av', or 'prg'
%   useLog10D2   - Stored on AR plotData when true

if nargin < 4 || isempty(useLog10D2)
  useLog10D2 = false;
end
if isempty(areas)
  areas = {};
end
plotData = struct('areas', {areas}, 'sessionTypes', {sessionTypes}, 'byType', struct());
if strcmpi(kind, 'ar')
  plotData.useLog10D2 = logical(useLog10D2);
end
nAreas = numel(areas);
for t = 1:numel(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  switch lower(kind)
    case 'ar'
      plotData.byType.(typeKey) = init_standard_ar_type(nAreas);
    case 'av'
      plotData.byType.(typeKey) = init_standard_av_type(nAreas);
    case 'prg'
      plotData.byType.(typeKey) = init_standard_prg_type(nAreas);
    otherwise
      error('make_empty_pipeline_out: unknown kind "%s"', kind);
  end
end
out = struct('plotData', plotData, 'batchResults', []);
end

function baseTable = build_prg_only_session_base_table(prgPlotData, sessionTypes, areaIdxPrg)
% BUILD_PRG_ONLY_SESSION_BASE_TABLE - Session rows from PRG when AR/AV absent

sessionTypeCol = {};
sessionNameCol = {};
sessionLabelCol = {};
d2MeanCol = [];
d2SemCol = [];
tauMeanCol = [];
tauSemCol = [];
alphaMeanCol = [];
alphaSemCol = [];

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(prgPlotData.byType, typeKey)
    continue;
  end
  prgType = prgPlotData.byType.(typeKey);
  if ~isfield(prgType, 'kappaMean') || areaIdxPrg > numel(prgType.kappaMean) ...
      || isempty(prgType.kappaMean{areaIdxPrg})
    continue;
  end
  names = get_type_session_names(prgType);
  numSess = numel(prgType.kappaMean{areaIdxPrg});
  for i = 1:numSess
    sessionName = names{min(i, numel(names))};
    sessionTypeCol{end + 1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = sessionName; %#ok<AGROW>
    sessionLabelCol{end + 1, 1} = get_session_display_label(prgType, i, sessionType); %#ok<AGROW>
    d2MeanCol(end + 1, 1) = nan; %#ok<AGROW>
    d2SemCol(end + 1, 1) = nan; %#ok<AGROW>
    tauMeanCol(end + 1, 1) = nan; %#ok<AGROW>
    tauSemCol(end + 1, 1) = 0; %#ok<AGROW>
    alphaMeanCol(end + 1, 1) = nan; %#ok<AGROW>
    alphaSemCol(end + 1, 1) = 0; %#ok<AGROW>
  end
end

baseTable = table(sessionTypeCol, sessionNameCol, sessionLabelCol, ...
  d2MeanCol, d2SemCol, tauMeanCol, tauSemCol, alphaMeanCol, alphaSemCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'sessionLabel', ...
  'd2Mean', 'd2Sem', 'tauMean', 'tauSem', 'alphaMean', 'alphaSem'});
end

function yLimPlot = compute_display_ylim_for_metrics(yVals, metricsToPlot, anchorMetric)
% COMPUTE_DISPLAY_YLIM_FOR_METRICS - Padded y-limits from plotted display values

yAnchor = yVals.(anchorMetric);
anchorFinite = yAnchor(isfinite(yAnchor));
if isempty(anchorFinite)
  yLimPlot = [];
  return;
end
yPad = max(0.05 * max(range(anchorFinite), eps), 0.02 * max(abs(anchorFinite)));
yLimPlot = [min(anchorFinite) - yPad, max(anchorFinite) + yPad];

yOthers = [];
for m = 1:numel(metricsToPlot)
  metricName = metricsToPlot{m};
  if strcmp(metricName, anchorMetric)
    continue;
  end
  vals = yVals.(metricName);
  yOthers = [yOthers; vals(:)]; %#ok<AGROW>
end
yOthers = yOthers(isfinite(yOthers));
if ~isempty(yOthers)
  yLimPlot(1) = min(yLimPlot(1), min(yOthers));
  yLimPlot(2) = max(yLimPlot(2), max(yOthers));
end
end

function sharedByArea = compute_shared_engagement_plot_scales(classViews, plotAreas, ...
    sessionTypes, metricsToPlot, anchorMetric, useAnchorAffineMap)
% COMPUTE_SHARED_ENGAGEMENT_PLOT_SCALES - Common maps + ylim for engaged/non-engaged
%
% Goal:
%   Fit affine maps on pooled engaged+non-engaged sessions and use one y-limit
%   per area so the primary (usually d2) axis matches across the pair of plots.

if nargin < 6 || isempty(useAnchorAffineMap)
  useAnchorAffineMap = true;
end
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
sharedByArea = struct();
engClasses = fieldnames(classViews);

for a = 1:numel(plotAreas)
  areaName = plotAreas{a};
  areaKey = matlab.lang.makeValidName(areaName);
  d2All = [];
  tauAll = [];
  alphaAll = [];
  tables = {};

  for iClass = 1:numel(engClasses)
    engClass = engClasses{iClass};
    arPlotData = classViews.(engClass).ar;
    avPlotData = classViews.(engClass).av;
    areaIdxAr = find(strcmp(arPlotData.areas, areaName), 1);
    areaIdxAv = find(strcmp(avPlotData.areas, areaName), 1);
    if isempty(areaIdxAr) || isempty(areaIdxAv)
      continue;
    end
    sessionTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
      areaIdxAr, areaIdxAv, metricsToPlot);
    if isempty(sessionTable)
      continue;
    end
    tables{end + 1} = sessionTable; %#ok<AGROW>
    d2All = [d2All; sessionTable.d2Mean(:)]; %#ok<AGROW>
    tauAll = [tauAll; sessionTable.tauMean(:)]; %#ok<AGROW>
    alphaAll = [alphaAll; sessionTable.alphaMean(:)]; %#ok<AGROW>
  end

  if isempty(tables)
    continue;
  end

  maps = compute_anchored_metric_maps(anchorMetric, d2All, tauAll, alphaAll, ...
    metricsToPlot, useAnchorAffineMap);

  % Prefer shared native d2 limits when d2 is plotted (comparison across classes)
  if ismember('d2', metricsToPlot)
    d2Finite = d2All(isfinite(d2All));
    if isempty(d2Finite)
      continue;
    end
    yPad = max(0.05 * max(range(d2Finite), eps), 0.02 * max(abs(d2Finite)));
    yLimD2 = [min(d2Finite) - yPad, max(d2Finite) + yPad];

    if ~useAnchorAffineMap
      nativePool = struct('d2', d2All, 'tau', tauAll, 'alpha', alphaAll);
      yLimPlot = compute_native_ylim_for_metric(nativePool.(anchorMetric));
      maps = compute_independent_range_maps(anchorMetric, nativePool, metricsToPlot, yLimPlot);
    elseif useAnchorAffineMap && strcmp(anchorMetric, 'd2')
      yLimPlot = yLimD2;
      % Expand slightly if other mapped markers fall outside
      yValsPool = struct( ...
        'd2', apply_metric_affine_map(d2All, maps.d2), ...
        'tau', apply_metric_affine_map(tauAll, maps.tau), ...
        'alpha', apply_metric_affine_map(alphaAll, maps.alpha));
      yLimOthers = compute_display_ylim_for_metrics(yValsPool, metricsToPlot, anchorMetric);
      if ~isempty(yLimOthers)
        yLimPlot(1) = min(yLimPlot(1), yLimOthers(1));
        yLimPlot(2) = max(yLimPlot(2), yLimOthers(2));
      end
    else
      yValsPool = struct( ...
        'd2', apply_metric_affine_map(d2All, maps.d2), ...
        'tau', apply_metric_affine_map(tauAll, maps.tau), ...
        'alpha', apply_metric_affine_map(alphaAll, maps.alpha));
      yLimPlot = compute_display_ylim_for_metrics(yValsPool, metricsToPlot, anchorMetric);
    end
  else
    if ~useAnchorAffineMap
      nativePool = struct('d2', d2All, 'tau', tauAll, 'alpha', alphaAll);
      yLimPlot = compute_native_ylim_for_metric(nativePool.(anchorMetric));
      maps = compute_independent_range_maps(anchorMetric, nativePool, metricsToPlot, yLimPlot);
    else
      yValsPool = struct( ...
        'd2', apply_metric_affine_map(d2All, maps.d2), ...
        'tau', apply_metric_affine_map(tauAll, maps.tau), ...
        'alpha', apply_metric_affine_map(alphaAll, maps.alpha));
      yLimPlot = compute_display_ylim_for_metrics(yValsPool, metricsToPlot, anchorMetric);
    end
  end

  if isempty(yLimPlot) || ~all(isfinite(yLimPlot))
    continue;
  end
  sharedByArea.(areaKey).maps = maps;
  sharedByArea.(areaKey).yLim = yLimPlot;
end
end

function position_figure_full_monitor(fig)
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
  targetPos = monitorPositions(end, :);
else
  targetPos = monitorPositions(1, :);
end
set(fig, 'Units', 'pixels', 'Position', targetPos);
end

%% -------------------------------------------------------------------------
%% Engagement split batch and plot-data views
%% -------------------------------------------------------------------------

function engOut = run_multimetric_engagement_batch(opts)
% RUN_MULTIMETRIC_ENGAGEMENT_BATCH - Interval/reach metrics by engagement class

analysesTag = 'd2 + avalanches + kurtosis';
if isfield(opts, 'analyses') && ~isempty(opts.analyses)
  analysesTag = strjoin(opts.analyses, ' + ');
end
fprintf('\n=== Engagement batch (%s) ===\n', analysesTag);
fprintf('Session types: %s\n', strjoin(opts.sessionTypes, ', '));

sessionTable = build_multimetric_engagement_session_table(opts.sessionTypes);
numSessions = size(sessionTable, 1);
fprintf('Engagement sessions: %d\n', numSessions);
if numSessions == 0
  error('No interval/reach sessions found for engagement batch.');
end

batchResults = repmat(struct( ...
  'sessionType', '', 'sessionName', '', 'subjectName', '', 'label', '', ...
  'success', false, 'skipReason', '', 'd2Split', [], 'avalanches', [], 'kurtosis', []), ...
  numSessions, 1);

for s = 1:numSessions
  sessionType = sessionTable.sessionType{s};
  sessionName = sessionTable.sessionName{s};
  subjectName = sessionTable.subjectName{s};

  fprintf('\n%s\n', repmat('=', 1, 80));
  fprintf('Engagement session %d/%d [%s]: %s\n', s, numSessions, sessionType, sessionName);

  batchResults(s).sessionType = sessionType;
  batchResults(s).sessionName = sessionName;
  batchResults(s).subjectName = subjectName;
  batchResults(s).label = sessionTable.label{s};
  batchResults(s).success = false;

  analyses = {'d2', 'avalanches', 'kurtosis'};
  if isfield(opts, 'analyses') && ~isempty(opts.analyses)
    analyses = opts.analyses;
  end
  [cachedPipelines, missingAnalyses] = try_load_engagement_pipeline_caches( ...
    sessionType, sessionName, subjectName, opts, analyses);

  % Prefer full-session AR/PRG caches: engagement d2/PRG are window splits of
  % those results. Avalanche class fits cannot be recovered from full-session AV.
  cachedArResults = [];
  cachedPrgResults = [];
  if any(strcmpi(analyses, 'd2'))
    cachedArResults = try_load_full_session_pipeline_cache( ...
      'ar', sessionType, sessionName, subjectName, opts);
    if ~isempty(cachedArResults)
      cachedPipelines.d2 = [];
      missingAnalyses = union_analysis_list(missingAnalyses, 'd2');
    end
  end
  if any(strcmpi(analyses, 'kurtosis'))
    cachedPrgResults = try_load_full_session_pipeline_cache( ...
      'prg', sessionType, sessionName, subjectName, opts);
    if ~isempty(cachedPrgResults)
      cachedPipelines.kurtosis = [];
      missingAnalyses = union_analysis_list(missingAnalyses, 'kurtosis');
    end
  end
  if any(strcmpi(analyses, 'avalanches')) && any(strcmpi(missingAnalyses, 'avalanches'))
    fprintf(['  Avalanches: full-session AV cache cannot be split by engagement ', ...
      '(class-specific segments and thresholds); running segment analysis.\n']);
  end

  if isempty(missingAnalyses)
    batchResults(s).d2Split = cachedPipelines.d2;
    batchResults(s).avalanches = cachedPipelines.avalanches;
    batchResults(s).kurtosis = cachedPipelines.kurtosis;
    batchResults(s).success = true;
    fprintf('  Engagement analysis completed (cached).\n');
    continue;
  end

  try
    engModOpts = build_multimetric_engagement_module_opts(opts, sessionType);
    engModOpts.analyses = missingAnalyses;
    engModOpts.useSessionCache = false;
    engModOpts.cachedArResults = cachedArResults;
    engModOpts.cachedPrgResults = cachedPrgResults;
    if strcmpi(sessionType, 'reach')
      sessionOut = reach_criticality_metrics_engagement(sessionName, engModOpts);
    elseif strcmpi(sessionType, 'semicircle')
      sessionOut = semicircle_criticality_metrics_engagement(subjectName, sessionName, engModOpts);
    else
      sessionOut = interval_criticality_metrics_engagement(subjectName, sessionName, engModOpts);
    end
    needD2 = any(strcmpi(analyses, 'd2'));
    needAv = any(strcmpi(analyses, 'avalanches'));
    if ~isempty(cachedPipelines.d2)
      sessionOut.d2 = cachedPipelines.d2;
    end
    if ~isempty(cachedPipelines.avalanches)
      sessionOut.avalanches = cachedPipelines.avalanches;
    end
    if ~isempty(cachedPipelines.kurtosis)
      sessionOut.kurtosis = cachedPipelines.kurtosis;
    end
    incomplete = (needD2 && (~isfield(sessionOut, 'd2') || isempty(sessionOut.d2))) ...
      || (needAv && (~isfield(sessionOut, 'avalanches') || isempty(sessionOut.avalanches)));
    if incomplete
      fprintf('  Incomplete engagement outputs; skipping.\n');
      batchResults(s).skipReason = 'incomplete engagement outputs';
      continue;
    end
    if isfield(sessionOut, 'd2')
      batchResults(s).d2Split = sessionOut.d2;
    end
    if isfield(sessionOut, 'avalanches')
      batchResults(s).avalanches = sessionOut.avalanches;
    end
    if isfield(sessionOut, 'kurtosis')
      batchResults(s).kurtosis = sessionOut.kurtosis;
    end
    if any(strcmpi(analyses, 'kurtosis')) && isempty(batchResults(s).kurtosis)
      fprintf('  Warning: kurtosis/PRG engagement split missing for this session.\n');
    end
    save_engagement_pipeline_caches(sessionOut, sessionType, sessionName, ...
      subjectName, opts, missingAnalyses);
    batchResults(s).success = true;
    fprintf('  Engagement analysis completed.\n');
  catch ME
    if is_skippable_engagement_session_error(ME)
      fprintf('  Skipping session: %s\n', ME.message);
      batchResults(s).skipReason = ME.message;
      continue;
    end
    fprintf('  Error: %s\n', ME.message);
    for st = 1:numel(ME.stack)
      fprintf('    %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
    end
    error('criticality_multiple_metrics_across_tasks:EngagementSessionFailed', ...
      'Engagement batch stopped at session %d/%d [%s] %s: %s', ...
      s, numSessions, sessionType, sessionName, ME.message);
  end
end

plotData = aggregate_multimetric_engagement_plot_data(batchResults, opts.sessionTypes, ...
  opts.useLog10D2);
batchMeta = struct( ...
  'sessionTypes', {opts.sessionTypes}, ...
  'collectStart', opts.collectStart, ...
  'collectEnd', opts.collectEnd, ...
  'd2Window', opts.d2Window, ...
  'brainArea', opts.brainArea, ...
  'useLog10D2', opts.useLog10D2, ...
  'powerLawFitMethod', opts.powerLawFitMethod, ...
  'avalancheDetectionMode', opts.avalancheDetectionMode);
if isfield(opts, 'thresholdMethod')
  batchMeta.thresholdMethod = opts.thresholdMethod;
end
if isfield(opts, 'binSizeD2')
  batchMeta.binSizeD2 = opts.binSizeD2;
end
if isfield(opts, 'binSizeAv')
  batchMeta.binSizeAv = opts.binSizeAv;
end
if isfield(opts, 'avWindow')
  batchMeta.avWindow = opts.avWindow;
end
if isfield(opts, 'engagementBufferBefore')
  batchMeta.engagementBufferBefore = opts.engagementBufferBefore;
end
if isfield(opts, 'engagementBufferAfter')
  batchMeta.engagementBufferAfter = opts.engagementBufferAfter;
end
if isfield(opts, 'engagementBuffer')
  batchMeta.engagementBuffer = opts.engagementBuffer;
end
if isfield(opts, 'minNonEngagedWindow')
  batchMeta.minNonEngagedWindow = opts.minNonEngagedWindow;
end
if isfield(opts, 'absorbSingleEvents')
  batchMeta.absorbSingleEvents = opts.absorbSingleEvents;
end
if isfield(opts, 'minTimeNonEngaged')
  batchMeta.minTimeNonEngaged = opts.minTimeNonEngaged;
end

engOut = struct('batchResults', batchResults, 'plotData', plotData, 'batchMeta', batchMeta);
end

function tf = is_skippable_engagement_session_error(ME)
% IS_SKIPPABLE_ENGAGEMENT_SESSION_ERROR - Expected per-session skip cases
%
% Includes TooFewNeurons / TooFewNeuronsForSubsample (matched on identifier,
% since the message text may not contain those tokens).

msg = ME.message;
id = ME.identifier;
tf = contains(msg, 'No valid areas to process') ...
  || contains(msg, 'insufficient neurons') ...
  || contains(msg, 'for subsampling') ...
  || contains(msg, 'not available') ...
  || contains(id, 'TooFewNeurons') ...
  || contains(id, 'TooFewNeuronsForSubsample');
end

function engModOpts = build_multimetric_engagement_module_opts(opts, sessionType)
% BUILD_MULTIMETRIC_ENGAGEMENT_MODULE_OPTS - Opts for reach/interval/semicircle engagement

if strcmpi(sessionType, 'reach')
  engModOpts = reach_criticality_metrics_engagement();
elseif strcmpi(sessionType, 'semicircle')
  engModOpts = semicircle_criticality_metrics_engagement();
else
  engModOpts = interval_criticality_metrics_engagement();
end

engModOpts.collectStart = opts.collectStart;
engModOpts.collectEnd = opts.collectEnd;
engModOpts.brainArea = opts.brainArea;
engModOpts.brainAreaCombinations = opts.brainAreaCombinations;
engModOpts.d2Window = opts.d2Window;
% Empty d2Window is resolved to loaded session duration inside engagement modules
if isfield(opts, 'binSizeD2') && ~isempty(opts.binSizeD2)
  engModOpts.binSizeD2 = opts.binSizeD2;
end
if isfield(opts, 'binSizeAv') && ~isempty(opts.binSizeAv)
  engModOpts.binSizeAv = opts.binSizeAv;
end
if isfield(opts, 'avWindow')
  engModOpts.avWindow = opts.avWindow;
end
% Engagement timing: reach uses reachBufferBefore/After;
% interval/semicircle use eventBufferBefore/After (legacy *Buffer still accepted)
[bufBefore, bufAfter] = resolve_engagement_buffer_pair( ...
  opts, 'engagementBufferBefore', 'engagementBufferAfter', 'engagementBuffer', 1);
if strcmpi(sessionType, 'reach')
  engModOpts.reachBufferBefore = bufBefore;
  engModOpts.reachBufferAfter = bufAfter;
else
  engModOpts.eventBufferBefore = bufBefore;
  engModOpts.eventBufferAfter = bufAfter;
end
if isfield(opts, 'minNonEngagedWindow') && ~isempty(opts.minNonEngagedWindow)
  engModOpts.minNonEngagedWindow = opts.minNonEngagedWindow;
end
if isfield(opts, 'absorbSingleEvents') && ~isempty(opts.absorbSingleEvents)
  if strcmpi(sessionType, 'reach')
    engModOpts.absorbSingleReaches = logical(opts.absorbSingleEvents);
  else
    engModOpts.absorbSingleEvents = logical(opts.absorbSingleEvents);
  end
end
engModOpts.useLog10D2 = opts.useLog10D2;
engModOpts.useSubsampling = opts.useSubsampling;
engModOpts.nSubsamples = opts.nSubsamples;
engModOpts.nNeuronsSubsample = opts.nNeuronsSubsample;
engModOpts.minNeuronsMultiple = opts.minNeuronsMultiple;
engModOpts.powerLawFitMethod = opts.powerLawFitMethod;
engModOpts.avalancheDetectionMode = opts.avalancheDetectionMode;
if isfield(opts, 'thresholdMethod') && ~isempty(opts.thresholdMethod)
  engModOpts.thresholdMethod = opts.thresholdMethod;
end
if isfield(opts, 'prgWindow')
  engModOpts.prgWindow = opts.prgWindow;
end
if isfield(opts, 'binSizePrg') && ~isempty(opts.binSizePrg)
  engModOpts.binSizePrg = opts.binSizePrg;
end
if isfield(opts, 'prgMethod') && ~isempty(opts.prgMethod)
  engModOpts.prgMethod = opts.prgMethod;
end
if isfield(opts, 'finalCutoffDivisor') && ~isempty(opts.finalCutoffDivisor)
  engModOpts.finalCutoffDivisor = opts.finalCutoffDivisor;
end
engModOpts.enableCircularPermutations = logical(opts.enablePermutations);
if opts.enablePermutations
  engModOpts.nShuffles = 5;
  engModOpts.nShufflesD2 = 10;
  engModOpts.nSurrogates = 10;
else
  engModOpts.nShuffles = 0;
  engModOpts.nShufflesD2 = 1;
  engModOpts.nSurrogates = 0;
end
engModOpts.analyses = {'d2', 'avalanches', 'kurtosis'};
if isfield(opts, 'analyses') && ~isempty(opts.analyses)
  engModOpts.analyses = opts.analyses;
end
engModOpts.makePlots = false;
engModOpts.saveFigure = false;
engModOpts.useSessionCache = false;
engModOpts.plotConfig = opts.plotConfig;
if strcmpi(sessionType, 'reach')
  engModOpts.runD2AccuracyCorrelation = false;
  engModOpts.runD2ReachRateCorrelation = false;
else
  engModOpts.runD2TrialRateCorrelation = false;
end
end

function analyses = union_analysis_list(analyses, analysisName)
% UNION_ANALYSIS_LIST - Append analysisName if not already in the cell list
if any(strcmpi(analyses, analysisName))
  return;
end
analyses{end + 1} = analysisName;
end

function sessionTable = build_multimetric_engagement_session_table(sessionTypes)
sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
labelCol = {};
for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  entries = get_multimetric_engagement_sessions(sessionType);
  for i = 1:numel(entries)
    sessionTypeCol{end + 1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = entries(i).sessionName; %#ok<AGROW>
    if isfield(entries, 'subjectName')
      subjectNameCol{end + 1, 1} = entries(i).subjectName; %#ok<AGROW>
    else
      subjectNameCol{end + 1, 1} = ''; %#ok<AGROW>
    end
    labelCol{end + 1, 1} = entries(i).sessionName; %#ok<AGROW>
  end
end
sessionTable = table(sessionTypeCol, sessionNameCol, subjectNameCol, labelCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'subjectName', 'label'});
end

function entries = get_multimetric_engagement_sessions(sessionType)
% GET_MULTIMETRIC_ENGAGEMENT_SESSIONS - Sessions for engagement batch only
%
% Engagement modules exist for interval/reach/semicircle. Spontaneous returns
% an empty list so it stays on the main (non-split) path.

if ~is_manuscript_engagement_session_type(sessionType)
  entries = struct('subjectName', {}, 'sessionName', {});
  return;
end
entries = manuscript_sessions_for_type(sessionType);
end

function plotData = aggregate_multimetric_engagement_plot_data(batchResults, sessionTypes, useLog10D2)
% AGGREGATE_MULTIMETRIC_ENGAGEMENT_PLOT_DATA - Per-session engaged/non-engaged metrics

plotData = struct();
plotData.areas = {};
sessionTypes = order_manuscript_session_types(sessionTypes);
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.useLog10D2 = useLog10D2;

metricFields = { ...
  'd2EngagedMean', 'd2EngagedSem', 'd2NonEngagedMean', 'd2NonEngagedSem', ...
  'tauEngaged', 'tauNonEngaged', 'alphaEngaged', 'alphaNonEngaged', ...
  'paramSDEngaged', 'paramSDNonEngaged', 'dccEngaged', 'dccNonEngaged', ...
  'decadesEngaged', 'decadesNonEngaged', ...
  'kappaEngagedMean', 'kappaEngagedSem', 'kappaNonEngagedMean', 'kappaNonEngagedSem', ...
  'djsEngagedMean', 'djsEngagedSem', 'djsNonEngagedMean', 'djsNonEngagedSem'};

for t = 1:numel(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  plotData.byType.(typeKey) = init_engagement_metric_type(metricFields, 0);
end

for s = 1:numel(batchResults)
  if ~batchResults(s).success
    continue;
  end
  sessionType = batchResults(s).sessionType;
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_engagement_metric_type(metricFields, numel(plotData.areas));
  end
  typeData = plotData.byType.(typeKey);
  d2Split = batchResults(s).d2Split;
  avByClass = batchResults(s).avalanches.byClass;
  prgSplit = [];
  if isfield(batchResults(s), 'kurtosis') && ~isempty(batchResults(s).kurtosis)
    prgSplit = batchResults(s).kurtosis;
  end
  areaNames = d2Split.areas;
  nonEngagedSec = get_batch_non_engaged_sec(batchResults(s));

  for a = 1:numel(areaNames)
    areaName = areaNames{a};
    areaIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaIdx)
      plotData.areas{end + 1} = areaName; %#ok<AGROW>
      areaIdx = numel(plotData.areas);
      plotData = extend_engagement_plot_areas(plotData, metricFields, areaIdx);
      typeData = plotData.byType.(typeKey);
    end
    typeData = ensure_engagement_area_slots(typeData, metricFields, areaIdx);

    engSummary = summarize_engagement_d2_vector(d2Split.d2{2}{a}, useLog10D2);
    nonSummary = summarize_engagement_d2_vector(d2Split.d2{3}{a}, useLog10D2);
    typeData.d2EngagedMean{areaIdx}(end + 1) = engSummary.mean;
    typeData.d2EngagedSem{areaIdx}(end + 1) = engSummary.sem;
    typeData.d2NonEngagedMean{areaIdx}(end + 1) = nonSummary.mean;
    typeData.d2NonEngagedSem{areaIdx}(end + 1) = nonSummary.sem;

    [tauEng, alphaEng, paramSDEng, dccEng, decadesEng] = get_engagement_area_av_scalars( ...
      avByClass.engaged, areaName);
    [tauNon, alphaNon, paramSDNon, dccNon, decadesNon] = get_engagement_area_av_scalars( ...
      avByClass.nonEngaged, areaName);
    typeData.tauEngaged{areaIdx}(end + 1) = tauEng;
    typeData.tauNonEngaged{areaIdx}(end + 1) = tauNon;
    typeData.alphaEngaged{areaIdx}(end + 1) = alphaEng;
    typeData.alphaNonEngaged{areaIdx}(end + 1) = alphaNon;
    typeData.paramSDEngaged{areaIdx}(end + 1) = paramSDEng;
    typeData.paramSDNonEngaged{areaIdx}(end + 1) = paramSDNon;
    typeData.dccEngaged{areaIdx}(end + 1) = dccEng;
    typeData.dccNonEngaged{areaIdx}(end + 1) = dccNon;
    typeData.decadesEngaged{areaIdx}(end + 1) = decadesEng;
    typeData.decadesNonEngaged{areaIdx}(end + 1) = decadesNon;

    [kappaEngMean, kappaEngSem, djsEngMean, djsEngSem] = ...
      summarize_engagement_prg_area(prgSplit, 'Engaged', areaName);
    [kappaNonMean, kappaNonSem, djsNonMean, djsNonSem] = ...
      summarize_engagement_prg_area(prgSplit, 'Non-engaged', areaName);
    typeData.kappaEngagedMean{areaIdx}(end + 1) = kappaEngMean;
    typeData.kappaEngagedSem{areaIdx}(end + 1) = kappaEngSem;
    typeData.kappaNonEngagedMean{areaIdx}(end + 1) = kappaNonMean;
    typeData.kappaNonEngagedSem{areaIdx}(end + 1) = kappaNonSem;
    typeData.djsEngagedMean{areaIdx}(end + 1) = djsEngMean;
    typeData.djsEngagedSem{areaIdx}(end + 1) = djsEngSem;
    typeData.djsNonEngagedMean{areaIdx}(end + 1) = djsNonMean;
    typeData.djsNonEngagedSem{areaIdx}(end + 1) = djsNonSem;
  end

  typeData.sessionLabels{end + 1} = batchResults(s).label;
  typeData.sessionNames{end + 1} = batchResults(s).sessionName;
  typeData.nonEngagedSec(end + 1) = nonEngagedSec; %#ok<AGROW>
  plotData.byType.(typeKey) = typeData;
end
end

function nonEngagedSec = get_batch_non_engaged_sec(batchResult)
% GET_BATCH_NON_ENGAGED_SEC - Total non-engaged time (s) for one engagement session
%
% Prefers avalanche segment duration; falls back to d2 window-based duration.

nonEngagedSec = nan;
if isfield(batchResult, 'avalanches') && isstruct(batchResult.avalanches) ...
    && isfield(batchResult.avalanches, 'durations') ...
    && isfield(batchResult.avalanches.durations, 'nonEngagedSec') ...
    && isfinite(batchResult.avalanches.durations.nonEngagedSec)
  nonEngagedSec = batchResult.avalanches.durations.nonEngagedSec;
  return;
end
if isfield(batchResult, 'd2Split') && isstruct(batchResult.d2Split) ...
    && isfield(batchResult.d2Split, 'durations') ...
    && isfield(batchResult.d2Split.durations, 'nonEngagedSec') ...
    && isfinite(batchResult.d2Split.durations.nonEngagedSec)
  nonEngagedSec = batchResult.d2Split.durations.nonEngagedSec;
end
end

function typeData = init_engagement_metric_type(metricFields, numAreas)
typeData = struct();
for m = 1:numel(metricFields)
  typeData.(metricFields{m}) = cell(1, numAreas);
  for a = 1:numAreas
    typeData.(metricFields{m}){a} = [];
  end
end
typeData.sessionLabels = {};
typeData.sessionNames = {};
typeData.nonEngagedSec = [];
end

function plotData = extend_engagement_plot_areas(plotData, metricFields, newAreaIdx)
typeNames = fieldnames(plotData.byType);
for i = 1:numel(typeNames)
  typeData = plotData.byType.(typeNames{i});
  typeData = ensure_engagement_area_slots(typeData, metricFields, newAreaIdx);
  plotData.byType.(typeNames{i}) = typeData;
end
end

function typeData = ensure_engagement_area_slots(typeData, metricFields, areaIdx)
for m = 1:numel(metricFields)
  fieldName = metricFields{m};
  while numel(typeData.(fieldName)) < areaIdx
    typeData.(fieldName){end + 1} = []; %#ok<AGROW>
  end
end
end

function stats = summarize_engagement_d2_vector(rawVec, useLog10D2)
% SUMMARIZE_ENGAGEMENT_D2_VECTOR - Mean/SEM of engagement-split d2 windows
%
% Engagement modules apply log10 in the split when useLog10D2 is true, so values
% are already on the plot scale.
stats = struct('mean', nan, 'sem', nan); %#ok<INUSD>
vec = rawVec(:);
vec = vec(isfinite(vec));
if isempty(vec)
  return;
end
stats.mean = mean(vec);
if numel(vec) > 1
  stats.sem = std(vec) / sqrt(numel(vec));
else
  stats.sem = 0;
end
end

function [tauVal, alphaVal, paramSDVal, dccVal, decadesVal] = get_engagement_area_av_scalars( ...
    avClassResult, areaName)
% GET_ENGAGEMENT_AREA_AV_SCALARS - tau, alpha, paramSD, dcc, decades for one area

tauVal = nan;
alphaVal = nan;
paramSDVal = nan;
dccVal = nan;
decadesVal = nan;
if ~isstruct(avClassResult) || ~isfield(avClassResult, 'areas') || ~isfield(avClassResult, 'byArea')
  return;
end
areaIdx = find(strcmp(avClassResult.areas, areaName), 1);
if isempty(areaIdx) || areaIdx > numel(avClassResult.byArea)
  return;
end
avData = avClassResult.byArea{areaIdx};
if ~isstruct(avData) || ~isfield(avData, 'hasAvalanches') || ~avData.hasAvalanches
  return;
end
if isfield(avData, 'tau') && isfinite(avData.tau)
  tauVal = avData.tau;
end
if isfield(avData, 'alpha') && isfinite(avData.alpha)
  alphaVal = avData.alpha;
end
if isfield(avData, 'paramSD') && isfinite(avData.paramSD)
  paramSDVal = avData.paramSD;
end
if isfield(avData, 'dcc') && isfinite(avData.dcc)
  dccVal = avData.dcc;
end
if isfield(avData, 'decades') && isfinite(avData.decades)
  decadesVal = avData.decades;
elseif isfield(avData, 'sizeFitInfo') && isstruct(avData.sizeFitInfo) ...
    && isfield(avData.sizeFitInfo, 'decades') && isfinite(avData.sizeFitInfo.decades)
  decadesVal = avData.sizeFitInfo.decades;
end
end

function [kappaMean, kappaSem, djsMean, djsSem] = summarize_engagement_prg_area( ...
    prgSplit, className, areaName)
% SUMMARIZE_ENGAGEMENT_PRG_AREA - Mean/SEM of kappa and D_JS for one class/area

kappaMean = nan;
kappaSem = nan;
djsMean = nan;
djsSem = nan;
if isempty(prgSplit) || ~isstruct(prgSplit) || ~isfield(prgSplit, 'areas') ...
    || ~isfield(prgSplit, 'classNames')
  return;
end
areaIdx = find(strcmp(prgSplit.areas, areaName), 1);
classIdx = find(strcmpi(prgSplit.classNames, className), 1);
if isempty(areaIdx) || isempty(classIdx)
  return;
end
if isfield(prgSplit, 'kappa') && classIdx <= numel(prgSplit.kappa) ...
    && areaIdx <= numel(prgSplit.kappa{classIdx})
  [kappaMean, kappaSem] = mean_sem_finite_vector(prgSplit.kappa{classIdx}{areaIdx});
end
if isfield(prgSplit, 'djs') && classIdx <= numel(prgSplit.djs) ...
    && areaIdx <= numel(prgSplit.djs{classIdx})
  [djsMean, djsSem] = mean_sem_finite_vector(prgSplit.djs{classIdx}{areaIdx});
end
end

function [meanVal, semVal] = mean_sem_finite_vector(vals)
meanVal = nan;
semVal = nan;
vals = vals(:);
vals = vals(isfinite(vals));
if isempty(vals)
  return;
end
meanVal = mean(vals);
if numel(vals) > 1
  semVal = std(vals) / sqrt(numel(vals));
else
  semVal = 0;
end
end

function [arView, avView, prgView] = build_engagement_class_metric_views(arPlotData, ...
    avPlotData, prgPlotData, engPlotData, engagementClass, sessionTypes, minTimeNonEngaged)
% BUILD_ENGAGEMENT_CLASS_METRIC_VIEWS - Remap engaged/non-engaged into standard plotData
%
% Variables:
%   minTimeNonEngaged - For nonEngaged class, blank (NaN) session metrics when
%                       total non-engaged time is below this threshold (s).
%                       Session slots remain so plots stay aligned.
%
% Returns AR/AV/PRG views. Interval/reach use engagement-split decades and
% PRG (kappa/D_JS); spontaneous (and other non-engagement types) copy from
% full-session AR/AV/PRG batches.

if nargin < 7 || isempty(minTimeNonEngaged)
  minTimeNonEngaged = 0;
end

engagementClass = lower(char(engagementClass));
if strcmp(engagementClass, 'nonengaged')
  engagementClass = 'nonEngaged';
end
if ~ismember(engagementClass, {'engaged', 'nonEngaged'})
  error('engagementClass must be ''engaged'' or ''nonEngaged''.');
end

arView = struct('areas', {{}}, 'sessionTypes', {sessionTypes}, 'byType', struct());
avView = struct('areas', {{}}, 'sessionTypes', {sessionTypes}, 'byType', struct());
prgView = struct('areas', {{}}, 'sessionTypes', {sessionTypes}, 'byType', struct());
if isfield(arPlotData, 'useLog10D2')
  arView.useLog10D2 = arPlotData.useLog10D2;
end

% Union of areas from spontaneous AR/AV/PRG and engagement
areaSet = {};
if isfield(arPlotData, 'areas')
  areaSet = [areaSet, arPlotData.areas]; %#ok<AGROW>
end
if isfield(avPlotData, 'areas')
  areaSet = [areaSet, avPlotData.areas]; %#ok<AGROW>
end
if isfield(prgPlotData, 'areas')
  areaSet = [areaSet, prgPlotData.areas]; %#ok<AGROW>
end
if isfield(engPlotData, 'areas')
  areaSet = [areaSet, engPlotData.areas]; %#ok<AGROW>
end
areaSet = unique(areaSet, 'stable');
arView.areas = areaSet;
avView.areas = areaSet;
prgView.areas = areaSet;

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  isEngType = is_manuscript_engagement_session_type(sessionType);

  arView.byType.(typeKey) = init_standard_ar_type(numel(areaSet));
  avView.byType.(typeKey) = init_standard_av_type(numel(areaSet));
  prgView.byType.(typeKey) = init_standard_prg_type(numel(areaSet));

  if isEngType
    if ~isfield(engPlotData.byType, typeKey)
      continue;
    end
    engType = engPlotData.byType.(typeKey);
    blankMask = false(1, 0);
    if strcmp(engagementClass, 'nonEngaged') && minTimeNonEngaged > 0
      blankMask = resolve_non_engaged_blank_mask(engType, minTimeNonEngaged, sessionType);
    end
    for a = 1:numel(areaSet)
      areaName = areaSet{a};
      engAreaIdx = find(strcmp(engPlotData.areas, areaName), 1);
      if isempty(engAreaIdx)
        continue;
      end
      if strcmp(engagementClass, 'engaged')
        arView.byType.(typeKey).d2Mean{a} = get_eng_series(engType, 'd2EngagedMean', engAreaIdx);
        arView.byType.(typeKey).d2Sem{a} = get_eng_series(engType, 'd2EngagedSem', engAreaIdx);
        avView.byType.(typeKey).tau{a} = get_eng_series(engType, 'tauEngaged', engAreaIdx);
        avView.byType.(typeKey).alpha{a} = get_eng_series(engType, 'alphaEngaged', engAreaIdx);
        avView.byType.(typeKey).paramSD{a} = get_eng_series(engType, 'paramSDEngaged', engAreaIdx);
        avView.byType.(typeKey).dcc{a} = get_eng_series(engType, 'dccEngaged', engAreaIdx);
        avView.byType.(typeKey).decades{a} = get_eng_series(engType, 'decadesEngaged', engAreaIdx);
        avView.byType.(typeKey).decadesSem{a} = zeros(size(avView.byType.(typeKey).decades{a}));
        prgView.byType.(typeKey).kappaMean{a} = get_eng_series(engType, 'kappaEngagedMean', engAreaIdx);
        prgView.byType.(typeKey).kappaSem{a} = get_eng_series(engType, 'kappaEngagedSem', engAreaIdx);
        prgView.byType.(typeKey).djsMean{a} = get_eng_series(engType, 'djsEngagedMean', engAreaIdx);
        prgView.byType.(typeKey).djsSem{a} = get_eng_series(engType, 'djsEngagedSem', engAreaIdx);
      else
        arView.byType.(typeKey).d2Mean{a} = blank_metric_series( ...
          get_eng_series(engType, 'd2NonEngagedMean', engAreaIdx), blankMask);
        arView.byType.(typeKey).d2Sem{a} = blank_metric_series( ...
          get_eng_series(engType, 'd2NonEngagedSem', engAreaIdx), blankMask);
        avView.byType.(typeKey).tau{a} = blank_metric_series( ...
          get_eng_series(engType, 'tauNonEngaged', engAreaIdx), blankMask);
        avView.byType.(typeKey).alpha{a} = blank_metric_series( ...
          get_eng_series(engType, 'alphaNonEngaged', engAreaIdx), blankMask);
        avView.byType.(typeKey).paramSD{a} = blank_metric_series( ...
          get_eng_series(engType, 'paramSDNonEngaged', engAreaIdx), blankMask);
        avView.byType.(typeKey).dcc{a} = blank_metric_series( ...
          get_eng_series(engType, 'dccNonEngaged', engAreaIdx), blankMask);
        avView.byType.(typeKey).decades{a} = blank_metric_series( ...
          get_eng_series(engType, 'decadesNonEngaged', engAreaIdx), blankMask);
        avView.byType.(typeKey).decadesSem{a} = zeros(size(avView.byType.(typeKey).decades{a}));
        prgView.byType.(typeKey).kappaMean{a} = blank_metric_series( ...
          get_eng_series(engType, 'kappaNonEngagedMean', engAreaIdx), blankMask);
        prgView.byType.(typeKey).kappaSem{a} = blank_metric_series( ...
          get_eng_series(engType, 'kappaNonEngagedSem', engAreaIdx), blankMask);
        prgView.byType.(typeKey).djsMean{a} = blank_metric_series( ...
          get_eng_series(engType, 'djsNonEngagedMean', engAreaIdx), blankMask);
        prgView.byType.(typeKey).djsSem{a} = blank_metric_series( ...
          get_eng_series(engType, 'djsNonEngagedSem', engAreaIdx), blankMask);
      end
    end
    arView.byType.(typeKey).sessionNames = get_field_or_empty(engType, 'sessionNames');
    arView.byType.(typeKey).sessionLabels = get_field_or_empty(engType, 'sessionLabels');
    avView.byType.(typeKey).sessionNames = get_field_or_empty(engType, 'sessionNames');
    avView.byType.(typeKey).sessionLabels = get_field_or_empty(engType, 'sessionLabels');
    prgView.byType.(typeKey).sessionNames = get_field_or_empty(engType, 'sessionNames');
    prgView.byType.(typeKey).sessionLabels = get_field_or_empty(engType, 'sessionLabels');
  else
    % Spontaneous (and other non-engagement types): copy from standard batches
    if isfield(arPlotData.byType, typeKey)
      arSrc = arPlotData.byType.(typeKey);
      for a = 1:numel(areaSet)
        srcIdx = find(strcmp(arPlotData.areas, areaSet{a}), 1);
        if isempty(srcIdx)
          continue;
        end
        arView.byType.(typeKey).d2Mean{a} = get_type_metric_cell(arSrc, 'd2Mean', srcIdx);
        arView.byType.(typeKey).d2Sem{a} = get_type_metric_cell(arSrc, 'd2Sem', srcIdx);
        arView.byType.(typeKey).d2ShuffleMean{a} = get_type_metric_cell(arSrc, 'd2ShuffleMean', srcIdx);
        arView.byType.(typeKey).d2ShuffleSem{a} = get_type_metric_cell(arSrc, 'd2ShuffleSem', srcIdx);
      end
      arView.byType.(typeKey).sessionNames = get_field_or_empty(arSrc, 'sessionNames');
      arView.byType.(typeKey).sessionLabels = get_field_or_empty(arSrc, 'sessionLabels');
    end
    if isfield(avPlotData.byType, typeKey)
      avSrc = avPlotData.byType.(typeKey);
      for a = 1:numel(areaSet)
        srcIdx = find(strcmp(avPlotData.areas, areaSet{a}), 1);
        if isempty(srcIdx)
          continue;
        end
        avView.byType.(typeKey).tau{a} = get_type_metric_cell(avSrc, 'tau', srcIdx);
        avView.byType.(typeKey).alpha{a} = get_type_metric_cell(avSrc, 'alpha', srcIdx);
        avView.byType.(typeKey).paramSD{a} = get_type_metric_cell(avSrc, 'paramSD', srcIdx);
        avView.byType.(typeKey).dcc{a} = get_type_metric_cell(avSrc, 'dcc', srcIdx);
        avView.byType.(typeKey).decades{a} = get_type_metric_cell(avSrc, 'decades', srcIdx);
        avView.byType.(typeKey).decadesSem{a} = get_type_metric_cell(avSrc, 'decadesSem', srcIdx);
        avView.byType.(typeKey).tauPermutedMean{a} = get_type_metric_cell(avSrc, 'tauPermutedMean', srcIdx);
        avView.byType.(typeKey).alphaPermutedMean{a} = get_type_metric_cell(avSrc, 'alphaPermutedMean', srcIdx);
        avView.byType.(typeKey).paramSDPermutedMean{a} = get_type_metric_cell(avSrc, 'paramSDPermutedMean', srcIdx);
        avView.byType.(typeKey).dccPermutedMean{a} = get_type_metric_cell(avSrc, 'dccPermutedMean', srcIdx);
        avView.byType.(typeKey).decadesPermutedMean{a} = get_type_metric_cell(avSrc, 'decadesPermutedMean', srcIdx);
      end
      avView.byType.(typeKey).sessionNames = get_field_or_empty(avSrc, 'sessionNames');
      avView.byType.(typeKey).sessionLabels = get_field_or_empty(avSrc, 'sessionLabels');
    end
    if isfield(prgPlotData, 'byType') && isfield(prgPlotData.byType, typeKey)
      prgSrc = prgPlotData.byType.(typeKey);
      for a = 1:numel(areaSet)
        srcIdx = find(strcmp(prgPlotData.areas, areaSet{a}), 1);
        if isempty(srcIdx)
          continue;
        end
        prgView.byType.(typeKey).kappaMean{a} = get_type_metric_cell(prgSrc, 'kappaMean', srcIdx);
        prgView.byType.(typeKey).kappaSem{a} = get_type_metric_cell(prgSrc, 'kappaSem', srcIdx);
        prgView.byType.(typeKey).djsMean{a} = get_type_metric_cell(prgSrc, 'djsMean', srcIdx);
        prgView.byType.(typeKey).djsSem{a} = get_type_metric_cell(prgSrc, 'djsSem', srcIdx);
        prgView.byType.(typeKey).kappaShuffleMean{a} = get_type_metric_cell(prgSrc, 'kappaShuffleMean', srcIdx);
        prgView.byType.(typeKey).kappaShuffleSem{a} = get_type_metric_cell(prgSrc, 'kappaShuffleSem', srcIdx);
        prgView.byType.(typeKey).djsShuffleMean{a} = get_type_metric_cell(prgSrc, 'djsShuffleMean', srcIdx);
        prgView.byType.(typeKey).djsShuffleSem{a} = get_type_metric_cell(prgSrc, 'djsShuffleSem', srcIdx);
      end
      prgView.byType.(typeKey).sessionNames = get_field_or_empty(prgSrc, 'sessionNames');
      prgView.byType.(typeKey).sessionLabels = get_field_or_empty(prgSrc, 'sessionLabels');
    end
  end
end
end

function blankMask = resolve_non_engaged_blank_mask(engType, minTimeNonEngaged, sessionType)
% RESOLVE_NON_ENGAGED_BLANK_MASK - True where non-engaged time is below threshold

nSess = 0;
if isfield(engType, 'sessionNames') && ~isempty(engType.sessionNames)
  nSess = numel(engType.sessionNames);
elseif isfield(engType, 'sessionLabels') && ~isempty(engType.sessionLabels)
  nSess = numel(engType.sessionLabels);
end
blankMask = false(1, nSess);
if nSess == 0 || ~(isfinite(minTimeNonEngaged) && minTimeNonEngaged > 0)
  return;
end

nonEngagedSec = [];
if isfield(engType, 'nonEngagedSec') && ~isempty(engType.nonEngagedSec)
  nonEngagedSec = engType.nonEngagedSec(:)';
end
for i = 1:nSess
  tSec = nan;
  if numel(nonEngagedSec) >= i
    tSec = nonEngagedSec(i);
  end
  if isfinite(tSec) && tSec < minTimeNonEngaged
    blankMask(i) = true;
    sessName = '';
    if isfield(engType, 'sessionNames') && numel(engType.sessionNames) >= i
      sessName = char(engType.sessionNames{i});
    elseif isfield(engType, 'sessionLabels') && numel(engType.sessionLabels) >= i
      sessName = char(engType.sessionLabels{i});
    end
    fprintf(['  Blanking non-engaged [%s] %s: non-engaged time %.1f s ', ...
      '< minTimeNonEngaged %.1f s\n'], sessionType, sessName, tSec, minTimeNonEngaged);
  end
end
end

function series = blank_metric_series(series, blankMask)
% BLANK_METRIC_SERIES - Set selected session indices to NaN (keep length)

if isempty(series) || isempty(blankMask) || ~any(blankMask)
  return;
end
series = series(:)';
n = numel(series);
for i = 1:min(n, numel(blankMask))
  if blankMask(i)
    series(i) = nan;
  end
end
end

function typeData = init_standard_ar_type(numAreas)
typeData = struct();
typeData.d2Mean = cell(1, numAreas);
typeData.d2Sem = cell(1, numAreas);
typeData.d2ShuffleMean = cell(1, numAreas);
typeData.d2ShuffleSem = cell(1, numAreas);
for a = 1:numAreas
  typeData.d2Mean{a} = [];
  typeData.d2Sem{a} = [];
  typeData.d2ShuffleMean{a} = [];
  typeData.d2ShuffleSem{a} = [];
end
typeData.sessionNames = {};
typeData.sessionLabels = {};
end

function typeData = init_standard_av_type(numAreas)
typeData = struct();
typeData.tau = cell(1, numAreas);
typeData.alpha = cell(1, numAreas);
typeData.paramSD = cell(1, numAreas);
typeData.dcc = cell(1, numAreas);
typeData.decades = cell(1, numAreas);
typeData.decadesSem = cell(1, numAreas);
typeData.tauPermutedMean = cell(1, numAreas);
typeData.alphaPermutedMean = cell(1, numAreas);
typeData.paramSDPermutedMean = cell(1, numAreas);
typeData.dccPermutedMean = cell(1, numAreas);
typeData.decadesPermutedMean = cell(1, numAreas);
for a = 1:numAreas
  typeData.tau{a} = [];
  typeData.alpha{a} = [];
  typeData.paramSD{a} = [];
  typeData.dcc{a} = [];
  typeData.decades{a} = [];
  typeData.decadesSem{a} = [];
  typeData.tauPermutedMean{a} = [];
  typeData.alphaPermutedMean{a} = [];
  typeData.paramSDPermutedMean{a} = [];
  typeData.dccPermutedMean{a} = [];
  typeData.decadesPermutedMean{a} = [];
end
typeData.sessionNames = {};
typeData.sessionLabels = {};
end

function typeData = init_standard_prg_type(numAreas)
typeData = struct();
typeData.kappaMean = cell(1, numAreas);
typeData.kappaSem = cell(1, numAreas);
typeData.djsMean = cell(1, numAreas);
typeData.djsSem = cell(1, numAreas);
typeData.kappaShuffleMean = cell(1, numAreas);
typeData.kappaShuffleSem = cell(1, numAreas);
typeData.djsShuffleMean = cell(1, numAreas);
typeData.djsShuffleSem = cell(1, numAreas);
for a = 1:numAreas
  typeData.kappaMean{a} = [];
  typeData.kappaSem{a} = [];
  typeData.djsMean{a} = [];
  typeData.djsSem{a} = [];
  typeData.kappaShuffleMean{a} = [];
  typeData.kappaShuffleSem{a} = [];
  typeData.djsShuffleMean{a} = [];
  typeData.djsShuffleSem{a} = [];
end
typeData.sessionNames = {};
typeData.sessionLabels = {};
end

function series = get_eng_series(typeData, fieldName, areaIdx)
series = [];
if ~isfield(typeData, fieldName) || areaIdx > numel(typeData.(fieldName))
  return;
end
series = typeData.(fieldName){areaIdx};
end

function series = get_type_metric_cell(typeData, fieldName, areaIdx)
series = [];
if ~isfield(typeData, fieldName) || areaIdx > numel(typeData.(fieldName))
  return;
end
series = typeData.(fieldName){areaIdx};
end

function val = get_field_or_empty(s, fieldName)
if isfield(s, fieldName)
  val = s.(fieldName);
else
  val = {};
end
end
