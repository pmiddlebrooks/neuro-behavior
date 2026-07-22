%%
% Criticality Multiple Metrics Across Task Types (Manuscript)
%
% Runs d2 (AR), avalanche (AV), and PRG batch analyses, saves outputs,
% plots aligned multi-metric session summaries on shared axes, and optionally
% a cross-session metric correlation matrix (pooled across task types).
%
% Variables:
%   sessionTypes, collectStart, collectEnd, d2Window, brainArea, areasToPlot
%   runArBatch, runAvBatch, runPrgBatch - Run each analysis pipeline
%   loadSavedResults   - If true, load saved .mat outputs when run*Batch false
%   plotResults        - Create combined d2/tau/alpha figure(s)
%   plotMetricPairScatters - 2x2 figure: d2 vs tau, d2 vs alpha,
%                            paramSD (crackling 1/σνz) vs (α-1)/(τ-1),
%                            d2 vs paramSD
%   plotCorrelationMatrix - Pearson corr heatmap across sessions (all tasks)
%   saveCombinedBatch  - Save merged outputs for plot-only reruns
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
%   anchorMetric       - 'd2', 'tau', or 'alpha' (primary / left axis)
%   metricsToPlot      - Subset of {'d2','tau','alpha'} markers to draw
%   splitByEngagement  - If true, interval/reach use engaged vs non-engaged
%                        analyses; make two plots (engaged and non-engaged),
%                        each including spontaneous alongside that class.
%                        Paired plots share d2-aligned y-limits for comparison.
%                        Correlation matrix always uses full-session metrics.
%
% Goal:
%   One session-grouped plot per brain area with d2, tau, and alpha per session.
%   Optionally anchor non-anchor metrics onto the chosen metric's y-scale via
%   affine maps. Optional pair scatters, separated metric panels, and
%   correlation matrix across sessions.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
collectStart = 0;
collectEnd = 45 * 60;
collectEnd = [];  % [] = full session
d2Window = [];
% One d2 estimate for the full collect window ([] when collectEnd is [])
% d2Window = collectEnd;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};

runArBatch = true;
runAvBatch = true;
runPrgBatch = true;
runEngagementBatch = true;
loadSavedResults = true;
plotResults = true;
plotMetricPairScatters = true;
plotSeparatedMetrics = true;
plotCorrelationMatrix = true;
saveCombinedBatch = false;
enablePermutations = false;
useAnchorAffineMap = false;  % false: native scales with independent right axes
anchorMetric = 'd2';  % 'd2', 'tau', or 'alpha' (primary / left axis)
metricsToPlot = {'d2', 'tau', 'alpha'};  % any non-empty subset
% metricsToPlot = {'d2', 'tau'};  % any non-empty subset
splitByEngagement = false;  % true: engaged / non-engaged plots (spontaneous on both)

useLog10D2 = true;
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 45;
minNeuronsMultiple = 1.2;

powerLawFitMethod = 'plfit2023';
avalancheDetectionMode = 'fixedBinMedian';

prgWindow = d2Window;
if isempty(prgWindow)
  prgWindow = 30;
end
finalCutoffDivisor = 16;
prgMethod = 'pca';

plotConfig = fill_manuscript_plot_config();

setup_criticality_manuscript_paths('criticality_multiple_metrics_across_tasks');
paths = get_paths();

combinedBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_multiple_metrics_across_tasks_batch.mat');
arBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_ar_across_tasks_batch.mat');
avBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_av_across_tasks_batch.mat');
prgBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_prg_across_tasks_batch.mat');
engBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_multiple_metrics_engagement_batch.mat');

fprintf('\n=== Criticality Multiple Metrics Across Tasks ===\n');
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
fprintf('enablePermutations: %d (observed metrics only when false)\n', enablePermutations);
fprintf('useAnchorAffineMap: %d\n', useAnchorAffineMap);
fprintf('anchorMetric: %s\n', anchorMetric);
fprintf('metricsToPlot: %s\n', strjoin(metricsToPlot, ', '));
fprintf('splitByEngagement: %d\n', splitByEngagement);
fprintf('plotMetricPairScatters: %d\n', plotMetricPairScatters);
fprintf('plotSeparatedMetrics: %d\n', plotSeparatedMetrics);
fprintf('plotCorrelationMatrix: %d\n', plotCorrelationMatrix);

% AR batch (d2) — full-session metrics across all requested session types
arOpts = struct( ...
  'sessionTypes', {sessionTypes}, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'd2Window', d2Window, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'areasToPlot', {areasToPlot}, ...
  'useLog10D2', useLog10D2, ...
  'useSubsampling', useSubsampling, ...
  'nSubsamples', nSubsamples, ...
  'nNeuronsSubsample', nNeuronsSubsample, ...
  'minNeuronsMultiple', minNeuronsMultiple, ...
  'enablePermutations', enablePermutations, ...
  'plotResults', false, ...
  'saveBatchResults', true, ...
  'batchResultsFile', arBatchFile);

if runArBatch
  arOut = criticality_ar_across_tasks(arOpts);
elseif loadSavedResults && isfile(arBatchFile)
  arOptsLoad = arOpts;
  arOptsLoad.runBatch = false;
  arOut = criticality_ar_across_tasks(arOptsLoad);
else
  error('AR batch required. Set runArBatch true or provide %s', arBatchFile);
end

% AV batch (tau, alpha, paramSD, decades, dcc)
avOpts = struct( ...
  'sessionTypes', {sessionTypes}, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'areasToPlot', {areasToPlot}, ...
  'powerLawFitMethod', powerLawFitMethod, ...
  'avalancheDetectionMode', avalancheDetectionMode, ...
  'useSubsampling', useSubsampling, ...
  'nSubsamples', nSubsamples, ...
  'nNeuronsSubsample', nNeuronsSubsample, ...
  'minNeuronsMultiple', minNeuronsMultiple, ...
  'enablePermutations', enablePermutations, ...
  'plotResults', false, ...
  'saveBatchResults', true, ...
  'batchResultsFile', avBatchFile);

if runAvBatch
  avOut = criticality_av_across_tasks(avOpts);
elseif loadSavedResults && isfile(avBatchFile)
  avOptsLoad = avOpts;
  avOptsLoad.runBatch = false;
  avOut = criticality_av_across_tasks(avOptsLoad);
else
  error('AV batch required. Set runAvBatch true or provide %s', avBatchFile);
end

% PRG batch (kurtosis / kappaMean, Jensen-Shannon / djsMean)
prgOpts = struct( ...
  'sessionTypes', {sessionTypes}, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'prgWindow', prgWindow, ...
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
  'plotResults', false, ...
  'saveBatchResults', true, ...
  'batchResultsFile', prgBatchFile);

if runPrgBatch
  prgOut = criticality_prg_across_tasks(prgOpts);
elseif loadSavedResults && isfile(prgBatchFile)
  prgOptsLoad = prgOpts;
  prgOptsLoad.runBatch = false;
  prgOut = criticality_prg_across_tasks(prgOptsLoad);
elseif plotCorrelationMatrix
  error('PRG batch required for correlation matrix. Set runPrgBatch true or provide %s', prgBatchFile);
else
  prgOut = [];
  fprintf('Skipping PRG batch (plotCorrelationMatrix false and runPrgBatch false).\n');
end

% Engagement batch (interval/reach engaged vs non-engaged d2 + tau/alpha)
engOut = [];
if splitByEngagement
  engSessionTypes = intersect(sessionTypes, {'interval', 'reach'}, 'stable');
  if isempty(engSessionTypes)
    error('splitByEngagement requires interval and/or reach in sessionTypes.');
  end
  engOpts = struct( ...
    'sessionTypes', {engSessionTypes}, ...
    'collectStart', collectStart, ...
    'collectEnd', collectEnd, ...
    'd2Window', d2Window, ...
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
    'batchResultsFile', engBatchFile, ...
    'plotConfig', plotConfig);

  if runEngagementBatch
    engOut = run_multimetric_engagement_batch(engOpts);
  elseif loadSavedResults && isfile(engBatchFile)
    loadedEng = load(engBatchFile, 'batchResults', 'plotData', 'batchMeta');
    engOut = struct( ...
      'batchResults', loadedEng.batchResults, ...
      'plotData', loadedEng.plotData, ...
      'batchMeta', loadedEng.batchMeta);
    fprintf('\nLoaded engagement batch: %s\n', engBatchFile);
  else
    error('Engagement batch required when splitByEngagement. Set runEngagementBatch true or provide %s', ...
      engBatchFile);
  end
end

combinedOut = struct();
combinedOut.ar = arOut;
combinedOut.av = avOut;
combinedOut.prg = prgOut;
combinedOut.engagement = engOut;
combinedOut.sessionTypes = sessionTypes;
combinedOut.collectStart = collectStart;
combinedOut.collectEnd = collectEnd;
combinedOut.d2Window = d2Window;
combinedOut.prgWindow = prgWindow;
combinedOut.brainArea = brainArea;
combinedOut.useLog10D2 = useLog10D2;
combinedOut.enablePermutations = enablePermutations;
combinedOut.anchorMetric = anchorMetric;
combinedOut.useAnchorAffineMap = useAnchorAffineMap;
combinedOut.metricsToPlot = metricsToPlot;
combinedOut.splitByEngagement = splitByEngagement;
combinedOut.plotMetricPairScatters = plotMetricPairScatters;
combinedOut.plotSeparatedMetrics = plotSeparatedMetrics;
combinedOut.plotCorrelationMatrix = plotCorrelationMatrix;

if saveCombinedBatch
  save(combinedBatchFile, '-struct', 'combinedOut', '-v7.3');
  fprintf('\nSaved combined batch: %s\n', combinedBatchFile);
end

% Combined plotting
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
  metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
  if useAnchorAffineMap && ~ismember(anchorMetric, metricsToPlot)
    error('anchorMetric "%s" must be included in metricsToPlot when useAnchorAffineMap is true.', ...
      anchorMetric);
  end

  fprintf('\n=== Combined plotting ===\n');
  if useAnchorAffineMap
    fprintf('Anchor metric: %s (affine map onto this scale)\n', anchorMetric);
  else
    fprintf('Anchor affine map: off (native metric scales)\n');
  end
  fprintf('Markers: %s\n', strjoin(metricsToPlot, ', '));

  if ~splitByEngagement
    plotAreas = intersect(arOut.plotData.areas, avOut.plotData.areas, 'stable');
    if ~isempty(areasToPlot)
      plotAreas = intersect(plotAreas, areasToPlot, 'stable');
    end
    if isempty(plotAreas)
      error('No common brain areas between AR and AV plotData.');
    end
    fprintf('Areas: %s\n', strjoin(plotAreas, ', '));
    plot_multimetric_d2_tau_alpha_across_tasks(arOut.plotData, avOut.plotData, plotAreas, ...
      sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
      plotConfig, anchorMetric, '', metricsToPlot, struct(), useAnchorAffineMap);
    if plotSeparatedMetrics
      if isempty(prgOut) || ~isfield(prgOut, 'plotData')
        error('PRG plotData required for separated metrics (kurtosis / D_{JS}). Set runPrgBatch true.');
      end
      plot_multimetric_separated_axes_across_tasks(arOut.plotData, avOut.plotData, ...
        prgOut.plotData, plotAreas, sessionTypes, collectStart, collectEnd, d2Window, ...
        paths, brainArea, useLog10D2, plotConfig, '', metricsToPlot, avOut.plotData, ...
        finalCutoffDivisor);
    end
    if plotMetricPairScatters
      plot_multimetric_pair_scatters_across_tasks(arOut.plotData, avOut.plotData, plotAreas, ...
        sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
        plotConfig, '');
    end
  else
    engagementClasses = {'engaged', 'nonEngaged'};
    classViews = struct();
    plotAreas = {};
    for iClass = 1:numel(engagementClasses)
      engClass = engagementClasses{iClass};
      [arView, avView] = build_engagement_class_metric_views( ...
        arOut.plotData, avOut.plotData, engOut.plotData, engClass, sessionTypes);
      classViews.(engClass).ar = arView;
      classViews.(engClass).av = avView;
      classAreas = intersect(arView.areas, avView.areas, 'stable');
      if isempty(plotAreas)
        plotAreas = classAreas;
      else
        plotAreas = intersect(plotAreas, classAreas, 'stable');
      end
    end
    if ~isempty(areasToPlot)
      plotAreas = intersect(plotAreas, areasToPlot, 'stable');
    end
    if isempty(plotAreas)
      error('No common brain areas for engagement paired plots.');
    end

    % Shared maps + y-limits across engaged/non-engaged so d2 axes match
    sharedByArea = compute_shared_engagement_plot_scales(classViews, plotAreas, ...
      sessionTypes, metricsToPlot, anchorMetric, useAnchorAffineMap);

    for iClass = 1:numel(engagementClasses)
      engClass = engagementClasses{iClass};
      fprintf('Areas (%s): %s\n', engClass, strjoin(plotAreas, ', '));
      plot_multimetric_d2_tau_alpha_across_tasks( ...
        classViews.(engClass).ar, classViews.(engClass).av, plotAreas, ...
        sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
        plotConfig, anchorMetric, engClass, metricsToPlot, sharedByArea, useAnchorAffineMap);
      if plotSeparatedMetrics
        if isempty(prgOut) || ~isfield(prgOut, 'plotData')
          error('PRG plotData required for separated metrics (kurtosis / D_{JS}).');
        end
        % Top row: engagement d2/tau/alpha; bottom: full-session decades + PRG
        plot_multimetric_separated_axes_across_tasks( ...
          classViews.(engClass).ar, classViews.(engClass).av, prgOut.plotData, plotAreas, ...
          sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
          plotConfig, engClass, metricsToPlot, avOut.plotData, finalCutoffDivisor);
      end
      if plotMetricPairScatters
        plot_multimetric_pair_scatters_across_tasks( ...
          classViews.(engClass).ar, classViews.(engClass).av, plotAreas, ...
          sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
          plotConfig, engClass);
      end
    end
  end
end

%% Cross-session metric correlation matrix (pooled across task types)
if plotCorrelationMatrix
  if isempty(prgOut) || ~isfield(prgOut, 'plotData')
    error('PRG plotData required for correlation matrix.');
  end
  if isempty(areasToPlot) && ~isempty(brainArea)
    areasToPlot = {brainArea};
  end
  corrAreas = intersect(arOut.plotData.areas, avOut.plotData.areas, 'stable');
  corrAreas = intersect(corrAreas, prgOut.plotData.areas, 'stable');
  if ~isempty(areasToPlot)
    corrAreas = intersect(corrAreas, areasToPlot, 'stable');
  end
  if isempty(corrAreas)
    error('No common brain areas among AR, AV, and PRG for correlation matrix.');
  end
  fprintf('\n=== Metric correlation matrix (sessions pooled across tasks) ===\n');
  fprintf('Areas: %s\n', strjoin(corrAreas, ', '));
  plot_metric_correlation_matrix_across_sessions( ...
    arOut, avOut, prgOut, corrAreas, sessionTypes, collectStart, collectEnd, ...
    d2Window, paths, brainArea, useLog10D2, plotConfig);
end

fprintf('\n=== Done ===\n');

%% Local functions

function plot_metric_correlation_matrix_across_sessions(arOut, avOut, prgOut, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, plotConfig)
% PLOT_METRIC_CORRELATION_MATRIX_ACROSS_SESSIONS - Pearson corr heatmap per area
%
% Variables:
%   arOut/avOut/prgOut - Batch outputs with plotData (and AR batchResults fallback)
%   areasToPlot        - Brain areas to plot
%   sessionTypes       - Session types pooled into one correlation
%
% Goal:
%   Each matrix entry is corr(metric_i, metric_j) across sessions (all tasks).

if nargin < 12 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

metricKeys = {'d2', 'tau', 'alpha', 'paramSD', 'decades', 'dcc', 'kurtosis', 'djs', ...
  'meanSpikesPerBinPerNeuron'};
metricLabels = {'d2', 'tau', 'alpha', '1/\sigma\nu z', 'decades', 'dcc', 'kurtosis', ...
  'JS distance', 'spikes/bin/neuron'};

saveDir = fullfile(paths.dropPath, 'criticality_manuscript', 'figures');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for iArea = 1:numel(areasToPlot)
  areaName = areasToPlot{iArea};
  areaIdxAr = find(strcmp(arOut.plotData.areas, areaName), 1);
  areaIdxAv = find(strcmp(avOut.plotData.areas, areaName), 1);
  areaIdxPrg = find(strcmp(prgOut.plotData.areas, areaName), 1);
  if isempty(areaIdxAr) || isempty(areaIdxAv) || isempty(areaIdxPrg)
    warning('Skipping correlation for area %s (missing in one pipeline).', areaName);
    continue;
  end

  sessionTable = build_correlation_metric_session_table( ...
    arOut, avOut, prgOut, sessionTypes, areaIdxAr, areaIdxAv, areaIdxPrg, areaName);
  if height(sessionTable) < 3
    warning('Skipping correlation for area %s: only %d sessions with metrics.', ...
      areaName, height(sessionTable));
    continue;
  end

  metricMat = nan(height(sessionTable), numel(metricKeys));
  for iMetric = 1:numel(metricKeys)
    metricMat(:, iMetric) = sessionTable.(metricKeys{iMetric});
  end

  corrMat = corrcoef(metricMat, 'Rows', 'pairwise');
  nPair = count_pairwise_session_counts(metricMat);

  fprintf('  %s: %d sessions in correlation table\n', areaName, height(sessionTable));
  for iMetric = 1:numel(metricKeys)
    nValid = sum(isfinite(metricMat(:, iMetric)));
    fprintf('    %s: %d finite\n', metricLabels{iMetric}, nValid);
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

  nMetric = numel(metricLabels);
  set(ax, 'XTick', 1:nMetric, 'XTickLabel', metricLabels, ...
    'YTick', 1:nMetric, 'YTickLabel', metricLabels, ...
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

  titleStr = sprintf('Metric correlations across sessions (%s)', areaName);
  if useLog10D2
    titleStr = [titleStr, '; d2 = log10(d2)']; %#ok<AGROW>
  end
  title(ax, titleStr, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');

  plotBase = make_correlation_matrix_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved correlation matrix: %s\n', fullfile(saveDir, plotBase));
end
end

function sessionTable = build_correlation_metric_session_table(arOut, avOut, prgOut, ...
    sessionTypes, areaIdxAr, areaIdxAv, areaIdxPrg, areaName)
% BUILD_CORRELATION_METRIC_SESSION_TABLE - One row per session, metrics joined by name
%
% Variables:
%   arOut/avOut/prgOut - Pipeline outputs
%   sessionTypes       - Types to pool
%   areaIdx*           - Area indices in each plotData
%   areaName           - Area name (for AR batchResults rate fallback)
%
% Goal:
%   Align d2, AV exponents, PRG kurtosis/JS, and mean spikes/bin/neuron.

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
  if ~isfield(arPlotData.byType, typeKey) || ~isfield(avPlotData.byType, typeKey) ...
      || ~isfield(prgPlotData.byType, typeKey)
    continue;
  end
  arType = arPlotData.byType.(typeKey);
  avType = avPlotData.byType.(typeKey);
  prgType = prgPlotData.byType.(typeKey);
  arNames = get_type_session_names(arType);
  if areaIdxAr > numel(arType.d2Mean) || isempty(arType.d2Mean{areaIdxAr})
    continue;
  end
  numAr = numel(arType.d2Mean{areaIdxAr});

  for i = 1:numAr
    sessionName = arNames{min(i, numel(arNames))};
    avIdx = find_matching_session_index(avType, sessionName, i);
    prgIdx = find_matching_session_index(prgType, sessionName, i);
    if isempty(avIdx) || isempty(prgIdx)
      continue;
    end

    d2Val = get_metric_series_value(arType.d2Mean{areaIdxAr}, i);
    tauVal = get_type_cell_metric(avType, 'tau', areaIdxAv, avIdx);
    alphaVal = get_type_cell_metric(avType, 'alpha', areaIdxAv, avIdx);
    paramSDVal = get_type_cell_metric(avType, 'paramSD', areaIdxAv, avIdx);
    decadesVal = get_type_cell_metric(avType, 'decades', areaIdxAv, avIdx);
    dccVal = get_type_cell_metric(avType, 'dcc', areaIdxAv, avIdx);
    kurtosisVal = get_type_cell_metric(prgType, 'kappaMean', areaIdxPrg, prgIdx);
    djsVal = get_type_cell_metric(prgType, 'djsMean', areaIdxPrg, prgIdx);
    meanSpikesVal = get_mean_spikes_per_bin_per_neuron_session( ...
      arType, arOut, sessionType, sessionName, areaIdxAr, i, areaName);

    % Keep session if at least two metrics are finite (pairwise corr handles the rest)
    metricVec = [d2Val, tauVal, alphaVal, paramSDVal, decadesVal, dccVal, ...
      kurtosisVal, djsVal, meanSpikesVal];
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
    collectStart, collectEnd, useLog10D2)
% MAKE_CORRELATION_MATRIX_PLOT_BASENAME - File stem for correlation heatmap
if isempty(brainArea)
  areaTag = areaName;
else
  areaTag = brainArea;
end
if isempty(d2Window)
  winTag = 'full';
else
  winTag = sprintf('win%.0fs', d2Window);
end
if isempty(collectEnd)
  timeTag = sprintf('%.0f-full', collectStart);
else
  timeTag = sprintf('%.0f-%.0fs', collectStart, collectEnd);
end
logTag = '';
if useLog10D2
  logTag = '_log10d2';
end
plotBase = sprintf('metric_corr_across_sessions_%s_%s_%s%s', areaTag, winTag, timeTag, logTag);
end

function plot_multimetric_d2_tau_alpha_across_tasks(arPlotData, avPlotData, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    plotConfig, anchorMetric, engagementTag, metricsToPlot, sharedByArea, useAnchorAffineMap)
% PLOT_MULTIMETRIC_D2_TAU_ALPHA_ACROSS_TASKS - Aligned d2/tau/alpha session plot
%
% Variables:
%   useAnchorAffineMap - If true, affine-map non-anchor metrics onto anchorMetric

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
    areaIdxAr, areaIdxAv, metricsToPlot);
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
  if isempty(d2Window)
    winTag = 'full';
  else
    winTag = sprintf('%.0fs', d2Window);
  end
  engTitle = format_engagement_title_tag(engagementTag);
  metricTitle = strjoin(metricsToPlot, ', ');
  if useAnchorAffineMap
    scaleTag = sprintf('anchor=%s', anchorMetric);
  else
    scaleTag = 'native scales';
  end
  if ~isempty(brainArea)
    titleStr = sprintf('%s (%s)%s — %s [%s, %s d2 windows]', ...
      metricTitle, scaleTag, engTitle, brainArea, collectTag, winTag);
  else
    titleStr = sprintf('%s (%s)%s — %s [%s, %s d2 windows]', ...
      metricTitle, scaleTag, engTitle, areaName, collectTag, winTag);
  end
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_multimetric_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, anchorMetric, engagementTag, metricsToPlot, ...
    useAnchorAffineMap);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
end

fprintf('\nAll combined figures saved to %s\n', saveDir);
end

function plot_multimetric_separated_axes_across_tasks(arPlotData, avPlotData, prgPlotData, ...
    areasToPlot, sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, ...
    useLog10D2, plotConfig, engagementTag, metricsToPlot, avPlotDataDecades, ...
    finalCutoffDivisor)
% PLOT_MULTIMETRIC_SEPARATED_AXES_ACROSS_TASKS - 2x4 panels of session metrics
%
% Layout:
%   Top:    D2 | Avalanche Sizes | Avalanche Durations | Crackling 1/σνz
%   Bottom: Scale Range | dcc | Renorm: Kurtosis | Renorm: JS-Distance
%
% Variables:
%   arPlotData / avPlotData - Sources for d2 / tau / alpha / paramSD / dcc
%                             (may be engagement views)
%   prgPlotData             - PRG plotData for kurtosis (kappaMean) and D_JS
%   avPlotDataDecades       - AV plotData used for decades (full-session when
%                             engagement av views lack decades); defaults to avPlotData
%   metricsToPlot           - Controls which of d2/tau/alpha appear in the top row
%   finalCutoffDivisor      - PRG kappa reported at N/finalCutoffDivisor (ylabel)
%
% Goal:
%   Same session-level data as the combined plot, each metric on its own axis.
%   Within each task type, consecutive session markers are connected by a line.

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
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
engagementTag = char(engagementTag);

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
% d2 / paramSD / decades / dcc / kurtosis / Djs: filled circles; tau square; alpha diamond
markerByKey = struct('d2', 'o', 'tau', 's', 'alpha', 'd', ...
  'paramSD', 'o', 'decades', 'o', 'dcc', 'o', 'kurtosis', 'o', 'djs', 'o');
fillByKey = struct('d2', true, 'tau', false, 'alpha', false, ...
  'paramSD', true, 'decades', true, 'dcc', true, 'kurtosis', true, 'djs', true);

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
  if isempty(areaIdxAr) || isempty(areaIdxAv)
    continue;
  end
  if isempty(areaIdxAvDec)
    areaIdxAvDec = areaIdxAv;
    avPlotDataDecades = avPlotData;
  end
  if isempty(areaIdxPrg)
    warning('Skipping separated metrics for %s: area missing in PRG plotData.', areaName);
    continue;
  end

  sessionTable = build_separated_metrics_session_table( ...
    arPlotData, avPlotData, prgPlotData, avPlotDataDecades, sessionTypes, ...
    areaIdxAr, areaIdxAv, areaIdxPrg, areaIdxAvDec, {'d2', 'tau', 'alpha'});
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

      faceColor = taskColor;
      if ~fillByKey.(metricKey)
        faceColor = 'none';
      end

      validLine = isfinite(xPos(:)) & isfinite(yPos(:));
      if sum(validLine) >= 2
        plot(ax, xPos(validLine), yPos(validLine), '-', ...
          'Color', 0.55 * taskColor + 0.45 * [1 1 1], ...
          'LineWidth', max(1, plotConfig.lineWidth - 0.25), 'HandleVisibility', 'off');
      end

      plot_metric_errorbar_group(ax, xPos, yPos, ySem, ...
        markerByKey.(metricKey), taskColor, faceColor, plotConfig);

      for i = 1:numSessions
        xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
        xtickLabels{end + 1} = char(sessionTable.sessionLabel(rowIdx(i))); %#ok<AGROW>
      end
      xCursor = xPos(end) + 1.5;
    end

    yLimPlot = compute_native_ylim_for_metric(sessionTable.(yField));
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
  if isempty(d2Window)
    winTag = 'full';
  else
    winTag = sprintf('%.0fs', d2Window);
  end
  engTitle = format_engagement_title_tag(engagementTag);
  if ~isempty(brainArea)
    titleStr = sprintf('Separated metrics%s — %s [%s, %s d2 windows]', ...
      engTitle, brainArea, collectTag, winTag);
  else
    titleStr = sprintf('Separated metrics%s — %s [%s, %s d2 windows]', ...
      engTitle, areaName, collectTag, winTag);
  end
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_separated_metrics_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, engagementTag, panelMetricKeys);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved separated metrics: %s\n', fullfile(saveDir, plotBase));
end
end

function sessionTable = build_separated_metrics_session_table(arPlotData, avPlotData, ...
    prgPlotData, avPlotDataDecades, sessionTypes, areaIdxAr, areaIdxAv, areaIdxPrg, ...
    areaIdxAvDec, topMetrics)
% BUILD_SEPARATED_METRICS_SESSION_TABLE - d2/tau/alpha + crackling + PRG metrics
%
% Variables:
%   avPlotDataDecades - AV source for decades (may differ from avPlotData)
%   topMetrics        - Which of d2/tau/alpha must be finite to keep a session
%
% Goal:
%   Align top-row metrics with paramSD/dcc (AV), decades (AV), and PRG kurtosis / D_JS.

if nargin < 10 || isempty(topMetrics)
  topMetrics = {'d2', 'tau', 'alpha'};
end
topMetrics = normalize_metrics_to_plot(topMetrics);

baseTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
  areaIdxAr, areaIdxAv, topMetrics);
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

for i = 1:nRow
  sessionType = baseTable.sessionType{i};
  sessionName = baseTable.sessionName{i};
  typeKey = matlab.lang.makeValidName(sessionType);
  % Within-type index for fallback (not the global table row)
  typeRows = find(strcmp(baseTable.sessionType, sessionType));
  withinTypeIdx = find(typeRows == i, 1);

  if isfield(avPlotData.byType, typeKey)
    avType = avPlotData.byType.(typeKey);
    avIdx = find_matching_session_index(avType, sessionName, withinTypeIdx);
    if ~isempty(avIdx)
      paramSDCol(i) = get_type_cell_metric(avType, 'paramSD', areaIdxAv, avIdx);
      dccCol(i) = get_type_cell_metric(avType, 'dcc', areaIdxAv, avIdx);
    end
  end

  if isfield(avPlotDataDecades.byType, typeKey)
    avDecType = avPlotDataDecades.byType.(typeKey);
    avDecIdx = find_matching_session_index(avDecType, sessionName, withinTypeIdx);
    if ~isempty(avDecIdx)
      decadesCol(i) = get_type_cell_metric(avDecType, 'decades', areaIdxAvDec, avDecIdx);
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
      if ~isfinite(kurtosisSemCol(i)), kurtosisSemCol(i) = 0; end
      if ~isfinite(djsSemCol(i)), djsSemCol(i) = 0; end
    end
  end
end

sessionTable = [baseTable, table(paramSDCol, paramSDSemCol, dccCol, dccSemCol, ...
  decadesCol, decadesSemCol, kurtosisCol, kurtosisSemCol, djsCol, djsSemCol, ...
  'VariableNames', {'paramSD', 'paramSDSem', 'dcc', 'dccSem', ...
  'decades', 'decadesSem', 'kurtosis', 'kurtosisSem', 'djs', 'djsSem'})];
end

function plot_multimetric_pair_scatters_across_tasks(arPlotData, avPlotData, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    plotConfig, engagementTag)
% PLOT_MULTIMETRIC_PAIR_SCATTERS_ACROSS_TASKS - 2x2 session metric scatters
%
% Panels (titles are Y vs X):
%   (1,1) Avalanche Sizes vs D2
%   (1,2) Avalanche Durations vs D2
%   (2,1) Measured vs Predicted Crackling Exponent
%   (2,2) Crackling Exponent vs D2
%
% Crackling naming:
%   Measured / observed crackling exponent = paramSD = 1/σνz from WLS ⟨S⟩(T)
%   Predicted crackling exponent = (α-1)/(τ-1) from the size/duration PDFs
%   dcc = |predicted - measured|
%
% Variables:
%   arPlotData / avPlotData - Aggregated plotData from AR / AV batches
%   areasToPlot             - Brain areas to plot (one figure each)
%   sessionTypes            - Session types (points colored by type)
%   engagementTag           - Optional engaged / nonEngaged suffix
%
% Goal:
%   Cross-session relationships among d2 and avalanche / crackling exponents.

if nargin < 11 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 12 || isempty(engagementTag)
  engagementTag = '';
end
engagementTag = char(engagementTag);

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
  if isempty(areaIdxAr) || isempty(areaIdxAv)
    continue;
  end

  sessionTable = build_pair_scatter_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv);
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
      plotConfig.scatterMarkerSize = 120;
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
  if isempty(d2Window)
    winTag = 'full';
  else
    winTag = sprintf('%.0fs', d2Window);
  end
  engTitle = format_engagement_title_tag(engagementTag);
  if ~isempty(brainArea)
    titleStr = sprintf('Session metric pairs%s — %s [%s, %s d2 windows]', ...
      engTitle, brainArea, collectTag, winTag);
  else
    titleStr = sprintf('Session metric pairs%s — %s [%s, %s d2 windows]', ...
      engTitle, areaName, collectTag, winTag);
  end
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_pair_scatter_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, engagementTag);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved pair scatters: %s\n', fullfile(saveDir, plotBase));
end
end

function sessionTable = build_pair_scatter_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv)
% BUILD_PAIR_SCATTER_SESSION_TABLE - d2/tau/alpha + paramSD and γ_pred
%
% Goal:
%   Align AR d2 with AV tau, alpha, and crackling paramSD (1/σνz). Also store
%   predicted crackling exponent (α-1)/(τ-1) for the identity-line panel.

baseTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
  areaIdxAr, areaIdxAv, {'d2', 'tau', 'alpha'});
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
    areaIdxAr, areaIdxAv, metricsToPlot)
% BUILD_MULTIMETRIC_SESSION_TABLE - Align d2/tau/alpha per session across pipelines

if nargin < 6 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
end
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
needD2 = ismember('d2', metricsToPlot);
needTau = ismember('tau', metricsToPlot);
needAlpha = ismember('alpha', metricsToPlot);
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
  if ~isfield(arPlotData.byType, typeKey)
    continue;
  end
  if needAv && ~isfield(avPlotData.byType, typeKey)
    continue;
  end
  arType = arPlotData.byType.(typeKey);
  if needD2 && (areaIdxAr > numel(arType.d2Mean) || isempty(arType.d2Mean{areaIdxAr}))
    continue;
  end
  avType = [];
  if needAv
    avType = avPlotData.byType.(typeKey);
    if areaIdxAv > numel(avType.tau)
      continue;
    end
  end

  arNames = get_type_session_names(arType);
  if needD2
    numAr = numel(arType.d2Mean{areaIdxAr});
  else
    numAr = numel(arNames);
  end
  for i = 1:numAr
    sessionName = arNames{min(i, numel(arNames))};
    avIdx = i;
    if needAv
      avIdx = find_matching_session_index(avType, sessionName, i);
      if isempty(avIdx)
        continue;
      end
    end

    d2Val = nan;
    tauVal = nan;
    alphaVal = nan;
    d2SemVal = nan;
    if needD2
      d2Val = arType.d2Mean{areaIdxAr}(i);
      d2SemVal = get_metric_series_value(arType.d2Sem{areaIdxAr}, i);
    end
    if needTau
      tauVal = avType.tau{areaIdxAv}(avIdx);
    end
    if needAlpha
      alphaVal = avType.alpha{areaIdxAv}(avIdx);
    end

    checkVals = [];
    if needD2, checkVals(end + 1) = d2Val; end %#ok<AGROW>
    if needTau, checkVals(end + 1) = tauVal; end %#ok<AGROW>
    if needAlpha, checkVals(end + 1) = alphaVal; end %#ok<AGROW>
    if isempty(checkVals) || ~all(isfinite(checkVals))
      continue;
    end

    sessionTypeCol{end + 1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = sessionName; %#ok<AGROW>
    sessionLabelCol{end + 1, 1} = get_session_display_label(arType, i, sessionType); %#ok<AGROW>
    d2MeanCol(end + 1, 1) = d2Val; %#ok<AGROW>
    d2SemCol(end + 1, 1) = d2SemVal; %#ok<AGROW>
    tauMeanCol(end + 1, 1) = tauVal; %#ok<AGROW>
    tauSemCol(end + 1, 1) = 0; %#ok<AGROW>
    alphaMeanCol(end + 1, 1) = alphaVal; %#ok<AGROW>
    alphaSemCol(end + 1, 1) = 0; %#ok<AGROW>
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
    collectStart, collectEnd, useLog10D2, engagementTag, metricsToPlot)
% MAKE_SEPARATED_METRICS_PLOT_BASENAME - File stem for separated metric panels
if nargin < 7 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 8 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha', 'paramSD', 'decades', 'dcc', 'kurtosis', 'djs'};
end
if ischar(metricsToPlot) || isstring(metricsToPlot)
  metricsToPlot = cellstr(metricsToPlot);
end
metricsToPlot = metricsToPlot(:)';
collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
if isempty(d2Window)
  winTag = 'full';
else
  winTag = sprintf('%.0fs', d2Window);
end
if ~isempty(brainArea)
  areaTag = brainArea;
else
  areaTag = areaName;
end
metricTag = strjoin(metricsToPlot, '-');
plotBase = sprintf('criticality_separated_metrics_%s_%s_win%s_%s', ...
  metricTag, areaTag, winTag, collectTag);
if ~isempty(engagementTag)
  plotBase = sprintf('%s_%s', plotBase, engagementTag);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
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
    useAnchorAffineMap)
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
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
if isempty(d2Window)
  winTag = 'full';
else
  winTag = sprintf('%.0fs', d2Window);
end
metricTag = strjoin(metricsToPlot, '-');
if ~isempty(brainArea)
  plotBase = sprintf('criticality_multiple_metrics_%s_%s_win%s_%s', ...
    metricTag, brainArea, winTag, collectTag);
else
  plotBase = sprintf('criticality_multiple_metrics_%s_%s_win%s_%s', ...
    metricTag, areaName, winTag, collectTag);
end
if useAnchorAffineMap
  plotBase = sprintf('%s_anchor%s', plotBase, anchorMetric);
else
  plotBase = [plotBase, '_native'];
end
if ~isempty(engagementTag)
  plotBase = sprintf('%s_%s', plotBase, engagementTag);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
end

function plotBase = make_pair_scatter_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, engagementTag)
% MAKE_PAIR_SCATTER_PLOT_BASENAME - File stem for 1x3 metric pair scatters
if nargin < 7 || isempty(engagementTag)
  engagementTag = '';
end
collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
if isempty(d2Window)
  winTag = 'full';
else
  winTag = sprintf('%.0fs', d2Window);
end
if ~isempty(brainArea)
  areaTag = brainArea;
else
  areaTag = areaName;
end
plotBase = sprintf('criticality_metric_pair_scatters_%s_win%s_%s', areaTag, winTag, collectTag);
if ~isempty(engagementTag)
  plotBase = sprintf('%s_%s', plotBase, engagementTag);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
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
% RUN_MULTIMETRIC_ENGAGEMENT_BATCH - Interval/reach d2 + AV by engagement class

fprintf('\n=== Engagement batch (d2 + avalanches) ===\n');
fprintf('Session types: %s\n', strjoin(opts.sessionTypes, ', '));

sessionTable = build_multimetric_engagement_session_table(opts.sessionTypes);
numSessions = size(sessionTable, 1);
fprintf('Engagement sessions: %d\n', numSessions);
if numSessions == 0
  error('No interval/reach sessions found for engagement batch.');
end

batchResults = repmat(struct( ...
  'sessionType', '', 'sessionName', '', 'subjectName', '', 'label', '', ...
  'success', false, 'skipReason', '', 'd2Split', [], 'avalanches', []), ...
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

  try
    engModOpts = build_multimetric_engagement_module_opts(opts, sessionType);
    if strcmpi(sessionType, 'reach')
      sessionOut = reach_criticality_metrics_engagement(sessionName, engModOpts);
    else
      sessionOut = interval_criticality_metrics_engagement(subjectName, sessionName, engModOpts);
    end
    if isempty(sessionOut.d2) || isempty(sessionOut.avalanches)
      fprintf('  Incomplete engagement outputs; skipping.\n');
      batchResults(s).skipReason = 'incomplete engagement outputs';
      continue;
    end
    batchResults(s).d2Split = sessionOut.d2;
    batchResults(s).avalanches = sessionOut.avalanches;
    batchResults(s).success = true;
    fprintf('  Engagement analysis completed.\n');
  catch ME
    if contains(ME.message, 'No valid areas to process') ...
        || contains(ME.message, 'insufficient neurons') ...
        || contains(ME.message, 'TooFewNeurons') ...
        || contains(ME.message, 'not available')
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

save(opts.batchResultsFile, 'batchResults', 'plotData', 'batchMeta', '-v7.3');
fprintf('\nSaved engagement batch: %s\n', opts.batchResultsFile);

engOut = struct('batchResults', batchResults, 'plotData', plotData, 'batchMeta', batchMeta);
end

function engModOpts = build_multimetric_engagement_module_opts(opts, sessionType)
% BUILD_MULTIMETRIC_ENGAGEMENT_MODULE_OPTS - Opts for reach/interval engagement

if strcmpi(sessionType, 'reach')
  engModOpts = reach_criticality_metrics_engagement();
else
  engModOpts = interval_criticality_metrics_engagement();
end

engModOpts.collectStart = opts.collectStart;
engModOpts.collectEnd = opts.collectEnd;
engModOpts.brainArea = opts.brainArea;
engModOpts.brainAreaCombinations = opts.brainAreaCombinations;
engModOpts.d2Window = opts.d2Window;
% Empty d2Window is resolved to loaded session duration inside engagement modules
engModOpts.useLog10D2 = opts.useLog10D2;
engModOpts.useSubsampling = opts.useSubsampling;
engModOpts.nSubsamples = opts.nSubsamples;
engModOpts.nNeuronsSubsample = opts.nNeuronsSubsample;
engModOpts.minNeuronsMultiple = opts.minNeuronsMultiple;
engModOpts.powerLawFitMethod = opts.powerLawFitMethod;
engModOpts.avalancheDetectionMode = opts.avalancheDetectionMode;
engModOpts.enableCircularPermutations = logical(opts.enablePermutations);
if opts.enablePermutations
  engModOpts.nShuffles = 5;
  engModOpts.nShufflesD2 = 10;
else
  engModOpts.nShuffles = 0;
  engModOpts.nShufflesD2 = 1;
end
engModOpts.analyses = {'d2', 'avalanches'};
engModOpts.makePlots = false;
engModOpts.saveFigure = false;
engModOpts.plotConfig = opts.plotConfig;
if strcmpi(sessionType, 'reach')
  engModOpts.runD2AccuracyCorrelation = false;
  engModOpts.runD2ReachRateCorrelation = false;
else
  engModOpts.runD2TrialRateCorrelation = false;
end
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
switch lower(sessionType)
  case 'interval'
    entries = interval_session_list();
  case 'reach'
    names = reach_session_list();
    entries = struct('subjectName', {}, 'sessionName', {});
    for i = 1:numel(names)
      entries(i).subjectName = ''; %#ok<AGROW>
      entries(i).sessionName = names{i}; %#ok<AGROW>
    end
  otherwise
    entries = struct('subjectName', {}, 'sessionName', {});
end
end

function plotData = aggregate_multimetric_engagement_plot_data(batchResults, sessionTypes, useLog10D2)
% AGGREGATE_MULTIMETRIC_ENGAGEMENT_PLOT_DATA - Per-session engaged/non-engaged metrics

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.useLog10D2 = useLog10D2;

metricFields = { ...
  'd2EngagedMean', 'd2EngagedSem', 'd2NonEngagedMean', 'd2NonEngagedSem', ...
  'tauEngaged', 'tauNonEngaged', 'alphaEngaged', 'alphaNonEngaged', ...
  'paramSDEngaged', 'paramSDNonEngaged', 'dccEngaged', 'dccNonEngaged'};

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
  areaNames = d2Split.areas;

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

    [tauEng, alphaEng, paramSDEng, dccEng] = get_engagement_area_av_scalars( ...
      avByClass.engaged, areaName);
    [tauNon, alphaNon, paramSDNon, dccNon] = get_engagement_area_av_scalars( ...
      avByClass.nonEngaged, areaName);
    typeData.tauEngaged{areaIdx}(end + 1) = tauEng;
    typeData.tauNonEngaged{areaIdx}(end + 1) = tauNon;
    typeData.alphaEngaged{areaIdx}(end + 1) = alphaEng;
    typeData.alphaNonEngaged{areaIdx}(end + 1) = alphaNon;
    typeData.paramSDEngaged{areaIdx}(end + 1) = paramSDEng;
    typeData.paramSDNonEngaged{areaIdx}(end + 1) = paramSDNon;
    typeData.dccEngaged{areaIdx}(end + 1) = dccEng;
    typeData.dccNonEngaged{areaIdx}(end + 1) = dccNon;
  end

  typeData.sessionLabels{end + 1} = batchResults(s).label;
  typeData.sessionNames{end + 1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
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

function [tauVal, alphaVal, paramSDVal, dccVal] = get_engagement_area_av_scalars(avClassResult, areaName)
% GET_ENGAGEMENT_AREA_AV_SCALARS - tau, alpha, paramSD (1/σνz), dcc for one area

tauVal = nan;
alphaVal = nan;
paramSDVal = nan;
dccVal = nan;
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
end

function [arView, avView] = build_engagement_class_metric_views(arPlotData, avPlotData, ...
    engPlotData, engagementClass, sessionTypes)
% BUILD_ENGAGEMENT_CLASS_METRIC_VIEWS - Remap engaged/non-engaged into standard plotData

engagementClass = lower(char(engagementClass));
if strcmp(engagementClass, 'nonengaged')
  engagementClass = 'nonEngaged';
end
if ~ismember(engagementClass, {'engaged', 'nonEngaged'})
  error('engagementClass must be ''engaged'' or ''nonEngaged''.');
end

arView = struct('areas', {{}}, 'sessionTypes', {sessionTypes}, 'byType', struct());
avView = struct('areas', {{}}, 'sessionTypes', {sessionTypes}, 'byType', struct());
if isfield(arPlotData, 'useLog10D2')
  arView.useLog10D2 = arPlotData.useLog10D2;
end

% Union of areas from spontaneous AR/AV and engagement
areaSet = {};
if isfield(arPlotData, 'areas')
  areaSet = [areaSet, arPlotData.areas]; %#ok<AGROW>
end
if isfield(avPlotData, 'areas')
  areaSet = [areaSet, avPlotData.areas]; %#ok<AGROW>
end
if isfield(engPlotData, 'areas')
  areaSet = [areaSet, engPlotData.areas]; %#ok<AGROW>
end
areaSet = unique(areaSet, 'stable');
arView.areas = areaSet;
avView.areas = areaSet;

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  isEngType = ismember(lower(sessionType), {'interval', 'reach'});

  arView.byType.(typeKey) = init_standard_ar_type(numel(areaSet));
  avView.byType.(typeKey) = init_standard_av_type(numel(areaSet));

  if isEngType
    if ~isfield(engPlotData.byType, typeKey)
      continue;
    end
    engType = engPlotData.byType.(typeKey);
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
      else
        arView.byType.(typeKey).d2Mean{a} = get_eng_series(engType, 'd2NonEngagedMean', engAreaIdx);
        arView.byType.(typeKey).d2Sem{a} = get_eng_series(engType, 'd2NonEngagedSem', engAreaIdx);
        avView.byType.(typeKey).tau{a} = get_eng_series(engType, 'tauNonEngaged', engAreaIdx);
        avView.byType.(typeKey).alpha{a} = get_eng_series(engType, 'alphaNonEngaged', engAreaIdx);
        avView.byType.(typeKey).paramSD{a} = get_eng_series(engType, 'paramSDNonEngaged', engAreaIdx);
        avView.byType.(typeKey).dcc{a} = get_eng_series(engType, 'dccNonEngaged', engAreaIdx);
      end
    end
    arView.byType.(typeKey).sessionNames = get_field_or_empty(engType, 'sessionNames');
    arView.byType.(typeKey).sessionLabels = get_field_or_empty(engType, 'sessionLabels');
    avView.byType.(typeKey).sessionNames = get_field_or_empty(engType, 'sessionNames');
    avView.byType.(typeKey).sessionLabels = get_field_or_empty(engType, 'sessionLabels');
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
      end
      avView.byType.(typeKey).sessionNames = get_field_or_empty(avSrc, 'sessionNames');
      avView.byType.(typeKey).sessionLabels = get_field_or_empty(avSrc, 'sessionLabels');
    end
  end
end
end

function typeData = init_standard_ar_type(numAreas)
typeData = struct();
typeData.d2Mean = cell(1, numAreas);
typeData.d2Sem = cell(1, numAreas);
for a = 1:numAreas
  typeData.d2Mean{a} = [];
  typeData.d2Sem{a} = [];
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
for a = 1:numAreas
  typeData.tau{a} = [];
  typeData.alpha{a} = [];
  typeData.paramSD{a} = [];
  typeData.dcc{a} = [];
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
