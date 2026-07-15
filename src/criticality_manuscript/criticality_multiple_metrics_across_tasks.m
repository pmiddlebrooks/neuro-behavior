%%
% Criticality Multiple Metrics Across Task Types (Manuscript)
%
% Runs d2 (AR) and avalanche tau/alpha (AV) batch analyses, saves outputs,
% and plots aligned multi-metric session summaries on shared axes.
%
% Variables:
%   sessionTypes, collectStart, collectEnd, d2Window, brainArea, areasToPlot
%   runArBatch, runAvBatch - Run each analysis pipeline
%   loadSavedResults   - If true, load saved .mat outputs when run*Batch false
%   plotResults        - Create combined d2/tau/alpha figure(s)
%   saveCombinedBatch  - Save merged outputs for plot-only reruns
%   enablePermutations - If false, observed metrics only (no shuffles; faster)
%   anchorMetric       - 'd2', 'tau', or 'alpha'; others affine-map onto this scale
%   metricsToPlot      - Subset of {'d2','tau','alpha'} markers to draw
%   splitByEngagement  - If true, interval/reach use engaged vs non-engaged
%                        analyses; make two plots (engaged and non-engaged),
%                        each including spontaneous alongside that class.
%                        Paired plots share d2-aligned y-limits for comparison.
%
% Goal:
%   One session-grouped plot per brain area with d2, tau, and alpha per session.
%   Plot is anchored on the chosen metric's native y-limits; the other metrics
%   are affine-mapped into that space to minimize within-session differences.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
collectStart = 0;
collectEnd = 45 * 60;
collectEnd = [];  % [] = full session
d2Window = 30;
% One d2 estimate for the full collect window ([] when collectEnd is [])
% d2Window = collectEnd;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};

runArBatch = true;
runAvBatch = true;
runEngagementBatch = true;
loadSavedResults = true;
plotResults = true;
saveCombinedBatch = false;
enablePermutations = false;
anchorMetric = 'd2';  % 'd2', 'tau', or 'alpha'
metricsToPlot = {'d2', 'tau', 'alpha'};  % any non-empty subset
% metricsToPlot = {'d2', 'tau'};  % any non-empty subset
splitByEngagement = true;  % true: engaged / non-engaged plots (spontaneous on both)

useLog10D2 = true;
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 45;
minNeuronsMultiple = 1.2;

powerLawFitMethod = 'plfit2023';
avalancheDetectionMode = 'fixedBinMedian';

plotConfig = fill_manuscript_plot_config();

setup_criticality_manuscript_paths('criticality_multiple_metrics_across_tasks');
paths = get_paths();

combinedBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_multiple_metrics_across_tasks_batch.mat');
arBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_ar_across_tasks_batch.mat');
avBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_av_across_tasks_batch.mat');
engBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_multiple_metrics_engagement_batch.mat');
if splitByEngagement
  arBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
    'criticality_ar_across_tasks_spontaneous_for_engagement_batch.mat');
  avBatchFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
    'criticality_av_across_tasks_spontaneous_for_engagement_batch.mat');
end

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
fprintf('anchorMetric: %s\n', anchorMetric);
fprintf('metricsToPlot: %s\n', strjoin(metricsToPlot, ', '));
fprintf('splitByEngagement: %d\n', splitByEngagement);

%% AR batch (d2) — full-session / all-window metrics (used for spontaneous, or all types)
arSessionTypes = sessionTypes;
if splitByEngagement
  % Interval/reach come from engagement batch; keep spontaneous (and any other
  % non-engagement types) on the shared AR pipeline.
  arSessionTypes = setdiff(sessionTypes, {'interval', 'reach'}, 'stable');
  if isempty(arSessionTypes)
    arSessionTypes = {'spontaneous'};
  end
end
arOpts = struct( ...
  'sessionTypes', {arSessionTypes}, ...
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

%% AV batch (tau, alpha)
avSessionTypes = arSessionTypes;
avOpts = struct( ...
  'sessionTypes', {avSessionTypes}, ...
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

%% Engagement batch (interval/reach engaged vs non-engaged d2 + tau/alpha)
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
combinedOut.engagement = engOut;
combinedOut.sessionTypes = sessionTypes;
combinedOut.collectStart = collectStart;
combinedOut.collectEnd = collectEnd;
combinedOut.d2Window = d2Window;
combinedOut.brainArea = brainArea;
combinedOut.useLog10D2 = useLog10D2;
combinedOut.enablePermutations = enablePermutations;
combinedOut.anchorMetric = anchorMetric;
combinedOut.metricsToPlot = metricsToPlot;
combinedOut.splitByEngagement = splitByEngagement;

if saveCombinedBatch
  save(combinedBatchFile, '-struct', 'combinedOut', '-v7.3');
  fprintf('\nSaved combined batch: %s\n', combinedBatchFile);
end

%% Combined plotting
if plotResults
  if isempty(areasToPlot) && ~isempty(brainArea)
    areasToPlot = {brainArea};
  end
  metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
  if ~ismember(anchorMetric, metricsToPlot)
    error('anchorMetric "%s" must be included in metricsToPlot.', anchorMetric);
  end

  fprintf('\n=== Combined plotting ===\n');
  fprintf('Anchor metric: %s\n', anchorMetric);
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
      plotConfig, anchorMetric, '', metricsToPlot, struct());
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
      sessionTypes, metricsToPlot, anchorMetric);

    for iClass = 1:numel(engagementClasses)
      engClass = engagementClasses{iClass};
      fprintf('Areas (%s): %s\n', engClass, strjoin(plotAreas, ', '));
      plot_multimetric_d2_tau_alpha_across_tasks( ...
        classViews.(engClass).ar, classViews.(engClass).av, plotAreas, ...
        sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
        plotConfig, anchorMetric, engClass, metricsToPlot, sharedByArea);
    end
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

function plot_multimetric_d2_tau_alpha_across_tasks(arPlotData, avPlotData, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    plotConfig, anchorMetric, engagementTag, metricsToPlot, sharedByArea)
% PLOT_MULTIMETRIC_D2_TAU_ALPHA_ACROSS_TASKS - Aligned d2/tau/alpha session plot

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
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);
anchorMetric = lower(char(anchorMetric));
validAnchors = {'d2', 'tau', 'alpha'};
if ~ismember(anchorMetric, validAnchors)
  error('anchorMetric must be one of: %s', strjoin(validAnchors, ', '));
end
if ~ismember(anchorMetric, metricsToPlot)
  error('anchorMetric "%s" must be included in metricsToPlot.', anchorMetric);
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
      sessionTable.tauMean, sessionTable.alphaMean, metricsToPlot);
  end

  yVals = struct();
  ySems = struct();
  yVals.d2 = apply_metric_affine_map(sessionTable.d2Mean, maps.d2);
  yVals.tau = apply_metric_affine_map(sessionTable.tauMean, maps.tau);
  yVals.alpha = apply_metric_affine_map(sessionTable.alphaMean, maps.alpha);
  ySems.d2 = abs(maps.d2.gain) * sessionTable.d2Sem;
  ySems.tau = abs(maps.tau.gain) * sessionTable.tauSem;
  ySems.alpha = abs(maps.alpha.gain) * sessionTable.alphaSem;

  fig = figure('Color', 'w', 'Name', sprintf('%s — %s', strjoin(metricsToPlot, ' '), areaName));
  position_figure_full_monitor(fig);
  axMain = axes(fig);
  hold(axMain, 'on');

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

    for m = 1:nMetrics
      metricName = metricsToPlot{m};
      faceColor = taskColor;
      if ~metricFill.(metricName)
        faceColor = 'none';
      end
      hMetric = plot_metric_errorbar_group(axMain, xPos + xOffsets(m), ...
        yVals.(metricName)(rowIdx), ySems.(metricName)(rowIdx), ...
        metricMarkers.(metricName), taskColor, faceColor, plotConfig);
      if isempty(legendHandles) || ~ismember(metricLabels.(metricName), legendLabels)
        legendHandles(end + 1) = hMetric; %#ok<AGROW>
        legendLabels{end + 1} = metricLabels.(metricName); %#ok<AGROW>
      end
    end

    for i = 1:numSessions
      xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
      xtickLabels{end + 1} = char(sessionTable.sessionLabel(rowIdx(i))); %#ok<AGROW>
    end
    xCursor = xPos(end) + 1.5;
  end

  if isfield(sharedByArea, areaKey) && isfield(sharedByArea.(areaKey), 'yLim') ...
      && numel(sharedByArea.(areaKey).yLim) == 2
    yLimPlot = sharedByArea.(areaKey).yLim;
  else
    yLimPlot = compute_display_ylim_for_metrics(yVals, metricsToPlot, anchorMetric);
  end
  if isempty(yLimPlot) || ~all(isfinite(yLimPlot))
    close(fig);
    continue;
  end
  ylim(axMain, yLimPlot);
  xlim(axMain, [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8]);
  set(axMain, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45);
  grid(axMain, 'off');
  xlabel(axMain, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
  ylabel(axMain, anchorLabel, 'FontSize', plotConfig.axisLabelFontSize, ...
    'Interpreter', ternary_metric_label_interpreter(anchorLabel));
  set(axMain, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
    'Box', 'off', 'TickDir', 'out');

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

  fprintf('  Anchor: %s | maps (display = gain * metric + offset):\n', anchorMetric);
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
  if ~isempty(brainArea)
    titleStr = sprintf('%s (anchor=%s)%s — %s [%s, %s d2 windows]', ...
      metricTitle, anchorMetric, engTitle, brainArea, collectTag, winTag);
  else
    titleStr = sprintf('%s (anchor=%s)%s — %s [%s, %s d2 windows]', ...
      metricTitle, anchorMetric, engTitle, areaName, collectTag, winTag);
  end
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_multimetric_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, anchorMetric, engagementTag, metricsToPlot);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
end

fprintf('\nAll combined figures saved to %s\n', saveDir);
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

function avIdx = find_matching_session_index(avType, sessionName, fallbackIdx)
avIdx = [];
if isfield(avType, 'sessionNames') && fallbackIdx <= numel(avType.sessionNames)
  names = get_type_session_names(avType);
  avIdx = find(strcmp(names, sessionName), 1);
end
if isempty(avIdx) && fallbackIdx <= numel(avType.sessionLabels)
  avIdx = fallbackIdx;
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

function maps = compute_anchored_metric_maps(anchorMetric, d2Vals, tauVals, alphaVals, metricsToPlot)
% COMPUTE_ANCHORED_METRIC_MAPS - Affine maps of non-anchor metrics into anchor space

if nargin < 5 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
end
metricsToPlot = normalize_metrics_to_plot(metricsToPlot);

metricVals = struct('d2', d2Vals(:), 'tau', tauVals(:), 'alpha', alphaVals(:));
anchorVals = metricVals.(anchorMetric);

maps = struct();
metricNames = {'d2', 'tau', 'alpha'};
for i = 1:numel(metricNames)
  name = metricNames{i};
  if ~ismember(name, metricsToPlot)
    maps.(name) = struct('gain', 1, 'offset', 0);
  elseif strcmp(name, anchorMetric)
    maps.(name) = struct('gain', 1, 'offset', 0);
  else
    maps.(name) = fit_metric_affine_map_to_anchor(metricVals.(name), anchorVals);
  end
end
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

function interp = ternary_metric_label_interpreter(labelText)
if contains(labelText, '_{')
  interp = 'tex';
else
  interp = 'none';
end
end

function plotBase = make_multimetric_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, anchorMetric, engagementTag, metricsToPlot)
if nargin < 7 || isempty(anchorMetric)
  anchorMetric = 'd2';
end
if nargin < 8 || isempty(engagementTag)
  engagementTag = '';
end
if nargin < 9 || isempty(metricsToPlot)
  metricsToPlot = {'d2', 'tau', 'alpha'};
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
plotBase = sprintf('%s_anchor%s', plotBase, anchorMetric);
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
    sessionTypes, metricsToPlot, anchorMetric)
% COMPUTE_SHARED_ENGAGEMENT_PLOT_SCALES - Common maps + ylim for engaged/non-engaged
%
% Goal:
%   Fit affine maps on pooled engaged+non-engaged sessions and use one y-limit
%   per area so the primary (usually d2) axis matches across the pair of plots.

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

  maps = compute_anchored_metric_maps(anchorMetric, d2All, tauAll, alphaAll, metricsToPlot);

  % Prefer shared native d2 limits when d2 is plotted (comparison across classes)
  if ismember('d2', metricsToPlot)
    d2Finite = d2All(isfinite(d2All));
    if isempty(d2Finite)
      continue;
    end
    yPad = max(0.05 * max(range(d2Finite), eps), 0.02 * max(abs(d2Finite)));
    yLimD2 = [min(d2Finite) - yPad, max(d2Finite) + yPad];

    if strcmp(anchorMetric, 'd2')
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
    yValsPool = struct( ...
      'd2', apply_metric_affine_map(d2All, maps.d2), ...
      'tau', apply_metric_affine_map(tauAll, maps.tau), ...
      'alpha', apply_metric_affine_map(alphaAll, maps.alpha));
    yLimPlot = compute_display_ylim_for_metrics(yValsPool, metricsToPlot, anchorMetric);
  end

  if isempty(yLimPlot) || ~all(isfinite(yLimPlot))
    continue;
  end
  sharedByArea.(areaKey) = struct('maps', maps, 'yLim', yLimPlot);
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
if isempty(engModOpts.d2Window)
  % Full-session one-window intent; AR clamps to loaded duration per session
  engModOpts.d2Window = 1e6;
end
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
  'tauEngaged', 'tauNonEngaged', 'alphaEngaged', 'alphaNonEngaged'};

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

    [tauEng, alphaEng] = get_engagement_area_av_exponents(avByClass.engaged, areaName);
    [tauNon, alphaNon] = get_engagement_area_av_exponents(avByClass.nonEngaged, areaName);
    typeData.tauEngaged{areaIdx}(end + 1) = tauEng;
    typeData.tauNonEngaged{areaIdx}(end + 1) = tauNon;
    typeData.alphaEngaged{areaIdx}(end + 1) = alphaEng;
    typeData.alphaNonEngaged{areaIdx}(end + 1) = alphaNon;
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

function [tauVal, alphaVal] = get_engagement_area_av_exponents(avClassResult, areaName)
tauVal = nan;
alphaVal = nan;
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
      else
        arView.byType.(typeKey).d2Mean{a} = get_eng_series(engType, 'd2NonEngagedMean', engAreaIdx);
        arView.byType.(typeKey).d2Sem{a} = get_eng_series(engType, 'd2NonEngagedSem', engAreaIdx);
        avView.byType.(typeKey).tau{a} = get_eng_series(engType, 'tauNonEngaged', engAreaIdx);
        avView.byType.(typeKey).alpha{a} = get_eng_series(engType, 'alphaNonEngaged', engAreaIdx);
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
for a = 1:numAreas
  typeData.tau{a} = [];
  typeData.alpha{a} = [];
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
