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
%
% Goal:
%   One session-grouped plot per brain area with d2, tau, and alpha per session.
%   Plot is anchored on the chosen metric's native y-limits; the other metrics
%   are affine-mapped into that space to minimize within-session differences.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
collectStart = 0;
collectEnd = 60 * 60;
% collectEnd = [];  % [] = full session
d2Window = 30;
% One d2 estimate for the full collect window ([] when collectEnd is [])
% d2Window = collectEnd;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};

runArBatch = true;
runAvBatch = true;
loadSavedResults = true;
plotResults = true;
saveCombinedBatch = true;
enablePermutations = false;
anchorMetric = 'd2';  % 'd2', 'tau', or 'alpha'

useLog10D2 = true;
useSubsampling = true;
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

%% AR batch (d2)
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

%% AV batch (tau, alpha)
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

combinedOut = struct();
combinedOut.ar = arOut;
combinedOut.av = avOut;
combinedOut.sessionTypes = sessionTypes;
combinedOut.collectStart = collectStart;
combinedOut.collectEnd = collectEnd;
combinedOut.d2Window = d2Window;
combinedOut.brainArea = brainArea;
combinedOut.useLog10D2 = useLog10D2;
combinedOut.enablePermutations = enablePermutations;
combinedOut.anchorMetric = anchorMetric;

if saveCombinedBatch
  save(combinedBatchFile, '-struct', 'combinedOut', '-v7.3');
  fprintf('\nSaved combined batch: %s\n', combinedBatchFile);
end

%% Combined plotting
if plotResults
  if isempty(areasToPlot) && ~isempty(brainArea)
    areasToPlot = {brainArea};
  end
  plotAreas = intersect(arOut.plotData.areas, avOut.plotData.areas, 'stable');
  if ~isempty(areasToPlot)
    plotAreas = intersect(plotAreas, areasToPlot, 'stable');
  end
  if isempty(plotAreas)
    error('No common brain areas between AR and AV plotData.');
  end

  fprintf('\n=== Combined plotting ===\n');
  fprintf('Areas: %s\n', strjoin(plotAreas, ', '));
  fprintf('Anchor metric: %s\n', anchorMetric);
  plot_multimetric_d2_tau_alpha_across_tasks(arOut.plotData, avOut.plotData, plotAreas, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    plotConfig, anchorMetric);
end

fprintf('\n=== Done ===\n');

%% Local functions

function plot_multimetric_d2_tau_alpha_across_tasks(arPlotData, avPlotData, areasToPlot, ...
    sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2, ...
    plotConfig, anchorMetric)
% PLOT_MULTIMETRIC_D2_TAU_ALPHA_ACROSS_TASKS - Aligned d2/tau/alpha session plot

if nargin < 11 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 12 || isempty(anchorMetric)
  anchorMetric = 'd2';
end
anchorMetric = lower(char(anchorMetric));
validAnchors = {'d2', 'tau', 'alpha'};
if ~ismember(anchorMetric, validAnchors)
  error('anchorMetric must be one of: %s', strjoin(validAnchors, ', '));
end

if useLog10D2
  d2Label = 'log_{10}(d2)';
else
  d2Label = 'd2';
end
metricLabels = struct('d2', d2Label, 'tau', 'tau', 'alpha', 'alpha');
anchorLabel = metricLabels.(anchorMetric);

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
    areaIdxAr, areaIdxAv);
  if isempty(sessionTable)
    fprintf('Skipping %s: no aligned sessions with d2, tau, and alpha.\n', areaName);
    continue;
  end

  % Anchor display on chosen metric; map others into that space
  maps = compute_anchored_metric_maps(anchorMetric, sessionTable.d2Mean, ...
    sessionTable.tauMean, sessionTable.alphaMean);
  yD2 = apply_metric_affine_map(sessionTable.d2Mean, maps.d2);
  yTau = apply_metric_affine_map(sessionTable.tauMean, maps.tau);
  yAlpha = apply_metric_affine_map(sessionTable.alphaMean, maps.alpha);
  yD2Sem = abs(maps.d2.gain) * sessionTable.d2Sem;
  yTauSem = abs(maps.tau.gain) * sessionTable.tauSem;
  yAlphaSem = abs(maps.alpha.gain) * sessionTable.alphaSem;

  fig = figure('Color', 'w', 'Name', sprintf('d2 tau alpha — %s', areaName));
  position_figure_full_monitor(fig);
  axMain = axes(fig);
  hold(axMain, 'on');

  xOffsets = [-0.12, 0, 0.12];
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

    hD2 = plot_metric_errorbar_group(axMain, xPos + xOffsets(1), yD2(rowIdx), yD2Sem(rowIdx), ...
      'o', taskColor, taskColor, plotConfig);
    hTau = plot_metric_errorbar_group(axMain, xPos + xOffsets(2), yTau(rowIdx), yTauSem(rowIdx), ...
      's', taskColor, 'none', plotConfig);
    hAlpha = plot_metric_errorbar_group(axMain, xPos + xOffsets(3), yAlpha(rowIdx), yAlphaSem(rowIdx), ...
      'd', taskColor, 'none', plotConfig);

    if isempty(legendHandles)
      legendHandles = [hD2, hTau, hAlpha];
      legendLabels = {d2Label, 'tau', 'alpha'};
    end

    for i = 1:numSessions
      xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
      xtickLabels{end + 1} = char(sessionTable.sessionLabel(rowIdx(i))); %#ok<AGROW>
    end
    xCursor = xPos(end) + 1.5;
  end

  % Y-limits follow anchor metric variation; allow slight expansion for mapped points
  switch anchorMetric
    case 'd2'
      yAnchor = yD2;
      yOthers = [yTau(:); yAlpha(:)];
    case 'tau'
      yAnchor = yTau;
      yOthers = [yD2(:); yAlpha(:)];
    otherwise
      yAnchor = yAlpha;
      yOthers = [yD2(:); yTau(:)];
  end
  anchorFinite = yAnchor(isfinite(yAnchor));
  if isempty(anchorFinite)
    close(fig);
    continue;
  end
  yPad = max(0.05 * max(range(anchorFinite), eps), 0.02 * max(abs(anchorFinite)));
  yLimPlot = [min(anchorFinite) - yPad, max(anchorFinite) + yPad];
  yOthers = yOthers(isfinite(yOthers));
  if ~isempty(yOthers)
    yLimPlot(1) = min(yLimPlot(1), min(yOthers));
    yLimPlot(2) = max(yLimPlot(2), max(yOthers));
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

  % Non-anchor metrics get secondary native-scale axes
  rightOffset = 1.0;
  for metricName = validAnchors
    name = metricName{1};
    if strcmp(name, anchorMetric)
      continue;
    end
    add_affine_metric_yaxis(axMain, maps.(name), 'right', metricLabels.(name), ...
      plotConfig, rightOffset);
    rightOffset = rightOffset + 0.1;
  end

  if ~isempty(legendHandles)
    legend(axMain, legendHandles, legendLabels, 'Location', 'best', ...
      'FontSize', plotConfig.legendFontSize);
  end
  hold(axMain, 'off');

  fprintf('  Anchor: %s | maps (display = gain * metric + offset):\n', anchorMetric);
  for metricName = validAnchors
    name = metricName{1};
    fprintf('    %s: gain=%.4g, offset=%.4g\n', name, maps.(name).gain, maps.(name).offset);
  end

  collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
  if isempty(d2Window)
    winTag = 'full';
  else
    winTag = sprintf('%.0fs', d2Window);
  end
  if ~isempty(brainArea)
    titleStr = sprintf('d2, tau, alpha (anchor=%s) — %s [%s, %s d2 windows]', ...
      anchorMetric, brainArea, collectTag, winTag);
  else
    titleStr = sprintf('d2, tau, alpha (anchor=%s) — %s [%s, %s d2 windows]', ...
      anchorMetric, areaName, collectTag, winTag);
  end
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

  plotBase = make_multimetric_plot_basename(areaName, brainArea, d2Window, ...
    collectStart, collectEnd, useLog10D2, anchorMetric);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
end

fprintf('\nAll combined figures saved to %s\n', saveDir);
end

function sessionTable = build_multimetric_session_table(arPlotData, avPlotData, sessionTypes, ...
    areaIdxAr, areaIdxAv)
% BUILD_MULTIMETRIC_SESSION_TABLE - Align d2/tau/alpha per session across pipelines

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
  if ~isfield(arPlotData.byType, typeKey) || ~isfield(avPlotData.byType, typeKey)
    continue;
  end
  arType = arPlotData.byType.(typeKey);
  avType = avPlotData.byType.(typeKey);
  if areaIdxAr > numel(arType.d2Mean) || areaIdxAv > numel(avType.tau)
    continue;
  end

  arNames = get_type_session_names(arType);
  numAr = numel(arType.d2Mean{areaIdxAr});
  for i = 1:numAr
    sessionName = arNames{i};
    avIdx = find_matching_session_index(avType, sessionName, i);
    if isempty(avIdx)
      continue;
    end

    d2Val = arType.d2Mean{areaIdxAr}(i);
    tauVal = avType.tau{areaIdxAv}(avIdx);
    alphaVal = avType.alpha{areaIdxAv}(avIdx);
    if ~all(isfinite([d2Val, tauVal, alphaVal]))
      continue;
    end

    sessionTypeCol{end + 1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = sessionName; %#ok<AGROW>
    sessionLabelCol{end + 1, 1} = get_session_display_label(arType, i, sessionType); %#ok<AGROW>
    d2MeanCol(end + 1, 1) = d2Val; %#ok<AGROW>
    d2SemCol(end + 1, 1) = get_metric_series_value(arType.d2Sem{areaIdxAr}, i); %#ok<AGROW>
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
end

function val = get_metric_series_value(metricSeries, idx)
val = nan;
if isempty(metricSeries) || idx > numel(metricSeries)
  return;
end
val = metricSeries(idx);
end

function maps = compute_anchored_metric_maps(anchorMetric, d2Vals, tauVals, alphaVals)
% COMPUTE_ANCHORED_METRIC_MAPS - Affine maps of non-anchor metrics into anchor space
%
% Variables:
%   anchorMetric - 'd2', 'tau', or 'alpha'
%   d2Vals, tauVals, alphaVals - Per-session metric vectors
%
% Goal:
%   Keep the anchor metric in native units. Map each other metric with
%   display = gain * metric + offset to minimize squared distance to the anchor.

metricVals = struct('d2', d2Vals(:), 'tau', tauVals(:), 'alpha', alphaVals(:));
anchorVals = metricVals.(anchorMetric);

maps = struct();
metricNames = {'d2', 'tau', 'alpha'};
for i = 1:numel(metricNames)
  name = metricNames{i};
  if strcmp(name, anchorMetric)
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
    collectStart, collectEnd, useLog10D2, anchorMetric)
if nargin < 7 || isempty(anchorMetric)
  anchorMetric = 'd2';
end
collectTag = format_multimetric_collect_tag(collectStart, collectEnd);
if isempty(d2Window)
  winTag = 'full';
else
  winTag = sprintf('%.0fs', d2Window);
end
if ~isempty(brainArea)
  plotBase = sprintf('criticality_multiple_metrics_d2_tau_alpha_%s_win%s_%s', ...
    brainArea, winTag, collectTag);
else
  plotBase = sprintf('criticality_multiple_metrics_d2_tau_alpha_%s_win%s_%s', ...
    areaName, winTag, collectTag);
end
plotBase = sprintf('%s_anchor%s', plotBase, anchorMetric);
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

function position_figure_full_monitor(fig)
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
  targetPos = monitorPositions(end, :);
else
  targetPos = monitorPositions(1, :);
end
set(fig, 'Units', 'pixels', 'Position', targetPos);
end
