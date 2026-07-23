%%
% Criticality d2 Sliding-Window Examples (Manuscript)
%
% Runs overlapping sliding-window d2 analysis on three example sessions
% (one spontaneous, one interval, one reach) with identical AR parameters,
% then plots all d2 traces on a single axis (time in minutes).
%
% Variables (configure in this section):
%   exampleSessions  - 3x1 struct array with fields:
%                      .sessionType, .sessionName, .subjectName ('' for reach)
%   dataSource       - 'spikes' or 'lfp'
%   collectStart     - Analysis window start (seconds from session onset)
%   collectEnd       - Analysis window end (seconds); [] = session end
%                      (reach: [] omits final 180 s via resolve_reach_collect_end)
%   brainArea              - Single or merged area to plot (e.g. 'M56', 'M23M56')
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   saveFigure       - Export PNG to dropPath/criticality_manuscript
%   plotD2PopActivity - If true, scatter d2 vs mean pop activity per window (3 panels)
%
% Goal:
%   Illustrate session-wise d2 time courses across task types using the same
%   overlapping sliding-window pipeline as run_criticality_ar.m, plus optional
%   d2 vs population-activity scatters (shared y-axis across example sessions).

%% Configuration — three example sessions (one per condition)
exampleSessions(1) = struct( ...
  'sessionType', 'spontaneous', ...
  'subjectName', 'ag25290', ...
  'sessionName', 'ag112321_1', ...
  'displayLabel', 'spontaneous');
exampleSessions(2) = struct( ...
  'sessionType', 'interval', ...
  'subjectName', 'ey9166', ...
  'sessionName', 'ey9166_2026_04_03', ...
  'displayLabel', 'interval');
exampleSessions(3) = struct( ...
  'sessionType', 'reach', ...
  'subjectName', '', ...
  'sessionName', 'Y16_23-Dec-2025 16_07_49_NeuroBeh', ...
  'displayLabel', 'reach');

dataSource = 'spikes';
collectStart = 0;
collectEnd = 30 * 60;
brainArea = 'M2356';
brainAreaCombinations = default_manuscript_brain_area_combinations();
saveFigure = true;
plotD2PopActivity = true;

% Overlapping sliding-window d2 settings (aligned with run_criticality_ar.m)
slidingWindowSize = 30;   % seconds
stepSize = 0.5;           % seconds; overlap when step < window
stepSize = 2;           % seconds; overlap when step < window
useLog10D2 = true;
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;

analysisConfig = struct();
analysisConfig.slidingWindowSize = slidingWindowSize;
analysisConfig.stepSize = stepSize;
analysisConfig.binSize = 0.025;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = true;
analysisConfig.nShuffles = 20;
analysisConfig.normalizeD2 = false;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 50;
analysisConfig.nMinNeurons = 10;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
% Paths
paths = get_paths();


fprintf('\n=== Criticality d2 sliding-window examples ===\n');
fprintf('Window: %.1f s, step: %.2f s (overlapping)\n', slidingWindowSize, stepSize);
fprintf('Brain area: %s\n', brainArea);
fprintf('useLog10D2: %d\n', useLog10D2);

validate_example_sessions(exampleSessions);

% Run analysis for each example session
numExamples = numel(exampleSessions);
exampleResults = repmat(struct(), numExamples, 1);

for e = 1:numExamples
  ex = exampleSessions(e);
  fprintf('\n--- Example %d/%d [%s]: %s ---\n', e, numExamples, ex.sessionType, ex.sessionName);

  opts = build_example_load_opts(ex.sessionType, collectStart, collectEnd);
  subjectNameForLoad = ex.subjectName;
  loadArgs = build_session_load_args(ex.sessionType, ex.sessionName, opts, subjectNameForLoad);
  dataStruct = load_sliding_window_data(ex.sessionType, dataSource, loadArgs{:});

  [dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations);
  if ~areaOk
    warning('criticality_d2_sliding_examples:MissingArea', ...
      'Brain area "%s" not available for %s; skipping.', brainArea, ex.sessionName);
    continue;
  end

  results = criticality_ar_analysis(dataStruct, analysisConfig);
  if ~isempty(brainArea)
    results = filter_ar_results_to_brain_area(results, brainArea);
  end

  trace = extract_d2_time_trace(results, useLog10D2, analysisConfig.normalizeD2);
  if isempty(trace.timeMin) || isempty(trace.d2)
    warning('criticality_d2_sliding_examples:NoD2', ...
      'No d2 trace for %s (%s).', ex.sessionName, brainArea);
    continue;
  end

  exampleResults(e).example = ex;
  exampleResults(e).results = results;
  exampleResults(e).trace = trace;
  fprintf('  %d windows, %.1f min span, mean d2 = %.4f\n', ...
    numel(trace.d2), max(trace.timeMin) - min(trace.timeMin), mean(trace.d2, 'omitnan'));
end

% Plot all traces on one axis
fig = plot_d2_sliding_examples(exampleResults, brainArea, slidingWindowSize, stepSize, useLog10D2);

if plotD2PopActivity
  figPop = plot_d2_popactivity_examples(exampleResults, brainArea, slidingWindowSize, useLog10D2);
  print_d2_popactivity_example_correlations(exampleResults, useLog10D2);
end

if saveFigure
  saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  plotPath = fullfile(saveDir, sprintf('criticality_d2_sliding_examples_%s_win%.0fs_step%.1fs.png', ...
    brainArea, slidingWindowSize, stepSize));
  exportgraphics(fig, plotPath, 'Resolution', 300);
  fprintf('\nSaved figure: %s\n', plotPath);
  if plotD2PopActivity
    plotPathPop = fullfile(saveDir, sprintf('criticality_d2_sliding_popactivity_%s_win%.0fs_step%.1fs.png', ...
      brainArea, slidingWindowSize, stepSize));
    exportgraphics(figPop, plotPathPop, 'Resolution', 300);
    fprintf('Saved figure: %s\n', plotPathPop);
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

function validate_example_sessions(exampleSessions)
% VALIDATE_EXAMPLE_SESSIONS - Require spontaneous, interval, and reach examples

requiredTypes = {'spontaneous', 'interval', 'reach'};
if numel(exampleSessions) ~= 3
  error('exampleSessions must contain exactly three entries.');
end
foundTypes = {exampleSessions.sessionType};
for t = 1:numel(requiredTypes)
  if ~any(strcmpi(foundTypes, requiredTypes{t}))
    error('exampleSessions must include one %s session.', requiredTypes{t});
  end
end
for e = 1:numel(exampleSessions)
  if isempty(exampleSessions(e).sessionName)
    error('exampleSessions(%d).sessionName is required.', e);
  end
  if any(strcmpi(exampleSessions(e).sessionType, {'spontaneous', 'interval'})) ...
      && isempty(exampleSessions(e).subjectName)
    error('exampleSessions(%d).subjectName is required for %s sessions.', ...
      e, exampleSessions(e).sessionType);
  end
end
end

function opts = build_example_load_opts(~, collectStart, collectEnd)
% BUILD_EXAMPLE_LOAD_OPTS - Loader opts for example sessions
%
% Variables:
%   collectStart - Analysis window start (s)
%   collectEnd   - Analysis window end (s); [] = full session (reach omits last 180 s)
%
% Goal:
%   Apply the same collectStart/collectEnd to spontaneous, interval, and reach.

opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 150;
end

function results = filter_ar_results_to_brain_area(results, brainArea)
% FILTER_AR_RESULTS_TO_BRAIN_AREA - Keep one area in AR results struct

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaNames = normalize_results_area_list(results.areas);
areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end
cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'mrBrPermuted', ...
  'd2PermutedMean', 'd2PermutedSEM', 'popActivityWindows', 'popActivityFull'};
results.areas = {areaNames{areaIdx}};
for f = 1:length(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
if isfield(results, 'binSize') && numel(results.binSize) >= areaIdx
  results.binSize = results.binSize(areaIdx);
end
if isfield(results, 'slidingWindowSize') && numel(results.slidingWindowSize) >= areaIdx
  results.slidingWindowSize = results.slidingWindowSize(areaIdx);
end
end

function areaNames = normalize_results_area_list(areasField)
% NORMALIZE_RESULTS_AREA_LIST - Flatten results.areas to a cell of name strings

areaNames = {};
if isempty(areasField)
  return;
end
if ischar(areasField)
  areaNames = {areasField};
  return;
end
if isstring(areasField)
  areaNames = cellstr(areasField(:));
  return;
end
if ~iscell(areasField)
  areaNames = {char(areasField)};
  return;
end
rawAreas = areasField(:);
for k = 1:numel(rawAreas)
  entry = rawAreas{k};
  if iscell(entry)
    for j = 1:numel(entry)
      if ischar(entry{j}) || isstring(entry{j})
        areaNames{end+1} = char(string(entry{j})); %#ok<AGROW>
      end
    end
  elseif ischar(entry) || isstring(entry)
    areaNames{end+1} = char(string(entry)); %#ok<AGROW>
  end
end
end

function trace = extract_d2_time_trace(results, useLog10D2, normalizeD2)
% EXTRACT_D2_TIME_TRACE - Window center times (minutes) and d2 for plotting
%
% Variables:
%   results      - Output from criticality_ar_analysis (single area)
%   useLog10D2   - Apply log10 transform for display
%   normalizeD2  - Use d2Normalized instead of raw d2 when true
%
% Returns:
%   trace - Struct with .timeMin (minutes) and .d2 column vectors

trace = struct('timeMin', [], 'd2', []);
if ~isfield(results, 'areas') || isempty(results.areas)
  return;
end
areaIdx = 1;
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
  return;
end
if areaIdx > numel(results.startS) || isempty(results.startS{areaIdx})
  return;
end

d2Vec = results.d2{areaIdx}(:);
if normalizeD2 && isfield(results, 'd2Normalized') ...
    && ~isempty(results.d2Normalized{areaIdx})
  d2Vec = results.d2Normalized{areaIdx}(:);
end
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end

timeSec = results.startS{areaIdx}(:);
validMask = isfinite(timeSec) & isfinite(d2Vec);
trace.timeMin = timeSec(validMask) / 60;
trace.d2 = d2Vec(validMask);
end

function fig = plot_d2_sliding_examples(exampleResults, brainArea, slidingWindowSize, stepSize, useLog10D2)
% PLOT_D2_SLIDING_EXAMPLES - Overlay d2 traces from example sessions on one axis

fig = figure('Color', 'w', 'Position', [100 100 1100 420], ...
  'Name', 'd2 sliding-window examples');
ax = axes(fig);
hold(ax, 'on');

plottedCount = 0;
for e = 1:numel(exampleResults)
  if ~isfield(exampleResults(e), 'trace') || isempty(exampleResults(e).trace.timeMin)
    continue;
  end
  trace = exampleResults(e).trace;
  ex = exampleResults(e).example;
  lineColor = colors_for_tasks(ex.sessionType);
  if isfield(ex, 'displayLabel') && ~isempty(ex.displayLabel)
    legendLabel = ex.displayLabel;
  else
    legendLabel = ex.sessionType;
  end
  plot(ax, trace.timeMin, trace.d2, '-', 'Color', lineColor, ...
    'LineWidth', 2, 'DisplayName', legendLabel);
  plottedCount = plottedCount + 1;
end

if plottedCount == 0
  error('No d2 traces available to plot.');
end

grid(ax, 'on');
xlabel(ax, 'Time (min)');
if useLog10D2
  ylabel(ax, 'log_{10}(d2)');
else
  ylabel(ax, 'd2');
end
title(ax, sprintf('%s d2 sliding-window examples (%.0f s window, %.1f s step)', ...
  brainArea, slidingWindowSize, stepSize), 'Interpreter', 'none');
legend(ax, 'Location', 'best');
hold(ax, 'off');
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive values

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function fig = plot_d2_popactivity_examples(exampleResults, brainArea, slidingWindowSize, useLog10D2)
% PLOT_D2_POPACTIVITY_EXAMPLES - d2 vs mean pop activity for each example session
%
% Variables:
%   exampleResults - Struct array with .results and .example per session
%
% Goal:
%   Match session_d2_distributions pop-activity scatter; shared y-limits across panels.

shuffleColor = [0.55, 0.55, 0.55];

numExamples = numel(exampleResults);
fig = figure('Color', 'w', 'Position', [100 100 380 * numExamples 420], ...
  'Name', 'd2 vs population activity (examples)');
tileLayout = tiledlayout(fig, 1, numExamples, 'TileSpacing', 'compact', 'Padding', 'compact');
d2YLabel = get_d2_axis_label(useLog10D2);

yLo = inf;
yHi = -inf;
panelData = cell(1, numExamples);
for e = 1:numExamples
  if ~isfield(exampleResults(e), 'results') || isempty(exampleResults(e).results)
    panelData{e} = [];
    continue;
  end
  results = exampleResults(e).results;
  if isempty(results.areas)
    panelData{e} = [];
    continue;
  end
  [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, 1, useLog10D2);
  shufVec = get_shuffled_mean_d2_per_window(results, 1, useLog10D2);
  if ~isempty(shufVec)
    shufVec = shufVec(1:numel(d2Vec));
  end
  panelData{e} = struct('d2Vec', d2Vec, 'popVec', popVec, 'validMask', validMask, ...
    'shufVec', shufVec, 'example', exampleResults(e).example);

  yVals = d2Vec(validMask);
  if ~isempty(shufVec)
    shufMask = validMask & isfinite(shufVec);
    yVals = [yVals; shufVec(shufMask)]; %#ok<AGROW>
  end
  yVals = yVals(isfinite(yVals));
  if ~isempty(yVals)
    yLo = min(yLo, min(yVals));
    yHi = max(yHi, max(yVals));
  end
end

if isfinite(yLo) && isfinite(yHi) && yHi > yLo
  yPad = 0.05 * (yHi - yLo);
  sharedYLim = [yLo - yPad, yHi + yPad];
else
  sharedYLim = [];
end

for e = 1:numExamples
  ax = nexttile(tileLayout);
  hold(ax, 'on');

  if isempty(panelData{e})
    title(ax, sprintf('Example %d (no data)', e));
    if ~isempty(sharedYLim)
      ylim(ax, sharedYLim);
    end
    continue;
  end

  pd = panelData{e};
  ex = pd.example;
  dataColor = colors_for_tasks(ex.sessionType);
  if isfield(ex, 'displayLabel') && ~isempty(ex.displayLabel)
    panelTitle = ex.displayLabel;
  else
    panelTitle = ex.sessionType;
  end

  if ~any(pd.validMask)
    title(ax, sprintf('%s (no data)', panelTitle), 'Interpreter', 'none');
    if ~isempty(sharedYLim)
      ylim(ax, sharedYLim);
    end
    continue;
  end

  scatter_open_translucent(ax, pd.popVec(pd.validMask), pd.d2Vec(pd.validMask), 24, ...
    dataColor, 'Data');

  rShuf = nan;
  if ~isempty(pd.shufVec)
    shufMask = pd.validMask & isfinite(pd.shufVec);
    if any(shufMask)
      scatter_open_translucent(ax, pd.popVec(shufMask), pd.shufVec(shufMask), 24, ...
        shuffleColor, 'Shuffled mean');
      rShuf = pearson_r(pd.popVec(shufMask), pd.shufVec(shufMask));
    end
  end

  rData = pearson_r(pd.popVec(pd.validMask), pd.d2Vec(pd.validMask));
  xlabel(ax, 'Mean pop activity (spikes/bin)');
  if e == 1
    ylabel(ax, d2YLabel);
  end
  title(ax, sprintf('%s | r_{data}=%.3f, r_{shuf}=%.3f, n=%d', ...
    panelTitle, rData, rShuf, sum(pd.validMask)), 'Interpreter', 'none');
  if ~isempty(sharedYLim)
    ylim(ax, sharedYLim);
  end
  legend(ax, 'Location', 'best');
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayout, sprintf('%s d2 vs mean population activity | %.0fs windows', ...
  brainArea, slidingWindowSize), 'FontSize', 12, 'Interpreter', 'none');
end

function print_d2_popactivity_example_correlations(exampleResults, useLog10D2)
% PRINT_D2_POPACTIVITY_EXAMPLE_CORRELATIONS - Command-window summary per example

fprintf('\n=== d2 vs mean pop activity (example sessions) ===\n');
for e = 1:numel(exampleResults)
  if ~isfield(exampleResults(e), 'results') || isempty(exampleResults(e).results) ...
      || isempty(exampleResults(e).results.areas)
    continue;
  end
  ex = exampleResults(e).example;
  if isfield(ex, 'displayLabel') && ~isempty(ex.displayLabel)
    label = ex.displayLabel;
  else
    label = ex.sessionType;
  end
  results = exampleResults(e).results;
  [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, 1, useLog10D2);
  shufVec = get_shuffled_mean_d2_per_window(results, 1, useLog10D2);
  if ~any(validMask)
    fprintf('  %s: no data\n', label);
    continue;
  end
  rData = pearson_r(popVec(validMask), d2Vec(validMask));
  rShuf = nan;
  if ~isempty(shufVec)
    shufVec = shufVec(1:numel(d2Vec));
    shufMask = validMask & isfinite(shufVec);
    if any(shufMask)
      rShuf = pearson_r(popVec(shufMask), shufVec(shufMask));
    end
  end
  fprintf('  %s: r(data)=%.3f, r(shuffled)=%.3f, n=%d\n', label, rData, rShuf, sum(validMask));
end
end

function d2Vec = get_aligned_d2_vector(results, areaIdx, useLog10D2)
% GET_ALIGNED_D2_VECTOR - d2 per window for one area (optional log10)

d2Vec = [];
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
  return;
end
d2Vec = results.d2{areaIdx}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
end

function [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, areaIdx, useLog10D2)
% GET_ALIGNED_D2_POPACTIVITY - Window-aligned d2 and pop activity vectors

d2Vec = [];
popVec = [];
validMask = false(0, 1);

d2Vec = get_aligned_d2_vector(results, areaIdx, useLog10D2);
if isempty(d2Vec)
  return;
end
if ~isfield(results, 'popActivityWindows') || areaIdx > numel(results.popActivityWindows) ...
    || isempty(results.popActivityWindows{areaIdx})
  return;
end

popVec = results.popActivityWindows{areaIdx}(:);
nWindows = min(numel(d2Vec), numel(popVec));
d2Vec = d2Vec(1:nWindows);
popVec = popVec(1:nWindows);
validMask = isfinite(d2Vec) & isfinite(popVec);
end

function scatter_open_translucent(ax, x, y, markerSize, faceColor, displayName)
% SCATTER_OPEN_TRANSLUCENT - Open circles with translucent fill

if nargin < 6
  displayName = '';
end

scatter(ax, x, y, markerSize, ...
  'Marker', 'o', ...
  'MarkerFaceColor', faceColor, ...
  'MarkerEdgeColor', faceColor, ...
  'MarkerFaceAlpha', 0.35, ...
  'LineWidth', 1, ...
  'DisplayName', displayName);
end

function shufVec = get_shuffled_mean_d2_per_window(results, areaIdx, useLog10D2)
% GET_SHUFFLED_MEAN_D2_PER_WINDOW - Mean shuffled d2 per window (subsampling-aware)

shufVec = get_per_window_shuffle_mean_d2(results, areaIdx, useLog10D2);
end

function rVal = pearson_r(x, y)
% PEARSON_R - Pearson correlation or NaN when undefined

rVal = nan;
if numel(x) < 2 || numel(y) < 2
  return;
end
cMat = corrcoef(x(:), y(:));
rVal = cMat(1, 2);
end

function yLabelText = get_d2_axis_label(useLog10D2)
% GET_D2_AXIS_LABEL - Y-axis label for d2 scatter plots

if useLog10D2
  yLabelText = 'log_{10}(d2)';
else
  yLabelText = 'd2';
end
end
