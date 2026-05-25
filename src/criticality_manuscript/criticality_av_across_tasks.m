%%
% Criticality Avalanche Analysis Across Task Types (Manuscript)
%
% Runs avalanche criticality on one fixed-duration window per session (not
% sliding windows), batches across session types, and plots metrics grouped
% by sessionType.
%
% Variables (configure in this section):
%   sessionTypes  - Cell array of session types to include
%   dataSource    - 'spikes' or 'lfp'
%   collectStart  - Window start (seconds from session onset), default 0
%   collectEnd    - Window end (seconds), default 40*60
%   brainArea     - Single area to analyze (e.g. 'M23', 'M1'); '' = all areas
%   areasToPlot   - Area names to plot; {} uses brainArea if set, else all areas
%   runBatch      - If true, run criticality_av_analysis per session
%   plotResults   - If true, create summary figures after batch
%
% Goal:
%   Compare avalanche metrics (dcc, kappa, decades, tau, alpha, paramSD)
%   across spontaneous, reach, interval, and other session types using
%   load_session_data and a common analysis window per session.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
dataSource = 'spikes';

collectStart = 0;              % seconds
collectEnd = 45 * 60;           % seconds (40 minutes)
windowDurationSec = collectEnd - collectStart;

brainArea = 'M56';             % one area per run; '' = analyze/plot all areas
areasToPlot = {};              % optional override, e.g. {'M23'}; {} -> brainArea if set
runBatch = true;
plotResults = true;

% Avalanche analysis settings (single full collect window per session)
analysisConfig = struct();
analysisConfig.slidingWindowSize = windowDurationSec;
analysisConfig.avStepSize = windowDurationSec;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.binSize = 0.05;   % seconds; same bin size for all areas
analysisConfig.analyzeDcc = true;
analysisConfig.analyzeKappa = true;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 5;
analysisConfig.enablePermutations = true;
analysisConfig.nShuffles = 20;
analysisConfig.makePlots = false;
analysisConfig.saveData = true;
analysisConfig.thresholdFlag = 1;
analysisConfig.thresholdPct = 1;
analysisConfig.nMinNeurons = 20;
analysisConfig.normalizeMetrics = true;
analysisConfig.includeM2356 = false;
if ~isempty(brainArea) && strcmpi(brainArea, 'M2356')
  analysisConfig.includeM2356 = true;
end

opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;

%% Paths
paths = get_paths();
scriptDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(scriptDir, '..');
addpath(srcPath);
addpath(fullfile(srcPath, 'reach_task'));
addpath(fullfile(srcPath, 'schall'));
addpath(fullfile(srcPath, 'spontaneous'));
addpath(fullfile(srcPath, 'interval_timing_task'));
addpath(fullfile(srcPath, 'criticality', 'scripts'));
addpath(fullfile(srcPath, 'criticality', 'analyses'));
addpath(fullfile(srcPath, 'session_prep', 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'sliding_window_prep', 'utils'));

fprintf('\n=== Criticality Avalanche Across Task Types ===\n');
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, windowDurationSec / 60);
fprintf('Session types: %s\n', strjoin(sessionTypes, ', '));
if ~isempty(brainArea)
  fprintf('Brain area: %s (single-area analysis)\n', brainArea);
else
  fprintf('Brain area: all areas in each session\n');
end

%% Build flat session table
sessionTable = build_session_table(sessionTypes);
numSessions = size(sessionTable, 1);
fprintf('Total sessions: %d\n', numSessions);

if numSessions == 0
  error('No sessions found for the requested session types.');
end

%% Batch avalanche analysis
batchResults = repmat(struct(), numSessions, 1);

if runBatch
  fprintf('\n=== Running avalanche analysis (single window per session) ===\n');
  for s = 1:numSessions
    sessionType = sessionTable.sessionType{s};
    sessionName = sessionTable.sessionName{s};
    subjectName = sessionTable.subjectName{s};

    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('Session %d/%d [%s]: %s\n', s, numSessions, sessionType, sessionName);
    if ~isempty(subjectName)
      fprintf('  subjectName: %s\n', subjectName);
    end

    batchResults(s).sessionType = sessionType;
    batchResults(s).sessionName = sessionName;
    batchResults(s).subjectName = subjectName;
    batchResults(s).label = sessionTable.label{s};
    batchResults(s).success = false;
    batchResults(s).results = [];

    try
      subjectNameForLoad = subjectName;
      loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
      dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

      [dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea);
      if ~areaOk
        fprintf('  Brain area "%s" not available in this session; skipping.\n', brainArea);
        continue;
      end

      config = analysisConfig;
      avResults = criticality_av_analysis(dataStruct, config);

      if ~isempty(brainArea)
        avResults = filter_av_results_to_brain_area(avResults, brainArea);
        if isempty(avResults.areas)
          fprintf('  No results for brain area "%s"; skipping.\n', brainArea);
          continue;
        end
      end

      batchResults(s).success = true;
      batchResults(s).results = avResults;
      fprintf('  Analysis completed.\n');
    catch ME
      fprintf('  Error: %s\n', ME.message);
      for st = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
      end
    end
  end
else
  error('runBatch must be true; loading precomputed results is not implemented in this script.');
end

%% Plotting: Aggregate metrics for plotting
plotData = aggregate_av_metrics(batchResults, sessionTypes);

if isempty(plotData.areas)
  error('No avalanche metrics extracted. Check that batch analyses succeeded.');
end

if isempty(areasToPlot) && ~isempty(brainArea)
  areasToPlot = {brainArea};
end
commonAreas = plotData.areas;
if ~isempty(areasToPlot)
  commonAreas = intersect(commonAreas, areasToPlot, 'stable');
  if isempty(commonAreas)
    error('None of areasToPlot are present in the aggregated results.');
  end
elseif ~isempty(brainArea)
  commonAreas = intersect(commonAreas, {brainArea}, 'stable');
  if isempty(commonAreas)
    error('No results for brainArea "%s". Check that sessions include this area.', brainArea);
  end
end

fprintf('\n=== Areas for plotting ===\n');
fprintf('  %s\n', strjoin(commonAreas, ', '));

% Summary figures grouped by session type
if plotResults
  plot_av_across_tasks(plotData, commonAreas, sessionTypes, collectStart, collectEnd, paths, brainArea);
end

fprintf('\n=== Done ===\n');

%% Local functions

function sessionTable = build_session_table(sessionTypes)
% BUILD_SESSION_TABLE - Flatten session lists from each session type

sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
labelCol = {};

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  entries = get_sessions_for_type(sessionType);
  nEntries = numel(entries);

  for i = 1:nEntries
    sessionTypeCol{end+1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end+1, 1} = entries(i).sessionName; %#ok<AGROW>
    if isfield(entries, 'subjectName')
      subjectNameCol{end+1, 1} = entries(i).subjectName; %#ok<AGROW>
    else
      subjectNameCol{end+1, 1} = ''; %#ok<AGROW>
    end
    labelCol{end+1, 1} = make_session_label(sessionType, entries(i)); %#ok<AGROW>
  end
end

sessionTable = table(sessionTypeCol, sessionNameCol, subjectNameCol, labelCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'subjectName', 'label'});
end

function entries = get_sessions_for_type(sessionType)
% GET_SESSIONS_FOR_TYPE - Struct array with subjectName and sessionName

switch lower(sessionType)
  case 'spontaneous'
    entries = spontaneous_session_list();
  case 'interval'
    entries = interval_session_list();
  case 'reach'
    names = reach_session_list();
    entries = struct('subjectName', {}, 'sessionName', {});
    for i = 1:length(names)
      entries(i).subjectName = '';
      entries(i).sessionName = names{i};
    end
  case 'schall'
    names = schall_session_list();
    entries = struct('subjectName', {}, 'sessionName', {});
    for i = 1:length(names)
      parts = strsplit(names{i}, '/');
      if numel(parts) >= 2
        entries(i).subjectName = parts{1};
        entries(i).sessionName = parts{2};
      else
        entries(i).subjectName = '';
        entries(i).sessionName = names{i};
      end
    end
  otherwise
    error('Unknown sessionType: %s', sessionType);
end

if ~isstruct(entries) || ~isfield(entries, 'sessionName')
  error('Session list for %s must return a struct array with sessionName.', sessionType);
end
end

function label = make_session_label(~, entry)
% MAKE_SESSION_LABEL - Short display label for plots
label = entry.sessionName;
end

function [dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea)
% APPLY_BRAIN_AREA_SELECTION - Restrict analysis to one brain area
%
% Variables:
%   dataStruct - From load_session_data
%   brainArea  - Area name (char); empty = no restriction
%
% Goal:
%   Set dataStruct.areasToTest when brainArea is already in dataStruct.areas.
%   M2356 is accepted when M23 and M56 exist (merged during analysis).
%
% Returns:
%   areaOk - false if brainArea was requested but cannot be analyzed

areaOk = true;
if isempty(brainArea)
  return;
end

if strcmpi(brainArea, 'M2356')
  areaOk = any(strcmp(dataStruct.areas, 'M23')) && any(strcmp(dataStruct.areas, 'M56'));
  if ~areaOk
    return;
  end
  fprintf('  Brain area M2356 (requires includeM2356 during analysis)\n');
  return;
end

areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  areaOk = false;
  return;
end

dataStruct.areasToTest = areaIdx;
fprintf('  Restricting analysis to area: %s\n', brainArea);
end

function results = filter_av_results_to_brain_area(results, brainArea)
% FILTER_AV_RESULTS_TO_BRAIN_AREA - Keep one area in avalanche results struct

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end

areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end

cellFields = {'dcc', 'dccNormalized', 'kappa', 'kappaNormalized', ...
  'decades', 'decadesNormalized', 'tau', 'tauNormalized', ...
  'alpha', 'alphaNormalized', 'paramSD', 'paramSDNormalized', 'startS', ...
  'dccPermuted', 'kappaPermuted', 'decadesPermuted', 'tauPermuted', ...
  'alphaPermuted', 'paramSDPermuted', ...
  'dccPermutedMean', 'kappaPermutedMean', 'decadesPermutedMean', ...
  'tauPermutedMean', 'alphaPermutedMean', 'paramSDPermutedMean'};

results.areas = results.areas(areaIdx);
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

function plotData = aggregate_av_metrics(batchResults, sessionTypes)
% AGGREGATE_AV_METRICS - Collect single-window metrics per session and area

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();

metricFields = get_av_metric_storage_fields();

for s = 1:length(batchResults)
  if ~batchResults(s).success || isempty(batchResults(s).results)
    continue;
  end

  results = batchResults(s).results;
  sessionType = batchResults(s).sessionType;

  if isempty(plotData.areas)
    for t = 1:length(sessionTypes)
      st = sessionTypes{t};
      plotData.byType.(matlab.lang.makeValidName(st)) = init_type_metrics(metricFields, 0);
    end
  end

  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_type_metrics(metricFields, length(plotData.areas));
  end
  typeData = plotData.byType.(typeKey);

  for a = 1:length(results.areas)
    areaName = results.areas{a};
    areaIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaIdx)
      plotData.areas{end+1} = areaName;
      areaIdx = length(plotData.areas);
      plotData = extend_plot_data_areas(plotData, sessionTypes, metricFields, areaIdx);
      typeData = plotData.byType.(typeKey);
    end

    for m = 1:length(metricFields)
      fieldName = metricFields{m};
      if ~isfield(results, fieldName) || a > length(results.(fieldName))
        val = nan;
      else
        val = extract_single_window_value(results.(fieldName){a});
      end
      typeData.(fieldName){areaIdx} = [typeData.(fieldName){areaIdx}, val];
    end
  end

  typeData.sessionLabels{end+1} = batchResults(s).label;
  typeData.sessionNames{end+1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
end
end

function plotData = extend_plot_data_areas(plotData, sessionTypes, metricFields, newAreaIdx)
% EXTEND_PLOT_DATA_AREAS - Grow metric storage when a new area appears

typeNames = fieldnames(plotData.byType);
for i = 1:length(typeNames)
  typeData = plotData.byType.(typeNames{i});
  for m = 1:length(metricFields)
    fieldName = metricFields{m};
    while length(typeData.(fieldName)) < newAreaIdx
      typeData.(fieldName){end+1} = [];
    end
  end
  plotData.byType.(typeNames{i}) = typeData;
end
end

function typeData = init_type_metrics(metricFields, numAreas)
% INIT_TYPE_METRICS - Empty storage per area for one session type

typeData = struct();
typeData.sessionLabels = {};
typeData.sessionNames = {};
for m = 1:length(metricFields)
  typeData.(metricFields{m}) = cell(1, numAreas);
  for a = 1:numAreas
    typeData.(metricFields{m}){a} = [];
  end
end
end

function val = extract_single_window_value(metricVec)
% EXTRACT_SINGLE_WINDOW_VALUE - One scalar from a single-window analysis

if isempty(metricVec)
  val = nan;
  return;
end
metricVec = metricVec(:);
valid = metricVec(~isnan(metricVec));
if isempty(valid)
  val = nan;
elseif numel(valid) == 1
  val = valid(1);
else
  val = mean(valid);
end
end

function plot_av_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, collectEnd, paths, brainArea)
% PLOT_AV_ACROSS_TASKS - 2x4 bar plots grouped by session type

if nargin < 8
  brainArea = '';
end

numAreas = length(areasToPlot);
numSessionTypes = length(sessionTypes);
figRows = 2;
figCols = 4;
metricPlotSpec = get_av_metric_plot_spec();
typeColors = lines(max(numSessionTypes, 3));
shuffleBarColor = [0.55, 0.55, 0.55];

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:numAreas
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx) || ~area_has_plot_data(plotData, sessionTypes, areaIdx, metricPlotSpec)
    continue;
  end

  fig = figure(4000 + a);
  clf(fig);
  position_figure_full_monitor(fig);

  for m = 1:size(metricPlotSpec, 1)
    panelRow = metricPlotSpec{m, 1};
    panelCol = metricPlotSpec{m, 2};
    fieldName = metricPlotSpec{m, 3};
    yLabelText = metricPlotSpec{m, 4};
    shuffledFieldName = metricPlotSpec{m, 5};
    plotShuffledMean = metricPlotSpec{m, 6};

    ax = subplot(figRows, figCols, (panelRow - 1) * figCols + panelCol);
    hold(ax, 'on');

    xCursor = 0;
    xticksCenters = [];
    xtickLabels = {};
    yValuesForLim = [];

    for t = 1:numSessionTypes
      sessionType = sessionTypes{t};
      typeKey = matlab.lang.makeValidName(sessionType);
      if ~isfield(plotData.byType, typeKey)
        continue;
      end
      typeData = plotData.byType.(typeKey);
      if ~isfield(typeData, fieldName) || areaIdx > length(typeData.(fieldName))
        continue;
      end

      values = typeData.(fieldName){areaIdx};
      values = values(:)';
      numBars = numel(values);
      if numBars == 0
        continue;
      end

      shuffleValues = nan(size(values));
      if plotShuffledMean && ~isempty(shuffledFieldName) ...
          && isfield(typeData, shuffledFieldName) ...
          && areaIdx <= length(typeData.(shuffledFieldName))
        shuffleValues = typeData.(shuffledFieldName){areaIdx};
        shuffleValues = shuffleValues(:)';
        if numel(shuffleValues) ~= numBars
          shuffleValues = nan(size(values));
        end
      end

      xPos = xCursor + (1:numBars);
      if plotShuffledMean && any(isfinite(shuffleValues))
        barData = [values(:), shuffleValues(:)];
        bh = bar(ax, xPos, barData, 'grouped');
        bh(1).FaceColor = typeColors(t, :);
        bh(1).EdgeColor = 'k';
        bh(1).LineWidth = 0.5;
        bh(2).FaceColor = shuffleBarColor;
        bh(2).EdgeColor = [0.2, 0.2, 0.2];
        bh(2).LineWidth = 0.5;
        yValuesForLim = [yValuesForLim, values(isfinite(values)), shuffleValues(isfinite(shuffleValues))]; %#ok<AGROW>
      else
        bar(ax, xPos, values, 0.75, ...
          'FaceColor', typeColors(t, :), 'EdgeColor', 'k', 'LineWidth', 0.5);
        yValuesForLim = [yValuesForLim, values(isfinite(values))]; %#ok<AGROW>
      end

      groupCenter = mean(xPos);
      xticksCenters(end+1) = groupCenter; %#ok<AGROW>
      xtickLabels{end+1} = sessionType; %#ok<AGROW>

      validVals = values(isfinite(values));
      if ~isempty(validVals)
        yline(ax, mean(validVals), '--', 'Color', typeColors(t, :), 'LineWidth', 1.5);
        yValuesForLim = [yValuesForLim, mean(validVals)]; %#ok<AGROW>
      end

      xCursor = xPos(end) + 1.5;
    end

    if ~isempty(xticksCenters)
      xticks(ax, xticksCenters);
      xticklabels(ax, xtickLabels);
    end
    ylabel(ax, yLabelText);
    title(ax, sprintf('%s — %s', areaName, yLabelText));
    apply_buffered_ylim(ax, yValuesForLim);
    grid(ax, 'on');
    hold(ax, 'off');

    if plotShuffledMean && panelRow == 1 && panelCol == 1
      legend(ax, {'session', 'shuffled mean'}, 'Location', 'best');
    end
  end

  if ~isempty(brainArea)
    titleStr = sprintf('Avalanche criticality — %s [%.0f–%.0f s]', brainArea, collectStart, collectEnd);
  else
    titleStr = sprintf('Avalanche criticality — %s [%.0f–%.0f s]', areaName, collectStart, collectEnd);
  end
  sgtitle(fig, titleStr, 'FontWeight', 'bold');

  if ~isempty(brainArea)
    plotBase = sprintf('criticality_av_across_tasks_%s_%.0f-%.0fs', brainArea, collectStart, collectEnd);
  else
    plotBase = sprintf('criticality_av_across_tasks_%s_%.0f-%.0fs', areaName, collectStart, collectEnd);
  end
  if numAreas > 1
    plotBase = sprintf('%s_area%s', plotBase, areaName);
  end

  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
end

fprintf('\nAll figures saved to %s\n', saveDir);
end

function metricFields = get_av_metric_storage_fields()
% GET_AV_METRIC_STORAGE_FIELDS - Fields aggregated for plotting

metricFields = {'dcc', 'dccNormalized', 'kappa', 'kappaNormalized', ...
  'dccPermutedMean', 'kappaPermutedMean', 'decadesPermutedMean', ...
  'tauPermutedMean', 'alphaPermutedMean', 'paramSDPermutedMean', ...
  'tau', 'alpha', 'paramSD', 'decades'};
end

function metricPlotSpec = get_av_metric_plot_spec()
% GET_AV_METRIC_PLOT_SPEC - row, col, field, ylabel, shuffledField, plotShuffled
%
% Top row: dcc and kappa (raw + normalized). Bottom row: decades and exponents.
% Shuffled means are shown only on raw panels (same axes as session values).

metricPlotSpec = {
  1, 1, 'dcc', 'dcc (raw)', 'dccPermutedMean', true
  1, 2, 'dccNormalized', 'dcc (normalized)', '', false
  1, 3, 'kappa', 'kappa (raw)', 'kappaPermutedMean', true
  1, 4, 'kappaNormalized', 'kappa (normalized)', '', false
  2, 1, 'decades', 'decades', 'decadesPermutedMean', true
  2, 2, 'tau', 'tau', 'tauPermutedMean', true
  2, 3, 'alpha', 'alpha', 'alphaPermutedMean', true
  2, 4, 'paramSD', 'paramSD', 'paramSDPermutedMean', true
  };
end

function position_figure_full_monitor(fig)
% POSITION_FIGURE_FULL_MONITOR - Size figure to fill a monitor
%
% Variables:
%   fig - Figure handle
%
% Goal:
%   Place the figure on the second monitor when available; otherwise the
%   primary monitor. Size matches the monitor pixel dimensions.

monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
  targetPos = monitorPositions(end, :);
  monitorLabel = 'second';
else
  targetPos = monitorPositions(1, :);
  monitorLabel = 'primary';
end

set(fig, 'Units', 'pixels', 'Position', targetPos);
fprintf('Figure positioned on %s monitor [%d %d %d %d]\n', monitorLabel, targetPos);
end

function apply_buffered_ylim(ax, yValues, bufferFrac)
% APPLY_BUFFERED_YLIM - y-limits with padding around plotted values

if nargin < 3
  bufferFrac = 0.05;
end

yValues = yValues(isfinite(yValues));
if isempty(yValues)
  return;
end

yMin = min(yValues);
yMax = max(yValues);
if yMin == yMax
  pad = max(abs(yMin) * bufferFrac, 0.05 * max(abs(yMin), 1));
  ylim(ax, [yMin - pad, yMax + pad]);
  return;
end

yRange = yMax - yMin;
ylim(ax, [yMin - bufferFrac * yRange, yMax + bufferFrac * yRange]);
end

function hasData = area_has_plot_data(plotData, sessionTypes, areaIdx, metricPlotSpec)
% AREA_HAS_PLOT_DATA - True if any session type has values for this area

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  for m = 1:size(metricPlotSpec, 1)
    fieldName = metricPlotSpec{m, 3};
    if isfield(typeData, fieldName) && areaIdx <= length(typeData.(fieldName)) ...
        && ~isempty(typeData.(fieldName){areaIdx})
      hasData = true;
      return;
    end
  end
end
end
