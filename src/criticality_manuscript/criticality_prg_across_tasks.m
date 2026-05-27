%%
% Criticality PRG Analysis Across Task Types (Manuscript)
%
% Runs momentum-space PRG kurtosis in non-overlapping windows of length
% prgWindow seconds, batches across session types, and plots session summaries
% grouped by sessionType.
%
% Variables (configure in this section):
%   sessionTypes   - Cell array of session types to include
%   dataSource     - 'spikes' or 'lfp'
%   collectStart   - Analysis window start (seconds from session onset)
%   collectEnd     - Analysis window end (seconds)
%   prgWindow      - Non-overlapping block length (seconds); blockWindowSize
%   brainArea      - Single area to analyze (e.g. 'M56'); '' = all valid areas
%   areasToPlot    - Area names to plot; {} uses brainArea if set
%   runBatch       - If true, run criticality_prg_analysis per session
%   plotResults    - If true, create summary figures after batch
%
% Goal:
%   Compare PRG kurtosis (kappa at N/finalCutoffDivisor) across spontaneous,
%   reach, interval, and other session types. Per session: mean and SEM of
%   valid window kappa values; surrogate: mean across windows of (mean
%   surrogate kappa per window), with SEM across those per-window surrogate means.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
dataSource = 'spikes';

collectStart = 0;
collectEnd = 45 * 60;

prgWindow = 30;  % seconds; non-overlapping blocks (blockWindowSize)

brainArea = 'M56';
areasToPlot = {};
runBatch = true;
plotResults = true;

% PRG settings (aligned with run_criticality_prg.m)
analysisConfig = struct();
analysisConfig.blockWindowSize = prgWindow;
analysisConfig.binSize = 0.2;
analysisConfig.cvThreshold = 5;
analysisConfig.cutoffDivisors = [1, 2, 4, 8, 16];
analysisConfig.finalCutoffDivisor = 16;
analysisConfig.kappaAxisMax = 20;
analysisConfig.enableSurrogates = true;
analysisConfig.nSurrogates = 1;
analysisConfig.surrogateMethod = 'isi';
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.nMinNeurons = 25;
analysisConfig.includeM2356 = false;
if ~isempty(brainArea) && strcmpi(brainArea, 'M2356')
  analysisConfig.includeM2356 = true;
end

opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 100;

%% Paths
paths = get_paths();
scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('criticality_prg_across_tasks'));
end
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
addpath(fullfile(srcPath, 'criticality'));

fprintf('\n=== Criticality PRG Across Task Types ===\n');
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('PRG blocks: %.1f s, non-overlapping\n', prgWindow);
fprintf('Kappa at N/%d; bin size: %.3f s; surrogates: %s\n', ...
  analysisConfig.finalCutoffDivisor, analysisConfig.binSize, analysisConfig.surrogateMethod);
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

%% Batch PRG analysis
batchResults = repmat(struct(), numSessions, 1);

if runBatch
  fprintf('\n=== Running PRG analysis (%.0f s non-overlapping blocks) ===\n', prgWindow);
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
      loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectName);
      dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

      [dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea);
      if ~areaOk
        fprintf('  Brain area "%s" not available in this session; skipping.\n', brainArea);
        continue;
      end

      config = analysisConfig;
      prgResults = criticality_prg_analysis(dataStruct, config);

      if ~isempty(brainArea)
        prgResults = filter_prg_results_to_brain_area(prgResults, brainArea);
        if isempty(prgResults.areas)
          fprintf('  No results for brain area "%s"; skipping.\n', brainArea);
          continue;
        end
      end

      batchResults(s).success = true;
      batchResults(s).results = prgResults;
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

%% Aggregate and plot
plotData = aggregate_prg_metrics(batchResults, sessionTypes, analysisConfig.finalCutoffDivisor);

if isempty(plotData.areas)
  error('No PRG metrics extracted. Check that batch analyses succeeded.');
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

if plotResults
  plot_prg_across_tasks(plotData, commonAreas, sessionTypes, collectStart, collectEnd, ...
    prgWindow, paths, brainArea);
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

function results = filter_prg_results_to_brain_area(results, brainArea)
% FILTER_PRG_RESULTS_TO_BRAIN_AREA - Keep one area in PRG results struct

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end

areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end

cellFields = {'kappa', 'kappaByCutoff', 'windowStartS', 'popCv', 'windowExcluded', ...
  'nNeuronsPerWindow', 'kappaSurrogate', 'nCutoffList'};

results.areas = results.areas(areaIdx);
for f = 1:length(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
end

function plotData = aggregate_prg_metrics(batchResults, sessionTypes, finalCutoffDivisor)
% AGGREGATE_PRG_METRICS - Per-session mean and SEM of window kappa and surrogate summary
%
% Variables:
%   finalCutoffDivisor - Reported in plot labels (N/divisor kurtosis)

if nargin < 3 || isempty(finalCutoffDivisor)
  finalCutoffDivisor = 16;
end

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.finalCutoffDivisor = finalCutoffDivisor;

metricFields = {'kappaMean', 'kappaSem', 'kappaShuffleMean', 'kappaShuffleSem'};

for s = 1:length(batchResults)
  if ~batchResults(s).success || isempty(batchResults(s).results)
    continue;
  end

  results = batchResults(s).results;
  sessionType = batchResults(s).sessionType;

  if isempty(plotData.areas)
    for t = 1:length(sessionTypes)
      st = sessionTypes{t};
      plotData.byType.(matlab.lang.makeValidName(st)) = init_type_prg_metrics(metricFields, 0);
    end
  end

  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_type_prg_metrics(metricFields, length(plotData.areas));
  end
  typeData = plotData.byType.(typeKey);

  for a = 1:length(results.areas)
    areaName = results.areas{a};
    areaIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaIdx)
      plotData.areas{end+1} = areaName;
      areaIdx = length(plotData.areas);
      plotData = extend_prg_plot_data_areas(plotData, sessionTypes, metricFields, areaIdx);
      typeData = plotData.byType.(typeKey);
    end

    summary = summarize_session_kappa_windows(results, a);
    for m = 1:length(metricFields)
      fieldName = metricFields{m};
      typeData.(fieldName){areaIdx} = [typeData.(fieldName){areaIdx}, summary.(fieldName)];
    end
  end

  typeData.sessionLabels{end+1} = batchResults(s).label;
  typeData.sessionNames{end+1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
end
end

function summary = summarize_session_kappa_windows(results, areaIdx)
% SUMMARIZE_SESSION_KAPPA_WINDOWS - Mean and SEM across valid windows; shuffle summary
%
% Variables:
%   results  - Output from criticality_prg_analysis
%   areaIdx  - Index into results.areas
%
% Returns:
%   summary - Struct with kappaMean, kappaSem, kappaShuffleMean, kappaShuffleSem
%
%   kappaShuffleSem - SEM across windows of (mean surrogate kappa in each window)

summary = struct('kappaMean', nan, 'kappaSem', nan, 'kappaShuffleMean', nan, ...
  'kappaShuffleSem', nan);

if areaIdx > length(results.kappa) || isempty(results.kappa{areaIdx})
  return;
end

kappaVec = results.kappa{areaIdx}(:);
nWin = numel(kappaVec);
excluded = false(nWin, 1);
if isfield(results, 'windowExcluded') && areaIdx <= length(results.windowExcluded) ...
    && ~isempty(results.windowExcluded{areaIdx})
  excluded = results.windowExcluded{areaIdx}(:);
  if numel(excluded) ~= nWin
    excluded = false(nWin, 1);
  end
end
validMask = isfinite(kappaVec) & ~excluded;
kappaValid = kappaVec(validMask);
if ~isempty(kappaValid)
  summary.kappaMean = mean(kappaValid);
  nK = numel(kappaValid);
  if nK > 1
    summary.kappaSem = std(kappaValid) / sqrt(nK);
  else
    summary.kappaSem = 0;
  end
end

if isfield(results, 'kappaSurrogate') && areaIdx <= length(results.kappaSurrogate) ...
    && ~isempty(results.kappaSurrogate{areaIdx})
  surrMat = results.kappaSurrogate{areaIdx};
  if size(surrMat, 1) == nWin
    surrMat = surrMat(validMask, :);
  end
  if size(surrMat, 2) > 1
    perWindowShuffleMean = nanmean(surrMat, 2);
  else
    perWindowShuffleMean = surrMat(:);
  end
  perWindowShuffleMean = perWindowShuffleMean(isfinite(perWindowShuffleMean));
  if ~isempty(perWindowShuffleMean)
    summary.kappaShuffleMean = mean(perWindowShuffleMean);
    nSh = numel(perWindowShuffleMean);
    if nSh > 1
      summary.kappaShuffleSem = std(perWindowShuffleMean) / sqrt(nSh);
    else
      summary.kappaShuffleSem = 0;
    end
  end
end
end

function plotData = extend_prg_plot_data_areas(plotData, sessionTypes, metricFields, newAreaIdx)
% EXTEND_PRG_PLOT_DATA_AREAS - Grow metric storage when a new area appears

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

function typeData = init_type_prg_metrics(metricFields, numAreas)
% INIT_TYPE_PRG_METRICS - Empty storage per area for one session type

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

function plot_prg_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, collectEnd, prgWindow, paths, brainArea)
% PLOT_PRG_ACROSS_TASKS - Session kappa with surrogate summary, grouped by session type

if nargin < 8 || isempty(brainArea)
  brainArea = '';
end

finalCutoffDivisor = 16;
if isfield(plotData, 'finalCutoffDivisor') && ~isempty(plotData.finalCutoffDivisor)
  finalCutoffDivisor = plotData.finalCutoffDivisor;
end

kappaYLabel = sprintf('\\kappa (N/%d, mean \\pm SEM across windows)', finalCutoffDivisor);
kappaTitleWord = sprintf('\\kappa (N/%d)', finalCutoffDivisor);

numSessionTypes = length(sessionTypes);
typeColors = lines(max(numSessionTypes, 3));
shuffleBarColor = [0.55, 0.55, 0.55];

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

for a = 1:length(areasToPlot)
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx) || ~area_has_prg_plot_data(plotData, sessionTypes, areaIdx)
    continue;
  end

  figKappa = figure(6000 + a);
  clf(figKappa);
  position_figure_full_monitor(figKappa);
  axKappa = axes(figKappa);
  plot_kappa_with_shuffle(axKappa, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor);
  ylabel(axKappa, kappaYLabel);
  title(axKappa, sprintf('%s — %s', areaName, kappaTitleWord));
  yline(axKappa, 3, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');

  if ~isempty(brainArea)
    titleStr = sprintf('PRG %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
      kappaTitleWord, brainArea, collectStart, collectEnd, prgWindow);
  else
    titleStr = sprintf('PRG %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
      kappaTitleWord, areaName, collectStart, collectEnd, prgWindow);
  end
  sgtitle(figKappa, titleStr, 'FontWeight', 'bold');

  plotBase = make_prg_plot_basename('criticality_prg_across_tasks', areaName, brainArea, ...
    prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor);
  exportgraphics(figKappa, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(figKappa, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
end

fprintf('\nAll figures saved to %s\n', saveDir);
end

function plot_kappa_with_shuffle(ax, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor)
% PLOT_KAPPA_WITH_SHUFFLE - Session mean kappa with SEM; surrogate mean with SEM beside each session

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendShown = false;

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if ~isfield(typeData, 'kappaMean') || areaIdx > length(typeData.kappaMean)
    continue;
  end

  kappaMeans = typeData.kappaMean{areaIdx};
  kappaSems = typeData.kappaSem{areaIdx};
  shuffleMeans = typeData.kappaShuffleMean{areaIdx};
  shuffleSems = typeData.kappaShuffleSem{areaIdx};
  kappaMeans = kappaMeans(:)';
  kappaSems = kappaSems(:)';
  shuffleMeans = shuffleMeans(:)';
  shuffleSems = shuffleSems(:)';
  if numel(shuffleSems) ~= numel(shuffleMeans)
    shuffleSems = nan(size(shuffleMeans));
  end
  numBars = numel(kappaMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, kappaMeans, kappaSems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 12, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', 'session \kappa');

  if any(isfinite(shuffleMeans))
    xShuffle = xPos + 0.22;
    semPlot = shuffleSems(:)';
    semPlot(~isfinite(semPlot)) = 0;
    errorbar(ax, xShuffle, shuffleMeans, semPlot, 's', ...
      'Color', shuffleBarColor, 'MarkerFaceColor', shuffleBarColor, ...
      'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 12, 'LineWidth', 1.2, ...
      'CapSize', 8, 'DisplayName', 'surrogate mean \pm SEM (across windows)');
  end

  groupCenter = mean(xPos);
  xticksCenters(end+1) = groupCenter; %#ok<AGROW>
  xtickLabels{end+1} = sessionType; %#ok<AGROW>

  validMeans = kappaMeans(isfinite(kappaMeans));
  if ~isempty(validMeans)
    yline(ax, mean(validMeans), '--', 'Color', typeColors(t, :), 'LineWidth', 1.5, ...
      'HandleVisibility', 'off');
  end

  xCursor = xPos(end) + 1.5;
  legendShown = true;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
end
grid(ax, 'on');
if legendShown
  legend(ax, {'session \kappa', 'surrogate mean \pm SEM (across windows)'}, 'Location', 'best');
end
hold(ax, 'off');
end

function plotBase = make_prg_plot_basename(prefix, areaName, brainArea, prgWindow, collectStart, collectEnd, multiArea, finalCutoffDivisor)
% MAKE_PRG_PLOT_BASENAME - Filename stem for saved figures

if nargin < 8 || isempty(finalCutoffDivisor)
  finalCutoffDivisor = 16;
end

if ~isempty(brainArea)
  plotBase = sprintf('%s_%s_win%.0fs_%.0f-%.0fs_N%d', prefix, brainArea, prgWindow, collectStart, collectEnd, finalCutoffDivisor);
else
  plotBase = sprintf('%s_%s_win%.0fs_%.0f-%.0fs_N%d', prefix, areaName, prgWindow, collectStart, collectEnd, finalCutoffDivisor);
end
if multiArea
  plotBase = sprintf('%s_area%s', plotBase, areaName);
end
end

function hasData = area_has_prg_plot_data(plotData, sessionTypes, areaIdx)
% AREA_HAS_PRG_PLOT_DATA - True if any session type has kappa values for this area

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'kappaMean') && areaIdx <= length(typeData.kappaMean) ...
      && ~isempty(typeData.kappaMean{areaIdx})
    hasData = true;
    return;
  end
end
end

function position_figure_full_monitor(fig)
% POSITION_FIGURE_FULL_MONITOR - Size figure to fill a monitor

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
