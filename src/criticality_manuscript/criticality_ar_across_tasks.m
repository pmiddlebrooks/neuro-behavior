%%
% Criticality AR (d2) Analysis Across Task Types (Manuscript)
%
% Runs d2 criticality in non-overlapping windows of length d2Window seconds,
% batches across session types, and plots session summaries grouped by sessionType.
%
% Variables (configure in this section):
%   sessionTypes   - Cell array of session types to include
%   dataSource     - 'spikes' or 'lfp'
%   collectStart   - Analysis window start (seconds from session onset)
%   collectEnd     - Analysis window end (seconds)
%   d2Window       - Non-overlapping window length (seconds); stepSize = d2Window
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' = all areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   areasToPlot            - Area names to plot; {} uses brainArea if set
%   runBatch       - If true, run criticality_ar_analysis per session
%   plotResults    - If true, create summary figures after batch
%   useLog10D2     - If true, aggregate and plot log10(d2); values <= 0 become NaN (default true)
%   useSubsampling - If true, d2 per window = mean across neuron subsamples; shuffled
%                    summary = mean across subsamples of per-subsample shuffle means
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings (see run_criticality_ar.m)
%
% Goal:
%   Compare d2 (raw + shuffled) and normalized d2 across spontaneous, reach,
%   interval, and other session types. Per session: mean and SEM of window d2 values;
%   shuffled: mean across windows of (mean shuffle d2 per window), with SEM across
%   windows of those per-window shuffle means.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
dataSource = 'spikes';

collectStart = 0;
collectEnd = 10 * 60;

d2Window = 30;  % seconds; non-overlapping windows (stepSize = d2Window)

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
% brainArea = 'M23M56';  % merge M23+M56 (preset in brainAreaCombinations)
areasToPlot = {};
runBatch = true;
plotResults = true;

% Display / summary scale (matches run_criticality_ar.m config.useLog10D2)
useLog10D2 = true;

% Neural subsampling (matches run_criticality_ar.m)
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.5;

% AR / d2 analysis settings (aligned with run_criticality_ar.m; no mrBr)
analysisConfig = struct();
analysisConfig.slidingWindowSize = d2Window;
analysisConfig.stepSize = d2Window;
analysisConfig.binSize = 0.025;
analysisConfig.binSize = 0.05;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = true;
analysisConfig.nShuffles = 10;
analysisConfig.normalizeD2 = true;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = 30;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 150;

%% Paths
paths = get_paths();
scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('criticality_ar_across_tasks'));
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

fprintf('\n=== Criticality d2 Across Task Types ===\n');
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('d2 windows: %.1f s, non-overlapping (step = window)\n', d2Window);
fprintf('useLog10D2 (aggregate/plot): %d\n', useLog10D2);
if useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
    nSubsamples, nNeuronsSubsample, minNeuronsMultiple);
else
  fprintf('Subsampling: off\n');
end
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

%% Batch d2 analysis
batchResults = repmat(struct(), numSessions, 1);

if runBatch
  fprintf('\n=== Running d2 analysis (%.0f s non-overlapping windows) ===\n', d2Window);
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

      [dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations);
      if ~areaOk
        fprintf('  Brain area "%s" not available in this session; skipping.\n', brainArea);
        continue;
      end

      config = analysisConfig;
      arResults = criticality_ar_analysis(dataStruct, config);

      if ~isempty(brainArea)
        arResults = filter_ar_results_to_brain_area(arResults, brainArea);
        if isempty(arResults.areas)
          fprintf('  No results for brain area "%s"; skipping.\n', brainArea);
          continue;
        end
      end

      batchResults(s).success = true;
      batchResults(s).results = arResults;
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
plotData = aggregate_ar_metrics(batchResults, sessionTypes, useLog10D2);

if isempty(plotData.areas)
  error('No d2 metrics extracted. Check that batch analyses succeeded.');
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
  plot_ar_across_tasks(plotData, commonAreas, sessionTypes, collectStart, collectEnd, ...
    d2Window, paths, brainArea, useLog10D2);
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

function results = filter_ar_results_to_brain_area(results, brainArea)
% FILTER_AR_RESULTS_TO_BRAIN_AREA - Keep one area in AR results struct

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end

areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end

cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'mrBrPermuted', ...
  'd2PermutedMean', 'd2PermutedSEM', 'popActivityWindows', 'popActivityFull'};

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

function plotData = aggregate_ar_metrics(batchResults, sessionTypes, useLog10D2)
% AGGREGATE_AR_METRICS - Per-session mean and SEM of window d2 and shuffle summary
%
% Variables:
%   useLog10D2 - If true, apply log10 to d2, shuffles, and normalized d2 before stats

if nargin < 3 || isempty(useLog10D2)
  useLog10D2 = false;
end

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.useLog10D2 = useLog10D2;

metricFields = {'d2Mean', 'd2Sem', 'd2ShuffleMean', 'd2ShuffleSem', 'd2NormMean', 'd2NormSem'};

for s = 1:length(batchResults)
  if ~batchResults(s).success || isempty(batchResults(s).results)
    continue;
  end

  results = batchResults(s).results;
  sessionType = batchResults(s).sessionType;

  if isempty(plotData.areas)
    for t = 1:length(sessionTypes)
      st = sessionTypes{t};
      plotData.byType.(matlab.lang.makeValidName(st)) = init_type_ar_metrics(metricFields, 0);
    end
  end

  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_type_ar_metrics(metricFields, length(plotData.areas));
  end
  typeData = plotData.byType.(typeKey);

  for a = 1:length(results.areas)
    areaName = results.areas{a};
    areaIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaIdx)
      plotData.areas{end+1} = areaName;
      areaIdx = length(plotData.areas);
      plotData = extend_ar_plot_data_areas(plotData, sessionTypes, metricFields, areaIdx);
      typeData = plotData.byType.(typeKey);
    end

    summary = summarize_session_d2_windows(results, a, useLog10D2);
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

function summary = summarize_session_d2_windows(results, areaIdx, useLog10D2)
% SUMMARIZE_SESSION_D2_WINDOWS - Mean and SEM across windows; shuffle = mean of window means
%
% Variables:
%   results      - Output from criticality_ar_analysis
%   areaIdx      - Index into results.areas
%   useLog10D2   - If true, use log10(d2) etc. (x>0 only; same as criticality_ar_plot)
%
% Returns:
%   summary - Struct with d2Mean, d2Sem, d2ShuffleMean, d2ShuffleSem, d2NormMean, d2NormSem
%
%   d2ShuffleSem - SEM across windows of per-window shuffle summary
%   (with subsampling: mean across subsamples of per-subsample shuffle means)

if nargin < 3 || isempty(useLog10D2)
  useLog10D2 = false;
end

summary = struct('d2Mean', nan, 'd2Sem', nan, 'd2ShuffleMean', nan, ...
  'd2ShuffleSem', nan, 'd2NormMean', nan, 'd2NormSem', nan);

if areaIdx > length(results.d2) || isempty(results.d2{areaIdx})
  return;
end

d2Vec = results.d2{areaIdx}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
d2Vec = d2Vec(isfinite(d2Vec));
if ~isempty(d2Vec)
  summary.d2Mean = mean(d2Vec);
  nD2 = numel(d2Vec);
  if nD2 > 1
    summary.d2Sem = std(d2Vec) / sqrt(nD2);
  else
    summary.d2Sem = 0;
  end
end

if isfield(results, 'd2Normalized') && areaIdx <= length(results.d2Normalized) ...
    && ~isempty(results.d2Normalized{areaIdx})
  d2NormVec = results.d2Normalized{areaIdx}(:);
  if useLog10D2
    d2NormVec = log10_safe_numeric(d2NormVec);
  end
  d2NormVec = d2NormVec(isfinite(d2NormVec));
  if ~isempty(d2NormVec)
    summary.d2NormMean = mean(d2NormVec);
    nNorm = numel(d2NormVec);
    if nNorm > 1
      summary.d2NormSem = std(d2NormVec) / sqrt(nNorm);
    else
      summary.d2NormSem = 0;
    end
  end
end

if isfield(results, 'd2Permuted') && areaIdx <= length(results.d2Permuted) ...
    && ~isempty(results.d2Permuted{areaIdx})
  perWindowShuffleMean = get_per_window_shuffle_mean_d2(results, areaIdx, useLog10D2);
  perWindowShuffleMean = perWindowShuffleMean(isfinite(perWindowShuffleMean));
  if ~isempty(perWindowShuffleMean)
    summary.d2ShuffleMean = mean(perWindowShuffleMean);
    nSh = numel(perWindowShuffleMean);
    if nSh > 1
      summary.d2ShuffleSem = std(perWindowShuffleMean) / sqrt(nSh);
    else
      summary.d2ShuffleSem = 0;
    end
  end
end
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive (matches criticality_ar_plot)

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function plotData = extend_ar_plot_data_areas(plotData, sessionTypes, metricFields, newAreaIdx)
% EXTEND_AR_PLOT_DATA_AREAS - Grow metric storage when a new area appears

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

function typeData = init_type_ar_metrics(metricFields, numAreas)
% INIT_TYPE_AR_METRICS - Empty storage per area for one session type

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

function plot_ar_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, collectEnd, d2Window, paths, brainArea, useLog10D2)
% PLOT_AR_ACROSS_TASKS - Raw+shuffled d2 and normalized d2 bar plots by session type

if nargin < 8 || isempty(brainArea)
  brainArea = '';
end
if nargin < 9 || isempty(useLog10D2)
  if isfield(plotData, 'useLog10D2')
    useLog10D2 = plotData.useLog10D2;
  else
    useLog10D2 = false;
  end
end
useLog10D2 = logical(useLog10D2);

if useLog10D2
  rawYLabel = 'log_{10}(d2) (mean \pm SEM across windows)';
  normYLabel = 'log_{10}(d2 normalized) (mean \pm SEM across windows)';
  rawTitleWord = 'log_{10}(d2)';
  normTitleWord = 'log_{10}(d2 normalized)';
else
  rawYLabel = 'd2 (mean \pm SEM across windows)';
  normYLabel = 'd2 normalized (mean \pm SEM across windows)';
  rawTitleWord = 'd2';
  normTitleWord = 'd2 normalized';
end

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
  if isempty(areaIdx) || ~area_has_ar_plot_data(plotData, sessionTypes, areaIdx)
    continue;
  end

  % Figure 1: raw d2 + shuffled mean
  figRaw = figure(5000 + 2 * a - 1);
  clf(figRaw);
  position_figure_full_monitor(figRaw);
  axRaw = axes(figRaw);
  plot_d2_raw_with_shuffle(axRaw, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor);
  ylabel(axRaw, rawYLabel);
  title(axRaw, sprintf('%s — %s', areaName, rawTitleWord));
  if ~isempty(brainArea)
    titleStrRaw = sprintf('%s (raw) — %s [%.0f–%.0f s, %.0fs windows]', ...
      rawTitleWord, brainArea, collectStart, collectEnd, d2Window);
  else
    titleStrRaw = sprintf('%s (raw) — %s [%.0f–%.0f s, %.0fs windows]', ...
      rawTitleWord, areaName, collectStart, collectEnd, d2Window);
  end
  sgtitle(figRaw, titleStrRaw, 'FontWeight', 'bold');

  plotBaseRaw = make_ar_plot_basename('criticality_ar_across_tasks_raw', areaName, brainArea, ...
    d2Window, collectStart, collectEnd, length(areasToPlot) > 1, useLog10D2);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.png']), 'Resolution', 300);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseRaw));

  % Figure 2: normalized d2
  figNorm = figure(5000 + 2 * a);
  clf(figNorm);
  position_figure_full_monitor(figNorm);
  axNorm = axes(figNorm);
  plot_d2_normalized(axNorm, plotData, sessionTypes, areaIdx, typeColors);
  ylabel(axNorm, normYLabel);
  title(axNorm, sprintf('%s — %s', areaName, normTitleWord));
  if ~isempty(brainArea)
    titleStrNorm = sprintf('%s — %s [%.0f–%.0f s, %.0fs windows]', ...
      normTitleWord, brainArea, collectStart, collectEnd, d2Window);
  else
    titleStrNorm = sprintf('%s — %s [%.0f–%.0f s, %.0fs windows]', ...
      normTitleWord, areaName, collectStart, collectEnd, d2Window);
  end
  sgtitle(figNorm, titleStrNorm, 'FontWeight', 'bold');

  plotBaseNorm = make_ar_plot_basename('criticality_ar_across_tasks_normalized', areaName, brainArea, ...
    d2Window, collectStart, collectEnd, length(areasToPlot) > 1, useLog10D2);
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.png']), 'Resolution', 300);
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseNorm));
end

fprintf('\nAll figures saved to %s\n', saveDir);
end

function plot_d2_raw_with_shuffle(ax, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor)
% PLOT_D2_RAW_WITH_SHUFFLE - Session mean d2 with SEM; shuffle mean with SEM beside each session

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
  if ~isfield(typeData, 'd2Mean') || areaIdx > length(typeData.d2Mean)
    continue;
  end

  d2Means = typeData.d2Mean{areaIdx};
  d2Sems = typeData.d2Sem{areaIdx};
  shuffleMeans = typeData.d2ShuffleMean{areaIdx};
  shuffleSems = typeData.d2ShuffleSem{areaIdx};
  d2Means = d2Means(:)';
  d2Sems = d2Sems(:)';
  shuffleMeans = shuffleMeans(:)';
  shuffleSems = shuffleSems(:)';
  if numel(shuffleSems) ~= numel(shuffleMeans)
    shuffleSems = nan(size(shuffleMeans));
  end
  numBars = numel(d2Means);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, d2Means, d2Sems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 24, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', 'session d2');

  if any(isfinite(shuffleMeans))
    xShuffle = xPos + 0.22;
    semPlot = shuffleSems(:)';
    semPlot(~isfinite(semPlot)) = 0;
    errorbar(ax, xShuffle, shuffleMeans, semPlot, 's', ...
      'Color', shuffleBarColor, 'MarkerFaceColor', shuffleBarColor, ...
      'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 24, 'LineWidth', 1.2, ...
      'CapSize', 8, 'DisplayName', 'shuffled mean \pm SEM (across windows)');
  end

  barLabels = get_session_bar_labels(typeData, numBars, sessionType);
  xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
  xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

  validMeans = d2Means(isfinite(d2Means));
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
  xtickangle(ax, 45);
end
grid(ax, 'on');
if legendShown
  legend(ax, {'session d2', 'shuffled mean \pm SEM (across windows)'}, 'Location', 'best');
end
hold(ax, 'off');
end

function plot_d2_normalized(ax, plotData, sessionTypes, areaIdx, typeColors)
% PLOT_D2_NORMALIZED - Session mean normalized d2 with SEM across windows

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if ~isfield(typeData, 'd2NormMean') || areaIdx > length(typeData.d2NormMean)
    continue;
  end

  normMeans = typeData.d2NormMean{areaIdx};
  normSems = typeData.d2NormSem{areaIdx};
  normMeans = normMeans(:)';
  normSems = normSems(:)';
  numBars = numel(normMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, normMeans, normSems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 24, 'LineWidth', 1.2, ...
      'CapSize', 8);

  barLabels = get_session_bar_labels(typeData, numBars, sessionType);
  xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
  xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

  validMeans = normMeans(isfinite(normMeans));
  if ~isempty(validMeans)
    yline(ax, mean(validMeans), '--', 'Color', typeColors(t, :), 'LineWidth', 1.5);
  end

  xCursor = xPos(end) + 1.5;
end

if ~isempty(xticksCenters)
  xticks(ax, xticksCenters);
  xticklabels(ax, xtickLabels);
  xtickangle(ax, 45);
end
grid(ax, 'on');
hold(ax, 'off');
end

function barLabels = get_session_bar_labels(typeData, numBars, sessionType)
% GET_SESSION_BAR_LABELS - One x-axis label per session bar (sessionName)
%
% Variables:
%   typeData    - Aggregated metrics for one session type
%   numBars     - Number of bars in the current group
%   sessionType - Fallback label if sessionNames unavailable
%
% Returns:
%   barLabels - 1 x numBars cell of char labels

if isfield(typeData, 'sessionNames') && numel(typeData.sessionNames) >= numBars
  barLabels = typeData.sessionNames(1:numBars);
elseif isfield(typeData, 'sessionLabels') && numel(typeData.sessionLabels) >= numBars
  barLabels = typeData.sessionLabels(1:numBars);
else
  barLabels = repmat({sessionType}, 1, numBars);
end
barLabels = cellfun(@char, barLabels, 'UniformOutput', false);
end

function plotBase = make_ar_plot_basename(prefix, areaName, brainArea, d2Window, collectStart, collectEnd, multiArea, useLog10D2)
% MAKE_AR_PLOT_BASENAME - Filename stem for saved figures

if nargin < 8 || isempty(useLog10D2)
  useLog10D2 = false;
else
  useLog10D2 = logical(useLog10D2);
end

if ~isempty(brainArea)
  plotBase = sprintf('%s_%s_win%.0fs_%.0f-%.0fs', prefix, brainArea, d2Window, collectStart, collectEnd);
else
  plotBase = sprintf('%s_%s_win%.0fs_%.0f-%.0fs', prefix, areaName, d2Window, collectStart, collectEnd);
end
if multiArea
  plotBase = sprintf('%s_area%s', plotBase, areaName);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
end

function hasData = area_has_ar_plot_data(plotData, sessionTypes, areaIdx)
% AREA_HAS_AR_PLOT_DATA - True if any session type has d2 values for this area

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'd2Mean') && areaIdx <= length(typeData.d2Mean) ...
      && ~isempty(typeData.d2Mean{areaIdx})
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
