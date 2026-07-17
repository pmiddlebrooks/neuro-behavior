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
%   collectEnd    - Window end (seconds); [] = full session; default 40*60
%   brainArea              - Single or merged area (e.g. 'M23', 'M23M56'); '' = all areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   areasToPlot            - Area names to plot; {} uses brainArea if set, else all areas
%   runBatch             - If true, run criticality_av_analysis per session
%   plotResults          - If true, create summary figures after batch
%   powerLawFitMethod    - 'clauset', 'plfit2023', or 'hybrid'
%   avalancheDetectionMode - 'fixedBinMedian' or 'meanIsiZero' (see below)
%   clausetPlfitPath     - Path to Clauset toolbox MATLAB Code folder
%   plfit2023Path        - Path to folder containing plfit2023.m
%   useSubsampling       - If true, metrics = mean across neuron subsamples per window
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%
%   avalancheDetectionMode:
%     'fixedBinMedian' - config.binSize (s) + per-window median cutoff (default)
%     'meanIsiZero'    - bin size = mean population ISI; zero cutoff (literature)
%
% Goal:
%   Compare avalanche metrics (dcc, kappa, decades, tau, alpha,
%   paramSD = crackling 1/σνz from ⟨S⟩~T^γ)
%   across spontaneous, reach, interval, and other session types using
%   load_session_data and a common analysis window per session.

function out = criticality_av_across_tasks(opts)
% CRITICALITY_AV_ACROSS_TASKS - Batch and plot avalanche metrics across session types
%
% Usage:
%   out = criticality_av_across_tasks();
%   out = criticality_av_across_tasks(struct('plotResults', false, 'saveBatchResults', true));

if nargin < 1 || isempty(opts)
  opts = struct();
end
opts = fill_criticality_av_across_tasks_opts(opts);
setup_criticality_manuscript_paths('criticality_av_across_tasks');
paths = get_paths();

if isempty(opts.batchResultsFile)
  opts.batchResultsFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
    'criticality_av_across_tasks_batch.mat');
end

if isempty(opts.collectEnd)
  windowDurationSec = [];
else
  windowDurationSec = opts.collectEnd - opts.collectStart;
end
[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();

fprintf('\n=== Criticality Avalanche Across Task Types ===\n');
fprintf('Power-law fit method: %s\n', opts.powerLawFitMethod);
fprintf('Avalanche detection mode: %s\n', opts.avalancheDetectionMode);
if opts.useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
    opts.nSubsamples, opts.nNeuronsSubsample, opts.minNeuronsMultiple);
else
  fprintf('Subsampling: off\n');
end
if isempty(opts.collectEnd)
  fprintf('Collect window: [%.1f, full] s\n', opts.collectStart);
else
  fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', opts.collectStart, opts.collectEnd, ...
    windowDurationSec / 60);
end
fprintf('Session types: %s\n', strjoin(opts.sessionTypes, ', '));
if ~isempty(opts.brainArea)
  fprintf('Brain area: %s (single-area analysis)\n', opts.brainArea);
else
  fprintf('Brain area: all areas in each session\n');
end

sessionTable = build_session_table(opts.sessionTypes);
numSessions = size(sessionTable, 1);
fprintf('Total sessions: %d\n', numSessions);
if numSessions == 0
  error('No sessions found for the requested session types.');
end

if opts.runBatch
  batchResults = run_av_across_tasks_batch(sessionTable, opts, clausetPlfitPath, plfit2023Path);
  plotData = aggregate_av_metrics(batchResults, opts.sessionTypes);
  batchMeta = pack_av_across_tasks_batch_meta(opts);
  if opts.saveBatchResults
    save(opts.batchResultsFile, 'batchResults', 'plotData', 'batchMeta', '-v7.3');
    fprintf('\nSaved batch results: %s\n', opts.batchResultsFile);
  end
else
  if ~isfile(opts.batchResultsFile)
    error('criticality_av_across_tasks:NoBatchFile', ...
      'Batch file not found: %s. Set runBatch true to compute.', opts.batchResultsFile);
  end
  loaded = load(opts.batchResultsFile, 'batchResults', 'plotData', 'batchMeta');
  batchResults = loaded.batchResults;
  plotData = loaded.plotData;
  batchMeta = loaded.batchMeta;
  fprintf('\nLoaded batch results: %s\n', opts.batchResultsFile);
end

if isempty(plotData.areas)
  error('No avalanche metrics extracted. Check that batch analyses succeeded.');
end

commonAreas = resolve_areas_to_plot(plotData.areas, opts.areasToPlot, opts.brainArea);

fprintf('\n=== Areas for plotting ===\n');
fprintf('  %s\n', strjoin(commonAreas, ', '));

if opts.plotResults
  plot_av_across_tasks(plotData, commonAreas, opts.sessionTypes, opts.collectStart, ...
    opts.collectEnd, paths, opts.brainArea);
  plot_av_crackling_relation_across_tasks(plotData, commonAreas, opts.sessionTypes, ...
    opts.collectStart, opts.collectEnd, paths, opts.brainArea);
end

fprintf('\n=== Done ===\n');

out = struct();
out.batchResults = batchResults;
out.plotData = plotData;
out.batchMeta = batchMeta;
out.paths = paths;
out.areasToPlot = commonAreas;
end

function opts = fill_criticality_av_across_tasks_opts(opts)
defaults = struct();
defaults.sessionTypes = {'spontaneous', 'interval', 'reach'};
defaults.dataSource = 'spikes';
defaults.collectStart = 0;
defaults.collectEnd = 45 * 60;
defaults.brainArea = 'M23M56';
defaults.brainAreaCombinations = default_manuscript_brain_area_combinations();
defaults.areasToPlot = {};
defaults.runBatch = true;
defaults.plotResults = true;
defaults.saveBatchResults = false;
defaults.batchResultsFile = '';
defaults.powerLawFitMethod = 'hybrid';
defaults.gofThreshold = 0.8;
defaults.avalancheDetectionMode = 'fixedBinMedian';
defaults.useSubsampling = false;
defaults.nSubsamples = 20;
defaults.nNeuronsSubsample = 20;
defaults.minNeuronsMultiple = 1.25;
defaults.firingRateCheckTime = [];
defaults.minFiringRate = 0.05;
defaults.maxFiringRate = 100;
defaults.enablePermutations = true;
defaults.nShuffles = 5;
preserveCollectEndEmpty = isfield(opts, 'collectEnd') && isempty(opts.collectEnd);
opts = merge_struct_defaults(opts, defaults);
if preserveCollectEndEmpty
  opts.collectEnd = [];
end
end

function batchMeta = pack_av_across_tasks_batch_meta(opts)
batchMeta = struct( ...
  'sessionTypes', {opts.sessionTypes}, ...
  'collectStart', opts.collectStart, ...
  'collectEnd', opts.collectEnd, ...
  'brainArea', opts.brainArea, ...
  'areasToPlot', {opts.areasToPlot}, ...
  'powerLawFitMethod', opts.powerLawFitMethod, ...
  'avalancheDetectionMode', opts.avalancheDetectionMode);
end

function batchResults = run_av_across_tasks_batch(sessionTable, opts, clausetPlfitPath, plfit2023Path)
if isempty(opts.collectEnd)
  windowDurationSec = [];
else
  windowDurationSec = opts.collectEnd - opts.collectStart;
end
analysisConfig = build_av_analysis_config(opts, windowDurationSec, clausetPlfitPath, plfit2023Path);
loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = opts.firingRateCheckTime;
loadOpts.collectStart = opts.collectStart;
loadOpts.collectEnd = opts.collectEnd;
loadOpts.minFiringRate = opts.minFiringRate;
loadOpts.maxFiringRate = opts.maxFiringRate;

numSessions = size(sessionTable, 1);
batchResults = repmat(struct(), numSessions, 1);
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
    loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
    dataStruct = load_session_data(sessionType, opts.dataSource, loadArgs{:});
    [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
      dataStruct, opts.brainArea, opts.brainAreaCombinations);
    if ~areaOk
      fprintf('  Brain area "%s" not available in this session; skipping.\n', opts.brainArea);
      continue;
    end

    sessionConfig = analysisConfig;
    sessionDuration = get_session_collect_duration(dataStruct, opts);
    durationToleranceSec = 1;
    useFullSessionWindow = isempty(windowDurationSec) ...
      || sessionDuration < (windowDurationSec - durationToleranceSec);
    if useFullSessionWindow
      if isempty(windowDurationSec)
        fprintf('  Using full session window (%.1f s).\n', sessionDuration);
      else
        fprintf('  Session duration %.1f s < requested %.1f s; using full session window.\n', ...
          sessionDuration, windowDurationSec);
      end
      sessionConfig.slidingWindowSize = sessionDuration;
      sessionConfig.avStepSize = sessionDuration;
    end

    avResults = criticality_av_analysis(dataStruct, sessionConfig);
    if ~isempty(opts.brainArea)
      avResults = filter_av_results_to_brain_area(avResults, opts.brainArea);
      if isempty(avResults.areas)
        fprintf('  No results for brain area "%s"; skipping.\n', opts.brainArea);
        continue;
      end
    end

    batchResults(s).success = true;
    batchResults(s).results = avResults;
    fprintf('  Analysis completed.\n');
  catch ME
    if is_skippable_session_analysis_error(ME)
      fprintf('  Skipping session (insufficient neurons / no valid areas): %s\n', ME.message);
      batchResults(s).skipReason = ME.message;
      continue;
    end
    fprintf('  Error: %s\n', ME.message);
    for st = 1:length(ME.stack)
      fprintf('    %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
    end
    error('criticality_av_across_tasks:SessionFailed', ...
      'Batch stopped at session %d/%d [%s] %s: %s', ...
      s, numSessions, sessionType, sessionName, ME.message);
  end
end
end

function tf = is_skippable_session_analysis_error(ME)
% IS_SKIPPABLE_SESSION_ANALYSIS_ERROR - True for expected per-session skip cases
tf = contains(ME.message, 'No valid areas to process') ...
  || contains(ME.message, 'insufficient neurons');
end

function analysisConfig = build_av_analysis_config(opts, windowDurationSec, clausetPlfitPath, plfit2023Path)
analysisConfig = struct();
if isempty(windowDurationSec)
  analysisConfig.slidingWindowSize = 1;
  analysisConfig.avStepSize = 1;
else
  analysisConfig.slidingWindowSize = windowDurationSec;
  analysisConfig.avStepSize = windowDurationSec;
end
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.avalancheDetectionMode = opts.avalancheDetectionMode;
if strcmpi(opts.avalancheDetectionMode, 'meanIsiZero')
  % bin size set per area in criticality_av_analysis
else
  analysisConfig.binSize = 0.05;
end
analysisConfig.analyzeDcc = true;
analysisConfig.analyzeKappa = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 5;
analysisConfig.enablePermutations = opts.enablePermutations;
analysisConfig.nShuffles = opts.nShuffles;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.thresholdFlag = 1;
analysisConfig.thresholdPct = 1;
analysisConfig.nMinNeurons = 25;
analysisConfig.useSubsampling = opts.useSubsampling;
analysisConfig.nSubsamples = opts.nSubsamples;
analysisConfig.nNeuronsSubsample = opts.nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
analysisConfig.normalizeMetrics = opts.enablePermutations;
analysisConfig.powerLawFitMethod = opts.powerLawFitMethod;
analysisConfig.gofThreshold = opts.gofThreshold;
analysisConfig.runClausetPlpva = false;
analysisConfig.clausetPlfitPath = clausetPlfitPath;
analysisConfig.plfit2023Path = plfit2023Path;
end

function sessionDuration = get_session_collect_duration(dataStruct, opts)
% GET_SESSION_COLLECT_DURATION - Actual loaded collect window length (s)

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
    && isfield(dataStruct.spikeData, 'collectStart')
  sessionDuration = dataStruct.spikeData.collectEnd - dataStruct.spikeData.collectStart;
elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
    && ~isempty(dataStruct.opts.collectEnd)
  collectStart = 0;
  if isfield(dataStruct.opts, 'collectStart') && ~isempty(dataStruct.opts.collectStart)
    collectStart = dataStruct.opts.collectStart;
  end
  sessionDuration = dataStruct.opts.collectEnd - collectStart;
elseif isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
  collectStart = opts.collectStart;
  if isempty(collectStart)
    collectStart = 0;
  end
  sessionDuration = max(dataStruct.spikeTimes) - collectStart;
else
  sessionDuration = opts.collectEnd - opts.collectStart;
  if isempty(sessionDuration)
    error('criticality_av_across_tasks:UnknownSessionDuration', ...
      'Could not determine session collect duration.');
  end
end
end

function commonAreas = resolve_areas_to_plot(plotAreas, areasToPlot, brainArea)
if isempty(areasToPlot) && ~isempty(brainArea)
  areasToPlot = {brainArea};
end
commonAreas = plotAreas;
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
end

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

  fig = figure(7000 + a);
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

      barLabels = get_session_bar_labels(typeData, numBars, sessionType);
      xticksCenters = [xticksCenters, xPos]; %#ok<AGROW>
      xtickLabels = [xtickLabels, barLabels]; %#ok<AGROW>

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
      xtickangle(ax, 45);
    end
    ylabel(ax, yLabelText, 'Interpreter', av_ylabel_interpreter(yLabelText));
    title(ax, sprintf('%s — %s', areaName, yLabelText), ...
      'Interpreter', av_ylabel_interpreter(yLabelText));
    apply_buffered_ylim(ax, yValuesForLim);
    grid(ax, 'on');
    hold(ax, 'off');

    if plotShuffledMean && panelRow == 1 && panelCol == 1
      legend(ax, {'session', 'shuffled mean'}, 'Location', 'best');
    end
  end

  collectTag = format_av_collect_window_tag(collectStart, collectEnd);
  if ~isempty(brainArea)
    titleStr = sprintf('Avalanche criticality — %s [%s]', brainArea, collectTag);
  else
    titleStr = sprintf('Avalanche criticality — %s [%s]', areaName, collectTag);
  end
  sgtitle(fig, titleStr, 'FontWeight', 'bold');

  if ~isempty(brainArea)
    plotBase = sprintf('criticality_av_across_tasks_%s_%s', brainArea, collectTag);
  else
    plotBase = sprintf('criticality_av_across_tasks_%s_%s', areaName, collectTag);
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
barLabels = cellfun(@(s) truncate_session_xtick_label(s, 7), barLabels, ...
  'UniformOutput', false);
end

function label = truncate_session_xtick_label(label, maxChars)
% TRUNCATE_SESSION_XTICK_LABEL - Cap session-name tick text length
if nargin < 2 || isempty(maxChars)
  maxChars = 7;
end
label = char(label);
if numel(label) > maxChars
  label = label(1:maxChars);
end
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
% paramSD is the crackling exponent 1/σνz from ⟨S⟩~T^γ (size_given_duration).
% Shuffled means are shown only on raw panels (same axes as session values).

metricPlotSpec = {
  1, 1, 'dcc', 'dcc (raw)', 'dccPermutedMean', true
  1, 2, 'dccNormalized', 'dcc (normalized)', '', false
  1, 3, 'kappa', 'kappa (raw)', 'kappaPermutedMean', true
  1, 4, 'kappaNormalized', 'kappa (normalized)', '', false
  2, 1, 'decades', 'decades', 'decadesPermutedMean', true
  2, 2, 'tau', 'tau', 'tauPermutedMean', true
  2, 3, 'alpha', 'alpha', 'alphaPermutedMean', true
  2, 4, 'paramSD', '1/\sigma\nu z (crackling)', 'paramSDPermutedMean', true
  };
end

function interp = av_ylabel_interpreter(labelText)
% AV_YLABEL_INTERPRETER - tex when label uses TeX escapes
if contains(labelText, '\')
  interp = 'tex';
else
  interp = 'none';
end
end

function plot_av_crackling_relation_across_tasks(plotData, areasToPlot, sessionTypes, ...
    collectStart, collectEnd, paths, brainArea)
% PLOT_AV_CRACKLING_RELATION_ACROSS_TASKS - paramSD vs γ_pred + dcc histogram
%
% Variables:
%   plotData      - Aggregated AV plotData (tau, alpha, paramSD, dcc cells)
%   areasToPlot   - Areas to figure
%   sessionTypes  - Session types (point / bar color)
%
% Goal:
%   Companion crackling-noise diagnostics: (1) measured 1/σνz vs γ_pred with
%   identity line; (2) session dcc distribution by task type.

if nargin < 7
  brainArea = '';
end

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end
typeColors = lines(max(numel(sessionTypes), 3));
collectTag = format_av_collect_window_tag(collectStart, collectEnd);

for a = 1:numel(areasToPlot)
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx)
    continue;
  end

  fig = figure(7100 + a);
  clf(fig);
  position_figure_full_monitor(fig);
  axScatter = subplot(1, 2, 1, 'Parent', fig);
  axDcc = subplot(1, 2, 2, 'Parent', fig);
  hold(axScatter, 'on');
  hold(axDcc, 'on');
  legendHandles = gobjects(0);
  legendLabels = {};
  nPlotted = 0;
  dccByType = cell(1, numel(sessionTypes));

  for t = 1:numel(sessionTypes)
    sessionType = sessionTypes{t};
    typeKey = matlab.lang.makeValidName(sessionType);
    if ~isfield(plotData.byType, typeKey)
      continue;
    end
    typeData = plotData.byType.(typeKey);
    tauVals = get_av_type_metric_vector(typeData, 'tau', areaIdx);
    alphaVals = get_av_type_metric_vector(typeData, 'alpha', areaIdx);
    paramSDVals = get_av_type_metric_vector(typeData, 'paramSD', areaIdx);
    dccVals = get_av_type_metric_vector(typeData, 'dcc', areaIdx);
    if isempty(tauVals) || isempty(alphaVals) || isempty(paramSDVals)
      continue;
    end
    n = min([numel(tauVals), numel(alphaVals), numel(paramSDVals)]);
    tauVals = tauVals(1:n);
    alphaVals = alphaVals(1:n);
    paramSDVals = paramSDVals(1:n);
    if numel(dccVals) >= n
      dccVals = dccVals(1:n);
    else
      dccVals = nan(1, n);
    end
    gammaPred = nan(size(tauVals));
    validTau = isfinite(tauVals) & isfinite(alphaVals) & (tauVals > 1);
    gammaPred(validTau) = (alphaVals(validTau) - 1) ./ (tauVals(validTau) - 1);
    valid = isfinite(gammaPred) & isfinite(paramSDVals);
    if any(valid)
      hSc = scatter(axScatter, gammaPred(valid), paramSDVals(valid), 60, ...
        'filled', 'MarkerFaceColor', typeColors(t, :), 'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.5, 'DisplayName', sessionType);
      legendHandles(end + 1) = hSc; %#ok<AGROW>
      legendLabels{end + 1} = sessionType; %#ok<AGROW>
      nPlotted = nPlotted + sum(valid);
      % Prefer stored dcc; else |γ_pred - paramSD|
      dccUse = dccVals;
      missingDcc = valid & ~isfinite(dccUse);
      dccUse(missingDcc) = abs(gammaPred(missingDcc) - paramSDVals(missingDcc));
      dccByType{t} = dccUse(valid & isfinite(dccUse));
    end
  end

  if nPlotted == 0
    close(fig);
    fprintf('Skipping crackling relation for %s: no finite sessions.\n', areaName);
    continue;
  end

  add_av_identity_line(axScatter);
  xlabel(axScatter, '(\alpha-1)/(\tau-1)', 'Interpreter', 'tex');
  ylabel(axScatter, '1/\sigma\nu z (paramSD)', 'Interpreter', 'tex');
  title(axScatter, sprintf('%s — crackling relation', areaName), 'Interpreter', 'none');
  if ~isempty(legendHandles)
    legend(axScatter, legendHandles, legendLabels, 'Location', 'best');
  end
  grid(axScatter, 'on');
  axis(axScatter, 'square');
  hold(axScatter, 'off');

  % dcc histogram / grouped counts by session type
  allDcc = [];
  for t = 1:numel(sessionTypes)
    allDcc = [allDcc, dccByType{t}(:)']; %#ok<AGROW>
  end
  if ~isempty(allDcc)
    nBins = max(5, min(15, ceil(sqrt(numel(allDcc)))));
    edges = linspace(min(allDcc), max(allDcc) + eps, nBins + 1);
    binCenters = edges(1:end-1) + diff(edges) / 2;
    barWidth = 0.8 * mean(diff(edges)) / max(numel(sessionTypes), 1);
    for t = 1:numel(sessionTypes)
      vals = dccByType{t};
      if isempty(vals)
        continue;
      end
      counts = histcounts(vals, edges);
      xPos = binCenters + (t - (numel(sessionTypes) + 1) / 2) * barWidth;
      bar(axDcc, xPos, counts, barWidth, 'FaceColor', typeColors(t, :), ...
        'EdgeColor', 'k', 'LineWidth', 0.5, 'DisplayName', sessionTypes{t});
    end
    xlabel(axDcc, 'dcc = |\gamma_{pred} - 1/\sigma\nu z|', 'Interpreter', 'tex');
    ylabel(axDcc, 'Session count');
    title(axDcc, sprintf('%s — dcc distribution', areaName), 'Interpreter', 'none');
    legend(axDcc, 'Location', 'best');
    grid(axDcc, 'on');
  end
  hold(axDcc, 'off');

  if ~isempty(brainArea)
    titleStr = sprintf('Crackling diagnostics — %s [%s]', brainArea, collectTag);
    plotBase = sprintf('criticality_av_crackling_relation_%s_%s', brainArea, collectTag);
  else
    titleStr = sprintf('Crackling diagnostics — %s [%s]', areaName, collectTag);
    plotBase = sprintf('criticality_av_crackling_relation_%s_%s', areaName, collectTag);
  end
  if numel(areasToPlot) > 1
    plotBase = sprintf('%s_area%s', plotBase, areaName);
  end
  sgtitle(fig, titleStr, 'FontWeight', 'bold', 'Interpreter', 'none');

  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved crackling relation: %s\n', fullfile(saveDir, plotBase));
end
end

function vals = get_av_type_metric_vector(typeData, fieldName, areaIdx)
% GET_AV_TYPE_METRIC_VECTOR - Pull one area's metric vector from typeData
vals = [];
if ~isfield(typeData, fieldName) || areaIdx > numel(typeData.(fieldName))
  return;
end
vals = typeData.(fieldName){areaIdx};
vals = vals(:)';
end

function add_av_identity_line(ax)
% ADD_AV_IDENTITY_LINE - y=x over finite scatter children
hold(ax, 'on');
sc = findobj(ax, 'Type', 'Scatter');
xAll = [];
yAll = [];
for i = 1:numel(sc)
  xAll = [xAll; sc(i).XData(:)]; %#ok<AGROW>
  yAll = [yAll; sc(i).YData(:)]; %#ok<AGROW>
end
valid = isfinite(xAll) & isfinite(yAll);
if ~any(valid)
  return;
end
lo = min([xAll(valid); yAll(valid)]);
hi = max([xAll(valid); yAll(valid)]);
pad = 0.05 * max(hi - lo, eps);
lim = [lo - pad, hi + pad];
xlim(ax, lim);
ylim(ax, lim);
plot(ax, lim, lim, 'k--', 'LineWidth', 1.25, 'HandleVisibility', 'off');
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

function tag = format_av_collect_window_tag(collectStart, collectEnd)
% FORMAT_AV_COLLECT_WINDOW_TAG - Filename/title fragment for collect window
if isempty(collectEnd)
  tag = sprintf('%.0f-full', collectStart);
else
  tag = sprintf('%.0f-%.0fs', collectStart, collectEnd);
end
end
