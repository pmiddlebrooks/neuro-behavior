%%
% Criticality AR (d2) Across Task Types — Engagement Split (Manuscript)
%
% Like criticality_ar_across_tasks.m, but for reach and interval sessions uses
% reach_criticality_metrics_engagement / interval_criticality_metrics_engagement
% (makePlots false) to obtain engaged vs non-engaged window d2 summaries.
% Spontaneous sessions use the standard all-window d2 pipeline.
%
% Variables (configure in this section):
%   sessionTypes   - Cell array of session types to include
%   dataSource     - 'spikes' or 'lfp'
%   collectStart   - Analysis window start (seconds from session onset)
%   collectEnd     - Analysis window end (seconds); [] = full session (default)
%   d2Window       - Non-overlapping window length (seconds)
%   brainArea, brainAreaCombinations, areasToPlot
%   runBatch, plotResults - Run analysis and/or figures independently
%   saveBatchResults, batchResultsFile - Save/load batch for plot-only reruns
%   useLog10D2, useSubsampling, nSubsamples, ...
%
% Goal:
%   Compare d2 across sessions grouped by task type. Reach/interval: engaged and
%   non-engaged bars side-by-side per session (+ shuffled mean). Spontaneous:
%   session mean vs shuffled (same as criticality_ar_across_tasks.m).

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
dataSource = 'spikes';

collectStart = 0;
collectEnd = [];

d2Window = 30;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};
runBatch = true;
plotResults = true;
saveBatchResults = true;
batchResultsFile = '';  % default: dropPath/criticality_manuscript/criticality_ar_across_tasks_engagement_batch.mat
useLog10D2 = true;

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.5;

analysisConfig = struct();
analysisConfig.slidingWindowSize = d2Window;
analysisConfig.stepSize = d2Window;
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
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 150;

plotConfig = fill_manuscript_plot_config();

% Paths

if isempty(batchResultsFile)
  batchResultsFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
    'criticality_ar_across_tasks_engagement_batch.mat');
end

fprintf('\n=== Criticality d2 Across Task Types (Engagement) ===\n');
fprintf('Collect window: [%.1f, %s] s\n', collectStart, format_collect_end_label(collectEnd));
fprintf('d2 windows: %.1f s, non-overlapping\n', d2Window);
fprintf('useLog10D2: %d\n', useLog10D2);
fprintf('Session types: %s\n', strjoin(sessionTypes, ', '));

%% Batch analysis
if runBatch
  sessionTable = build_engagement_session_table(sessionTypes);
  numSessions = size(sessionTable, 1);
  fprintf('Total sessions: %d\n', numSessions);
  if numSessions == 0
    error('No sessions found for the requested session types.');
  end

  batchResults = repmat(struct(), numSessions, 1);
  fprintf('\n=== Running d2 analysis ===\n');
  for s = 1:numSessions
    sessionType = sessionTable.sessionType{s};
    sessionName = sessionTable.sessionName{s};
    subjectName = sessionTable.subjectName{s};

    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('Session %d/%d [%s]: %s\n', s, numSessions, sessionType, sessionName);

    batchResults(s).sessionType = sessionType;
    batchResults(s).sessionName = sessionName;
    batchResults(s).subjectName = subjectName;
    batchResults(s).label = sessionTable.label{s};
    batchResults(s).success = false;
    batchResults(s).useEngagement = is_engagement_session_type(sessionType);
    batchResults(s).results = [];
    batchResults(s).d2Split = [];

    try
      if batchResults(s).useEngagement
        engOpts = build_engagement_batch_opts(opts, analysisConfig, brainArea, ...
          brainAreaCombinations, plotConfig, sessionType);
        if strcmpi(sessionType, 'reach')
          engOut = reach_criticality_metrics_engagement(sessionName, engOpts);
        else
          engOut = interval_criticality_metrics_engagement(subjectName, sessionName, engOpts);
        end
        batchResults(s).d2Split = engOut.d2;
        batchResults(s).results = engOut.arResultsD2;
        batchResults(s).success = ~isempty(engOut.d2) && ~isempty(engOut.arResultsD2);
      else
        loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectName);
        dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});
        [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
          dataStruct, brainArea, brainAreaCombinations);
        if ~areaOk
          fprintf('  Brain area "%s" not available; skipping.\n', brainArea);
          continue;
        end
        arResults = criticality_ar_analysis(dataStruct, analysisConfig);
        if ~isempty(brainArea)
          arResults = filter_ar_results_to_brain_area(arResults, brainArea);
          if isempty(arResults.areas)
            fprintf('  No results for brain area "%s"; skipping.\n', brainArea);
            continue;
          end
        end
        batchResults(s).results = arResults;
        batchResults(s).success = true;
      end
      fprintf('  Analysis completed.\n');
    catch ME
      fprintf('  Error: %s\n', ME.message);
      for st = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
      end
    end
  end
  plotData = aggregate_engagement_ar_metrics(batchResults, sessionTypes, useLog10D2);
  batchMeta = struct( ...
    'sessionTypes', {sessionTypes}, ...
    'useLog10D2', useLog10D2, ...
    'collectStart', collectStart, ...
    'collectEnd', collectEnd, ...
    'd2Window', d2Window, ...
    'brainArea', brainArea, ...
    'areasToPlot', {areasToPlot}, ...
    'plotConfig', plotConfig);
  if saveBatchResults
    save(batchResultsFile, 'batchResults', 'plotData', 'batchMeta', '-v7.3');
    fprintf('\nSaved batch results: %s\n', batchResultsFile);
  end
else
  if ~isfile(batchResultsFile)
    error('criticality_ar_across_tasks_engagement:NoBatchFile', ...
      'Batch file not found: %s. Set runBatch true to compute.', batchResultsFile);
  end
  loaded = load(batchResultsFile, 'batchResults', 'plotData', 'batchMeta');
  batchResults = loaded.batchResults;
  plotData = loaded.plotData;
  batchMeta = loaded.batchMeta;
  sessionTypes = batchMeta.sessionTypes;
  useLog10D2 = batchMeta.useLog10D2;
  collectStart = batchMeta.collectStart;
  collectEnd = batchMeta.collectEnd;
  d2Window = batchMeta.d2Window;
  brainArea = batchMeta.brainArea;
  areasToPlot = batchMeta.areasToPlot;
  if isfield(batchMeta, 'plotConfig') && ~isempty(batchMeta.plotConfig)
    plotConfig = fill_manuscript_plot_config(batchMeta.plotConfig);
  end
  fprintf('\nLoaded batch results: %s\n', batchResultsFile);
end

%% Plotting
if plotResults
  if isempty(plotData.areas)
    error('No d2 metrics in plotData. Re-run batch with runBatch true.');
  end

  if isempty(areasToPlot) && ~isempty(brainArea)
    areasToPlot = {brainArea};
  end
  commonAreas = plotData.areas;
  if ~isempty(areasToPlot)
    commonAreas = intersect(commonAreas, areasToPlot, 'stable');
    if isempty(commonAreas)
      error('None of areasToPlot are present in plotData.');
    end
  end

  fprintf('\n=== Plotting ===\n');
  fprintf('Areas: %s\n', strjoin(commonAreas, ', '));
  plot_engagement_ar_across_tasks(plotData, commonAreas, sessionTypes, collectStart, ...
    collectEnd, d2Window, paths, brainArea, useLog10D2, plotConfig);
end

fprintf('\n=== Done ===\n');

%% Local functions

function sessionTable = build_engagement_session_table(sessionTypes)
sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
labelCol = {};

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  entries = get_engagement_sessions_for_type(sessionType);
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

function entries = get_engagement_sessions_for_type(sessionType)
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
  otherwise
    error('Unknown sessionType: %s', sessionType);
end
end

function tf = is_engagement_session_type(sessionType)
tf = any(strcmpi(sessionType, {'reach', 'interval'}));
end

function engOpts = build_engagement_batch_opts(opts, analysisConfig, brainArea, ...
    brainAreaCombinations, plotConfig, sessionType)
% BUILD_ENGAGEMENT_BATCH_OPTS - Engagement-module opts for batch (d2 only, no plots)

if strcmpi(sessionType, 'reach')
  engOpts = reach_criticality_metrics_engagement();
else
  engOpts = interval_criticality_metrics_engagement();
end

engOpts.collectStart = opts.collectStart;
engOpts.collectEnd = opts.collectEnd;
engOpts.minFiringRate = opts.minFiringRate;
engOpts.maxFiringRate = opts.maxFiringRate;
engOpts.firingRateCheckTime = opts.firingRateCheckTime;
engOpts.dataSource = 'spikes';
engOpts.brainArea = brainArea;
engOpts.brainAreaCombinations = brainAreaCombinations;
engOpts.d2Window = analysisConfig.slidingWindowSize;
engOpts.useLog10D2 = analysisConfig.useLog10D2;
engOpts.nShufflesD2 = analysisConfig.nShuffles;
engOpts.useSubsampling = analysisConfig.useSubsampling;
engOpts.nSubsamples = analysisConfig.nSubsamples;
engOpts.nNeuronsSubsample = analysisConfig.nNeuronsSubsample;
engOpts.minNeuronsMultiple = analysisConfig.minNeuronsMultiple;
engOpts.nMinNeurons = analysisConfig.nMinNeurons;
engOpts.analyses = {'d2'};
engOpts.makePlots = false;
engOpts.saveFigure = false;
engOpts.plotConfig = plotConfig;

if strcmpi(sessionType, 'reach')
  engOpts.runD2AccuracyCorrelation = false;
  engOpts.runD2ReachRateCorrelation = false;
else
  engOpts.runD2TrialRateCorrelation = false;
end
end

function plotData = aggregate_engagement_ar_metrics(batchResults, sessionTypes, useLog10D2)
% AGGREGATE_ENGAGEMENT_AR_METRICS - Per-session metrics for engagement and total modes

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.useLog10D2 = useLog10D2;

metricFields = {...
  'd2Mean', 'd2Sem', 'd2ShuffleMean', 'd2ShuffleSem', ...
  'd2NormMean', 'd2NormSem', ...
  'd2EngagedMean', 'd2EngagedSem', 'd2NonEngagedMean', 'd2NonEngagedSem', ...
  'd2NormEngagedMean', 'd2NormEngagedSem', 'd2NormNonEngagedMean', 'd2NormNonEngagedSem'};

for s = 1:length(batchResults)
  if ~batchResults(s).success
    continue;
  end

  sessionType = batchResults(s).sessionType;
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_engagement_type_metrics(metricFields, 0);
    plotData.byType.(typeKey).useEngagement = is_engagement_session_type(sessionType);
  end
  typeData = plotData.byType.(typeKey);

  if batchResults(s).useEngagement
    d2Split = batchResults(s).d2Split;
    results = batchResults(s).results;
    areaNames = d2Split.areas;
    engagedIdx = 2;
    nonEngagedIdx = 3;
  else
    results = batchResults(s).results;
    areaNames = results.areas;
  end

  for a = 1:length(areaNames)
    areaName = areaNames{a};
    areaIdx = find(strcmp(plotData.areas, areaName), 1);
    if isempty(areaIdx)
      plotData.areas{end + 1} = areaName;
      areaIdx = numel(plotData.areas);
    end
    typeData = ensure_type_data_area_slots(typeData, metricFields, areaIdx);

    if batchResults(s).useEngagement
      summaryEng = summarize_vector_mean_sem(d2Split.d2{engagedIdx}{a});
      summaryNon = summarize_vector_mean_sem(d2Split.d2{nonEngagedIdx}{a});
      summaryNormEng = summarize_vector_mean_sem(d2Split.d2Normalized{engagedIdx}{a});
      summaryNormNon = summarize_vector_mean_sem(d2Split.d2Normalized{nonEngagedIdx}{a});
      shuffleSummary = summarize_session_d2_shuffle(results, a, useLog10D2);

      typeData.d2EngagedMean{areaIdx}(end + 1) = summaryEng.mean;
      typeData.d2EngagedSem{areaIdx}(end + 1) = summaryEng.sem;
      typeData.d2NonEngagedMean{areaIdx}(end + 1) = summaryNon.mean;
      typeData.d2NonEngagedSem{areaIdx}(end + 1) = summaryNon.sem;
      typeData.d2NormEngagedMean{areaIdx}(end + 1) = summaryNormEng.mean;
      typeData.d2NormEngagedSem{areaIdx}(end + 1) = summaryNormEng.sem;
      typeData.d2NormNonEngagedMean{areaIdx}(end + 1) = summaryNormNon.mean;
      typeData.d2NormNonEngagedSem{areaIdx}(end + 1) = summaryNormNon.sem;
      typeData.d2ShuffleMean{areaIdx}(end + 1) = shuffleSummary.mean;
      typeData.d2ShuffleSem{areaIdx}(end + 1) = shuffleSummary.sem;
    else
      summary = summarize_session_d2_windows(results, a, useLog10D2);
      typeData.d2Mean{areaIdx}(end + 1) = summary.d2Mean;
      typeData.d2Sem{areaIdx}(end + 1) = summary.d2Sem;
      typeData.d2ShuffleMean{areaIdx}(end + 1) = summary.d2ShuffleMean;
      typeData.d2ShuffleSem{areaIdx}(end + 1) = summary.d2ShuffleSem;
      typeData.d2NormMean{areaIdx}(end + 1) = summary.d2NormMean;
      typeData.d2NormSem{areaIdx}(end + 1) = summary.d2NormSem;
    end
  end

  typeData.sessionLabels{end + 1} = batchResults(s).label;
  typeData.sessionNames{end + 1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
end
end

function stats = summarize_vector_mean_sem(vec)
stats = struct('mean', nan, 'sem', nan);
vec = vec(:);
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

function summary = summarize_session_d2_shuffle(results, areaIdx, useLog10D2)
summary = struct('mean', nan, 'sem', nan);
if ~isfield(results, 'd2Permuted') || areaIdx > numel(results.d2Permuted) ...
    || isempty(results.d2Permuted{areaIdx})
  return;
end
perWindowShuffleMean = get_per_window_shuffle_mean_d2(results, areaIdx, useLog10D2);
stats = summarize_vector_mean_sem(perWindowShuffleMean);
summary.mean = stats.mean;
summary.sem = stats.sem;
end

function summary = summarize_session_d2_windows(results, areaIdx, useLog10D2)
summary = struct('d2Mean', nan, 'd2Sem', nan, 'd2ShuffleMean', nan, ...
  'd2ShuffleSem', nan, 'd2NormMean', nan, 'd2NormSem', nan);

if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
  return;
end

d2Vec = results.d2{areaIdx}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
stats = summarize_vector_mean_sem(d2Vec);
summary.d2Mean = stats.mean;
summary.d2Sem = stats.sem;

if isfield(results, 'd2Normalized') && areaIdx <= numel(results.d2Normalized) ...
    && ~isempty(results.d2Normalized{areaIdx})
  d2NormVec = results.d2Normalized{areaIdx}(:);
  if useLog10D2
    d2NormVec = log10_safe_numeric(d2NormVec);
  end
  statsNorm = summarize_vector_mean_sem(d2NormVec);
  summary.d2NormMean = statsNorm.mean;
  summary.d2NormSem = statsNorm.sem;
end

shuffleStats = summarize_session_d2_shuffle(results, areaIdx, useLog10D2);
summary.d2ShuffleMean = shuffleStats.mean;
summary.d2ShuffleSem = shuffleStats.sem;
end

function y = log10_safe_numeric(x)
validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function typeData = ensure_type_data_area_slots(typeData, metricFields, areaIdx)
% ENSURE_TYPE_DATA_AREA_SLOTS - Grow per-area metric cells before indexed write

for m = 1:length(metricFields)
  fieldName = metricFields{m};
  if ~isfield(typeData, fieldName) || isempty(typeData.(fieldName))
    typeData.(fieldName) = {};
  end
  while numel(typeData.(fieldName)) < areaIdx
    typeData.(fieldName){end + 1} = [];
  end
end
end

function typeData = init_engagement_type_metrics(metricFields, numAreas)
typeData = struct();
typeData.sessionLabels = {};
typeData.sessionNames = {};
typeData.useEngagement = false;
for m = 1:length(metricFields)
  typeData.(metricFields{m}) = cell(1, max(numAreas, 0));
  for a = 1:max(numAreas, 0)
    typeData.(metricFields{m}){a} = [];
  end
end
end

function results = filter_ar_results_to_brain_area(results, brainArea)
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
end

function plot_engagement_ar_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, ...
    collectEnd, d2Window, paths, brainArea, useLog10D2, plotConfig)
% PLOT_ENGAGEMENT_AR_ACROSS_TASKS - Raw and normalized d2 bar plots by session type

if nargin < 10 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

if useLog10D2
  rawYLabel = 'log_{10}(d2) (mean \pm SEM across windows)';
  normYLabel = 'log_{10}(d2 normalized) (mean \pm SEM across windows)';
  rawTitlePlain = 'log10(d2)';
  normTitlePlain = 'log10(d2 normalized)';
  labelInterpreter = 'tex';
else
  rawYLabel = 'd2 (mean \pm SEM across windows)';
  normYLabel = 'd2 normalized (mean \pm SEM across windows)';
  rawTitlePlain = 'd2';
  normTitlePlain = 'd2 normalized';
  labelInterpreter = 'none';
end

shuffleColor = manuscript_plot_colors().shuffled;
saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

collectTag = format_collect_window_tag(collectStart, collectEnd);

for a = 1:length(areasToPlot)
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx) || ~area_has_engagement_plot_data(plotData, sessionTypes, areaIdx)
    continue;
  end

  figRaw = figure('Color', 'w', 'Name', sprintf('d2 engagement raw — %s', areaName));
  position_figure_full_monitor(figRaw);
  axRaw = axes(figRaw);
  plot_d2_raw_engagement_bars(axRaw, plotData, sessionTypes, areaIdx, shuffleColor, plotConfig);
  apply_manuscript_axes_style(axRaw, plotConfig, '', rawYLabel, '', labelInterpreter);
  sgtitle(figRaw, sprintf('%s (raw, engagement) — %s | %s | %.0fs windows', ...
    rawTitlePlain, format_brain_area_title(brainArea, areaName), collectTag, d2Window), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none', 'FontWeight', 'bold');

  plotBaseRaw = make_engagement_ar_plot_basename('criticality_ar_across_tasks_engagement_raw', ...
    areaName, brainArea, d2Window, collectStart, collectEnd, useLog10D2);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.png']), 'Resolution', 300);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseRaw));

  figNorm = figure('Color', 'w', 'Name', sprintf('d2 engagement normalized — %s', areaName));
  position_figure_full_monitor(figNorm);
  axNorm = axes(figNorm);
  plot_d2_normalized_engagement_bars(axNorm, plotData, sessionTypes, areaIdx, plotConfig);
  apply_manuscript_axes_style(axNorm, plotConfig, '', normYLabel, '', labelInterpreter);
  sgtitle(figNorm, sprintf('%s (normalized, engagement) — %s | %s | %.0fs windows', ...
    normTitlePlain, format_brain_area_title(brainArea, areaName), collectTag, d2Window), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none', 'FontWeight', 'bold');

  plotBaseNorm = make_engagement_ar_plot_basename('criticality_ar_across_tasks_engagement_normalized', ...
    areaName, brainArea, d2Window, collectStart, collectEnd, useLog10D2);
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.png']), 'Resolution', 300);
  exportgraphics(figNorm, fullfile(saveDir, [plotBaseNorm, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseNorm));
end

fprintf('\nAll figures saved to %s\n', saveDir);
end

function plot_d2_raw_engagement_bars(ax, plotData, sessionTypes, areaIdx, shuffleColor, plotConfig)
% PLOT_D2_RAW_ENGAGEMENT_BARS - Engaged/non-engaged pairs or total d2 + shuffle

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendHandles = [];
legendLabels = {};

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  taskColor = colors_for_tasks(sessionType);

  if isfield(typeData, 'useEngagement') && typeData.useEngagement
    engagedMeans = typeData.d2EngagedMean{areaIdx};
    engagedSems = typeData.d2EngagedSem{areaIdx};
    nonMeans = typeData.d2NonEngagedMean{areaIdx};
    nonSems = typeData.d2NonEngagedSem{areaIdx};
    shuffleMeans = typeData.d2ShuffleMean{areaIdx};
    shuffleSems = typeData.d2ShuffleSem{areaIdx};
    numSessions = numel(engagedMeans);
    if numSessions == 0
      continue;
    end

    for i = 1:numSessions
      xBase = xCursor + i;
      xEng = xBase - 0.12;
      xNon = xBase + 0.12;
      xShuf = xBase + 0.34;

      hEng = errorbar(ax, xEng, engagedMeans(i), engagedSems(i), 'o', ...
        'Color', taskColor, 'MarkerFaceColor', taskColor, ...
        'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
        'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
      hNon = errorbar(ax, xNon, nonMeans(i), nonSems(i), 'o', ...
        'Color', taskColor, 'MarkerFaceColor', 'none', ...
        'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
        'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
      if isfinite(shuffleMeans(i))
        hShuf = errorbar(ax, xShuf, shuffleMeans(i), shuffleSems(i), 's', ...
          'Color', shuffleColor, 'MarkerFaceColor', shuffleColor, ...
          'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
          'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
          'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
        if isempty(legendHandles) || numel(legendHandles) < 3
          legendHandles = [hEng, hNon, hShuf]; %#ok<AGROW>
          legendLabels = {'engaged d2', 'non-engaged d2', 'shuffled mean'};
        end
      elseif isempty(legendHandles)
        legendHandles = [hEng, hNon]; %#ok<AGROW>
        legendLabels = {'engaged d2', 'non-engaged d2'};
      end

      xticksCenters(end + 1) = xBase; %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xCursor + numSessions + 1.2;
  else
    d2Means = typeData.d2Mean{areaIdx};
    d2Sems = typeData.d2Sem{areaIdx};
    shuffleMeans = typeData.d2ShuffleMean{areaIdx};
    shuffleSems = typeData.d2ShuffleSem{areaIdx};
    numSessions = numel(d2Means);
    if numSessions == 0
      continue;
    end

    xPos = xCursor + (1:numSessions);
    hData = errorbar(ax, xPos, d2Means, d2Sems, 'o', 'Color', taskColor, ...
      'MarkerFaceColor', taskColor, 'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
      'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
    if any(isfinite(shuffleMeans))
      semPlot = shuffleSems;
      semPlot(~isfinite(semPlot)) = 0;
      hShuf = errorbar(ax, xPos + 0.22, shuffleMeans, semPlot, 's', ...
        'Color', shuffleColor, 'MarkerFaceColor', shuffleColor, ...
        'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
        'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
        'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
      if isempty(legendHandles)
        legendHandles = [hData, hShuf];
        legendLabels = {'session d2', 'shuffled mean'};
      end
    end

    for i = 1:numSessions
      xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xPos(end) + 1.5;
  end
end

if ~isempty(xticksCenters)
  set(ax, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45);
end
grid(ax, 'off');
if ~isempty(legendHandles)
  legend(ax, legendHandles, legendLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize);
end
hold(ax, 'off');
end

function plot_d2_normalized_engagement_bars(ax, plotData, sessionTypes, areaIdx, plotConfig)
% PLOT_D2_NORMALIZED_ENGAGEMENT_BARS - Normalized d2 bars by engagement class

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
  taskColor = colors_for_tasks(sessionType);

  if isfield(typeData, 'useEngagement') && typeData.useEngagement
    engagedMeans = typeData.d2NormEngagedMean{areaIdx};
    engagedSems = typeData.d2NormEngagedSem{areaIdx};
    nonMeans = typeData.d2NormNonEngagedMean{areaIdx};
    nonSems = typeData.d2NormNonEngagedSem{areaIdx};
    numSessions = numel(engagedMeans);
    if numSessions == 0
      continue;
    end

    for i = 1:numSessions
      xBase = xCursor + i;
      errorbar(ax, xBase - 0.12, engagedMeans(i), engagedSems(i), 'o', ...
        'Color', taskColor, 'MarkerFaceColor', taskColor, ...
        'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
        'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
      errorbar(ax, xBase + 0.12, nonMeans(i), nonSems(i), 'o', ...
        'Color', taskColor, 'MarkerFaceColor', 'none', ...
        'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
        'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
      xticksCenters(end + 1) = xBase; %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xCursor + numSessions + 1.2;
  else
    normMeans = typeData.d2NormMean{areaIdx};
    normSems = typeData.d2NormSem{areaIdx};
    numSessions = numel(normMeans);
    if numSessions == 0
      continue;
    end
    xPos = xCursor + (1:numSessions);
    errorbar(ax, xPos, normMeans, normSems, 'o', 'Color', taskColor, ...
      'MarkerFaceColor', taskColor, 'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
      'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize);
    for i = 1:numSessions
      xticksCenters(end + 1) = xPos(i); %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xPos(end) + 1.5;
  end
end

if ~isempty(xticksCenters)
  set(ax, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45);
end
grid(ax, 'off');
hold(ax, 'off');
end

function label = get_session_bar_label(typeData, sessionIdx, sessionType)
if isfield(typeData, 'sessionNames') && numel(typeData.sessionNames) >= sessionIdx
  label = char(typeData.sessionNames{sessionIdx});
elseif isfield(typeData, 'sessionLabels') && numel(typeData.sessionLabels) >= sessionIdx
  label = char(typeData.sessionLabels{sessionIdx});
else
  label = sessionType;
end
end

function hasData = area_has_engagement_plot_data(plotData, sessionTypes, areaIdx)
hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'useEngagement') && typeData.useEngagement
    if isfield(typeData, 'd2EngagedMean') && areaIdx <= numel(typeData.d2EngagedMean) ...
        && ~isempty(typeData.d2EngagedMean{areaIdx})
      hasData = true;
      return;
    end
  elseif isfield(typeData, 'd2Mean') && areaIdx <= numel(typeData.d2Mean) ...
      && ~isempty(typeData.d2Mean{areaIdx})
    hasData = true;
    return;
  end
end
end

function plotBase = make_engagement_ar_plot_basename(prefix, areaName, brainArea, ...
    d2Window, collectStart, collectEnd, useLog10D2)
collectTag = format_collect_window_tag(collectStart, collectEnd);
if ~isempty(brainArea)
  plotBase = sprintf('%s_%s_win%.0fs_%s', prefix, brainArea, d2Window, collectTag);
else
  plotBase = sprintf('%s_%s_win%.0fs_%s', prefix, areaName, d2Window, collectTag);
end
if useLog10D2
  plotBase = [plotBase, '_log10'];
end
end

function tag = format_collect_window_tag(collectStart, collectEnd)
if isempty(collectEnd)
  tag = sprintf('%.0f-full', collectStart);
else
  tag = sprintf('%.0f-%.0f', collectStart, collectEnd);
end
end

function label = format_collect_end_label(collectEnd)
if isempty(collectEnd)
  label = 'full';
else
  label = sprintf('%.1f', collectEnd);
end
end

function titleText = format_brain_area_title(brainArea, areaName)
if ~isempty(brainArea)
  titleText = brainArea;
else
  titleText = areaName;
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
