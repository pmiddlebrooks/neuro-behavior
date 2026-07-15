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
%   useLog10D2, useViolinPlots - Bar means (default) or per-window violin distributions
%   useSubsampling, nSubsamples, ...
%
% Goal:
%   Compare d2 across sessions grouped by task type. Reach/interval: engaged and
%   non-engaged side-by-side per session (+ shuffled mean in bar mode). Optional
%   violin mode shows window distributions with means. Spontaneous: session total.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
sessionTypes = {'interval', 'reach'};
dataSource = 'spikes';

collectStart = 0;
collectEnd = [];

d2Window = 30;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};
runBatch = true;
plotResults = true;
saveBatchResults = false;
batchResultsFile = '';  % default: dropPath/criticality_manuscript/criticality_ar_across_tasks_engagement_batch.mat
useLog10D2 = true;
useViolinPlots = false;

useSubsampling = true;
nSubsamples = 25;
nNeuronsSubsample = 50;
minNeuronsMultiple = 1.2;

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
fprintf('useViolinPlots: %d\n', useViolinPlots);
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
  if exist('batchResults', 'var') && ~isempty(batchResults)
    plotData = aggregate_engagement_ar_metrics(batchResults, sessionTypes, useLog10D2);
  end
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
    collectEnd, d2Window, paths, brainArea, useLog10D2, useViolinPlots, plotConfig);
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
valueFields = {...
  'd2Values', 'd2NormValues', ...
  'd2EngagedValues', 'd2NonEngagedValues', ...
  'd2NormEngagedValues', 'd2NormNonEngagedValues'};

for s = 1:length(batchResults)
  if ~batchResults(s).success
    continue;
  end

  sessionType = batchResults(s).sessionType;
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_engagement_type_metrics(metricFields, valueFields, 0);
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
    typeData = ensure_type_data_area_slots(typeData, metricFields, valueFields, areaIdx);

    if batchResults(s).useEngagement
      % d2Split values are already on the plot scale (log10 applied in split_* when useLog10D2)
      onPlotScale = useLog10D2;
      summaryEng = summarize_d2_vector_for_plot(d2Split.d2{engagedIdx}{a}, useLog10D2, onPlotScale);
      summaryNon = summarize_d2_vector_for_plot(d2Split.d2{nonEngagedIdx}{a}, useLog10D2, onPlotScale);
      summaryNormEng = summarize_d2_vector_for_plot(d2Split.d2Normalized{engagedIdx}{a}, useLog10D2, onPlotScale);
      summaryNormNon = summarize_d2_vector_for_plot(d2Split.d2Normalized{nonEngagedIdx}{a}, useLog10D2, onPlotScale);
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
      typeData.d2EngagedValues{areaIdx}{end + 1} = get_d2_plot_vector(d2Split.d2{engagedIdx}{a}, useLog10D2, onPlotScale);
      typeData.d2NonEngagedValues{areaIdx}{end + 1} = get_d2_plot_vector(d2Split.d2{nonEngagedIdx}{a}, useLog10D2, onPlotScale);
      typeData.d2NormEngagedValues{areaIdx}{end + 1} = get_d2_plot_vector(d2Split.d2Normalized{engagedIdx}{a}, useLog10D2, onPlotScale);
      typeData.d2NormNonEngagedValues{areaIdx}{end + 1} = get_d2_plot_vector(d2Split.d2Normalized{nonEngagedIdx}{a}, useLog10D2, onPlotScale);
    else
      summary = summarize_session_d2_windows(results, a, useLog10D2);
      typeData.d2Mean{areaIdx}(end + 1) = summary.d2Mean;
      typeData.d2Sem{areaIdx}(end + 1) = summary.d2Sem;
      typeData.d2ShuffleMean{areaIdx}(end + 1) = summary.d2ShuffleMean;
      typeData.d2ShuffleSem{areaIdx}(end + 1) = summary.d2ShuffleSem;
      typeData.d2NormMean{areaIdx}(end + 1) = summary.d2NormMean;
      typeData.d2NormSem{areaIdx}(end + 1) = summary.d2NormSem;
      typeData.d2Values{areaIdx}{end + 1} = get_d2_plot_vector(results.d2{a}, useLog10D2);
      if isfield(results, 'd2Normalized') && a <= numel(results.d2Normalized) ...
          && ~isempty(results.d2Normalized{a})
        typeData.d2NormValues{areaIdx}{end + 1} = get_d2_plot_vector(results.d2Normalized{a}, useLog10D2);
      else
        typeData.d2NormValues{areaIdx}{end + 1} = [];
      end
    end
  end

  typeData.sessionLabels{end + 1} = batchResults(s).label;
  typeData.sessionNames{end + 1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
end
end

function vec = get_d2_plot_vector(rawVec, useLog10D2, alreadyOnPlotScale)
% GET_D2_PLOT_VECTOR - Finite per-window d2 values for plotting (optional log10)
%
% Variables:
%   rawVec            - Per-window values (linear d2 or already log10'd)
%   useLog10D2        - Request log10 display scale
%   alreadyOnPlotScale - If true, skip log10 (e.g. d2Split from engagement modules)

if nargin < 3 || isempty(alreadyOnPlotScale)
  alreadyOnPlotScale = false;
end
vec = rawVec(:);
vec = vec(isfinite(vec));
if useLog10D2 && ~alreadyOnPlotScale
  vec = log10_safe_numeric(vec);
  vec = vec(isfinite(vec));
end
end

function stats = summarize_d2_vector_for_plot(rawVec, useLog10D2, alreadyOnPlotScale)
% SUMMARIZE_D2_VECTOR_FOR_PLOT - Mean/SEM on the same scale as violin/bar plots

if nargin < 3 || isempty(alreadyOnPlotScale)
  alreadyOnPlotScale = false;
end
stats = summarize_vector_mean_sem(get_d2_plot_vector(rawVec, useLog10D2, alreadyOnPlotScale));
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

function typeData = ensure_type_data_area_slots(typeData, metricFields, valueFields, areaIdx)
% ENSURE_TYPE_DATA_AREA_SLOTS - Grow per-area metric and value cells before write

for m = 1:length(metricFields)
  fieldName = metricFields{m};
  if ~isfield(typeData, fieldName) || isempty(typeData.(fieldName))
    typeData.(fieldName) = {};
  end
  while numel(typeData.(fieldName)) < areaIdx
    typeData.(fieldName){end + 1} = [];
  end
end

for m = 1:length(valueFields)
  fieldName = valueFields{m};
  if ~isfield(typeData, fieldName) || isempty(typeData.(fieldName))
    typeData.(fieldName) = {};
  end
  while numel(typeData.(fieldName)) < areaIdx
    typeData.(fieldName){end + 1} = {};
  end
end
end

function typeData = init_engagement_type_metrics(metricFields, valueFields, numAreas)
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
for m = 1:length(valueFields)
  typeData.(valueFields{m}) = cell(1, max(numAreas, 0));
  for a = 1:max(numAreas, 0)
    typeData.(valueFields{m}){a} = {};
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
    collectEnd, d2Window, paths, brainArea, useLog10D2, useViolinPlots, plotConfig)
% PLOT_ENGAGEMENT_AR_ACROSS_TASKS - Raw and normalized d2 across sessions

if nargin < 10 || isempty(useViolinPlots)
  useViolinPlots = false;
end
if nargin < 11 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

if useViolinPlots && ~plot_data_has_violin_vectors(plotData, sessionTypes)
  warning('criticality_ar_across_tasks_engagement:NoViolinData', ...
    'Per-window d2 vectors missing in plotData; re-run batch (runBatch true). Using bar plots.');
  useViolinPlots = false;
end

if useLog10D2
  if useViolinPlots
    rawYLabel = 'log_{10}(d2) per window';
    normYLabel = 'log_{10}(d2 normalized) per window';
  else
    rawYLabel = 'log_{10}(d2) (mean \pm SEM across windows)';
    normYLabel = 'log_{10}(d2 normalized) (mean \pm SEM across windows)';
  end
  rawTitlePlain = 'log10(d2)';
  normTitlePlain = 'log10(d2 normalized)';
  labelInterpreter = 'tex';
else
  if useViolinPlots
    rawYLabel = 'd2 per window';
    normYLabel = 'd2 normalized per window';
  else
    rawYLabel = 'd2 (mean \pm SEM across windows)';
    normYLabel = 'd2 normalized (mean \pm SEM across windows)';
  end
  rawTitlePlain = 'd2';
  normTitlePlain = 'd2 normalized';
  labelInterpreter = 'none';
end

shuffleColor = manuscript_plot_colors().shuffled;
nonEngagedColor = [0.28, 0.28, 0.28];
saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

collectTag = format_collect_window_tag(collectStart, collectEnd);
if useViolinPlots
  rawPrefix = 'criticality_ar_across_tasks_engagement_raw_violin';
  normPrefix = 'criticality_ar_across_tasks_engagement_normalized_violin';
else
  rawPrefix = 'criticality_ar_across_tasks_engagement_raw';
  normPrefix = 'criticality_ar_across_tasks_engagement_normalized';
end

for a = 1:length(areasToPlot)
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx) || ~area_has_engagement_plot_data(plotData, sessionTypes, areaIdx)
    continue;
  end

  figRaw = figure('Color', 'w', 'Name', sprintf('d2 engagement raw — %s', areaName));
  position_figure_full_monitor(figRaw);
  axRaw = axes(figRaw);
  if useViolinPlots
    plot_d2_raw_engagement_violins(axRaw, plotData, sessionTypes, areaIdx, ...
      nonEngagedColor, plotConfig);
  else
    plot_d2_raw_engagement_bars(axRaw, plotData, sessionTypes, areaIdx, shuffleColor, plotConfig);
  end
  apply_manuscript_axes_style(axRaw, plotConfig, '', rawYLabel, '', labelInterpreter);
  sgtitle(figRaw, sprintf('%s (raw, engagement) — %s | %s | %.0fs windows', ...
    rawTitlePlain, format_brain_area_title(brainArea, areaName), collectTag, d2Window), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none', 'FontWeight', 'bold');

  plotBaseRaw = make_engagement_ar_plot_basename(rawPrefix, ...
    areaName, brainArea, d2Window, collectStart, collectEnd, useLog10D2);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.png']), 'Resolution', 300);
  exportgraphics(figRaw, fullfile(saveDir, [plotBaseRaw, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseRaw));

  figNorm = figure('Color', 'w', 'Name', sprintf('d2 engagement normalized — %s', areaName));
  position_figure_full_monitor(figNorm);
  axNorm = axes(figNorm);
  if useViolinPlots
    plot_d2_normalized_engagement_violins(axNorm, plotData, sessionTypes, areaIdx, ...
      nonEngagedColor, plotConfig);
  else
    plot_d2_normalized_engagement_bars(axNorm, plotData, sessionTypes, areaIdx, plotConfig);
  end
  apply_manuscript_axes_style(axNorm, plotConfig, '', normYLabel, '', labelInterpreter);
  sgtitle(figNorm, sprintf('%s (normalized, engagement) — %s | %s | %.0fs windows', ...
    normTitlePlain, format_brain_area_title(brainArea, areaName), collectTag, d2Window), ...
    'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none', 'FontWeight', 'bold');

  plotBaseNorm = make_engagement_ar_plot_basename(normPrefix, ...
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

function plot_d2_raw_engagement_violins(ax, plotData, sessionTypes, areaIdx, nonEngagedColor, plotConfig)
% PLOT_D2_RAW_ENGAGEMENT_VIOLINS - Mirrored per-window d2 violins by engagement class

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendHandles = [];
legendLabels = {};
violinOpts = struct('width', 0.22, 'faceAlpha', 0.5, 'showMean', true, 'plotConfig', plotConfig);

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  taskColor = colors_for_tasks(sessionType);

  if isfield(typeData, 'useEngagement') && typeData.useEngagement
    numSessions = numel(typeData.d2EngagedMean{areaIdx});
    if numSessions == 0
      continue;
    end

    for i = 1:numSessions
      xJoin = xCursor + i;
      engVec = get_type_session_values(typeData, 'd2EngagedValues', areaIdx, i);
      nonVec = get_type_session_values(typeData, 'd2NonEngagedValues', areaIdx, i);
      pairHandles = plot_engagement_mirror_violin_pair(ax, xJoin, engVec, nonVec, ...
        taskColor, nonEngagedColor, violinOpts);
      if isempty(legendHandles) && ~isempty(pairHandles.engaged.violin) ...
          && ~isempty(pairHandles.nonEngaged.violin)
        legendHandles = [pairHandles.engaged.violin, pairHandles.nonEngaged.violin, ...
          pairHandles.engaged.mean];
        legendLabels = {'engaged d2', 'non-engaged d2', 'mean'};
      end

      xticksCenters(end + 1) = xJoin; %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xCursor + numSessions + 1.2;
  else
    numSessions = numel(typeData.d2Mean{areaIdx});
    if numSessions == 0
      continue;
    end

    for i = 1:numSessions
      xBase = xCursor + i;
      sessVec = get_type_session_values(typeData, 'd2Values', areaIdx, i);
      hSess = plot_manuscript_symmetric_violin(ax, xBase, sessVec, taskColor, violinOpts);
      if isempty(legendHandles) && ~isempty(hSess.violin)
        legendHandles = [hSess.violin, hSess.mean];
        legendLabels = {'session d2', 'mean'};
      end
      xticksCenters(end + 1) = xBase; %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xCursor + numSessions + 1.5;
  end
end

finalize_engagement_violin_axes(ax, xticksCenters, xtickLabels, legendHandles, legendLabels, plotConfig);
end

function plot_d2_normalized_engagement_violins(ax, plotData, sessionTypes, areaIdx, nonEngagedColor, plotConfig)
% PLOT_D2_NORMALIZED_ENGAGEMENT_VIOLINS - Mirrored normalized per-window d2 violins

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendHandles = [];
legendLabels = {};
violinOpts = struct('width', 0.22, 'faceAlpha', 0.5, 'showMean', true, 'plotConfig', plotConfig);

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  taskColor = colors_for_tasks(sessionType);

  if isfield(typeData, 'useEngagement') && typeData.useEngagement
    numSessions = numel(typeData.d2NormEngagedMean{areaIdx});
    if numSessions == 0
      continue;
    end

    for i = 1:numSessions
      xJoin = xCursor + i;
      engVec = get_type_session_values(typeData, 'd2NormEngagedValues', areaIdx, i);
      nonVec = get_type_session_values(typeData, 'd2NormNonEngagedValues', areaIdx, i);
      pairHandles = plot_engagement_mirror_violin_pair(ax, xJoin, engVec, nonVec, ...
        taskColor, nonEngagedColor, violinOpts);
      if isempty(legendHandles) && ~isempty(pairHandles.engaged.violin) ...
          && ~isempty(pairHandles.nonEngaged.violin)
        legendHandles = [pairHandles.engaged.violin, pairHandles.nonEngaged.violin, ...
          pairHandles.engaged.mean];
        legendLabels = {'engaged d2 normalized', 'non-engaged d2 normalized', 'mean'};
      end

      xticksCenters(end + 1) = xJoin; %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xCursor + numSessions + 1.2;
  else
    numSessions = numel(typeData.d2NormMean{areaIdx});
    if numSessions == 0
      continue;
    end

    for i = 1:numSessions
      xBase = xCursor + i;
      sessVec = get_type_session_values(typeData, 'd2NormValues', areaIdx, i);
      hSess = plot_manuscript_symmetric_violin(ax, xBase, sessVec, taskColor, violinOpts);
      if isempty(legendHandles) && ~isempty(hSess.violin)
        legendHandles = [hSess.violin, hSess.mean];
        legendLabels = {'session d2 normalized', 'mean'};
      end
      xticksCenters(end + 1) = xBase; %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    xCursor = xCursor + numSessions + 1.5;
  end
end

finalize_engagement_violin_axes(ax, xticksCenters, xtickLabels, legendHandles, legendLabels, plotConfig);
end

function pairHandles = plot_engagement_mirror_violin_pair(ax, xJoin, engagedVec, nonEngagedVec, ...
    engagedColor, nonEngagedColor, opts)
% PLOT_ENGAGEMENT_MIRROR_VIOLIN_PAIR - Back-to-back engaged/non-engaged half violins

pairHandles = struct('engaged', struct('violin', [], 'mean', []), ...
  'nonEngaged', struct('violin', [], 'mean', []));
pairHandles.engaged = plot_manuscript_half_violin(ax, xJoin, engagedVec, engagedColor, 'left', opts);
pairHandles.nonEngaged = plot_manuscript_half_violin(ax, xJoin, nonEngagedVec, nonEngagedColor, 'right', opts);
end

function handles = plot_manuscript_symmetric_violin(ax, xCenter, values, color, opts)
% PLOT_MANUSCRIPT_SYMMETRIC_VIOLIN - Full symmetric violin for single-class sessions

handles = plot_manuscript_half_violin(ax, xCenter, values, color, 'both', opts);
end

function handles = plot_manuscript_half_violin(ax, xEdge, values, color, side, opts)
% PLOT_MANUSCRIPT_HALF_VIOLIN - Half or full kernel-density violin with mean marker
%
% Variables:
%   ax     - Axes handle
%   xEdge  - Flat vertical edge (engaged/non-engaged join) or center ('both')
%   values - Per-window observation vector
%   color  - RGB face/edge color
%   side   - 'left', 'right', or 'both' (symmetric)
%   opts   - width, faceAlpha, showMean, plotConfig

handles = struct('violin', [], 'mean', []);
if nargin < 6 || isempty(opts)
  opts = struct();
end

values = values(:);
values = values(isfinite(values));
if isempty(values)
  return;
end

width = get_violin_opt(opts, 'width', 0.22);
faceAlpha = get_violin_opt(opts, 'faceAlpha', 0.5);
showMean = get_violin_opt(opts, 'showMean', true);
plotConfig = get_violin_opt(opts, 'plotConfig', fill_manuscript_plot_config());
edgeColor = min(color * 0.75 + 0.25, 1);
lineWidth = plotConfig.lineWidth;
meanVal = mean(values);

if strcmp(side, 'both')
  if numel(values) == 1
    halfW = width * 0.12;
    yVal = values(1);
    yPad = max(abs(yVal) * 0.01, 0.01);
    handles.violin = patch(ax, ...
      [xEdge - halfW, xEdge + halfW, xEdge + halfW, xEdge - halfW], ...
      [yVal - yPad, yVal - yPad, yVal + yPad, yVal + yPad], color, ...
      'FaceAlpha', faceAlpha, 'EdgeColor', edgeColor, 'LineWidth', lineWidth);
  elseif numel(values) == 2
    yMin = min(values);
    yMax = max(values);
    halfW = width * 0.2;
    handles.violin = patch(ax, ...
      [xEdge - halfW, xEdge + halfW, xEdge + halfW, xEdge - halfW], ...
      [yMin, yMin, yMax, yMax], color, ...
      'FaceAlpha', faceAlpha, 'EdgeColor', edgeColor, 'LineWidth', lineWidth);
  else
    [density, xi] = ksdensity(values);
    if max(density) <= 0 || ~all(isfinite(density))
      return;
    end
    density = density / max(density) * width;
    xPoly = [xEdge - density, xEdge + flip(density)];
    yPoly = [xi, flip(xi)];
    handles.violin = patch(ax, xPoly, yPoly, color, ...
      'FaceAlpha', faceAlpha, 'EdgeColor', edgeColor, 'LineWidth', lineWidth);
  end
  if showMean
    handles.mean = plot(ax, xEdge, meanVal, 'd', ...
      'Color', edgeColor, 'MarkerFaceColor', color, ...
      'MarkerSize', plotConfig.markerSize, 'LineWidth', 1);
  end
  return;
end

if numel(values) == 1
  yVal = values(1);
  yPad = max(abs(yVal) * 0.01, 0.01);
  if strcmp(side, 'left')
    xPoly = [xEdge - width * 0.12, xEdge, xEdge, xEdge - width * 0.12];
  else
    xPoly = [xEdge, xEdge + width * 0.12, xEdge + width * 0.12, xEdge];
  end
  yPoly = [yVal - yPad, yVal - yPad, yVal + yPad, yVal + yPad];
  handles.violin = patch(ax, xPoly, yPoly, color, ...
    'FaceAlpha', faceAlpha, 'EdgeColor', edgeColor, 'LineWidth', lineWidth);
elseif numel(values) == 2
  yMin = min(values);
  yMax = max(values);
  halfW = width * 0.2;
  if strcmp(side, 'left')
    xPoly = [xEdge - halfW, xEdge, xEdge, xEdge - halfW];
  else
    xPoly = [xEdge, xEdge + halfW, xEdge + halfW, xEdge];
  end
  yPoly = [yMin, yMin, yMax, yMax];
  handles.violin = patch(ax, xPoly, yPoly, color, ...
    'FaceAlpha', faceAlpha, 'EdgeColor', edgeColor, 'LineWidth', lineWidth);
else
  [density, xi] = ksdensity(values);
  if max(density) <= 0 || ~all(isfinite(density))
    return;
  end
  density = density / max(density) * width;
  if strcmp(side, 'left')
    xPoly = [xEdge - density, xEdge * ones(size(density))];
  else
    xPoly = [xEdge * ones(size(density)), xEdge + density];
  end
  yPoly = [xi, flip(xi)];
  handles.violin = patch(ax, xPoly, yPoly, color, ...
    'FaceAlpha', faceAlpha, 'EdgeColor', edgeColor, 'LineWidth', lineWidth);
end

if showMean
  if strcmp(side, 'left')
    xMean = xEdge - width * 0.08;
  else
    xMean = xEdge + width * 0.08;
  end
  handles.mean = plot(ax, xMean, meanVal, 'd', ...
    'Color', edgeColor, 'MarkerFaceColor', color, ...
    'MarkerSize', plotConfig.markerSize, 'LineWidth', 1);
end
end

function finalize_engagement_violin_axes(ax, xticksCenters, xtickLabels, legendHandles, legendLabels, plotConfig)
% FINALIZE_ENGAGEMENT_VIOLIN_AXES - Tick labels, limits, and legend for violin panels

if ~isempty(xticksCenters)
  xPad = 0.6;
  xlim(ax, [min(xticksCenters) - xPad, max(xticksCenters) + xPad]);
  set(ax, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45);
end
grid(ax, 'off');
if ~isempty(legendHandles)
  legend(ax, legendHandles, legendLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize);
end
hold(ax, 'off');
end

function vec = get_type_session_values(typeData, fieldName, areaIdx, sessionIdx)
% GET_TYPE_SESSION_VALUES - Per-session window vector from plotData type struct

vec = [];
if ~isfield(typeData, fieldName) || areaIdx > numel(typeData.(fieldName))
  return;
end
areaValues = typeData.(fieldName){areaIdx};
if isempty(areaValues) || sessionIdx > numel(areaValues)
  return;
end
vec = areaValues{sessionIdx};
if ~isnumeric(vec)
  vec = [];
end
end

function hasVectors = plot_data_has_violin_vectors(plotData, sessionTypes)
% PLOT_DATA_HAS_VIOLIN_VECTORS - True if plotData stores usable per-window vectors

hasVectors = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'useEngagement') && typeData.useEngagement
    fieldName = 'd2EngagedValues';
  else
    fieldName = 'd2Values';
  end
  if type_data_has_session_vectors(typeData, fieldName)
    hasVectors = true;
    return;
  end
end
end

function hasVectors = type_data_has_session_vectors(typeData, fieldName)
% TYPE_DATA_HAS_SESSION_VECTORS - Any finite values stored for a type/field

hasVectors = false;
if ~isfield(typeData, fieldName) || isempty(typeData.(fieldName))
  return;
end
for a = 1:numel(typeData.(fieldName))
  areaValues = typeData.(fieldName){a};
  if isempty(areaValues)
    continue;
  end
  for s = 1:numel(areaValues)
    if isnumeric(areaValues{s}) && any(isfinite(areaValues{s}(:)))
      hasVectors = true;
      return;
    end
  end
end
end

function val = get_violin_opt(opts, fieldName, defaultVal)
if isfield(opts, fieldName) && ~isempty(opts.(fieldName))
  val = opts.(fieldName);
else
  val = defaultVal;
end
end

function label = get_session_bar_label(typeData, sessionIdx, sessionType)
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
if nargin < 2 || isempty(maxChars)
  maxChars = 7;
end
label = char(label);
if numel(label) > maxChars
  label = label(1:maxChars);
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
