%%
% Criticality Avalanche Across Task Types — Engagement Split (Manuscript)
%
% Like criticality_av_across_tasks.m, but for reach and interval sessions uses
% reach_criticality_metrics_engagement / interval_criticality_metrics_engagement
% (analyses={'avalanches'}, makePlots false) for engaged vs non-engaged metrics.
% Spontaneous sessions use the standard single-window avalanche pipeline.
%
% Variables (configure in this section):
%   sessionTypes, dataSource, collectStart, collectEnd
%   brainArea, brainAreaCombinations, areasToPlot
%   runBatch, plotResults, saveBatchResults, batchResultsFile
%   powerLawFitMethod, avalancheDetectionMode, gofThreshold
%   enablePermutations, nShuffles (spontaneous total; engagement total class only)
%   useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple
%   metricsToPlot - subset of {'dcc','decades','tau','alpha','paramSD'}
%
% Goal:
%   Compare avalanche exponents and related scalars across session types.
%   Reach/interval: engaged vs non-engaged side-by-side per session.
%   Spontaneous: full-window session values (+ shuffle mean when available).

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
dataSource = 'spikes';

collectStart = 0;
collectEnd = [];  % [] = full session

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};

runBatch = true;
plotResults = true;
saveBatchResults = true;
batchResultsFile = '';  % default under dropPath/criticality_manuscript/

powerLawFitMethod = 'plfit2023';
avalancheDetectionMode = 'fixedBinMedian';
gofThreshold = 0.8;
enablePermutations = false;
nShuffles = 5;

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 45;
minNeuronsMultiple = 1.2;
nMinNeurons = 25;

minNonEngagedWindow = 30;
reachBuffer = 1;
absorbSingleEvents = true;

metricsToPlot = {'dcc', 'decades', 'tau', 'alpha'};

opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 100;

plotConfig = fill_manuscript_plot_config();

%% Paths
setup_criticality_manuscript_paths('criticality_av_across_tasks_engagement');
paths = get_paths();
if isempty(batchResultsFile)
  batchResultsFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
    'criticality_av_across_tasks_engagement_batch.mat');
end
[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();

fprintf('\n=== Criticality Avalanche Across Task Types (Engagement) ===\n');
fprintf('Collect window: [%.1f, %s] s\n', collectStart, format_collect_end_label(collectEnd));
fprintf('Power-law fit: %s | detection: %s\n', powerLawFitMethod, avalancheDetectionMode);
fprintf('enablePermutations: %d\n', enablePermutations);
fprintf('Session types: %s\n', strjoin(sessionTypes, ', '));
fprintf('metricsToPlot: %s\n', strjoin(metricsToPlot, ', '));
if ~isempty(brainArea)
  fprintf('Brain area: %s\n', brainArea);
end

%% Batch analysis
if runBatch
  sessionTable = build_engagement_session_table(sessionTypes);
  numSessions = size(sessionTable, 1);
  fprintf('Total sessions: %d\n', numSessions);
  if numSessions == 0
    error('No sessions found for the requested session types.');
  end

  analysisConfig = build_spontaneous_av_config(collectStart, collectEnd, ...
    powerLawFitMethod, avalancheDetectionMode, gofThreshold, enablePermutations, ...
    nShuffles, useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple, ...
    nMinNeurons, clausetPlfitPath, plfit2023Path);

  batchResults = repmat(struct(), numSessions, 1);
  fprintf('\n=== Running avalanche analysis ===\n');
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
    batchResults(s).avalanches = [];
    batchResults(s).skipReason = '';

    try
      if batchResults(s).useEngagement
        engOpts = build_av_engagement_batch_opts(opts, brainArea, brainAreaCombinations, ...
          plotConfig, sessionType, powerLawFitMethod, avalancheDetectionMode, ...
          gofThreshold, enablePermutations, nShuffles, useSubsampling, nSubsamples, ...
          nNeuronsSubsample, minNeuronsMultiple, nMinNeurons, minNonEngagedWindow, ...
          reachBuffer, absorbSingleEvents);
        if strcmpi(sessionType, 'reach')
          engOut = reach_criticality_metrics_engagement(sessionName, engOpts);
        else
          engOut = interval_criticality_metrics_engagement(subjectName, sessionName, engOpts);
        end
        if isempty(engOut.avalanches) || ~isfield(engOut.avalanches, 'byClass')
          fprintf('  No avalanche engagement outputs; skipping.\n');
          batchResults(s).skipReason = 'incomplete avalanche engagement outputs';
          continue;
        end
        batchResults(s).avalanches = engOut.avalanches;
        batchResults(s).success = true;
      else
        loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectName);
        dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});
        [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
          dataStruct, brainArea, brainAreaCombinations);
        if ~areaOk
          fprintf('  Brain area "%s" not available; skipping.\n', brainArea);
          batchResults(s).skipReason = 'brain area unavailable';
          continue;
        end

        sessionConfig = analysisConfig;
        sessionDuration = get_session_collect_duration(dataStruct, opts);
        if isempty(opts.collectEnd) || sessionDuration < ((opts.collectEnd - opts.collectStart) - 1)
          fprintf('  Using full session window (%.1f s).\n', sessionDuration);
          sessionConfig.slidingWindowSize = sessionDuration;
          sessionConfig.avStepSize = sessionDuration;
        end

        avResults = criticality_av_analysis(dataStruct, sessionConfig);
        if ~isempty(brainArea)
          avResults = filter_av_results_to_brain_area(avResults, brainArea);
          if isempty(avResults.areas)
            fprintf('  No results for brain area "%s"; skipping.\n', brainArea);
            batchResults(s).skipReason = 'no results for brain area';
            continue;
          end
        end
        batchResults(s).results = avResults;
        batchResults(s).success = true;
      end
      fprintf('  Analysis completed.\n');
    catch ME
      if is_skippable_session_analysis_error(ME)
        fprintf('  Skipping session: %s\n', ME.message);
        batchResults(s).skipReason = ME.message;
        continue;
      end
      fprintf('  Error: %s\n', ME.message);
      for st = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
      end
    end
  end

  plotData = aggregate_engagement_av_metrics(batchResults, sessionTypes);
  batchMeta = struct( ...
    'sessionTypes', {sessionTypes}, ...
    'collectStart', collectStart, ...
    'collectEnd', collectEnd, ...
    'brainArea', brainArea, ...
    'areasToPlot', {areasToPlot}, ...
    'powerLawFitMethod', powerLawFitMethod, ...
    'avalancheDetectionMode', avalancheDetectionMode, ...
    'metricsToPlot', {metricsToPlot}, ...
    'plotConfig', plotConfig);
  if saveBatchResults
    save(batchResultsFile, 'batchResults', 'plotData', 'batchMeta', '-v7.3');
    fprintf('\nSaved batch results: %s\n', batchResultsFile);
  end
else
  if ~isfile(batchResultsFile)
    error('criticality_av_across_tasks_engagement:NoBatchFile', ...
      'Batch file not found: %s. Set runBatch true to compute.', batchResultsFile);
  end
  loaded = load(batchResultsFile, 'batchResults', 'plotData', 'batchMeta');
  batchResults = loaded.batchResults;
  plotData = loaded.plotData;
  batchMeta = loaded.batchMeta;
  sessionTypes = batchMeta.sessionTypes;
  collectStart = batchMeta.collectStart;
  collectEnd = batchMeta.collectEnd;
  brainArea = batchMeta.brainArea;
  areasToPlot = batchMeta.areasToPlot;
  if isfield(batchMeta, 'metricsToPlot') && ~isempty(batchMeta.metricsToPlot)
    metricsToPlot = batchMeta.metricsToPlot;
  end
  if isfield(batchMeta, 'plotConfig') && ~isempty(batchMeta.plotConfig)
    plotConfig = fill_manuscript_plot_config(batchMeta.plotConfig);
  end
  fprintf('\nLoaded batch results: %s\n', batchResultsFile);
end

%% Plotting
if plotResults
  if exist('batchResults', 'var') && ~isempty(batchResults)
    plotData = aggregate_engagement_av_metrics(batchResults, sessionTypes);
  end
  if isempty(plotData.areas)
    error('No avalanche metrics in plotData. Re-run batch with runBatch true.');
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
  plot_engagement_av_across_tasks(plotData, commonAreas, sessionTypes, collectStart, ...
    collectEnd, paths, brainArea, metricsToPlot, plotConfig);
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
      entries(i).subjectName = ''; %#ok<AGROW>
      entries(i).sessionName = names{i}; %#ok<AGROW>
    end
  otherwise
    error('Unknown sessionType for engagement AV: %s', sessionType);
end
end

function tf = is_engagement_session_type(sessionType)
tf = any(strcmpi(sessionType, {'reach', 'interval'}));
end

function engOpts = build_av_engagement_batch_opts(opts, brainArea, brainAreaCombinations, ...
    plotConfig, sessionType, powerLawFitMethod, avalancheDetectionMode, gofThreshold, ...
    enablePermutations, nShuffles, useSubsampling, nSubsamples, nNeuronsSubsample, ...
    minNeuronsMultiple, nMinNeurons, minNonEngagedWindow, reachBuffer, absorbSingleEvents)
% BUILD_AV_ENGAGEMENT_BATCH_OPTS - Engagement-module opts (avalanches only)

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
engOpts.analyses = {'avalanches'};
engOpts.makePlots = false;
engOpts.saveFigure = false;
engOpts.plotConfig = plotConfig;
engOpts.powerLawFitMethod = powerLawFitMethod;
engOpts.avalancheDetectionMode = avalancheDetectionMode;
engOpts.gofThreshold = gofThreshold;
engOpts.enableCircularPermutations = logical(enablePermutations);
engOpts.nShuffles = nShuffles;
engOpts.useSubsampling = useSubsampling;
engOpts.nSubsamples = nSubsamples;
engOpts.nNeuronsSubsample = nNeuronsSubsample;
engOpts.minNeuronsMultiple = minNeuronsMultiple;
engOpts.nMinNeurons = nMinNeurons;
engOpts.minNonEngagedWindow = minNonEngagedWindow;
if strcmpi(sessionType, 'reach')
  engOpts.reachBuffer = reachBuffer;
  engOpts.absorbSingleReaches = absorbSingleEvents;
else
  engOpts.eventBuffer = reachBuffer;
  engOpts.absorbSingleEvents = absorbSingleEvents;
end
end

function analysisConfig = build_spontaneous_av_config(collectStart, collectEnd, ...
    powerLawFitMethod, avalancheDetectionMode, gofThreshold, enablePermutations, ...
    nShuffles, useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple, ...
    nMinNeurons, clausetPlfitPath, plfit2023Path)
if isempty(collectEnd)
  windowDurationSec = [];
else
  windowDurationSec = collectEnd - collectStart;
end
analysisConfig = struct();
if isempty(windowDurationSec)
  analysisConfig.slidingWindowSize = 1;
  analysisConfig.avStepSize = 1;
else
  analysisConfig.slidingWindowSize = windowDurationSec;
  analysisConfig.avStepSize = windowDurationSec;
end
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.avalancheDetectionMode = avalancheDetectionMode;
if ~strcmpi(avalancheDetectionMode, 'meanIsiZero')
  analysisConfig.binSize = 0.05;
end
analysisConfig.analyzeDcc = true;
analysisConfig.analyzeKappa = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 5;
analysisConfig.enablePermutations = enablePermutations;
analysisConfig.nShuffles = nShuffles;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.thresholdFlag = 1;
analysisConfig.thresholdPct = 1;
analysisConfig.nMinNeurons = nMinNeurons;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
analysisConfig.normalizeMetrics = enablePermutations;
analysisConfig.powerLawFitMethod = powerLawFitMethod;
analysisConfig.gofThreshold = gofThreshold;
analysisConfig.runClausetPlpva = false;
analysisConfig.clausetPlfitPath = clausetPlfitPath;
analysisConfig.plfit2023Path = plfit2023Path;
end

function sessionDuration = get_session_collect_duration(dataStruct, opts)
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
end
end

function tf = is_skippable_session_analysis_error(ME)
tf = contains(ME.message, 'No valid areas to process') ...
  || contains(ME.message, 'insufficient neurons') ...
  || contains(ME.message, 'TooFewNeurons') ...
  || contains(ME.message, 'not available');
end

function results = filter_av_results_to_brain_area(results, brainArea)
if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end
cellFields = {'dcc', 'dccNormalized', 'kappa', 'kappaNormalized', 'decades', 'tau', ...
  'alpha', 'paramSD', 'startS', 'dccPermuted', 'kappaPermuted', 'decadesPermuted', ...
  'tauPermuted', 'alphaPermuted', 'paramSDPermuted', 'dccPermutedMean', ...
  'kappaPermutedMean', 'decadesPermutedMean', 'tauPermutedMean', 'alphaPermutedMean', ...
  'paramSDPermutedMean'};
results.areas = results.areas(areaIdx);
for f = 1:numel(cellFields)
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

function plotData = aggregate_engagement_av_metrics(batchResults, sessionTypes)
% AGGREGATE_ENGAGEMENT_AV_METRICS - Per-session avalanche scalars by engagement class

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();

metricPairs = get_engagement_av_metric_pairs();
spontFields = [metricPairs(:, 2)', ...
  strcat(metricPairs(:, 2)', 'PermutedMean')];
engFields = [metricPairs(:, 3)', metricPairs(:, 4)'];

for s = 1:numel(batchResults)
  if ~batchResults(s).success
    continue;
  end

  sessionType = batchResults(s).sessionType;
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    plotData.byType.(typeKey) = init_engagement_av_type(spontFields, engFields, 0);
    plotData.byType.(typeKey).useEngagement = is_engagement_session_type(sessionType);
  end
  typeData = plotData.byType.(typeKey);

  if batchResults(s).useEngagement
    avByClass = batchResults(s).avalanches.byClass;
    if isfield(batchResults(s).avalanches, 'areaNames') ...
        && ~isempty(batchResults(s).avalanches.areaNames)
      areaNames = batchResults(s).avalanches.areaNames;
    elseif isfield(avByClass, 'engaged') && isfield(avByClass.engaged, 'areas')
      areaNames = avByClass.engaged.areas;
    else
      continue;
    end

    for a = 1:numel(areaNames)
      areaName = areaNames{a};
      areaIdx = find(strcmp(plotData.areas, areaName), 1);
      if isempty(areaIdx)
        plotData.areas{end + 1} = areaName; %#ok<AGROW>
        areaIdx = numel(plotData.areas);
        plotData = extend_engagement_av_areas(plotData, spontFields, engFields, areaIdx);
        typeData = plotData.byType.(typeKey);
      end
      typeData = ensure_engagement_av_area_slots(typeData, spontFields, engFields, areaIdx);

      for m = 1:size(metricPairs, 1)
        metricName = metricPairs{m, 1};
        engField = metricPairs{m, 3};
        nonField = metricPairs{m, 4};
        typeData.(engField){areaIdx}(end + 1) = get_engagement_area_av_metric( ...
          avByClass.engaged, areaName, metricName); %#ok<AGROW>
        typeData.(nonField){areaIdx}(end + 1) = get_engagement_area_av_metric( ...
          avByClass.nonEngaged, areaName, metricName); %#ok<AGROW>
      end
    end
  else
    results = batchResults(s).results;
    if isempty(results) || ~isfield(results, 'areas')
      continue;
    end
    for a = 1:numel(results.areas)
      areaName = results.areas{a};
      areaIdx = find(strcmp(plotData.areas, areaName), 1);
      if isempty(areaIdx)
        plotData.areas{end + 1} = areaName; %#ok<AGROW>
        areaIdx = numel(plotData.areas);
        plotData = extend_engagement_av_areas(plotData, spontFields, engFields, areaIdx);
        typeData = plotData.byType.(typeKey);
      end
      typeData = ensure_engagement_av_area_slots(typeData, spontFields, engFields, areaIdx);

      for m = 1:size(metricPairs, 1)
        metricName = metricPairs{m, 1};
        spontField = metricPairs{m, 2};
        typeData.(spontField){areaIdx}(end + 1) = extract_single_window_value( ...
          get_result_metric_cell(results, metricName, a)); %#ok<AGROW>
        shuffleField = [metricName, 'PermutedMean'];
        if isfield(results, shuffleField)
          typeData.(shuffleField){areaIdx}(end + 1) = extract_single_window_value( ...
            get_result_metric_cell(results, shuffleField, a)); %#ok<AGROW>
        else
          typeData.(shuffleField){areaIdx}(end + 1) = nan; %#ok<AGROW>
        end
      end
    end
  end

  typeData.sessionLabels{end + 1} = batchResults(s).label;
  typeData.sessionNames{end + 1} = batchResults(s).sessionName;
  plotData.byType.(typeKey) = typeData;
end
end

function pairs = get_engagement_av_metric_pairs()
% Rows: metricName, spontaneousField, engagedField, nonEngagedField
pairs = {
  'dcc', 'dcc', 'dccEngaged', 'dccNonEngaged'
  'decades', 'decades', 'decadesEngaged', 'decadesNonEngaged'
  'tau', 'tau', 'tauEngaged', 'tauNonEngaged'
  'alpha', 'alpha', 'alphaEngaged', 'alphaNonEngaged'
  'paramSD', 'paramSD', 'paramSDEngaged', 'paramSDNonEngaged'
  };
end

function typeData = init_engagement_av_type(spontFields, engFields, numAreas)
typeData = struct();
allFields = unique([spontFields, engFields], 'stable');
for f = 1:numel(allFields)
  typeData.(allFields{f}) = cell(1, numAreas);
  for a = 1:numAreas
    typeData.(allFields{f}){a} = [];
  end
end
typeData.sessionLabels = {};
typeData.sessionNames = {};
typeData.useEngagement = false;
end

function plotData = extend_engagement_av_areas(plotData, spontFields, engFields, newAreaIdx)
typeNames = fieldnames(plotData.byType);
for i = 1:numel(typeNames)
  typeData = plotData.byType.(typeNames{i});
  typeData = ensure_engagement_av_area_slots(typeData, spontFields, engFields, newAreaIdx);
  plotData.byType.(typeNames{i}) = typeData;
end
end

function typeData = ensure_engagement_av_area_slots(typeData, spontFields, engFields, areaIdx)
allFields = unique([spontFields, engFields], 'stable');
for f = 1:numel(allFields)
  fieldName = allFields{f};
  if ~isfield(typeData, fieldName)
    typeData.(fieldName) = {};
  end
  while numel(typeData.(fieldName)) < areaIdx
    typeData.(fieldName){end + 1} = []; %#ok<AGROW>
  end
end
end

function val = get_engagement_area_av_metric(avClassResult, areaName, metricName)
val = nan;
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
if isfield(avData, metricName) && isfinite(avData.(metricName))
  val = avData.(metricName);
  return;
end
if strcmp(metricName, 'decades') && isfield(avData, 'sizeFitInfo') ...
    && isstruct(avData.sizeFitInfo) && isfield(avData.sizeFitInfo, 'decades') ...
    && isfinite(avData.sizeFitInfo.decades)
  val = avData.sizeFitInfo.decades;
end
end

function metricVec = get_result_metric_cell(results, fieldName, areaIdx)
metricVec = [];
if isfield(results, fieldName) && areaIdx <= numel(results.(fieldName))
  metricVec = results.(fieldName){areaIdx};
end
end

function val = extract_single_window_value(metricVec)
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

function plot_engagement_av_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, ...
    collectEnd, paths, brainArea, metricsToPlot, plotConfig)
% PLOT_ENGAGEMENT_AV_ACROSS_TASKS - Engaged/non-engaged avalanche scalars by session

if nargin < 9 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
metricsToPlot = normalize_av_engagement_metrics(metricsToPlot);
metricPairs = get_engagement_av_metric_pairs();
plotRows = metricPairs(ismember(metricPairs(:, 1), metricsToPlot), :);

nMetrics = size(plotRows, 1);
nCols = min(2, nMetrics);
nRows = ceil(nMetrics / nCols);

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end
collectTag = format_collect_window_tag(collectStart, collectEnd);

for a = 1:numel(areasToPlot)
  areaName = areasToPlot{a};
  areaIdx = find(strcmp(plotData.areas, areaName), 1);
  if isempty(areaIdx) || ~area_has_engagement_av_plot_data(plotData, sessionTypes, areaIdx, plotRows)
    continue;
  end

  fig = figure('Color', 'w', 'Name', sprintf('AV engagement — %s', areaName));
  position_figure_full_monitor(fig);

  for m = 1:nMetrics
    ax = subplot(nRows, nCols, m);
    metricName = plotRows{m, 1};
    spontField = plotRows{m, 2};
    engField = plotRows{m, 3};
    nonField = plotRows{m, 4};
    plot_av_engagement_metric_panel(ax, plotData, sessionTypes, areaIdx, ...
      spontField, engField, nonField, [metricName, 'PermutedMean'], plotConfig);
    ylabel(ax, metricName, 'FontSize', plotConfig.axisLabelFontSize);
    title(ax, sprintf('%s — %s', areaName, metricName), ...
      'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
    apply_manuscript_axes_style(ax, plotConfig, '', '', '', 'none');
  end

  if ~isempty(brainArea)
    titleStr = sprintf('Avalanche engagement — %s [%s]', brainArea, collectTag);
  else
    titleStr = sprintf('Avalanche engagement — %s [%s]', areaName, collectTag);
  end
  sgtitle(fig, titleStr, 'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', ...
    'Interpreter', 'none');

  if ~isempty(brainArea)
    plotBase = sprintf('criticality_av_across_tasks_engagement_%s_%s', brainArea, collectTag);
  else
    plotBase = sprintf('criticality_av_across_tasks_engagement_%s_%s', areaName, collectTag);
  end
  if numel(areasToPlot) > 1
    plotBase = sprintf('%s_area%s', plotBase, areaName);
  end
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
end

fprintf('\nAll figures saved to %s\n', saveDir);
end

function plot_av_engagement_metric_panel(ax, plotData, sessionTypes, areaIdx, ...
    spontField, engField, nonField, shuffleField, plotConfig)
% PLOT_AV_ENGAGEMENT_METRIC_PANEL - One metric: engaged/non pairs or spontaneous

hold(ax, 'on');
xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendHandles = [];
legendLabels = {};
yValuesForLim = [];
shuffleColor = manuscript_plot_colors().shuffled;

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  taskColor = colors_for_tasks(sessionType);

  if isfield(typeData, 'useEngagement') && typeData.useEngagement
    if ~isfield(typeData, engField) || areaIdx > numel(typeData.(engField))
      continue;
    end
    engVals = typeData.(engField){areaIdx}(:)';
    nonVals = typeData.(nonField){areaIdx}(:)';
    numSessions = numel(engVals);
    if numSessions == 0
      continue;
    end
    for i = 1:numSessions
      xBase = xCursor + i;
      hEng = plot(ax, xBase - 0.12, engVals(i), 'o', ...
        'Color', taskColor, 'MarkerFaceColor', taskColor, ...
        'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
        'LineWidth', plotConfig.lineWidth);
      hNon = plot(ax, xBase + 0.12, nonVals(i), 'o', ...
        'Color', taskColor, 'MarkerFaceColor', 'none', ...
        'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
        'LineWidth', plotConfig.lineWidth);
      if isempty(legendHandles)
        legendHandles = [hEng, hNon];
        legendLabels = {'engaged', 'non-engaged'};
      end
      xticksCenters(end + 1) = xBase; %#ok<AGROW>
      xtickLabels{end + 1} = get_session_bar_label(typeData, i, sessionType); %#ok<AGROW>
    end
    yValuesForLim = [yValuesForLim, engVals(isfinite(engVals)), nonVals(isfinite(nonVals))]; %#ok<AGROW>
    xCursor = xCursor + numSessions + 1.5;
  else
    if ~isfield(typeData, spontField) || areaIdx > numel(typeData.(spontField))
      continue;
    end
    vals = typeData.(spontField){areaIdx}(:)';
    numSessions = numel(vals);
    if numSessions == 0
      continue;
    end
    xPos = xCursor + (1:numSessions);
    hData = plot(ax, xPos, vals, 'o', 'Color', taskColor, 'MarkerFaceColor', taskColor, ...
      'MarkerSize', plotConfig.scatterMarkerSize / 4, 'LineWidth', plotConfig.lineWidth);
    yValuesForLim = [yValuesForLim, vals(isfinite(vals))]; %#ok<AGROW>

    if isfield(typeData, shuffleField) && areaIdx <= numel(typeData.(shuffleField))
      shuffleVals = typeData.(shuffleField){areaIdx}(:)';
      if numel(shuffleVals) == numSessions && any(isfinite(shuffleVals))
        hShuf = plot(ax, xPos + 0.18, shuffleVals, 's', ...
          'Color', shuffleColor, 'MarkerFaceColor', shuffleColor, ...
          'MarkerSize', plotConfig.scatterMarkerSize / 4, ...
          'LineWidth', plotConfig.lineWidth);
        yValuesForLim = [yValuesForLim, shuffleVals(isfinite(shuffleVals))]; %#ok<AGROW>
        if isempty(legendHandles)
          legendHandles = [hData, hShuf];
          legendLabels = {'session', 'shuffled'};
        end
      elseif isempty(legendHandles)
        legendHandles = hData;
        legendLabels = {'session'};
      end
    elseif isempty(legendHandles)
      legendHandles = hData;
      legendLabels = {'session'};
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
apply_buffered_ylim(ax, yValuesForLim);
grid(ax, 'off');
if ~isempty(legendHandles)
  legend(ax, legendHandles, legendLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize);
end
hold(ax, 'off');
end

function metricsToPlot = normalize_av_engagement_metrics(metricsToPlot)
if ischar(metricsToPlot) || isstring(metricsToPlot)
  metricsToPlot = cellstr(metricsToPlot);
end
valid = {'dcc', 'decades', 'tau', 'alpha', 'paramSD'};
metricsToPlot = lower(metricsToPlot(:)');
unknown = setdiff(metricsToPlot, valid);
if ~isempty(unknown)
  error('metricsToPlot unknown: %s', strjoin(unknown, ', '));
end
metricsToPlot = intersect(valid, metricsToPlot, 'stable');
if isempty(metricsToPlot)
  error('metricsToPlot must include at least one valid metric.');
end
end

function hasData = area_has_engagement_av_plot_data(plotData, sessionTypes, areaIdx, plotRows)
hasData = false;
for t = 1:numel(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  for m = 1:size(plotRows, 1)
    if isfield(typeData, 'useEngagement') && typeData.useEngagement
      fieldName = plotRows{m, 3};
    else
      fieldName = plotRows{m, 2};
    end
    if isfield(typeData, fieldName) && areaIdx <= numel(typeData.(fieldName)) ...
        && ~isempty(typeData.(fieldName){areaIdx}) ...
        && any(isfinite(typeData.(fieldName){areaIdx}))
      hasData = true;
      return;
    end
  end
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
if nargin < 2 || isempty(maxChars)
  maxChars = 7;
end
label = char(label);
if numel(label) > maxChars
  label = label(1:maxChars);
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

function apply_buffered_ylim(ax, yValues)
yValues = yValues(isfinite(yValues));
if isempty(yValues)
  return;
end
yMin = min(yValues);
yMax = max(yValues);
if yMin == yMax
  pad = max(0.05 * abs(yMin), 0.05);
  ylim(ax, [yMin - pad, yMax + pad]);
  return;
end
bufferFrac = 0.08;
yRange = yMax - yMin;
ylim(ax, [yMin - bufferFrac * yRange, yMax + bufferFrac * yRange]);
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
