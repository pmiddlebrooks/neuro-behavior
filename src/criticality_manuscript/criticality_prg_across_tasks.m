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
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' = all valid areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   areasToPlot            - Area names to plot; {} uses brainArea if set
%   runBatch       - If true, run criticality_prg_analysis per session
%   plotResults    - If true, create summary figures after batch
%   prgMethod        - 'pca' (momentum-space) or 'icg' (real-space ICG, Morales 2023)
%   surrogateMethod  - 'isi' (ISI shuffle per unit) or 'circular' (circshift per neuron)
%   useSubsampling   - If true, kappa/D_JS per window = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%
% Goal:
%   Compare PRG kurtosis (kappa at N/finalCutoffDivisor) and Jensen-Shannon distance
%   (D_JS vs Gaussian) across session types, including values normalized by per-window
%   surrogate means. Per session: mean and SEM across valid windows.

%% Configuration
sessionTypes = {'spontaneous', 'interval', 'reach'};
dataSource = 'spikes';

collectStart = 0;
collectEnd = 45 * 60;

prgWindow = 30;  % seconds; non-overlapping blocks (blockWindowSize)

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
areasToPlot = {};
runBatch = true;
plotResults = true;

useSubsampling = false;
nSubsamples = 30;
nNeuronsSubsample = 32;
minNeuronsMultiple = 1.25;

% Surrogate null: 'isi' (Cambrainha paper) or 'circular' (per-neuron circshift on binned data)
surrogateMethod = 'circular';

% PRG settings (aligned with run_criticality_prg.m)
analysisConfig = struct();
analysisConfig.prgMethod = 'pca';  % 'pca' or 'icg'
analysisConfig.blockWindowSize = prgWindow;
analysisConfig.binSize = 0.05;
analysisConfig.cvThreshold = 5;
analysisConfig.cutoffDivisors = [1, 2, 4, 8, 16];
analysisConfig.finalCutoffDivisor = 16;
analysisConfig.kappaAxisMax = 20;
analysisConfig.enableSurrogates = true;
analysisConfig.nSurrogates = 10;
analysisConfig.surrogateMethod = surrogateMethod;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.nMinNeurons = 32;
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
fprintf('PRG method: %s\n', analysisConfig.prgMethod);
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('PRG blocks: %.1f s, non-overlapping\n', prgWindow);
fprintf('Kappa at N/%d; bin size: %.3f s; surrogates: %s\n', ...
  analysisConfig.finalCutoffDivisor, analysisConfig.binSize, analysisConfig.surrogateMethod);
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

      [dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations);
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
    prgWindow, paths, brainArea, analysisConfig.prgMethod);
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
  'nNeuronsPerWindow', 'kappaSurrogate', 'djs', 'djsSurrogate', 'nCutoffList'};

results.areas = results.areas(areaIdx);
for f = 1:length(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
end

function plotData = aggregate_prg_metrics(batchResults, sessionTypes, finalCutoffDivisor)
% AGGREGATE_PRG_METRICS - Per-session mean and SEM of window kappa / D_JS and surrogates
%
% Variables:
%   finalCutoffDivisor - Reported in plot labels (N/divisor)

if nargin < 3 || isempty(finalCutoffDivisor)
  finalCutoffDivisor = 16;
end

plotData = struct();
plotData.areas = {};
plotData.sessionTypes = sessionTypes;
plotData.byType = struct();
plotData.finalCutoffDivisor = finalCutoffDivisor;

metricFields = {'kappaMean', 'kappaSem', 'kappaShuffleMean', 'kappaShuffleSem', ...
  'kappaNormMean', 'kappaNormSem', ...
  'djsMean', 'djsSem', 'djsShuffleMean', 'djsShuffleSem', ...
  'djsNormMean', 'djsNormSem'};

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
%   summary - kappaMean, kappaSem, kappaShuffleMean, kappaShuffleSem,
%             kappaNormMean, kappaNormSem,
%             djsMean, djsSem, djsShuffleMean, djsShuffleSem,
%             djsNormMean, djsNormSem
%
%   Normalized metrics: per-window value / mean(surrogate for that window), then
%   mean and SEM across valid windows. *ShuffleSem - SEM across windows of per-window surrogate means

summary = struct('kappaMean', nan, 'kappaSem', nan, 'kappaShuffleMean', nan, ...
  'kappaShuffleSem', nan, 'kappaNormMean', nan, 'kappaNormSem', nan, ...
  'djsMean', nan, 'djsSem', nan, 'djsShuffleMean', nan, 'djsShuffleSem', nan, ...
  'djsNormMean', nan, 'djsNormSem', nan);

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

if isfield(results, 'djs') && areaIdx <= length(results.djs) && ~isempty(results.djs{areaIdx})
  djsVec = results.djs{areaIdx}(:);
  if numel(djsVec) == nWin
    djsValid = djsVec(validMask);
    djsValid = djsValid(isfinite(djsValid));
    if ~isempty(djsValid)
      summary.djsMean = mean(djsValid);
      nD = numel(djsValid);
      if nD > 1
        summary.djsSem = std(djsValid) / sqrt(nD);
      else
        summary.djsSem = 0;
      end
    end
  end
end

if isfield(results, 'kappaSurrogate') && areaIdx <= length(results.kappaSurrogate) ...
    && ~isempty(results.kappaSurrogate{areaIdx})
  surrMat = results.kappaSurrogate{areaIdx};
  if size(surrMat, 1) == nWin
    perWindowShuffleMean = get_per_window_shuffle_mean_matrix(surrMat, results);

    normVec = normalize_prg_by_surrogate(kappaVec, perWindowShuffleMean);
    normValid = normVec(validMask);
    normValid = normValid(isfinite(normValid));
    if ~isempty(normValid)
      summary.kappaNormMean = mean(normValid);
      nNorm = numel(normValid);
      if nNorm > 1
        summary.kappaNormSem = std(normValid) / sqrt(nNorm);
      else
        summary.kappaNormSem = 0;
      end
    end

    perWindowShuffleMean = perWindowShuffleMean(validMask);
    shuffleValid = perWindowShuffleMean(isfinite(perWindowShuffleMean));
    if ~isempty(shuffleValid)
      summary.kappaShuffleMean = mean(shuffleValid);
      nSh = numel(shuffleValid);
      if nSh > 1
        summary.kappaShuffleSem = std(shuffleValid) / sqrt(nSh);
      else
        summary.kappaShuffleSem = 0;
      end
    end
  end
end

if isfield(results, 'djsSurrogate') && areaIdx <= length(results.djsSurrogate) ...
    && ~isempty(results.djsSurrogate{areaIdx}) ...
    && isfield(results, 'djs') && areaIdx <= length(results.djs) ...
    && ~isempty(results.djs{areaIdx})
  djsVec = results.djs{areaIdx}(:);
  djsSurrMat = results.djsSurrogate{areaIdx};
  if numel(djsVec) == nWin && size(djsSurrMat, 1) == nWin
    perWindowDjsShuffleMean = get_per_window_shuffle_mean_matrix(djsSurrMat, results);

    normVec = normalize_prg_by_surrogate(djsVec, perWindowDjsShuffleMean);
    normValid = normVec(validMask);
    normValid = normValid(isfinite(normValid));
    if ~isempty(normValid)
      summary.djsNormMean = mean(normValid);
      nNorm = numel(normValid);
      if nNorm > 1
        summary.djsNormSem = std(normValid) / sqrt(nNorm);
      else
        summary.djsNormSem = 0;
      end
    end

    perWindowDjsShuffleMean = perWindowDjsShuffleMean(validMask);
    shuffleValid = perWindowDjsShuffleMean(isfinite(perWindowDjsShuffleMean));
    if ~isempty(shuffleValid)
      summary.djsShuffleMean = mean(shuffleValid);
      nDjsSh = numel(shuffleValid);
      if nDjsSh > 1
        summary.djsShuffleSem = std(shuffleValid) / sqrt(nDjsSh);
      else
        summary.djsShuffleSem = 0;
      end
    end
  end
end
end

function normVec = normalize_prg_by_surrogate(dataVec, surrogateMeanVec)
% NORMALIZE_PRG_BY_SURROGATE - Per-window ratio to mean surrogate value
%
% Variables:
%   dataVec           - Metric per window (column vector)
%   surrogateMeanVec  - Mean surrogate metric per window
%
% Goal:
%   Match criticality_av_analysis normalization: metric / mean(surrogate).

dataVec = dataVec(:);
surrogateMeanVec = surrogateMeanVec(:);
normVec = nan(size(dataVec));
if numel(dataVec) ~= numel(surrogateMeanVec)
  return;
end

validDenom = isfinite(dataVec) & isfinite(surrogateMeanVec) & surrogateMeanVec > 0;
normVec(validDenom) = dataVec(validDenom) ./ surrogateMeanVec(validDenom);
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

function plot_prg_across_tasks(plotData, areasToPlot, sessionTypes, collectStart, collectEnd, prgWindow, paths, brainArea, prgMethod)
% PLOT_PRG_ACROSS_TASKS - Session kappa and D_JS with surrogate summary, by session type

if nargin < 8 || isempty(brainArea)
  brainArea = '';
end
if nargin < 9 || isempty(prgMethod)
  prgMethod = 'pca';
end

finalCutoffDivisor = 16;
if isfield(plotData, 'finalCutoffDivisor') && ~isempty(plotData.finalCutoffDivisor)
  finalCutoffDivisor = plotData.finalCutoffDivisor;
end

kappaYLabel = sprintf('\\kappa (N/%d, mean \\pm SEM across windows)', finalCutoffDivisor);
kappaTitleWord = sprintf('\\kappa (N/%d)', finalCutoffDivisor);
djsYLabel = sprintf('D_{JS} (N/%d vs Gaussian, mean \\pm SEM across windows)', finalCutoffDivisor);
djsTitleWord = sprintf('D_{JS} (N/%d)', finalCutoffDivisor);

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
    titleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
      prgMethod, kappaTitleWord, brainArea, collectStart, collectEnd, prgWindow);
  else
    titleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
      prgMethod, kappaTitleWord, areaName, collectStart, collectEnd, prgWindow);
  end
  sgtitle(figKappa, titleStr, 'FontWeight', 'bold');

  plotBase = make_prg_plot_basename('criticality_prg_across_tasks', areaName, brainArea, ...
    prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, prgMethod);
  exportgraphics(figKappa, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(figKappa, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));

  if area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, 'kappaNormMean')
    kappaNormYLabel = sprintf('\\kappa / surrogate (N/%d, mean \\pm SEM across windows)', finalCutoffDivisor);
    kappaNormTitleWord = sprintf('\\kappa / surrogate (N/%d)', finalCutoffDivisor);

    figKappaNorm = figure(6200 + a);
    clf(figKappaNorm);
    position_figure_full_monitor(figKappaNorm);
    axKappaNorm = axes(figKappaNorm);
    plot_prg_normalized_sessions(axKappaNorm, plotData, sessionTypes, areaIdx, ...
      typeColors, 'kappaNormMean', 'kappaNormSem', 'session \kappa / surrogate');
    ylabel(axKappaNorm, kappaNormYLabel);
    title(axKappaNorm, sprintf('%s — %s', areaName, kappaNormTitleWord));
    yline(axKappaNorm, 1, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');

    if ~isempty(brainArea)
      kappaNormTitleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
        prgMethod, kappaNormTitleWord, brainArea, collectStart, collectEnd, prgWindow);
    else
      kappaNormTitleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
        prgMethod, kappaNormTitleWord, areaName, collectStart, collectEnd, prgWindow);
    end
    sgtitle(figKappaNorm, kappaNormTitleStr, 'FontWeight', 'bold');

    plotBaseNorm = make_prg_plot_basename('criticality_prg_kappa_norm_across_tasks', areaName, brainArea, ...
      prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, prgMethod);
    exportgraphics(figKappaNorm, fullfile(saveDir, [plotBaseNorm, '.png']), 'Resolution', 300);
    exportgraphics(figKappaNorm, fullfile(saveDir, [plotBaseNorm, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseNorm));
  end

  if area_has_prg_djs_plot_data(plotData, sessionTypes, areaIdx)
    figDjs = figure(6100 + a);
    clf(figDjs);
    position_figure_full_monitor(figDjs);
    axDjs = axes(figDjs);
    plot_djs_with_shuffle(axDjs, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor);
    ylabel(axDjs, djsYLabel);
    title(axDjs, sprintf('%s — %s vs Gaussian', areaName, djsTitleWord));

    if ~isempty(brainArea)
      djsTitleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
        prgMethod, djsTitleWord, brainArea, collectStart, collectEnd, prgWindow);
    else
      djsTitleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
        prgMethod, djsTitleWord, areaName, collectStart, collectEnd, prgWindow);
    end
    sgtitle(figDjs, djsTitleStr, 'FontWeight', 'bold');

    plotBaseDjs = make_prg_plot_basename('criticality_prg_djs_across_tasks', areaName, brainArea, ...
      prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, prgMethod);
    exportgraphics(figDjs, fullfile(saveDir, [plotBaseDjs, '.png']), 'Resolution', 300);
    exportgraphics(figDjs, fullfile(saveDir, [plotBaseDjs, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseDjs));
  end

  if area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, 'djsNormMean')
    djsNormYLabel = sprintf('D_{JS} / surrogate (N/%d vs Gaussian, mean \\pm SEM across windows)', finalCutoffDivisor);
    djsNormTitleWord = sprintf('D_{JS} / surrogate (N/%d)', finalCutoffDivisor);

    figDjsNorm = figure(6300 + a);
    clf(figDjsNorm);
    position_figure_full_monitor(figDjsNorm);
    axDjsNorm = axes(figDjsNorm);
    plot_prg_normalized_sessions(axDjsNorm, plotData, sessionTypes, areaIdx, ...
      typeColors, 'djsNormMean', 'djsNormSem', 'session D_{JS} / surrogate');
    ylabel(axDjsNorm, djsNormYLabel);
    title(axDjsNorm, sprintf('%s — %s vs Gaussian', areaName, djsNormTitleWord));
    yline(axDjsNorm, 1, '--', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1, 'HandleVisibility', 'off');

    if ~isempty(brainArea)
      djsNormTitleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
        prgMethod, djsNormTitleWord, brainArea, collectStart, collectEnd, prgWindow);
    else
      djsNormTitleStr = sprintf('PRG (%s) %s — %s [%.0f–%.0f s, %.0fs blocks]', ...
        prgMethod, djsNormTitleWord, areaName, collectStart, collectEnd, prgWindow);
    end
    sgtitle(figDjsNorm, djsNormTitleStr, 'FontWeight', 'bold');

    plotBaseDjsNorm = make_prg_plot_basename('criticality_prg_djs_norm_across_tasks', areaName, brainArea, ...
      prgWindow, collectStart, collectEnd, length(areasToPlot) > 1, finalCutoffDivisor, prgMethod);
    exportgraphics(figDjsNorm, fullfile(saveDir, [plotBaseDjsNorm, '.png']), 'Resolution', 300);
    exportgraphics(figDjsNorm, fullfile(saveDir, [plotBaseDjsNorm, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, plotBaseDjsNorm));
  end
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
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 24, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', 'session \kappa');

  if any(isfinite(shuffleMeans))
    xShuffle = xPos + 0.22;
    semPlot = shuffleSems(:)';
    semPlot(~isfinite(semPlot)) = 0;
    errorbar(ax, xShuffle, shuffleMeans, semPlot, 's', ...
      'Color', shuffleBarColor, 'MarkerFaceColor', shuffleBarColor, ...
      'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 24, 'LineWidth', 1.2, ...
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

function plot_prg_normalized_sessions(ax, plotData, sessionTypes, areaIdx, typeColors, meanField, semField, legendLabel)
% PLOT_PRG_NORMALIZED_SESSIONS - Session mean normalized metric with SEM by session type
%
% Variables:
%   meanField, semField - Fields in plotData.byType (e.g. kappaNormMean, kappaNormSem)
%   legendLabel         - Display name for errorbar series

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
  if ~isfield(typeData, meanField) || areaIdx > length(typeData.(meanField))
    continue;
  end

  metricMeans = typeData.(meanField){areaIdx};
  metricSems = typeData.(semField){areaIdx};
  metricMeans = metricMeans(:)';
  metricSems = metricSems(:)';
  if numel(metricSems) ~= numel(metricMeans)
    metricSems = nan(size(metricMeans));
  end
  numBars = numel(metricMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, metricMeans, metricSems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 24, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', legendLabel);

  groupCenter = mean(xPos);
  xticksCenters(end+1) = groupCenter; %#ok<AGROW>
  xtickLabels{end+1} = sessionType; %#ok<AGROW>

  validMeans = metricMeans(isfinite(metricMeans));
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
  legend(ax, legendLabel, 'Location', 'best');
end
hold(ax, 'off');
end

function plot_djs_with_shuffle(ax, plotData, sessionTypes, areaIdx, typeColors, shuffleBarColor)
% PLOT_DJS_WITH_SHUFFLE - Session mean D_JS with SEM; surrogate mean with SEM beside each session

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
  if ~isfield(typeData, 'djsMean') || areaIdx > length(typeData.djsMean)
    continue;
  end

  djsMeans = typeData.djsMean{areaIdx};
  djsSems = typeData.djsSem{areaIdx};
  shuffleMeans = typeData.djsShuffleMean{areaIdx};
  shuffleSems = typeData.djsShuffleSem{areaIdx};
  djsMeans = djsMeans(:)';
  djsSems = djsSems(:)';
  shuffleMeans = shuffleMeans(:)';
  shuffleSems = shuffleSems(:)';
  if numel(shuffleSems) ~= numel(shuffleMeans)
    shuffleSems = nan(size(shuffleMeans));
  end
  numBars = numel(djsMeans);
  if numBars == 0
    continue;
  end

  xPos = xCursor + (1:numBars);
  errorbar(ax, xPos, djsMeans, djsSems, 'o', 'Color', typeColors(t, :), ...
    'MarkerFaceColor', typeColors(t, :), 'MarkerSize', 24, 'LineWidth', 1.2, ...
    'CapSize', 8, 'DisplayName', 'session D_{JS}');

  if any(isfinite(shuffleMeans))
    xShuffle = xPos + 0.22;
    semPlot = shuffleSems(:)';
    semPlot(~isfinite(semPlot)) = 0;
    errorbar(ax, xShuffle, shuffleMeans, semPlot, 's', ...
      'Color', shuffleBarColor, 'MarkerFaceColor', shuffleBarColor, ...
      'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 24, 'LineWidth', 1.2, ...
      'CapSize', 8, 'DisplayName', 'surrogate mean \pm SEM (across windows)');
  end

  groupCenter = mean(xPos);
  xticksCenters(end+1) = groupCenter; %#ok<AGROW>
  xtickLabels{end+1} = sessionType; %#ok<AGROW>

  validMeans = djsMeans(isfinite(djsMeans));
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
  legend(ax, {'session D_{JS}', 'surrogate mean \pm SEM (across windows)'}, 'Location', 'best');
end
hold(ax, 'off');
end

function plotBase = make_prg_plot_basename(prefix, areaName, brainArea, prgWindow, collectStart, collectEnd, multiArea, finalCutoffDivisor, prgMethod)
% MAKE_PRG_PLOT_BASENAME - Filename stem for saved figures

if nargin < 8 || isempty(finalCutoffDivisor)
  finalCutoffDivisor = 16;
end
if nargin < 9 || isempty(prgMethod)
  prgMethod = 'pca';
end

if ~isempty(brainArea)
  plotBase = sprintf('%s_%s_%s_win%.0fs_%.0f-%.0fs_N%d', prefix, prgMethod, brainArea, prgWindow, collectStart, collectEnd, finalCutoffDivisor);
else
  plotBase = sprintf('%s_%s_%s_win%.0fs_%.0f-%.0fs_N%d', prefix, prgMethod, areaName, prgWindow, collectStart, collectEnd, finalCutoffDivisor);
end
if multiArea
  plotBase = sprintf('%s_area%s', plotBase, areaName);
end
end

function hasData = area_has_prg_djs_plot_data(plotData, sessionTypes, areaIdx)
% AREA_HAS_PRG_DJS_PLOT_DATA - True if any session type has D_JS values for this area

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'djsMean') && areaIdx <= length(typeData.djsMean) ...
      && ~isempty(typeData.djsMean{areaIdx})
    hasData = true;
    return;
  end
end
end

function hasData = area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, metricField)
% AREA_HAS_PRG_METRIC_PLOT_DATA - True if any session type has values for metricField

hasData = false;
for t = 1:length(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, metricField) && areaIdx <= length(typeData.(metricField)) ...
      && ~isempty(typeData.(metricField){areaIdx})
    hasData = true;
    return;
  end
end
end

function hasData = area_has_prg_plot_data(plotData, sessionTypes, areaIdx)
% AREA_HAS_PRG_PLOT_DATA - True if any session type has kappa values for this area

hasData = area_has_prg_metric_plot_data(plotData, sessionTypes, areaIdx, 'kappaMean');
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
