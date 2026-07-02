%%
% Session Avalanche Metrics vs Bin Size (Manuscript)
%
% For one session and brain area, sweeps fixedBinMedian avalanche analysis
% across bin sizes and plots tau, alpha, paramSD, decades, and dcc.
%
% Variables (configure in this section):
%   sessionType        - 'spontaneous', 'interval', 'reach', 'schall'
%   sessionName        - Session identifier
%   subjectName        - Required for spontaneous/interval; '' for reach
%   dataSource         - 'spikes' or 'lfp'
%   collectStart       - Window start (seconds from session onset)
%   collectEnd         - Window end (seconds)
%   brainArea              - Single or merged area (e.g. 'M23M56')
%   brainAreaCombinations  - Merged areas from default_manuscript_brain_area_combinations
%   binSizes           - Vector of bin widths (seconds) to test
%   powerLawFitMethods - Which fits to sweep (subset of 'clauset', 'plfit2023', 'hybrid')
%                          e.g. {'hybrid'} or {'clauset', 'hybrid'}
%   useSubsampling     - If true, metrics = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%   saveFigure         - Export PNG/EPS to dropPath/criticality_manuscript
%   saveAnalysisResults - If true, save binSizeResults after %% Analysis
%   analysisResultsFile - Path to .mat cache; '' = default
%
% Sections:
%   %% Analysis  - load session, sweep bin sizes, optionally cache results
%   %% Plotting  - metric-vs-bin-size figure (loads .mat if not in workspace)
%
% Goal:
%   Compare avalanche criticality metrics as a function of fixed bin size
%   for one brain area using the same bin/threshold pipeline as
%   session_avalanche_distributions.m and metrics from compute_av_metrics_from_pop_activity.

%% Configuration
% sessionType = 'interval';
% subjectName = 'ey9166';
% sessionName = 'ey9166_2026_04_03';
% dataSource = 'spikes';

collectStart = 0;
collectEnd = 45 * 60;
windowDurationSec = collectEnd - collectStart;

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
binSizes = 0.01:0.005:0.1;
saveFigure = true;

% Which power-law fit methods to sweep (any non-empty subset):
% powerLawFitMethods = {'hybrid'};
% powerLawFitMethods = {'clauset', 'hybrid'};
powerLawFitMethods = {'clauset', 'plfit2023', 'hybrid'};
powerLawFitMethods = normalize_power_law_fit_methods(powerLawFitMethods);
gofThreshold = 0.8;
runClausetPlpva = false;

useSubsampling = true;
nSubsamples = 20;
nNeuronsSubsample = 50;
minNeuronsMultiple = 1.25;

saveAnalysisResults = false;
analysisResultsFile = '';

plotConfig = struct();
plotConfig.axisLabelFontSize = 14;
plotConfig.tickLabelFontSize = 12;
plotConfig.axesLineWidth = 1.5;
plotConfig.markerSize = 6;
plotConfig.markerFaceAlpha = 0.35;
plotConfig.figureWidthInches = 6.5;

opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;

analysisConfig = struct();
analysisConfig.avalancheDetectionMode = 'fixedBinMedian';
analysisConfig.thresholdFlag = 1;
analysisConfig.thresholdPct = 1;
analysisConfig.nMinNeurons = 20;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
analysisConfig.gofThreshold = gofThreshold;
analysisConfig.runClausetPlpva = runClausetPlpva;

% Paths
paths = get_paths();
scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('session_avalanche_params_per_bin_size'));
end
srcPath = fullfile(scriptDir, '..');
addpath(scriptDir, '-begin');
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

[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();
analysisConfig.clausetPlfitPath = clausetPlfitPath;
analysisConfig.plfit2023Path = plfit2023Path;

fprintf('\n=== Session Avalanche Metrics vs Bin Size ===\n');
fprintf('Power-law fit methods: %s\n', strjoin(powerLawFitMethods, ', '));
fprintf('Avalanche detection mode: fixedBinMedian\n');
fprintf('Bin sizes (s): %s\n', mat2str(binSizes, 3));
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Brain area: %s\n', brainArea);
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, windowDurationSec / 60);

%% Analysis — load session, sweep bin sizes, optionally cache results
subjectNameForLoad = subjectName;
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

[dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations);
if ~areaOk
  error('Brain area "%s" not available in this session.', brainArea);
end

runMeta = struct( ...
  'sessionType', sessionType, ...
  'sessionName', sessionName, ...
  'subjectName', subjectName, ...
  'dataSource', dataSource, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'powerLawFitMethods', {powerLawFitMethods}, ...
  'binSizes', binSizes, ...
  'useSubsampling', useSubsampling, ...
  'saveFigure', saveFigure);

binSizeResults = sweep_avalanche_metrics_over_bin_sizes(dataStruct, analysisConfig, runMeta);

if saveAnalysisResults
  resultsFile = resolve_bin_size_results_file(paths, sessionName, analysisResultsFile);
  save(resultsFile, 'binSizeResults', '-v7.3');
  fprintf('\nSaved analysis results: %s\n', resultsFile);
end

% Plotting — metric-vs-bin-size figure (re-run to tweak formatting)
if ~exist('binSizeResults', 'var') || isempty(binSizeResults)
  resultsFile = resolve_bin_size_results_file(paths, sessionName, analysisResultsFile);
  if ~isfile(resultsFile)
    error(['binSizeResults not in workspace and results file not found: %s. ', ...
      'Run %% Analysis first or set analysisResultsFile.'], resultsFile);
  end
  loaded = load(resultsFile, 'binSizeResults');
  binSizeResults = loaded.binSizeResults;
  fprintf('\nLoaded analysis results: %s\n', resultsFile);
end

plotConfig.sessionType = binSizeResults.meta.sessionType;
plot_bin_size_avalanche_metrics(binSizeResults, paths, plotConfig);

fprintf('\n=== Done ===\n');

%% Local functions

function resultsFile = resolve_bin_size_results_file(paths, sessionName, analysisResultsFile)
% RESOLVE_BIN_SIZE_RESULTS_FILE - Default path for cached bin-size sweep results

if ~isempty(analysisResultsFile)
  resultsFile = analysisResultsFile;
  return;
end

saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
sessionTag = matlab.lang.makeValidName(sessionName);
resultsFile = fullfile(saveDir, sprintf('session_avalanche_bin_size_%s.mat', sessionTag));
end

function binSizeResults = sweep_avalanche_metrics_over_bin_sizes(dataStruct, analysisConfig, runMeta)
% SWEEP_AVALANCHE_METRICS_OVER_BIN_SIZES - Metrics for one area across bin sizes and fit methods
%
% Variables:
%   dataStruct      - Loaded session data (already restricted to brainArea)
%   analysisConfig  - Base config (binSize and powerLawFitMethod set per sweep)
%   runMeta         - Session metadata including binSizes and powerLawFitMethods
%
% Returns:
%   binSizeResults - Struct with metrics per fit method

runMeta = runMeta(1);
binSizes = runMeta.binSizes(:)';
fitMethods = normalize_power_law_fit_methods(runMeta.powerLawFitMethods);
metricNames = {'tau', 'alpha', 'paramSD', 'decades', 'dcc'};
nBins = numel(binSizes);
nMethods = numel(fitMethods);

areasToAnalyze = resolve_areas_to_analyze(dataStruct, runMeta.brainArea, analysisConfig.nMinNeurons);
if isempty(areasToAnalyze)
  error('No areas meet minimum neuron count (%d) for "%s".', ...
    analysisConfig.nMinNeurons, runMeta.brainArea);
end
if numel(areasToAnalyze) > 1
  warning('session_avalanche_params_per_bin_size:MultipleAreas', ...
    'Multiple areas match "%s"; analyzing area index %d (%s).', ...
    runMeta.brainArea, areasToAnalyze(1), dataStruct.areas{areasToAnalyze(1)});
end
areaIndex = areasToAnalyze(1);

binSizeResults = struct();
binSizeResults.meta = runMeta;
binSizeResults.areaName = dataStruct.areas{areaIndex};
binSizeResults.binSizes = binSizes;
binSizeResults.fitMethods = fitMethods;
binSizeResults.metricsByMethod = struct();

for iMethod = 1:nMethods
  methodTag = matlab.lang.makeValidName(fitMethods{iMethod});
  binSizeResults.metricsByMethod.(methodTag) = struct();
  for iMetric = 1:numel(metricNames)
    binSizeResults.metricsByMethod.(methodTag).(metricNames{iMetric}) = nan(1, nBins);
  end
end

fprintf('\nSweeping %d bin sizes x %d fit methods for area %s...\n', ...
  nBins, nMethods, binSizeResults.areaName);

for iBin = 1:nBins
  config = analysisConfig;
  config.binSize = binSizes(iBin);

  fprintf('\n--- binSize = %.3f s (%.1f ms) ---\n', binSizes(iBin), binSizes(iBin) * 1000);
  avalancheSamples = collect_avalanche_samples_for_bin_size(dataStruct, areaIndex, config, ...
    runMeta.collectStart, runMeta.collectEnd);

  for iMethod = 1:nMethods
    config.powerLawFitMethod = fitMethods{iMethod};
    areaMetrics = compute_av_metrics_from_avalanche_samples(avalancheSamples, config);
    methodTag = matlab.lang.makeValidName(fitMethods{iMethod});

    for iMetric = 1:numel(metricNames)
      metricName = metricNames{iMetric};
      binSizeResults.metricsByMethod.(methodTag).(metricName)(iBin) = areaMetrics.(metricName);
    end

    fprintf('  [%s] tau = %.3f, alpha = %.3f, paramSD = %.3f, decades = %.3f, dcc = %.3f\n', ...
      fitMethods{iMethod}, areaMetrics.tau, areaMetrics.alpha, areaMetrics.paramSD, ...
      areaMetrics.decades, areaMetrics.dcc);
  end
end
end

function avalancheSamples = collect_avalanche_samples_for_bin_size(dataStruct, areaIndex, analysisConfig, collectStart, collectEnd)
% COLLECT_AVALANCHE_SAMPLES_FOR_BIN_SIZE - Sizes/durations per subsample (or once)
%
% Returns:
%   avalancheSamples - Struct array with fields .sizes and .durations

timeRange = [collectStart, collectEnd];
neuronIds = dataStruct.idLabel{areaIndex};
binSize = analysisConfig.binSize;

aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIds, timeRange, binSize);

useSubsampling = isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling;
if useSubsampling
  numNeuronsArea = size(aDataMat, 2);
  nSubsamplesArea = analysisConfig.nSubsamples;
  nNeuronsSubsampleArea = min(analysisConfig.nNeuronsSubsample, numNeuronsArea);
  avalancheSamples = repmat(struct('sizes', [], 'durations', []), 1, nSubsamplesArea);

  for s = 1:nSubsamplesArea
    if nNeuronsSubsampleArea == numNeuronsArea
      colIdx = 1:numNeuronsArea;
    else
      colIdx = randperm(numNeuronsArea, nNeuronsSubsampleArea);
    end
    wPopActivity = sum(aDataMat(:, colIdx), 2);
    [avalancheSamples(s).sizes, avalancheSamples(s).durations] = ...
      extract_sizes_durations_from_pop_activity(wPopActivity, analysisConfig);
  end
else
  wPopActivity = sum(aDataMat, 2);
  [sizes, durations] = extract_sizes_durations_from_pop_activity(wPopActivity, analysisConfig);
  avalancheSamples = struct('sizes', sizes, 'durations', durations);
end
end

function areaMetrics = compute_av_metrics_from_avalanche_samples(avalancheSamples, analysisConfig)
% COMPUTE_AV_METRICS_FROM_AVALANCHE_SAMPLES - Power-law metrics from pooled avalanche samples

metricNames = {'tau', 'alpha', 'paramSD', 'decades', 'dcc'};
areaMetrics = struct();
for iMetric = 1:numel(metricNames)
  areaMetrics.(metricNames{iMetric}) = nan;
end

nSamples = numel(avalancheSamples);
metricSweeps = struct();
for iMetric = 1:numel(metricNames)
  metricSweeps.(metricNames{iMetric}) = nan(1, nSamples);
end

for s = 1:nSamples
  sampleMetrics = compute_av_metrics_from_sizes_durations( ...
    avalancheSamples(s).sizes, avalancheSamples(s).durations, analysisConfig);
  for iMetric = 1:numel(metricNames)
    metricName = metricNames{iMetric};
    metricSweeps.(metricName)(s) = sampleMetrics.(metricName);
  end
end

for iMetric = 1:numel(metricNames)
  metricName = metricNames{iMetric};
  areaMetrics.(metricName) = nanmean(metricSweeps.(metricName));
end
end

function [sizes, durations] = extract_sizes_durations_from_pop_activity(wPopActivity, analysisConfig)
% EXTRACT_SIZES_DURATIONS_FROM_POP_ACTIVITY - Avalanche vectors from population trace

sizes = [];
durations = [];

wPopActivity = apply_avalanche_population_threshold(wPopActivity(:), analysisConfig);
zeroBins = find(wPopActivity == 0);
if ~(numel(zeroBins) > 1 && any(diff(zeroBins) > 1))
  return;
end

[sizes, durations] = getAvalanches(wPopActivity', 0.5, 1);
sizes = sizes(:);
durations = durations(:);
end

function areaMetrics = compute_av_metrics_from_sizes_durations(sizes, durations, analysisConfig)
% COMPUTE_AV_METRICS_FROM_SIZES_DURATIONS - Metrics for one avalanche sample

metricNames = {'tau', 'alpha', 'paramSD', 'decades', 'dcc'};
areaMetrics = struct();
for iMetric = 1:numel(metricNames)
  areaMetrics.(metricNames{iMetric}) = nan;
end

if isempty(sizes) || isempty(durations)
  return;
end

plMetrics = avalanche_power_law_metrics(sizes, durations, analysisConfig);
areaMetrics.dcc = distance_to_criticality(plMetrics.tau, plMetrics.alpha, plMetrics.paramSD);
areaMetrics.decades = plMetrics.decades;
areaMetrics.tau = plMetrics.tau;
areaMetrics.alpha = plMetrics.alpha;
areaMetrics.paramSD = plMetrics.paramSD;
end

function areasToAnalyze = resolve_areas_to_analyze(dataStruct, brainArea, nMinNeurons)
% RESOLVE_AREAS_TO_ANALYZE - Area indices to process

if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
  candidateAreas = dataStruct.areasToTest(:)';
elseif ~isempty(brainArea)
  areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
  if isempty(areaIdx)
    areasToAnalyze = [];
    return;
  end
  candidateAreas = areaIdx;
else
  candidateAreas = 1:numel(dataStruct.areas);
end

areasToAnalyze = [];
for areaIndex = candidateAreas
  if numel(dataStruct.idMatIdx{areaIndex}) >= nMinNeurons
    areasToAnalyze(end+1) = areaIndex; %#ok<AGROW>
  end
end
end

function plot_bin_size_avalanche_metrics(binSizeResults, paths, plotConfig)
% PLOT_BIN_SIZE_AVALANCHE_METRICS - Line plots of metrics vs bin size per fit method

runMeta = binSizeResults.meta(1);
plotConfig = fill_default_bin_size_plot_config(plotConfig);
binSizesMs = binSizeResults.binSizes * 1000;
fitMethods = normalize_power_law_fit_methods(binSizeResults.fitMethods);

metricPanels = {
  'tau', '\tau'
  'alpha', '\alpha'
  'paramSD', 'paramSD'
  'decades', 'decades'
  'dcc', 'dcc'
  };

figName = sprintf('Avalanche metrics vs bin size | %s', runMeta.sessionName);
fig = get_task_figure_handle(runMeta.sessionType, 'bin_size_sweep', '', figName);
tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for iPanel = 1:size(metricPanels, 1)
  metricName = metricPanels{iPanel, 1};
  yLabelText = metricPanels{iPanel, 2};

  ax = nexttile(iPanel);
  hold(ax, 'on');
  legendHandles = gobjects(numel(fitMethods), 1);

  for iMethod = 1:numel(fitMethods)
    methodTag = matlab.lang.makeValidName(fitMethods{iMethod});
    methodColor = get_power_law_fit_method_color(fitMethods{iMethod});
    metricValues = binSizeResults.metricsByMethod.(methodTag).(metricName);

    legendHandles(iMethod) = scatter(ax, binSizesMs, metricValues, plotConfig.markerSize .^ 2, ...
      'filled', 'MarkerEdgeColor', methodColor, 'MarkerFaceColor', methodColor, ...
      'MarkerFaceAlpha', plotConfig.markerFaceAlpha, 'DisplayName', fitMethods{iMethod});
    plot(ax, binSizesMs, metricValues, '-', 'Color', methodColor, 'LineWidth', 1.5, ...
      'HandleVisibility', 'off');
  end

  set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
  xlabel(ax, 'Bin size (ms)', 'FontSize', plotConfig.axisLabelFontSize);
  ylabel(ax, yLabelText, 'FontSize', plotConfig.axisLabelFontSize);
  title(ax, yLabelText);
  legend(ax, legendHandles, fitMethods, 'Location', 'best', ...
    'FontSize', plotConfig.tickLabelFontSize);
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(fig, sprintf('%s | %s | fixedBinMedian [%.0f–%.0f s]', ...
  runMeta.sessionName, binSizeResults.areaName, runMeta.collectStart, runMeta.collectEnd), ...
  'FontWeight', 'bold', 'Interpreter', 'none');

apply_portrait_figure_size(fig, plotConfig.figureWidthInches, 3, 2);

if runMeta.saveFigure
  saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  areaTag = matlab.lang.makeValidName(binSizeResults.areaName);
  methodTag = strjoin(cellfun(@matlab.lang.makeValidName, fitMethods, 'UniformOutput', false), '_');
  plotBase = sprintf('session_avalanche_bin_size_%s_%s_%s_%.0f-%.0fs', ...
    runMeta.sessionName, areaTag, methodTag, runMeta.collectStart, runMeta.collectEnd);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
end
end

function fitMethods = normalize_power_law_fit_methods(powerLawFitMethods)
% NORMALIZE_POWER_LAW_FIT_METHODS - Validate user-selected fit methods to sweep
%
% Variables:
%   powerLawFitMethods - Char, string, or cell of method names
%
% Returns:
%   fitMethods - Row cell of unique valid method names

validMethods = {'clauset', 'plfit2023', 'hybrid'};

if nargin < 1 || isempty(powerLawFitMethods)
  error('powerLawFitMethods must be a non-empty char, string, or cell array.');
end

if ischar(powerLawFitMethods) || isstring(powerLawFitMethods)
  powerLawFitMethods = cellstr(powerLawFitMethods);
end
if ~iscell(powerLawFitMethods)
  error('powerLawFitMethods must be a char, string, or cell array.');
end

fitMethods = cell(1, numel(powerLawFitMethods));
nValid = 0;
for i = 1:numel(powerLawFitMethods)
  methodName = lower(strtrim(char(powerLawFitMethods{i})));
  if isempty(methodName)
    continue;
  end
  if ~any(strcmp(methodName, validMethods))
    error('Unknown powerLawFitMethod "%s". Use ''clauset'', ''plfit2023'', or ''hybrid''.', ...
      powerLawFitMethods{i});
  end
  if any(strcmp(fitMethods(1:nValid), methodName))
    continue;
  end
  nValid = nValid + 1;
  fitMethods{nValid} = methodName;
end
fitMethods = fitMethods(1:nValid);

if isempty(fitMethods)
  error('powerLawFitMethods is empty after validation.');
end
end

function methodColor = get_power_law_fit_method_color(methodName)
% GET_POWER_LAW_FIT_METHOD_COLOR - Stable color per fit method name

switch lower(strtrim(char(methodName)))
  case 'clauset'
    methodColor = [0.15, 0.45, 0.85];
  case 'plfit2023'
    methodColor = [0.90, 0.40, 0.15];
  case 'hybrid'
    methodColor = [0.20, 0.65, 0.30];
  otherwise
    methodColor = [0.35, 0.35, 0.35];
end
end

function plotConfig = fill_default_bin_size_plot_config(plotConfig)
if nargin < 1 || isempty(plotConfig)
  plotConfig = struct();
end
if ~isfield(plotConfig, 'axisLabelFontSize') || isempty(plotConfig.axisLabelFontSize)
  plotConfig.axisLabelFontSize = 14;
end
if ~isfield(plotConfig, 'tickLabelFontSize') || isempty(plotConfig.tickLabelFontSize)
  plotConfig.tickLabelFontSize = 12;
end
if ~isfield(plotConfig, 'axesLineWidth') || isempty(plotConfig.axesLineWidth)
  plotConfig.axesLineWidth = 1.5;
end
if ~isfield(plotConfig, 'markerSize') || isempty(plotConfig.markerSize)
  plotConfig.markerSize = 6;
end
if ~isfield(plotConfig, 'markerFaceAlpha') || isempty(plotConfig.markerFaceAlpha)
  plotConfig.markerFaceAlpha = 0.35;
end
if ~isfield(plotConfig, 'figureWidthInches') || isempty(plotConfig.figureWidthInches)
  plotConfig.figureWidthInches = 6.5;
end
end

function fig = get_task_figure_handle(sessionType, plotKind, cellType, figName)
% GET_TASK_FIGURE_HANDLE - Reuse or create a task-specific figure window

figNumber = figure_number_for_task(sessionType, plotKind, cellType);
fig = figure(figNumber);
set(fig, 'Color', 'w', 'Name', figName);
clf(fig);
end

function apply_portrait_figure_size(fig, figureWidthInches, nRows, nCols)
% APPLY_PORTRAIT_FIGURE_SIZE - Size figure for portrait PDF export

layoutPadIn = 0.55;
titlePadIn = 0.45;
usableWidth = max(figureWidthInches - layoutPadIn, 1);
panelWidth = usableWidth / max(nCols, 1);
panelHeight = panelWidth;
figHeight = nRows * panelHeight + layoutPadIn + titlePadIn;
figHeight = min(figHeight, 9.5);

set(fig, 'Units', 'inches', 'Position', [1, 1, figureWidthInches, figHeight]);
set(fig, 'PaperUnits', 'inches', 'PaperSize', [figureWidthInches, figHeight], ...
  'PaperPositionMode', 'auto');
end
