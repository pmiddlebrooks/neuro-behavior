%%
% Criticality Avalanche Size/Duration Examples (Manuscript)
%
% Runs single-window avalanche analysis on three example sessions
% (one spontaneous, one interval, one reach) with identical parameters,
% then plots size and duration CCDFs in a 1x6 figure:
%   spontaneous size | spontaneous duration |
%   interval size    | interval duration    |
%   reach size       | reach duration
%
% Variables (configure in this section):
%   exampleSessions  - 3x1 struct array with fields:
%                      .sessionType, .sessionName, .subjectName ('' for reach),
%                      .displayLabel
%   dataSource       - 'spikes' or 'lfp'
%   collectStart     - Analysis window start (seconds from session onset)
%   collectEnd       - Analysis window end (seconds); [] = session end
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56')
%   brainAreaCombinations  - Merged areas: default_manuscript_brain_area_combinations()
%   powerLawFitMethod      - 'clauset', 'plfit2023', or 'hybrid'
%   avalancheDetectionMode - 'fixedBinMedian' or 'meanIsiZero'
%   enableCircularPermutations - Overlay pooled circular-shuffle CCDFs
%   saveFigure       - Export PNG/EPS to dropPath/criticality_manuscript
%
% Goal:
%   Illustrate avalanche size and duration distributions across task types
%   using the same single-window pipeline as session_avalanche_distributions.m.

%% Paths
setup_criticality_manuscript_paths('criticality_av_examples');
paths = get_paths();

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
collectEnd = [];
brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
saveFigure = true;

powerLawFitMethod = 'plfit2023';
runClausetPlpva = false;
gofThreshold = 0.8;
avalancheDetectionMode = 'fixedBinMedian';

useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;

enableCircularPermutations = true;
nShuffles = 5;

plotConfig = struct();
plotConfig.observedMarkerSize = 5;
plotConfig.shuffleMarkerSize = 4;
plotConfig.fitLineWidth = 2.5;
plotConfig.tileSpacing = 'compact';
plotConfig.tilePadding = 'compact';
plotConfig.legendLocation = 'southwest';
plotConfig.axisLabelFontSize = 12;
plotConfig.tickLabelFontSize = 10;
plotConfig.axesLineWidth = 1.5;
plotConfig.observedMarkerFaceAlpha = 0.35;

[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();

fprintf('\n=== Criticality avalanche examples ===\n');
fprintf('Power-law fit method: %s\n', powerLawFitMethod);
fprintf('Avalanche detection mode: %s\n', avalancheDetectionMode);
fprintf('Brain area: %s\n', brainArea);
if enableCircularPermutations
  fprintf('Circular permutations: %d shuffles per area\n', nShuffles);
end

validate_example_sessions(exampleSessions);

numExamples = numel(exampleSessions);
exampleResults = repmat(struct('example', [], 'avData', [], 'collectEnd', nan), numExamples, 1);

for e = 1:numExamples
  ex = exampleSessions(e);
  fprintf('\n--- Example %d/%d [%s]: %s ---\n', e, numExamples, ex.sessionType, ex.sessionName);

  opts = build_example_load_opts(collectStart, collectEnd);
  loadArgs = build_session_load_args(ex.sessionType, ex.sessionName, opts, ex.subjectName);
  dataStruct = load_session_data(ex.sessionType, dataSource, loadArgs{:});

  sessionCollectEnd = collectEnd;
  if isempty(sessionCollectEnd)
    sessionCollectEnd = resolve_session_collect_end(dataStruct, collectStart);
  end
  windowDurationSec = sessionCollectEnd - collectStart;
  fprintf('  Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
    collectStart, sessionCollectEnd, windowDurationSec / 60);

  analysisConfig = build_example_av_analysis_config( ...
    windowDurationSec, avalancheDetectionMode, powerLawFitMethod, gofThreshold, ...
    runClausetPlpva, useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple, ...
    enableCircularPermutations, nShuffles, clausetPlfitPath, plfit2023Path);

  [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
    dataStruct, brainArea, brainAreaCombinations);
  if ~areaOk
    warning('criticality_av_examples:MissingArea', ...
      'Brain area "%s" not available for %s; skipping.', brainArea, ex.sessionName);
    continue;
  end

  areasToAnalyze = resolve_areas_to_analyze(dataStruct, brainArea, analysisConfig.nMinNeurons);
  if isempty(areasToAnalyze)
    warning('criticality_av_examples:NoAreas', ...
      'No areas meet nMinNeurons for %s; skipping.', ex.sessionName);
    continue;
  end

  areaIndex = areasToAnalyze(1);
  areaName = dataStruct.areas{areaIndex};
  fprintf('  Area %s...\n', areaName);
  avData = extract_area_avalanches(dataStruct, areaIndex, analysisConfig, ...
    collectStart, sessionCollectEnd);
  avData.areaName = areaName;

  if ~avData.hasAvalanches
    warning('criticality_av_examples:NoAvalanches', ...
      'No avalanches for %s (%s).', ex.sessionName, areaName);
    continue;
  end

  print_area_avalanche_summary(avData, enableCircularPermutations);
  exampleResults(e).example = ex;
  exampleResults(e).avData = avData;
  exampleResults(e).collectEnd = sessionCollectEnd;
end

fig = plot_av_examples_size_duration(exampleResults, brainArea, collectStart, plotConfig);

if saveFigure
  saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  plotBase = sprintf('criticality_av_examples_%s_%s', brainArea, powerLawFitMethod);
  if enableCircularPermutations
    plotBase = sprintf('%s_circ%d', plotBase, nShuffles);
  end
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
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

function opts = build_example_load_opts(collectStart, collectEnd)
% BUILD_EXAMPLE_LOAD_OPTS - Loader opts for example sessions
%
% Variables:
%   collectStart - Analysis window start (s)
%   collectEnd   - Analysis window end (s); [] = full session
%
% Goal:
%   Apply the same collectStart/collectEnd to spontaneous, interval, and reach.

opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;
end

function analysisConfig = build_example_av_analysis_config( ...
  windowDurationSec, avalancheDetectionMode, powerLawFitMethod, gofThreshold, ...
  runClausetPlpva, useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple, ...
  enableCircularPermutations, nShuffles, clausetPlfitPath, plfit2023Path)
% BUILD_EXAMPLE_AV_ANALYSIS_CONFIG - Avalanche config matching session_avalanche_distributions

analysisConfig = struct();
analysisConfig.slidingWindowSize = windowDurationSec;
analysisConfig.avStepSize = windowDurationSec;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.avalancheDetectionMode = avalancheDetectionMode;
if ~strcmpi(avalancheDetectionMode, 'meanIsiZero')
  analysisConfig.binSize = 0.05;
end
analysisConfig.thresholdFlag = 1;
analysisConfig.thresholdPct = 1;
analysisConfig.nMinNeurons = 20;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
analysisConfig.pcaFlag = 0;
analysisConfig.gofThreshold = gofThreshold;
analysisConfig.powerLawFitMethod = powerLawFitMethod;
analysisConfig.runClausetPlpva = runClausetPlpva;
analysisConfig.enableCircularPermutations = enableCircularPermutations;
analysisConfig.nShuffles = nShuffles;
analysisConfig.clausetPlfitPath = clausetPlfitPath;
analysisConfig.plfit2023Path = plfit2023Path;
end

function collectEnd = resolve_session_collect_end(dataStruct, collectStart)
% RESOLVE_SESSION_COLLECT_END - Full-session end time (s) when collectEnd is empty

collectEnd = [];
if isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
  collectEnd = max(dataStruct.spikeTimes);
elseif isfield(dataStruct, 'opts') && isstruct(dataStruct.opts) ...
    && isfield(dataStruct.opts, 'collectEnd') && ~isempty(dataStruct.opts.collectEnd)
  collectEnd = dataStruct.opts.collectEnd;
end

if isempty(collectEnd) || ~isfinite(collectEnd) || collectEnd <= collectStart
  error(['Could not resolve collectEnd for the full session. ', ...
    'Set collectEnd explicitly in the configuration.']);
end
end

function areasToAnalyze = resolve_areas_to_analyze(dataStruct, brainArea, nMinNeurons)
% RESOLVE_AREAS_TO_ANALYZE - Area indices that meet neuron-count threshold

if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
  candidateAreas = dataStruct.areasToTest(:);
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

function avData = extract_area_avalanches(dataStruct, areaIndex, analysisConfig, collectStart, collectEnd)
% EXTRACT_AREA_AVALANCHES - Bin, threshold, and fit one collect window
%
% Variables:
%   dataStruct      - From load_session_data
%   areaIndex       - Index into dataStruct.areas
%   analysisConfig  - Bin size and threshold settings
%   collectStart    - Window start (s)
%   collectEnd      - Window end (s)
%
% Goal:
%   Match session_avalanche_distributions single-window avalanche extraction.

avData = struct('hasAvalanches', false, 'sizes', [], 'durations', [], ...
  'tau', nan, 'alpha', nan, 'paramSD', nan, 'dcc', nan, 'scalingRelation', nan, ...
  'minSizeFit', nan, 'maxSizeFit', nan, ...
  'minDurFit', nan, 'maxDurFit', nan, 'sizeFitInfo', struct(), ...
  'durFitInfo', struct(), 'nAvalanches', 0, 'binSize', nan, ...
  'shuffleSizes', [], 'shuffleDurations', [], 'shuffleTau', nan, 'shuffleAlpha', nan, ...
  'nShufflesCompleted', 0);

timeRange = [collectStart, collectEnd];
neuronIds = dataStruct.idLabel{areaIndex};
binSizeVec = resolve_avalanche_bin_sizes(dataStruct, areaIndex, timeRange, analysisConfig);
binSize = binSizeVec(areaIndex);
avData.binSize = binSize;

aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIds, timeRange, binSize);

[sizes, durations, hasAvalanches] = compute_avalanche_sizes_durations_from_binned( ...
  aDataMat, analysisConfig);
if ~hasAvalanches
  return;
end

sizeFit = fit_avalanche_power_law(sizes, analysisConfig);
durFit = fit_avalanche_power_law(durations, analysisConfig);

avData.hasAvalanches = true;
avData.sizes = sizes;
avData.durations = durations;
avData.tau = sizeFit.exponent;
avData.alpha = durFit.exponent;
avData.minSizeFit = sizeFit.fitMin;
avData.maxSizeFit = sizeFit.fitMax;
avData.minDurFit = durFit.fitMin;
avData.maxDurFit = durFit.fitMax;
avData.sizeFitInfo = sizeFit;
avData.durFitInfo = durFit;
avData.nAvalanches = numel(sizes);

% Measured 1/σνz: ⟨S⟩(T) only for durations in the α power-law fit range
if isfinite(durFit.fitMin) && isfinite(durFit.fitMax) && durFit.fitMin < durFit.fitMax ...
    && numel(sizes) >= 2 && numel(durations) >= 2
  sizesCol = sizes(:);
  dursCol = durations(:);
  inDurFitRange = isfinite(sizesCol) & isfinite(dursCol) ...
    & sizesCol > 0 & dursCol > 0 ...
    & dursCol >= durFit.fitMin & dursCol <= durFit.fitMax;
  if nnz(inDurFitRange) >= 2 && numel(unique(dursCol(inDurFitRange))) >= 2
    [avData.paramSD, ~, ~] = size_given_duration( ...
      sizesCol(inDurFitRange), dursCol(inDurFitRange), ...
      'durmin', durFit.fitMin, 'durmax', durFit.fitMax);
  end
end
if isfinite(avData.tau) && isfinite(avData.alpha) && avData.tau > 1
  avData.scalingRelation = (avData.alpha - 1) / (avData.tau - 1);
end
if isfinite(avData.paramSD) && isfinite(avData.scalingRelation)
  avData.dcc = abs(avData.scalingRelation - avData.paramSD);
end

if isfield(analysisConfig, 'enableCircularPermutations') && analysisConfig.enableCircularPermutations
  nShufflesArea = 10;
  if isfield(analysisConfig, 'nShuffles') && ~isempty(analysisConfig.nShuffles)
    nShufflesArea = analysisConfig.nShuffles;
  end
  [avData.shuffleSizes, avData.shuffleDurations, avData.nShufflesCompleted] = ...
    pooled_circular_shuffle_avalanches(aDataMat, analysisConfig, nShufflesArea);
  if ~isempty(avData.shuffleSizes)
    shuffleSizeFit = fit_avalanche_power_law(avData.shuffleSizes, analysisConfig);
    avData.shuffleTau = shuffleSizeFit.exponent;
  end
  if ~isempty(avData.shuffleDurations)
    shuffleDurFit = fit_avalanche_power_law(avData.shuffleDurations, analysisConfig);
    avData.shuffleAlpha = shuffleDurFit.exponent;
  end
end
end

function [sizes, durations, hasAvalanches] = compute_avalanche_sizes_durations_from_binned(aDataMat, analysisConfig)
% COMPUTE_AVALANCHE_SIZES_DURATIONS_FROM_BINNED - Avalanche vectors from binned spike matrix

sizes = [];
durations = [];
hasAvalanches = false;

useSubsampling = isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling;
if useSubsampling
  numNeuronsArea = size(aDataMat, 2);
  nSubsamplesArea = analysisConfig.nSubsamples;
  nNeuronsSubsampleArea = min(analysisConfig.nNeuronsSubsample, numNeuronsArea);
  for s = 1:nSubsamplesArea
    if nNeuronsSubsampleArea == numNeuronsArea
      colIdx = 1:numNeuronsArea;
    else
      colIdx = randperm(numNeuronsArea, nNeuronsSubsampleArea);
    end
    wPopActivity = sum(aDataMat(:, colIdx), 2);
    avMetrics = compute_av_metrics_from_pop_activity(wPopActivity, analysisConfig);
    if ~isfinite(avMetrics.kappa)
      continue;
    end
    wPopActivity = apply_avalanche_population_threshold(wPopActivity, analysisConfig);
    zeroBins = find(wPopActivity == 0);
    if ~(numel(zeroBins) > 1 && any(diff(zeroBins) > 1))
      continue;
    end
    [sizesSub, dursSub] = getAvalanches(wPopActivity', 0.5, 1);
    sizes = [sizes; sizesSub(:)]; %#ok<AGROW>
    durations = [durations; dursSub(:)]; %#ok<AGROW>
  end
else
  wPopActivity = sum(aDataMat, 2);
  wPopActivity = apply_avalanche_population_threshold(wPopActivity, analysisConfig);
  zeroBins = find(wPopActivity == 0);
  if ~(numel(zeroBins) > 1 && any(diff(zeroBins) > 1))
    return;
  end
  [sizes, durations] = getAvalanches(wPopActivity', 0.5, 1);
end

sizes = sizes(:);
durations = durations(:);
hasAvalanches = ~isempty(sizes) && ~isempty(durations);
end

function [shuffleSizes, shuffleDurations, nCompleted] = pooled_circular_shuffle_avalanches(aDataMat, analysisConfig, nShuffles)
% POOLED_CIRCULAR_SHUFFLE_AVALANCHES - Pool avalanches across circular neuron shuffles

shuffleSizes = [];
shuffleDurations = [];
nCompleted = 0;

for shuffleIdx = 1:nShuffles
  permutedMat = circular_shuffle_binned_matrix(aDataMat);
  [sizesSub, durationsSub, hasAvalanches] = compute_avalanche_sizes_durations_from_binned( ...
    permutedMat, analysisConfig);
  if ~hasAvalanches
    continue;
  end
  shuffleSizes = [shuffleSizes; sizesSub(:)]; %#ok<AGROW>
  shuffleDurations = [shuffleDurations; durationsSub(:)]; %#ok<AGROW>
  nCompleted = nCompleted + 1;
end
end

function permutedMat = circular_shuffle_binned_matrix(aDataMat)
% CIRCULAR_SHUFFLE_BINNED_MATRIX - Independent circular shift per neuron column

permutedMat = aDataMat;
winSamples = size(aDataMat, 1);
if winSamples < 1
  return;
end

for neuronIdx = 1:size(aDataMat, 2)
  shiftAmount = randi([1, winSamples]);
  permutedMat(:, neuronIdx) = circshift(aDataMat(:, neuronIdx), shiftAmount);
end
end

function print_area_avalanche_summary(avData, enableCircularPermutations)
% PRINT_AREA_AVALANCHE_SUMMARY - Command-window fit summary for one area

fprintf('  Size:  tau = %.3f, x in [%.3g, %.3g]', ...
  avData.tau, avData.minSizeFit, avData.maxSizeFit);
print_fit_diagnostics(avData.sizeFitInfo);
fprintf('\n');
binSize = resolve_avalanche_duration_bin_size(avData);
fprintf('  Dur:   alpha = %.3f, x in [%.3g, %.3g] s', ...
  avData.alpha, avData.minDurFit * binSize, avData.maxDurFit * binSize);
print_fit_diagnostics(avData.durFitInfo);
fprintf('\n');
fprintf('  n = %d avalanches\n', avData.nAvalanches);
if enableCircularPermutations && ~isempty(avData.shuffleSizes)
  fprintf('  Shuffle (pooled): n = %d avalanches over %d shuffles\n', ...
    numel(avData.shuffleSizes), avData.nShufflesCompleted);
end
end

function print_fit_diagnostics(fitInfo)
% PRINT_FIT_DIAGNOSTICS - Optional p-value / tail count for command window

if isempty(fitInfo) || ~isstruct(fitInfo)
  return;
end
extras = {};
if isfield(fitInfo, 'nTail') && fitInfo.nTail > 0
  extras{end+1} = sprintf('nTail=%d', fitInfo.nTail); %#ok<AGROW>
end
if isfield(fitInfo, 'pValue') && isfinite(fitInfo.pValue)
  extras{end+1} = sprintf('p=%.3f', fitInfo.pValue); %#ok<AGROW>
end
if ~isempty(extras)
  fprintf(' (%s)', strjoin(extras, ', '));
end
end

function fig = plot_av_examples_size_duration(exampleResults, brainArea, collectStart, plotConfig)
% PLOT_AV_EXAMPLES_SIZE_DURATION - 1x6 size/duration CCDFs across example sessions
%
% Variables:
%   exampleResults - Struct array with .example and .avData per session
%   brainArea      - Area label for sgtitle
%   collectStart   - Shared window start (s)
%   plotConfig     - Marker / font options
%
% Goal:
%   One row: size then duration for spontaneous, interval, reach (in that order).

plotConfig = fill_default_avalanche_plot_config(plotConfig);

fig = figure('Color', 'w', 'Position', [40 80 1680 320], ...
  'Name', 'Avalanche size/duration examples');
tileLayout = tiledlayout(fig, 1, 6, ...
  'TileSpacing', plotConfig.tileSpacing, 'Padding', plotConfig.tilePadding);

sessionOrder = {'spontaneous', 'interval', 'reach'};
plottedCount = 0;

for s = 1:numel(sessionOrder)
  sessionType = sessionOrder{s};
  resultIdx = find_example_result_index(exampleResults, sessionType);
  axSize = nexttile(tileLayout, (s - 1) * 2 + 1);
  axDur = nexttile(tileLayout, (s - 1) * 2 + 2);

  if isempty(resultIdx) || isempty(exampleResults(resultIdx).avData) ...
      || ~exampleResults(resultIdx).avData.hasAvalanches
    title(axSize, sprintf('%s size (no data)', sessionType), 'Interpreter', 'none');
    title(axDur, sprintf('%s duration (no data)', sessionType), 'Interpreter', 'none');
    continue;
  end

  ex = exampleResults(resultIdx).example;
  avData = exampleResults(resultIdx).avData;
  if isfield(ex, 'displayLabel') && ~isempty(ex.displayLabel)
    panelLabel = ex.displayLabel;
  else
    panelLabel = ex.sessionType;
  end

  panelPlotConfig = plotConfig;
  panelPlotConfig.sessionType = ex.sessionType;

  plot_avalanche_ccdf_with_fit(axSize, avData.sizes, avData.tau, ...
    avData.minSizeFit, avData.maxSizeFit, 'Sizes', '\tau', avData.sizeFitInfo, ...
    avData.shuffleSizes, panelPlotConfig);
  title(axSize, sprintf('%s | size (\\tau = %.2f)', panelLabel, avData.tau), ...
    'Interpreter', 'tex');

  binSize = resolve_avalanche_duration_bin_size(avData);
  plot_avalanche_ccdf_with_fit(axDur, avData.durations * binSize * 1000, avData.alpha, ...
    avData.minDurFit * binSize * 1000, avData.maxDurFit * binSize * 1000, ...
    'Durations (ms)', '\alpha', avData.durFitInfo, ...
    avData.shuffleDurations * binSize * 1000, panelPlotConfig);
  title(axDur, sprintf('%s | duration (\\alpha = %.2f)', panelLabel, avData.alpha), ...
    'Interpreter', 'tex');

  plottedCount = plottedCount + 1;
end

if plottedCount == 0
  error('No avalanche results available to plot.');
end

sgtitle(tileLayout, sprintf('%s avalanche size & duration examples | start %.0f s', ...
  brainArea, collectStart), 'FontSize', 13, 'Interpreter', 'none');
end

function resultIdx = find_example_result_index(exampleResults, sessionType)
% FIND_EXAMPLE_RESULT_INDEX - Index of exampleResults matching sessionType

resultIdx = [];
for e = 1:numel(exampleResults)
  if ~isfield(exampleResults(e), 'example') || isempty(exampleResults(e).example)
    continue;
  end
  if strcmpi(exampleResults(e).example.sessionType, sessionType)
    resultIdx = e;
    return;
  end
end
end

function plot_avalanche_ccdf_with_fit(ax, values, exponent, fitMin, fitMax, xLabelText, ...
  exponentLabel, fitInfo, shuffleValues, plotConfig)
% PLOT_AVALANCHE_CCDF_WITH_FIT - Log-log CCDF with optional shuffle overlay
%
% Variables:
%   ax            - Axes handle
%   values        - Avalanche sizes or durations
%   exponent      - Power-law exponent (tau or alpha)
%   fitMin, fitMax - Fitted scaling range
%   xLabelText    - X-axis label
%   exponentLabel - Display name for exponent ('\tau' or '\alpha')
%   fitInfo       - Optional struct from fit_avalanche_power_law
%   shuffleValues - Optional pooled shuffle avalanches for overlay CCDF
%   plotConfig    - Marker / font options (includes .sessionType for color)

if nargin < 8 || isempty(fitInfo)
  fitInfo = struct();
end
if nargin < 9
  shuffleValues = [];
end
if nargin < 10 || isempty(plotConfig)
  plotConfig = struct();
end
plotConfig = fill_default_avalanche_plot_config(plotConfig);

values = values(values > 0 & isfinite(values));
if isempty(values)
  cla(ax);
  title(ax, sprintf('%s (no data)', xLabelText));
  return;
end

uniqueVals = unique(values);
cdfY = arrayfun(@(x) mean(values >= x), uniqueVals);

observedColor = colors_for_tasks(plotConfig.sessionType);
markerArea = plotConfig.observedMarkerSize .^ 2;

hold(ax, 'on');
scatter(ax, uniqueVals, cdfY, markerArea, 'filled', ...
  'MarkerEdgeColor', observedColor, 'MarkerFaceColor', observedColor, ...
  'MarkerFaceAlpha', plotConfig.observedMarkerFaceAlpha, ...
  'DisplayName', 'Obs');

shufflePlotted = false;
shuffleValues = shuffleValues(shuffleValues > 0 & isfinite(shuffleValues));
if ~isempty(shuffleValues)
  uniqueShuffle = unique(shuffleValues);
  cdfShuffle = arrayfun(@(x) mean(shuffleValues >= x), uniqueShuffle);
  plot(ax, uniqueShuffle, cdfShuffle, 'o', 'Color', [0.55, 0.55, 0.55], ...
    'MarkerFaceColor', [0.75, 0.75, 0.75], 'MarkerSize', plotConfig.shuffleMarkerSize, ...
    'DisplayName', 'Shuff');
  shufflePlotted = true;
end

fitPlotted = false;
if isfinite(exponent) && exponent > 1 && isfinite(fitMin) && isfinite(fitMax) ...
    && fitMin > 0 && fitMax > fitMin
  % Fit overlay available via fitMin/fitMax; kept off to match session script style
  fitPlotted = false; %#ok<NASGU>
end

set(ax, 'XScale', 'log', 'YScale', 'log', 'FontSize', plotConfig.tickLabelFontSize, ...
  'LineWidth', plotConfig.axesLineWidth);
axis(ax, 'square');
xlabel(ax, xLabelText, 'FontSize', plotConfig.axisLabelFontSize);
ylabel(ax, 'P(X \geq x)', 'FontSize', plotConfig.axisLabelFontSize);
if shufflePlotted
  legend(ax, 'Location', plotConfig.legendLocation, 'FontSize', plotConfig.tickLabelFontSize);
end

pText = '';
if isstruct(fitInfo) && isfield(fitInfo, 'pValue') && isfinite(fitInfo.pValue)
  pText = sprintf(', p=%.2f', fitInfo.pValue);
end
title(ax, sprintf('%s (%s = %.2f%s)', xLabelText, exponentLabel, exponent, pText));
hold(ax, 'off');
end

function plotConfig = fill_default_avalanche_plot_config(plotConfig)
% FILL_DEFAULT_AVALANCHE_PLOT_CONFIG - Default CCDF marker/font settings

if ~isfield(plotConfig, 'observedMarkerSize') || isempty(plotConfig.observedMarkerSize)
  plotConfig.observedMarkerSize = 5;
end
if ~isfield(plotConfig, 'shuffleMarkerSize') || isempty(plotConfig.shuffleMarkerSize)
  plotConfig.shuffleMarkerSize = 4;
end
if ~isfield(plotConfig, 'fitLineWidth') || isempty(plotConfig.fitLineWidth)
  plotConfig.fitLineWidth = 2.5;
end
if ~isfield(plotConfig, 'tileSpacing') || isempty(plotConfig.tileSpacing)
  plotConfig.tileSpacing = 'compact';
end
if ~isfield(plotConfig, 'tilePadding') || isempty(plotConfig.tilePadding)
  plotConfig.tilePadding = 'compact';
end
if ~isfield(plotConfig, 'legendLocation') || isempty(plotConfig.legendLocation)
  plotConfig.legendLocation = 'southwest';
end
if ~isfield(plotConfig, 'axisLabelFontSize') || isempty(plotConfig.axisLabelFontSize)
  plotConfig.axisLabelFontSize = 12;
end
if ~isfield(plotConfig, 'tickLabelFontSize') || isempty(plotConfig.tickLabelFontSize)
  plotConfig.tickLabelFontSize = 10;
end
if ~isfield(plotConfig, 'axesLineWidth') || isempty(plotConfig.axesLineWidth)
  plotConfig.axesLineWidth = 1.5;
end
if ~isfield(plotConfig, 'observedMarkerFaceAlpha') || isempty(plotConfig.observedMarkerFaceAlpha)
  plotConfig.observedMarkerFaceAlpha = 0.35;
end
if ~isfield(plotConfig, 'sessionType')
  plotConfig.sessionType = '';
end
end

function binSize = resolve_avalanche_duration_bin_size(avData)
% RESOLVE_AVALANCHE_DURATION_BIN_SIZE - Bin width (s) for duration conversion

if isfield(avData, 'binSize') && isscalar(avData.binSize) && isfinite(avData.binSize) && avData.binSize > 0
  binSize = avData.binSize;
else
  binSize = 0.05;
  warning('criticality_av_examples:missingBinSize', ...
    'avData.binSize missing; assuming %.3f s for duration conversion.', binSize);
end
end
