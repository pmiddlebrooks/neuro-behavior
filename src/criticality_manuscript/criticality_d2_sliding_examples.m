%%
% Criticality d2 Sliding-Window Examples (Manuscript)
%
% Runs overlapping sliding-window d2 analysis on three example sessions
% (one spontaneous, one interval, one reach) with identical AR parameters,
% then plots all d2 traces on a single axis (time in minutes).
%
% Variables (configure in this section):
%   exampleSessions  - 3x1 struct array with fields:
%                      .sessionType, .sessionName, .subjectName ('' for reach)
%   dataSource       - 'spikes' or 'lfp'
%   collectStart     - Analysis window start (seconds from session onset)
%   collectEnd       - Analysis window end (seconds); [] = session end (reach)
%   brainArea        - Area to plot (e.g. 'M56')
%   saveFigure       - Export PNG to dropPath/criticality_manuscript
%
% Goal:
%   Illustrate session-wise d2 time courses across task types using the same
%   overlapping sliding-window pipeline as run_criticality_ar.m.

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
  'sessionName', 'AB6_27-Mar-2025 14_04_12_NeuroBeh', ...
  'displayLabel', 'reach');

dataSource = 'spikes';
collectStart = 0;
collectEnd = 45 * 60;
brainArea = 'M56';
saveFigure = false;

% Overlapping sliding-window d2 settings (aligned with run_criticality_ar.m)
slidingWindowSize = 30;   % seconds
stepSize = 0.5;           % seconds; overlap when step < window
stepSize = 30;           % seconds; overlap when step < window
useLog10D2 = true;
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;

analysisConfig = struct();
analysisConfig.slidingWindowSize = slidingWindowSize;
analysisConfig.stepSize = stepSize;
analysisConfig.binSize = 0.025;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = true;
analysisConfig.nShuffles = 20;
analysisConfig.normalizeD2 = false;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 50;
analysisConfig.nMinNeurons = 10;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
analysisConfig.includeM2356 = false;
if ~isempty(brainArea) && strcmpi(brainArea, 'M2356')
  analysisConfig.includeM2356 = true;
end

%% Paths
paths = get_paths();
scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('criticality_d2_sliding_examples'));
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
addpath(fullfile(srcPath, 'sliding_window_prep', 'data_prep'));
addpath(fullfile(srcPath, 'sliding_window_prep', 'utils'));
addpath(fullfile(srcPath, 'criticality'));

fprintf('\n=== Criticality d2 sliding-window examples ===\n');
fprintf('Window: %.1f s, step: %.2f s (overlapping)\n', slidingWindowSize, stepSize);
fprintf('Brain area: %s\n', brainArea);
fprintf('useLog10D2: %d\n', useLog10D2);

validate_example_sessions(exampleSessions);

%% Run analysis for each example session
numExamples = numel(exampleSessions);
exampleResults = repmat(struct(), numExamples, 1);

for e = 1:numExamples
  ex = exampleSessions(e);
  fprintf('\n--- Example %d/%d [%s]: %s ---\n', e, numExamples, ex.sessionType, ex.sessionName);

  opts = build_example_load_opts(ex.sessionType, collectStart, collectEnd);
  subjectNameForLoad = ex.subjectName;
  loadArgs = build_session_load_args(ex.sessionType, ex.sessionName, opts, subjectNameForLoad);
  dataStruct = load_sliding_window_data(ex.sessionType, dataSource, loadArgs{:});

  [dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea);
  if ~areaOk
    warning('criticality_d2_sliding_examples:MissingArea', ...
      'Brain area "%s" not available for %s; skipping.', brainArea, ex.sessionName);
    continue;
  end

  results = criticality_ar_analysis(dataStruct, analysisConfig);
  if ~isempty(brainArea)
    results = filter_ar_results_to_brain_area(results, brainArea);
  end

  trace = extract_d2_time_trace(results, useLog10D2, analysisConfig.normalizeD2);
  if isempty(trace.timeMin) || isempty(trace.d2)
    warning('criticality_d2_sliding_examples:NoD2', ...
      'No d2 trace for %s (%s).', ex.sessionName, brainArea);
    continue;
  end

  exampleResults(e).example = ex;
  exampleResults(e).results = results;
  exampleResults(e).trace = trace;
  fprintf('  %d windows, %.1f min span, mean d2 = %.4f\n', ...
    numel(trace.d2), max(trace.timeMin) - min(trace.timeMin), mean(trace.d2, 'omitnan'));
end

%% Plot all traces on one axis
fig = plot_d2_sliding_examples(exampleResults, brainArea, slidingWindowSize, stepSize, useLog10D2);

if saveFigure
  saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  plotPath = fullfile(saveDir, sprintf('criticality_d2_sliding_examples_%s_win%.0fs_step%.1fs.png', ...
    brainArea, slidingWindowSize, stepSize));
  exportgraphics(fig, plotPath, 'Resolution', 300);
  fprintf('\nSaved figure: %s\n', plotPath);
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

function opts = build_example_load_opts(sessionType, collectStart, collectEnd)
% BUILD_EXAMPLE_LOAD_OPTS - Loader opts per session type (matches run_criticality_ar.m)

opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.minFiringRate = 0.05;
opts.maxFiringRate = 150;

if strcmpi(sessionType, 'reach') || strcmpi(sessionType, 'hong')
  opts.collectEnd = [];
else
  opts.collectEnd = collectEnd;
end
end

function [dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea)
% APPLY_BRAIN_AREA_SELECTION - Restrict analysis to one brain area

areaOk = true;
if isempty(brainArea)
  return;
end
if strcmpi(brainArea, 'M2356')
  areaOk = any(strcmp(dataStruct.areas, 'M23')) && any(strcmp(dataStruct.areas, 'M56'));
  return;
end
areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  areaOk = false;
  return;
end
dataStruct.areasToTest = areaIdx;
end

function results = filter_ar_results_to_brain_area(results, brainArea)
% FILTER_AR_RESULTS_TO_BRAIN_AREA - Keep one area in AR results struct

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaNames = normalize_results_area_list(results.areas);
areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end
cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'mrBrPermuted', ...
  'd2PermutedMean', 'd2PermutedSEM', 'popActivityWindows', 'popActivityFull'};
results.areas = {areaNames{areaIdx}};
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

function areaNames = normalize_results_area_list(areasField)
% NORMALIZE_RESULTS_AREA_LIST - Flatten results.areas to a cell of name strings

areaNames = {};
if isempty(areasField)
  return;
end
if ischar(areasField)
  areaNames = {areasField};
  return;
end
if isstring(areasField)
  areaNames = cellstr(areasField(:));
  return;
end
if ~iscell(areasField)
  areaNames = {char(areasField)};
  return;
end
rawAreas = areasField(:);
for k = 1:numel(rawAreas)
  entry = rawAreas{k};
  if iscell(entry)
    for j = 1:numel(entry)
      if ischar(entry{j}) || isstring(entry{j})
        areaNames{end+1} = char(string(entry{j})); %#ok<AGROW>
      end
    end
  elseif ischar(entry) || isstring(entry)
    areaNames{end+1} = char(string(entry)); %#ok<AGROW>
  end
end
end

function trace = extract_d2_time_trace(results, useLog10D2, normalizeD2)
% EXTRACT_D2_TIME_TRACE - Window center times (minutes) and d2 for plotting
%
% Variables:
%   results      - Output from criticality_ar_analysis (single area)
%   useLog10D2   - Apply log10 transform for display
%   normalizeD2  - Use d2Normalized instead of raw d2 when true
%
% Returns:
%   trace - Struct with .timeMin (minutes) and .d2 column vectors

trace = struct('timeMin', [], 'd2', []);
if ~isfield(results, 'areas') || isempty(results.areas)
  return;
end
areaIdx = 1;
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
  return;
end
if areaIdx > numel(results.startS) || isempty(results.startS{areaIdx})
  return;
end

d2Vec = results.d2{areaIdx}(:);
if normalizeD2 && isfield(results, 'd2Normalized') ...
    && ~isempty(results.d2Normalized{areaIdx})
  d2Vec = results.d2Normalized{areaIdx}(:);
end
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end

timeSec = results.startS{areaIdx}(:);
validMask = isfinite(timeSec) & isfinite(d2Vec);
trace.timeMin = timeSec(validMask) / 60;
trace.d2 = d2Vec(validMask);
end

function fig = plot_d2_sliding_examples(exampleResults, brainArea, slidingWindowSize, stepSize, useLog10D2)
% PLOT_D2_SLIDING_EXAMPLES - Overlay d2 traces from example sessions on one axis

conditionColors = [ ...
  0.15, 0.45, 0.75; ...
  0.85, 0.35, 0.15; ...
  0.20, 0.65, 0.35];

fig = figure('Color', 'w', 'Position', [100 100 1100 420], ...
  'Name', 'd2 sliding-window examples');
ax = axes(fig);
hold(ax, 'on');

plottedCount = 0;
for e = 1:numel(exampleResults)
  if ~isfield(exampleResults(e), 'trace') || isempty(exampleResults(e).trace.timeMin)
    continue;
  end
  trace = exampleResults(e).trace;
  ex = exampleResults(e).example;
  lineColor = get_example_condition_color(ex.sessionType, conditionColors);
  if isfield(ex, 'displayLabel') && ~isempty(ex.displayLabel)
    legendLabel = ex.displayLabel;
  else
    legendLabel = ex.sessionType;
  end
  plot(ax, trace.timeMin, trace.d2, '-', 'Color', lineColor, ...
    'LineWidth', 2, 'DisplayName', legendLabel);
  plottedCount = plottedCount + 1;
end

if plottedCount == 0
  error('No d2 traces available to plot.');
end

grid(ax, 'on');
xlabel(ax, 'Time (min)');
if useLog10D2
  ylabel(ax, 'log_{10}(d2)');
else
  ylabel(ax, 'd2');
end
title(ax, sprintf('%s d2 sliding-window examples (%.0f s window, %.1f s step)', ...
  brainArea, slidingWindowSize, stepSize), 'Interpreter', 'none');
legend(ax, 'Location', 'best');
hold(ax, 'off');
end

function lineColor = get_example_condition_color(sessionType, conditionColors)
% GET_EXAMPLE_CONDITION_COLOR - Stable color per session type

switch lower(sessionType)
  case 'spontaneous'
    lineColor = conditionColors(1, :);
  case 'interval'
    lineColor = conditionColors(2, :);
  case 'reach'
    lineColor = conditionColors(3, :);
  otherwise
    lineColor = conditionColors(1, :);
end
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive values

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end
