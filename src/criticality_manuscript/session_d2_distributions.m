%%
% Session d2 Distributions (Manuscript)
%
% For one session, runs the same AR/d2 pipeline as criticality_ar_across_tasks.m
% (non-overlapping windows) and plots overlapping probability densities of
% window-wise d2 for real vs shuffled data.
%
% Variables (configure in this section):
%   sessionType      - 'spontaneous', 'interval', 'reach', 'schall'
%   sessionName      - Session identifier
%   subjectName      - Required for spontaneous/interval; '' for reach
%   dataSource       - 'spikes' or 'lfp'
%   collectStart     - Window start (seconds from session onset)
%   collectEnd       - Window end (seconds)
%   d2Window         - Non-overlapping window length (seconds)
%   brainArea        - Area to analyze (e.g. 'M56'); '' uses all valid areas
%   useLog10D2       - If true, plot log10(d2) and log10(shuffled d2)
%   useSubsampling   - If true, d2 per window = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling (run_criticality_ar.m)
%   plotD2PopActivity - If true, scatter d2 vs mean pop activity (+ shuffled)
%   behaviorLabelSets - Cell of structs (.name, .numeratorIDs, .denominatorIDs)
%                       for d2 vs behavior proportion figure (spontaneous)
%   saveFigure       - Export PNG/EPS to dropPath/criticality_manuscript
%
% Goal:
%   Visualize real d2 vs shuffled d2 distributions for one session across
%   windows, where shuffled values are the mean across permutations per window.

%% Configuration
% sessionType = 'interval';
% subjectName = 'ey9166';
% sessionName = 'ey9166_2026_04_03';
% dataSource = 'spikes';

collectStart = 0;
collectEnd = 45 * 60;

d2Window = 30;  % seconds; non-overlapping windows

brainArea = 'M56';
useLog10D2 = true;
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 20;
minNeuronsMultiple = 1.25;
plotD2PopActivity = true;
saveFigure = false;

% Behavior label sets for d2 correlation figure (proportion per window)
% denominatorIDs empty -> fraction of all behavior frames in the window
behaviorLabelSets = {
  struct('name', 'Loco', 'numeratorIDs', [1 2 13:15], 'denominatorIDs', [])
  struct('name', 'Groom', 'numeratorIDs', 6:10, 'denominatorIDs', [])
  };

opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 150;

analysisConfig = struct();
analysisConfig.slidingWindowSize = d2Window;
analysisConfig.stepSize = d2Window;
analysisConfig.binSize = 0.025;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = true;
analysisConfig.nShuffles = 50;
analysisConfig.normalizeD2 = true;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = 25;
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
  scriptDir = fileparts(which('session_d2_distributions'));
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

fprintf('\n=== Session d2 Distributions ===\n');
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('d2 windows: %.1f s; binSize: %.3f s; nShuffles: %d\n', ...
  d2Window, analysisConfig.binSize, analysisConfig.nShuffles);
fprintf('useLog10D2: %d\n', useLog10D2);
if useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
    nSubsamples, nNeuronsSubsample, minNeuronsMultiple);
else
  fprintf('Subsampling: off\n');
end

%% Load session and run d2 analysis
subjectNameForLoad = subjectName;
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

[dataStruct, areaOk] = apply_brain_area_selection(dataStruct, brainArea);
if ~areaOk
  error('Brain area "%s" not available in this session.', brainArea);
end

results = criticality_ar_analysis(dataStruct, analysisConfig);

if ~isempty(brainArea)
  results = filter_ar_results_to_brain_area(results, brainArea);
  if isempty(results.areas)
    error('No d2 results for brain area "%s".', brainArea);
  end
end

print_session_d2_summary(results, useLog10D2);

%% Build distributions and plot
plotData = build_d2_distribution_data(results, useLog10D2);
if isempty(plotData.areas)
  error(['No valid d2 distribution data found. Check d2 values and shuffled ' ...
    'permutation outputs for this session.']);
end

fig = plot_d2_distributions(plotData, sessionType, sessionName, d2Window, collectStart, collectEnd, useLog10D2);

if saveFigure
  saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  areaTag = format_areas_label(plotData.areas);
  plotBase = sprintf('session_d2_distributions_%s_%s_win%.0fs_%.0f-%.0fs', ...
    sessionName, areaTag, d2Window, collectStart, collectEnd);
  if useLog10D2
    plotBase = [plotBase, '_log10'];
  end
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
  exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
  fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
end

%% d2 vs mean population activity (real and shuffled mean per window)
if plotD2PopActivity
  figPop = plot_d2_vs_popactivity(results, useLog10D2, d2Window);
  print_d2_popactivity_correlations(results, useLog10D2);
  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    areaTag = format_areas_label(plotData.areas);
    plotBase = sprintf('session_d2_vs_popactivity_%s_%s_win%.0fs_%.0f-%.0fs', ...
      sessionName, areaTag, d2Window, collectStart, collectEnd);
    if useLog10D2
      plotBase = [plotBase, '_log10'];
    end
    exportgraphics(figPop, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(figPop, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
  end
end

%% d2 vs behavior label-set proportions (one scatter per set)
if ~isempty(behaviorLabelSets)
  if ~isfield(dataStruct, 'bhvID') || isempty(dataStruct.bhvID)
    warning(['behaviorLabelSets configured but bhvID is unavailable for this session; ', ...
      'skipping behavior scatter figure.']);
  else
    figBhv = plot_d2_vs_behavior_sets(results, dataStruct, behaviorLabelSets, ...
      d2Window, useLog10D2);
    print_d2_behavior_correlations(results, dataStruct, behaviorLabelSets, d2Window, useLog10D2);
    if saveFigure
      saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
      if ~exist(saveDir, 'dir')
        mkdir(saveDir);
      end
      areaTag = format_areas_label(plotData.areas);
      plotBase = sprintf('session_d2_vs_behavior_%s_%s_win%.0fs_%.0f-%.0fs', ...
        sessionName, areaTag, d2Window, collectStart, collectEnd);
      if useLog10D2
        plotBase = [plotBase, '_log10'];
      end
      exportgraphics(figBhv, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
      exportgraphics(figBhv, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
      fprintf('Saved figure: %s\n', fullfile(saveDir, plotBase));
    end
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

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
fprintf('  Restricting analysis to area: %s\n', brainArea);
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

function print_session_d2_summary(results, useLog10D2)
% PRINT_SESSION_D2_SUMMARY - Window counts and mean d2 per area

fprintf('\n=== Session d2 summary ===\n');
for a = 1:numel(results.areas)
  if a > numel(results.d2) || isempty(results.d2{a})
    fprintf('  %s: no d2 data\n', results.areas{a});
    continue;
  end

  d2Vec = results.d2{a}(:);
  if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
  end
  d2Vec = d2Vec(isfinite(d2Vec));

  nPermRows = 0;
  if isfield(results, 'd2Permuted') && a <= numel(results.d2Permuted) && ~isempty(results.d2Permuted{a})
    nPermRows = size(results.d2Permuted{a}, 1);
  end

  if isempty(d2Vec)
    fprintf('  %s: no finite d2 values (perm rows: %d)\n', results.areas{a}, nPermRows);
  else
    fprintf('  %s: %d finite d2 windows, mean = %.4f (perm rows: %d)\n', ...
      results.areas{a}, numel(d2Vec), mean(d2Vec), nPermRows);
  end
end
end

function plotData = build_d2_distribution_data(results, useLog10D2)
% BUILD_D2_DISTRIBUTION_DATA - Collect real d2 and per-window shuffled means
%
% Variables:
%   results    - Output from criticality_ar_analysis
%   useLog10D2 - If true, transform values with log10_safe_numeric
%
% Goal:
%   Build per-area vectors for overlapping histogram/PDF plots:
%   - realD2 values across windows
%   - shuffledMeanD2 values where each element is mean across permutations for one window

plotData = struct();
plotData.areas = {};
plotData.realD2 = {};
plotData.shuffledMeanD2 = {};

for a = 1:numel(results.areas)
  if a > numel(results.d2) || isempty(results.d2{a})
    continue;
  end

  d2Vec = results.d2{a}(:);
  if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
  end
  d2Vec = d2Vec(isfinite(d2Vec));
  if isempty(d2Vec)
    continue;
  end

  shuffledVec = get_per_window_shuffle_mean_d2(results, a, useLog10D2);
  shuffledVec = shuffledVec(isfinite(shuffledVec));

  plotData.areas{end+1} = results.areas{a}; %#ok<AGROW>
  plotData.realD2{end+1} = d2Vec; %#ok<AGROW>
  plotData.shuffledMeanD2{end+1} = shuffledVec; %#ok<AGROW>
end
end

function fig = plot_d2_distributions(plotData, sessionType, sessionName, d2Window, collectStart, collectEnd, useLog10D2)
% PLOT_D2_DISTRIBUTIONS - Overlapping PDFs of real d2 and shuffled mean d2
%
% Variables:
%   plotData - Struct from build_d2_distribution_data
%
% Goal:
%   Plot one tile per area, with shared x-limits and identical bin edges.

numAreas = numel(plotData.areas);
allVals = [];
for a = 1:numAreas
  allVals = [allVals; plotData.realD2{a}(:)]; %#ok<AGROW>
  allVals = [allVals; plotData.shuffledMeanD2{a}(:)]; %#ok<AGROW>
end
allVals = allVals(isfinite(allVals));
if isempty(allVals)
  error('No finite d2 values available for plotting.');
end

[binEdges, xMin, xMax] = build_shared_bin_edges(allVals, 28);

fig = figure('Color', 'w', 'Position', [120 120 900 260 * numAreas], ...
  'Name', 'd2 distributions');
tileLayout = tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
  ax = nexttile(tileLayout);
  hold(ax, 'on');

  realVals = plotData.realD2{a};
  shuffledVals = plotData.shuffledMeanD2{a};

  histogram(ax, realVals, binEdges, 'Normalization', 'pdf', ...
    'FaceColor', [0.15 0.45 0.75], 'FaceAlpha', 0.45, 'EdgeColor', 'none', ...
    'DisplayName', 'Data');

  if ~isempty(shuffledVals)
    histogram(ax, shuffledVals, binEdges, 'Normalization', 'pdf', ...
      'FaceColor', [0.55 0.55 0.55], 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
      'DisplayName', 'Shuffled mean (per window)');
  end

  xline(ax, 0, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1, 'HandleVisibility', 'off');
  xlim(ax, [xMin, xMax]);
  grid(ax, 'on');
  if useLog10D2
    xlabel(ax, 'log_{10}(d2)');
  else
    xlabel(ax, 'd2');
  end
  ylabel(ax, 'Probability density');
  title(ax, plotData.areas{a}, 'Interpreter', 'none');
  legend(ax, 'Location', 'northeast');
  hold(ax, 'off');
end

if useLog10D2
  xLabelTitle = 'log_{10}(d2)';
else
  xLabelTitle = 'd2';
end
sgtitle(tileLayout, sprintf( ...
  'Distribution of %s | real vs shuffled mean per window | %s | %.0fs windows%s [%.0f-%.0f s]', ...
  xLabelTitle, sessionType, d2Window, make_title_suffix(sessionName), collectStart, collectEnd), ...
  'FontSize', 12, 'Interpreter', 'none');
end

function [binEdges, xMin, xMax] = build_shared_bin_edges(allVals, nBinsTarget)
% BUILD_SHARED_BIN_EDGES - Shared histogram bin edges and x-limits

xMin = min(allVals);
xMax = max(allVals);
xSpan = xMax - xMin;
if xSpan <= 0 || ~isfinite(xSpan)
  pad = max(0.5, abs(xMin) * 0.05 + eps);
  xMin = xMin - pad;
  xMax = xMax + pad;
else
  pad = 0.03 * xSpan;
  xMin = xMin - pad;
  xMax = xMax + pad;
end

nBins = max(8, round(nBinsTarget));
binEdges = linspace(xMin, xMax, nBins + 1);
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive values

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function suffixStr = make_title_suffix(sessionName)
% MAKE_TITLE_SUFFIX - Optional session-name suffix for figure titles

if isempty(sessionName)
  suffixStr = '';
else
  suffixStr = [' | ' sessionName];
end
end

function label = format_areas_label(areaNames)
% FORMAT_AREAS_LABEL - Underscore-safe tag for filenames/titles

if iscell(areaNames)
  areaNames = areaNames(:)';
  label = strjoin(areaNames, '_');
else
  label = char(areaNames);
end
label = matlab.lang.makeValidName(label);
end

function fig = plot_d2_vs_popactivity(results, useLog10D2, d2Window)
% PLOT_D2_VS_POPACTIVITY - Scatter d2 and shuffled mean d2 vs pop activity per window

if nargin < 3
  d2Window = results.params.slidingWindowSize;
end

if ~isfield(results, 'popActivityWindows')
  error('results.popActivityWindows not found.');
end

numAreas = numel(results.areas);
fig = figure('Color', 'w', 'Position', [140 140 420 * numAreas 420], ...
  'Name', 'd2 vs population activity');
tileLayout = tiledlayout(fig, 1, numAreas, 'TileSpacing', 'compact', 'Padding', 'compact');
d2YLabel = get_d2_axis_label(useLog10D2);

for a = 1:numAreas
  ax = nexttile(tileLayout);
  hold(ax, 'on');

  [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, a, useLog10D2);
  if ~any(validMask)
    title(ax, sprintf('%s (no data)', results.areas{a}), 'Interpreter', 'none');
    continue;
  end

  scatter_open_translucent(ax, popVec(validMask), d2Vec(validMask), 24, ...
    [0.15 0.45 0.75], 'Data');

  shufVec = get_shuffled_mean_d2_per_window(results, a, useLog10D2);
  if ~isempty(shufVec)
    shufVec = shufVec(1:numel(d2Vec));
    shufMask = validMask & isfinite(shufVec);
    if any(shufMask)
      scatter_open_translucent(ax, popVec(shufMask), shufVec(shufMask), 24, ...
        [0.55 0.55 0.55], 'Shuffled mean');
    end
  end

  rData = pearson_r(popVec(validMask), d2Vec(validMask));
  rShuf = nan;
  if ~isempty(shufVec)
    shufMask = validMask & isfinite(shufVec);
    if any(shufMask)
      rShuf = pearson_r(popVec(shufMask), shufVec(shufMask));
    end
  end

  xlabel(ax, 'Mean pop activity (spikes/bin)');
  ylabel(ax, d2YLabel);
  title(ax, sprintf('%s | r_{data}=%.3f, r_{shuf}=%.3f, n=%d', ...
    results.areas{a}, rData, rShuf, sum(validMask)), 'Interpreter', 'none');
  legend(ax, 'Location', 'best');
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayout, sprintf('d2 vs mean population activity per %.0fs window', d2Window), ...
  'FontSize', 12);
end

function print_d2_popactivity_correlations(results, useLog10D2)
% PRINT_D2_POPACTIVITY_CORRELATIONS - Command-window summary

fprintf('\n=== d2 vs mean pop activity correlations ===\n');
for a = 1:numel(results.areas)
  [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, a, useLog10D2);
  shufVec = get_shuffled_mean_d2_per_window(results, a, useLog10D2);
  if ~any(validMask)
    fprintf('  %s: no data\n', results.areas{a});
    continue;
  end
  rData = pearson_r(popVec(validMask), d2Vec(validMask));
  rShuf = nan;
  if ~isempty(shufVec)
    shufMask = validMask & isfinite(shufVec);
    if any(shufMask)
      rShuf = pearson_r(popVec(shufMask), shufVec(shufMask));
    end
  end
  fprintf('  %s: r(data)=%.3f, r(shuffled)=%.3f, n=%d\n', ...
    results.areas{a}, rData, rShuf, sum(validMask));
end
end

function print_d2_behavior_correlations(results, dataStruct, behaviorLabelSets, d2Window, useLog10D2)
% PRINT_D2_BEHAVIOR_CORRELATIONS - Command-window summary per label set

fprintf('\n=== d2 vs behavior proportion correlations ===\n');
for a = 1:numel(results.areas)
  if a > numel(results.startS) || isempty(results.startS{a})
    continue;
  end
  centerTimes = results.startS{a};
  bhvProportions = compute_behavior_proportions_by_set( ...
    dataStruct, centerTimes, d2Window, behaviorLabelSets);
  d2Vec = get_aligned_d2_vector(results, a, useLog10D2);
  nWindows = min(numel(d2Vec), numel(centerTimes));
  d2Vec = d2Vec(1:nWindows);
  for s = 1:numel(behaviorLabelSets)
    [d2Plot, propPlot, validMask] = align_window_vectors(d2Vec, bhvProportions{s}, nWindows);
    if ~any(validMask)
      fprintf('  %s | %s: no data\n', results.areas{a}, behaviorLabelSets{s}.name);
      continue;
    end
    rVal = pearson_r(propPlot(validMask), d2Plot(validMask));
    fprintf('  %s | %s: r=%.3f, n=%d\n', results.areas{a}, behaviorLabelSets{s}.name, ...
      rVal, sum(validMask));
  end
end
end

 function fig = plot_d2_vs_behavior_sets(results, dataStruct, behaviorLabelSets, d2Window, useLog10D2)
% PLOT_D2_VS_BEHAVIOR_SETS - One scatter per behavior label set (single row)

numSets = numel(behaviorLabelSets);
numAreas = numel(results.areas);
fig = figure('Color', 'w', 'Position', [120 120 380 * numSets 420], ...
  'Name', 'd2 vs behavior proportions');
tileLayout = tiledlayout(fig, numAreas, numSets, 'TileSpacing', 'compact', 'Padding', 'compact');
d2YLabel = get_d2_axis_label(useLog10D2);

for a = 1:numAreas
  if a > numel(results.startS) || isempty(results.startS{a})
    continue;
  end
  centerTimes = results.startS{a};
  d2Vec = get_aligned_d2_vector(results, a, useLog10D2);
  nWindows = min(numel(d2Vec), numel(centerTimes));
  if nWindows == 0
    warning('No windows for area %s; skipping behavior scatters.', results.areas{a});
    continue;
  end
  d2Vec = d2Vec(1:nWindows);
  centerTimes = centerTimes(1:nWindows);

  bhvProportions = compute_behavior_proportions_by_set( ...
    dataStruct, centerTimes, d2Window, behaviorLabelSets);

  for s = 1:numSets
    ax = nexttile(tileLayout);
    hold(ax, 'on');

    [d2Plot, propPlot, validMask] = align_window_vectors(d2Vec, bhvProportions{s}, nWindows);
    if ~any(validMask)
      title(ax, sprintf('%s — %s (no data)', results.areas{a}, behaviorLabelSets{s}.name), ...
        'Interpreter', 'none');
      continue;
    end

    scatter_open_translucent(ax, propPlot(validMask), d2Plot(validMask), 50, ...
      [0.15 0.45 0.75]);

    rVal = pearson_r(propPlot(validMask), d2Plot(validMask));
    xlabel(ax, 'Behavior proportion');
    if s == 1
      ylabel(ax, d2YLabel);
    end
    title(ax, sprintf('%s | r=%.3f, n=%d', behaviorLabelSets{s}.name, rVal, sum(validMask)), ...
      'Interpreter', 'none');
    grid(ax, 'on');
    xlim(ax, [0, 1]);
    hold(ax, 'off');
  end
end

if numAreas == 1
  rowTitle = results.areas{1};
else
  rowTitle = strjoin(results.areas, ', ');
end
sgtitle(tileLayout, sprintf('d2 vs behavior proportions | %s | %.0fs windows', ...
  rowTitle, d2Window), 'FontSize', 12, 'Interpreter', 'none');
end

function bhvProportions = compute_behavior_proportions_by_set(dataStruct, centerTimes, winSize, behaviorLabelSets)
% COMPUTE_BEHAVIOR_PROPORTIONS_BY_SET - Per-window proportion for each label set

numSets = numel(behaviorLabelSets);
numWindows = numel(centerTimes);
bhvProportions = cell(1, numSets);
for s = 1:numSets
  bhvProportions{s} = nan(numWindows, 1);
end

if ~isfield(dataStruct, 'bhvID') || isempty(dataStruct.bhvID) || ~isfield(dataStruct, 'fsBhv')
  return;
end

bhvID = dataStruct.bhvID(:);
bhvID(bhvID < 0) = nan;
bhvBinSize = 1 / dataStruct.fsBhv;

for w = 1:numWindows
  centerTime = centerTimes(w);
  winStartTime = centerTime - winSize / 2;
  winEndTime = centerTime + winSize / 2;
  bhvStartIdx = max(1, round(winStartTime / bhvBinSize) + 1);
  bhvEndIdx = min(numel(bhvID), round(winEndTime / bhvBinSize));
  if bhvStartIdx > bhvEndIdx
    continue;
  end

  windowBhvID = bhvID(bhvStartIdx:bhvEndIdx);
  windowBhvID = windowBhvID(isfinite(windowBhvID));
  if isempty(windowBhvID)
    continue;
  end

  for s = 1:numSets
    labelSet = behaviorLabelSets{s};
    numerIds = labelSet.numeratorIDs(:)';
    if isfield(labelSet, 'denominatorIDs') && ~isempty(labelSet.denominatorIDs)
      denomIds = labelSet.denominatorIDs(:)';
      denomMask = ismember(windowBhvID, denomIds);
      denominatorCount = sum(denomMask);
      if denominatorCount == 0
        continue;
      end
      numeratorCount = sum(ismember(windowBhvID(denomMask), numerIds));
      bhvProportions{s}(w) = numeratorCount / denominatorCount;
    else
      numeratorCount = sum(ismember(windowBhvID, numerIds));
      bhvProportions{s}(w) = numeratorCount / numel(windowBhvID);
    end
  end
end
end

function [vecA, vecB, validMask] = align_window_vectors(vecA, vecB, nWindows)
% ALIGN_WINDOW_VECTORS - Column vectors of equal length and element-wise finite mask
%
% Variables:
%   vecA, vecB  - Window-aligned values (any shape)
%   nWindows    - Optional cap on length (min with vector lengths)
%
% Goal:
%   Avoid row/column broadcast when building isfinite masks for scatter plots.

if nargin < 3 || isempty(nWindows)
  nWindows = min(numel(vecA), numel(vecB));
end
nWindows = min([nWindows, numel(vecA), numel(vecB)]);
vecA = vecA(1:nWindows);
vecB = vecB(1:nWindows);
vecA = vecA(:);
vecB = vecB(:);
validMask = isfinite(vecA) & isfinite(vecB);
end

function d2Vec = get_aligned_d2_vector(results, areaIdx, useLog10D2)
% GET_ALIGNED_D2_VECTOR - d2 per window for one area (optional log10)

d2Vec = [];
if areaIdx > numel(results.d2) || isempty(results.d2{areaIdx})
  return;
end
d2Vec = results.d2{areaIdx}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
end

function [d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, areaIdx, useLog10D2)
% GET_ALIGNED_D2_POPACTIVITY - Window-aligned d2 and pop activity vectors

d2Vec = [];
popVec = [];
validMask = false(0, 1);

d2Vec = get_aligned_d2_vector(results, areaIdx, useLog10D2);
if isempty(d2Vec)
  return;
end
if ~isfield(results, 'popActivityWindows') || areaIdx > numel(results.popActivityWindows) ...
    || isempty(results.popActivityWindows{areaIdx})
  return;
end

popVec = results.popActivityWindows{areaIdx}(:);
nWindows = min(numel(d2Vec), numel(popVec));
d2Vec = d2Vec(1:nWindows);
popVec = popVec(1:nWindows);
validMask = isfinite(d2Vec) & isfinite(popVec);
end

function scatter_open_translucent(ax, x, y, markerSize, faceColor, displayName)
% SCATTER_OPEN_TRANSLUCENT - Open circles with translucent fill

if nargin < 6
  displayName = '';
end

scatter(ax, x, y, markerSize, ...
  'Marker', 'o', ...
  'MarkerFaceColor', faceColor, ...
  'MarkerEdgeColor', faceColor, ...
  'MarkerFaceAlpha', 0.35, ...
  'LineWidth', 1, ...
  'DisplayName', displayName);
end

function shufVec = get_shuffled_mean_d2_per_window(results, areaIdx, useLog10D2)
% GET_SHUFFLED_MEAN_D2_PER_WINDOW - Mean shuffled d2 per window (subsampling-aware)

shufVec = get_per_window_shuffle_mean_d2(results, areaIdx, useLog10D2);
end

function rVal = pearson_r(x, y)
% PEARSON_R - Pearson correlation or NaN when undefined

rVal = nan;
if numel(x) < 2 || numel(y) < 2
  return;
end
cMat = corrcoef(x(:), y(:));
rVal = cMat(1, 2);
end

function yLabelText = get_d2_axis_label(useLog10D2)
% GET_D2_AXIS_LABEL - Y-axis label for d2 scatter plots

if useLog10D2
  yLabelText = 'log_{10}(d2)';
else
  yLabelText = 'd2';
end
end

function refAreaIdx = find_first_area_with_start_times(results)
% FIND_FIRST_AREA_WITH_START_TIMES - Index of first area with startS

refAreaIdx = find(~cellfun(@isempty, results.startS), 1);
if isempty(refAreaIdx)
  error('No window center times (startS) found in results.');
end
end
