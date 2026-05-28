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
saveFigure = false;

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

  shuffledVec = [];
  if isfield(results, 'd2Permuted') && a <= numel(results.d2Permuted) && ~isempty(results.d2Permuted{a})
    d2Perm = results.d2Permuted{a};
    if useLog10D2
      d2Perm = log10_safe_numeric(d2Perm);
    end
    if size(d2Perm, 2) > 1
      shuffledVec = nanmean(d2Perm, 2);
    else
      shuffledVec = d2Perm(:);
    end
    shuffledVec = shuffledVec(isfinite(shuffledVec));
  end

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
