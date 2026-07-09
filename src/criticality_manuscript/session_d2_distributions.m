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
%   brainArea              - Single or merged area (e.g. 'M56', 'M23M56'); '' uses all valid areas
%   brainAreaCombinations  - Merged areas: struct('name', 'M23M56', 'areas', {{'M23','M56'}})
%   useLog10D2       - If true, plot log10(d2) and log10(shuffled d2)
%   useSubsampling   - If true, d2 per window = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling (run_criticality_ar.m)
%   plotD2PopActivity - If true, scatter d2 vs mean pop activity (+ shuffled)
%   saveFigure       - Export PNG/EPS to dropPath/criticality_manuscript
%   plotConfig       - Axis fonts/line widths (see fill_manuscript_plot_config)
%   splitExcitatoryInhibitory - If true, run combined (E+I), excitatory, and inhibitory;
%                               also plots mean +/- SEM summary across windows;
%                               d2 vs pop activity on one figure (shared y-axis)
%   widthCutoff        - Peak-to-trough width threshold in ms (narrow <= cutoff = I)
%                        Waveforms: spontaneous/interval waveforms.mat; reach
%                        reach_task/data/WaveformDATA/*_Neural_WFs.mat
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
collectEnd = [];

d2Window = 30;  % seconds; non-overlapping windows

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
useLog10D2 = true;
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 32;
minNeuronsMultiple = 1.25;
plotD2PopActivity = true;
saveFigure = false;
plotConfig = fill_manuscript_plot_config();

splitExcitatoryInhibitory = true;
widthCutoff = 0.35;  % ms; peak-to-trough width (narrow <= cutoff = inhibitory)

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
analysisConfig.nMinNeurons = 20;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;

% Paths

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
if splitExcitatoryInhibitory
  fprintf('E/I split: on (widthCutoff = %.3f ms)\n', widthCutoff);
end

%% Load session and run d2 analysis
subjectNameForLoad = '';
if exist('subjectName', 'var') && ~isempty(subjectName)
  subjectNameForLoad = subjectName;
end
loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectNameForLoad);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

[dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, brainAreaCombinations, false);
if ~areaOk
  error('Brain area "%s" not available in this session.', brainArea);
end

if splitExcitatoryInhibitory
  eiCheck = check_session_ei_neuron_counts(dataStruct, paths, widthCutoff, brainArea, ...
    brainAreaCombinations, analysisConfig.nMinNeurons);
  if ~eiCheck.isOk
    return;
  end
end

cellTypesToRun = get_session_cell_types_to_run(splitExcitatoryInhibitory);
if splitExcitatoryInhibitory
  eiSummary = init_session_ei_summary({'d2'}, {get_d2_axis_label(useLog10D2)});
  eiPopActivityResults = cell(1, numel(cellTypesToRun));
end

for iCellRun = 1:numel(cellTypesToRun)
  cellType = cellTypesToRun{iCellRun};
  dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, widthCutoff, splitExcitatoryInhibitory);

  [dataStructRun, ~] = apply_manuscript_brain_area_selection(dataStructRun, brainArea, brainAreaCombinations);

  results = criticality_ar_analysis(dataStructRun, analysisConfig);

  if ~isempty(brainArea)
    results = filter_ar_results_to_brain_area(results, brainArea);
    if isempty(results.areas)
      error('No d2 results for brain area "%s" (%s).', brainArea, cell_type_label(cellType));
    end
  end

  print_session_d2_summary(results, useLog10D2);

  if splitExcitatoryInhibitory
    eiSummary = set_session_ei_summary_population(eiSummary, cellType, ...
      extract_d2_summary_metric_values(results, useLog10D2));
  end

  %% Build distributions and plot
  plotData = build_d2_distribution_data(results, useLog10D2);
  if isempty(plotData.areas)
    error(['No valid d2 distribution data found (%s). Check d2 values and shuffled ' ...
      'permutation outputs for this session.'], cell_type_label(cellType));
  end

  fig = plot_d2_distributions(plotData, sessionType, sessionName, d2Window, collectStart, collectEnd, useLog10D2, plotConfig);
  if splitExcitatoryInhibitory
    sgtitle(fig, sprintf('%s | %s | width cutoff %.3f ms', ...
      sessionName, cell_type_label(cellType), widthCutoff), 'Interpreter', 'none');
  end

  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    areaTag = format_areas_label(plotData.areas);
    plotBase = sprintf('session_d2_distributions_%s_%s_win%.0fs_%.0f-%.0fs%s', ...
      sessionName, areaTag, d2Window, collectStart, collectEnd, cell_type_file_tag(cellType));
    if useLog10D2
      plotBase = [plotBase, '_log10'];
    end
    exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved figure: %s\n', fullfile(saveDir, plotBase));
  end

  %% d2 vs mean population activity (real and shuffled mean per window)
  if plotD2PopActivity
    if splitExcitatoryInhibitory
      eiPopActivityResults{iCellRun} = struct('cellType', cellType, 'results', results);
      print_d2_popactivity_correlations(results, useLog10D2, cell_type_label(cellType));
    else
      figPop = plot_d2_vs_popactivity(results, useLog10D2, d2Window, plotConfig);
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
  end
end

if plotD2PopActivity && splitExcitatoryInhibitory
  figPopEi = plot_d2_vs_popactivity_ei_split(eiPopActivityResults, useLog10D2, d2Window, ...
    plotConfig, sessionName, widthCutoff);
  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    areaTag = format_areas_label(brainArea);
    if isempty(areaTag)
      areaTag = format_areas_label(eiPopActivityResults{1}.results.areas);
    end
    plotBase = sprintf('session_d2_vs_popactivity_%s_%s_win%.0fs_%.0f-%.0fs_ei_split', ...
      sessionName, areaTag, d2Window, collectStart, collectEnd);
    if useLog10D2
      plotBase = [plotBase, '_log10'];
    end
    exportgraphics(figPopEi, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(figPopEi, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved E/I pop-activity figure: %s\n', fullfile(saveDir, plotBase));
  end
end

if splitExcitatoryInhibitory
  areaTag = format_areas_label(brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  summaryTitle = sprintf('%s | %s | d2 mean +/- SEM across windows', sessionName, areaTag);
  figEiSummary = plot_session_ei_summary(eiSummary, summaryTitle, get_d2_axis_label(useLog10D2), [], [], plotConfig);
  if saveFigure
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    plotBase = sprintf('session_d2_ei_summary_%s_%s_win%.0fs_%.0f-%.0fs%s', ...
      sessionName, areaTag, d2Window, collectStart, collectEnd, session_ei_summary_file_tag());
    if useLog10D2
      plotBase = [plotBase, '_log10'];
    end
    exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
    exportgraphics(figEiSummary, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
    fprintf('\nSaved E/I summary figure: %s\n', fullfile(saveDir, plotBase));
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

function metricValues = extract_d2_summary_metric_values(results, useLog10D2)
% EXTRACT_D2_SUMMARY_METRIC_VALUES - Window-wise d2 values for E/I summary plot

metricValues = struct('d2', []);
if isempty(results.areas) || isempty(results.d2)
  return;
end

d2Vec = results.d2{1}(:);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end
metricValues.d2 = d2Vec(isfinite(d2Vec));
end

function yLabelText = get_d2_axis_label(useLog10D2)
if useLog10D2
  yLabelText = 'log_{10}(d2)';
else
  yLabelText = 'd2';
end
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

function fig = plot_d2_distributions(plotData, sessionType, sessionName, d2Window, collectStart, collectEnd, useLog10D2, plotConfig)
% PLOT_D2_DISTRIBUTIONS - Overlapping PDFs of real d2 and shuffled mean d2
%
% Variables:
%   plotData - Struct from build_d2_distribution_data
%   plotConfig - Manuscript axis/scatter styling
%
% Goal:
%   Plot one tile per area, with shared x-limits and identical bin edges.

if nargin < 8 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

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

[binEdges, xMin, xMax] = build_shared_histogram_bin_edges(allVals, 28);
if useLog10D2
  xLabelText = 'log_{10}(d2)';
  labelInterpreter = 'tex';
else
  xLabelText = 'd2';
  labelInterpreter = 'none';
end

fig = figure('Color', 'w', 'Position', [120 120 900 280 * numAreas], ...
  'Name', 'd2 distributions');
tileLayout = tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
  ax = nexttile(tileLayout);
  plot_real_shuffled_histogram_pdfs(ax, plotData.realD2{a}, plotData.shuffledMeanD2{a}, ...
    binEdges, xMin, xMax, plotConfig, useLog10D2);
  apply_manuscript_axes_style(ax, plotConfig, xLabelText, 'Probability density', ...
    plotData.areas{a}, labelInterpreter);
end

sgtitle(tileLayout, sprintf( ...
  'Distribution of %s | real vs shuffled mean per window | %s | %.0fs windows%s [%.0f-%.0f s]', ...
  xLabelText, sessionType, d2Window, make_title_suffix(sessionName), collectStart, collectEnd), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
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

function fig = plot_d2_vs_popactivity(results, useLog10D2, d2Window, plotConfig)
% PLOT_D2_VS_POPACTIVITY - Scatter d2 and shuffled mean d2 vs pop activity per window

if nargin < 3 || isempty(d2Window)
  d2Window = results.params.slidingWindowSize;
end
if nargin < 4 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

if ~isfield(results, 'popActivityWindows')
  error('results.popActivityWindows not found.');
end

numAreas = numel(results.areas);
fig = figure('Color', 'w', 'Position', [140 140 420 * numAreas 420], ...
  'Name', 'd2 vs population activity');
tileLayout = tiledlayout(fig, 1, numAreas, 'TileSpacing', 'compact', 'Padding', 'compact');
d2YLabel = get_d2_axis_label(useLog10D2);
if useLog10D2
  labelInterpreter = 'tex';
else
  labelInterpreter = 'none';
end

allYVals = [];
axesList = gobjects(numAreas, 1);
for a = 1:numAreas
  ax = nexttile(tileLayout);
  axesList(a) = ax;
  [yVals, ~, ~, ~] = plot_d2_popactivity_panel(ax, results, a, useLog10D2, plotConfig, ...
    results.areas{a}, d2YLabel, labelInterpreter, true);
  allYVals = [allYVals; yVals(:)]; %#ok<AGROW>
end
apply_shared_popactivity_ylim(axesList, allYVals);

sgtitle(tileLayout, sprintf('d2 vs mean population activity per %.0fs window', d2Window), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function fig = plot_d2_vs_popactivity_ei_split(eiResultsCell, useLog10D2, d2Window, ...
    plotConfig, sessionName, widthCutoff)
% PLOT_D2_VS_POPACTIVITY_EI_SPLIT - Combined, excitatory, and inhibitory on one figure
%
% Variables:
%   eiResultsCell - Cell of struct with .cellType and .results from each E/I run
%
% Goal:
%   One row per brain area, one column per population (combined, E, I) with shared
%   y-limits across all panels for direct comparison.

if nargin < 4 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if isempty(eiResultsCell)
  error('No E/I pop-activity results to plot.');
end

refResults = eiResultsCell{1}.results;
if ~isfield(refResults, 'popActivityWindows')
  error('results.popActivityWindows not found.');
end

numAreas = numel(refResults.areas);
numCols = numel(eiResultsCell);
fig = figure('Color', 'w', ...
  'Position', [120 120 380 * numCols max(360, 340 * numAreas)], ...
  'Name', 'd2 vs population activity (E/I split)');
tileLayout = tiledlayout(fig, numAreas, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
d2YLabel = get_d2_axis_label(useLog10D2);
if useLog10D2
  labelInterpreter = 'tex';
else
  labelInterpreter = 'none';
end

allYVals = [];
axesList = gobjects(numAreas, numCols);
for col = 1:numCols
  entry = eiResultsCell{col};
  results = entry.results;
  panelTitle = cell_type_label(entry.cellType);
  for a = 1:numAreas
    ax = nexttile(tileLayout);
    axesList(a, col) = ax;
    areaTitle = panelTitle;
    if numAreas > 1
      areaTitle = sprintf('%s | %s', panelTitle, results.areas{a});
    end
    showYLabel = (col == 1);
    [yVals, ~, ~, ~] = plot_d2_popactivity_panel(ax, results, a, useLog10D2, plotConfig, ...
      areaTitle, d2YLabel, labelInterpreter, showYLabel);
    allYVals = [allYVals; yVals(:)]; %#ok<AGROW>
  end
end
apply_shared_popactivity_ylim(axesList(:), allYVals);

sgtitle(tileLayout, sprintf('%s | d2 vs mean population activity | %.0fs windows | width cutoff %.3f ms', ...
  sessionName, d2Window, widthCutoff), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function [yVals, rData, rShuf, nValid] = plot_d2_popactivity_panel(ax, results, areaIdx, ...
    useLog10D2, plotConfig, panelTitle, d2YLabel, labelInterpreter, showYLabel)
% PLOT_D2_POPACTIVITY_PANEL - One scatter panel of d2 vs mean pop activity

if nargin < 10 || isempty(showYLabel)
  showYLabel = true;
end
if nargin < 9 || isempty(labelInterpreter)
  labelInterpreter = 'none';
end
if nargin < 8 || isempty(d2YLabel)
  d2YLabel = get_d2_axis_label(useLog10D2);
end

plotColors = manuscript_plot_colors();
hold(ax, 'on');

[d2Vec, popVec, validMask] = get_aligned_d2_popactivity(results, areaIdx, useLog10D2);
yVals = collect_d2_popactivity_y_values(results, areaIdx, useLog10D2);
rData = nan;
rShuf = nan;
nValid = 0;

if ~any(validMask)
  yLabelText = d2YLabel;
  if ~showYLabel
    yLabelText = '';
  end
  apply_manuscript_axes_style(ax, plotConfig, 'Mean pop activity (spikes/bin)', yLabelText, ...
    sprintf('%s (no data)', panelTitle), labelInterpreter);
  hold(ax, 'off');
  return;
end

scatter_manuscript_open(ax, popVec(validMask), d2Vec(validMask), plotConfig, ...
  plotColors.data, 'Data');
add_manuscript_scatter_trendline(ax, popVec(validMask), d2Vec(validMask), plotConfig);

shufVec = get_shuffled_mean_d2_per_window(results, areaIdx, useLog10D2);
if ~isempty(shufVec)
  shufVec = shufVec(1:numel(d2Vec));
  shufMask = validMask & isfinite(shufVec);
  if any(shufMask)
      scatter_manuscript_open(ax, popVec(shufMask), shufVec(shufMask), plotConfig, ...
        plotColors.shuffled, 'Shuffled mean');
  end
end

rData = pearson_r(popVec(validMask), d2Vec(validMask));
if ~isempty(shufVec)
  shufMask = validMask & isfinite(shufVec);
  if any(shufMask)
    rShuf = pearson_r(popVec(shufMask), shufVec(shufMask));
  end
end
nValid = sum(validMask);

yLabelText = d2YLabel;
if ~showYLabel
  yLabelText = '';
end
apply_manuscript_axes_style(ax, plotConfig, 'Mean pop activity (spikes/bin)', yLabelText, ...
  sprintf('%s | r_{data}=%.3f, r_{shuf}=%.3f, n=%d', panelTitle, rData, rShuf, nValid), ...
  labelInterpreter);
legend(ax, 'Location', 'best', 'FontSize', plotConfig.legendFontSize);
grid(ax, 'on');
hold(ax, 'off');
end

function yVals = collect_d2_popactivity_y_values(results, areaIdx, useLog10D2)
% COLLECT_D2_POPACTIVITY_Y_VALUES - Finite d2 y-values (data + shuffled) for y-limits

[d2Vec, ~, validMask] = get_aligned_d2_popactivity(results, areaIdx, useLog10D2);
yVals = d2Vec(validMask);
shufVec = get_shuffled_mean_d2_per_window(results, areaIdx, useLog10D2);
if ~isempty(shufVec) && ~isempty(d2Vec)
  shufVec = shufVec(1:numel(d2Vec));
  shufMask = validMask & isfinite(shufVec);
  yVals = [yVals(:); shufVec(shufMask)]; %#ok<AGROW>
end
yVals = yVals(isfinite(yVals));
end

function apply_shared_popactivity_ylim(axesList, allYVals)
% APPLY_SHARED_POPACTIVITY_YLIM - Match y-limits across pop-activity scatter panels

axesList = axesList(isgraphics(axesList));
if isempty(axesList)
  return;
end
allYVals = allYVals(isfinite(allYVals));
if isempty(allYVals)
  return;
end

yMin = min(allYVals);
yMax = max(allYVals);
ySpan = yMax - yMin;
if ySpan <= 0 || ~isfinite(ySpan)
  pad = max(0.1, abs(yMin) * 0.05 + eps);
else
  pad = 0.05 * ySpan;
end
sharedYLim = [yMin - pad, yMax + pad];
for iAx = 1:numel(axesList)
  ylim(axesList(iAx), sharedYLim);
end
end

function print_d2_popactivity_correlations(results, useLog10D2, populationLabel)
% PRINT_D2_POPACTIVITY_CORRELATIONS - Command-window summary

if nargin < 3
  populationLabel = '';
end
if isempty(populationLabel)
  fprintf('\n=== d2 vs mean pop activity correlations ===\n');
else
  fprintf('\n=== d2 vs mean pop activity correlations (%s) ===\n', populationLabel);
end
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

function refAreaIdx = find_first_area_with_start_times(results)
% FIND_FIRST_AREA_WITH_START_TIMES - Index of first area with startS

refAreaIdx = find(~cellfun(@isempty, results.startS), 1);
if isempty(refAreaIdx)
  error('No window center times (startS) found in results.');
end
end
