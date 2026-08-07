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
%   nPermutations    - Number of circular permutations per window for shuffled d2
%   plotD2PopActivity - If true, scatter d2 vs mean pop activity (+ shuffled)
%   plotD2Timeline   - If true, plot mean pop per d2 window, d2, and ethogram vs time
%   useRelativeTime  - If true, timeline x-axis is relative to collectStart (default false)
%   binSize          - Spike bin width (s) for d2 analysis (and window popActivity)
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
% collectStart = 1.2*10^4;
collectEnd = 45 * 60;
collectEnd = 1.2*10^4;
collectEnd = [];

d2Window = 5*60;  % seconds; non-overlapping windows

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
useLog10D2 = true;
useSubsampling = false;
nSubsamples = 20;
nNeuronsSubsample = 40;
minNeuronsMultiple = 1.1;
nPermutations = 5;  % circular shuffles per window for shuffled d2 distribution
plotD2PopActivity = true;
plotD2Timeline = true;  % mean pop per d2 window | d2 vs time | ethogram
useRelativeTime = false;  % false: absolute session time (default); true: t=0 at collectStart
binSize = 0.04;  % s; spike binning for d2 (and window mean popActivity)
saveFigure = false;
plotConfig = fill_manuscript_plot_config();

splitExcitatoryInhibitory = false;
widthCutoff = 0.35;  % ms; peak-to-trough width (narrow <= cutoff = inhibitory)

opts = neuro_behavior_options();
opts.firingRateCheckTime = 5 * 60;
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 200;

analysisConfig = struct();
analysisConfig.slidingWindowSize = d2Window;
analysisConfig.stepSize = d2Window;
analysisConfig.binSize = binSize;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = nPermutations > 0;
analysisConfig.nShuffles = nPermutations;
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
fprintf('d2 windows: %.1f s; binSize: %.3f s; nPermutations: %d\n', ...
  d2Window, binSize, nPermutations);
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

% Load session and run d2 analysis
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

  % Build distributions and plot
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

  % d2 vs mean population activity (real and shuffled mean per window)
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

  % popActivity | d2 over time | ethogram (time-aligned)
  if plotD2Timeline
    figTime = plot_d2_pop_ethogram_timeline(dataStruct, results, ...
      collectStart, collectEnd, d2Window, binSize, useLog10D2, plotConfig, ...
      sessionName, cellType, useRelativeTime);
    if ~isempty(figTime) && saveFigure
      saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
      if ~exist(saveDir, 'dir')
        mkdir(saveDir);
      end
      areaTag = format_areas_label(plotData.areas);
      if isempty(collectEnd)
        collectTag = sprintf('%.0f-full', collectStart);
      else
        collectTag = sprintf('%.0f-%.0f', collectStart, collectEnd);
      end
      plotBase = sprintf('session_d2_timeline_%s_%s_win%.0fs_%ss%s', ...
        sessionName, areaTag, d2Window, collectTag, cell_type_file_tag(cellType));
      if useLog10D2
        plotBase = [plotBase, '_log10'];
      end
      exportgraphics(figTime, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
      exportgraphics(figTime, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
      fprintf('Saved timeline figure: %s\n', fullfile(saveDir, plotBase));
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

function fig = plot_d2_pop_ethogram_timeline(dataStructBhv, results, ...
    collectStart, collectEnd, d2Window, binSize, useLog10D2, plotConfig, ...
    sessionName, cellType, useRelativeTime)
% PLOT_D2_POP_ETHOGRAM_TIMELINE - Stacked mean-pop | d2 | ethogram vs time
%
% Variables:
%   dataStructBhv - Session used for bhvID / fsBhv and duration
%   results       - criticality_ar_analysis output (d2, startS, popActivityWindows)
%   binSize       - Spike bin width (s) used in d2 analysis (title only)
%   useRelativeTime - If true, shift x-axis so t=0 at collectStart (default false)
%
% Layout (per brain area column):
%   Top:    mean popActivity per d2 window (results.popActivityWindows)
%   Middle: window-wise d2 (and shuffled mean when present)
%   Bottom: behavior ethogram
%
% Timebase: results.startS and bhvID use absolute session time by default.

if nargin < 8 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 9 || isempty(sessionName)
  sessionName = '';
end
if nargin < 10
  cellType = '';
end
if nargin < 11 || isempty(useRelativeTime)
  useRelativeTime = false;
end

dataPrepPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data_prep');
if exist(dataPrepPath, 'dir')
  addpath(dataPrepPath);
end

fig = [];
numAreas = numel(results.areas);
if numAreas < 1
  warning('session_d2_distributions:NoTimelineAreas', 'No areas for timeline plot.');
  return;
end

bhvRec = session_d2_behavior_record(dataStructBhv);
tMaxAbs = session_d2_resolve_timeline_tmax([], results, collectStart, collectEnd, d2Window, ...
  dataStructBhv);
tMinAbs = collectStart;
if isempty(tMinAbs) || ~isfinite(tMinAbs)
  tMinAbs = session_time_origin(dataStructBhv);
end
if useRelativeTime
  tMin = 0;
  tMax = tMaxAbs - tMinAbs;
  timeShift = tMinAbs;
  xLabelText = 'Time from collectStart (s)';
else
  tMin = tMinAbs;
  tMax = tMaxAbs;
  timeShift = 0;
  xLabelText = 'Time (s)';
end

plotColors = manuscript_plot_colors();
d2YLabel = get_d2_axis_label(useLog10D2);
fig = figure('Color', 'w', 'Name', sprintf('d2 timeline — %s', sessionName), ...
  'Position', [100 80 max(720, 420 * numAreas) 780]);

axesToLink = gobjects(0);
for a = 1:numAreas
  areaName = results.areas{a};
  tWin = [];
  if isfield(results, 'startS') && a <= numel(results.startS) && ~isempty(results.startS{a})
    tWin = results.startS{a}(:) - timeShift;
  end

  axPop = subplot(3, numAreas, a, 'Parent', fig);
  hold(axPop, 'on');
  popVec = [];
  if isfield(results, 'popActivityWindows') && a <= numel(results.popActivityWindows) ...
      && ~isempty(results.popActivityWindows{a})
    popVec = results.popActivityWindows{a}(:);
  end
  if ~isempty(popVec) && ~isempty(tWin)
    nPlot = min(numel(popVec), numel(tWin));
    plot(axPop, tWin(1:nPlot), popVec(1:nPlot), '-o', ...
      'Color', [0.15 0.15 0.15], 'MarkerFaceColor', [0.15 0.15 0.15], ...
      'MarkerSize', 5, 'LineWidth', plotConfig.axesLineWidth);
  else
    text(axPop, mean([tMin tMax]), 0.5, 'no window popActivity', ...
      'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
  end
  xlim(axPop, [tMin, tMax]);
  ylabel(axPop, 'mean pop', 'FontSize', plotConfig.axisLabelFontSize);
  title(axPop, areaName, 'Interpreter', 'none', 'FontSize', plotConfig.titleFontSize);
  set(axPop, 'XTickLabel', [], 'Box', 'off', 'TickDir', 'out', ...
    'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
  hold(axPop, 'off');

  axD2 = subplot(3, numAreas, numAreas + a, 'Parent', fig);
  hold(axD2, 'on');
  d2Vec = get_aligned_d2_vector(results, a, useLog10D2);
  if ~isempty(d2Vec) && ~isempty(tWin)
    nPlot = min(numel(d2Vec), numel(tWin));
    tD2 = tWin(1:nPlot);
    d2Vec = d2Vec(1:nPlot);
    plot(axD2, tD2, d2Vec, '-o', 'Color', plotColors.data, ...
      'MarkerFaceColor', plotColors.data, 'MarkerSize', 5, ...
      'LineWidth', plotConfig.axesLineWidth, 'DisplayName', 'Data');
    shufVec = get_shuffled_mean_d2_per_window(results, a, useLog10D2);
    if ~isempty(shufVec)
      shufVec = shufVec(1:nPlot);
      shufMask = isfinite(shufVec) & isfinite(tD2);
      if any(shufMask)
        plot(axD2, tD2(shufMask), shufVec(shufMask), '-o', 'Color', plotColors.shuffled, ...
          'MarkerFaceColor', plotColors.shuffled, 'MarkerSize', 4, ...
          'LineWidth', max(0.8, plotConfig.axesLineWidth - 0.3), ...
          'DisplayName', 'Shuffled mean');
      end
    end
    legend(axD2, 'Location', 'best', 'FontSize', plotConfig.legendFontSize);
  else
    text(axD2, mean([tMin tMax]), 0.5, 'no d2 values', ...
      'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
  end
  xlim(axD2, [tMin, tMax]);
  ylabel(axD2, d2YLabel, 'FontSize', plotConfig.axisLabelFontSize, ...
    'Interpreter', ternary_tex_if_log10(useLog10D2));
  set(axD2, 'XTickLabel', [], 'Box', 'off', 'TickDir', 'out', ...
    'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
  hold(axD2, 'off');

  axEth = subplot(3, numAreas, 2 * numAreas + a, 'Parent', fig);
  session_d2_plot_behavior_ethogram(axEth, bhvRec, tMin, tMax);
  xlabel(axEth, xLabelText, 'FontSize', plotConfig.axisLabelFontSize);
  set(axEth, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);

  axesToLink = [axesToLink; axPop; axD2; axEth]; %#ok<AGROW>
end
linkaxes(axesToLink, 'x');

cellTag = '';
if ~isempty(cellType) && ~strcmpi(cellType, 'combined')
  cellTag = sprintf(' | %s', cell_type_label(cellType));
end
sgtitle(fig, sprintf('%s%s | mean pop / d2 (%.0fs windows, bin=%.0f ms) / ethogram', ...
  sessionName, cellTag, d2Window, binSize * 1000), ...
  'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
fprintf('Plotted d2 timeline (%d area(s), t=[%.1f, %.1f] s).\n', numAreas, tMin, tMax);
end

function interp = ternary_tex_if_log10(useLog10D2)
if useLog10D2
  interp = 'tex';
else
  interp = 'none';
end
end

function tMax = session_d2_resolve_timeline_tmax(popTime, results, collectStart, collectEnd, ...
    d2Window, dataStruct)
% SESSION_D2_RESOLVE_TIMELINE_TMAX - Right edge of shared time axis

tMax = nan;
if ~isempty(popTime)
  tMax = max(popTime);
end
if isfield(results, 'startS')
  for a = 1:numel(results.startS)
    if ~isempty(results.startS{a})
      tMax = max(tMax, max(results.startS{a}) + d2Window / 2);
    end
  end
end
if ~isempty(collectEnd) && isfinite(collectEnd)
  tMax = max(tMax, collectEnd);
else
  durationSec = session_d2_session_duration_sec(dataStruct, collectStart);
  if isfinite(durationSec)
    tMax = max(tMax, collectStart + durationSec);
  end
end
if ~isfinite(tMax) || tMax <= collectStart
  tMax = collectStart + 1;
end
end

function durationSec = session_d2_session_duration_sec(dataStruct, collectStart)
% SESSION_D2_SESSION_DURATION_SEC - Loaded collect window length (s)

durationSec = nan;
if nargin < 2 || isempty(collectStart)
  collectStart = 0;
end
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
    && ~isempty(dataStruct.spikeData.collectEnd)
  startVal = collectStart;
  if isfield(dataStruct.spikeData, 'collectStart') && ~isempty(dataStruct.spikeData.collectStart)
    startVal = dataStruct.spikeData.collectStart;
  end
  durationSec = dataStruct.spikeData.collectEnd - startVal;
  return;
end
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
    && ~isempty(dataStruct.opts.collectEnd)
  startVal = collectStart;
  if isfield(dataStruct.opts, 'collectStart') && ~isempty(dataStruct.opts.collectStart)
    startVal = dataStruct.opts.collectStart;
  end
  durationSec = dataStruct.opts.collectEnd - startVal;
  return;
end
if isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
  durationSec = max(dataStruct.spikeTimes) - collectStart;
end
end

function bhvRec = session_d2_behavior_record(dataStruct)
% SESSION_D2_BEHAVIOR_RECORD - bhvID + fsBhv for ethogram plotting

bhvRec = struct('bhvID', [], 'fsBhv', nan, 'bhvTimeOrigin', 0);
if isfield(dataStruct, 'bhvID') && ~isempty(dataStruct.bhvID)
  bhvRec.bhvID = dataStruct.bhvID(:);
end
if isfield(dataStruct, 'fsBhv') && ~isempty(dataStruct.fsBhv)
  bhvRec.fsBhv = dataStruct.fsBhv;
elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'fsBhv') ...
    && ~isempty(dataStruct.opts.fsBhv)
  bhvRec.fsBhv = dataStruct.opts.fsBhv;
end
bhvRec.bhvTimeOrigin = session_time_origin(dataStruct);
end

function session_d2_plot_behavior_ethogram(ax, bhvRec, tMin, tMax)
% SESSION_D2_PLOT_BEHAVIOR_ETHOGRAM - Colored behavior runs aligned to time
%
% Variables:
%   ax     - Target axes
%   bhvRec - Struct with .bhvID, .fsBhv, .bhvTimeOrigin
%   tMin, tMax - Shared x-limits (s). bhvID(1) maps to tMin (absolute collect
%                start by default, or 0 when plotting relative time).

hold(ax, 'on');
bhvID = bhvRec.bhvID;
fsBhv = bhvRec.fsBhv;
if isempty(bhvID) || ~(isfinite(fsBhv) && fsBhv > 0)
  text(ax, mean([tMin tMax]), 0.5, 'no behavior labels', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.5 0.5 0.5]);
  xlim(ax, [tMin, tMax]);
  ylim(ax, [0 1]);
  set(ax, 'YTick', [], 'Box', 'off');
  hold(ax, 'off');
  return;
end

bhvID = bhvID(:);
nFrame = numel(bhvID);
frameStarts = tMin + ((0:nFrame-1)' ) / fsBhv;
frameEnds = tMin + (1:nFrame)' / fsBhv;

uniqueCodes = unique(bhvID);
codeColorMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
for iCode = 1:numel(uniqueCodes)
  code = double(uniqueCodes(iCode));
  c = colors_for_behaviors(code);
  if size(c, 1) == 1 && size(c, 2) == 3
    codeColorMap(code) = c;
  else
    codeColorMap(code) = [0.7 0.7 0.7];
  end
end

runCode = bhvID(1);
runStart = frameStarts(1);
for i = 2:nFrame
  if bhvID(i) ~= runCode
    session_d2_fill_ethogram_run(ax, runStart, frameStarts(i), runCode, codeColorMap);
    runCode = bhvID(i);
    runStart = frameStarts(i);
  end
end
session_d2_fill_ethogram_run(ax, runStart, frameEnds(end), runCode, codeColorMap);

xlim(ax, [tMin, tMax]);
ylim(ax, [0 1]);
ylabel(ax, 'bhv', 'FontSize', 9);
set(ax, 'YTick', [], 'Box', 'off', 'TickDir', 'out');
hold(ax, 'off');
end

function session_d2_fill_ethogram_run(ax, tStart, tEnd, code, codeColorMap)
% SESSION_D2_FILL_ETHOGRAM_RUN - One colored rectangle for a behavior bout

if ~(isfinite(tStart) && isfinite(tEnd)) || tEnd <= tStart
  return;
end
code = double(code);
if isKey(codeColorMap, code)
  faceColor = codeColorMap(code);
else
  faceColor = [0.7 0.7 0.7];
end
fill(ax, [tStart, tEnd, tEnd, tStart], [0, 0, 1, 1], faceColor, ...
  'EdgeColor', 'none', 'HandleVisibility', 'off');
end
