function out = reach_criticality_metrics_engagement(sessionName, opts)
% REACH_CRITICALITY_METRICS_ENGAGEMENT - Criticality metrics by reach-task engagement
%
% Variables:
%   sessionName - Reach session identifier (no .mat extension)
%   opts        - Options struct (session loading + analysis selection). Fields:
%     Session loading (usual neuro_behavior_options overrides):
%       .collectStart, .collectEnd, .minFiringRate, .maxFiringRate,
%       .firingRateCheckTime
%     Analysis selection:
%       .analyses - Cell of {'d2','kurtosis','avalanches'} (any subset; default all)
%     Engagement:
%       .reachBuffer         - Seconds around each reach treated as engaged (default 1).
%                              d2/kurtosis: windows overlapping [reach-buffer, reach+buffer]
%                              count as engaged even without a reach onset inside.
%                              Avalanches: buffer excluded from non-engaged gaps.
%       .minNonEngagedWindow - Min gap without reaches (s) for non-engaged avalanche segments
%       .absorbSingleReaches - If true (default), isolated single reaches flanked by
%                              qualifying non-engaged gaps are merged into non-engaged
%                              time (avalanche segments and d2/kurtosis windows).
%     Shared analysis:
%       .brainArea, .brainAreaCombinations, .dataSource, .saveFigure, .outputDir
%       .makePlots - If false, skip all figures (batch mode; default true)
%       .plotConfig - axisLabelFontSize, tickLabelFontSize, axesLineWidth, ...
%       .nMinNeurons - Minimum neurons in analysis area(s) (default 20)
%       .useSubsampling - If true, metrics = mean across neuron subsamples
%       .nSubsamples, .nNeuronsSubsample, .minNeuronsMultiple - subsampling settings
%     d2:
%       .d2Window, .useLog10D2, .nShufflesD2
%       .runD2AccuracyCorrelation - If true (default), correlate d2 with reach
%                                   accuracy across total and engaged windows
%     Kurtosis (PRG):
%       .prgWindow, .prgMethod, .surrogateMethod, .nSurrogates
%     Avalanches:
%       .powerLawFitMethod, .avalancheDetectionMode, .runClausetPlpva, .gofThreshold
%       .enableCircularPermutations - If true, overlay shuffle CCDF for total only
%       .nShuffles - Number of circular permutations for total shuffle (default 5)
%
% Goal:
%   Test whether criticality metrics differ with task engagement by separating
%   time near reaches vs away from reaches (opts.reachBuffer, default 1 s):
%     d2 / kurtosis - windows overlapping any [reach +/- reachBuffer] = engaged
%                     (isolated single reaches may be absorbed; see absorbSingleReaches)
%     avalanches    - continuous gaps >= minNonEngagedWindow (outside reach
%                     buffers) = non-engaged; complement = engaged; pool
%                     avalanches across segments of each class (+ full session)
%   Plot total, engaged, and non-engaged metrics on shared axes and print
%   collected durations for each class.
%
% Returns:
%   With no inputs: default options struct (same fields as opts above).
%   Otherwise: struct with durations, segments, results, figHandles, config

setup_reach_criticality_metrics_engagement_paths();

if nargin == 0
  out = fill_engagement_opts_defaults(struct());
  return;
end
if nargin < 1 || isempty(sessionName)
  error('reach_criticality_metrics_engagement:MissingSession', ...
    'sessionName is required.');
end
if nargin < 2 || isempty(opts)
  opts = struct();
end
opts = fill_engagement_opts_defaults(opts);

sessionType = 'reach';
dataSource = opts.dataSource;
collectStart = opts.collectStart;
collectEnd = opts.collectEnd;

fprintf('\n=== Reach criticality metrics by engagement ===\n');
fprintf('Session: %s\n', sessionName);
fprintf('Analyses: %s\n', strjoin(opts.analyses, ', '));
fprintf('nMinNeurons: %d\n', opts.nMinNeurons);
if opts.useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
    opts.nSubsamples, opts.nNeuronsSubsample, opts.minNeuronsMultiple);
else
  fprintf('Subsampling: off\n');
end

%% Load session
loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = opts.firingRateCheckTime;
loadOpts.collectStart = collectStart;
loadOpts.collectEnd = collectEnd;
loadOpts.minFiringRate = opts.minFiringRate;
loadOpts.maxFiringRate = opts.maxFiringRate;

loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, '');
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

if ~isfield(dataStruct, 'reachStart') || isempty(dataStruct.reachStart)
  error('reach_criticality_metrics_engagement:NoReaches', ...
    'dataStruct.reachStart is required (reach session load failed?).');
end

if isempty(collectEnd)
  if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd')
    collectEnd = dataStruct.spikeData.collectEnd;
  else
    collectEnd = max(dataStruct.spikeTimes);
  end
  opts.collectEnd = collectEnd;
end
if isempty(collectStart)
  collectStart = 0;
  if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
    collectStart = dataStruct.spikeData.collectStart;
  end
  opts.collectStart = collectStart;
end

[dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
  dataStruct, opts.brainArea, opts.brainAreaCombinations, true);
if ~areaOk
  error('Brain area "%s" not available in this session.', opts.brainArea);
end

nUnits = count_session_neurons_for_brain_area(dataStruct, opts.brainArea);
fprintf('Neurons in analysis area(s): %d (min %d)\n', nUnits, opts.nMinNeurons);
if nUnits < opts.nMinNeurons
  error('reach_criticality_metrics_engagement:TooFewNeurons', ...
    'Only %d neurons in analysis area(s) (min %d).', nUnits, opts.nMinNeurons);
end
if opts.useSubsampling
  minNeuronsForSubsample = round(opts.nNeuronsSubsample * opts.minNeuronsMultiple);
  if nUnits < minNeuronsForSubsample
    error('reach_criticality_metrics_engagement:TooFewNeuronsForSubsample', ...
      'Only %d neurons; need at least %d for subsampling (%d x %.2f).', ...
      nUnits, minNeuronsForSubsample, opts.nNeuronsSubsample, opts.minNeuronsMultiple);
  end
end

reachStartAll = dataStruct.reachStart(:);
reachInCollect = reachStartAll >= collectStart & reachStartAll <= collectEnd;
reachStart = reachStartAll(reachInCollect);
reachClass = [];
if isfield(dataStruct, 'reachClass') && ~isempty(dataStruct.reachClass)
  reachClass = dataStruct.reachClass(:);
  if numel(reachClass) == numel(reachStartAll)
    reachClass = reachClass(reachInCollect);
  else
    reachClass = [];
  end
end
fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
  collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('Reaches in collect window: %d\n', numel(reachStart));
reachStartEngaged = filter_engaged_reach_onsets(reachStart, collectStart, collectEnd, ...
  opts.minNonEngagedWindow, opts.reachBuffer, opts.absorbSingleReaches);
if opts.absorbSingleReaches && numel(reachStartEngaged) < numel(reachStart)
  fprintf('absorbSingleReaches: %d isolated reach(s) merged into non-engaged time\n', ...
    numel(reachStart) - numel(reachStartEngaged));
end

paths = get_paths();
[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();

% Continuous engagement segments (avalanche definition; also used for overview plot)
[engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
  collectStart, collectEnd, reachStart, opts.minNonEngagedWindow, opts.reachBuffer, ...
  opts.absorbSingleReaches);
engagedSegs = label_segments(engagedSegs, 'Engaged');
nonEngagedSegs = label_segments(nonEngagedSegs, 'Non-engaged');

out = struct();
out.sessionName = sessionName;
out.config = opts;
out.reachStart = reachStart;
out.reachStartEngaged = reachStartEngaged;
out.segments = struct('engaged', engagedSegs, 'nonEngaged', nonEngagedSegs);
out.durations = struct();
out.d2 = [];
out.d2AccuracyCorrelation = [];
out.kurtosis = [];
out.avalanches = [];
out.figHandles = struct();

plotConfig = opts.plotConfig;

if opts.makePlots
  out.figHandles.segments = plot_reach_engagement_segments( ...
    reachStart, reachClass, engagedSegs, nonEngagedSegs, sessionName, ...
    collectStart, collectEnd, opts.minNonEngagedWindow, opts.reachBuffer, plotConfig);
end

%% d2: classify non-overlapping windows by reach buffer overlap
if ismember('d2', opts.analyses)
  fprintf('\n--- d2 ---\n');
  fprintf('reachBuffer: %.1f s (windows overlapping reach +/- buffer = engaged)\n', ...
    opts.reachBuffer);
  arConfig = build_ar_config(opts);
  resultsD2 = criticality_ar_analysis(dataStruct, arConfig);
  if ~isempty(opts.brainArea)
    resultsD2 = filter_ar_results_to_brain_area(resultsD2, opts.brainArea);
  end
  if isempty(resultsD2.areas)
    error('No d2 results for brain area "%s".', opts.brainArea);
  end

  d2Split = split_d2_by_reach_engagement(resultsD2, reachStartEngaged, collectStart, ...
    opts.d2Window, opts.useLog10D2, opts.reachBuffer);
  out.d2 = d2Split;
  out.durations.d2 = d2Split.durations;
  print_engagement_durations('d2', d2Split.durations);

  if opts.makePlots
    out.figHandles.d2 = plot_engagement_d2_distributions( ...
      d2Split, sessionType, sessionName, opts.d2Window, collectStart, collectEnd, ...
      opts.useLog10D2, plotConfig);
  end

  if opts.runD2AccuracyCorrelation
    if isempty(reachClass)
      warning('reach_criticality_metrics_engagement:NoReachClass', ...
        'Skipping d2-accuracy correlation: reachClass unavailable.');
    else
      fprintf('\n--- d2 vs reach accuracy (total and engaged windows) ---\n');
      d2AccCorr = build_d2_accuracy_correlation(resultsD2, reachStart, reachClass, ...
        reachStartEngaged, collectStart, opts.d2Window, opts.useLog10D2, opts.reachBuffer);
      out.d2AccuracyCorrelation = d2AccCorr;
      print_d2_accuracy_correlation(d2AccCorr);
      if opts.makePlots
        out.figHandles.d2AccuracyCorrelation = plot_d2_accuracy_correlation( ...
          d2AccCorr, sessionName, opts.d2Window, opts.useLog10D2, plotConfig);
      end
    end
  end
end

%% Kurtosis (PRG): classify non-overlapping blocks by reach buffer overlap
if ismember('kurtosis', opts.analyses)
  fprintf('\n--- Kurtosis (PRG) ---\n');
  fprintf('reachBuffer: %.1f s (blocks overlapping reach +/- buffer = engaged)\n', ...
    opts.reachBuffer);
  prgConfig = build_prg_config(opts);
  resultsPrg = criticality_prg_analysis(dataStruct, prgConfig);
  if ~isempty(opts.brainArea)
    resultsPrg = filter_prg_results_to_brain_area(resultsPrg, opts.brainArea);
  end
  if isempty(resultsPrg.areas)
    error('No PRG results for brain area "%s".', opts.brainArea);
  end

  prgSplit = split_prg_by_reach_engagement(resultsPrg, reachStartEngaged, opts.prgWindow, ...
    opts.reachBuffer);
  out.kurtosis = prgSplit;
  out.durations.kurtosis = prgSplit.durations;
  print_engagement_durations('kurtosis', prgSplit.durations);

  if opts.makePlots
    out.figHandles.kurtosis = plot_engagement_kurtosis_distributions( ...
      prgSplit, sessionType, sessionName, opts.prgWindow, collectStart, collectEnd, ...
      prgConfig.finalCutoffDivisor, prgConfig.prgMethod, plotConfig);
  end
end

%% Avalanches: pool across engaged / non-engaged continuous segments
if ismember('avalanches', opts.analyses)
  fprintf('\n--- Avalanches ---\n');
  fprintf('minNonEngagedWindow: %.1f s; reachBuffer: %.1f s\n', ...
    opts.minNonEngagedWindow, opts.reachBuffer);

  totalSeg = struct('start', collectStart, 'end', collectEnd, 'label', 'Total');
  engagedSegs = out.segments.engaged;
  nonEngagedSegs = out.segments.nonEngaged;

  avDurations = struct();
  avDurations.totalSec = collectEnd - collectStart;
  avDurations.engagedSec = sum_segment_durations(engagedSegs);
  avDurations.nonEngagedSec = sum_segment_durations(nonEngagedSegs);
  avDurations.nEngagedSegments = numel(engagedSegs);
  avDurations.nNonEngagedSegments = numel(nonEngagedSegs);
  out.durations.avalanches = avDurations;
  print_engagement_durations('avalanches', avDurations);
  print_segment_list('Engaged', engagedSegs);
  print_segment_list('Non-engaged', nonEngagedSegs);

  avConfig = build_av_config(opts, clausetPlfitPath, plfit2023Path);
  areasToAnalyze = resolve_areas_to_analyze(dataStruct, opts.brainArea, avConfig.nMinNeurons);
  if isempty(areasToAnalyze)
    error('No areas meet minimum neuron count (%d).', avConfig.nMinNeurons);
  end
  areaNames = dataStruct.areas(areasToAnalyze);

  avByClass = struct();
  fprintf('Total (full session)');
  if avConfig.enableCircularPermutations
    fprintf(', with %d circular shuffles', avConfig.nShuffles);
  end
  fprintf('...\n');
  avByClass.total = run_pooled_avalanche_analysis( ...
    dataStruct, areasToAnalyze, avConfig, totalSeg, avConfig.enableCircularPermutations);
  fprintf('Engaged...\n');
  avByClass.engaged = run_pooled_avalanche_analysis( ...
    dataStruct, areasToAnalyze, avConfig, engagedSegs, false);
  fprintf('Non-engaged...\n');
  avByClass.nonEngaged = run_pooled_avalanche_analysis( ...
    dataStruct, areasToAnalyze, avConfig, nonEngagedSegs, false);

  out.avalanches = struct();
  out.avalanches.segments = struct( ...
    'total', totalSeg, 'engaged', engagedSegs, 'nonEngaged', nonEngagedSegs);
  out.avalanches.areaNames = areaNames;
  out.avalanches.byClass = avByClass;
  out.avalanches.durations = avDurations;

  if opts.makePlots
    out.figHandles.avalanches = plot_engagement_avalanche_distributions( ...
      avByClass, areaNames, sessionName, collectStart, collectEnd, avDurations, plotConfig);
  end
end

%% Summary bar plots (mean +/- SEM) for d2, decades, dcc, kappa, D_JS
avByClass = [];
if isfield(out, 'avalanches') && ~isempty(out.avalanches) && isfield(out.avalanches, 'byClass')
  avByClass = out.avalanches.byClass;
end
if ~isempty(out.d2) || ~isempty(out.kurtosis) || ~isempty(avByClass)
  summaryStats = build_engagement_metric_summary(out.d2, out.kurtosis, avByClass, opts.useLog10D2);
  out.summary = summaryStats;
  if opts.makePlots
    out.figHandles.summary = plot_engagement_metric_summary( ...
      summaryStats, sessionName, collectStart, collectEnd, plotConfig);
  end
  print_engagement_metric_summary(summaryStats);
end

%% Optional figure export
if opts.saveFigure && opts.makePlots
  saveDir = opts.outputDir;
  if isempty(saveDir)
    saveDir = fullfile(paths.dropPath, 'reach_task', 'results', ...
      matlab.lang.makeValidName(sessionName));
  end
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  areaTag = format_areas_label(opts.brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  baseName = sprintf('reach_criticality_metrics_engagement_%s_%s_%.0f-%.0fs', ...
    sessionName, areaTag, collectStart, collectEnd);
  figFields = fieldnames(out.figHandles);
  for iFig = 1:numel(figFields)
    figName = figFields{iFig};
    fig = out.figHandles.(figName);
    if ~isgraphics(fig)
      continue;
    end
    plotBase = fullfile(saveDir, sprintf('%s_%s', baseName, figName));
    exportgraphics(fig, [plotBase, '.png'], 'Resolution', 300);
    exportgraphics(fig, [plotBase, '.eps'], 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', plotBase);
  end
end

fprintf('\n=== Done ===\n');
end

%% -------------------------------------------------------------------------
%% Defaults and paths
%% -------------------------------------------------------------------------

function opts = fill_engagement_opts_defaults(opts)
% FILL_ENGAGEMENT_OPTS_DEFAULTS - Session-load and analysis option defaults

if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
  opts.collectStart = 0;
end
if ~isfield(opts, 'collectEnd')
  opts.collectEnd = [];
end
if ~isfield(opts, 'minFiringRate') || isempty(opts.minFiringRate)
  opts.minFiringRate = 0.1;
end
if ~isfield(opts, 'maxFiringRate') || isempty(opts.maxFiringRate)
  opts.maxFiringRate = 100;
end
if ~isfield(opts, 'firingRateCheckTime')
  opts.firingRateCheckTime = [];
end
if ~isfield(opts, 'dataSource') || isempty(opts.dataSource)
  opts.dataSource = 'spikes';
end
if ~isfield(opts, 'analyses') || isempty(opts.analyses)
  opts.analyses = {'d2', 'kurtosis', 'avalanches'};
end
if ischar(opts.analyses) || isstring(opts.analyses)
  opts.analyses = cellstr(opts.analyses);
end
opts.analyses = lower(opts.analyses(:)');
validAnalyses = {'d2', 'kurtosis', 'avalanches'};
unknown = setdiff(opts.analyses, validAnalyses);
if ~isempty(unknown)
  error('reach_criticality_metrics_engagement:BadAnalyses', ...
    'Unknown analyses: %s. Use d2, kurtosis, and/or avalanches.', strjoin(unknown, ', '));
end

if ~isfield(opts, 'minNonEngagedWindow') || isempty(opts.minNonEngagedWindow)
  opts.minNonEngagedWindow = 30;
end
if ~isfield(opts, 'reachBuffer') || isempty(opts.reachBuffer)
  opts.reachBuffer = 1;
end
if ~isfield(opts, 'absorbSingleReaches') || isempty(opts.absorbSingleReaches)
  opts.absorbSingleReaches = true;
end
if ~isfield(opts, 'brainArea')
  opts.brainArea = 'M23M56';
end
if ~isfield(opts, 'brainAreaCombinations') || isempty(opts.brainAreaCombinations)
  opts.brainAreaCombinations = default_manuscript_brain_area_combinations();
end
if ~isfield(opts, 'saveFigure') || isempty(opts.saveFigure)
  opts.saveFigure = false;
end
if ~isfield(opts, 'makePlots') || isempty(opts.makePlots)
  opts.makePlots = true;
end
if ~isfield(opts, 'outputDir')
  opts.outputDir = '';
end

if ~isfield(opts, 'nMinNeurons') || isempty(opts.nMinNeurons)
  opts.nMinNeurons = 20;
end
if ~isfield(opts, 'useSubsampling') || isempty(opts.useSubsampling)
  opts.useSubsampling = false;
end
if ~isfield(opts, 'nSubsamples') || isempty(opts.nSubsamples)
  opts.nSubsamples = 20;
end
if ~isfield(opts, 'nNeuronsSubsample') || isempty(opts.nNeuronsSubsample)
  opts.nNeuronsSubsample = 32;
end
if ~isfield(opts, 'minNeuronsMultiple') || isempty(opts.minNeuronsMultiple)
  opts.minNeuronsMultiple = 1.25;
end

if ~isfield(opts, 'd2Window') || isempty(opts.d2Window)
  opts.d2Window = 30;
end
if ~isfield(opts, 'useLog10D2') || isempty(opts.useLog10D2)
  opts.useLog10D2 = true;
end
if ~isfield(opts, 'nShufflesD2') || isempty(opts.nShufflesD2)
  opts.nShufflesD2 = 50;
end
if ~isfield(opts, 'runD2AccuracyCorrelation') || isempty(opts.runD2AccuracyCorrelation)
  opts.runD2AccuracyCorrelation = true;
end

if ~isfield(opts, 'prgWindow') || isempty(opts.prgWindow)
  opts.prgWindow = 30;
end
if ~isfield(opts, 'prgMethod') || isempty(opts.prgMethod)
  opts.prgMethod = 'pca';
end
if ~isfield(opts, 'surrogateMethod') || isempty(opts.surrogateMethod)
  opts.surrogateMethod = 'isi';
end
if ~isfield(opts, 'nSurrogates') || isempty(opts.nSurrogates)
  opts.nSurrogates = 1;
end

if ~isfield(opts, 'powerLawFitMethod') || isempty(opts.powerLawFitMethod)
  opts.powerLawFitMethod = 'plfit2023';
end
if ~isfield(opts, 'avalancheDetectionMode') || isempty(opts.avalancheDetectionMode)
  opts.avalancheDetectionMode = 'fixedBinMedian';
end
if ~isfield(opts, 'runClausetPlpva') || isempty(opts.runClausetPlpva)
  opts.runClausetPlpva = false;
end
if ~isfield(opts, 'gofThreshold') || isempty(opts.gofThreshold)
  opts.gofThreshold = 0.8;
end
if ~isfield(opts, 'enableCircularPermutations') || isempty(opts.enableCircularPermutations)
  opts.enableCircularPermutations = true;
end
if ~isfield(opts, 'nShuffles') || isempty(opts.nShuffles)
  opts.nShuffles = 5;
end
if ~isfield(opts, 'plotConfig') || isempty(opts.plotConfig)
  opts.plotConfig = struct();
end
opts.plotConfig = fill_default_engagement_plot_config(opts.plotConfig);
end

function plotConfig = fill_default_engagement_plot_config(plotConfig)
% FILL_DEFAULT_ENGAGEMENT_PLOT_CONFIG - Manuscript-style axis fonts and line widths

if nargin < 1 || isempty(plotConfig)
  plotConfig = struct();
end
if ~isfield(plotConfig, 'axisLabelFontSize') || isempty(plotConfig.axisLabelFontSize)
  plotConfig.axisLabelFontSize = 14;
end
if ~isfield(plotConfig, 'tickLabelFontSize') || isempty(plotConfig.tickLabelFontSize)
  plotConfig.tickLabelFontSize = 12;
end
if ~isfield(plotConfig, 'titleFontSize') || isempty(plotConfig.titleFontSize)
  plotConfig.titleFontSize = 13;
end
if ~isfield(plotConfig, 'sgtitleFontSize') || isempty(plotConfig.sgtitleFontSize)
  plotConfig.sgtitleFontSize = 14;
end
if ~isfield(plotConfig, 'axesLineWidth') || isempty(plotConfig.axesLineWidth)
  plotConfig.axesLineWidth = 1.5;
end
if ~isfield(plotConfig, 'markerSize') || isempty(plotConfig.markerSize)
  plotConfig.markerSize = 6;
end
if ~isfield(plotConfig, 'lineWidth') || isempty(plotConfig.lineWidth)
  plotConfig.lineWidth = 1.5;
end
if ~isfield(plotConfig, 'errorCapSize') || isempty(plotConfig.errorCapSize)
  plotConfig.errorCapSize = 8;
end
if ~isfield(plotConfig, 'legendFontSize') || isempty(plotConfig.legendFontSize)
  plotConfig.legendFontSize = 11;
end
if ~isfield(plotConfig, 'shuffleMarkerSize') || isempty(plotConfig.shuffleMarkerSize)
  plotConfig.shuffleMarkerSize = 4;
end
end

function apply_engagement_axes_style(ax, plotConfig, xLabelText, yLabelText, titleText, textInterpreter)
% APPLY_ENGAGEMENT_AXES_STYLE - Thicker axes and larger fonts (manuscript style)
%
% Variables:
%   ax              - Axes handle
%   plotConfig      - Font/line-width settings
%   xLabelText      - X label ('' or omit to skip)
%   yLabelText      - Y label ('' or omit to skip)
%   titleText       - Title ('' or omit to skip)
%   textInterpreter - Optional interpreter for labels/title (default 'none')

if nargin < 6 || isempty(textInterpreter)
  textInterpreter = 'none';
end

set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
  'Box', 'off', 'TickDir', 'out');
if nargin >= 3 && ~isempty(xLabelText)
  xlabel(ax, xLabelText, 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', textInterpreter);
end
if nargin >= 4 && ~isempty(yLabelText)
  ylabel(ax, yLabelText, 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', textInterpreter);
end
if nargin >= 5 && ~isempty(titleText)
  title(ax, titleText, 'FontSize', plotConfig.titleFontSize, 'Interpreter', textInterpreter);
end
end

function setup_reach_criticality_metrics_engagement_paths()
% SETUP_REACH_CRITICALITY_METRICS_ENGAGEMENT_PATHS - Add neuro-behavior paths if present
%
% Goal:
%   Resolve the real function location (Editor temp copies use a fake path) and
%   only addpath directories that exist, avoiding warnings when src is already on path.

scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_']) || ~isfolder(fullfile(scriptDir, '..', 'reach_task'))
  resolved = which('reach_criticality_metrics_engagement');
  if ~isempty(resolved)
    scriptDir = fileparts(resolved);
  end
end
srcPath = fullfile(scriptDir, '..');
pathDirs = {
  srcPath
  fullfile(srcPath, 'reach_task')
  fullfile(srcPath, 'schall')
  fullfile(srcPath, 'spontaneous')
  fullfile(srcPath, 'interval_timing_task')
  fullfile(srcPath, 'criticality', 'scripts')
  fullfile(srcPath, 'criticality', 'analyses')
  fullfile(srcPath, 'criticality_manuscript')
  fullfile(srcPath, 'session_prep', 'data_prep')
  fullfile(srcPath, 'session_prep', 'utils')
  fullfile(srcPath, 'data_prep')
  fullfile(srcPath, 'sliding_window_prep', 'utils')
  fullfile(srcPath, 'criticality')
  };
for iDir = 1:numel(pathDirs)
  if isfolder(pathDirs{iDir})
    addpath(pathDirs{iDir});
  end
end
end

function arConfig = build_ar_config(opts)
% BUILD_AR_CONFIG - d2 / AR settings (non-overlapping windows)

arConfig = struct();
arConfig.slidingWindowSize = opts.d2Window;
arConfig.stepSize = opts.d2Window;
arConfig.binSize = 0.025;
arConfig.useOptimalBinWindowFunction = false;
arConfig.analyzeD2 = true;
arConfig.analyzeMrBr = false;
arConfig.pcaFlag = 0;
arConfig.pcaFirstFlag = 1;
arConfig.nDim = 4;
arConfig.enablePermutations = true;
arConfig.nShuffles = opts.nShufflesD2;
arConfig.normalizeD2 = true;
arConfig.useLog10D2 = opts.useLog10D2;
arConfig.makePlots = false;
arConfig.saveData = false;
arConfig.pOrder = 10;
arConfig.critType = 2;
arConfig.minSpikesPerBin = 2.5;
arConfig.minBinsPerWindow = 1000;
arConfig.maxSpikesPerBin = 100;
arConfig.nMinNeurons = opts.nMinNeurons;
arConfig.useSubsampling = opts.useSubsampling;
arConfig.nSubsamples = opts.nSubsamples;
arConfig.nNeuronsSubsample = opts.nNeuronsSubsample;
arConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
end

function prgConfig = build_prg_config(opts)
% BUILD_PRG_CONFIG - PRG kurtosis settings (non-overlapping blocks)

prgConfig = struct();
prgConfig.prgMethod = opts.prgMethod;
prgConfig.blockWindowSize = opts.prgWindow;
prgConfig.binSize = 0.05;
prgConfig.cvThreshold = 5;
prgConfig.cutoffDivisors = [1, 2, 4, 8, 16];
prgConfig.finalCutoffDivisor = 16;
prgConfig.kappaAxisMax = 20;
prgConfig.enableSurrogates = true;
prgConfig.nSurrogates = opts.nSurrogates;
prgConfig.surrogateMethod = opts.surrogateMethod;
prgConfig.makePlots = false;
prgConfig.saveData = false;
prgConfig.plotTimeSeries = false;
prgConfig.nMinNeurons = opts.nMinNeurons;
prgConfig.useSubsampling = opts.useSubsampling;
prgConfig.nSubsamples = opts.nSubsamples;
prgConfig.nNeuronsSubsample = opts.nNeuronsSubsample;
prgConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
end

function avConfig = build_av_config(opts, clausetPlfitPath, plfit2023Path)
% BUILD_AV_CONFIG - Avalanche detection and power-law fit settings

avConfig = struct();
avConfig.useOptimalBinWindowFunction = false;
avConfig.avalancheDetectionMode = opts.avalancheDetectionMode;
if strcmpi(opts.avalancheDetectionMode, 'fixedBinMedian')
  avConfig.binSize = 0.05;
end
avConfig.thresholdFlag = 1;
avConfig.thresholdPct = 1;
avConfig.nMinNeurons = opts.nMinNeurons;
avConfig.useSubsampling = opts.useSubsampling;
avConfig.nSubsamples = opts.nSubsamples;
avConfig.nNeuronsSubsample = opts.nNeuronsSubsample;
avConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
avConfig.pcaFlag = 0;
avConfig.gofThreshold = opts.gofThreshold;
avConfig.powerLawFitMethod = opts.powerLawFitMethod;
avConfig.runClausetPlpva = opts.runClausetPlpva;
avConfig.clausetPlfitPath = clausetPlfitPath;
avConfig.plfit2023Path = plfit2023Path;
avConfig.enableCircularPermutations = opts.enableCircularPermutations;
avConfig.nShuffles = opts.nShuffles;
end

%% -------------------------------------------------------------------------
%% Engagement segment definitions (avalanches)
%% -------------------------------------------------------------------------

function [engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
    collectStart, collectEnd, reachStart, minNonEngagedWindow, reachBuffer, absorbSingleReaches)
% DEFINE_REACH_ENGAGEMENT_SEGMENTS - Continuous engaged / non-engaged intervals
%
% Variables:
%   collectStart, collectEnd - Session analysis bounds (s)
%   reachStart               - Reach onset times in collect window (s)
%   minNonEngagedWindow      - Minimum gap without reaches (s)
%   reachBuffer              - Seconds around each reach excluded from non-engaged
%   absorbSingleReaches      - If true, merge isolated single reaches into flanking gaps
%
% Goal:
%   Each reach occupies [reach - reachBuffer, reach + reachBuffer]. Non-engaged =
%   gaps between these buffered neighborhoods (and session edges) with duration
%   >= minNonEngagedWindow. Engaged = complement (includes all reach buffers).
%   When absorbSingleReaches is true, an occupied interval with exactly one reach
%   that is flanked on both sides by qualifying non-engaged gaps is absorbed: the
%   reach buffer and both gaps form one non-engaged segment.

if nargin < 5 || isempty(reachBuffer)
  reachBuffer = 0;
end
if nargin < 6 || isempty(absorbSingleReaches)
  absorbSingleReaches = true;
end
reachBuffer = max(0, reachBuffer);

reachStart = sort(reachStart(:));
reachStart = reachStart(reachStart >= collectStart & reachStart <= collectEnd);

% Merge overlapping reach neighborhoods into occupied intervals
occupied = merge_reach_buffer_intervals(reachStart, reachBuffer, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if absorbSingleReaches && ~isempty(occupied)
  absorbedMask = get_absorbed_single_reach_occupied_mask( ...
    reachStart, collectStart, collectEnd, minNonEngagedWindow, reachBuffer);
end

nonEngagedSegs = struct('start', {}, 'end', {});
cursor = collectStart;
iOcc = 1;
while iOcc <= numel(occupied)
  if absorbedMask(iOcc)
    gapStart = cursor;
    if iOcc < numel(occupied)
      gapEnd = occupied(iOcc + 1).start;
    else
      gapEnd = collectEnd;
    end
    if (gapEnd - gapStart) >= minNonEngagedWindow
      nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
      nonEngagedSegs(end).end = gapEnd;
    end
    cursor = gapEnd;
  else
    gapStart = cursor;
    gapEnd = occupied(iOcc).start;
    if (gapEnd - gapStart) >= minNonEngagedWindow
      nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
      nonEngagedSegs(end).end = gapEnd;
    end
    cursor = occupied(iOcc).end;
  end
  iOcc = iOcc + 1;
end
if (collectEnd - cursor) >= minNonEngagedWindow
  nonEngagedSegs(end + 1).start = cursor;
  nonEngagedSegs(end).end = collectEnd;
end

engagedSegs = complement_segments(collectStart, collectEnd, nonEngagedSegs);
end

function absorbedMask = get_absorbed_single_reach_occupied_mask( ...
    reachStart, collectStart, collectEnd, minNonEngagedWindow, reachBuffer)
% GET_ABSORBED_SINGLE_REACH_OCCUPIED_MASK - Isolated single reaches to absorb
%
% Variables:
%   reachStart, collectStart, collectEnd, minNonEngagedWindow, reachBuffer
%
% Goal:
%   Mark merged reach-buffer intervals that contain exactly one reach and are
%   flanked on both sides by gaps >= minNonEngagedWindow.

occupied = merge_reach_buffer_intervals(reachStart, reachBuffer, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if isempty(occupied)
  return;
end

for iOcc = 1:numel(occupied)
  nReachesInOcc = sum(reachStart >= occupied(iOcc).start & reachStart <= occupied(iOcc).end);
  if nReachesInOcc ~= 1
    continue;
  end
  if iOcc == 1
    gapBeforeStart = collectStart;
  else
    gapBeforeStart = occupied(iOcc - 1).end;
  end
  gapBeforeEnd = occupied(iOcc).start;
  gapAfterStart = occupied(iOcc).end;
  if iOcc == numel(occupied)
    gapAfterEnd = collectEnd;
  else
    gapAfterEnd = occupied(iOcc + 1).start;
  end
  if (gapBeforeEnd - gapBeforeStart) >= minNonEngagedWindow && ...
      (gapAfterEnd - gapAfterStart) >= minNonEngagedWindow
    absorbedMask(iOcc) = true;
  end
end
end

function reachStartEngaged = filter_engaged_reach_onsets(reachStart, collectStart, ...
    collectEnd, minNonEngagedWindow, reachBuffer, absorbSingleReaches)
% FILTER_ENGAGED_REACH_ONSETS - Reach onsets that count as engaged
%
% Variables:
%   reachStart, collectStart, collectEnd, minNonEngagedWindow, reachBuffer
%   absorbSingleReaches - If true, drop isolated single reaches absorbed into gaps
%
% Goal:
%   Return reach onsets whose buffers should be treated as engaged for d2/kurtosis.

reachStart = sort(reachStart(:));
if nargin < 6 || isempty(absorbSingleReaches)
  absorbSingleReaches = true;
end
if ~absorbSingleReaches || isempty(reachStart)
  reachStartEngaged = reachStart;
  return;
end

occupied = merge_reach_buffer_intervals(reachStart, reachBuffer, collectStart, collectEnd);
absorbedMask = get_absorbed_single_reach_occupied_mask( ...
  reachStart, collectStart, collectEnd, minNonEngagedWindow, reachBuffer);
keep = true(size(reachStart));
for iReach = 1:numel(reachStart)
  for iOcc = 1:numel(occupied)
    if absorbedMask(iOcc) && reachStart(iReach) >= occupied(iOcc).start && ...
        reachStart(iReach) <= occupied(iOcc).end
      keep(iReach) = false;
      break;
    end
  end
end
reachStartEngaged = reachStart(keep);
end

function occupied = merge_reach_buffer_intervals(reachStart, reachBuffer, collectStart, collectEnd)
% MERGE_REACH_BUFFER_INTERVALS - Union of [reach-buffer, reach+buffer] in collect window
%
% Variables:
%   reachStart               - Reach onset times (s)
%   reachBuffer              - Half-width around each reach (s)
%   collectStart, collectEnd - Clip bounds (s)
%
% Goal:
%   Return non-overlapping occupied intervals sorted by start time.

occupied = struct('start', {}, 'end', {});
if isempty(reachStart)
  return;
end

starts = max(collectStart, reachStart - reachBuffer);
ends = min(collectEnd, reachStart + reachBuffer);
valid = ends > starts;
starts = starts(valid);
ends = ends(valid);
if isempty(starts)
  return;
end

[starts, ord] = sort(starts);
ends = ends(ord);

occupied(1).start = starts(1);
occupied(1).end = ends(1);
for i = 2:numel(starts)
  if starts(i) <= occupied(end).end
    occupied(end).end = max(occupied(end).end, ends(i));
  else
    occupied(end + 1).start = starts(i); %#ok<AGROW>
    occupied(end).end = ends(i);
  end
end
end

function engagedSegs = complement_segments(collectStart, collectEnd, nonEngagedSegs)
% COMPLEMENT_SEGMENTS - Intervals in [collectStart, collectEnd] not in nonEngagedSegs

engagedSegs = struct('start', {}, 'end', {});
if isempty(nonEngagedSegs)
  engagedSegs(1).start = collectStart;
  engagedSegs(1).end = collectEnd;
  return;
end

starts = [nonEngagedSegs.start];
ends = [nonEngagedSegs.end];
[starts, ord] = sort(starts);
ends = ends(ord);

cursor = collectStart;
for i = 1:numel(starts)
  if starts(i) > cursor + eps
    engagedSegs(end + 1).start = cursor; %#ok<AGROW>
    engagedSegs(end).end = starts(i);
  end
  cursor = max(cursor, ends(i));
end
if cursor < collectEnd - eps
  engagedSegs(end + 1).start = cursor;
  engagedSegs(end).end = collectEnd;
end
end

function segs = label_segments(segs, labelText)
% LABEL_SEGMENTS - Attach a common label to each segment struct

for i = 1:numel(segs)
  segs(i).label = labelText;
end
end

function totalSec = sum_segment_durations(segs)
% SUM_SEGMENT_DURATIONS - Total duration across segment structs

totalSec = 0;
for i = 1:numel(segs)
  totalSec = totalSec + (segs(i).end - segs(i).start);
end
end

function print_segment_list(labelText, segs)
% PRINT_SEGMENT_LIST - Command-window list of segment intervals

fprintf('  %s segments (%d):\n', labelText, numel(segs));
if isempty(segs)
  fprintf('    (none)\n');
  return;
end
for i = 1:numel(segs)
  fprintf('    [%7.1f, %7.1f] s (%.1f s)\n', ...
    segs(i).start, segs(i).end, segs(i).end - segs(i).start);
end
end

function fig = plot_reach_engagement_segments(reachStartSec, reachClass, engagedSegs, ...
    nonEngagedSegs, sessionName, collectStart, collectEnd, minNonEngagedWindow, ...
    reachBuffer, plotConfig)
% PLOT_REACH_ENGAGEMENT_SEGMENTS - Reaches and engaged / non-engaged intervals
%
% Variables:
%   reachStartSec        - Reach onset times in collect window (s)
%   reachClass           - Reach class labels (optional; for Block 2 marker)
%   engagedSegs          - Struct array with .start, .end
%   nonEngagedSegs       - Struct array with .start, .end
%   sessionName          - Session id for title
%   collectStart/End     - Analysis bounds (s)
%   minNonEngagedWindow  - Gap threshold used for non-engaged (s)
%   reachBuffer          - Buffer around reaches excluded from non-engaged (s)
%   plotConfig           - Axis fonts and line widths
%
% Goal:
%   Schematic overview like reach_task_engagement: vertical lines at reaches,
%   shaded engaged (blue) and non-engaged (orange) segments.

plotConfig = fill_default_engagement_plot_config(plotConfig);
yMin = 0;
yMax = 1;
engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];

fig = figure('Color', 'w', 'Name', 'Reach engagement segments', ...
  'Position', [100 400 1100 240]);
ax = axes(fig);
hold(ax, 'on');

segmentHandles = gobjects(0);
segmentLabels = {};

hNon = add_segment_patches(ax, nonEngagedSegs, nonEngagedColor, yMin, yMax);
if ~isempty(hNon)
  segmentHandles(end + 1) = hNon; %#ok<AGROW>
  segmentLabels{end + 1} = sprintf('Non-engaged (n=%d)', numel(nonEngagedSegs)); %#ok<AGROW>
end

hEng = add_segment_patches(ax, engagedSegs, engagedColor, yMin, yMax);
if ~isempty(hEng)
  segmentHandles(end + 1) = hEng; %#ok<AGROW>
  segmentLabels{end + 1} = sprintf('Engaged (n=%d)', numel(engagedSegs)); %#ok<AGROW>
end

for iReach = 1:numel(reachStartSec)
  x = reachStartSec(iReach);
  plot(ax, [x, x], [yMin, yMax], 'Color', [0.35, 0.35, 0.35], 'LineWidth', 0.75, ...
    'HandleVisibility', 'off');
end

if ~isempty(reachClass)
  block2Reaches = (reachClass == 3) | (reachClass == 4);
  if any(block2Reaches)
    block2StartTime = reachStartSec(find(block2Reaches, 1, 'first'));
    hBlock2 = plot(ax, [block2StartTime, block2StartTime], [yMin, yMax], 'r', ...
      'LineWidth', plotConfig.lineWidth + 0.5, 'DisplayName', 'Block 2 start');
    segmentHandles(end + 1) = hBlock2; %#ok<AGROW>
    segmentLabels{end + 1} = 'Block 2 start'; %#ok<AGROW>
  end
end

xlim(ax, [collectStart, collectEnd]);
ylim(ax, [yMin, yMax]);
yticks(ax, []);
titleText = sprintf( ...
  ['%s — reach engagement segments [%.0f–%.0f s]\n', ...
  'minNonEngagedWindow=%.1f s, reachBuffer=%.1f s | %d reaches'], ...
  sessionName, collectStart, collectEnd, minNonEngagedWindow, reachBuffer, ...
  numel(reachStartSec));
apply_engagement_axes_style(ax, plotConfig, 'Time (s)', 'Engagement (schematic)', titleText);

if ~isempty(segmentHandles)
  legend(ax, segmentHandles, segmentLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize);
end
hold(ax, 'off');
end

function h = add_segment_patches(ax, segs, colorVal, yMin, yMax)
% ADD_SEGMENT_PATCHES - Shade engagement intervals; return one legend handle

h = gobjects(0);
for i = 1:numel(segs)
  t0 = segs(i).start;
  t1 = segs(i).end;
  hi = patch(ax, [t0, t1, t1, t0], [yMin, yMin, yMax, yMax], colorVal, ...
    'FaceAlpha', 0.35, 'EdgeColor', 'none', 'HandleVisibility', 'off');
  if isempty(h)
    h = hi;
    set(h, 'HandleVisibility', 'on');
  end
end
end

function summaryStats = build_engagement_metric_summary(d2Split, prgSplit, avByClass, useLog10D2)
% BUILD_ENGAGEMENT_METRIC_SUMMARY - Mean +/- SEM per engagement class
%
% Variables:
%   d2Split     - Output of split_d2_by_reach_engagement, or []
%   prgSplit    - Output of split_prg_by_reach_engagement, or []
%   avByClass   - Avalanche results by class (.total, .engaged, .nonEngaged), or []
%   useLog10D2  - If true, d2 label uses log10(d2)
%
% Goal:
%   Pool window-wise values across areas within each engagement class and
%   compute mean, SEM, and n for d2, kappa, and D_JS. Avalanche decades and dcc
%   use one scalar per area per class (mean +/- SEM across areas).

classNames = engagement_class_names();
summaryStats = struct();
summaryStats.classNames = classNames;
summaryStats.metrics = {};

if ~isempty(d2Split)
  d2Label = 'd2';
  if useLog10D2
    d2Label = 'log_{10}(d2)';
  end
  metric = struct('name', 'd2', 'label', d2Label);
  metric.stats = class_metric_mean_sem(d2Split.d2);
  summaryStats.metrics{end + 1} = metric;
end

if ~isempty(avByClass)
  metric = struct('name', 'decades', 'label', 'decades');
  metric.stats = class_avalanche_scalar_mean_sem(avByClass, 'decades');
  summaryStats.metrics{end + 1} = metric;

  metric = struct('name', 'dcc', 'label', 'dcc');
  metric.stats = class_avalanche_scalar_mean_sem(avByClass, 'dcc');
  summaryStats.metrics{end + 1} = metric;
end

if ~isempty(prgSplit)
  metric = struct('name', 'kappa', 'label', '\kappa');
  metric.stats = class_metric_mean_sem(prgSplit.kappa);
  summaryStats.metrics{end + 1} = metric;

  metric = struct('name', 'djs', 'label', 'D_{JS}');
  metric.stats = class_metric_mean_sem(prgSplit.djs);
  summaryStats.metrics{end + 1} = metric;
end
end

function stats = class_avalanche_scalar_mean_sem(avByClass, metricName)
% CLASS_AVALANCHE_SCALAR_MEAN_SEM - Per-class mean/SEM for pooled avalanche metrics
%
% Variables:
%   avByClass   - Struct with .total, .engaged, .nonEngaged avalanche results
%   metricName  - Scalar field on avData (e.g. decades, tau, dcc)
%
% Goal:
%   For each engagement class, collect the metric from each area with avalanches
%   and summarize with mean +/- SEM across areas.

classFields = {'total', 'engaged', 'nonEngaged'};
nClasses = numel(classFields);
stats = repmat(struct('mean', nan, 'sem', nan, 'n', 0), 1, nClasses);
for c = 1:nClasses
  if ~isfield(avByClass, classFields{c})
    continue;
  end
  avClassResult = avByClass.(classFields{c});
  vals = [];
  if isstruct(avClassResult) && isfield(avClassResult, 'byArea')
    for a = 1:numel(avClassResult.byArea)
      value = area_avalanche_metric_value(avClassResult.byArea{a}, metricName);
      if isfinite(value)
        vals(end + 1) = value; %#ok<AGROW>
      end
    end
  end
  stats(c).n = numel(vals);
  if stats(c).n == 0
    continue;
  end
  stats(c).mean = mean(vals);
  if stats(c).n > 1
    stats(c).sem = std(vals, 0) / sqrt(stats(c).n);
  else
    stats(c).sem = 0;
  end
end
end

function value = area_avalanche_metric_value(avData, metricName)
% AREA_AVALANCHE_METRIC_VALUE - Scalar avalanche metric from one area

value = nan;
if ~isstruct(avData) || ~isfield(avData, 'hasAvalanches') || ~avData.hasAvalanches
  return;
end
if isfield(avData, metricName) && isfinite(avData.(metricName))
  value = avData.(metricName);
  return;
end
if strcmp(metricName, 'decades') && isfield(avData, 'sizeFitInfo') ...
    && isstruct(avData.sizeFitInfo) && isfield(avData.sizeFitInfo, 'decades') ...
    && isfinite(avData.sizeFitInfo.decades)
  value = avData.sizeFitInfo.decades;
end
end

function stats = class_metric_mean_sem(classMetric)
% CLASS_METRIC_MEAN_SEM - Per-class mean/SEM after pooling areas
%
% Variables:
%   classMetric - Cell {class}{area} of value vectors
%
% Returns:
%   stats - Struct array (.mean, .sem, .n) length = nClasses

nClasses = numel(classMetric);
stats = repmat(struct('mean', nan, 'sem', nan, 'n', 0), 1, nClasses);
for c = 1:nClasses
  vals = [];
  for a = 1:numel(classMetric{c})
    vals = [vals; classMetric{c}{a}(:)]; %#ok<AGROW>
  end
  vals = vals(isfinite(vals));
  stats(c).n = numel(vals);
  if stats(c).n == 0
    continue;
  end
  stats(c).mean = mean(vals);
  if stats(c).n > 1
    stats(c).sem = std(vals, 0) / sqrt(stats(c).n);
  else
    stats(c).sem = 0;
  end
end
end

function fig = plot_engagement_metric_summary(summaryStats, sessionName, collectStart, ...
    collectEnd, plotConfig)
% PLOT_ENGAGEMENT_METRIC_SUMMARY - Bar plots of mean +/- SEM per metric
%
% Variables:
%   summaryStats - From build_engagement_metric_summary
%   plotConfig   - Axis fonts and line widths
%
% Goal:
%   One axes per metric (d2, decades, dcc, kappa, D_JS), bars for Total / Engaged / Non-engaged.

plotConfig = fill_default_engagement_plot_config(plotConfig);
classNames = summaryStats.classNames;
classColors = engagement_class_colors();
nMetrics = numel(summaryStats.metrics);
if nMetrics == 0
  fig = gobjects(0);
  return;
end

fig = figure('Color', 'w', 'Name', 'Engagement metric summary', ...
  'Position', [120 120 340 * nMetrics 380]);
tileLayout = tiledlayout(fig, 1, nMetrics, 'TileSpacing', 'compact', 'Padding', 'compact');

xPos = 1:numel(classNames);
for m = 1:nMetrics
  metric = summaryStats.metrics{m};
  ax = nexttile(tileLayout);
  hold(ax, 'on');

  means = [metric.stats.mean];
  sems = [metric.stats.sem];
  for c = 1:numel(classNames)
    if ~isfinite(means(c))
      continue;
    end
    bar(ax, xPos(c), means(c), 0.65, ...
      'FaceColor', classColors(c, :), 'EdgeColor', 'none', 'FaceAlpha', 0.9, ...
      'DisplayName', sprintf('%s (n=%d)', classNames{c}, metric.stats(c).n));
    if isfinite(sems(c)) && sems(c) > 0
      errorbar(ax, xPos(c), means(c), sems(c), 'Color', [0.15, 0.15, 0.15], ...
        'LineStyle', 'none', 'LineWidth', plotConfig.lineWidth, ...
        'CapSize', plotConfig.errorCapSize, 'HandleVisibility', 'off');
    end
  end

  set(ax, 'XTick', xPos, 'XTickLabel', classNames);
  xtickangle(ax, 20);
  apply_engagement_axes_style(ax, plotConfig);
  ylabel(ax, metric.label, 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', 'tex');
  title(ax, metric.label, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'tex');
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayout, sprintf('%s — engagement summary [%.0f–%.0f s] (mean \\pm SEM)', ...
  sessionName, collectStart, collectEnd), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function print_engagement_metric_summary(summaryStats)
% PRINT_ENGAGEMENT_METRIC_SUMMARY - Command-window mean +/- SEM table

fprintf('\n=== Engagement metric summary (mean +/- SEM) ===\n');
classNames = summaryStats.classNames;
for m = 1:numel(summaryStats.metrics)
  metric = summaryStats.metrics{m};
  fprintf('  %s:\n', metric.name);
  for c = 1:numel(classNames)
    s = metric.stats(c);
    if s.n == 0 || ~isfinite(s.mean)
      fprintf('    %-12s  n=0\n', classNames{c});
    else
      fprintf('    %-12s  %.4f +/- %.4f  (n=%d)\n', ...
        classNames{c}, s.mean, s.sem, s.n);
    end
  end
end
end

function print_engagement_durations(analysisName, durations)
% PRINT_ENGAGEMENT_DURATIONS - Total / engaged / non-engaged time summary

fprintf('\n=== %s durations ===\n', analysisName);
if isfield(durations, 'totalSec')
  fprintf('  Total:       %8.1f s (%.2f min)\n', durations.totalSec, durations.totalSec / 60);
  fprintf('  Engaged:     %8.1f s (%.2f min)', durations.engagedSec, durations.engagedSec / 60);
  if isfield(durations, 'nEngagedWindows')
    fprintf('  [%d windows]', durations.nEngagedWindows);
  elseif isfield(durations, 'nEngagedSegments')
    fprintf('  [%d segments]', durations.nEngagedSegments);
  end
  fprintf('\n');
  fprintf('  Non-engaged: %8.1f s (%.2f min)', durations.nonEngagedSec, durations.nonEngagedSec / 60);
  if isfield(durations, 'nNonEngagedWindows')
    fprintf('  [%d windows]', durations.nNonEngagedWindows);
  elseif isfield(durations, 'nNonEngagedSegments')
    fprintf('  [%d segments]', durations.nNonEngagedSegments);
  end
  fprintf('\n');
end
end

%% -------------------------------------------------------------------------
%% d2 split and plot
%% -------------------------------------------------------------------------

function d2Split = split_d2_by_reach_engagement(results, reachStart, collectStart, ...
    d2Window, useLog10D2, reachBuffer)
% SPLIT_D2_BY_REACH_ENGAGEMENT - Partition window-wise d2 by reach engagement
%
% Variables:
%   results      - Output of criticality_ar_analysis
%   reachStart   - Reach onsets in collect window (absolute s)
%   collectStart - Collect window start (s); startS is relative to this
%   d2Window     - Non-overlapping window length (s)
%   useLog10D2   - If true, store log10(d2) for plotting
%   reachBuffer  - Seconds around each reach that count as engaged (s)
%
% Goal:
%   Windows overlapping any [reach - reachBuffer, reach + reachBuffer] are engaged
%   (including windows with no reach onset but within the buffer). Others are
%   non-engaged. Total uses all windows. Duration = nWindows * d2Window per class.

if nargin < 6 || isempty(reachBuffer)
  reachBuffer = 0;
end

classNames = engagement_class_names();
d2Split = struct();
d2Split.areas = normalize_results_area_list(results.areas);
d2Split.classNames = classNames;
d2Split.d2 = cell(1, numel(classNames));
d2Split.d2Normalized = cell(1, numel(classNames));
d2Split.windowMask = cell(1, numel(d2Split.areas));
for c = 1:numel(classNames)
  d2Split.d2{c} = cell(1, numel(d2Split.areas));
  d2Split.d2Normalized{c} = cell(1, numel(d2Split.areas));
end

nEngagedWindows = 0;
nNonEngagedWindows = 0;
nTotalWindows = 0;
countedDurations = false;

for a = 1:numel(d2Split.areas)
  if a > numel(results.startS) || isempty(results.startS{a})
    continue;
  end
  % startS is window center relative to collectStart
  centerRel = results.startS{a}(:);
  winStartAbs = collectStart + centerRel - d2Window / 2;
  winEndAbs = collectStart + centerRel + d2Window / 2;
  engagedMask = window_overlaps_reach_buffer(winStartAbs, winEndAbs, reachStart, reachBuffer);
  nonEngagedMask = ~engagedMask;
  d2Split.windowMask{a} = struct('engaged', engagedMask, 'nonEngaged', nonEngagedMask);

  % Window grid is shared across areas; count session time once
  if ~countedDurations
    nTotalWindows = numel(engagedMask);
    nEngagedWindows = sum(engagedMask);
    nNonEngagedWindows = sum(nonEngagedMask);
    countedDurations = true;
  end

  d2Raw = results.d2{a}(:);
  d2Norm = [];
  if isfield(results, 'd2Normalized') && a <= numel(results.d2Normalized) ...
      && ~isempty(results.d2Normalized{a})
    d2Norm = results.d2Normalized{a}(:);
  end
  if useLog10D2
    d2Raw = log10_safe_numeric(d2Raw);
    if ~isempty(d2Norm)
      d2Norm = log10_safe_numeric(d2Norm);
    end
  end

  masks = {true(size(engagedMask)), engagedMask, nonEngagedMask};
  for c = 1:numel(classNames)
    m = masks{c};
    vals = d2Raw(m);
    d2Split.d2{c}{a} = vals(isfinite(vals));
    if ~isempty(d2Norm)
      valsN = d2Norm(m);
      d2Split.d2Normalized{c}{a} = valsN(isfinite(valsN));
    else
      d2Split.d2Normalized{c}{a} = [];
    end
  end

  fprintf('  %s: total=%d, engaged=%d, non-engaged=%d finite d2 windows\n', ...
    d2Split.areas{a}, numel(d2Split.d2{1}{a}), numel(d2Split.d2{2}{a}), numel(d2Split.d2{3}{a}));
end

d2Split.durations = struct( ...
  'totalSec', nTotalWindows * d2Window, ...
  'engagedSec', nEngagedWindows * d2Window, ...
  'nonEngagedSec', nNonEngagedWindows * d2Window, ...
  'nTotalWindows', nTotalWindows, ...
  'nEngagedWindows', nEngagedWindows, ...
  'nNonEngagedWindows', nNonEngagedWindows);
end

function fig = plot_engagement_d2_distributions(d2Split, sessionType, sessionName, ...
    d2Window, collectStart, collectEnd, useLog10D2, plotConfig)
% PLOT_ENGAGEMENT_D2_DISTRIBUTIONS - Overlapping PDFs for total / engaged / non-engaged

plotConfig = fill_default_engagement_plot_config(plotConfig);
classNames = d2Split.classNames;
classColors = engagement_class_colors();
numAreas = numel(d2Split.areas);
if numAreas == 0
  error('No areas with d2 data to plot.');
end

allVals = collect_class_metric_values(d2Split.d2);
[binEdges, xMin, xMax] = build_shared_bin_edges(allVals, 28);

fig = figure('Color', 'w', 'Position', [120 120 900 280 * numAreas], ...
  'Name', 'Engagement d2 distributions');
tileLayout = tiledlayout(numAreas, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

if useLog10D2
  xLabelText = 'log_{10}(d2)';
else
  xLabelText = 'd2';
end

for a = 1:numAreas
  ax = nexttile(tileLayout);
  plot_class_metric_histogram(ax, d2Split.d2, a, classNames, classColors, ...
    binEdges, xMin, xMax, useLog10D2, nan, plotConfig);
  apply_engagement_axes_style(ax, plotConfig, xLabelText, 'Probability density', ...
    d2Split.areas{a}, 'tex');
end

dur = d2Split.durations;
sgtitle(tileLayout, sprintf( ...
  ['d2 by engagement | %s | %s | %.0fs windows [%.0f-%.0f s]\n', ...
  'durations: total %.1f min, engaged %.1f min, non-engaged %.1f min'], ...
  sessionType, sessionName, d2Window, collectStart, collectEnd, ...
  dur.totalSec / 60, dur.engagedSec / 60, dur.nonEngagedSec / 60), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function corrResult = build_d2_accuracy_correlation(results, reachStart, reachClass, ...
    reachStartEngaged, collectStart, d2Window, useLog10D2, reachBuffer)
% BUILD_D2_ACCURACY_CORRELATION - d2 and reach accuracy per window by class
%
% Variables:
%   results           - Output of criticality_ar_analysis
%   reachStart        - All reach onsets in collect window (absolute s)
%   reachClass        - Reach outcome labels (2/4 = correct, 1/3 = error)
%   reachStartEngaged - Reach onsets used for engaged-window classification
%   collectStart      - Collect window start (s); startS is relative to this
%   d2Window          - Non-overlapping window length (s)
%   useLog10D2        - If true, use log10(d2) for correlation
%   reachBuffer       - Seconds around each reach that count as engaged (s)
%
% Goal:
%   For each d2 window in Total and Engaged classes, compute reach accuracy
%   as the fraction of correct reaches (class 2 or 4) among reaches with onsets
%   in the window. Return aligned per-window vectors and Pearson r per area.

reachStart = reachStart(:);
reachClass = reachClass(:);
isCorrectReach = ismember(reachClass, [2, 4]);

classNames = {'Total', 'Engaged'};
corrResult = struct();
corrResult.classNames = classNames;
corrResult.areas = {};
corrResult.byClass = struct('total', {{}}, 'engaged', {{}});

for a = 1:numel(results.areas)
  if a > numel(results.startS) || isempty(results.startS{a})
    continue;
  end

  centerRel = results.startS{a}(:);
  d2Vec = results.d2{a}(:);
  nWindows = min(numel(centerRel), numel(d2Vec));
  if nWindows == 0
    continue;
  end

  centerRel = centerRel(1:nWindows);
  d2Vec = d2Vec(1:nWindows);
  if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
  end

  winStartAbs = collectStart + centerRel - d2Window / 2;
  winEndAbs = collectStart + centerRel + d2Window / 2;
  engagedMask = window_overlaps_reach_buffer(winStartAbs, winEndAbs, reachStartEngaged, reachBuffer);
  accuracyVec = compute_window_reach_accuracy(reachStart, isCorrectReach, winStartAbs, winEndAbs);

  classMasks = {true(nWindows, 1), engagedMask};
  classFields = {'total', 'engaged'};
  for c = 1:numel(classNames)
    classMask = classMasks{c};
    validMask = classMask & isfinite(d2Vec) & isfinite(accuracyVec);
    areaResult = struct();
    areaResult.areaName = results.areas{a};
    areaResult.className = classNames{c};
    areaResult.d2 = d2Vec;
    areaResult.accuracy = accuracyVec;
    areaResult.classMask = classMask;
    areaResult.validMask = validMask;
    areaResult.nClassWindows = sum(classMask);
    areaResult.nValidWindows = sum(validMask);
    areaResult.rPearson = pearson_r(d2Vec(validMask), accuracyVec(validMask));
    corrResult.byClass.(classFields{c}){end + 1} = areaResult; %#ok<AGROW>
  end

  corrResult.areas{end + 1} = results.areas{a}; %#ok<AGROW>
end
end

function accuracyVec = compute_window_reach_accuracy(reachStart, isCorrectReach, winStartAbs, winEndAbs)
% COMPUTE_WINDOW_REACH_ACCURACY - Fraction correct among reaches in each window
%
% Variables:
%   reachStart, isCorrectReach - Reach onsets (s) and logical correct labels
%   winStartAbs, winEndAbs     - Window bounds [start, end) in absolute seconds
%
% Goal:
%   For each window, accuracy = mean(isCorrectReach) over reaches with onset
%   in [winStart, winEnd). Windows with no reaches return NaN.

nWindows = numel(winStartAbs);
accuracyVec = nan(nWindows, 1);
for w = 1:nWindows
  reachInWin = reachStart >= winStartAbs(w) & reachStart < winEndAbs(w);
  if ~any(reachInWin)
    continue;
  end
  accuracyVec(w) = mean(isCorrectReach(reachInWin));
end
end

function print_d2_accuracy_correlation(corrResult)
% PRINT_D2_ACCURACY_CORRELATION - Command-window summary per class and area

fprintf('Accuracy = fraction correct (reachClass 2 or 4) among reaches in each window\n');
classFields = {'total', 'engaged'};
if isempty(corrResult.areas)
  fprintf('  No areas with d2 window data.\n');
  return;
end
for c = 1:numel(corrResult.classNames)
  className = corrResult.classNames{c};
  byArea = corrResult.byClass.(classFields{c});
  fprintf('  %s:\n', className);
  for a = 1:numel(byArea)
    areaResult = byArea{a};
    if areaResult.nValidWindows == 0
      fprintf('    %s: no windows with d2 and >=1 reach\n', areaResult.areaName);
      continue;
    end
    fprintf('    %s: r=%.3f, n=%d windows with reaches (%d %s windows)\n', ...
      areaResult.areaName, areaResult.rPearson, areaResult.nValidWindows, ...
      areaResult.nClassWindows, lower(className));
  end
end
end

function fig = plot_d2_accuracy_correlation(corrResult, sessionName, d2Window, ...
    useLog10D2, plotConfig)
% PLOT_D2_ACCURACY_CORRELATION - Scatter of reach accuracy vs d2 by class
%
% Variables:
%   corrResult   - Output of build_d2_accuracy_correlation
%   sessionName  - Session identifier for title
%   d2Window     - Window length (s)
%   useLog10D2   - If true, y-axis uses log10(d2)
%   plotConfig   - Axis fonts and line widths
%
% Goal:
%   One row per area, one column per class (Total, Engaged) with scatter and r.

plotConfig = fill_default_engagement_plot_config(plotConfig);
classNames = corrResult.classNames;
classFields = {'total', 'engaged'};
classColors = engagement_class_colors();  % Total, Engaged, Non-engaged
numAreas = numel(corrResult.areas);
numClasses = numel(classNames);
if numAreas == 0
  warning('reach_criticality_metrics_engagement:NoD2AccuracyData', ...
    'No areas available for d2-accuracy correlation plot.');
  fig = gobjects(0);
  return;
end

if useLog10D2
  d2YLabel = 'log_{10}(d2)';
else
  d2YLabel = 'd2';
end
labelInterpreter = 'none';
if useLog10D2
  labelInterpreter = 'tex';
end

fig = figure('Color', 'w', 'Name', 'd2 vs reach accuracy', ...
  'Position', [120 120 420 * numClasses 340 * numAreas]);
tileLayout = tiledlayout(fig, numAreas, numClasses, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
  for c = 1:numClasses
    areaResult = corrResult.byClass.(classFields{c}){a};
    ax = nexttile(tileLayout);
    hold(ax, 'on');

    validMask = areaResult.validMask;
    if ~any(validMask)
      title(ax, sprintf('%s — %s (no windows with reaches)', ...
        areaResult.areaName, classNames{c}), 'Interpreter', 'none');
      apply_engagement_axes_style(ax, plotConfig);
      hold(ax, 'off');
      continue;
    end

    accPlot = areaResult.accuracy(validMask);
    d2Plot = areaResult.d2(validMask);
    scatter(ax, accPlot, d2Plot, 48, classColors(c, :), 'filled', ...
      'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');

    rVal = areaResult.rPearson;
    apply_engagement_axes_style(ax, plotConfig, 'Reach accuracy', d2YLabel, ...
      sprintf('%s | %s | r=%.3f, n=%d', areaResult.areaName, classNames{c}, ...
      rVal, sum(validMask)), labelInterpreter);
    xlim(ax, [0, 1]);
    grid(ax, 'on');
    hold(ax, 'off');
  end
end

sgtitle(tileLayout, sprintf('%s — d2 vs reach accuracy | %.0fs windows', ...
  sessionName, d2Window), 'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
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

%% -------------------------------------------------------------------------
%% Kurtosis (PRG) split and plot
%% -------------------------------------------------------------------------

function prgSplit = split_prg_by_reach_engagement(results, reachStart, prgWindow, reachBuffer)
% SPLIT_PRG_BY_REACH_ENGAGEMENT - Partition PRG windows by reach engagement
%
% Variables:
%   results     - Output of criticality_prg_analysis
%   reachStart  - Reach onsets (absolute s)
%   prgWindow   - Non-overlapping block length (s)
%   reachBuffer - Seconds around each reach that count as engaged (s)
%
% Goal:
%   Blocks overlapping any [reach - reachBuffer, reach + reachBuffer] are engaged
%   (including blocks with no reach onset but within the buffer). Duration =
%   nWindows * prgWindow per class (all blocks, including CV-excluded); metric
%   vectors use valid (non-excluded) windows only.

if nargin < 4 || isempty(reachBuffer)
  reachBuffer = 0;
end

classNames = engagement_class_names();
prgSplit = struct();
prgSplit.areas = normalize_results_area_list(results.areas);
prgSplit.classNames = classNames;
prgSplit.kappa = cell(1, numel(classNames));
prgSplit.djs = cell(1, numel(classNames));
for c = 1:numel(classNames)
  prgSplit.kappa{c} = cell(1, numel(prgSplit.areas));
  prgSplit.djs{c} = cell(1, numel(prgSplit.areas));
end

nEngagedWindows = 0;
nNonEngagedWindows = 0;
nTotalWindows = 0;
countedDurations = false;

for a = 1:numel(prgSplit.areas)
  if a > numel(results.windowStartS) || isempty(results.windowStartS{a})
    continue;
  end
  winStartAbs = results.windowStartS{a}(:);
  winEndAbs = winStartAbs + prgWindow;
  engagedMask = window_overlaps_reach_buffer(winStartAbs, winEndAbs, reachStart, reachBuffer);
  nonEngagedMask = ~engagedMask;

  % Block grid is shared across areas; count session time once
  if ~countedDurations
    nTotalWindows = numel(engagedMask);
    nEngagedWindows = sum(engagedMask);
    nNonEngagedWindows = sum(nonEngagedMask);
    countedDurations = true;
  end

  validMask = get_prg_valid_window_mask(results, a);
  kappaVec = results.kappa{a}(:);
  djsVec = [];
  if isfield(results, 'djs') && a <= numel(results.djs) && ~isempty(results.djs{a})
    djsVec = results.djs{a}(:);
  end

  masks = {true(size(engagedMask)), engagedMask, nonEngagedMask};
  for c = 1:numel(classNames)
    m = masks{c} & validMask;
    valsK = kappaVec(m);
    prgSplit.kappa{c}{a} = valsK(isfinite(valsK));
    if ~isempty(djsVec)
      valsD = djsVec(m);
      prgSplit.djs{c}{a} = valsD(isfinite(valsD));
    else
      prgSplit.djs{c}{a} = [];
    end
  end

  fprintf('  %s: total=%d, engaged=%d, non-engaged=%d valid kappa windows\n', ...
    prgSplit.areas{a}, numel(prgSplit.kappa{1}{a}), numel(prgSplit.kappa{2}{a}), ...
    numel(prgSplit.kappa{3}{a}));
end

prgSplit.durations = struct( ...
  'totalSec', nTotalWindows * prgWindow, ...
  'engagedSec', nEngagedWindows * prgWindow, ...
  'nonEngagedSec', nNonEngagedWindows * prgWindow, ...
  'nTotalWindows', nTotalWindows, ...
  'nEngagedWindows', nEngagedWindows, ...
  'nNonEngagedWindows', nNonEngagedWindows);
end

function fig = plot_engagement_kurtosis_distributions(prgSplit, sessionType, sessionName, ...
    prgWindow, collectStart, collectEnd, finalCutoffDivisor, prgMethod, plotConfig)
% PLOT_ENGAGEMENT_KURTOSIS_DISTRIBUTIONS - Kappa and D_JS PDFs by engagement class

plotConfig = fill_default_engagement_plot_config(plotConfig);
classNames = prgSplit.classNames;
classColors = engagement_class_colors();
numAreas = numel(prgSplit.areas);
if numAreas == 0
  error('No areas with PRG data to plot.');
end

allKappa = collect_class_metric_values(prgSplit.kappa);
allDjs = collect_class_metric_values(prgSplit.djs, false);
[kappaEdges, kappaMin, kappaMax] = build_shared_bin_edges(allKappa, 28);
hasDjs = ~isempty(allDjs);
if hasDjs
  [djsEdges, djsMin, djsMax] = build_shared_bin_edges(allDjs, 28, [0, 1]);
  nCols = 2;
else
  nCols = 1;
end

fig = figure('Color', 'w', 'Position', [140 140 470 * nCols 280 * numAreas], ...
  'Name', 'Engagement kurtosis distributions');
tileLayout = tiledlayout(numAreas, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
  axK = nexttile(tileLayout);
  plot_class_metric_histogram(axK, prgSplit.kappa, a, classNames, classColors, ...
    kappaEdges, kappaMin, kappaMax, false, 3, plotConfig);
  apply_engagement_axes_style(axK, plotConfig, ...
    sprintf('Kurtosis \\kappa (N/%d)', finalCutoffDivisor), 'Probability density', ...
    sprintf('%s — \\kappa', prgSplit.areas{a}), 'tex');

  if hasDjs
    axD = nexttile(tileLayout);
    plot_class_metric_histogram(axD, prgSplit.djs, a, classNames, classColors, ...
      djsEdges, djsMin, djsMax, false, nan, plotConfig);
    apply_engagement_axes_style(axD, plotConfig, 'D_{JS}', 'Probability density', ...
      sprintf('%s — D_{JS}', prgSplit.areas{a}), 'tex');
  end
end

dur = prgSplit.durations;
sgtitle(tileLayout, sprintf( ...
  ['PRG (%s) by engagement | %s | %s | %.0fs blocks [%.0f-%.0f s]\n', ...
  'durations: total %.1f min, engaged %.1f min, non-engaged %.1f min'], ...
  prgMethod, sessionType, sessionName, prgWindow, collectStart, collectEnd, ...
  dur.totalSec / 60, dur.engagedSec / 60, dur.nonEngagedSec / 60), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

%% -------------------------------------------------------------------------
%% Avalanche pool and plot
%% -------------------------------------------------------------------------

function avResult = run_pooled_avalanche_analysis(dataStruct, areasToAnalyze, avConfig, ...
    segments, computeShuffles)
% RUN_POOLED_AVALANCHE_ANALYSIS - Pool avalanches across segments, fit per area
%
% Variables:
%   segments        - Struct array with .start, .end (one or many)
%   computeShuffles - If true, also pool circular-shuffle avalanches (total only)

if nargin < 4 || isempty(segments)
  segments = struct('start', {}, 'end', {});
end
if nargin < 5 || isempty(computeShuffles)
  computeShuffles = false;
end

avResult = struct('areas', {{}}, 'byArea', {{}}, 'segments', segments);
for aIdx = 1:numel(areasToAnalyze)
  areaIndex = areasToAnalyze(aIdx);
  areaName = dataStruct.areas{areaIndex};
  avData = extract_pooled_area_avalanches(dataStruct, areaIndex, avConfig, segments, ...
    computeShuffles);
  avData.areaName = areaName;
  avResult.areas{end + 1} = areaName; %#ok<AGROW>
  avResult.byArea{end + 1} = avData; %#ok<AGROW>

  if avData.hasAvalanches
    fprintf('  AV %s: n=%d, tau=%.2f, alpha=%.2f, decades=%.2f, dcc=%.3f, (alpha-1)/(tau-1)=%.3f\n', ...
      areaName, avData.nAvalanches, avData.tau, avData.alpha, avData.decades, avData.dcc, avData.scalingRelation);
    if computeShuffles && ~isempty(avData.shuffleSizes)
      fprintf('  AV %s shuffle (total): n=%d over %d shuffles\n', ...
        areaName, numel(avData.shuffleSizes), avData.nShufflesCompleted);
    end
  else
    fprintf('  AV %s: no avalanches\n', areaName);
  end
end
end

function avData = extract_pooled_area_avalanches(dataStruct, areaIndex, analysisConfig, ...
    segments, computeShuffles)
% EXTRACT_POOLED_AREA_AVALANCHES - Collect avalanches across segments then fit
%
% Variables:
%   segments        - Struct array (.start, .end) in seconds
%   computeShuffles - If true, also pool circular-shuffle avalanches
%
% Goal:
%   Bin and detect avalanches per segment, pool sizes/durations, fit power laws
%   on the pooled sample (matches multi-segment engagement design).

if nargin < 5 || isempty(computeShuffles)
  computeShuffles = false;
end

avData = empty_avalanche_data();
if isempty(segments)
  return;
end

allSizes = [];
allDurations = [];
allShuffleSizes = [];
allShuffleDurations = [];
nShufflesCompleted = 0;
binSizeUsed = nan;
minSegDur = 0.2;
if isfield(analysisConfig, 'binSize') && isfinite(analysisConfig.binSize)
  minSegDur = max(minSegDur, analysisConfig.binSize * 4);
end
for i = 1:numel(segments)
  segStart = segments(i).start;
  segEnd = segments(i).end;
  if segEnd - segStart < minSegDur
    continue;
  end
  segAv = extract_area_avalanches(dataStruct, areaIndex, analysisConfig, segStart, segEnd, ...
    computeShuffles);
  if ~segAv.hasAvalanches
    continue;
  end
  allSizes = [allSizes; segAv.sizes(:)]; %#ok<AGROW>
  allDurations = [allDurations; segAv.durations(:)]; %#ok<AGROW>
  if computeShuffles && ~isempty(segAv.shuffleSizes)
    allShuffleSizes = [allShuffleSizes; segAv.shuffleSizes(:)]; %#ok<AGROW>
    allShuffleDurations = [allShuffleDurations; segAv.shuffleDurations(:)]; %#ok<AGROW>
    nShufflesCompleted = nShufflesCompleted + segAv.nShufflesCompleted;
  end
  if ~isfinite(binSizeUsed)
    binSizeUsed = segAv.binSize;
  end
end

if isempty(allSizes) || isempty(allDurations)
  return;
end

plMetrics = avalanche_power_law_metrics(allSizes, allDurations, analysisConfig);

avData.hasAvalanches = true;
avData.sizes = allSizes;
avData.durations = allDurations;
avData.tau = plMetrics.tau;
avData.alpha = plMetrics.alpha;
avData.paramSD = plMetrics.paramSD;
avData.decades = plMetrics.decades;
avData.dcc = distance_to_criticality(plMetrics.tau, plMetrics.alpha, plMetrics.paramSD);
avData.scalingRelation = compute_avalanche_scaling_relation(avData.tau, avData.alpha);
avData.minSizeFit = plMetrics.minavS;
avData.maxSizeFit = plMetrics.maxavS;
avData.minDurFit = plMetrics.minavD;
avData.maxDurFit = plMetrics.maxavD;
avData.sizeFitInfo = struct('exponent', plMetrics.tau, 'fitMin', plMetrics.minavS, ...
  'fitMax', plMetrics.maxavS, 'decades', plMetrics.decades);
avData.durFitInfo = struct('exponent', plMetrics.alpha, 'fitMin', plMetrics.minavD, ...
  'fitMax', plMetrics.maxavD);
avData.nAvalanches = numel(allSizes);
avData.binSize = binSizeUsed;
avData.nSegments = numel(segments);
avData.shuffleSizes = allShuffleSizes;
avData.shuffleDurations = allShuffleDurations;
avData.nShufflesCompleted = nShufflesCompleted;
end

function avData = extract_area_avalanches(dataStruct, areaIndex, analysisConfig, ...
    collectStart, collectEnd, computeShuffles)
% EXTRACT_AREA_AVALANCHES - Bin, threshold, and detect avalanches in one interval
%
% Variables:
%   computeShuffles - If true, also run circular permutations on this window

if nargin < 6 || isempty(computeShuffles)
  computeShuffles = false;
end

avData = empty_avalanche_data();
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

avData.hasAvalanches = true;
avData.sizes = sizes(:);
avData.durations = durations(:);
avData.nAvalanches = numel(sizes);

if computeShuffles
  nShufflesArea = 5;
  if isfield(analysisConfig, 'nShuffles') && ~isempty(analysisConfig.nShuffles)
    nShufflesArea = analysisConfig.nShuffles;
  end
  [avData.shuffleSizes, avData.shuffleDurations, avData.nShufflesCompleted] = ...
    pooled_circular_shuffle_avalanches(aDataMat, analysisConfig, nShufflesArea);
end
end

function avData = empty_avalanche_data()
avData = struct('hasAvalanches', false, 'sizes', [], 'durations', [], ...
  'tau', nan, 'alpha', nan, 'paramSD', nan, 'decades', nan, 'dcc', nan, ...
  'scalingRelation', nan, ...
  'minSizeFit', nan, 'maxSizeFit', nan, ...
  'minDurFit', nan, 'maxDurFit', nan, 'sizeFitInfo', struct(), ...
  'durFitInfo', struct(), 'nAvalanches', 0, 'binSize', nan, 'nSegments', 0, ...
  'shuffleSizes', [], 'shuffleDurations', [], 'nShufflesCompleted', 0);
end

function [shuffleSizes, shuffleDurations, nCompleted] = pooled_circular_shuffle_avalanches( ...
    aDataMat, analysisConfig, nShuffles)
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

function [sizes, durations, hasAvalanches] = compute_avalanche_sizes_durations_from_binned( ...
    aDataMat, analysisConfig)
% COMPUTE_AVALANCHE_SIZES_DURATIONS_FROM_BINNED - Avalanche vectors from binned matrix

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

function fig = plot_engagement_avalanche_distributions(avByClass, areaNames, sessionName, ...
    collectStart, collectEnd, avDurations, plotConfig)
% PLOT_ENGAGEMENT_AVALANCHE_DISTRIBUTIONS - Size/duration CCDFs for three classes
%
% Goal:
%   Overlay total / engaged / non-engaged CCDFs, plus circular-shuffle CCDF for
%   the full-session (total) data only.

plotConfig = fill_default_engagement_plot_config(plotConfig);
classNames = engagement_class_names();
classColors = engagement_class_colors();
classFields = {'total', 'engaged', 'nonEngaged'};
shuffleColor = [0.55, 0.55, 0.55];
nAreas = numel(areaNames);

fig = figure('Color', 'w', 'Position', [80 80 max(900, 380 * nAreas) 440], ...
  'Name', 'Engagement avalanche distributions');
tiledlayout(fig, nAreas, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for aIdx = 1:nAreas
  axSize = nexttile((aIdx - 1) * 2 + 1);
  axDur = nexttile((aIdx - 1) * 2 + 2);
  hold(axSize, 'on');
  hold(axDur, 'on');

  for c = 1:numel(classFields)
    avData = avByClass.(classFields{c}).byArea{aIdx};
    if ~avData.hasAvalanches
      continue;
    end
    displayName = sprintf('%s (\\tau=%.2f, n=%d)', classNames{c}, avData.tau, avData.nAvalanches);
    plot_empirical_ccdf(axSize, avData.sizes, classColors(c, :), displayName, plotConfig);

    binSize = resolve_avalanche_duration_bin_size(avData);
    displayNameDur = sprintf('%s (\\alpha=%.2f, n=%d)', classNames{c}, avData.alpha, avData.nAvalanches);
    plot_empirical_ccdf(axDur, avData.durations * binSize * 1000, classColors(c, :), ...
      displayNameDur, plotConfig);

    % Shuffle overlay for full-session (total) only
    if strcmp(classFields{c}, 'total') && isfield(avData, 'shuffleSizes') ...
        && ~isempty(avData.shuffleSizes)
      shuffleName = sprintf('Shuffle total (n=%d)', numel(avData.shuffleSizes));
      plot_empirical_ccdf(axSize, avData.shuffleSizes, shuffleColor, shuffleName, ...
        plotConfig, true);
      plot_empirical_ccdf(axDur, avData.shuffleDurations * binSize * 1000, shuffleColor, ...
        shuffleName, plotConfig, true);
    end
  end

  set(axSize, 'XScale', 'log', 'YScale', 'log');
  set(axDur, 'XScale', 'log', 'YScale', 'log');
  apply_engagement_axes_style(axSize, plotConfig, 'Avalanche size', 'P(X \geq x)', ...
    sprintf('%s — size', areaNames{aIdx}), 'tex');
  apply_engagement_axes_style(axDur, plotConfig, 'Avalanche duration (ms)', 'P(X \geq x)', ...
    sprintf('%s — duration', areaNames{aIdx}), 'tex');
  grid(axSize, 'on');
  grid(axDur, 'on');
  legend(axSize, 'Location', 'southwest', 'Interpreter', 'tex', ...
    'FontSize', plotConfig.legendFontSize);
  legend(axDur, 'Location', 'southwest', 'Interpreter', 'tex', ...
    'FontSize', plotConfig.legendFontSize);
  hold(axSize, 'off');
  hold(axDur, 'off');
end

sgtitle(fig, sprintf( ...
  ['%s — avalanche CCDFs by engagement [%.0f–%.0f s]\n', ...
  'durations: total %.1f min, engaged %.1f min, non-engaged %.1f min'], ...
  sessionName, collectStart, collectEnd, ...
  avDurations.totalSec / 60, avDurations.engagedSec / 60, avDurations.nonEngagedSec / 60), ...
  'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
end

function plot_empirical_ccdf(ax, values, lineColor, displayName, plotConfig, isShuffle)
% PLOT_EMPIRICAL_CCDF - Log-log CCDF markers for one engagement class or shuffle
%
% Variables:
%   isShuffle - If true, use open gray markers (manuscript shuffle style)

if nargin < 6 || isempty(isShuffle)
  isShuffle = false;
end
plotConfig = fill_default_engagement_plot_config(plotConfig);
values = values(values > 0 & isfinite(values));
if isempty(values)
  return;
end
uniqueVals = unique(values);
cdfY = arrayfun(@(x) mean(values >= x), uniqueVals);
if isShuffle
  plot(ax, uniqueVals, cdfY, 'o', 'Color', lineColor, ...
    'MarkerFaceColor', [0.75, 0.75, 0.75], 'MarkerSize', plotConfig.shuffleMarkerSize, ...
    'LineWidth', 1, 'DisplayName', displayName);
else
  plot(ax, uniqueVals, cdfY, 'o-', 'Color', lineColor, ...
    'MarkerFaceColor', lineColor, 'MarkerSize', plotConfig.markerSize, ...
    'LineWidth', plotConfig.lineWidth, 'DisplayName', displayName);
end
end

function binSize = resolve_avalanche_duration_bin_size(avData)
if isfield(avData, 'binSize') && isscalar(avData.binSize) && isfinite(avData.binSize) && avData.binSize > 0
  binSize = avData.binSize;
else
  binSize = 0.05;
end
end

function scalingRelation = compute_avalanche_scaling_relation(tau, alpha)
% COMPUTE_AVALANCHE_SCALING_RELATION - (alpha-1)/(tau-1)

scalingRelation = nan;
if isfinite(tau) && isfinite(alpha) && tau > 1
  scalingRelation = (alpha - 1) / (tau - 1);
end
end

%% -------------------------------------------------------------------------
%% Shared plotting / classification helpers
%% -------------------------------------------------------------------------

function names = engagement_class_names()
names = {'Total', 'Engaged', 'Non-engaged'};
end

function colors = engagement_class_colors()
% ENGAGEMENT_CLASS_COLORS - Total (gray), engaged (blue), non-engaged (orange)

colors = [0.45, 0.45, 0.45; 0.15, 0.45, 0.75; 0.85, 0.35, 0.15];
end

function isEngaged = window_overlaps_reach_buffer(winStartAbs, winEndAbs, reachStart, reachBuffer)
% WINDOW_OVERLAPS_REACH_BUFFER - True if window overlaps any reach buffer interval
%
% Variables:
%   winStartAbs, winEndAbs - Window bounds [start, end) in absolute seconds
%   reachStart             - Reach onset times (s)
%   reachBuffer            - Half-width around each reach treated as engaged (s)
%
% Goal:
%   Engaged if the window overlaps [reach - reachBuffer, reach + reachBuffer] for
%   any reach, so near-reach windows without an onset inside still count.

if nargin < 4 || isempty(reachBuffer)
  reachBuffer = 0;
end
reachBuffer = max(0, reachBuffer);

isEngaged = false(size(winStartAbs));
if isempty(reachStart)
  return;
end
reachStart = reachStart(:);
for w = 1:numel(winStartAbs)
  % Half-open window [winStart, winEnd) vs closed buffer [reach-buf, reach+buf]
  isEngaged(w) = any(winStartAbs(w) <= reachStart + reachBuffer & ...
    winEndAbs(w) > reachStart - reachBuffer);
end
end

function plot_class_metric_histogram(ax, classMetric, areaIdx, classNames, classColors, ...
    binEdges, xMin, xMax, drawZeroRef, refValue, plotConfig)
% PLOT_CLASS_METRIC_HISTOGRAM - Overlapping PDFs for total / engaged / non-engaged

plotConfig = fill_default_engagement_plot_config(plotConfig);
hold(ax, 'on');
for c = 1:numel(classNames)
  vals = classMetric{c}{areaIdx};
  if isempty(vals)
    continue;
  end
  histogram(ax, vals, binEdges, 'Normalization', 'pdf', ...
    'FaceColor', classColors(c, :), 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
    'DisplayName', sprintf('%s (n=%d)', classNames{c}, numel(vals)));
end
if drawZeroRef
  xline(ax, 0, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', plotConfig.lineWidth, ...
    'HandleVisibility', 'off');
end
if isfinite(refValue)
  xline(ax, refValue, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', plotConfig.lineWidth, ...
    'HandleVisibility', 'off');
end
xlim(ax, [xMin, xMax]);
grid(ax, 'on');
legend(ax, 'Location', 'northeast', 'FontSize', plotConfig.legendFontSize);
hold(ax, 'off');
end

function allVals = collect_class_metric_values(classMetric, requireData)
% COLLECT_CLASS_METRIC_VALUES - Pool finite values across classes and areas

if nargin < 2
  requireData = true;
end
allVals = [];
for c = 1:numel(classMetric)
  for a = 1:numel(classMetric{c})
    allVals = [allVals; classMetric{c}{a}(:)]; %#ok<AGROW>
  end
end
allVals = allVals(isfinite(allVals));
if requireData && isempty(allVals)
  error('No finite values available for histogram binning.');
end
end

function [binEdges, xMin, xMax] = build_shared_bin_edges(allVals, nBinsTarget, forcedLimits)
% BUILD_SHARED_BIN_EDGES - Shared histogram bin edges and x-limits

if nargin < 3
  forcedLimits = [];
end
if isempty(allVals)
  if ~isempty(forcedLimits)
    xMin = forcedLimits(1);
    xMax = forcedLimits(2);
  else
    xMin = 0;
    xMax = 1;
  end
else
  xMin = min(allVals);
  xMax = max(allVals);
  if ~isempty(forcedLimits)
    xMin = min(xMin, forcedLimits(1));
    xMax = max(xMax, forcedLimits(2));
  end
end

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
validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function validMask = get_prg_valid_window_mask(results, areaIdx)
% GET_PRG_VALID_WINDOW_MASK - Finite, non-CV-excluded windows

validMask = [];
if areaIdx > numel(results.kappa) || isempty(results.kappa{areaIdx})
  return;
end
validMask = isfinite(results.kappa{areaIdx}(:));
if isfield(results, 'windowExcluded') && areaIdx <= numel(results.windowExcluded) ...
    && ~isempty(results.windowExcluded{areaIdx})
  excluded = results.windowExcluded{areaIdx}(:);
  if numel(excluded) == numel(validMask)
    validMask = validMask & ~excluded;
  end
end
end

function areaNames = normalize_results_area_list(areasField)
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
        areaNames{end + 1} = char(string(entry{j})); %#ok<AGROW>
      end
    end
  elseif ischar(entry) || isstring(entry)
    areaNames{end + 1} = char(string(entry)); %#ok<AGROW>
  end
end
end

function results = filter_ar_results_to_brain_area(results, brainArea)
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

function results = filter_prg_results_to_brain_area(results, brainArea)
if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaNames = normalize_results_area_list(results.areas);
areaIdx = find(strcmp(areaNames, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end
cellFields = {'kappa', 'kappaByCutoff', 'windowStartS', 'popCv', 'windowExcluded', ...
  'nNeuronsPerWindow', 'kappaSurrogate', 'djs', 'djsSurrogate', 'nCutoffList'};
results.areas = {areaNames{areaIdx}};
for f = 1:length(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
end

function nUnits = count_session_neurons_for_brain_area(dataStruct, brainArea)
% COUNT_SESSION_NEURONS_FOR_BRAIN_AREA - Neurons in selected analysis area(s)

nUnits = 0;
if ~isfield(dataStruct, 'idMatIdx') || isempty(dataStruct.idMatIdx)
  return;
end

if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
  for areaIdx = dataStruct.areasToTest(:)'
    nUnits = nUnits + numel(dataStruct.idMatIdx{areaIdx});
  end
  return;
end

if isempty(brainArea)
  for areaIdx = 1:numel(dataStruct.idMatIdx)
    nUnits = nUnits + numel(dataStruct.idMatIdx{areaIdx});
  end
  return;
end

areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  return;
end
nUnits = numel(dataStruct.idMatIdx{areaIdx});
end

function areasToAnalyze = resolve_areas_to_analyze(dataStruct, brainArea, nMinNeurons)
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
    areasToAnalyze(end + 1) = areaIndex; %#ok<AGROW>
  end
end
end

function label = format_areas_label(areaNames)
if isempty(areaNames)
  label = '';
  return;
end
if iscell(areaNames)
  areaNames = areaNames(:)';
  label = strjoin(areaNames, '_');
else
  label = char(areaNames);
end
label = matlab.lang.makeValidName(label);
end
