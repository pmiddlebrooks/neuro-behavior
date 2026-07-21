function out = interval_criticality_metrics_engagement(subjectName, sessionName, opts)
% INTERVAL_CRITICALITY_METRICS_ENGAGEMENT - Criticality metrics by interval-task engagement
%
% Variables:
%   subjectName - Subject folder (e.g. 'ey9166')
%   sessionName - Session identifier (e.g. 'ey9166_2026_04_03')
%   opts        - Options struct (session loading + analysis selection). Fields:
%     Session loading (usual neuro_behavior_options overrides):
%       .collectStart, .collectEnd, .minFiringRate, .maxFiringRate,
%       .firingRateCheckTime
%     Analysis selection:
%       .analyses - Cell of {'d2','kurtosis','avalanches'} (any subset; default all)
%     Engagement:
%       .eventBuffer         - Seconds around each correct/error beam break treated
%                              as engaged (default 1). d2/kurtosis: windows overlapping
%                              [event +/- eventBuffer] count as engaged. Avalanches:
%                              buffer excluded from non-engaged gaps.
%       .minNonEngagedWindow - Min gap without beam-break events (s) for non-engaged
%                              avalanche segments
%       .absorbSingleEvents  - If true (default), isolated single beam-break events
%                              flanked by qualifying non-engaged gaps are merged into
%                              non-engaged time (avalanche segments and d2/kurtosis windows).
%                              Alias: absorbSingleReaches
%       .minLeaveSec         - Min confirmed leave duration for trial parsing (default 0.1)
%     Shared analysis:
%       .brainArea, .brainAreaCombinations, .dataSource, .saveFigure, .outputDir
%       .plotConfig - axisLabelFontSize, tickLabelFontSize, axesLineWidth, ...
%       .useSubsampling, .nSubsamples, .nNeuronsSubsample, .minNeuronsMultiple
%     d2:
%       .d2Window, .useLog10D2, .nShufflesD2
%       .sessionInterval - Target interval duration (s; default 5)
%       .rewardAttemptBeforeSec - Exclude error trials with poke time before
%                                 sessionInterval - this value (default 1)
%       .runD2TrialRateCorrelation - If true (default), correlate d2 with trial
%                                    count/rate across all d2 windows
%     Kurtosis (PRG):
%       .prgWindow, .prgMethod, .surrogateMethod, .nSurrogates
%     Avalanches:
%       .powerLawFitMethod, .avalancheDetectionMode, .runClausetPlpva, .gofThreshold
%       .enableCircularPermutations - If true, overlay shuffle CCDF for total only
%       .nShuffles - Number of circular permutations for total shuffle (default 5)
%
% Goal:
%   Same analyses as reach_criticality_metrics_engagement, but engagement events
%   are correct and error beam breaks (REWARD / ERROR outcomes) from the interval
%   task log rather than reaches. Isolated single events may be absorbed; see
%   absorbSingleEvents.
%
% Returns:
%   With no inputs: default options struct (same fields as opts above).
%   Otherwise: struct with durations, segments, results, figHandles, config

setup_interval_criticality_metrics_engagement_paths();

if nargin == 0
  out = fill_engagement_opts_defaults(struct());
  return;
end
if nargin < 2 || isempty(subjectName) || isempty(sessionName)
  error('interval_criticality_metrics_engagement:MissingSession', ...
    'subjectName and sessionName are required.');
end
if nargin < 3 || isempty(opts)
  opts = struct();
end
opts = fill_engagement_opts_defaults(opts);

sessionType = 'interval';
dataSource = opts.dataSource;
collectStart = opts.collectStart;
collectEnd = opts.collectEnd;

fprintf('\n=== Interval criticality metrics by engagement ===\n');
fprintf('Session: %s / %s\n', subjectName, sessionName);
fprintf('Analyses: %s\n', strjoin(opts.analyses, ', '));

%% Load session
loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = opts.firingRateCheckTime;
loadOpts.collectStart = collectStart;
loadOpts.collectEnd = collectEnd;
loadOpts.minFiringRate = opts.minFiringRate;
loadOpts.maxFiringRate = opts.maxFiringRate;

loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

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

% Full-session / oversized d2Window → one window over the loaded collect range
sessionDuration = collectEnd - collectStart;
if isempty(opts.d2Window) || opts.d2Window > (sessionDuration + 1)
  if isempty(opts.d2Window)
    prevD2WindowStr = '[]';
  else
    prevD2WindowStr = sprintf('%.1f', opts.d2Window);
  end
  fprintf('  d2Window clamped to session duration %.1f s (was %s).\n', ...
    sessionDuration, prevD2WindowStr);
  opts.d2Window = sessionDuration;
end

[dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
  dataStruct, opts.brainArea, opts.brainAreaCombinations, true);
if ~areaOk
  error('Brain area "%s" not available in this session.', opts.brainArea);
end

paths = get_paths();
[eventTimes, eventTypes, trials] = load_interval_beam_break_events( ...
  paths, subjectName, sessionName, opts.minLeaveSec);
eventInCollect = eventTimes >= collectStart & eventTimes <= collectEnd;
eventTimes = eventTimes(eventInCollect);
eventTypes = eventTypes(eventInCollect);
nCorrect = sum(eventTypes == "correct");
nError = sum(eventTypes == "error");

fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
  collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('Beam-break events in collect window: %d (%d correct, %d error)\n', ...
  numel(eventTimes), nCorrect, nError);
if isempty(eventTimes)
  error('interval_criticality_metrics_engagement:NoEvents', ...
    'No correct/error beam-break events found in collect window.');
end
eventTimesEngaged = filter_engaged_event_times(eventTimes, collectStart, collectEnd, ...
  opts.minNonEngagedWindow, opts.eventBuffer, opts.absorbSingleEvents);
if opts.absorbSingleEvents && numel(eventTimesEngaged) < numel(eventTimes)
  fprintf('absorbSingleEvents: %d isolated event(s) merged into non-engaged time\n', ...
    numel(eventTimes) - numel(eventTimesEngaged));
end

[eventTimesForRate, nExcludedEarlyErrors] = filter_interval_events_for_trial_rate( ...
  trials, opts.sessionInterval, opts.rewardAttemptBeforeSec);
eventInCollectRate = eventTimesForRate >= collectStart & eventTimesForRate <= collectEnd;
eventTimesForRate = eventTimesForRate(eventInCollectRate);
fprintf(['Trial-rate analysis: %d beam-break events in collect window ', ...
  '(%d early errors excluded; poke < %.1f s)\n'], ...
  numel(eventTimesForRate), nExcludedEarlyErrors, ...
  opts.sessionInterval - opts.rewardAttemptBeforeSec);

[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();

% Continuous engagement segments (avalanche definition; also used for overview plot)
[engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
  collectStart, collectEnd, eventTimes, opts.minNonEngagedWindow, opts.eventBuffer, ...
  opts.absorbSingleEvents);
engagedSegs = label_segments(engagedSegs, 'Engaged');
nonEngagedSegs = label_segments(nonEngagedSegs, 'Non-engaged');

out = struct();
out.subjectName = subjectName;
out.sessionName = sessionName;
out.config = opts;
out.eventTimes = eventTimes;
out.eventTimesEngaged = eventTimesEngaged;
out.eventTimesForRate = eventTimesForRate;
out.nExcludedEarlyErrors = nExcludedEarlyErrors;
out.eventTypes = eventTypes;
out.trials = trials;
out.segments = struct('engaged', engagedSegs, 'nonEngaged', nonEngagedSegs);
out.durations = struct();
out.d2 = [];
out.d2TrialRateCorrelation = [];
out.kurtosis = [];
out.avalanches = [];
out.figHandles = struct();

plotConfig = opts.plotConfig;

if opts.makePlots
  out.figHandles.segments = plot_beam_break_engagement_segments( ...
    eventTimes, eventTypes, engagedSegs, nonEngagedSegs, sessionName, ...
    collectStart, collectEnd, opts.minNonEngagedWindow, opts.eventBuffer, plotConfig);
end

%% d2: classify non-overlapping windows by beam-break buffer overlap
if ismember('d2', opts.analyses)
  fprintf('\n--- d2 ---\n');
  fprintf('eventBuffer: %.1f s (windows overlapping event +/- buffer = engaged)\n', ...
    opts.eventBuffer);
  arConfig = build_ar_config(opts);
  resultsD2 = criticality_ar_analysis(dataStruct, arConfig);
  if ~isempty(opts.brainArea)
    resultsD2 = filter_ar_results_to_brain_area(resultsD2, opts.brainArea);
  end
  if isempty(resultsD2.areas)
    error('No d2 results for brain area "%s".', opts.brainArea);
  end

  d2Split = split_d2_by_reach_engagement(resultsD2, eventTimesEngaged, collectStart, ...
    opts.d2Window, opts.useLog10D2, opts.eventBuffer);
  out.d2 = d2Split;
  out.arResultsD2 = resultsD2;
  out.durations.d2 = d2Split.durations;
  print_engagement_durations('d2', d2Split.durations);

  if opts.makePlots
    out.figHandles.d2 = plot_engagement_d2_distributions( ...
      d2Split, sessionType, sessionName, opts.d2Window, collectStart, collectEnd, ...
      opts.useLog10D2, plotConfig);
  end

  if opts.runD2TrialRateCorrelation
    fprintf('\n--- d2 vs trial rate (all windows) ---\n');
    d2RateCorr = build_d2_trial_rate_correlation(resultsD2, eventTimesForRate, ...
      collectStart, opts.d2Window, opts.useLog10D2);
    out.d2TrialRateCorrelation = d2RateCorr;
    print_d2_trial_rate_correlation(d2RateCorr, opts.d2Window, opts.sessionInterval, ...
      opts.rewardAttemptBeforeSec);
    if opts.makePlots
      out.figHandles.d2TrialRateCorrelation = plot_d2_trial_rate_correlation( ...
        d2RateCorr, sessionName, opts.d2Window, opts.useLog10D2, plotConfig);
    end
  end
end

%% Kurtosis (PRG): classify non-overlapping blocks by beam-break buffer overlap
if ismember('kurtosis', opts.analyses)
  fprintf('\n--- Kurtosis (PRG) ---\n');
  fprintf('eventBuffer: %.1f s (blocks overlapping event +/- buffer = engaged)\n', ...
    opts.eventBuffer);
  prgConfig = build_prg_config(opts);
  resultsPrg = criticality_prg_analysis(dataStruct, prgConfig);
  if ~isempty(opts.brainArea)
    resultsPrg = filter_prg_results_to_brain_area(resultsPrg, opts.brainArea);
  end
  if isempty(resultsPrg.areas)
    error('No PRG results for brain area "%s".', opts.brainArea);
  end

  prgSplit = split_prg_by_reach_engagement(resultsPrg, eventTimesEngaged, opts.prgWindow, ...
    opts.eventBuffer);
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
  fprintf('minNonEngagedWindow: %.1f s; eventBuffer: %.1f s\n', ...
    opts.minNonEngagedWindow, opts.eventBuffer);

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
    saveDir = fullfile(paths.dropPath, 'interval_timing_task', 'results', ...
      subjectName, matlab.lang.makeValidName(sessionName));
  end
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  areaTag = format_areas_label(opts.brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  baseName = sprintf('interval_criticality_metrics_engagement_%s_%s_%.0f-%.0fs', ...
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
  error('interval_criticality_metrics_engagement:BadAnalyses', ...
    'Unknown analyses: %s. Use d2, kurtosis, and/or avalanches.', strjoin(unknown, ', '));
end

if ~isfield(opts, 'minNonEngagedWindow') || isempty(opts.minNonEngagedWindow)
  opts.minNonEngagedWindow = 30;
end
if ~isfield(opts, 'eventBuffer') || isempty(opts.eventBuffer)
  if isfield(opts, 'reachBuffer') && ~isempty(opts.reachBuffer)
    opts.eventBuffer = opts.reachBuffer;
  else
    opts.eventBuffer = 1;
  end
end
if ~isfield(opts, 'absorbSingleEvents') || isempty(opts.absorbSingleEvents)
  if isfield(opts, 'absorbSingleReaches') && ~isempty(opts.absorbSingleReaches)
    opts.absorbSingleEvents = opts.absorbSingleReaches;
  else
    opts.absorbSingleEvents = true;
  end
end
if ~isfield(opts, 'minLeaveSec') || isempty(opts.minLeaveSec)
  opts.minLeaveSec = 0.1;
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

if ~isfield(opts, 'useSubsampling') || isempty(opts.useSubsampling)
  opts.useSubsampling = false;
end
if ~isfield(opts, 'nSubsamples') || isempty(opts.nSubsamples)
  opts.nSubsamples = 20;
end
if ~isfield(opts, 'nNeuronsSubsample') || isempty(opts.nNeuronsSubsample)
  opts.nNeuronsSubsample = 20;
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
if ~isfield(opts, 'sessionInterval') || isempty(opts.sessionInterval)
  opts.sessionInterval = 5;
end
if ~isfield(opts, 'rewardAttemptBeforeSec') || isempty(opts.rewardAttemptBeforeSec)
  opts.rewardAttemptBeforeSec = 1;
end
if ~isfield(opts, 'runD2TrialRateCorrelation') || isempty(opts.runD2TrialRateCorrelation)
  opts.runD2TrialRateCorrelation = true;
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

plotConfig = fill_manuscript_plot_config(plotConfig);
end

function apply_engagement_axes_style(ax, plotConfig, xLabelText, yLabelText, titleText, textInterpreter)
% APPLY_ENGAGEMENT_AXES_STYLE - Thicker axes and larger fonts (manuscript style)

apply_manuscript_axes_style(ax, plotConfig, xLabelText, yLabelText, titleText, textInterpreter);
end

function setup_interval_criticality_metrics_engagement_paths()
% SETUP_INTERVAL_CRITICALITY_METRICS_ENGAGEMENT_PATHS - Add neuro-behavior paths

scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  scriptDir = fileparts(which('interval_criticality_metrics_engagement'));
end
srcPath = fullfile(scriptDir, '..');
addpath(srcPath);
addpath(fullfile(srcPath, 'reach_task'));
addpath(fullfile(srcPath, 'schall'));
addpath(fullfile(srcPath, 'spontaneous'));
addpath(fullfile(srcPath, 'interval_timing_task'));
addpath(fullfile(srcPath, 'criticality', 'scripts'));
addpath(fullfile(srcPath, 'criticality', 'analyses'));
addpath(fullfile(srcPath, 'criticality_manuscript'));
addpath(fullfile(srcPath, 'session_prep', 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'sliding_window_prep', 'utils'));
addpath(fullfile(srcPath, 'criticality'));
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
arConfig.nMinNeurons = 10;
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
prgConfig.nMinNeurons = 20;
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
avConfig.nMinNeurons = 20;
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
%% Interval beam-break events (correct / error)
%% -------------------------------------------------------------------------

function [eventTimes, eventTypes, trials] = load_interval_beam_break_events( ...
    paths, subjectName, sessionName, minLeaveSec)
% LOAD_INTERVAL_BEAM_BREAK_EVENTS - Correct and error outcome times from session log
%
% Variables:
%   paths, subjectName, sessionName - Session location under intervalDataPath
%   minLeaveSec                     - Min confirmed leave duration (s)
%
% Goal:
%   Parse revised_interval_*.csv and return session-relative outcome times for
%   correct (REWARD) and error (ERROR) beam breaks, matching interval_session_performance.

sessionDir = fullfile(paths.intervalDataPath, subjectName, sessionName);
if ~exist(sessionDir, 'dir')
  error('interval_criticality_metrics_engagement:SessionNotFound', ...
    'Session directory not found: %s', sessionDir);
end

csvPath = find_interval_csv(sessionDir);
fprintf('Loading interval log: %s\n', csvPath);
logTable = parse_interval_log(csvPath);
trials = extract_interval_trials(logTable, minLeaveSec);

eventTimes = trials.outcomeTimeSec(:);
eventTypes = trials.type(:);
[eventTimes, ord] = sort(eventTimes);
eventTypes = eventTypes(ord);
end

function csvPath = find_interval_csv(sessionDir)
% FIND_INTERVAL_CSV - Most recent revised_interval_*.csv in session folder

csvFiles = dir(fullfile(sessionDir, 'revised_interval_*.csv'));
if isempty(csvFiles)
  error('interval_criticality_metrics_engagement:NoCsv', ...
    'No revised_interval_*.csv found in %s', sessionDir);
end
[~, newestIdx] = max([csvFiles.datenum]);
csvPath = fullfile(sessionDir, csvFiles(newestIdx).name);
end

function logTable = parse_interval_log(csvPath)
% PARSE_INTERVAL_LOG - Read Arduino/Processing interval task CSV

rawTable = readtable(csvPath, 'TextType', 'string');
varNames = lower(string(rawTable.Properties.VariableNames));

timeCol = find(contains(varNames, 'timestamp'), 1);
eventCol = find(strcmp(varNames, 'event') | contains(varNames, 'event'), 1);
valueCol = find(strcmp(varNames, 'value') | contains(varNames, 'value'), 1);

if isempty(timeCol) || isempty(eventCol) || isempty(valueCol)
  error('interval_criticality_metrics_engagement:BadCsv', ...
    'CSV must contain timestamp, event, and value columns: %s', csvPath);
end

timestampMs = rawTable{:, timeCol};
if iscell(timestampMs)
  timestampMs = cellfun(@str2double, timestampMs);
elseif isstring(timestampMs) || ischar(timestampMs)
  timestampMs = str2double(string(timestampMs));
end
timestampMs = double(timestampMs);

eventNames = string(rawTable{:, eventCol});
eventValues = rawTable{:, valueCol};
if iscell(eventValues) || isstring(eventValues)
  eventValues = str2double(string(eventValues));
end
eventValues = double(eventValues);

validRows = ~isnan(timestampMs) & eventNames ~= "" & ~ismissing(eventNames);
logTable = table(timestampMs(validRows), eventNames(validRows), eventValues(validRows), ...
  'VariableNames', {'timestampMs', 'event', 'value'});
logTable = sortrows(logTable, 'timestampMs');
end

function trials = extract_interval_trials(logTable, minLeaveSec)
% EXTRACT_INTERVAL_TRIALS - Segment ERROR/REWARD trials from event log
%
% Goal: For each ERROR or post-training REWARD, record poke time since last leave
% and session-relative outcome time (same logic as interval_session_performance).

minLeaveMs = minLeaveSec * 1000;

leavePending = false;
leaveConfirmStartMs = NaN;
initialExitMs = NaN;
leaveTimeMs = NaN;
timerArmed = false;
beamState = 0;
firstRewardSeen = false;

sessionOriginMs = min(logTable.timestampMs);
trialTypes = strings(0, 1);
pokeTimesSec = [];
outcomeTimesSec = [];

eventNames = logTable.event;
timestampsMs = logTable.timestampMs;
eventValues = logTable.value;
nEvents = height(logTable);

for eventIdx = 1:nEvents
  eventTimeMs = timestampsMs(eventIdx);
  eventName = eventNames(eventIdx);
  eventValue = eventValues(eventIdx);

  if leavePending && beamState == 0 && (eventTimeMs - leaveConfirmStartMs) >= minLeaveMs
    leaveTimeMs = initialExitMs;
    timerArmed = true;
    leavePending = false;
  end

  if eventName == "B"
    beamState = eventValue;
    if eventValue == 0
      leavePending = true;
      initialExitMs = eventTimeMs;
      leaveConfirmStartMs = eventTimeMs;
    else
      leavePending = false;
    end
  elseif eventName == "ERROR"
    if timerArmed && ~isnan(leaveTimeMs)
      trialTypes(end + 1, 1) = "error"; %#ok<AGROW>
      pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
      outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
    end
    timerArmed = false;
    leaveTimeMs = NaN;
  elseif eventName == "REWARD"
    if ~firstRewardSeen
      firstRewardSeen = true;
    elseif timerArmed && ~isnan(leaveTimeMs)
      trialTypes(end + 1, 1) = "correct"; %#ok<AGROW>
      pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
      outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
    end
    timerArmed = false;
    leaveTimeMs = NaN;
  end
end

trials = table(trialTypes, pokeTimesSec, outcomeTimesSec, ...
  'VariableNames', {'type', 'pokeTimeSec', 'outcomeTimeSec'});
end

%% -------------------------------------------------------------------------
%% Engagement segment definitions (avalanches)
%% -------------------------------------------------------------------------

function [engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
    collectStart, collectEnd, eventTimes, minNonEngagedWindow, eventBuffer, absorbSingleEvents)
% DEFINE_REACH_ENGAGEMENT_SEGMENTS - Continuous engaged / non-engaged intervals
%
% Variables:
%   collectStart, collectEnd - Session analysis bounds (s)
%   eventTimes               - Beam-break event times in collect window (s)
%   minNonEngagedWindow      - Minimum gap without events (s)
%   eventBuffer              - Seconds around each event excluded from non-engaged
%   absorbSingleEvents       - If true, merge isolated single events into flanking gaps
%
% Goal:
%   Each event occupies [event - eventBuffer, event + eventBuffer]. Non-engaged =
%   gaps between these buffered neighborhoods (and session edges) with duration
%   >= minNonEngagedWindow. Engaged = complement (includes all event buffers).
%   When absorbSingleEvents is true, an occupied interval with exactly one event
%   that is flanked on both sides by qualifying non-engaged gaps is absorbed: the
%   event buffer and both gaps form one non-engaged segment.

if nargin < 5 || isempty(eventBuffer)
  eventBuffer = 0;
end
if nargin < 6 || isempty(absorbSingleEvents)
  absorbSingleEvents = true;
end
eventBuffer = max(0, eventBuffer);

eventTimes = sort(eventTimes(:));
eventTimes = eventTimes(eventTimes >= collectStart & eventTimes <= collectEnd);

% Merge overlapping event neighborhoods into occupied intervals
occupied = merge_event_buffer_intervals(eventTimes, eventBuffer, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if absorbSingleEvents && ~isempty(occupied)
  absorbedMask = get_absorbed_single_event_occupied_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, eventBuffer);
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

function absorbedMask = get_absorbed_single_event_occupied_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, eventBuffer)
% GET_ABSORBED_SINGLE_EVENT_OCCUPIED_MASK - Isolated single events to absorb
%
% Variables:
%   eventTimes, collectStart, collectEnd, minNonEngagedWindow, eventBuffer
%
% Goal:
%   Mark merged event-buffer intervals that contain exactly one event and are
%   flanked on both sides by gaps >= minNonEngagedWindow.

occupied = merge_event_buffer_intervals(eventTimes, eventBuffer, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if isempty(occupied)
  return;
end

for iOcc = 1:numel(occupied)
  nEventsInOcc = sum(eventTimes >= occupied(iOcc).start & eventTimes <= occupied(iOcc).end);
  if nEventsInOcc ~= 1
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

function eventTimesEngaged = filter_engaged_event_times(eventTimes, collectStart, ...
    collectEnd, minNonEngagedWindow, eventBuffer, absorbSingleEvents)
% FILTER_ENGAGED_EVENT_TIMES - Beam-break events that count as engaged
%
% Variables:
%   eventTimes, collectStart, collectEnd, minNonEngagedWindow, eventBuffer
%   absorbSingleEvents - If true, drop isolated single events absorbed into gaps
%
% Goal:
%   Return event times whose buffers should be treated as engaged for d2/kurtosis.

eventTimes = sort(eventTimes(:));
if nargin < 6 || isempty(absorbSingleEvents)
  absorbSingleEvents = true;
end
if ~absorbSingleEvents || isempty(eventTimes)
  eventTimesEngaged = eventTimes;
  return;
end

occupied = merge_event_buffer_intervals(eventTimes, eventBuffer, collectStart, collectEnd);
absorbedMask = get_absorbed_single_event_occupied_mask( ...
  eventTimes, collectStart, collectEnd, minNonEngagedWindow, eventBuffer);
keep = true(size(eventTimes));
for iEvent = 1:numel(eventTimes)
  for iOcc = 1:numel(occupied)
    if absorbedMask(iOcc) && eventTimes(iEvent) >= occupied(iOcc).start && ...
        eventTimes(iEvent) <= occupied(iOcc).end
      keep(iEvent) = false;
      break;
    end
  end
end
eventTimesEngaged = eventTimes(keep);
end

function occupied = merge_event_buffer_intervals(reachStart, eventBuffer, collectStart, collectEnd)
% MERGE_REACH_BUFFER_INTERVALS - Union of [reach-buffer, reach+buffer] in collect window
%
% Variables:
%   reachStart               - Reach onset times (s)
%   eventBuffer              - Half-width around each reach (s)
%   collectStart, collectEnd - Clip bounds (s)
%
% Goal:
%   Return non-overlapping occupied intervals sorted by start time.

occupied = struct('start', {}, 'end', {});
if isempty(reachStart)
  return;
end

starts = max(collectStart, reachStart - eventBuffer);
ends = min(collectEnd, reachStart + eventBuffer);
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

function fig = plot_beam_break_engagement_segments(eventTimes, eventTypes, engagedSegs, ...
    nonEngagedSegs, sessionName, collectStart, collectEnd, minNonEngagedWindow, ...
    eventBuffer, plotConfig)
% PLOT_BEAM_BREAK_ENGAGEMENT_SEGMENTS - Beam breaks and engaged / non-engaged intervals
%
% Variables:
%   eventTimes           - Correct/error beam-break times in collect window (s)
%   eventTypes           - String array "correct" / "error" aligned with eventTimes
%   engagedSegs          - Struct array with .start, .end
%   nonEngagedSegs       - Struct array with .start, .end
%   sessionName          - Session id for title
%   collectStart/End     - Analysis bounds (s)
%   minNonEngagedWindow  - Gap threshold used for non-engaged (s)
%   eventBuffer          - Buffer around events excluded from non-engaged (s)
%   plotConfig           - Axis fonts and line widths
%
% Goal:
%   Schematic overview: vertical lines at correct (green) and error (red)
%   beam breaks, shaded engaged (blue) and non-engaged (orange) segments.

plotConfig = fill_default_engagement_plot_config(plotConfig);
yMin = 0;
yMax = 1;
engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];
correctColor = [0.2, 0.65, 0.25];
errorColor = [0.85, 0.2, 0.2];

fig = figure('Color', 'w', 'Name', 'Interval engagement segments', ...
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

correctMask = eventTypes == "correct";
errorMask = eventTypes == "error";
hCorrect = gobjects(0);
hError = gobjects(0);
for iEvent = 1:numel(eventTimes)
  x = eventTimes(iEvent);
  if correctMask(iEvent)
    hLine = plot(ax, [x, x], [yMin, yMax], 'Color', correctColor, 'LineWidth', 0.9, ...
      'HandleVisibility', 'off');
    if isempty(hCorrect)
      hCorrect = hLine;
      set(hCorrect, 'HandleVisibility', 'on', 'DisplayName', ...
        sprintf('Correct (n=%d)', sum(correctMask)));
    end
  elseif errorMask(iEvent)
    hLine = plot(ax, [x, x], [yMin, yMax], 'Color', errorColor, 'LineWidth', 0.9, ...
      'HandleVisibility', 'off');
    if isempty(hError)
      hError = hLine;
      set(hError, 'HandleVisibility', 'on', 'DisplayName', ...
        sprintf('Error (n=%d)', sum(errorMask)));
    end
  else
    plot(ax, [x, x], [yMin, yMax], 'Color', [0.35, 0.35, 0.35], 'LineWidth', 0.75, ...
      'HandleVisibility', 'off');
  end
end
if ~isempty(hCorrect)
  segmentHandles(end + 1) = hCorrect; %#ok<AGROW>
  segmentLabels{end + 1} = get(hCorrect, 'DisplayName'); %#ok<AGROW>
end
if ~isempty(hError)
  segmentHandles(end + 1) = hError; %#ok<AGROW>
  segmentLabels{end + 1} = get(hError, 'DisplayName'); %#ok<AGROW>
end

xlim(ax, [collectStart, collectEnd]);
ylim(ax, [yMin, yMax]);
yticks(ax, []);
titleText = sprintf( ...
  ['%s — beam-break engagement segments [%.0f–%.0f s]\n', ...
  'minNonEngagedWindow=%.1f s, eventBuffer=%.1f s | %d events (%d correct, %d error)'], ...
  sessionName, collectStart, collectEnd, minNonEngagedWindow, eventBuffer, ...
  numel(eventTimes), sum(correctMask), sum(errorMask));
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
%   metricName  - Scalar field on avData (e.g. decades, dcc)
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
    d2Window, useLog10D2, eventBuffer)
% SPLIT_D2_BY_REACH_ENGAGEMENT - Partition window-wise d2 by reach engagement
%
% Variables:
%   results      - Output of criticality_ar_analysis
%   reachStart   - Reach onsets in collect window (absolute s)
%   collectStart - Collect window start (s); startS is relative to this
%   d2Window     - Non-overlapping window length (s)
%   useLog10D2   - If true, store log10(d2) for plotting
%   eventBuffer  - Seconds around each reach that count as engaged (s)
%
% Goal:
%   Windows overlapping any [reach - eventBuffer, reach + eventBuffer] are engaged
%   (including windows with no reach onset but within the buffer). Others are
%   non-engaged. Total uses all windows. Duration = nWindows * d2Window per class.

if nargin < 6 || isempty(eventBuffer)
  eventBuffer = 0;
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
  engagedMask = window_overlaps_event_buffer(winStartAbs, winEndAbs, reachStart, eventBuffer);
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

function [eventTimesFiltered, nExcludedEarlyErrors] = filter_interval_events_for_trial_rate( ...
    trials, sessionInterval, rewardAttemptBeforeSec)
% FILTER_INTERVAL_EVENTS_FOR_TRIAL_RATE - Beam-break times for trial-rate correlation
%
% Variables:
%   trials                 - Output of extract_interval_trials (type, pokeTimeSec, ...)
%   sessionInterval        - Target interval (s)
%   rewardAttemptBeforeSec - Exclude errors with poke earlier than interval minus this
%
% Goal:
%   Return outcome times for correct trials and errors in the reward-seeking
%   window (poke >= sessionInterval - rewardAttemptBeforeSec), matching
%   interval_session_performance reward-attempt filtering.

pokeMinSec = sessionInterval - rewardAttemptBeforeSec;
earlyErrorMask = trials.type == "error" & trials.pokeTimeSec < pokeMinSec;
nExcludedEarlyErrors = sum(earlyErrorMask);
keepMask = ~earlyErrorMask;
eventTimesFiltered = trials.outcomeTimeSec(keepMask);
end

function corrResult = build_d2_trial_rate_correlation(results, eventTimes, ...
    collectStart, d2Window, useLog10D2)
% BUILD_D2_TRIAL_RATE_CORRELATION - d2 and trial count/rate per window (total)
%
% Variables:
%   results      - Output of criticality_ar_analysis
%   eventTimes   - Filtered beam-break outcome times in collect window (absolute s)
%   collectStart - Collect window start (s); startS is relative to this
%   d2Window     - Non-overlapping window length (s)
%   useLog10D2   - If true, use log10(d2) for correlation
%
% Goal:
%   For each d2 window, count beam-break events and compute trial rate (count /
%   d2Window). Correlate d2 with trial count across all windows.

eventTimes = eventTimes(:);

corrResult = struct();
corrResult.d2Window = d2Window;
corrResult.areas = {};
corrResult.byArea = {};

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
  trialCountVec = compute_window_event_count(eventTimes, winStartAbs, winEndAbs);
  trialRateVec = trialCountVec / d2Window;

  validMask = isfinite(d2Vec);
  areaResult = struct();
  areaResult.areaName = results.areas{a};
  areaResult.d2 = d2Vec;
  areaResult.nTrials = trialCountVec;
  areaResult.trialRate = trialRateVec;
  areaResult.validMask = validMask;
  areaResult.nWindows = nWindows;
  areaResult.nValidWindows = sum(validMask);
  areaResult.rPearson = pearson_r(d2Vec(validMask), trialCountVec(validMask));
  areaResult.rPearsonRate = pearson_r(d2Vec(validMask), trialRateVec(validMask));
  corrResult.byArea{end + 1} = areaResult; %#ok<AGROW>
  corrResult.areas{end + 1} = results.areas{a}; %#ok<AGROW>
end
end

function eventCountVec = compute_window_event_count(eventTimes, winStartAbs, winEndAbs)
% COMPUTE_WINDOW_EVENT_COUNT - Beam-break events per d2 window
%
% Variables:
%   eventTimes  - Outcome times (s)
%   winStartAbs - Window start times (s), one per window
%   winEndAbs   - Window end times (s); interval is [winStart, winEnd)
%
% Goal:
%   For each window, return the number of events in [winStart, winEnd).

nWindows = numel(winStartAbs);
eventCountVec = zeros(nWindows, 1);
eventTimes = eventTimes(:);
for w = 1:nWindows
  eventCountVec(w) = sum(eventTimes >= winStartAbs(w) & eventTimes < winEndAbs(w));
end
end

function print_d2_trial_rate_correlation(corrResult, d2Window, sessionInterval, ...
    rewardAttemptBeforeSec)
% PRINT_D2_TRIAL_RATE_CORRELATION - Command-window summary per area (all windows)

fprintf(['Trial count = beam breaks per d2 window; rate = count / %.0f s\n', ...
  'Early errors excluded: poke < %.1f s (interval %.0f s, buffer %.1f s before)\n'], ...
  d2Window, sessionInterval - rewardAttemptBeforeSec, sessionInterval, rewardAttemptBeforeSec);
if isempty(corrResult.areas)
  fprintf('  No areas with d2 window data.\n');
  return;
end
for a = 1:numel(corrResult.byArea)
  areaResult = corrResult.byArea{a};
  if areaResult.nValidWindows == 0
    fprintf('  %s: no windows with finite d2\n', areaResult.areaName);
    continue;
  end
  nTrialVals = areaResult.nTrials(areaResult.validMask);
  fprintf(['  %s: r=%.3f (count), r=%.3f (rate), n=%d windows ', ...
    '(trial count %.0f-%.0f, mean %.2f)\n'], ...
    areaResult.areaName, areaResult.rPearson, areaResult.rPearsonRate, ...
    areaResult.nValidWindows, min(nTrialVals), max(nTrialVals), mean(nTrialVals));
end
end

function fig = plot_d2_trial_rate_correlation(corrResult, sessionName, d2Window, ...
    useLog10D2, plotConfig)
% PLOT_D2_TRIAL_RATE_CORRELATION - Scatter of trial count vs d2 (all windows)

plotConfig = fill_default_engagement_plot_config(plotConfig);
classColors = engagement_class_colors();
totalColor = classColors(1, :);
numAreas = numel(corrResult.areas);
if numAreas == 0
  warning('interval_criticality_metrics_engagement:NoD2TrialRateData', ...
    'No areas available for d2-trial-rate correlation plot.');
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

fig = figure('Color', 'w', 'Name', 'd2 vs trial rate', ...
  'Position', [120 120 420 * numAreas 340]);
tileLayout = tiledlayout(fig, 1, numAreas, 'TileSpacing', 'compact', 'Padding', 'compact');

for a = 1:numAreas
  areaResult = corrResult.byArea{a};
  ax = nexttile(tileLayout);
  hold(ax, 'on');

  validMask = areaResult.validMask;
  if ~any(validMask)
    title(ax, sprintf('%s (no finite d2)', areaResult.areaName), 'Interpreter', 'none');
    apply_engagement_axes_style(ax, plotConfig);
    hold(ax, 'off');
    continue;
  end

  nTrialPlot = areaResult.nTrials(validMask);
  d2Plot = areaResult.d2(validMask);
  scatter(ax, nTrialPlot, d2Plot, 48, totalColor, 'filled', ...
    'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');

  rVal = areaResult.rPearson;
  apply_engagement_axes_style(ax, plotConfig, 'Trials per window', d2YLabel, ...
    sprintf('%s | r=%.3f, n=%d', areaResult.areaName, rVal, sum(validMask)), ...
    labelInterpreter);
  if max(nTrialPlot) > min(nTrialPlot)
    xlim(ax, [min(nTrialPlot) - 0.5, max(nTrialPlot) + 0.5]);
  end
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayout, sprintf('%s — d2 vs trial count | %.0fs windows', ...
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

function prgSplit = split_prg_by_reach_engagement(results, reachStart, prgWindow, eventBuffer)
% SPLIT_PRG_BY_REACH_ENGAGEMENT - Partition PRG windows by reach engagement
%
% Variables:
%   results     - Output of criticality_prg_analysis
%   reachStart  - Reach onsets (absolute s)
%   prgWindow   - Non-overlapping block length (s)
%   eventBuffer - Seconds around each reach that count as engaged (s)
%
% Goal:
%   Blocks overlapping any [reach - eventBuffer, reach + eventBuffer] are engaged
%   (including blocks with no reach onset but within the buffer). Duration =
%   nWindows * prgWindow per class (all blocks, including CV-excluded); metric
%   vectors use valid (non-excluded) windows only.

if nargin < 4 || isempty(eventBuffer)
  eventBuffer = 0;
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
  engagedMask = window_overlaps_event_buffer(winStartAbs, winEndAbs, reachStart, eventBuffer);
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
    fprintf(['  AV %s: n=%d, tau=%.2f, alpha=%.2f, paramSD=%.3f, ', ...
      'decades=%.2f, dcc=%.3f, (alpha-1)/(tau-1)=%.3f\n'], ...
      areaName, avData.nAvalanches, avData.tau, avData.alpha, avData.paramSD, ...
      avData.decades, avData.dcc, avData.scalingRelation);
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
% PLOT_ENGAGEMENT_AVALANCHE_DISTRIBUTIONS - Size/duration CCDFs + ⟨S⟩(T) by class
%
% Goal:
%   Overlay total / engaged / non-engaged CCDFs, plus circular-shuffle CCDF for
%   the full-session (total) data only. Third column: crackling ⟨S⟩(T) with WLS fit.

plotConfig = fill_default_engagement_plot_config(plotConfig);
classNames = engagement_class_names();
classColors = engagement_class_colors();
classFields = {'total', 'engaged', 'nonEngaged'};
shuffleColor = [0.55, 0.55, 0.55];
nAreas = numel(areaNames);

fig = figure('Color', 'w', 'Position', [80 80 max(1100, 420 * nAreas) 440], ...
  'Name', 'Engagement avalanche distributions');
tiledlayout(fig, nAreas, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for aIdx = 1:nAreas
  axSize = nexttile((aIdx - 1) * 3 + 1);
  axDur = nexttile((aIdx - 1) * 3 + 2);
  axCrack = nexttile((aIdx - 1) * 3 + 3);
  hold(axSize, 'on');
  hold(axDur, 'on');
  hold(axCrack, 'on');

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

    plot_engagement_size_given_duration(axCrack, avData, classColors(c, :), classNames{c}, plotConfig);

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
  set(axCrack, 'XScale', 'log', 'YScale', 'log');
  apply_engagement_axes_style(axSize, plotConfig, 'Avalanche size', 'P(X \geq x)', ...
    sprintf('%s — size', areaNames{aIdx}), 'tex');
  apply_engagement_axes_style(axDur, plotConfig, 'Avalanche duration (ms)', 'P(X \geq x)', ...
    sprintf('%s — duration', areaNames{aIdx}), 'tex');
  apply_engagement_axes_style(axCrack, plotConfig, 'Duration (bins)', '\langleS\rangle(T)', ...
    sprintf('%s — crackling', areaNames{aIdx}), 'tex');
  grid(axSize, 'on');
  grid(axDur, 'on');
  grid(axCrack, 'on');
  legend(axSize, 'Location', 'southwest', 'Interpreter', 'tex', ...
    'FontSize', plotConfig.legendFontSize);
  legend(axDur, 'Location', 'southwest', 'Interpreter', 'tex', ...
    'FontSize', plotConfig.legendFontSize);
  legend(axCrack, 'Location', 'northwest', 'Interpreter', 'tex', ...
    'FontSize', plotConfig.legendFontSize);
  hold(axSize, 'off');
  hold(axDur, 'off');
  hold(axCrack, 'off');
end

sgtitle(fig, sprintf( ...
  ['%s — avalanche CCDFs + crackling by engagement [%.0f–%.0f s]\n', ...
  'durations: total %.1f min, engaged %.1f min, non-engaged %.1f min'], ...
  sessionName, collectStart, collectEnd, ...
  avDurations.totalSec / 60, avDurations.engagedSec / 60, avDurations.nonEngagedSec / 60), ...
  'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
end

function plot_engagement_size_given_duration(ax, avData, lineColor, className, plotConfig)
% PLOT_ENGAGEMENT_SIZE_GIVEN_DURATION - ⟨S⟩|T markers + WLS crackling slope

sizes = avData.sizes(:);
durations = avData.durations(:);
valid = isfinite(sizes) & isfinite(durations) & sizes > 0 & durations > 0;
sizes = sizes(valid);
durations = durations(valid);
if numel(sizes) < 2
  return;
end

unqDurations = unique(durations);
meanSize = nan(size(unqDurations));
for iDur = 1:numel(unqDurations)
  meanSize(iDur) = mean(sizes(durations == unqDurations(iDur)));
end
validMean = isfinite(meanSize) & meanSize > 0;
unqDurations = unqDurations(validMean);
meanSize = meanSize(validMean);

paramSD = nan;
if isfield(avData, 'paramSD') && isfinite(avData.paramSD)
  paramSD = avData.paramSD;
end
displayName = sprintf('%s (1/\\sigma\\nu z=%.2f)', className, paramSD);
plot(ax, unqDurations, meanSize, 'o-', 'Color', lineColor, ...
  'MarkerFaceColor', lineColor, 'MarkerSize', plotConfig.markerSize, ...
  'LineWidth', plotConfig.lineWidth, 'DisplayName', displayName);

durMin = nan;
durMax = nan;
if isfield(avData, 'minDurFit'), durMin = avData.minDurFit; end
if isfield(avData, 'maxDurFit'), durMax = avData.maxDurFit; end
if ~(isfinite(durMin) && isfinite(durMax) && durMin <= durMax)
  return;
end
[fitParamSD, ~, logCoeff] = size_given_duration(sizes, durations, ...
  'durmin', durMin, 'durmax', durMax);
if ~(isfinite(fitParamSD) && isfinite(logCoeff) && durMin > 0 && durMax > durMin)
  return;
end
xFit = logspace(log10(durMin), log10(durMax), 60);
yFit = 10 .^ (fitParamSD * log10(xFit) + logCoeff);
plot(ax, xFit, yFit, '-', 'Color', lineColor, 'LineWidth', max(1.5, plotConfig.lineWidth), ...
  'HandleVisibility', 'off');
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

colors = manuscript_plot_colors().engagementClasses;
end

function isEngaged = window_overlaps_event_buffer(winStartAbs, winEndAbs, reachStart, eventBuffer)
% WINDOW_OVERLAPS_REACH_BUFFER - True if window overlaps any reach buffer interval
%
% Variables:
%   winStartAbs, winEndAbs - Window bounds [start, end) in absolute seconds
%   reachStart             - Reach onset times (s)
%   eventBuffer            - Half-width around each reach treated as engaged (s)
%
% Goal:
%   Engaged if the window overlaps [reach - eventBuffer, reach + eventBuffer] for
%   any reach, so near-reach windows without an onset inside still count.

if nargin < 4 || isempty(eventBuffer)
  eventBuffer = 0;
end
eventBuffer = max(0, eventBuffer);

isEngaged = false(size(winStartAbs));
if isempty(reachStart)
  return;
end
reachStart = reachStart(:);
for w = 1:numel(winStartAbs)
  % Half-open window [winStart, winEnd) vs closed buffer [reach-buf, reach+buf]
  isEngaged(w) = any(winStartAbs(w) <= reachStart + eventBuffer & ...
    winEndAbs(w) > reachStart - eventBuffer);
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
[binEdges, xMin, xMax] = build_shared_histogram_bin_edges(allVals, nBinsTarget, forcedLimits);
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
