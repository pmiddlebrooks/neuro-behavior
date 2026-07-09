function out = spontaneous_criticality_metrics_behaviors(subjectName, sessionName, opts)
% SPONTANEOUS_CRITICALITY_METRICS_BEHAVIORS - d2 vs spontaneous behavior features
%
% Variables:
%   subjectName - Subject folder under spontaneous data (e.g. 'ag25290')
%   sessionName - Session identifier (e.g. 'ag112321_1')
%   opts        - Options struct. Fields:
%     Session loading (neuro_behavior_options overrides):
%       .collectStart, .collectEnd, .minFiringRate, .maxFiringRate,
%       .firingRateCheckTime, .dataSource, .fsBhv
%     Analysis:
%       .analyses - Cell of {'d2'} (default); future: sequences, clusters
%       .brainArea, .brainAreaCombinations
%       .useSubsampling, .nSubsamples, .nNeuronsSubsample, .minNeuronsMultiple
%     Behavior labels:
%       .smoothBehaviorLabels, .behaviorSmoothingWindow
%       .useBehaviorRecoding, .behaviorRecodingRules
%       .behaviorLabelSets - Cell of structs (.name, .numeratorIDs, .denominatorIDs)
%                            for d2 vs behavior proportion scatters
%       .useRawLabelsForLabelSets - Use raw B-SOiD IDs for label sets (default true)
%       .runD2BehaviorLabelSetCorrelation - d2 vs label-set proportions (default true)
%     d2:
%       .d2Window, .useLog10D2, .normalizeD2, .nShufflesD2
%       .runD2EntropyCorrelation   - Correlate d2 with label entropy (default true)
%       .runD2SwitchRateCorrelation - Correlate d2 with switch rate (default true)
%     Output:
%       .makePlots, .saveFigure, .saveResults, .outputDir, .plotConfig
%
% Goal:
%   For each non-overlapping d2 window, compute behavior switch rate and label-
%   distribution entropy from frame-wise behavior labels. Correlate d2 with each
%   feature per brain area (all windows). Optionally correlate d2 with predefined
%   behavior label-set proportions (Inv, Loco, Itch, Groom by default). Future
%   extensions may target windows with specific sequences or clusters (see criticality_spontaneous_sequences_d2,
%   criticality_spontaneous_clusters).
%
% Returns:
%   With no inputs: default options struct.
%   Otherwise: struct with config, behavior metrics, d2 correlations, figHandles.

setup_spontaneous_criticality_metrics_behaviors_paths();

if nargin == 0
  out = fill_behavior_opts_defaults(struct());
  return;
end
if nargin < 2 || isempty(subjectName) || isempty(sessionName)
  error('spontaneous_criticality_metrics_behaviors:MissingSession', ...
    'subjectName and sessionName are required.');
end
if nargin < 3 || isempty(opts)
  opts = struct();
end
opts = fill_behavior_opts_defaults(opts);

sessionType = 'spontaneous';
dataSource = opts.dataSource;
collectStart = opts.collectStart;
collectEnd = opts.collectEnd;

fprintf('\n=== Spontaneous criticality metrics by behavior ===\n');
fprintf('Session: %s / %s\n', subjectName, sessionName);
fprintf('Analyses: %s\n', strjoin(opts.analyses, ', '));

%% Load neural session
loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = opts.firingRateCheckTime;
loadOpts.collectStart = collectStart;
loadOpts.collectEnd = collectEnd;
loadOpts.minFiringRate = opts.minFiringRate;
loadOpts.maxFiringRate = opts.maxFiringRate;
loadOpts.fsBhv = opts.fsBhv;

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

[dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
  dataStruct, opts.brainArea, opts.brainAreaCombinations, true);
if ~areaOk
  error('Brain area "%s" not available in this session.', opts.brainArea);
end

fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
  collectStart, collectEnd, (collectEnd - collectStart) / 60);

paths = get_paths();
behaviorData = load_spontaneous_behavior_labels(paths, subjectName, sessionName, opts);
behaviorData = align_behavior_to_collect_window(behaviorData, collectStart, collectEnd);

fprintf('Behavior labels: %d frames (fs=%.2f Hz)\n', ...
  numel(behaviorData.labels), behaviorData.fsBhv);

out = struct();
out.subjectName = subjectName;
out.sessionName = sessionName;
out.config = opts;
out.behavior = behaviorData;
out.d2 = [];
out.d2EntropyCorrelation = [];
out.d2SwitchRateCorrelation = [];
out.d2BehaviorLabelSetCorrelation = [];
out.behaviorPerWindow = [];
out.figHandles = struct();

plotConfig = opts.plotConfig;
saveDir = resolve_spontaneous_behavior_save_dir(dataStruct, subjectName, sessionName, opts);

%% d2 and behavior correlations
if ismember('d2', opts.analyses)
  fprintf('\n--- d2 ---\n');
  fprintf('d2Window: %.1f s (non-overlapping)\n', opts.d2Window);

  arConfig = build_ar_config(opts);
  resultsD2 = criticality_ar_analysis(dataStruct, arConfig);
  if ~isempty(opts.brainArea)
    resultsD2 = filter_ar_results_to_brain_area(resultsD2, opts.brainArea);
  end
  if isempty(resultsD2.areas)
    error('No d2 results for brain area "%s".', opts.brainArea);
  end

  out.d2 = resultsD2;
  behaviorPerWindow = compute_behavior_metrics_for_d2_windows( ...
    resultsD2, behaviorData, collectStart, opts.d2Window);
  out.behaviorPerWindow = behaviorPerWindow;

  fprintf('Windows: %d | mean switch rate %.3f/s | mean entropy %.3f bits\n', ...
    behaviorPerWindow.nWindows, nanmean(behaviorPerWindow.switchRate), ...
    nanmean(behaviorPerWindow.entropy));

  if opts.runD2EntropyCorrelation
    fprintf('\n--- d2 vs behavior entropy (all windows) ---\n');
    d2EntropyCorr = build_d2_behavior_feature_correlation(resultsD2, ...
      behaviorPerWindow.entropy, collectStart, opts.d2Window, opts.useLog10D2, 'entropy');
    out.d2EntropyCorrelation = d2EntropyCorr;
    print_d2_behavior_correlation(d2EntropyCorr, 'Label entropy (bits)', opts.d2Window);
    if opts.makePlots
      out.figHandles.d2EntropyCorrelation = plot_d2_behavior_correlation( ...
        d2EntropyCorr, sessionName, opts.d2Window, opts.useLog10D2, ...
        'Label entropy (bits)', plotConfig);
    end
  end

  if opts.runD2SwitchRateCorrelation
    fprintf('\n--- d2 vs behavior switch rate (all windows) ---\n');
    d2SwitchCorr = build_d2_behavior_feature_correlation(resultsD2, ...
      behaviorPerWindow.switchRate, collectStart, opts.d2Window, opts.useLog10D2, 'switchRate');
    out.d2SwitchRateCorrelation = d2SwitchCorr;
    print_d2_behavior_correlation(d2SwitchCorr, 'Switch rate (switches/s)', opts.d2Window);
    if opts.makePlots
      out.figHandles.d2SwitchRateCorrelation = plot_d2_behavior_correlation( ...
        d2SwitchCorr, sessionName, opts.d2Window, opts.useLog10D2, ...
        'Switch rate (switches/s)', plotConfig);
    end
  end

  if opts.runD2BehaviorLabelSetCorrelation && ~isempty(opts.behaviorLabelSets)
    fprintf('\n--- d2 vs behavior label-set proportions ---\n');
    labelSetCorr = build_d2_behavior_label_set_correlation(resultsD2, behaviorData, ...
      collectStart, opts.d2Window, opts.useLog10D2, opts.behaviorLabelSets, ...
      opts.useRawLabelsForLabelSets);
    out.d2BehaviorLabelSetCorrelation = labelSetCorr;
    print_d2_behavior_label_set_correlations(labelSetCorr, opts.d2Window);
    if opts.makePlots
      out.figHandles.d2BehaviorLabelSets = plot_d2_vs_behavior_label_sets( ...
        resultsD2, behaviorData, collectStart, opts.behaviorLabelSets, ...
        opts.d2Window, opts.useLog10D2, opts.useRawLabelsForLabelSets, ...
        sessionName, plotConfig);
    end
  end
end

out.saveDir = saveDir;

if opts.saveFigure && ~isempty(fieldnames(out.figHandles))
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  areaTag = format_areas_label(opts.brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  baseName = sprintf('spontaneous_criticality_metrics_behaviors_%s_%s_W%.0f', ...
    sessionName, areaTag, opts.d2Window);
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

if opts.saveResults
  if ~exist(saveDir, 'dir')
    mkdir(saveDir);
  end
  resultsPath = fullfile(saveDir, sprintf( ...
    'spontaneous_criticality_metrics_behaviors_%s_W%.0f.mat', sessionName, opts.d2Window));
  results = out; %#ok<NASGU>
  save(resultsPath, 'results');
  out.savedResultsPath = resultsPath;
  fprintf('Saved results: %s\n', resultsPath);
end

fprintf('\n=== Done ===\n');
end

%% -------------------------------------------------------------------------
%% Defaults and paths
%% -------------------------------------------------------------------------

function opts = fill_behavior_opts_defaults(opts)
% FILL_BEHAVIOR_OPTS_DEFAULTS - Session-load and analysis option defaults

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
if ~isfield(opts, 'fsBhv') || isempty(opts.fsBhv)
  opts.fsBhv = 60;
end

if ~isfield(opts, 'analyses') || isempty(opts.analyses)
  opts.analyses = {'d2'};
end
if ischar(opts.analyses) || isstring(opts.analyses)
  opts.analyses = cellstr(opts.analyses);
end
opts.analyses = lower(opts.analyses(:)');
validAnalyses = {'d2'};
unknown = setdiff(opts.analyses, validAnalyses);
if ~isempty(unknown)
  error('spontaneous_criticality_metrics_behaviors:BadAnalyses', ...
    'Unknown analyses: %s. Use d2 (sequences/clusters planned).', strjoin(unknown, ', '));
end

if ~isfield(opts, 'brainArea')
  opts.brainArea = 'M23M56';
end
if ~isfield(opts, 'brainAreaCombinations') || isempty(opts.brainAreaCombinations)
  opts.brainAreaCombinations = default_manuscript_brain_area_combinations();
end

if ~isfield(opts, 'useSubsampling') || isempty(opts.useSubsampling)
  opts.useSubsampling = true;
end
if ~isfield(opts, 'nSubsamples') || isempty(opts.nSubsamples)
  opts.nSubsamples = 30;
end
if ~isfield(opts, 'nNeuronsSubsample') || isempty(opts.nNeuronsSubsample)
  opts.nNeuronsSubsample = 20;
end
if ~isfield(opts, 'minNeuronsMultiple') || isempty(opts.minNeuronsMultiple)
  opts.minNeuronsMultiple = 1.2;
end

if ~isfield(opts, 'smoothBehaviorLabels') || isempty(opts.smoothBehaviorLabels)
  opts.smoothBehaviorLabels = false;
end
if ~isfield(opts, 'behaviorSmoothingWindow') || isempty(opts.behaviorSmoothingWindow)
  opts.behaviorSmoothingWindow = 0.25;
end
if ~isfield(opts, 'useBehaviorRecoding') || isempty(opts.useBehaviorRecoding)
  opts.useBehaviorRecoding = false;
end
if ~isfield(opts, 'behaviorRecodingRules') || isempty(opts.behaviorRecodingRules)
  opts.behaviorRecodingRules = default_spontaneous_behavior_recoding_rules();
end
if ~isfield(opts, 'behaviorLabelSets')
  opts.behaviorLabelSets = default_spontaneous_behavior_label_sets();
end
if isstruct(opts.behaviorLabelSets)
  opts.behaviorLabelSets = {opts.behaviorLabelSets};
end
if ~isfield(opts, 'useRawLabelsForLabelSets') || isempty(opts.useRawLabelsForLabelSets)
  opts.useRawLabelsForLabelSets = true;
end
if ~isfield(opts, 'runD2BehaviorLabelSetCorrelation') || isempty(opts.runD2BehaviorLabelSetCorrelation)
  opts.runD2BehaviorLabelSetCorrelation = true;
end

if ~isfield(opts, 'd2Window') || isempty(opts.d2Window)
  opts.d2Window = 30;
end
if ~isfield(opts, 'useLog10D2') || isempty(opts.useLog10D2)
  opts.useLog10D2 = true;
end
if ~isfield(opts, 'normalizeD2') || isempty(opts.normalizeD2)
  opts.normalizeD2 = false;
end
if ~isfield(opts, 'nShufflesD2') || isempty(opts.nShufflesD2)
  opts.nShufflesD2 = 8;
end
if ~isfield(opts, 'runD2EntropyCorrelation') || isempty(opts.runD2EntropyCorrelation)
  opts.runD2EntropyCorrelation = true;
end
if ~isfield(opts, 'runD2SwitchRateCorrelation') || isempty(opts.runD2SwitchRateCorrelation)
  opts.runD2SwitchRateCorrelation = true;
end

if ~isfield(opts, 'makePlots') || isempty(opts.makePlots)
  opts.makePlots = true;
end
if ~isfield(opts, 'saveFigure') || isempty(opts.saveFigure)
  opts.saveFigure = true;
end
if ~isfield(opts, 'saveResults') || isempty(opts.saveResults)
  opts.saveResults = false;
end
if ~isfield(opts, 'outputDir')
  opts.outputDir = '';
end
if ~isfield(opts, 'plotConfig') || isempty(opts.plotConfig)
  opts.plotConfig = struct();
end
opts.plotConfig = fill_default_behavior_plot_config(opts.plotConfig);
end

function rules = default_spontaneous_behavior_recoding_rules()
% DEFAULT_SPONTANEOUS_BEHAVIOR_RECODING_RULES - Super-category labels (behavior_vs_d2)

rules = {...
  [0 1 2], 1;
  [3], 2;
  4, 3;
  5:12, 4;
  13:15, 5;
  };
end

function labelSets = default_spontaneous_behavior_label_sets()
% DEFAULT_SPONTANEOUS_BEHAVIOR_LABEL_SETS - Raw B-SOiD label groups for proportion scatters
%
% denominatorIDs empty -> fraction of all behavior frames in the window.

labelSets = {
  struct('name', 'Inv', 'numeratorIDs', [0 1 2], 'denominatorIDs', [])
  struct('name', 'Loco', 'numeratorIDs', [15], 'denominatorIDs', [])
  struct('name', 'Itch', 'numeratorIDs', [11:12], 'denominatorIDs', [])
  struct('name', 'Groom', 'numeratorIDs', 5:10, 'denominatorIDs', [])
  };
end

function plotConfig = fill_default_behavior_plot_config(plotConfig)
% FILL_DEFAULT_BEHAVIOR_PLOT_CONFIG - Axis fonts and line widths

plotConfig = fill_manuscript_plot_config(plotConfig);
end

function setup_spontaneous_criticality_metrics_behaviors_paths()
% SETUP_SPONTANEOUS_CRITICALITY_METRICS_BEHAVIORS_PATHS - Add neuro-behavior paths

scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_']) || ~isfolder(fullfile(scriptDir, '..', 'spontaneous'))
  resolved = which('spontaneous_criticality_metrics_behaviors');
  if ~isempty(resolved)
    scriptDir = fileparts(resolved);
  end
end
srcPath = fullfile(scriptDir, '..');
pathDirs = {
  srcPath
  fullfile(srcPath, 'spontaneous')
  fullfile(srcPath, 'reach_task')
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

function saveDir = resolve_spontaneous_behavior_save_dir(dataStruct, subjectName, sessionName, opts)
% RESOLVE_SPONTANEOUS_BEHAVIOR_SAVE_DIR - Output directory for figures and .mat

if ~isempty(opts.outputDir)
  saveDir = opts.outputDir;
elseif isfield(dataStruct, 'saveDir') && ~isempty(dataStruct.saveDir)
  saveDir = dataStruct.saveDir;
else
  paths = get_paths();
  saveDir = fullfile(paths.dropPath, 'spontaneous', 'results', ...
    subjectName, matlab.lang.makeValidName(sessionName));
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
arConfig.enablePermutations = opts.normalizeD2;
arConfig.nShuffles = opts.nShufflesD2;
arConfig.normalizeD2 = opts.normalizeD2;
arConfig.useLog10D2 = opts.useLog10D2;
arConfig.makePlots = false;
arConfig.saveData = false;
arConfig.pOrder = 10;
arConfig.critType = 2;
arConfig.minSpikesPerBin = 3;
arConfig.minBinsPerWindow = 1000;
arConfig.maxSpikesPerBin = 100;
arConfig.nMinNeurons = 10;
arConfig.useSubsampling = opts.useSubsampling;
arConfig.nSubsamples = opts.nSubsamples;
arConfig.nNeuronsSubsample = opts.nNeuronsSubsample;
arConfig.minNeuronsMultiple = opts.minNeuronsMultiple;
end

%% -------------------------------------------------------------------------
%% Behavior loading and per-window metrics
%% -------------------------------------------------------------------------

function behaviorData = load_spontaneous_behavior_labels(paths, subjectName, sessionName, opts)
% LOAD_SPONTANEOUS_BEHAVIOR_LABELS - Frame-wise labels from behavior_labels*.csv

sessionFolder = fullfile(paths.spontaneousDataPath, subjectName, sessionName);

csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
if isempty(csvFiles)
  error('spontaneous_criticality_metrics_behaviors:NoBehaviorCsv', ...
    'No behavior_labels*.csv found in %s', sessionFolder);
elseif numel(csvFiles) > 1
  warning('Multiple behavior_labels*.csv files found. Using first: %s', csvFiles(1).name);
end

behaviorTable = readtable(fullfile(sessionFolder, csvFiles(1).name));
if ~ismember('Code', behaviorTable.Properties.VariableNames)
  error('spontaneous_criticality_metrics_behaviors:BadBehaviorCsv', ...
    'Behavior CSV must contain a ''Code'' column.');
end

labelsRaw = behaviorTable.Code(:);
if opts.smoothBehaviorLabels
  smoothOpts = struct();
  smoothOpts.fsBhv = opts.fsBhv;
  smoothOpts.smoothingWindow = opts.behaviorSmoothingWindow;
  smoothOpts.summarize = false;
  labelsUsed = behavior_label_smoothing(labelsRaw, smoothOpts);
else
  labelsUsed = labelsRaw;
end

if opts.useBehaviorRecoding
  labelsUsed = recode_behavior_labels(labelsUsed, opts.behaviorRecodingRules);
end

behaviorData = struct();
behaviorData.labels = labelsUsed;
behaviorData.labelsRaw = labelsRaw;
behaviorData.fsBhv = opts.fsBhv;
behaviorData.csvPath = fullfile(sessionFolder, csvFiles(1).name);
behaviorData.sessionFolder = sessionFolder;
end

function behaviorData = align_behavior_to_collect_window(behaviorData, collectStart, collectEnd)
% ALIGN_BEHAVIOR_TO_COLLECT_WINDOW - Time axis and trim to collect window
%
% Frame k is at collectStart + k/fsBhv (same convention as behavior_vs_d2).

fsBhv = behaviorData.fsBhv;
labels = behaviorData.labels(:);
nFrames = numel(labels);
timeAxisSec = collectStart + (0:(nFrames - 1))' ./ fsBhv;
inCollect = timeAxisSec >= collectStart & timeAxisSec <= collectEnd;
behaviorData.labels = labels(inCollect);
if isfield(behaviorData, 'labelsRaw')
  behaviorData.labelsRaw = behaviorData.labelsRaw(inCollect);
end
behaviorData.timeAxisSec = timeAxisSec(inCollect);
behaviorData.collectStart = collectStart;
behaviorData.collectEnd = collectEnd;
end

function behaviorPerWindow = compute_behavior_metrics_for_d2_windows( ...
    resultsD2, behaviorData, collectStart, d2Window)
% COMPUTE_BEHAVIOR_METRICS_FOR_D2_WINDOWS - Switch rate and entropy per d2 window

refAreaIdx = find(~cellfun(@isempty, resultsD2.startS), 1, 'first');
if isempty(refAreaIdx)
  error('spontaneous_criticality_metrics_behaviors:NoD2Windows', ...
    'No d2 window centers in results.');
end

centerRel = resultsD2.startS{refAreaIdx}(:);
nWindows = numel(centerRel);
winStartAbs = collectStart + centerRel - d2Window / 2;
winEndAbs = collectStart + centerRel + d2Window / 2;
winCenterAbs = collectStart + centerRel;

switchRate = nan(nWindows, 1);
entropyBits = nan(nWindows, 1);
timeAxisSec = behaviorData.timeAxisSec;
labels = behaviorData.labels;

for w = 1:nWindows
  inWindow = timeAxisSec >= winStartAbs(w) & timeAxisSec < winEndAbs(w);
  labelsWindow = labels(inWindow);
  switchRate(w) = calculate_switch_rate(labelsWindow, d2Window);
  entropyBits(w) = calculate_label_entropy(labelsWindow);
end

behaviorPerWindow = struct();
behaviorPerWindow.nWindows = nWindows;
behaviorPerWindow.windowStartSec = winStartAbs;
behaviorPerWindow.windowEndSec = winEndAbs;
behaviorPerWindow.windowCenterSec = winCenterAbs;
behaviorPerWindow.switchRate = switchRate;
behaviorPerWindow.entropy = entropyBits;
end

function switchRate = calculate_switch_rate(labelsWindow, windowDurationSec)
% CALCULATE_SWITCH_RATE - Adjacent label changes per second in one window

switchRate = nan;
if isempty(labelsWindow) || numel(labelsWindow) < 2 || windowDurationSec <= 0
  return;
end
labelsWindow = labelsWindow(:);
validMask = ~isnan(labelsWindow);
labelsWindow = labelsWindow(validMask);
if numel(labelsWindow) < 2
  return;
end
numSwitches = sum(diff(labelsWindow) ~= 0);
switchRate = numSwitches / windowDurationSec;
end

function entropyBits = calculate_label_entropy(labelsWindow)
% CALCULATE_LABEL_ENTROPY - Shannon entropy (bits) of label distribution in window

entropyBits = nan;
if isempty(labelsWindow)
  return;
end
labelsWindow = labelsWindow(:);
validMask = ~isnan(labelsWindow);
labelsWindow = labelsWindow(validMask);
if isempty(labelsWindow)
  return;
end
[~, ~, groupIdx] = unique(labelsWindow);
counts = accumarray(groupIdx, 1);
probVec = counts / sum(counts);
entropyBits = -sum(probVec .* log2(probVec));
end

function labelsOut = recode_behavior_labels(labelsIn, behaviorRecodingRules)
% RECODE_BEHAVIOR_LABELS - Collapse behavior labels into super-categories

labelsOut = labelsIn;
if isempty(behaviorRecodingRules)
  return;
end
if ~iscell(behaviorRecodingRules) || size(behaviorRecodingRules, 2) ~= 2
  error('behaviorRecodingRules must be an Nx2 cell array: { [oldLabels], newLabel }.');
end
for iRule = 1:size(behaviorRecodingRules, 1)
  oldLabels = behaviorRecodingRules{iRule, 1};
  newLabel = behaviorRecodingRules{iRule, 2};
  if isempty(oldLabels) || ~isnumeric(oldLabels) || ~isscalar(newLabel) || ~isnumeric(newLabel)
    error('Invalid behavior recoding rule at row %d.', iRule);
  end
  mask = ismember(labelsOut, oldLabels);
  labelsOut(mask) = newLabel;
end
end

%% -------------------------------------------------------------------------
%% d2 vs behavior correlations and plots
%% -------------------------------------------------------------------------

function corrResult = build_d2_behavior_label_set_correlation(results, behaviorData, ...
    collectStart, d2Window, useLog10D2, behaviorLabelSets, useRawLabels)
% BUILD_D2_BEHAVIOR_LABEL_SET_CORRELATION - Pearson r for each behavior label set

corrResult = struct();
corrResult.behaviorLabelSets = behaviorLabelSets;
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

  bhvProportions = compute_behavior_proportions_by_set( ...
    behaviorData, centerRel, collectStart, d2Window, behaviorLabelSets, useRawLabels);

  areaResult = struct();
  areaResult.areaName = results.areas{a};
  areaResult.d2 = d2Vec;
  areaResult.proportions = bhvProportions;
  areaResult.labelSetNames = cell(1, numel(behaviorLabelSets));
  areaResult.rPearson = nan(1, numel(behaviorLabelSets));
  areaResult.nValidWindows = zeros(1, numel(behaviorLabelSets));

  for s = 1:numel(behaviorLabelSets)
    areaResult.labelSetNames{s} = behaviorLabelSets{s}.name;
    [d2Plot, propPlot, validMask] = align_window_vectors(d2Vec, bhvProportions{s}, nWindows);
    areaResult.nValidWindows(s) = sum(validMask);
    if any(validMask)
      areaResult.rPearson(s) = pearson_r(propPlot(validMask), d2Plot(validMask));
    end
  end

  corrResult.byArea{end + 1} = areaResult; %#ok<AGROW>
  corrResult.areas{end + 1} = results.areas{a}; %#ok<AGROW>
end
end

function print_d2_behavior_label_set_correlations(corrResult, d2Window)
% PRINT_D2_BEHAVIOR_LABEL_SET_CORRELATIONS - Command-window summary per label set

fprintf('Behavior label-set proportions per %.0f s d2 window\n', d2Window);
if isempty(corrResult.areas)
  fprintf('  No areas with d2 window data.\n');
  return;
end
for a = 1:numel(corrResult.byArea)
  areaResult = corrResult.byArea{a};
  for s = 1:numel(areaResult.labelSetNames)
    if areaResult.nValidWindows(s) == 0
      fprintf('  %s | %s: no data\n', areaResult.areaName, areaResult.labelSetNames{s});
      continue;
    end
    fprintf('  %s | %s: r=%.3f, n=%d\n', areaResult.areaName, ...
      areaResult.labelSetNames{s}, areaResult.rPearson(s), areaResult.nValidWindows(s));
  end
end
end

function fig = plot_d2_vs_behavior_label_sets(results, behaviorData, collectStart, ...
    behaviorLabelSets, d2Window, useLog10D2, useRawLabels, sessionName, plotConfig)
% PLOT_D2_VS_BEHAVIOR_LABEL_SETS - One scatter per behavior label set

if nargin < 9 || isempty(plotConfig)
  plotConfig = fill_default_behavior_plot_config();
end

plotColors = manuscript_plot_colors();
numSets = numel(behaviorLabelSets);
numAreas = numel(results.areas);
fig = figure('Color', 'w', 'Position', [120 120 380 * numSets 420], ...
  'Name', 'd2 vs behavior proportions');
tileLayout = tiledlayout(fig, numAreas, numSets, 'TileSpacing', 'compact', 'Padding', 'compact');
if useLog10D2
  d2YLabel = 'log_{10}(d2)';
  labelInterpreter = 'tex';
else
  d2YLabel = 'd2';
  labelInterpreter = 'none';
end

for a = 1:numAreas
  if a > numel(results.startS) || isempty(results.startS{a})
    continue;
  end
  centerRel = results.startS{a}(:);
  d2Vec = results.d2{a}(:);
  nWindows = min(numel(d2Vec), numel(centerRel));
  if nWindows == 0
    warning('No windows for area %s; skipping behavior scatters.', results.areas{a});
    continue;
  end
  centerRel = centerRel(1:nWindows);
  d2Vec = d2Vec(1:nWindows);
  if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
  end

  bhvProportions = compute_behavior_proportions_by_set( ...
    behaviorData, centerRel, collectStart, d2Window, behaviorLabelSets, useRawLabels);

  for s = 1:numSets
    ax = nexttile(tileLayout);
    hold(ax, 'on');

    [d2Plot, propPlot, validMask] = align_window_vectors(d2Vec, bhvProportions{s}, nWindows);
    if ~any(validMask)
      apply_behavior_axes_style(ax, plotConfig, 'Behavior proportion', d2YLabel, ...
        sprintf('%s — %s (no data)', results.areas{a}, behaviorLabelSets{s}.name), ...
        labelInterpreter);
      hold(ax, 'off');
      continue;
    end

    xVals = propPlot(validMask);
    yVals = d2Plot(validMask);
    scatter_manuscript_open(ax, xVals, yVals, plotConfig, plotColors.data);
    add_manuscript_scatter_trendline(ax, xVals, yVals, plotConfig);

    rVal = pearson_r(xVals, yVals);
    titleText = sprintf('%s | r=%.3f, n=%d', behaviorLabelSets{s}.name, rVal, sum(validMask));
    if s == 1
      apply_behavior_axes_style(ax, plotConfig, 'Behavior proportion', d2YLabel, ...
        titleText, labelInterpreter);
    else
      apply_behavior_axes_style(ax, plotConfig, 'Behavior proportion', '', ...
        titleText, labelInterpreter);
    end
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
sgtitle(tileLayout, sprintf('%s — d2 vs behavior proportions | %s | %.0fs windows', ...
  sessionName, rowTitle, d2Window), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function bhvProportions = compute_behavior_proportions_by_set( ...
    behaviorData, centerTimesRel, collectStart, winSize, behaviorLabelSets, useRawLabels)
% COMPUTE_BEHAVIOR_PROPORTIONS_BY_SET - Per-window proportion for each label set

if nargin < 6 || isempty(useRawLabels)
  useRawLabels = true;
end

numSets = numel(behaviorLabelSets);
numWindows = numel(centerTimesRel);
bhvProportions = cell(1, numSets);
for s = 1:numSets
  bhvProportions{s} = nan(numWindows, 1);
end

if ~isfield(behaviorData, 'timeAxisSec') || isempty(behaviorData.timeAxisSec)
  return;
end
if useRawLabels && isfield(behaviorData, 'labelsRaw') && ~isempty(behaviorData.labelsRaw)
  labels = behaviorData.labelsRaw(:);
else
  labels = behaviorData.labels(:);
end
timeAxisSec = behaviorData.timeAxisSec(:);
labels(labels < 0) = nan;

for w = 1:numWindows
  centerAbs = collectStart + centerTimesRel(w);
  winStartAbs = centerAbs - winSize / 2;
  winEndAbs = centerAbs + winSize / 2;
  inWindow = timeAxisSec >= winStartAbs & timeAxisSec < winEndAbs;
  windowBhvID = labels(inWindow);
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

function corrResult = build_d2_behavior_feature_correlation(results, featureVec, ...
    collectStart, d2Window, useLog10D2, featureName)
% BUILD_D2_BEHAVIOR_FEATURE_CORRELATION - Pearson r between d2 and behavior feature

featureVec = featureVec(:);
corrResult = struct();
corrResult.featureName = featureName;
corrResult.d2Window = d2Window;
corrResult.areas = {};
corrResult.byArea = {};

for a = 1:numel(results.areas)
  if a > numel(results.startS) || isempty(results.startS{a})
    continue;
  end

  d2Vec = results.d2{a}(:);
  nWindows = min(numel(d2Vec), numel(featureVec));
  if nWindows == 0
    continue;
  end

  d2Vec = d2Vec(1:nWindows);
  featVec = featureVec(1:nWindows);
  if useLog10D2
    d2Vec = log10_safe_numeric(d2Vec);
  end

  validMask = isfinite(d2Vec) & isfinite(featVec);
  areaResult = struct();
  areaResult.areaName = results.areas{a};
  areaResult.d2 = d2Vec;
  areaResult.feature = featVec;
  areaResult.validMask = validMask;
  areaResult.nWindows = nWindows;
  areaResult.nValidWindows = sum(validMask);
  areaResult.rPearson = pearson_r(d2Vec(validMask), featVec(validMask));
  corrResult.byArea{end + 1} = areaResult; %#ok<AGROW>
  corrResult.areas{end + 1} = results.areas{a}; %#ok<AGROW>
end
end

function print_d2_behavior_correlation(corrResult, featureLabel, d2Window)
% PRINT_D2_BEHAVIOR_CORRELATION - Command-window summary per area

fprintf('%s per %.0f s d2 window\n', featureLabel, d2Window);
if isempty(corrResult.areas)
  fprintf('  No areas with d2 window data.\n');
  return;
end
for a = 1:numel(corrResult.byArea)
  areaResult = corrResult.byArea{a};
  if areaResult.nValidWindows == 0
    fprintf('  %s: no windows with finite d2 and %s\n', areaResult.areaName, corrResult.featureName);
    continue;
  end
  featVals = areaResult.feature(areaResult.validMask);
  fprintf('  %s: r=%.3f, n=%d windows (%s %.3f-%.3f, mean %.3f)\n', ...
    areaResult.areaName, areaResult.rPearson, areaResult.nValidWindows, ...
    corrResult.featureName, min(featVals), max(featVals), mean(featVals));
end
end

function fig = plot_d2_behavior_correlation(corrResult, sessionName, d2Window, ...
    useLog10D2, xLabelText, plotConfig)
% PLOT_D2_BEHAVIOR_CORRELATION - Scatter of behavior feature vs d2 per area

plotConfig = fill_default_behavior_plot_config(plotConfig);
numAreas = numel(corrResult.areas);
if numAreas == 0
  warning('spontaneous_criticality_metrics_behaviors:NoD2BehaviorData', ...
    'No areas available for d2-behavior correlation plot.');
  fig = gobjects(0);
  return;
end

if useLog10D2
  d2YLabel = 'log_{10}(d2)';
  labelInterpreter = 'tex';
else
  d2YLabel = 'd2';
  labelInterpreter = 'none';
end

fig = figure('Color', 'w', 'Name', sprintf('d2 vs %s', corrResult.featureName), ...
  'Position', [120 120 420 * numAreas 340]);
tileLayout = tiledlayout(fig, 1, numAreas, 'TileSpacing', 'compact', 'Padding', 'compact');
plotColors = manuscript_plot_colors();

for a = 1:numAreas
  areaResult = corrResult.byArea{a};
  ax = nexttile(tileLayout);
  hold(ax, 'on');

  validMask = areaResult.validMask;
  if ~any(validMask)
    title(ax, sprintf('%s (no finite data)', areaResult.areaName), 'Interpreter', 'none');
    apply_behavior_axes_style(ax, plotConfig);
    hold(ax, 'off');
    continue;
  end

  xVals = areaResult.feature(validMask);
  yVals = areaResult.d2(validMask);
  scatter_manuscript_filled(ax, xVals, yVals, plotConfig, plotColors.data);
  add_manuscript_scatter_trendline(ax, xVals, yVals, plotConfig);

  rVal = areaResult.rPearson;
  apply_behavior_axes_style(ax, plotConfig, xLabelText, d2YLabel, ...
    sprintf('%s | r=%.3f, n=%d', areaResult.areaName, rVal, sum(validMask)), ...
    labelInterpreter);
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayout, sprintf('%s — d2 vs %s | %.0fs windows', ...
  sessionName, xLabelText, d2Window), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function apply_behavior_axes_style(ax, plotConfig, xLabelText, yLabelText, titleText, textInterpreter)
% APPLY_BEHAVIOR_AXES_STYLE - Manuscript-style axis formatting

apply_manuscript_axes_style(ax, plotConfig, xLabelText, yLabelText, titleText, textInterpreter);
end

%% -------------------------------------------------------------------------
%% Low-level helpers
%% -------------------------------------------------------------------------

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with nonpositive/nonfinite -> NaN

y = nan(size(x));
ok = isfinite(x) & x > 0;
y(ok) = log10(x(ok));
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

function label = format_areas_label(areaNames)
% FORMAT_AREAS_LABEL - Filesystem-safe label from area name(s)

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

function areaNames = normalize_results_area_list(areasField)
% NORMALIZE_RESULTS_AREA_LIST - Flatten heterogeneous area name fields to cellstr

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
% FILTER_AR_RESULTS_TO_BRAIN_AREA - Keep one area from criticality_ar_analysis output

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
