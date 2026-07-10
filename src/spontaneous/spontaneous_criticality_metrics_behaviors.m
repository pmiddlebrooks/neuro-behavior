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
%       .entropyGroups - Cell of structs (.name, .sourceIDs, optional .code)
%                         for entropy/switch-rate label stream
%       .useEntropyGroups - If true, entropy/switch use entropyGroups recoding;
%                           if false, use smoothed raw labels (default false)
%       .behaviorPropGroups - Cell of structs for d2 vs behavior proportion
%                             scatters (matched on smoothed raw labels)
%       .runD2BehaviorGroupCorrelation - d2 vs behaviorPropGroups (default true)
%     d2:
%       .d2Window, .useLog10D2, .normalizeD2, .nShufflesD2
%       .runD2EntropyCorrelation   - Correlate d2 with label entropy (default true)
%       .runD2SwitchRateCorrelation - Correlate d2 with switch rate (default true)
%       .d2BehaviorNonzeroMin - Fixed min behavior x for intensity (Spearman) part
%                              of hurdle stats on group proportions (default 0.1)
%       .d2BehaviorIntensityMaxFraction - For entropy and switch rate, intensity
%                              cutoff is this fraction of the session max x among
%                              valid windows (default 0.1 = 10% of max)
%     Output:
%       .makePlots, .saveFigure, .saveResults, .outputDir, .plotConfig
%
% Goal:
%   For each non-overlapping d2 window, compute behavior switch rate and label-
%   distribution entropy from frame-wise behavior labels. Correlate d2 with each
%   feature per brain area (all windows). Optionally correlate d2 with behavior
%   group proportions (Inv, Groom, Itch, Loco, etc. by default). Future
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
out.d2BehaviorGroupCorrelation = [];
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
      behaviorPerWindow.entropy, collectStart, opts.d2Window, opts.useLog10D2, ...
      'entropy', opts.d2BehaviorIntensityMaxFraction);
    out.d2EntropyCorrelation = d2EntropyCorr;
    print_d2_behavior_correlation(d2EntropyCorr, 'Label entropy (bits)', opts.d2Window);
  end

  if opts.runD2SwitchRateCorrelation
    fprintf('\n--- d2 vs behavior switch rate (all windows) ---\n');
    d2SwitchCorr = build_d2_behavior_feature_correlation(resultsD2, ...
      behaviorPerWindow.switchRate, collectStart, opts.d2Window, opts.useLog10D2, ...
      'switchRate', opts.d2BehaviorIntensityMaxFraction);
    out.d2SwitchRateCorrelation = d2SwitchCorr;
    print_d2_behavior_correlation(d2SwitchCorr, 'Switch rate (switches/s)', opts.d2Window);
  end

  if opts.makePlots && (opts.runD2EntropyCorrelation || opts.runD2SwitchRateCorrelation)
    entropyCorr = [];
    switchCorr = [];
    if opts.runD2EntropyCorrelation
      entropyCorr = out.d2EntropyCorrelation;
    end
    if opts.runD2SwitchRateCorrelation
      switchCorr = out.d2SwitchRateCorrelation;
    end
    out.figHandles.d2BehaviorFeatures = plot_d2_behavior_features_combined( ...
      entropyCorr, switchCorr, sessionName, opts.d2Window, opts.useLog10D2, plotConfig, ...
      behavior_plot_uses_combined_labels(opts, false));
  end

  if opts.runD2BehaviorGroupCorrelation && ~isempty(opts.behaviorPropGroups)
    fprintf('\n--- d2 vs behavior group proportions ---\n');
    groupCorr = build_d2_behavior_group_correlation(resultsD2, behaviorData, ...
      collectStart, opts.d2Window, opts.useLog10D2, opts.behaviorPropGroups, ...
      opts.d2BehaviorNonzeroMin);
    out.d2BehaviorGroupCorrelation = groupCorr;
    print_d2_behavior_group_correlations(groupCorr, opts.d2Window);
    if opts.makePlots
      out.figHandles.d2BehaviorGroups = plot_d2_vs_behavior_groups( ...
        groupCorr, sessionName, opts.d2Window, opts.useLog10D2, plotConfig, ...
        behavior_plot_uses_combined_labels(opts, true));
    end
  end
end

% Legacy output field name for saved scripts/results.
if ~isempty(out.d2BehaviorGroupCorrelation)
  out.d2BehaviorLabelSetCorrelation = out.d2BehaviorGroupCorrelation;
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
    plotSuffix = '';
    if strcmp(figName, 'd2BehaviorFeatures')
      plotSuffix = entropy_groups_filename_tag(opts.useEntropyGroups);
    elseif strcmp(figName, 'd2BehaviorGroups') ...
        && behavior_plot_uses_combined_labels(opts, true)
      plotSuffix = combined_labels_filename_tag(true);
    end
    plotBase = fullfile(saveDir, sprintf('%s_%s%s', baseName, figName, plotSuffix));
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
  opts.useSubsampling = false;
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
opts = migrate_legacy_behavior_opts(opts);
if ~isfield(opts, 'entropyGroups') || isempty(opts.entropyGroups)
  opts.entropyGroups = default_spontaneous_entropy_groups();
end
if ~isfield(opts, 'behaviorPropGroups') || isempty(opts.behaviorPropGroups)
  opts.behaviorPropGroups = default_spontaneous_behavior_prop_groups();
end
opts.entropyGroups = normalize_behavior_groups(opts.entropyGroups);
opts.behaviorPropGroups = normalize_behavior_groups(opts.behaviorPropGroups);
if isstruct(opts.entropyGroups)
  opts.entropyGroups = {opts.entropyGroups};
end
if isstruct(opts.behaviorPropGroups)
  opts.behaviorPropGroups = {opts.behaviorPropGroups};
end
if ~isfield(opts, 'useEntropyGroups') || isempty(opts.useEntropyGroups)
  opts.useEntropyGroups = false;
end
if ~isfield(opts, 'runD2BehaviorGroupCorrelation') || isempty(opts.runD2BehaviorGroupCorrelation)
  opts.runD2BehaviorGroupCorrelation = true;
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
if ~isfield(opts, 'd2BehaviorNonzeroMin') || isempty(opts.d2BehaviorNonzeroMin)
  opts.d2BehaviorNonzeroMin = 0.1;
end
if ~isfield(opts, 'd2BehaviorIntensityMaxFraction') || isempty(opts.d2BehaviorIntensityMaxFraction)
  opts.d2BehaviorIntensityMaxFraction = 0.1;
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

function groups = default_spontaneous_entropy_groups()
% DEFAULT_SPONTANEOUS_ENTROPY_GROUPS - Coarse combos for entropy/switch rate
%
% Goal:
%   Super-category grouping (behavior_vs_d2 style) when useEntropyGroups is true.

groups = {
  struct('name', 'Inv', 'sourceIDs', [0 1 2])
  struct('name', 'GroomItch', 'sourceIDs', 5:12)
  struct('name', 'Loco', 'sourceIDs', [13:15])
  };
groups = normalize_behavior_groups(groups);
end

function groups = default_spontaneous_behavior_prop_groups()
% DEFAULT_SPONTANEOUS_BEHAVIOR_PROP_GROUPS - Groups for proportion scatters
%
% Goal:
%   Finer behavior categories for d2 vs group proportion analyses on raw IDs.

groups = {
  struct('name', 'Inv', 'sourceIDs', [0 1 2])
  struct('name', 'Groom', 'sourceIDs', 5:10)
  struct('name', 'Itch', 'sourceIDs', [11 12])
  struct('name', 'Loco', 'sourceIDs', [13:15])
  };
groups = normalize_behavior_groups(groups);
end

function opts = migrate_legacy_behavior_opts(opts)
% MIGRATE_LEGACY_BEHAVIOR_OPTS - Map deprecated behavior opts to split groups

if (~isfield(opts, 'behaviorPropGroups') || isempty(opts.behaviorPropGroups)) ...
    && isfield(opts, 'behaviorLabelSets') && ~isempty(opts.behaviorLabelSets)
  opts.behaviorPropGroups = label_sets_to_behavior_groups(opts.behaviorLabelSets);
end
if (~isfield(opts, 'entropyGroups') || isempty(opts.entropyGroups)) ...
    && isfield(opts, 'behaviorRecodingRules') && ~isempty(opts.behaviorRecodingRules)
  opts.entropyGroups = recoding_rules_to_behavior_groups(opts.behaviorRecodingRules);
end
if (~isfield(opts, 'entropyGroups') || isempty(opts.entropyGroups)) ...
    && (~isfield(opts, 'behaviorPropGroups') || isempty(opts.behaviorPropGroups)) ...
    && isfield(opts, 'behaviorGroups') && ~isempty(opts.behaviorGroups)
  opts.entropyGroups = opts.behaviorGroups;
  opts.behaviorPropGroups = opts.behaviorGroups;
elseif (~isfield(opts, 'entropyGroups') || isempty(opts.entropyGroups)) ...
    && isfield(opts, 'behaviorGroups') && ~isempty(opts.behaviorGroups)
  opts.entropyGroups = opts.behaviorGroups;
elseif (~isfield(opts, 'behaviorPropGroups') || isempty(opts.behaviorPropGroups)) ...
    && isfield(opts, 'behaviorGroups') && ~isempty(opts.behaviorGroups)
  opts.behaviorPropGroups = opts.behaviorGroups;
end
if (~isfield(opts, 'useEntropyGroups') || isempty(opts.useEntropyGroups)) ...
    && isfield(opts, 'useBehaviorGroups') && ~isempty(opts.useBehaviorGroups)
  opts.useEntropyGroups = opts.useBehaviorGroups;
elseif (~isfield(opts, 'useEntropyGroups') || isempty(opts.useEntropyGroups)) ...
    && isfield(opts, 'useBehaviorRecoding') && ~isempty(opts.useBehaviorRecoding)
  opts.useEntropyGroups = opts.useBehaviorRecoding;
end
if (~isfield(opts, 'runD2BehaviorGroupCorrelation') ...
    || isempty(opts.runD2BehaviorGroupCorrelation)) ...
    && isfield(opts, 'runD2BehaviorLabelSetCorrelation') ...
    && ~isempty(opts.runD2BehaviorLabelSetCorrelation)
  opts.runD2BehaviorGroupCorrelation = opts.runD2BehaviorLabelSetCorrelation;
end
end

function groups = label_sets_to_behavior_groups(labelSets)
% LABEL_SETS_TO_BEHAVIOR_GROUPS - Convert legacy label-set structs

if isstruct(labelSets)
  labelSets = {labelSets};
end
groups = cell(1, numel(labelSets));
for iGroup = 1:numel(labelSets)
  labelSet = labelSets{iGroup};
  groups{iGroup} = struct('name', labelSet.name);
  if isfield(labelSet, 'numeratorIDs')
    groups{iGroup}.sourceIDs = labelSet.numeratorIDs;
  else
    groups{iGroup}.sourceIDs = [];
  end
  if isfield(labelSet, 'denominatorIDs') && ~isempty(labelSet.denominatorIDs)
    groups{iGroup}.denominatorIDs = labelSet.denominatorIDs;
  end
end
end

function groups = recoding_rules_to_behavior_groups(recodingRules)
% RECODING_RULES_TO_BEHAVIOR_GROUPS - Convert legacy Nx2 recoding rules

if isempty(recodingRules)
  groups = {};
  return;
end
if ~iscell(recodingRules) || size(recodingRules, 2) ~= 2
  error('behaviorRecodingRules must be an Nx2 cell array: { [oldLabels], newLabel }.');
end
groups = cell(size(recodingRules, 1), 1);
for iRule = 1:size(recodingRules, 1)
  oldLabels = recodingRules{iRule, 1};
  newLabel = recodingRules{iRule, 2};
  groups{iRule} = struct( ...
    'name', sprintf('Group%d', newLabel), ...
    'sourceIDs', oldLabels, ...
    'code', newLabel);
end
groups = normalize_behavior_groups(groups);
end

function groups = normalize_behavior_groups(groups)
% NORMALIZE_BEHAVIOR_GROUPS - Validate groups and assign sequential codes

if isstruct(groups)
  groups = {groups};
end
if isempty(groups)
  return;
end
for iGroup = 1:numel(groups)
  group = groups{iGroup};
  if ~isstruct(group) || ~isfield(group, 'name') || isempty(group.name)
    error('Each behavior group must be a struct with a non-empty .name field.');
  end
  if ~isfield(group, 'sourceIDs') || isempty(group.sourceIDs)
    error('Behavior group "%s" must define .sourceIDs.', group.name);
  end
  group.sourceIDs = group.sourceIDs(:)';
  if ~isfield(group, 'code') || isempty(group.code)
    group.code = iGroup;
  end
  groups{iGroup} = group;
end
end

function recodingRules = behavior_groups_to_recoding_rules(behaviorGroups)
% BEHAVIOR_GROUPS_TO_RECODING_RULES - Frame recoding rules from behaviorGroups

behaviorGroups = normalize_behavior_groups(behaviorGroups);
recodingRules = cell(numel(behaviorGroups), 2);
for iGroup = 1:numel(behaviorGroups)
  group = behaviorGroups{iGroup};
  recodingRules{iGroup, 1} = group.sourceIDs;
  recodingRules{iGroup, 2} = group.code;
end
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
  labelsSmooth = behavior_label_smoothing(labelsRaw, smoothOpts);
else
  labelsSmooth = labelsRaw;
end

if opts.useEntropyGroups
  labelsEntropy = recode_behavior_labels(labelsSmooth, ...
    behavior_groups_to_recoding_rules(opts.entropyGroups));
else
  labelsEntropy = labelsSmooth;
end

behaviorData = struct();
behaviorData.labels = labelsSmooth;
behaviorData.labelsEntropy = labelsEntropy;
behaviorData.labelsRaw = labelsRaw;
behaviorData.entropyGroups = opts.entropyGroups;
behaviorData.behaviorPropGroups = opts.behaviorPropGroups;
behaviorData.useEntropyGroups = opts.useEntropyGroups;
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
if isfield(behaviorData, 'labelsEntropy')
  behaviorData.labelsEntropy = behaviorData.labelsEntropy(inCollect);
end
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
if isfield(behaviorData, 'labelsEntropy') && ~isempty(behaviorData.labelsEntropy)
  labels = behaviorData.labelsEntropy;
else
  labels = behaviorData.labels;
end

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

function corrResult = build_d2_behavior_group_correlation(results, behaviorData, ...
    collectStart, d2Window, useLog10D2, behaviorPropGroups, nonzeroMin)
% BUILD_D2_BEHAVIOR_GROUP_CORRELATION - Two-part hurdle stats per behavior group
%
% Variables:
%   results, behaviorData, collectStart, d2Window, useLog10D2, behaviorPropGroups
%   nonzeroMin - Min proportion for Spearman intensity part (x > nonzeroMin)
%
% Goal:
%   Part A: Wilcoxon rank-sum on d2 for windows with x=0 vs x>0.
%   Part B: Spearman rho between d2 and x for windows with x > nonzeroMin.

if nargin < 7 || isempty(nonzeroMin)
  nonzeroMin = 0.1;
end

behaviorPropGroups = normalize_behavior_groups(behaviorPropGroups);
corrResult = struct();
corrResult.behaviorPropGroups = behaviorPropGroups;
corrResult.d2Window = d2Window;
corrResult.nonzeroMin = nonzeroMin;
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

  bhvProportions = compute_behavior_proportions_by_group( ...
    behaviorData, centerRel, collectStart, d2Window, behaviorPropGroups);

  areaResult = struct();
  areaResult.areaName = results.areas{a};
  areaResult.d2 = d2Vec;
  areaResult.proportions = bhvProportions;
  areaResult.groupNames = cell(1, numel(behaviorPropGroups));
  areaResult.nValidWindows = zeros(1, numel(behaviorPropGroups));
  areaResult.hurdle = repmat(empty_d2_behavior_hurdle_stats(), 1, numel(behaviorPropGroups));

  for g = 1:numel(behaviorPropGroups)
    areaResult.groupNames{g} = behaviorPropGroups{g}.name;
    [d2Plot, propPlot, validMask] = align_window_vectors(d2Vec, bhvProportions{g}, nWindows);
    areaResult.nValidWindows(g) = sum(validMask);
    if areaResult.nValidWindows(g) >= 1
      areaResult.hurdle(g) = compute_d2_behavior_hurdle_stats( ...
        propPlot(validMask), d2Plot(validMask), nonzeroMin);
    end
  end

  corrResult.byArea{end + 1} = areaResult; %#ok<AGROW>
  corrResult.areas{end + 1} = results.areas{a}; %#ok<AGROW>
end
end

function print_d2_behavior_group_correlations(corrResult, d2Window)
% PRINT_D2_BEHAVIOR_GROUP_CORRELATIONS - Command-window hurdle summary

fprintf('Behavior group proportions per %.0f s d2 window', d2Window);
if isfield(corrResult, 'nonzeroMin') && isfinite(corrResult.nonzeroMin)
  fprintf(' (intensity cutoff x > %.2f)', corrResult.nonzeroMin);
end
fprintf('\n');
if isempty(corrResult.areas)
  fprintf('  No areas with d2 window data.\n');
  return;
end
for a = 1:numel(corrResult.byArea)
  areaResult = corrResult.byArea{a};
  for g = 1:numel(areaResult.groupNames)
    if areaResult.nValidWindows(g) == 0
      fprintf('  %s | %s: no data\n', areaResult.areaName, areaResult.groupNames{g});
      continue;
    end
    print_d2_behavior_hurdle_line(sprintf('%s | %s', areaResult.areaName, ...
      areaResult.groupNames{g}), areaResult.hurdle(g));
  end
end
end

function fig = plot_d2_vs_behavior_groups(groupCorr, sessionName, d2Window, ...
    useLog10D2, plotConfig, useCombinedLabels)
% PLOT_D2_VS_BEHAVIOR_GROUPS - One scatter per behavior group

if nargin < 5 || isempty(plotConfig)
  plotConfig = fill_default_behavior_plot_config();
end
if nargin < 6 || isempty(useCombinedLabels)
  useCombinedLabels = false;
end

plotColors = manuscript_plot_colors();
behaviorPropGroups = groupCorr.behaviorPropGroups;
numGroups = numel(behaviorPropGroups);
numAreas = numel(groupCorr.areas);
fig = figure('Color', 'w', 'Position', [120 120 380 * numGroups 420], ...
  'Name', 'd2 vs behavior proportions');
tileLayout = tiledlayout(fig, numAreas, numGroups, 'TileSpacing', 'compact', 'Padding', 'compact');
if useLog10D2
  d2YLabel = 'log_{10}(d2)';
  labelInterpreter = 'tex';
else
  d2YLabel = 'd2';
  labelInterpreter = 'none';
end

for a = 1:numAreas
  areaResult = groupCorr.byArea{a};
  for g = 1:numGroups
    ax = nexttile(tileLayout);
    hold(ax, 'on');

    [d2Plot, propPlot, validMask] = align_window_vectors( ...
      areaResult.d2, areaResult.proportions{g}, numel(areaResult.d2));
    if ~any(validMask)
      apply_behavior_axes_style(ax, plotConfig, 'Behavior proportion', d2YLabel, ...
        sprintf('%s (no data)', behaviorPropGroups{g}.name), labelInterpreter);
      hold(ax, 'off');
      continue;
    end

    xVals = propPlot(validMask);
    yVals = d2Plot(validMask);
    scatter_manuscript_filled(ax, xVals, yVals, plotConfig, plotColors.data);
    add_behavior_nonzero_cutoff_line(ax, areaResult.hurdle(g).effectiveNonzeroMin);
    xlim(ax, [0, 1]);

    if numAreas > 1
      add_behavior_area_row_label(ax, areaResult.areaName, plotConfig);
    end
    if g == 1
      apply_behavior_axes_style(ax, plotConfig, 'Behavior proportion', d2YLabel, ...
        behaviorPropGroups{g}.name, labelInterpreter);
    else
      apply_behavior_axes_style(ax, plotConfig, 'Behavior proportion', '', ...
        behaviorPropGroups{g}.name, labelInterpreter);
    end
    grid(ax, 'on');
    add_hurdle_stats_textbox(ax, xVals, yVals, areaResult.hurdle(g), plotConfig, numAreas > 1);
    hold(ax, 'off');
  end
end

if numAreas == 1
  rowTitle = groupCorr.areas{1};
else
  rowTitle = strjoin(groupCorr.areas, ', ');
end
sgtitle(tileLayout, sprintf('%s — d2 vs behavior proportions%s | %s | %.0fs windows', ...
  sessionName, combined_labels_title_suffix(useCombinedLabels), rowTitle, d2Window), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function bhvProportions = compute_behavior_proportions_by_group( ...
    behaviorData, centerTimesRel, collectStart, winSize, behaviorPropGroups)
% COMPUTE_BEHAVIOR_PROPORTIONS_BY_GROUP - Per-window proportion for each group
%
% Variables:
%   behaviorData, centerTimesRel, collectStart, winSize, behaviorPropGroups
%
% Goal:
%   Fraction of window frames in each behaviorPropGroups category using smoothed
%   raw labels (behaviorData.labels) and each group's sourceIDs.

behaviorPropGroups = normalize_behavior_groups(behaviorPropGroups);
numGroups = numel(behaviorPropGroups);
numWindows = numel(centerTimesRel);
bhvProportions = cell(1, numGroups);
for g = 1:numGroups
  bhvProportions{g} = nan(numWindows, 1);
end

if ~isfield(behaviorData, 'timeAxisSec') || isempty(behaviorData.timeAxisSec)
  return;
end
labels = behaviorData.labels(:);
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

  for g = 1:numGroups
    group = behaviorPropGroups{g};
    numerIds = group.sourceIDs(:)';
    if isfield(group, 'denominatorIDs') && ~isempty(group.denominatorIDs)
      denomIds = group.denominatorIDs(:)';
      denomMask = ismember(windowBhvID, denomIds);
      denominatorCount = sum(denomMask);
      if denominatorCount == 0
        continue;
      end
      numeratorCount = sum(ismember(windowBhvID(denomMask), numerIds));
      bhvProportions{g}(w) = numeratorCount / denominatorCount;
    else
      numeratorCount = sum(ismember(windowBhvID, numerIds));
      bhvProportions{g}(w) = numeratorCount / numel(windowBhvID);
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
    collectStart, d2Window, useLog10D2, featureName, intensityMaxFraction)
% BUILD_D2_BEHAVIOR_FEATURE_CORRELATION - Two-part hurdle stats for d2 vs feature
%
% Variables:
%   results, featureVec, collectStart, d2Window, useLog10D2, featureName
%   intensityMaxFraction - Intensity cutoff = fraction * max(x) among valid windows
%
% Goal:
%   Part A: Wilcoxon rank-sum on d2 for x=0 vs x>0 windows.
%   Part B: Spearman rho between d2 and x for x > intensityMaxFraction * max(x).

if nargin < 7 || isempty(intensityMaxFraction)
  intensityMaxFraction = 0.1;
end

featureVec = featureVec(:);
corrResult = struct();
corrResult.featureName = featureName;
corrResult.d2Window = d2Window;
corrResult.intensityMaxFraction = intensityMaxFraction;
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
  if areaResult.nValidWindows >= 1
    areaResult.hurdle = compute_d2_behavior_hurdle_stats( ...
      featVec(validMask), d2Vec(validMask), [], intensityMaxFraction);
  else
    areaResult.hurdle = empty_d2_behavior_hurdle_stats();
  end
  corrResult.byArea{end + 1} = areaResult; %#ok<AGROW>
  corrResult.areas{end + 1} = results.areas{a}; %#ok<AGROW>
end
end

function print_d2_behavior_correlation(corrResult, featureLabel, d2Window)
% PRINT_D2_BEHAVIOR_CORRELATION - Command-window hurdle summary per area

fprintf('%s per %.0f s d2 window', featureLabel, d2Window);
if isfield(corrResult, 'intensityMaxFraction') && isfinite(corrResult.intensityMaxFraction)
  fprintf(' (intensity cutoff x > %.0f%% of max)', 100 * corrResult.intensityMaxFraction);
end
fprintf('\n');
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
  fprintf('  %s (%s %.3f-%.3f, mean %.3f): ', areaResult.areaName, ...
    corrResult.featureName, min(featVals), max(featVals), mean(featVals));
  print_d2_behavior_hurdle_line('', areaResult.hurdle, false);
end
end

function fig = plot_d2_behavior_features_combined(entropyCorr, switchCorr, sessionName, ...
    d2Window, useLog10D2, plotConfig, useCombinedLabels)
% PLOT_D2_BEHAVIOR_FEATURES_COMBINED - Entropy and switch-rate scatters (1 x 2)
%
% Variables:
%   entropyCorr, switchCorr - Outputs of build_d2_behavior_feature_correlation ([] to skip)
%   sessionName, d2Window, useLog10D2, plotConfig
%   useCombinedLabels       - If true, title notes smoothed/recoded labels
%
% Goal:
%   One figure with label entropy (left) and switch rate (right) when both are
%   available; otherwise a single panel. One row per brain area.

if nargin < 7 || isempty(useCombinedLabels)
  useCombinedLabels = false;
end

plotConfig = fill_default_behavior_plot_config(plotConfig);
panels = {};
if ~isempty(entropyCorr) && isstruct(entropyCorr) && ~isempty(entropyCorr.areas)
  panels{end + 1} = struct('corrResult', entropyCorr, 'xLabel', 'Label entropy (bits)'); %#ok<AGROW>
end
if ~isempty(switchCorr) && isstruct(switchCorr) && ~isempty(switchCorr.areas)
  panels{end + 1} = struct('corrResult', switchCorr, 'xLabel', 'Switch rate (switches/s)'); %#ok<AGROW>
end
if isempty(panels)
  warning('spontaneous_criticality_metrics_behaviors:NoD2BehaviorData', ...
    'No behavior-feature correlations available for combined plot.');
  fig = gobjects(0);
  return;
end

numPanels = numel(panels);
numAreas = numel(panels{1}.corrResult.areas);
for p = 2:numPanels
  numAreas = max(numAreas, numel(panels{p}.corrResult.areas));
end

if useLog10D2
  d2YLabel = 'log_{10}(d2)';
  labelInterpreter = 'tex';
else
  d2YLabel = 'd2';
  labelInterpreter = 'none';
end

fig = figure('Color', 'w', 'Name', 'd2 vs behavior entropy and switch rate', ...
  'Position', [120 120 420 * numPanels 340 * max(1, numAreas)]);
tileLayout = tiledlayout(fig, max(1, numAreas), numPanels, ...
  'TileSpacing', 'compact', 'Padding', 'compact');
plotColors = manuscript_plot_colors();

for a = 1:numAreas
  for p = 1:numPanels
    corrResult = panels{p}.corrResult;
    if a > numel(corrResult.byArea)
      continue;
    end
    areaResult = corrResult.byArea{a};
    ax = nexttile(tileLayout);
    hold(ax, 'on');

    validMask = areaResult.validMask;
    if ~any(validMask)
      apply_behavior_axes_style(ax, plotConfig, panels{p}.xLabel, d2YLabel, ...
        '(no finite data)', labelInterpreter);
      hold(ax, 'off');
      continue;
    end

    xVals = areaResult.feature(validMask);
    yVals = areaResult.d2(validMask);
    scatter_manuscript_filled(ax, xVals, yVals, plotConfig, plotColors.data);
    add_behavior_nonzero_cutoff_line(ax, areaResult.hurdle.effectiveNonzeroMin);

    if numAreas > 1
      add_behavior_area_row_label(ax, areaResult.areaName, plotConfig);
    end
    if p == 1
      apply_behavior_axes_style(ax, plotConfig, panels{p}.xLabel, d2YLabel, '', ...
        labelInterpreter);
    else
      apply_behavior_axes_style(ax, plotConfig, panels{p}.xLabel, '', '', ...
        labelInterpreter);
    end
    grid(ax, 'on');
    add_hurdle_stats_textbox(ax, xVals, yVals, areaResult.hurdle, plotConfig, numAreas > 1);
    hold(ax, 'off');
  end
end

featureNames = cellfun(@(panel) panel.xLabel, panels, 'UniformOutput', false);
sgtitle(tileLayout, sprintf('%s — d2 vs %s%s | %.0fs windows', ...
  sessionName, strjoin(featureNames, ' and '), combined_labels_title_suffix(useCombinedLabels), ...
  d2Window), 'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
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
    apply_behavior_axes_style(ax, plotConfig, xLabelText, d2YLabel, ...
      '(no finite data)', labelInterpreter);
    hold(ax, 'off');
    continue;
  end

  xVals = areaResult.feature(validMask);
  yVals = areaResult.d2(validMask);
  scatter_manuscript_filled(ax, xVals, yVals, plotConfig, plotColors.data);
  add_behavior_nonzero_cutoff_line(ax, areaResult.hurdle.effectiveNonzeroMin);

  if numAreas > 1
    add_behavior_area_row_label(ax, areaResult.areaName, plotConfig);
  end
  apply_behavior_axes_style(ax, plotConfig, xLabelText, d2YLabel, '', labelInterpreter);
  grid(ax, 'on');
  add_hurdle_stats_textbox(ax, xVals, yVals, areaResult.hurdle, plotConfig, numAreas > 1);
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

function useCombinedLabels = behavior_plot_uses_combined_labels(opts, forProportions)
% BEHAVIOR_PLOT_USES_COMBINED_LABELS - True when plot uses non-raw B-SOiD labels
%
% Variables:
%   opts            - Behavior options struct
%   forProportions  - If true, tag proportion plots (smoothing only); else
%                     entropy/switch plots (smoothing or entropyGroups)
%
% Goal:
%   Raw labels = unsmoothed IDs from behavior_labels CSV.
%   Entropy/switch combined = smoothing and/or entropyGroups recoding.
%   Proportion combined = temporal smoothing only.

if nargin < 2 || isempty(forProportions)
  forProportions = false;
end
if forProportions
  useCombinedLabels = opts.smoothBehaviorLabels;
else
  useCombinedLabels = opts.smoothBehaviorLabels || opts.useEntropyGroups;
end
end

function suffix = combined_labels_title_suffix(useCombinedLabels)
% COMBINED_LABELS_TITLE_SUFFIX - Figure title tag for non-raw behavior labels

if useCombinedLabels
  suffix = ' | combined';
else
  suffix = '';
end
end

function tag = combined_labels_filename_tag(useCombinedLabels)
% COMBINED_LABELS_FILENAME_TAG - Filesystem tag for non-raw behavior labels

if useCombinedLabels
  tag = '_combined';
else
  tag = '';
end
end

function tag = entropy_groups_filename_tag(useEntropyGroups)
% ENTROPY_GROUPS_FILENAME_TAG - Filename tag for entropy/switch-rate label stream
%
% Variables:
%   useEntropyGroups - If true, entropy/switch used entropyGroups recoding
%
% Goal:
%   Distinguish raw per-label IDs vs grouped label stream in saved figure names.

if useEntropyGroups
  tag = '_GroupedLabels';
else
  tag = '_RawLabels';
end
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

[rVal, ~] = pearson_rp(x, y);
end

function [rVal, pVal] = pearson_rp(x, y)
% PEARSON_RP - Pearson correlation and two-tailed p-value
%
% Variables:
%   x, y - Paired sample vectors
%
% Goal:
%   corr(x, y) returns scalars for two vectors (not a 2x2 matrix). Fall back
%   to a t-test on r when the p-value is unavailable.

rVal = nan;
pVal = nan;
x = x(:);
y = y(:);
n = min(numel(x), numel(y));
if n < 2
  return;
end
x = x(1:n);
y = y(1:n);
valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);
n = numel(x);
if n < 2 || std(x, 0) == 0 || std(y, 0) == 0
  return;
end

[rOut, pOut] = corr(x, y, 'Type', 'Pearson');
if isscalar(rOut)
  rVal = rOut;
  if isscalar(pOut)
    pVal = pOut;
  end
elseif isequal(size(rOut), [2, 2])
  rVal = rOut(1, 2);
  if isequal(size(pOut), [2, 2])
    pVal = pOut(1, 2);
  end
end

if ~isfinite(pVal) && isfinite(rVal) && n >= 3 && abs(rVal) < 1
  tStat = rVal * sqrt((n - 2) / (1 - rVal^2));
  pVal = 2 * tcdf(-abs(tStat), n - 2);
end
end

function pStr = format_correlation_p(pVal)
% FORMAT_CORRELATION_P - Title-friendly p-value string

if ~isfinite(pVal)
  pStr = 'p=n/a';
elseif pVal < 0.001
  pStr = sprintf('p=%.1e', pVal);
else
  pStr = sprintf('p=%.3f', pVal);
end
end

function hurdle = empty_d2_behavior_hurdle_stats()
% EMPTY_D2_BEHAVIOR_HURDLE_STATS - NaN-filled hurdle struct template

hurdle = struct();
hurdle.cutoffMode = '';
hurdle.nonzeroMin = nan;
hurdle.intensityMaxFraction = nan;
hurdle.xMax = nan;
hurdle.effectiveNonzeroMin = nan;
hurdle.nZero = 0;
hurdle.nNonzero = 0;
hurdle.nAboveMin = 0;
hurdle.pPresence = nan;
hurdle.medianD2Zero = nan;
hurdle.medianD2Nonzero = nan;
hurdle.rhoSpearman = nan;
hurdle.pSpearman = nan;
end

function hurdle = compute_d2_behavior_hurdle_stats(x, y, nonzeroMin, intensityMaxFraction)
% COMPUTE_D2_BEHAVIOR_HURDLE_STATS - Two-part hurdle: presence + intensity
%
% Variables:
%   x, y                   - Paired behavior feature (x) and d2 (y)
%   nonzeroMin             - Fixed intensity cutoff when intensityMaxFraction empty
%   intensityMaxFraction   - If set, intensity cutoff = fraction * max(x) among valid
%
% Goal:
%   Part A: Wilcoxon rank-sum on d2 for windows with x == 0 vs x > 0.
%   Part B: Spearman rho between d2 and x for windows above the intensity cutoff.

hurdle = empty_d2_behavior_hurdle_stats();
x = x(:);
y = y(:);
n = min(numel(x), numel(y));
if n < 1
  return;
end
x = x(1:n);
y = y(1:n);
valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);
if isempty(x)
  return;
end

if nargin >= 4 && ~isempty(intensityMaxFraction) && isfinite(intensityMaxFraction) ...
    && intensityMaxFraction > 0
  hurdle.cutoffMode = 'maxFraction';
  hurdle.intensityMaxFraction = intensityMaxFraction;
  hurdle.xMax = max(x);
  if isfinite(hurdle.xMax) && hurdle.xMax > 0
    hurdle.effectiveNonzeroMin = intensityMaxFraction * hurdle.xMax;
  else
    hurdle.effectiveNonzeroMin = 0;
  end
else
  if nargin < 3 || isempty(nonzeroMin)
    nonzeroMin = 0.1;
  end
  hurdle.cutoffMode = 'fixed';
  hurdle.nonzeroMin = nonzeroMin;
  hurdle.effectiveNonzeroMin = nonzeroMin;
end

zeroMask = x == 0;
nonzeroMask = x > 0;
aboveMinMask = x > hurdle.effectiveNonzeroMin;
hurdle.nZero = sum(zeroMask);
hurdle.nNonzero = sum(nonzeroMask);
hurdle.nAboveMin = sum(aboveMinMask);

if hurdle.nZero >= 1 && hurdle.nNonzero >= 1
  d2Zero = y(zeroMask);
  d2Nonzero = y(nonzeroMask);
  hurdle.medianD2Zero = median(d2Zero);
  hurdle.medianD2Nonzero = median(d2Nonzero);
  hurdle.pPresence = ranksum_test_safe(d2Zero, d2Nonzero);
end

if hurdle.nAboveMin >= 2
  [hurdle.rhoSpearman, hurdle.pSpearman] = spearman_rp(x(aboveMinMask), y(aboveMinMask));
end
end

function statText = format_hurdle_stats_box_text(hurdle)
% FORMAT_HURDLE_STATS_BOX_TEXT - Multi-line hurdle stats for in-axes text box

if isfinite(hurdle.rhoSpearman)
  rhoStr = sprintf('ρ=%.2f', hurdle.rhoSpearman);
else
  rhoStr = 'ρ=n/a';
end
statText = sprintf('x=0 vs >0: %s\nx>%.2f: %s %s', ...
  format_correlation_p(hurdle.pPresence), hurdle.effectiveNonzeroMin, ...
  rhoStr, format_correlation_p(hurdle.pSpearman));
end

function add_hurdle_stats_textbox(ax, xVals, yVals, hurdle, plotConfig, reserveTopLeft)
% ADD_HURDLE_STATS_TEXTBOX - Hurdle stats in boxed text at least-crowded corner
%
% Variables:
%   ax, xVals, yVals - Axes and plotted data
%   hurdle           - Output of compute_d2_behavior_hurdle_stats
%   plotConfig       - Manuscript plot config
%   reserveTopLeft   - If true, avoid NW corner (e.g. when area label is there)

if nargin < 6 || isempty(reserveTopLeft)
  reserveTopLeft = false;
end
excludeCornerIdx = 0;
if reserveTopLeft
  excludeCornerIdx = 1;
end
corner = pick_least_crowded_corner(ax, xVals, yVals, excludeCornerIdx);
fontSize = plotConfig.tickLabelFontSize;
if isfield(plotConfig, 'titleFontSize') && ~isempty(plotConfig.titleFontSize)
  fontSize = max(fontSize, plotConfig.titleFontSize - 1);
end
text(ax, corner.x, corner.y, format_hurdle_stats_box_text(hurdle), ...
  'Units', 'normalized', 'VerticalAlignment', corner.vertAlign, ...
  'HorizontalAlignment', corner.horizAlign, 'FontSize', fontSize, ...
  'Interpreter', 'none', 'BackgroundColor', [1, 1, 1], ...
  'EdgeColor', [0.35, 0.35, 0.35], 'Margin', 4, 'Clipping', 'on');
end

function add_behavior_area_row_label(ax, areaName, plotConfig)
% ADD_BEHAVIOR_AREA_ROW_LABEL - Brain-area tag for multi-row behavior figures

text(ax, 0.03, 0.97, areaName, 'Units', 'normalized', ...
  'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
  'FontSize', plotConfig.tickLabelFontSize, 'FontWeight', 'bold', ...
  'Interpreter', 'none', 'Clipping', 'on');
end

function corner = pick_least_crowded_corner(ax, xVals, yVals, excludeCornerIdx)
% PICK_LEAST_CROWDED_CORNER - Corner with fewest scatter points (normalized axes)
%
% Variables:
%   ax               - Target axes
%   xVals, yVals     - Plotted data in data coordinates
%   excludeCornerIdx - Optional 1-4 to skip NW, NE, SW, SE

if nargin < 4 || isempty(excludeCornerIdx)
  excludeCornerIdx = 0;
end
xl = xlim(ax);
yl = ylim(ax);
if ~all(isfinite(xl)) || diff(xl) <= 0
  xPad = 0.05 * max(range(xVals), eps);
  xl = [min(xVals) - xPad, max(xVals) + xPad];
  xlim(ax, xl);
end
if ~all(isfinite(yl)) || diff(yl) <= 0
  yPad = 0.05 * max(range(yVals), eps);
  yl = [min(yVals) - yPad, max(yVals) + yPad];
  ylim(ax, yl);
end
xNorm = (xVals - xl(1)) / diff(xl);
yNorm = (yVals - yl(1)) / diff(yl);
counts = [ ...
  sum(xNorm < 0.5 & yNorm >= 0.5); ...
  sum(xNorm >= 0.5 & yNorm >= 0.5); ...
  sum(xNorm < 0.5 & yNorm < 0.5); ...
  sum(xNorm >= 0.5 & yNorm < 0.5)];
if excludeCornerIdx >= 1 && excludeCornerIdx <= 4
  counts(excludeCornerIdx) = inf;
end
[~, cornerIdx] = min(counts);
cornerSpecs = { ...
  0.03, 0.97, 'left', 'top'; ...
  0.97, 0.97, 'right', 'top'; ...
  0.03, 0.03, 'left', 'bottom'; ...
  0.97, 0.03, 'right', 'bottom'};
corner = struct();
corner.x = cornerSpecs{cornerIdx, 1};
corner.y = cornerSpecs{cornerIdx, 2};
corner.horizAlign = cornerSpecs{cornerIdx, 3};
corner.vertAlign = cornerSpecs{cornerIdx, 4};
end

function print_d2_behavior_hurdle_line(prefix, hurdle, includePrefix)
% PRINT_D2_BEHAVIOR_HURDLE_LINE - Command-window line for hurdle stats
%
% Variables:
%   prefix         - Optional leading label (e.g. area name)
%   hurdle         - Output of compute_d2_behavior_hurdle_stats
%   includePrefix  - If false, omit prefix and leading spaces (default true)

if nargin < 3 || isempty(includePrefix)
  includePrefix = true;
end
if strcmp(hurdle.cutoffMode, 'maxFraction') && isfinite(hurdle.xMax)
  cutoffNote = sprintf('x>%.2f (%.0f%% of max %.3f)', hurdle.effectiveNonzeroMin, ...
    100 * hurdle.intensityMaxFraction, hurdle.xMax);
else
  cutoffNote = sprintf('x>%.2f', hurdle.effectiveNonzeroMin);
end
lineText = sprintf(['presence x=0 vs >0 %s (n0=%d, n>0=%d, ' ...
  'median d2 %.3f vs %.3f) | intensity %s %s %s (n=%d)'], ...
  format_correlation_p(hurdle.pPresence), hurdle.nZero, hurdle.nNonzero, ...
  hurdle.medianD2Zero, hurdle.medianD2Nonzero, cutoffNote, ...
  format_spearman_rho(hurdle.rhoSpearman), format_correlation_p(hurdle.pSpearman), ...
  hurdle.nAboveMin);
if includePrefix && ~isempty(prefix)
  fprintf('  %s: %s\n', prefix, lineText);
else
  fprintf('%s\n', lineText);
end
end

function rhoStr = format_spearman_rho(rhoVal)
% FORMAT_SPEARMAN_RHO - Command-window Spearman rho string

if isfinite(rhoVal)
  rhoStr = sprintf('ρ=%.3f', rhoVal);
else
  rhoStr = 'ρ=n/a';
end
end

function add_behavior_nonzero_cutoff_line(ax, nonzeroMin)
% ADD_BEHAVIOR_NONZERO_CUTOFF_LINE - Vertical marker at intensity cutoff

if ~isfinite(nonzeroMin) || nonzeroMin <= 0
  return;
end
xline(ax, nonzeroMin, '--', 'Color', [0.45, 0.45, 0.45], 'LineWidth', 0.9, ...
  'HandleVisibility', 'off');
end

function [rhoVal, pVal] = spearman_rp(x, y)
% SPEARMAN_RP - Spearman correlation and two-tailed p-value
%
% Variables:
%   x, y - Paired sample vectors
%
% Goal:
%   Rank-based association for the intensity (x > cutoff) subset.

rhoVal = nan;
pVal = nan;
x = x(:);
y = y(:);
n = min(numel(x), numel(y));
if n < 2
  return;
end
x = x(1:n);
y = y(1:n);
valid = isfinite(x) & isfinite(y);
x = x(valid);
y = y(valid);
n = numel(x);
if n < 2 || numel(unique(x)) < 2 || numel(unique(y)) < 2
  return;
end

[rhoOut, pOut] = corr(x, y, 'Type', 'Spearman');
if isscalar(rhoOut)
  rhoVal = rhoOut;
  if isscalar(pOut)
    pVal = pOut;
  end
elseif isequal(size(rhoOut), [2, 2])
  rhoVal = rhoOut(1, 2);
  if isequal(size(pOut), [2, 2])
    pVal = pOut(1, 2);
  end
end
end

function pVal = ranksum_test_safe(x, y)
% RANKSUM_TEST_SAFE - Wilcoxon rank-sum; NaN if either group is empty

pVal = nan;
x = x(isfinite(x));
y = y(isfinite(y));
if numel(x) < 1 || numel(y) < 1
  return;
end
pVal = ranksum(x, y);
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
