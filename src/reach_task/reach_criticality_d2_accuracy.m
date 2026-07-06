function out = reach_criticality_d2_accuracy(sessionName, opts)
% REACH_CRITICALITY_D2_ACCURACY - Pre/post-reach d2 by reach outcome (correct vs error)
%
% Variables:
%   sessionName - Reach session identifier (no .mat extension)
%   opts        - Options struct. Fields:
%     Session loading (neuro_behavior_options overrides):
%       .collectStart, .collectEnd, .minFiringRate, .maxFiringRate,
%       .firingRateCheckTime, .dataSource
%     Window alignment:
%       .d2Window        - Fixed window length for d2 (s; default 10)
%       .preReachBuffer  - Pre: gap between window end and reach onset (ms).
%                          Post: exclusion zone before next reach onset (ms).
%       .postReachBuffer - Pre: exclusion zone after previous reach onset (ms).
%                          Post: gap between reach onset and window start (ms).
%       .windowBuffer    - Margin inside recording bounds (s; default 0.5)
%     d2:
%       .pOrder, .critType, .normalizeD2, .nShuffles, .meanSubtract,
%       .useLog10D2, .useOptimalBinWindowFunction, .binSizeManual,
%       .minSpikesPerBin, .minBinsPerWindow, .pcaFlag
%     Analysis:
%       .brainArea, .brainAreaCombinations, .nMinNeurons
%     Output:
%       .makePlots, .saveFigure, .saveResults, .outputDir, .plotConfig
%
% Goal:
%   Pre-reach: fixed window ending preReachBuffer before reach onset; exclude
%   overlap with [prev reach, prev reach + postReachBuffer].
%   Post-reach: fixed window starting postReachBuffer after reach onset; exclude
%   overlap with [next reach - preReachBuffer, next reach].
%   Compare d2 between correct (reachClass 2/4) and error (1/3) per area
%   (Wilcoxon rank-sum and two-sample t-test). Plot pre-reach (top row) and
%   post-reach (bottom row).
%
% Returns:
%   With no inputs: default options struct.
%   Otherwise: struct with config, preReach/postReach results, figHandles,
%              and optional saved paths.

setup_reach_criticality_d2_accuracy_paths();

if nargin == 0
  out = fill_d2_accuracy_opts_defaults(struct());
  return;
end
if nargin < 1 || isempty(sessionName)
  error('reach_criticality_d2_accuracy:MissingSession', 'sessionName is required.');
end
if nargin < 2 || isempty(opts)
  opts = struct();
end
opts = fill_d2_accuracy_opts_defaults(opts);

sessionType = 'reach';
dataSource = opts.dataSource;
collectStart = opts.collectStart;
collectEnd = opts.collectEnd;

fprintf('\n=== Reach criticality d2 accuracy ===\n');
fprintf('Session: %s\n', sessionName);
fprintf('d2Window=%.2fs | preReachBuffer=%d ms | postReachBuffer=%d ms\n', ...
  opts.d2Window, opts.preReachBuffer, opts.postReachBuffer);

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
  error('reach_criticality_d2_accuracy:NoReaches', ...
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

if ~isempty(opts.brainArea)
  [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
    dataStruct, opts.brainArea, opts.brainAreaCombinations, true);
  if ~areaOk
    error('Brain area "%s" not available in this session.', opts.brainArea);
  end
end

areas = dataStruct.areas;
numAreas = numel(areas);
if isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
  areasToTest = dataStruct.areasToTest;
else
  areasToTest = 1:numAreas;
end

for a = areasToTest
  nUnits = numel(dataStruct.idMatIdx{a});
  if nUnits < opts.nMinNeurons
    warning('reach_criticality_d2_accuracy:TooFewNeurons', ...
      'Area %s has %d neurons (min %d); results may be unreliable.', ...
      areas{a}, nUnits, opts.nMinNeurons);
  end
end

reachStartAll = dataStruct.reachStart(:);
reachInCollect = reachStartAll >= collectStart & reachStartAll <= collectEnd;
reachStart = reachStartAll(reachInCollect);
nReach = numel(reachStart);

reachClass = [];
if isfield(dataStruct, 'reachClass') && ~isempty(dataStruct.reachClass)
  reachClass = dataStruct.reachClass(:);
  if numel(reachClass) == numel(reachStartAll)
    reachClass = reachClass(reachInCollect);
  else
    reachClass = [];
  end
end
if isempty(reachClass)
  error('reach_criticality_d2_accuracy:NoReachClass', ...
    'reachClass required for correct vs error comparison.');
end

isCorrectReach = ismember(reachClass, [2, 4]);
isErrorReach = ismember(reachClass, [1, 3]);
if ~any(isCorrectReach) || ~any(isErrorReach)
  warning('reach_criticality_d2_accuracy:ImbalancedOutcomes', ...
    'Few or no reaches in one outcome class (correct=%d, error=%d).', ...
    sum(isCorrectReach), sum(isErrorReach));
end

if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectStart')
  timeRange = [dataStruct.spikeData.collectStart, dataStruct.spikeData.collectEnd];
else
  timeRange = [collectStart, collectEnd];
end
tRecStart = timeRange(1);
tRecEnd = timeRange(2);

fprintf('Collect window: [%.1f, %.1f] s | reaches: %d (correct=%d, error=%d)\n', ...
  collectStart, collectEnd, nReach, sum(isCorrectReach), sum(isErrorReach));

saveDir = resolve_d2_accuracy_save_dir(dataStruct, sessionName, opts);
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

preReachBufferSec = opts.preReachBuffer / 1000;
postReachBufferSec = opts.postReachBuffer / 1000;

%% Bin sizes
binSize = zeros(1, numAreas);
if opts.useOptimalBinWindowFunction
  for a = areasToTest
    neuronIDs = dataStruct.idLabel{a};
    fr = calculate_firing_rate_from_spikes(dataStruct.spikeTimes, ...
      dataStruct.spikeClusters, neuronIDs, timeRange);
    [binSize(a), ~] = find_optimal_bin_and_window(fr, opts.minSpikesPerBin, opts.minBinsPerWindow);
  end
else
  binSize(:) = opts.binSizeManual;
end

%% Pre-reach window eligibility
[preWinStart, preWinEnd, preWinCenter, preEligible, preExcludeOverlapPrev, ...
  preExcludeRecording] = build_pre_reach_windows(reachStart, opts.d2Window, ...
  preReachBufferSec, postReachBufferSec, opts.windowBuffer, tRecStart, tRecEnd);

fprintf('Pre-reach eligible: %d / %d (recording=%d, prev overlap=%d)\n', ...
  sum(preEligible), nReach, sum(preExcludeRecording), sum(preExcludeOverlapPrev));

%% Post-reach window eligibility
[postWinStart, postWinEnd, postWinCenter, postEligible, postExcludeOverlapNext, ...
  postExcludeRecording] = build_post_reach_windows(reachStart, opts.d2Window, ...
  preReachBufferSec, postReachBufferSec, opts.windowBuffer, tRecStart, tRecEnd);

fprintf('Post-reach eligible: %d / %d (recording=%d, next overlap=%d)\n', ...
  sum(postEligible), nReach, sum(postExcludeRecording), sum(postExcludeOverlapNext));

%% Per-area d2 (pre-reach)
d2PerReachPre = cell(1, numAreas);
d2PerReachNormPre = cell(1, numAreas);
for a = areasToTest
  d2PerReachPre{a} = nan(nReach, 1);
  d2PerReachNormPre{a} = nan(nReach, 1);
end

for a = areasToTest
  fprintf('\nArea %s: pre-reach d2 by outcome...\n', areas{a});
  [d2PerReachPre{a}, d2PerReachNormPre{a}] = compute_aligned_reach_d2_for_area( ...
    dataStruct, a, preEligible, preWinCenter, opts.d2Window, ...
    binSize(a), timeRange, opts);
end

d2AnalysisPre = build_d2_analysis_vectors(d2PerReachPre, d2PerReachNormPre, ...
  areasToTest, numAreas, opts.normalizeD2, opts.useLog10D2);
statsByAreaPre = build_d2_accuracy_stats_by_area(d2AnalysisPre, areas, areasToTest, ...
  preEligible, isCorrectReach, isErrorReach, 'pre-reach');
print_d2_accuracy_stats(statsByAreaPre, opts.useLog10D2, 'Pre-reach');

%% Per-area d2 (post-reach)
d2PerReachPost = cell(1, numAreas);
d2PerReachNormPost = cell(1, numAreas);
for a = areasToTest
  d2PerReachPost{a} = nan(nReach, 1);
  d2PerReachNormPost{a} = nan(nReach, 1);
end

for a = areasToTest
  fprintf('\nArea %s: post-reach d2 by outcome...\n', areas{a});
  [d2PerReachPost{a}, d2PerReachNormPost{a}] = compute_aligned_reach_d2_for_area( ...
    dataStruct, a, postEligible, postWinCenter, opts.d2Window, ...
    binSize(a), timeRange, opts);
end

d2AnalysisPost = build_d2_analysis_vectors(d2PerReachPost, d2PerReachNormPost, ...
  areasToTest, numAreas, opts.normalizeD2, opts.useLog10D2);
statsByAreaPost = build_d2_accuracy_stats_by_area(d2AnalysisPost, areas, areasToTest, ...
  postEligible, isCorrectReach, isErrorReach, 'post-reach');
print_d2_accuracy_stats(statsByAreaPost, opts.useLog10D2, 'Post-reach');

%% Output struct
out = struct();
out.sessionName = sessionName;
out.config = opts;
out.areas = areas;
out.areasToTest = areasToTest;
out.reachStart = reachStart;
out.reachClass = reachClass;
out.isCorrectReach = isCorrectReach;
out.isErrorReach = isErrorReach;
out.binSize = binSize;
out.saveDir = saveDir;
out.figHandles = struct();

out.preReach = struct();
out.preReach.winStartReach = preWinStart;
out.preReach.winEndReach = preWinEnd;
out.preReach.winCenterReach = preWinCenter;
out.preReach.eligibleReach = preEligible;
out.preReach.excludeOverlapPrev = preExcludeOverlapPrev;
out.preReach.excludeRecording = preExcludeRecording;
out.preReach.d2PerReach = d2PerReachPre;
out.preReach.d2PerReachNorm = d2PerReachNormPre;
out.preReach.d2Analysis = d2AnalysisPre;
out.preReach.statsByArea = statsByAreaPre;

out.postReach = struct();
out.postReach.winStartReach = postWinStart;
out.postReach.winEndReach = postWinEnd;
out.postReach.winCenterReach = postWinCenter;
out.postReach.eligibleReach = postEligible;
out.postReach.excludeOverlapNext = postExcludeOverlapNext;
out.postReach.excludeRecording = postExcludeRecording;
out.postReach.d2PerReach = d2PerReachPost;
out.postReach.d2PerReachNorm = d2PerReachNormPost;
out.postReach.d2Analysis = d2AnalysisPost;
out.postReach.statsByArea = statsByAreaPost;

plotConfig = opts.plotConfig;
if opts.makePlots
  out.figHandles.comparison = plot_d2_accuracy_comparison( ...
    statsByAreaPre, statsByAreaPost, sessionName, opts, plotConfig);
end

if opts.saveFigure && opts.makePlots && isgraphics(out.figHandles.comparison)
  areaTag = format_areas_label(opts.brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  plotBase = fullfile(saveDir, sprintf('reach_criticality_d2_accuracy_%s_%s_W%.1f_pre%d_post%d', ...
    sessionName, areaTag, opts.d2Window, opts.preReachBuffer, opts.postReachBuffer));
  exportgraphics(out.figHandles.comparison, [plotBase, '.png'], 'Resolution', 300);
  exportgraphics(out.figHandles.comparison, [plotBase, '.eps'], 'ContentType', 'vector');
  out.savedFigurePng = [plotBase, '.png'];
  fprintf('Saved figure: %s\n', out.savedFigurePng);
end

if opts.saveResults
  resultsPath = fullfile(saveDir, sprintf('reach_criticality_d2_accuracy_W%.1f_pre%d_post%d.mat', ...
    opts.d2Window, opts.preReachBuffer, opts.postReachBuffer));
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

function opts = fill_d2_accuracy_opts_defaults(opts)
% FILL_D2_ACCURACY_OPTS_DEFAULTS - Session-load and analysis option defaults

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

if ~isfield(opts, 'd2Window') || isempty(opts.d2Window)
  opts.d2Window = 10;
end
if ~isfield(opts, 'preReachBuffer') || isempty(opts.preReachBuffer)
  opts.preReachBuffer = 250;
end
if ~isfield(opts, 'postReachBuffer') || isempty(opts.postReachBuffer)
  opts.postReachBuffer = 250;
end
if ~isfield(opts, 'windowBuffer') || isempty(opts.windowBuffer)
  opts.windowBuffer = 0.5;
end

if ~isfield(opts, 'pOrder') || isempty(opts.pOrder)
  opts.pOrder = 10;
end
if ~isfield(opts, 'critType') || isempty(opts.critType)
  opts.critType = 2;
end
if ~isfield(opts, 'normalizeD2') || isempty(opts.normalizeD2)
  opts.normalizeD2 = false;
end
if ~isfield(opts, 'nShuffles') || isempty(opts.nShuffles)
  opts.nShuffles = 10;
end
if ~isfield(opts, 'meanSubtract') || isempty(opts.meanSubtract)
  opts.meanSubtract = false;
end
if ~isfield(opts, 'useLog10D2') || isempty(opts.useLog10D2)
  opts.useLog10D2 = true;
end
if ~isfield(opts, 'useOptimalBinWindowFunction') || isempty(opts.useOptimalBinWindowFunction)
  opts.useOptimalBinWindowFunction = false;
end
if ~isfield(opts, 'binSizeManual') || isempty(opts.binSizeManual)
  opts.binSizeManual = 0.025;
end
if ~isfield(opts, 'minSpikesPerBin') || isempty(opts.minSpikesPerBin)
  opts.minSpikesPerBin = 3;
end
if ~isfield(opts, 'minBinsPerWindow') || isempty(opts.minBinsPerWindow)
  opts.minBinsPerWindow = 1000;
end
if ~isfield(opts, 'pcaFlag') || isempty(opts.pcaFlag)
  opts.pcaFlag = 0;
end

if ~isfield(opts, 'brainArea')
  opts.brainArea = '';
end
if ~isfield(opts, 'brainAreaCombinations') || isempty(opts.brainAreaCombinations)
  opts.brainAreaCombinations = default_manuscript_brain_area_combinations();
end
if ~isfield(opts, 'nMinNeurons') || isempty(opts.nMinNeurons)
  opts.nMinNeurons = 20;
end

if ~isfield(opts, 'makePlots') || isempty(opts.makePlots)
  opts.makePlots = true;
end
if ~isfield(opts, 'saveFigure') || isempty(opts.saveFigure)
  opts.saveFigure = false;
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
opts.plotConfig = fill_default_d2_accuracy_plot_config(opts.plotConfig);
end

function plotConfig = fill_default_d2_accuracy_plot_config(plotConfig)
% FILL_DEFAULT_D2_ACCURACY_PLOT_CONFIG - Axis fonts and line widths

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
if ~isfield(plotConfig, 'errorCapSize') || isempty(plotConfig.errorCapSize)
  plotConfig.errorCapSize = 8;
end
end

function setup_reach_criticality_d2_accuracy_paths()
% SETUP_REACH_CRITICALITY_D2_ACCURACY_PATHS - Add neuro-behavior paths if present

scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_']) || ~isfolder(fullfile(scriptDir, '..', 'reach_task'))
  resolved = which('reach_criticality_d2_accuracy');
  if ~isempty(resolved)
    scriptDir = fileparts(resolved);
  end
end
srcPath = fullfile(scriptDir, '..');
pathDirs = {
  srcPath
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

function saveDir = resolve_d2_accuracy_save_dir(dataStruct, sessionName, opts)
% RESOLVE_D2_ACCURACY_SAVE_DIR - Output directory for figures and .mat results

if ~isempty(opts.outputDir)
  saveDir = opts.outputDir;
elseif isfield(dataStruct, 'saveDir') && ~isempty(dataStruct.saveDir)
  saveDir = dataStruct.saveDir;
else
  paths = get_paths();
  saveDir = fullfile(paths.dropPath, 'reach_task', 'results', ...
    matlab.lang.makeValidName(sessionName));
end
end

%% -------------------------------------------------------------------------
%% Windowing and d2 computation
%% -------------------------------------------------------------------------

function [winStartReach, winEndReach, winCenterReach, eligibleReach, ...
    excludeOverlapPrev, excludeRecording] = build_pre_reach_windows( ...
    reachStart, d2Window, preReachBufferSec, postReachBufferSec, windowBuffer, ...
    tRecStart, tRecEnd)
% BUILD_PRE_REACH_WINDOWS - Fixed pre-reach windows and eligibility masks
%
% Window ends preReachBuffer before aligned reach onset. Exclude overlap with
% [prev reach, prev reach + postReachBuffer].

nReach = numel(reachStart);
winStartReach = nan(nReach, 1);
winEndReach = nan(nReach, 1);
winCenterReach = nan(nReach, 1);
eligibleReach = false(nReach, 1);
excludeOverlapPrev = false(nReach, 1);
excludeRecording = false(nReach, 1);

for r = 1:nReach
  tReach = reachStart(r);
  winEndReach(r) = tReach - preReachBufferSec;
  winStartReach(r) = winEndReach(r) - d2Window;
  winCenterReach(r) = (winStartReach(r) + winEndReach(r)) / 2;

  if winStartReach(r) < tRecStart + windowBuffer || winEndReach(r) > tRecEnd - windowBuffer
    excludeRecording(r) = true;
    continue;
  end

  if r > 1
    tPrevReach = reachStart(r - 1);
    forbiddenEnd = tPrevReach + postReachBufferSec;
    if winStartReach(r) < forbiddenEnd && winEndReach(r) > tPrevReach
      excludeOverlapPrev(r) = true;
      continue;
    end
  end

  eligibleReach(r) = true;
end
end

function [winStartReach, winEndReach, winCenterReach, eligibleReach, ...
    excludeOverlapNext, excludeRecording] = build_post_reach_windows( ...
    reachStart, d2Window, preReachBufferSec, postReachBufferSec, windowBuffer, ...
    tRecStart, tRecEnd)
% BUILD_POST_REACH_WINDOWS - Fixed post-reach windows and eligibility masks
%
% Window starts postReachBuffer after aligned reach onset. Exclude overlap with
% [next reach - preReachBuffer, next reach].

nReach = numel(reachStart);
winStartReach = nan(nReach, 1);
winEndReach = nan(nReach, 1);
winCenterReach = nan(nReach, 1);
eligibleReach = false(nReach, 1);
excludeOverlapNext = false(nReach, 1);
excludeRecording = false(nReach, 1);

for r = 1:nReach
  tReach = reachStart(r);
  winStartReach(r) = tReach + postReachBufferSec;
  winEndReach(r) = winStartReach(r) + d2Window;
  winCenterReach(r) = (winStartReach(r) + winEndReach(r)) / 2;

  if winStartReach(r) < tRecStart + windowBuffer || winEndReach(r) > tRecEnd - windowBuffer
    excludeRecording(r) = true;
    continue;
  end

  if r < nReach
    tNextReach = reachStart(r + 1);
    forbiddenStart = tNextReach - preReachBufferSec;
    if winStartReach(r) < tNextReach && winEndReach(r) > forbiddenStart
      excludeOverlapNext(r) = true;
      continue;
    end
  end

  eligibleReach(r) = true;
end
end

function [d2RawVec, d2NormVec] = compute_aligned_reach_d2_for_area( ...
    dataStruct, areaIdx, eligibleReach, winCenterReach, d2Window, ...
    binSizeArea, timeRange, opts)
% COMPUTE_ALIGNED_REACH_D2_FOR_AREA - d2 per reach for one brain area

nReach = numel(winCenterReach);
d2RawVec = nan(nReach, 1);
d2NormVec = nan(nReach, 1);

neuronIDs = dataStruct.idLabel{areaIdx};
aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIDs, timeRange, binSizeArea);
nT = size(aDataMat, 1);

if opts.pcaFlag
  [coeff, score, ~, ~, explained, mu] = pca(aDataMat);
  forDim = find(cumsum(explained) > 30, 1);
  forDim = max(3, min(6, forDim));
  nDimUse = 1:forDim;
  aDataMat = score(:, nDimUse) * coeff(:, nDimUse)' + mu;
end

popTraceCell = cell(nReach, 1);
windowMatCell = cell(nReach, 1);

for r = 1:nReach
  if ~eligibleReach(r)
    continue;
  end
  centerTime = winCenterReach(r);
  [i0, i1, idxOk] = window_indices_strict(centerTime, d2Window, binSizeArea, nT);
  if ~idxOk
    continue;
  end
  wMat = aDataMat(i0:i1, :);
  popTraceCell{r} = mean(wMat, 2);
  windowMatCell{r} = wMat;
end

if opts.meanSubtract
  validIdx = find(eligibleReach & ~cellfun(@isempty, popTraceCell));
  if ~isempty(validIdx)
    nBinsRef = numel(popTraceCell{validIdx(1)});
    popMat = nan(numel(validIdx), nBinsRef);
    for ii = 1:numel(validIdx)
      popMat(ii, :) = popTraceCell{validIdx(ii)}(:)';
    end
    meanPerBin = nanmean(popMat, 1);
    for ii = 1:numel(validIdx)
      r = validIdx(ii);
      popTraceCell{r} = popTraceCell{r} - meanPerBin(:);
    end
  end
end

for r = 1:nReach
  popTrace = popTraceCell{r};
  if isempty(popTrace)
    continue;
  end
  d2Val = compute_d2_from_pop_trace_local(popTrace, opts.pOrder, opts.critType);
  d2RawVec(r) = d2Val;

  if opts.normalizeD2 && ~isempty(windowMatCell{r})
    wMat = windowMatCell{r};
    nBins = size(wMat, 1);
    nNeu = size(wMat, 2);
    d2Shuf = nan(opts.nShuffles, 1);
    for sh = 1:opts.nShuffles
      permutedMat = zeros(size(wMat));
      for n = 1:nNeu
        permutedMat(:, n) = circshift(wMat(:, n), randi(nBins));
      end
      shTrace = mean(permutedMat, 2);
      if opts.meanSubtract
        shTrace = shTrace - nanmean(shTrace);
      end
      d2Shuf(sh) = compute_d2_from_pop_trace_local(shTrace, opts.pOrder, opts.critType);
    end
    meanShuf = nanmean(d2Shuf);
    if isfinite(d2Val) && isfinite(meanShuf) && meanShuf > 0
      d2NormVec(r) = d2Val / meanShuf;
    end
  end
end
end

function d2Analysis = build_d2_analysis_vectors(d2PerReach, d2PerReachNorm, ...
    areasToTest, numAreas, normalizeD2, useLog10D2)
% BUILD_D2_ANALYSIS_VECTORS - Select raw/normalized d2 and optional log10

d2Analysis = cell(1, numAreas);
for a = areasToTest
  if normalizeD2
    d2Analysis{a} = d2PerReachNorm{a};
  else
    d2Analysis{a} = d2PerReach{a};
  end
  if useLog10D2
    d2Analysis{a} = apply_log10_safe_vector(d2Analysis{a});
  end
end
end

function statsByArea = build_d2_accuracy_stats_by_area(d2Analysis, areas, ...
    areasToTest, eligibleReach, isCorrectReach, isErrorReach, timingLabel)
% BUILD_D2_ACCURACY_STATS_BY_AREA - Per-area correct vs error comparison stats

if nargin < 8 || isempty(timingLabel)
  timingLabel = '';
end

statsByArea = struct([]);
for colIdx = 1:numel(areasToTest)
  a = areasToTest(colIdx);
  d2Vec = d2Analysis{a};
  maskCorrect = eligibleReach & isCorrectReach & isfinite(d2Vec);
  maskError = eligibleReach & isErrorReach & isfinite(d2Vec);
  d2Correct = d2Vec(maskCorrect);
  d2Error = d2Vec(maskError);

  statsByArea(colIdx).areaName = areas{a};
  statsByArea(colIdx).areaIdx = a;
  statsByArea(colIdx).nCorrect = numel(d2Correct);
  statsByArea(colIdx).nError = numel(d2Error);
  statsByArea(colIdx).meanCorrect = nanmean(d2Correct);
  statsByArea(colIdx).meanError = nanmean(d2Error);
  statsByArea(colIdx).semCorrect = sem_local(d2Correct);
  statsByArea(colIdx).semError = sem_local(d2Error);
  statsByArea(colIdx).d2Correct = d2Correct;
  statsByArea(colIdx).d2Error = d2Error;
  [statsByArea(colIdx).pRanksum, statsByArea(colIdx).zRanksum] = ...
    ranksum_test_safe(d2Correct, d2Error);
  [statsByArea(colIdx).pTtest2, statsByArea(colIdx).tTtest2] = ...
    ttest2_safe(d2Correct, d2Error);
  statsByArea(colIdx).meanDiff = statsByArea(colIdx).meanCorrect - statsByArea(colIdx).meanError;

  if isempty(timingLabel)
    fprintf('  %s: correct n=%d mean=%.4f | error n=%d mean=%.4f | p_ranksum=%.4g\n', ...
      areas{a}, statsByArea(colIdx).nCorrect, statsByArea(colIdx).meanCorrect, ...
      statsByArea(colIdx).nError, statsByArea(colIdx).meanError, statsByArea(colIdx).pRanksum);
  else
    fprintf('  %s [%s]: correct n=%d mean=%.4f | error n=%d mean=%.4f | p_ranksum=%.4g\n', ...
      areas{a}, timingLabel, statsByArea(colIdx).nCorrect, statsByArea(colIdx).meanCorrect, ...
      statsByArea(colIdx).nError, statsByArea(colIdx).meanError, statsByArea(colIdx).pRanksum);
  end
end
end

%% -------------------------------------------------------------------------
%% Plotting and printing
%% -------------------------------------------------------------------------

function fig = plot_d2_accuracy_comparison(statsByAreaPre, statsByAreaPost, ...
    sessionName, opts, plotConfig)
% PLOT_D2_ACCURACY_COMPARISON - Pre (top) and post (bottom) correct vs error d2

plotConfig = fill_default_d2_accuracy_plot_config(plotConfig);
nCol = numel(statsByAreaPre);

if opts.useLog10D2
  yLabelStr = 'log_{10}(d2)';
  yLabInterpreter = 'tex';
else
  yLabelStr = 'd2';
  yLabInterpreter = 'none';
end

fig = figure('Color', 'w', 'Name', 'Pre/post-reach d2: correct vs error', ...
  'NumberTitle', 'off');
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
  set(fig, 'Position', monitorPositions(end, :));
else
  set(fig, 'Position', monitorPositions(1, :));
end

nRow = 2;
useTight = exist('tight_subplot', 'file');
if useTight
  ha = tight_subplot(nRow, nCol, [0.06 0.06], [0.08 0.12], [0.08 0.04]);
else
  ha = gobjects(nRow * nCol, 1);
  for ii = 1:(nRow * nCol)
    ha(ii) = subplot(nRow, nCol, ii);
  end
end

rowStats = {statsByAreaPre, statsByAreaPost};
rowTitles = {'Pre-reach', 'Post-reach'};

[yLimGlobal, yTicksGlobal, yTickLabelsGlobal] = compute_d2_accuracy_y_axis( ...
  statsByAreaPre, statsByAreaPost);

for rowIdx = 1:nRow
  for colIdx = 1:nCol
    ax = ha((rowIdx - 1) * nCol + colIdx);
    st = rowStats{rowIdx}(colIdx);
    if rowIdx == 1
      panelTitle = st.areaName;
    else
      panelTitle = sprintf('n_{corr}=%d, n_{err}=%d', st.nCorrect, st.nError);
    end
    plot_d2_accuracy_panel(ax, st, plotConfig, panelTitle, yLimGlobal);
    if colIdx == 1
      ylabel(ax, sprintf('%s\n%s', rowTitles{rowIdx}, yLabelStr), ...
        'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', yLabInterpreter);
    end
    set(ax, 'YLim', yLimGlobal, 'YTick', yTicksGlobal, 'YTickLabel', yTickLabelsGlobal, ...
      'YTickLabelMode', 'manual', 'TickLabelInterpreter', 'none');
    make_axis_tick_labels_visible(ax);
    set(ax, 'XTick', [1, 2], 'XTickLabel', {'Correct', 'Error'}, ...
      'XTickLabelMode', 'manual', 'TickLabelInterpreter', 'none');
  end
end

sgtitle(sprintf(['Pre/post-reach d2 | %s | W=%.1fs, preBuf=%d ms, postBuf=%d ms'], ...
  sessionName, opts.d2Window, opts.preReachBuffer, opts.postReachBuffer), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none');
end

function [yLimGlobal, yTicksGlobal, yTickLabelsGlobal] = compute_d2_accuracy_y_axis( ...
    statsByAreaPre, statsByAreaPost)
% COMPUTE_D2_ACCURACY_Y_AXIS - Shared y-limits and tick labels across panels

allVals = [];
for st = [statsByAreaPre, statsByAreaPost]
  allVals = [allVals; st.d2Correct(:); st.d2Error(:)]; %#ok<AGROW>
end
allVals = allVals(isfinite(allVals));

if isempty(allVals)
  yLimGlobal = [0, 1];
elseif numel(unique(allVals)) == 1
  yPad = max(0.05, 0.05 * abs(allVals(1)));
  yLimGlobal = [allVals(1) - yPad, allVals(1) + yPad];
else
  yMinData = min(allVals);
  yMaxData = max(allVals);
  yPad = 0.12 * (yMaxData - yMinData);
  yLimGlobal = [yMinData - yPad, yMaxData + yPad];
end

nTickY = 5;
yTicksGlobal = linspace(yLimGlobal(1), yLimGlobal(2), nTickY);
yTickLabelsGlobal = cell(nTickY, 1);
for tt = 1:nTickY
  yTickLabelsGlobal{tt} = sprintf('%.2f', yTicksGlobal(tt));
end
end

function plot_d2_accuracy_panel(ax, st, plotConfig, titleStr, yLimGlobal)
% PLOT_D2_ACCURACY_PANEL - Single correct vs error panel

if nargin < 5 || isempty(yLimGlobal)
  yLimGlobal = [0, 1];
end
if nargin < 4 || isempty(titleStr)
  titleStr = sprintf('n_{corr}=%d, n_{err}=%d', st.nCorrect, st.nError);
end

correctColor = [0.15, 0.45, 0.75];
errorColor = [0.85, 0.33, 0.1];
hold(ax, 'on');

groupData = {st.d2Correct, st.d2Error};
groupNames = {'Correct', 'Error'};
xPos = 1:2;
allVals = [groupData{1}(:); groupData{2}(:)];
allGrp = [ones(numel(groupData{1}), 1); 2 * ones(numel(groupData{2}), 1)];

if ~isempty(allVals)
  axes(ax);
  boxplot(allVals, allGrp, 'Labels', groupNames, 'Symbol', '', 'Widths', 0.35);
  for g = 1:2
    vals = groupData{g};
    if isempty(vals)
      continue;
    end
    ptColor = correctColor * (g == 1) + errorColor * (g == 2);
    scatter(ax, xPos(g) + 0.06 * (rand(numel(vals), 1) - 0.5), vals, ...
      20, 'filled', 'MarkerFaceAlpha', 0.45, ...
      'MarkerFaceColor', ptColor, 'MarkerEdgeColor', 'none');
    errorbar(ax, xPos(g), nanmean(vals), sem_local(vals), 'k', ...
      'LineWidth', plotConfig.axesLineWidth, 'CapSize', plotConfig.errorCapSize);
  end
end

sigStr = p_to_star(st.pRanksum);
yText = yLimGlobal(2) - 0.02 * (yLimGlobal(2) - yLimGlobal(1));
text(ax, 1.5, yText, sprintf('p=%.3g %s', st.pRanksum, sigStr), ...
  'HorizontalAlignment', 'center', 'FontSize', plotConfig.tickLabelFontSize, ...
  'VerticalAlignment', 'top');

hold(ax, 'off');
set(ax, 'XTick', xPos, 'XTickLabel', groupNames, 'XTickLabelMode', 'manual', ...
  'TickLabelInterpreter', 'none', ...
  'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth);
xlim(ax, [0.4, 2.6]);
ylim(ax, yLimGlobal);
title(ax, titleStr, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
grid(ax, 'on');
end

function make_axis_tick_labels_visible(axPanel)
% MAKE_AXIS_TICK_LABELS_VISIBLE - Restore tick labels hidden by tight_subplot
% Does not change tick label mode (preserves manual Correct/Error x-labels).

if isprop(axPanel, 'XTickLabelVisible')
  set(axPanel, 'XTickLabelVisible', 'on', 'YTickLabelVisible', 'on');
end
end

function print_d2_accuracy_stats(statsByArea, useLog10D2, timingLabel)
% PRINT_D2_ACCURACY_STATS - Command-window summary per area

if nargin < 3 || isempty(timingLabel)
  timingLabel = 'Pre-reach';
end
if useLog10D2
  metricLabel = 'log10(d2)';
else
  metricLabel = 'd2';
end
fprintf('\n=== %s %s: correct vs error ===\n', timingLabel, metricLabel);
fprintf('%-8s  n_corr  n_err   mean_corr+/-SEM   mean_err+/-SEM    diff      p_ranksum  p_ttest2\n', 'Area');
for i = 1:numel(statsByArea)
  st = statsByArea(i);
  fprintf('%-8s  %5d  %5d   %7.4f+/-%.4f   %7.4f+/-%.4f   %+.4f   %9.3g   %9.3g\n', ...
    st.areaName, st.nCorrect, st.nError, ...
    st.meanCorrect, st.semCorrect, st.meanError, st.semError, ...
    st.meanDiff, st.pRanksum, st.pTtest2);
end
if strcmpi(timingLabel, 'Pre-reach')
  fprintf(['Pre: window ends preReachBuffer before reach; excludes overlap with ', ...
    'prev reach + postReachBuffer.\n']);
elseif strcmpi(timingLabel, 'Post-reach')
  fprintf(['Post: window starts postReachBuffer after reach; excludes overlap with ', ...
    'next reach - preReachBuffer.\n']);
end
end

%% -------------------------------------------------------------------------
%% Low-level helpers
%% -------------------------------------------------------------------------

function [startIdx, endIdx, ok] = window_indices_strict(centerTime, slidingWindowSize, binSize, numTimePoints)
% window_indices_strict - bin indices for centered window; ok false if clamped

centerIdx = round(centerTime / binSize) + 1;
winSamples = round(slidingWindowSize / binSize);
if winSamples < 1
  winSamples = 1;
end
halfWin = round(winSamples / 2);
startIdx = centerIdx - halfWin + 1;
endIdx = startIdx + winSamples - 1;
ok = startIdx >= 1 && endIdx <= numTimePoints && endIdx > startIdx;
end

function d2Val = compute_d2_from_pop_trace_local(popTrace, pOrder, critType)
% compute_d2_from_pop_trace_local - d2 from population activity trace

d2Val = nan;
if isempty(popTrace) || numel(popTrace) < pOrder + 2
  return;
end
try
  [varphi, ~] = myYuleWalker3(double(popTrace), pOrder);
  d2Val = getFixedPointDistance2(pOrder, critType, varphi);
catch
  d2Val = nan;
end
end

function vOut = apply_log10_safe_vector(vIn)
% apply_log10_safe_vector - log10 with nonpositive/nonfinite -> NaN

vOut = nan(size(vIn));
ok = isfinite(vIn) & vIn > 0;
vOut(ok) = log10(vIn(ok));
end

function s = sem_local(x)
% sem_local - standard error of the mean (omit NaN)

x = x(:);
x = x(isfinite(x));
n = numel(x);
if n < 2
  s = nan;
  return;
end
s = std(x, 0) / sqrt(n);
end

function [pVal, zVal] = ranksum_test_safe(x, y)
% ranksum_test_safe - Wilcoxon rank-sum; NaN if either group too small

pVal = nan;
zVal = nan;
x = x(isfinite(x));
y = y(isfinite(y));
if numel(x) < 1 || numel(y) < 1
  return;
end
[pVal, ~, stats] = ranksum(x, y);
if isfield(stats, 'zval')
  zVal = stats.zval;
end
end

function [pVal, tStat] = ttest2_safe(x, y)
% ttest2_safe - two-sample t-test; NaN if either group has < 2 samples

pVal = nan;
tStat = nan;
x = x(isfinite(x));
y = y(isfinite(y));
if numel(x) < 2 || numel(y) < 2
  return;
end
[~, pVal, ~, stats] = ttest2(x, y);
if isfield(stats, 'tstat')
  tStat = stats.tstat;
end
end

function starStr = p_to_star(pVal)
% p_to_star - significance marker from p-value

if ~isfinite(pVal)
  starStr = '';
elseif pVal < 0.001
  starStr = '***';
elseif pVal < 0.01
  starStr = '**';
elseif pVal < 0.05
  starStr = '*';
else
  starStr = 'ns';
end
end

function label = format_areas_label(areaNames)
% format_areas_label - filesystem-safe label from area name(s)

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
