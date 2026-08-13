function avData = extract_pooled_area_avalanches(dataStruct, areaIndex, analysisConfig, ...
    segments, computeShuffles)
% EXTRACT_POOLED_AREA_AVALANCHES - Detect per segment/window, pool, fit once
%
% Variables:
%   dataStruct      - Session data with spikeTimes / idLabel
%   areaIndex       - Area index
%   analysisConfig  - AV config; optional .avWindow, .sharedThresholdByArea
%   segments        - Struct array (.start, .end) in seconds
%   computeShuffles - If true, also pool circular-shuffle avalanches
%
% Goal:
%   Detect avalanches in each segment (or avWindow tile), pool sizes/durations,
%   and fit power laws once on the pooled sample. When avWindow is set, each
%   tile uses a threshold from that tile's own population activity. When
%   avWindow is empty, reuse sharedThresholdByArea (collect-range for total;
%   engagement callers pass class-specific cutoffs for engaged / non-engaged).

if nargin < 5 || isempty(computeShuffles)
  computeShuffles = false;
end

avData = empty_avalanche_data();
if isempty(segments)
  return;
end

% Attach collect-range shared neuron subsets (+ thresholds when avWindow empty)
if isfield(analysisConfig, 'sharedThresholdByArea') ...
    && numel(analysisConfig.sharedThresholdByArea) >= areaIndex ...
    && ~isempty(analysisConfig.sharedThresholdByArea{areaIndex})
  analysisConfig = merge_shared_av_threshold_into_config( ...
    analysisConfig, analysisConfig.sharedThresholdByArea{areaIndex});
end

avWindow = resolve_effective_av_window(analysisConfig);
useLocalThresh = ~isempty(avWindow);
if useLocalThresh
  analysisConfig.avWindow = avWindow;
  % Keep fixed neuron subsets; drop collect-range cutoffs (recomputed per tile)
  if isfield(analysisConfig, 'thresholdPerSubsample')
    analysisConfig = rmfield(analysisConfig, 'thresholdPerSubsample');
  end
  if isfield(analysisConfig, 'fixedPopulationThreshold')
    analysisConfig = rmfield(analysisConfig, 'fixedPopulationThreshold');
  end
  analysisConfig.useLocalWindowThreshold = true;
end

minSegDur = 0.2;
if isfield(analysisConfig, 'binSize') && isfinite(analysisConfig.binSize)
  minSegDur = max(minSegDur, analysisConfig.binSize * 4);
end

analysisWindows = tile_segments_into_av_windows(segments, avWindow, minSegDur);

allSizes = [];
allDurations = [];
allShuffleSizes = [];
allShuffleDurations = [];
nShufflesCompleted = 0;
binSizeUsed = nan;
for i = 1:numel(analysisWindows)
  segStart = analysisWindows(i).start;
  segEnd = analysisWindows(i).end;
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
avData.nSegments = numel(analysisWindows);
avData.shuffleSizes = allShuffleSizes;
avData.shuffleDurations = allShuffleDurations;
avData.nShufflesCompleted = nShufflesCompleted;

if computeShuffles && ~isempty(allShuffleSizes) && ~isempty(allShuffleDurations)
  plShuf = avalanche_power_law_metrics(allShuffleSizes, allShuffleDurations, analysisConfig);
  avData.shuffleTau = plShuf.tau;
  avData.shuffleAlpha = plShuf.alpha;
  avData.shuffleParamSD = plShuf.paramSD;
  avData.shuffleDecades = plShuf.decades;
  avData.shuffleDcc = distance_to_criticality(plShuf.tau, plShuf.alpha, plShuf.paramSD);
  avData.shuffleKappa = compute_kappa(allShuffleSizes);
end
avData.kappa = compute_kappa(allSizes);
end

function avData = extract_area_avalanches(dataStruct, areaIndex, analysisConfig, ...
    collectStart, collectEnd, computeShuffles)
% EXTRACT_AREA_AVALANCHES - Bin, threshold, and detect avalanches in one interval

if nargin < 6 || isempty(computeShuffles)
  computeShuffles = false;
end

avData = empty_avalanche_data();
timeRange = [collectStart, collectEnd];
neuronIds = dataStruct.idLabel{areaIndex};
if isfield(analysisConfig, 'sharedCollectBinSize') ...
    && isfinite(analysisConfig.sharedCollectBinSize)
  binSize = analysisConfig.sharedCollectBinSize;
else
  binSizeVec = resolve_avalanche_bin_sizes(dataStruct, areaIndex, timeRange, analysisConfig);
  binSize = binSizeVec(areaIndex);
end
avData.binSize = binSize;

aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIds, timeRange, binSize);

if isfield(analysisConfig, 'useLocalWindowThreshold') && analysisConfig.useLocalWindowThreshold
  analysisConfig = attach_local_avalanche_window_threshold(analysisConfig, aDataMat);
end

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
  'tau', nan, 'alpha', nan, 'paramSD', nan, 'decades', nan, 'dcc', nan, 'kappa', nan, ...
  'scalingRelation', nan, ...
  'minSizeFit', nan, 'maxSizeFit', nan, ...
  'minDurFit', nan, 'maxDurFit', nan, 'sizeFitInfo', struct(), ...
  'durFitInfo', struct(), 'nAvalanches', 0, 'binSize', nan, 'nSegments', 0, ...
  'shuffleSizes', [], 'shuffleDurations', [], 'nShufflesCompleted', 0, ...
  'shuffleTau', nan, 'shuffleAlpha', nan, 'shuffleParamSD', nan, ...
  'shuffleDecades', nan, 'shuffleDcc', nan, 'shuffleKappa', nan);
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
