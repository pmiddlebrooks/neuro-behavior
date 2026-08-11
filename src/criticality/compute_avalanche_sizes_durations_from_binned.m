function [sizes, durations, hasAvalanches] = compute_avalanche_sizes_durations_from_binned( ...
    aDataMat, analysisConfig)
% COMPUTE_AVALANCHE_SIZES_DURATIONS_FROM_BINNED - Avalanche vectors from binned matrix
%
% Variables:
%   aDataMat       - Binned spikes [timeBins x neurons]
%   analysisConfig - AV config; optional shared-threshold fields:
%     .fixedPopulationThreshold - scalar cutoff (full collect range)
%     .neuronIdxSubsamples      - cell of column indices (fixed across segments)
%     .thresholdPerSubsample    - cutoff per subsample from full collect range
%     .avWindow                 - if set, tile this matrix and use a local
%                                 threshold per tile (then pool events)
%
% Goal:
%   Detect avalanches after applying the population threshold. When avWindow
%   is set, split the binned trace into tiles, recompute the cutoff from each
%   tile's pop activity, and pool sizes/durations. Callers still fit once on
%   the collected events.

sizes = [];
durations = [];
hasAvalanches = false;

skipTile = isfield(analysisConfig, 'skipAvWindowTile') && analysisConfig.skipAvWindowTile;
if ~skipTile && use_local_av_window_thresholds(analysisConfig)
  [sizes, durations, hasAvalanches] = compute_tiled_window_avalanches( ...
    aDataMat, analysisConfig);
  return;
end

useSubsampling = isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling;
if useSubsampling
  numNeuronsArea = size(aDataMat, 2);
  nSubsamplesArea = analysisConfig.nSubsamples;
  nNeuronsSubsampleArea = min(analysisConfig.nNeuronsSubsample, numNeuronsArea);
  hasSharedSubsamples = isfield(analysisConfig, 'neuronIdxSubsamples') ...
    && iscell(analysisConfig.neuronIdxSubsamples) ...
    && numel(analysisConfig.neuronIdxSubsamples) >= nSubsamplesArea;
  hasSharedThresholds = isfield(analysisConfig, 'thresholdPerSubsample') ...
    && numel(analysisConfig.thresholdPerSubsample) >= nSubsamplesArea;

  for s = 1:nSubsamplesArea
    if hasSharedSubsamples
      colIdx = analysisConfig.neuronIdxSubsamples{s};
      colIdx = colIdx(colIdx >= 1 & colIdx <= numNeuronsArea);
      if isempty(colIdx)
        continue;
      end
    elseif nNeuronsSubsampleArea == numNeuronsArea
      colIdx = 1:numNeuronsArea;
    else
      colIdx = randperm(numNeuronsArea, nNeuronsSubsampleArea);
    end

    wPopActivity = sum(aDataMat(:, colIdx), 2);
    fixedThresh = [];
    if hasSharedThresholds && isfinite(analysisConfig.thresholdPerSubsample(s))
      fixedThresh = analysisConfig.thresholdPerSubsample(s);
    elseif isfield(analysisConfig, 'fixedPopulationThreshold') ...
        && isfinite(analysisConfig.fixedPopulationThreshold)
      fixedThresh = analysisConfig.fixedPopulationThreshold;
    end

    wPopActivity = apply_avalanche_population_threshold( ...
      wPopActivity, analysisConfig, fixedThresh);
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
  fixedThresh = [];
  if isfield(analysisConfig, 'fixedPopulationThreshold') ...
      && isfinite(analysisConfig.fixedPopulationThreshold)
    fixedThresh = analysisConfig.fixedPopulationThreshold;
  end
  wPopActivity = apply_avalanche_population_threshold( ...
    wPopActivity, analysisConfig, fixedThresh);
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

function [sizes, durations, hasAvalanches] = compute_tiled_window_avalanches( ...
    aDataMat, analysisConfig)
% COMPUTE_TILED_WINDOW_AVALANCHES - Per-tile local thresholds, then pool events

sizes = [];
durations = [];
hasAvalanches = false;

avWindow = resolve_effective_av_window(analysisConfig);
binSize = nan;
if isfield(analysisConfig, 'binSize') && ~isempty(analysisConfig.binSize) ...
    && isfinite(analysisConfig.binSize(1))
  binSize = analysisConfig.binSize(1);
elseif isfield(analysisConfig, 'sharedCollectBinSize') ...
    && isfinite(analysisConfig.sharedCollectBinSize)
  binSize = analysisConfig.sharedCollectBinSize;
end

nBins = size(aDataMat, 1);
if isempty(avWindow) || ~isfinite(binSize) || binSize <= 0 || nBins < 1
  cfgOne = analysisConfig;
  cfgOne.skipAvWindowTile = true;
  cfgOne = attach_local_avalanche_window_threshold(cfgOne, aDataMat);
  [sizes, durations, hasAvalanches] = compute_avalanche_sizes_durations_from_binned( ...
    aDataMat, cfgOne);
  return;
end

nBinsPerWindow = max(1, round(avWindow / binSize));
minBins = max(2, round(0.2 / binSize));
cfgBase = analysisConfig;
cfgBase.skipAvWindowTile = true;
cfgBase.avWindow = [];
if isfield(cfgBase, 'thresholdPerSubsample')
  cfgBase = rmfield(cfgBase, 'thresholdPerSubsample');
end
if isfield(cfgBase, 'fixedPopulationThreshold')
  cfgBase = rmfield(cfgBase, 'fixedPopulationThreshold');
end

startBin = 1;
while startBin <= nBins
  endBin = min(startBin + nBinsPerWindow - 1, nBins);
  if (endBin - startBin + 1) >= minBins
    chunkMat = aDataMat(startBin:endBin, :);
    cfgChunk = attach_local_avalanche_window_threshold(cfgBase, chunkMat);
    [sizesSub, dursSub, hasSub] = compute_avalanche_sizes_durations_from_binned( ...
      chunkMat, cfgChunk);
    if hasSub
      sizes = [sizes; sizesSub(:)]; %#ok<AGROW>
      durations = [durations; dursSub(:)]; %#ok<AGROW>
    end
  end
  startBin = startBin + nBinsPerWindow;
end

sizes = sizes(:);
durations = durations(:);
hasAvalanches = ~isempty(sizes) && ~isempty(durations);
end
