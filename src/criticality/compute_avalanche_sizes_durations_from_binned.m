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
%
% Goal:
%   Detect avalanches after applying the population threshold. When shared
%   subsample indices / thresholds are present, reuse them instead of drawing
%   new neuron subsets or recomputing cutoffs from this segment alone.

sizes = [];
durations = [];
hasAvalanches = false;

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
