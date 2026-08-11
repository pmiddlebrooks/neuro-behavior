function analysisConfig = attach_local_avalanche_window_threshold(analysisConfig, aDataMat)
% ATTACH_LOCAL_AVALANCHE_WINDOW_THRESHOLD - Cutoff(s) from this window's pop
%
% Variables:
%   analysisConfig - AV config; reuses .neuronIdxSubsamples when present
%   aDataMat       - Binned spikes for this window [timeBins x neurons]
%
% Goal:
%   Replace collect-range cutoffs with thresholds computed from this window's
%   population activity (per subsample when those neuron indices are fixed).

if isempty(aDataMat) || size(aDataMat, 1) < 1
  return;
end

useSubsampling = isfield(analysisConfig, 'useSubsampling') && analysisConfig.useSubsampling;
hasSharedSubsamples = useSubsampling ...
  && isfield(analysisConfig, 'neuronIdxSubsamples') ...
  && iscell(analysisConfig.neuronIdxSubsamples) ...
  && ~isempty(analysisConfig.neuronIdxSubsamples);

if hasSharedSubsamples
  nSubsamplesArea = numel(analysisConfig.neuronIdxSubsamples);
  thresholdPerSubsample = nan(1, nSubsamplesArea);
  numNeuronsArea = size(aDataMat, 2);
  for s = 1:nSubsamplesArea
    colIdx = analysisConfig.neuronIdxSubsamples{s};
    colIdx = colIdx(colIdx >= 1 & colIdx <= numNeuronsArea);
    if isempty(colIdx)
      continue;
    end
    popWin = sum(aDataMat(:, colIdx), 2);
    thresholdPerSubsample(s) = compute_avalanche_population_threshold(popWin, analysisConfig);
  end
  analysisConfig.thresholdPerSubsample = thresholdPerSubsample;
  if isfield(analysisConfig, 'fixedPopulationThreshold')
    analysisConfig = rmfield(analysisConfig, 'fixedPopulationThreshold');
  end
else
  popWin = sum(aDataMat, 2);
  analysisConfig.fixedPopulationThreshold = ...
    compute_avalanche_population_threshold(popWin, analysisConfig);
  if isfield(analysisConfig, 'thresholdPerSubsample')
    analysisConfig = rmfield(analysisConfig, 'thresholdPerSubsample');
  end
end
end
