function analysisConfig = merge_shared_av_threshold_into_config(analysisConfig, threshInfo)
% MERGE_SHARED_AV_THRESHOLD_INTO_CONFIG - Attach collect-range threshold fields
%
% Variables:
%   analysisConfig - AV analysis config (modified copy returned)
%   threshInfo     - From prepare_shared_avalanche_threshold_info / prepare_area_...
%
% Goal:
%   Copy fixed threshold and/or per-subsample neuron indices + thresholds onto
%   the config so segment/window avalanche detection shares cutoffs.

if nargin < 2 || isempty(threshInfo)
  return;
end

if isfield(threshInfo, 'useSubsampling') && threshInfo.useSubsampling
  analysisConfig.neuronIdxSubsamples = threshInfo.neuronIdxSubsamples;
  analysisConfig.thresholdPerSubsample = threshInfo.thresholdPerSubsample;
  if isfield(analysisConfig, 'fixedPopulationThreshold')
    analysisConfig = rmfield(analysisConfig, 'fixedPopulationThreshold');
  end
elseif isfield(threshInfo, 'fixedThreshold') && isfinite(threshInfo.fixedThreshold)
  analysisConfig.fixedPopulationThreshold = threshInfo.fixedThreshold;
end

if isfield(threshInfo, 'binSize') && isfinite(threshInfo.binSize)
  analysisConfig.sharedCollectBinSize = threshInfo.binSize;
end
end
