function threshInfo = prepare_shared_avalanche_threshold_info(aDataMatFull, config)
% PREPARE_SHARED_AVALANCHE_THRESHOLD_INFO - Collect-range thresholds (+ subsample idxs)
%
% Variables:
%   aDataMatFull - Binned spikes for the full collect range [timeBins x neurons]
%   config       - AV analysis config (thresholdMethod, useSubsampling, ...)
%
% Goal:
%   Compute the population avalanche cutoff from aDataMatFull (typically the
%   full collect range for the total class, or concatenated engaged /
%   non-engaged segments for those classes). With useSubsampling, draw neuron
%   subsets once and compute one cutoff per subsample from that matrix.

threshInfo = struct( ...
  'useSubsampling', false, ...
  'neuronIdxSubsamples', {{}}, ...
  'thresholdPerSubsample', [], ...
  'fixedThreshold', nan);

if isempty(aDataMatFull) || size(aDataMatFull, 1) < 1
  return;
end

useSubsampling = isfield(config, 'useSubsampling') && config.useSubsampling;
threshInfo.useSubsampling = useSubsampling;
numNeuronsArea = size(aDataMatFull, 2);
computeThreshold = ~is_mean_isi_zero_avalanche_mode(config);
if isfield(config, 'thresholdFlag') && ~config.thresholdFlag
  computeThreshold = false;
end

if useSubsampling
  nSubsamplesArea = config.nSubsamples;
  nNeuronsSubsampleArea = min(config.nNeuronsSubsample, numNeuronsArea);
  neuronIdxSubsamples = cell(1, nSubsamplesArea);
  thresholdPerSubsample = nan(1, nSubsamplesArea);
  for s = 1:nSubsamplesArea
    if nNeuronsSubsampleArea == numNeuronsArea
      neuronIdxSubsamples{s} = 1:numNeuronsArea;
    else
      neuronIdxSubsamples{s} = randperm(numNeuronsArea, nNeuronsSubsampleArea);
    end
    if computeThreshold
      popFull = sum(aDataMatFull(:, neuronIdxSubsamples{s}), 2);
      thresholdPerSubsample(s) = compute_avalanche_population_threshold(popFull, config);
    end
  end
  threshInfo.neuronIdxSubsamples = neuronIdxSubsamples;
  threshInfo.thresholdPerSubsample = thresholdPerSubsample;
elseif computeThreshold
  popFull = sum(aDataMatFull, 2);
  threshInfo.fixedThreshold = compute_avalanche_population_threshold(popFull, config);
end
end
