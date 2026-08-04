function wPopActivity = apply_avalanche_population_threshold(wPopActivity, config, fixedThreshold)
% APPLY_AVALANCHE_POPULATION_THRESHOLD - Threshold population activity for avalanches
%
% Variables:
%   wPopActivity   - Population spike counts per bin (column vector)
%   config         - avalancheDetectionMode, thresholdFlag, thresholdMethod,
%                    thresholdPct (median only); optional fixedPopulationThreshold
%   fixedThreshold - Optional scalar cutoff; if finite, use instead of recomputing
%
% Goal:
%   meanIsiZero: zero cutoff (activity > 0 defines avalanches).
%   fixedBinMedian (default): bins below cutoff are zeroed.
%     If fixedThreshold or config.fixedPopulationThreshold is set, that cutoff
%     is used (shared across engagement classes / windows).
%     Otherwise cutoff from this trace:
%       thresholdMethod:
%         'median' (default) - thresholdPct * median(population activity)
%         'quantile10'       - 10th percentile of population activity

if nargin < 3
  fixedThreshold = [];
end

wPopActivity = wPopActivity(:);

if is_mean_isi_zero_avalanche_mode(config)
  wPopActivity(wPopActivity <= 0) = 0;
  return;
end

useThreshold = true;
if isfield(config, 'thresholdFlag') && ~config.thresholdFlag
  useThreshold = false;
end

if ~useThreshold
  wPopActivity(wPopActivity <= 0) = 0;
  return;
end

if ~(isfinite(fixedThreshold))
  if isfield(config, 'fixedPopulationThreshold') ...
      && isfinite(config.fixedPopulationThreshold)
    fixedThreshold = config.fixedPopulationThreshold;
  end
end

if isfinite(fixedThreshold)
  threshSpikes = fixedThreshold;
else
  threshSpikes = compute_avalanche_population_threshold(wPopActivity, config);
end

if ~isfinite(threshSpikes)
  wPopActivity(wPopActivity <= 0) = 0;
  return;
end

wPopActivity(wPopActivity < threshSpikes) = 0;
end
