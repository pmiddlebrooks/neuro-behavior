function wPopActivity = apply_avalanche_population_threshold(wPopActivity, config)
% APPLY_AVALANCHE_POPULATION_THRESHOLD - Threshold population activity for avalanches
%
% Variables:
%   wPopActivity - Population spike counts per bin (column vector)
%   config       - avalancheDetectionMode, thresholdFlag, thresholdPct
%
% Goal:
%   meanIsiZero: zero cutoff (activity > 0 defines avalanches).
%   fixedBinMedian (default): median * thresholdPct cutoff per window.

wPopActivity = wPopActivity(:);

if is_mean_isi_zero_avalanche_mode(config)
  wPopActivity(wPopActivity <= 0) = 0;
  return;
end

useMedian = true;
if isfield(config, 'thresholdFlag') && ~config.thresholdFlag
  useMedian = false;
end

if ~useMedian
  wPopActivity(wPopActivity <= 0) = 0;
  return;
end

thresholdPct = 1;
if isfield(config, 'thresholdPct') && ~isempty(config.thresholdPct)
  thresholdPct = config.thresholdPct;
end

threshSpikes = thresholdPct * median(wPopActivity);
wPopActivity(wPopActivity < threshSpikes) = 0;
end
