function threshSpikes = compute_avalanche_population_threshold(wPopActivity, config)
% COMPUTE_AVALANCHE_POPULATION_THRESHOLD - Scalar cutoff from pop activity
%
% Variables:
%   wPopActivity - Population spike counts per bin
%   config       - avalancheDetectionMode, thresholdFlag, thresholdMethod,
%                  thresholdPct (median only)
%
% Goal:
%   Return the spike-count cutoff used by apply_avalanche_population_threshold
%   (NaN when thresholding is disabled / meanIsiZero).

threshSpikes = nan;
wPopActivity = wPopActivity(:);
if isempty(wPopActivity)
  return;
end

if is_mean_isi_zero_avalanche_mode(config)
  return;
end

useThreshold = true;
if isfield(config, 'thresholdFlag') && ~config.thresholdFlag
  useThreshold = false;
end
if ~useThreshold
  return;
end

thresholdMethod = 'median';
if isfield(config, 'thresholdMethod') && ~isempty(config.thresholdMethod)
  thresholdMethod = char(config.thresholdMethod);
end

switch lower(thresholdMethod)
  case 'median'
    thresholdPct = 1;
    if isfield(config, 'thresholdPct') && ~isempty(config.thresholdPct)
      thresholdPct = config.thresholdPct;
    end
    threshSpikes = thresholdPct * median(wPopActivity);
  case {'quantile10', 'p10', 'q10'}
    threshSpikes = quantile(wPopActivity, 0.1);
  otherwise
    error('compute_avalanche_population_threshold:BadThresholdMethod', ...
      ['Unknown config.thresholdMethod ''%s''. ', ...
      'Use ''median'' or ''quantile10''.'], thresholdMethod);
end
end
