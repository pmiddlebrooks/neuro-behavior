function permMetrics = run_av_window_circular_shuffles(windowData, config)
% RUN_AV_WINDOW_CIRCULAR_SHUFFLES - Avalanche metrics for circular-shuffled window data
%
% Variables:
%   windowData - [timeBins x neurons] binned activity for one window
%   config     - Must include .nShuffles; optional .pcaFlag (not supported here)
%
% Returns:
%   permMetrics - Struct arrays length nShuffles with fields dcc, kappa, decades, tau, alpha, paramSD

nShuffles = config.nShuffles;
config.compareTailModels = false;
permMetrics = repmat(struct('dcc', nan, 'kappa', nan, 'decades', nan, ...
  'tau', nan, 'alpha', nan, 'paramSD', nan, ...
  'sizeDecision', '', 'durDecision', '', ...
  'sizeVuongRExp', nan, 'sizeVuongPExp', nan, ...
  'durVuongRExp', nan, 'durVuongPExp', nan), 1, nShuffles);

winSamples = size(windowData, 1);
nNeurons = size(windowData, 2);
if winSamples < 1 || nNeurons < 1
  return;
end

for shuffle = 1:nShuffles
  permutedWindowData = windowData;
  for n = 1:nNeurons
    shiftAmount = randi([1, winSamples]);
    permutedWindowData(:, n) = circshift(permutedWindowData(:, n), shiftAmount);
  end

  if isfield(config, 'pcaFlag') && config.pcaFlag
    [coeffPerm, scorePerm, ~, ~, explainedPerm, muPerm] = pca(permutedWindowData);
    forDimPerm = find(cumsum(explainedPerm) > 30, 1);
    forDimPerm = max(3, min(6, forDimPerm));
    nDimPerm = 1:forDimPerm;
    permutedWindowData = scorePerm(:, nDimPerm) * coeffPerm(:, nDimPerm)' + muPerm;
  end

  wPopActivityPerm = sum(permutedWindowData, 2);
  fixedThreshold = [];
  if isfield(config, 'fixedPopulationThreshold') && isfinite(config.fixedPopulationThreshold)
    fixedThreshold = config.fixedPopulationThreshold;
  end
  permMetrics(shuffle) = compute_av_metrics_from_pop_activity( ...
    wPopActivityPerm, config, fixedThreshold);
end
end
