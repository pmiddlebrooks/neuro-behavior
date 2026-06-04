function metrics = compute_av_metrics_from_pop_activity(wPopActivity, config)
% COMPUTE_AV_METRICS_FROM_POP_ACTIVITY - Avalanche metrics for one population trace
%
% Variables:
%   wPopActivity - Population activity per bin (column vector)
%   config       - Analysis config passed to avalanche_power_law_metrics
%
% Returns:
%   metrics - Struct with dcc, kappa, decades, tau, alpha, paramSD (NaN if no avalanches)

metrics = struct('dcc', nan, 'kappa', nan, 'decades', nan, ...
  'tau', nan, 'alpha', nan, 'paramSD', nan);

wPopActivity = apply_avalanche_population_threshold(wPopActivity(:), config);
zeroBins = find(wPopActivity == 0);
if ~(numel(zeroBins) > 1 && any(diff(zeroBins) > 1))
  return;
end

[sizes, durs] = getAvalanches(wPopActivity', 0.5, 1);
plMetrics = avalanche_power_law_metrics(sizes, durs, config);

metrics.dcc = distance_to_criticality(plMetrics.tau, plMetrics.alpha, plMetrics.paramSD);
metrics.kappa = compute_kappa(sizes);
metrics.decades = plMetrics.decades;
metrics.tau = plMetrics.tau;
metrics.alpha = plMetrics.alpha;
metrics.paramSD = plMetrics.paramSD;
end
