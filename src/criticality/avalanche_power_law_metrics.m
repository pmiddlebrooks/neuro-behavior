function metrics = avalanche_power_law_metrics(sizes, durs, config)
% AVALANCHE_POWER_LAW_METRICS - Tau, alpha, decades, paramSD from avalanche data
%
% Variables:
%   sizes  - Avalanche sizes
%   durs   - Avalanche durations
%   config - Passed to fit_avalanche_power_law (powerLawFitMethod, paths, etc.)
%
% Goal:
%   Single call site for power-law fits used in criticality_av_analysis.
%
% Returns:
%   metrics - Struct with tau, alpha, decades, minavS, maxavS, minavD, maxavD,
%             paramSD (crackling 1/σνz from WLS ⟨S⟩~T^γ over the α duration
%             power-law range [minavD, maxavD]; see size_given_duration)

metrics = struct('tau', nan, 'alpha', nan, 'decades', nan, ...
  'minavS', nan, 'maxavS', nan, 'minavD', nan, 'maxavD', nan, 'paramSD', nan);

sizeFit = fit_avalanche_power_law(sizes, config);
durFit = fit_avalanche_power_law(durs, config);

metrics.tau = sizeFit.exponent;
metrics.alpha = durFit.exponent;
metrics.decades = sizeFit.decades;
metrics.minavS = sizeFit.fitMin;
metrics.maxavS = sizeFit.fitMax;
metrics.minavD = durFit.fitMin;
metrics.maxavD = durFit.fitMax;

% Measured crackling 1/σνz: fit ⟨S⟩(T) only on avalanches whose duration lies
% in the same power-law range used for α (and for comparing to (α-1)/(τ-1)).
if isfinite(durFit.fitMin) && isfinite(durFit.fitMax) && durFit.fitMin < durFit.fitMax ...
    && numel(sizes) >= 2 && numel(durs) >= 2
  sizesCol = sizes(:);
  dursCol = durs(:);
  inDurFitRange = isfinite(sizesCol) & isfinite(dursCol) ...
    & sizesCol > 0 & dursCol > 0 ...
    & dursCol >= durFit.fitMin & dursCol <= durFit.fitMax;
  if nnz(inDurFitRange) >= 2 && numel(unique(dursCol(inDurFitRange))) >= 2
    [metrics.paramSD, ~, ~] = size_given_duration( ...
      sizesCol(inDurFitRange), dursCol(inDurFitRange), ...
      'durmin', durFit.fitMin, 'durmax', durFit.fitMax);
  end
end
end
