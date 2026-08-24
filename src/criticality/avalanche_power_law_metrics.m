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
%             power-law range [minavD, maxavD]; see size_given_duration),
%             sizeFit, durFit (full fit structs, including tailComparison)

metrics = struct('tau', nan, 'alpha', nan, 'decades', nan, ...
  'minavS', nan, 'maxavS', nan, 'minavD', nan, 'maxavD', nan, 'paramSD', nan, ...
  'sizeFit', struct(), 'durFit', struct(), ...
  'sizeDecision', '', 'durDecision', '', ...
  'sizeVuongRExp', nan, 'sizeVuongPExp', nan, ...
  'durVuongRExp', nan, 'durVuongPExp', nan);

sizeFit = fit_avalanche_power_law(sizes, config);
durFit = fit_avalanche_power_law(durs, config);

metrics.tau = sizeFit.exponent;
metrics.alpha = durFit.exponent;
metrics.decades = sizeFit.decades;
metrics.minavS = sizeFit.fitMin;
metrics.maxavS = sizeFit.fitMax;
metrics.minavD = durFit.fitMin;
metrics.maxavD = durFit.fitMax;
metrics.sizeFit = sizeFit;
metrics.durFit = durFit;
metrics.sizeDecision = get_tail_decision(sizeFit);
metrics.durDecision = get_tail_decision(durFit);
[metrics.sizeVuongRExp, metrics.sizeVuongPExp] = get_vuong_vs_exp(sizeFit);
[metrics.durVuongRExp, metrics.durVuongPExp] = get_vuong_vs_exp(durFit);

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

function decision = get_tail_decision(fitResult)
decision = '';
if isstruct(fitResult) && isfield(fitResult, 'tailComparison') ...
    && isstruct(fitResult.tailComparison) && isfield(fitResult.tailComparison, 'decision')
  decision = fitResult.tailComparison.decision;
end
end

function [vuongR, vuongP] = get_vuong_vs_exp(fitResult)
vuongR = nan;
vuongP = nan;
if ~isstruct(fitResult) || ~isfield(fitResult, 'tailComparison') ...
    || ~isstruct(fitResult.tailComparison) || ~isfield(fitResult.tailComparison, 'vuongVsExp')
  return;
end
vuong = fitResult.tailComparison.vuongVsExp;
if isfield(vuong, 'R')
  vuongR = vuong.R;
end
if isfield(vuong, 'pValue')
  vuongP = vuong.pValue;
end
end
