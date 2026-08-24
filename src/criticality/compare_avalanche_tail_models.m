function comparison = compare_avalanche_tail_models(values, fitResult)
% COMPARE_AVALANCHE_TAIL_MODELS - Test power-law vs exponential / lognormal / truncated PL
%
% Variables:
%   values    - Positive avalanche sizes or durations (same units as the PL fit)
%   fitResult - Struct from fit_avalanche_power_law (exponent, fitMin, fitMax)
%
% Goal:
%   On the same discrete support [fitMin, fitMax] used for the reported power-law,
%   compare that power-law to a truncated exponential, a truncated lognormal, and
%   a power-law with exponential cutoff. Uses Vuong likelihood-ratio tests for
%   non-nested pairs and a nested LR test for truncated PL vs pure PL (Clauset,
%   Shalizi, Newman 2009). Does not re-select xmin/xmax.
%
% Returns:
%   comparison - Struct with per-model log-likelihood / AIC, Vuong stats, KS D,
%                a decision string, and CCDF curves for overlay.

comparison = empty_tail_comparison();
if nargin < 2 || ~isstruct(fitResult)
  return;
end

values = values(:);
values = values(isfinite(values) & values > 0);
fitMin = nan;
fitMax = nan;
plExponent = nan;
if isfield(fitResult, 'fitMin')
  fitMin = fitResult.fitMin;
end
if isfield(fitResult, 'fitMax')
  fitMax = fitResult.fitMax;
end
if isfield(fitResult, 'exponent')
  plExponent = fitResult.exponent;
end
if ~(isfinite(fitMin) && isfinite(fitMax) && fitMax > fitMin && isfinite(plExponent) && plExponent > 1)
  return;
end

tailValues = values(values >= fitMin & values <= fitMax);
tailValues = round(tailValues);
tailValues = tailValues(tailValues > 0);
nTail = numel(tailValues);
comparison.nTail = nTail;
comparison.fitMin = fitMin;
comparison.fitMax = fitMax;
comparison.plExponent = plExponent;
if isfield(fitResult, 'decades') && isfinite(fitResult.decades)
  comparison.decades = fitResult.decades;
elseif fitMin > 0
  comparison.decades = log10(fitMax / fitMin);
end

minTailN = 20;
minDecades = 1.0;
if nTail < minTailN || ~(isfinite(comparison.decades) && comparison.decades >= minDecades)
  comparison.decision = 'insufficientRange';
  comparison.decisionReason = sprintf( ...
    'nTail=%d, decades=%.2f (need nTail>=%d and decades>=%.1f to distinguish)', ...
    nTail, comparison.decades, minTailN, minDecades);
  return;
end

xmin = max(1, round(fitMin));
xmax = max(xmin + 1, round(fitMax));
support = (xmin:xmax)';
z = tailValues;
z = max(xmin, min(xmax, z));

% Reported power-law (do not re-fit tau)
logPmfPl = discrete_log_pmf_power_law(support, plExponent);
llPlEach = lookup_log_pmf(z, support, logPmfPl);
if any(~isfinite(llPlEach))
  comparison.decision = 'fitFailed';
  comparison.decisionReason = 'Power-law likelihood was non-finite on the fitted range.';
  return;
end
llPl = sum(llPlEach);
comparison.powerLaw.logLikelihood = llPl;
comparison.powerLaw.nParams = 1;
comparison.powerLaw.aic = aic_from_ll(llPl, 1);
comparison.powerLaw.ksD = ks_distance_from_log_pmf(z, support, logPmfPl);
comparison.powerLaw.logPmf = logPmfPl;

% Truncated discrete exponential
[lambdaExp, logPmfExp, llExp, llExpEach, okExp] = fit_truncated_exponential(z, support);
comparison.exponential.lambda = lambdaExp;
comparison.exponential.logLikelihood = llExp;
comparison.exponential.nParams = 1;
comparison.exponential.aic = aic_from_ll(llExp, 1);
comparison.exponential.ksD = ks_distance_from_log_pmf(z, support, logPmfExp);
comparison.exponential.ok = okExp;
comparison.exponential.logPmf = logPmfExp;
if okExp
  comparison.vuongVsExp = vuong_test(llPlEach, llExpEach);
else
  comparison.vuongVsExp = empty_vuong();
end

% Truncated discrete lognormal
[muLn, sigmaLn, logPmfLn, llLn, llLnEach, okLn] = fit_truncated_lognormal(z, support);
comparison.lognormal.mu = muLn;
comparison.lognormal.sigma = sigmaLn;
comparison.lognormal.logLikelihood = llLn;
comparison.lognormal.nParams = 2;
comparison.lognormal.aic = aic_from_ll(llLn, 2);
comparison.lognormal.ksD = ks_distance_from_log_pmf(z, support, logPmfLn);
comparison.lognormal.ok = okLn;
comparison.lognormal.logPmf = logPmfLn;
if okLn
  comparison.vuongVsLognormal = vuong_test(llPlEach, llLnEach);
else
  comparison.vuongVsLognormal = empty_vuong();
end

% Power-law with exponential cutoff (nested in pure PL when lambda = 0)
[tauCut, lambdaCut, logPmfCut, llCut, okCut] = fit_truncated_power_law(z, support, plExponent);
comparison.truncatedPowerLaw.exponent = tauCut;
comparison.truncatedPowerLaw.lambda = lambdaCut;
comparison.truncatedPowerLaw.logLikelihood = llCut;
comparison.truncatedPowerLaw.nParams = 2;
comparison.truncatedPowerLaw.aic = aic_from_ll(llCut, 2);
comparison.truncatedPowerLaw.ksD = ks_distance_from_log_pmf(z, support, logPmfCut);
comparison.truncatedPowerLaw.ok = okCut;
comparison.truncatedPowerLaw.logPmf = logPmfCut;
if okCut && isfinite(llCut) && isfinite(llPl)
  lrStat = 2 * (llCut - llPl);
  comparison.nestedTruncatedVsPl.lrStat = lrStat;
  if lrStat > 0
    comparison.nestedTruncatedVsPl.pValue = 1 - gammainc(lrStat / 2, 0.5, 'lower');
  else
    comparison.nestedTruncatedVsPl.pValue = 1;
  end
end

comparison.support = support;
comparison = add_ccdf_curves(comparison, xmin, xmax);
comparison = assign_tail_decision(comparison, minTailN, minDecades);
end

function comparison = empty_tail_comparison()
comparison = struct();
comparison.nTail = 0;
comparison.fitMin = nan;
comparison.fitMax = nan;
comparison.plExponent = nan;
comparison.decades = nan;
comparison.decision = 'notTested';
comparison.decisionReason = '';
comparison.powerLaw = empty_model_fit(1);
comparison.exponential = empty_model_fit(1);
comparison.exponential.lambda = nan;
comparison.exponential.ok = false;
comparison.lognormal = empty_model_fit(2);
comparison.lognormal.mu = nan;
comparison.lognormal.sigma = nan;
comparison.lognormal.ok = false;
comparison.truncatedPowerLaw = empty_model_fit(2);
comparison.truncatedPowerLaw.exponent = nan;
comparison.truncatedPowerLaw.lambda = nan;
comparison.truncatedPowerLaw.ok = false;
comparison.vuongVsExp = empty_vuong();
comparison.vuongVsLognormal = empty_vuong();
comparison.nestedTruncatedVsPl = struct('lrStat', nan, 'pValue', nan);
comparison.support = [];
comparison.xGrid = [];
comparison.ccdfPowerLaw = [];
comparison.ccdfExponential = [];
comparison.ccdfLognormal = [];
comparison.ccdfTruncatedPowerLaw = [];
end

function modelFit = empty_model_fit(nParams)
modelFit = struct('logLikelihood', nan, 'nParams', nParams, ...
  'aic', nan, 'ksD', nan, 'logPmf', []);
end

function vuong = empty_vuong()
vuong = struct('R', nan, 'stat', nan, 'pValue', nan, 'preferPowerLaw', false, ...
  'preferAlternative', false);
end

function aic = aic_from_ll(logLikelihood, nParams)
aic = nan;
if isfinite(logLikelihood)
  aic = 2 * nParams - 2 * logLikelihood;
end
end

function logPmf = discrete_log_pmf_power_law(support, exponent)
logUnnorm = -exponent * log(support);
logPmf = logUnnorm - log_sum_exp(logUnnorm);
end

function [lambdaHat, logPmf, ll, llEach, ok] = fit_truncated_exponential(z, support)
lambdaHat = nan;
logPmf = [];
ll = nan;
llEach = nan(size(z));
ok = false;
lambda0 = 1 / max(mean(z) - min(support) + 1, eps);
nllFun = @(logLam) exponential_nll(exp(logLam), z, support);
[logLamHat, nll, exitFlag] = run_bounded_search(nllFun, log(max(lambda0, 1e-8)));
if ~(isfinite(nll) && exitFlag > 0)
  return;
end
lambdaHat = exp(logLamHat);
logPmf = discrete_log_pmf_exponential(support, lambdaHat);
llEach = lookup_log_pmf(z, support, logPmf);
ll = sum(llEach);
ok = all(isfinite(llEach));
end

function nll = exponential_nll(lambda, z, support)
if ~(isfinite(lambda) && lambda > 0)
  nll = inf;
  return;
end
logPmf = discrete_log_pmf_exponential(support, lambda);
llEach = lookup_log_pmf(z, support, logPmf);
if any(~isfinite(llEach))
  nll = inf;
  return;
end
nll = -sum(llEach);
end

function logPmf = discrete_log_pmf_exponential(support, lambda)
logUnnorm = -lambda * support;
logPmf = logUnnorm - log_sum_exp(logUnnorm);
end

function [muHat, sigmaHat, logPmf, ll, llEach, ok] = fit_truncated_lognormal(z, support)
muHat = nan;
sigmaHat = nan;
logPmf = [];
ll = nan;
llEach = nan(size(z));
ok = false;
logZ = log(z);
mu0 = mean(logZ);
sigma0 = max(std(logZ, 1), 0.1);
nllFun = @(p) lognormal_nll(p(1), exp(p(2)), z, support);
[pHat, nll, exitFlag] = run_bounded_search(nllFun, [mu0; log(sigma0)]);
if ~(isfinite(nll) && exitFlag > 0)
  return;
end
muHat = pHat(1);
sigmaHat = exp(pHat(2));
logPmf = discrete_log_pmf_lognormal(support, muHat, sigmaHat);
llEach = lookup_log_pmf(z, support, logPmf);
ll = sum(llEach);
ok = all(isfinite(llEach));
end

function nll = lognormal_nll(mu, sigma, z, support)
if ~(isfinite(mu) && isfinite(sigma) && sigma > 0)
  nll = inf;
  return;
end
logPmf = discrete_log_pmf_lognormal(support, mu, sigma);
llEach = lookup_log_pmf(z, support, logPmf);
if any(~isfinite(llEach))
  nll = inf;
  return;
end
nll = -sum(llEach);
end

function logPmf = discrete_log_pmf_lognormal(support, mu, sigma)
logUnnorm = -log(support) - ((log(support) - mu) .^ 2) / (2 * sigma ^ 2);
logPmf = logUnnorm - log_sum_exp(logUnnorm);
end

function [tauHat, lambdaHat, logPmf, ll, ok] = fit_truncated_power_law(z, support, tau0)
tauHat = nan;
lambdaHat = nan;
logPmf = [];
ll = nan;
ok = false;
lambda0 = 1 / max(mean(z), 1);
nllFun = @(p) truncated_pl_nll(p(1), exp(p(2)), z, support);
[pHat, nll, exitFlag] = run_bounded_search(nllFun, [tau0; log(max(lambda0, 1e-8))]);
if ~(isfinite(nll) && exitFlag > 0)
  return;
end
tauHat = pHat(1);
lambdaHat = exp(pHat(2));
logPmf = discrete_log_pmf_truncated_pl(support, tauHat, lambdaHat);
llEach = lookup_log_pmf(z, support, logPmf);
ll = sum(llEach);
ok = all(isfinite(llEach));
end

function nll = truncated_pl_nll(tau, lambda, z, support)
if ~(isfinite(tau) && tau > 1 && isfinite(lambda) && lambda >= 0)
  nll = inf;
  return;
end
logPmf = discrete_log_pmf_truncated_pl(support, tau, lambda);
llEach = lookup_log_pmf(z, support, logPmf);
if any(~isfinite(llEach))
  nll = inf;
  return;
end
nll = -sum(llEach);
end

function logPmf = discrete_log_pmf_truncated_pl(support, tau, lambda)
logUnnorm = -tau * log(support) - lambda * support;
logPmf = logUnnorm - log_sum_exp(logUnnorm);
end

function [pHat, nll, exitFlag] = run_bounded_search(nllFun, p0)
pHat = p0;
nll = inf;
exitFlag = 0;
opts = optimset('Display', 'off', 'MaxFunEvals', 400, 'MaxIter', 200, 'TolX', 1e-6, ...
  'TolFun', 1e-6);
try
  [pHat, nll, exitFlag] = fminsearch(nllFun, p0, opts);
catch
  return;
end
end

function llEach = lookup_log_pmf(z, support, logPmf)
llEach = nan(size(z));
if isempty(logPmf) || numel(logPmf) ~= numel(support)
  return;
end
[tf, loc] = ismember(z, support);
llEach(tf) = logPmf(loc(tf));
end

function ksD = ks_distance_from_log_pmf(z, support, logPmf)
ksD = nan;
if isempty(logPmf) || any(~isfinite(logPmf))
  return;
end
pmf = exp(logPmf - max(logPmf));
pmf = pmf / sum(pmf);
modelCdf = cumsum(pmf);
[unq, ~, ic] = unique(z);
empCounts = accumarray(ic, 1);
empCdfUnq = cumsum(empCounts) / numel(z);
[tf, loc] = ismember(unq, support);
if ~all(tf)
  return;
end
ksD = max(abs(empCdfUnq - modelCdf(loc)));
end

function vuong = vuong_test(llModelA, llModelB)
% Vuong (1989) / Clauset 2009: A is power-law, B is the alternative
vuong = empty_vuong();
lrEach = llModelA(:) - llModelB(:);
lrEach = lrEach(isfinite(lrEach));
n = numel(lrEach);
if n < 8
  return;
end
rBar = mean(lrEach);
sigma = std(lrEach, 1);
if ~(isfinite(sigma) && sigma > 0)
  return;
end
vuong.R = sum(lrEach);
vuong.stat = rBar * sqrt(n) / sigma;
vuong.pValue = erfc(abs(vuong.stat) / sqrt(2));
vuongAlpha = 0.1;
vuong.preferPowerLaw = vuong.pValue < vuongAlpha && vuong.R > 0;
vuong.preferAlternative = vuong.pValue < vuongAlpha && vuong.R < 0;
end

function comparison = add_ccdf_curves(comparison, xmin, xmax)
nDec = max(comparison.decades, 0.5);
nPts = min(200, max(20, round(40 * nDec)));
xGrid = unique(round(logspace(log10(xmin), log10(xmax), nPts)'));
xGrid = xGrid(xGrid >= xmin & xGrid <= xmax);
comparison.xGrid = xGrid;
comparison.ccdfPowerLaw = discrete_ccdf_from_log_pmf(xGrid, comparison.support, comparison.powerLaw.logPmf);
comparison.ccdfExponential = discrete_ccdf_from_log_pmf(xGrid, comparison.support, comparison.exponential.logPmf);
comparison.ccdfLognormal = discrete_ccdf_from_log_pmf(xGrid, comparison.support, comparison.lognormal.logPmf);
comparison.ccdfTruncatedPowerLaw = discrete_ccdf_from_log_pmf(xGrid, comparison.support, ...
  comparison.truncatedPowerLaw.logPmf);
end

function ccdf = discrete_ccdf_from_log_pmf(xGrid, support, logPmf)
ccdf = nan(size(xGrid));
if isempty(logPmf) || any(~isfinite(logPmf))
  return;
end
pmf = exp(logPmf - max(logPmf));
pmf = pmf / sum(pmf);
sf = flipud(cumsum(flipud(pmf)));
[tf, loc] = ismember(xGrid, support);
ccdf(tf) = sf(loc(tf));
end

function comparison = assign_tail_decision(comparison, minTailN, minDecades)
if comparison.nTail < minTailN || ~(isfinite(comparison.decades) && comparison.decades >= minDecades)
  comparison.decision = 'insufficientRange';
  comparison.decisionReason = sprintf('nTail=%d, decades=%.2f', comparison.nTail, comparison.decades);
  return;
end

vuongExp = comparison.vuongVsExp;
vuongLn = comparison.vuongVsLognormal;
nested = comparison.nestedTruncatedVsPl;
aicPl = comparison.powerLaw.aic;
aicExp = comparison.exponential.aic;
aicLn = comparison.lognormal.aic;
aicCut = comparison.truncatedPowerLaw.aic;

if vuongExp.preferAlternative
  comparison.decision = 'exponentialPreferred';
  comparison.decisionReason = sprintf( ...
    'Vuong PL vs exponential: R=%.2f, p=%.3f (exponential preferred)', ...
    vuongExp.R, vuongExp.pValue);
  return;
end
if vuongLn.preferAlternative
  comparison.decision = 'lognormalPreferred';
  comparison.decisionReason = sprintf( ...
    'Vuong PL vs lognormal: R=%.2f, p=%.3f (lognormal preferred)', ...
    vuongLn.R, vuongLn.pValue);
  return;
end

truncBeatsPl = isfinite(nested.pValue) && nested.pValue < 0.1 && isfinite(aicCut) ...
  && isfinite(aicPl) && aicCut < aicPl;
truncBeatsExp = isfinite(aicCut) && isfinite(aicExp) && aicCut < aicExp;
if truncBeatsPl && truncBeatsExp
  comparison.decision = 'truncatedPowerLawPreferred';
  comparison.decisionReason = sprintf( ...
    ['Nested truncated-PL vs PL: LR=%.2f, p=%.3f; AIC truncated=%.1f, PL=%.1f. ', ...
    'Scale-free body with an exponential cutoff (typical finite-size form).'], ...
    nested.lrStat, nested.pValue, aicCut, aicPl);
  return;
end

if ~vuongExp.preferPowerLaw
  comparison.decision = 'indistinguishableFromExponential';
  comparison.decisionReason = sprintf( ...
    'Vuong PL vs exponential not significant (R=%.2f, p=%.3f)', ...
    vuongExp.R, vuongExp.pValue);
  return;
end

if isfinite(aicLn) && isfinite(aicPl) && aicLn < aicPl && ~vuongLn.preferPowerLaw
  comparison.decision = 'indistinguishableFromLognormal';
  comparison.decisionReason = sprintf( ...
    'Vuong PL vs lognormal not significant (R=%.2f, p=%.3f); AIC lognormal=%.1f, PL=%.1f', ...
    vuongLn.R, vuongLn.pValue, aicLn, aicPl);
  return;
end

comparison.decision = 'powerLawPreferred';
comparison.decisionReason = sprintf( ...
  'Vuong favors PL vs exponential (R=%.2f, p=%.3f) and vs lognormal (R=%.2f, p=%.3f)', ...
  vuongExp.R, vuongExp.pValue, vuongLn.R, vuongLn.pValue);
end

function y = log_sum_exp(v)
v = v(isfinite(v));
if isempty(v)
  y = nan;
  return;
end
m = max(v);
y = m + log(sum(exp(v - m)));
end
