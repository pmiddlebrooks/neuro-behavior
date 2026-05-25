function fitResult = fit_avalanche_power_law(values, config)
% FIT_AVALANCHE_POWER_LAW - Power-law fit for avalanche sizes or durations
%
% Variables:
%   values - Positive scalar vector (sizes or durations)
%   config - Struct with fields:
%     .powerLawFitMethod - 'clauset' (default) or 'plfit2023'
%     .clausetPlfitPath   - Path to Clauset toolbox MATLAB Code folder
%     .gofThreshold       - For plfit2023 only (default 0.8)
%     .runClausetPlpva    - If true, run plpva (slow; clauset only)
%
% Goal:
%   Unified interface for Clauset plfit and legacy plfit2023 used across
%   avalanche criticality analyses.
%
% Returns:
%   fitResult - Struct with exponent, fitMin, fitMax, decades, method, etc.

fitResult = struct('exponent', nan, 'fitMin', nan, 'fitMax', nan, ...
  'decades', nan, 'method', '', 'logLikelihood', nan, ...
  'pValue', nan, 'ksGof', nan, 'nTail', 0);

values = values(isfinite(values) & values > 0);
if numel(values) < 2
  return;
end

method = 'clauset';
if nargin >= 2 && isstruct(config)
  if isfield(config, 'powerLawFitMethod') && ~isempty(config.powerLawFitMethod)
    method = lower(strtrim(config.powerLawFitMethod));
  end
else
  config = struct();
end
fitResult.method = method;

switch method
  case 'clauset'
    fitResult = fit_power_law_clauset(values, config);
  case 'plfit2023'
    fitResult = fit_power_law_plfit2023(values, config);
  otherwise
    error('Unknown powerLawFitMethod "%s". Use ''clauset'' or ''plfit2023''.', method);
end
end

function fitResult = fit_power_law_clauset(values, config)
% FIT_POWER_LAW_CLAUSET - Clauset et al. plfit (KS-optimal xmin)

fitResult = struct('exponent', nan, 'fitMin', nan, 'fitMax', nan, ...
  'decades', nan, 'method', 'clauset', 'logLikelihood', nan, ...
  'pValue', nan, 'ksGof', nan, 'nTail', 0);

if ~isfield(config, 'clausetPlfitPath') || isempty(config.clausetPlfitPath)
  error('config.clausetPlfitPath is required when powerLawFitMethod is ''clauset''.');
end
% setup_clauset_plfit_path(config.clausetPlfitPath);

fitValues = round(values(:));
fitValues = fitValues(fitValues > 0);
if numel(fitValues) < 2 || numel(unique(fitValues)) < 2
  return;
end

[alphaHat, xminHat, logL] = plfit(fitValues, 'nosmall');
if isempty(alphaHat) || ~isfinite(alphaHat) || isempty(xminHat) || ~isfinite(xminHat)
  return;
end

tailValues = fitValues(fitValues >= xminHat);
if isempty(tailValues)
  return;
end

fitMax = max(tailValues);
fitResult.exponent = alphaHat;
fitResult.fitMin = xminHat;
fitResult.fitMax = fitMax;
fitResult.logLikelihood = logL;
fitResult.nTail = numel(tailValues);
if fitMax > xminHat
  fitResult.decades = log10(fitMax / xminHat);
end

runPlpva = isfield(config, 'runClausetPlpva') && config.runClausetPlpva;
if runPlpva && exist('plpva', 'file') == 2
  [pVal, ksGof] = plpva(fitValues, xminHat, 'silent');
  fitResult.pValue = pVal;
  fitResult.ksGof = ksGof;
end
end

function fitResult = fit_power_law_plfit2023(values, config)
% FIT_POWER_LAW_PLFIT2023 - Legacy plfit2023 wrapper

fitResult = struct('exponent', nan, 'fitMin', nan, 'fitMax', nan, ...
  'decades', nan, 'method', 'plfit2023', 'logLikelihood', nan, ...
  'pValue', nan, 'ksGof', nan, 'nTail', 0);

gof = 0.8;
if isfield(config, 'gofThreshold') && ~isempty(config.gofThreshold)
  gof = config.gofThreshold;
end

plotAv = 0;
[exponent, plr, fitMin, fitMax, ~, ~, ~] = plfit2023(values(:), gof, plotAv, 0);
if ~isfinite(exponent)
  return;
end

fitResult.exponent = exponent;
fitResult.fitMin = fitMin;
fitResult.fitMax = fitMax;
fitResult.decades = plr;
fitResult.nTail = sum(values >= fitMin & values <= fitMax);
end

function setup_clauset_plfit_path(clausetPlfitPath)
% SETUP_CLAUSET_PLFIT_PATH - Prepend Clauset toolbox to path

if ~exist(clausetPlfitPath, 'dir')
  error('Clauset plfit path not found: %s', clausetPlfitPath);
end
addpath(clausetPlfitPath, '-begin');
if exist('plfit', 'file') ~= 2
  error('plfit.m not found after addpath(%s).', clausetPlfitPath);
end
end
