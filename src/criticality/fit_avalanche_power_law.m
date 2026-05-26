function fitResult = fit_avalanche_power_law(values, config)
% FIT_AVALANCHE_POWER_LAW - Power-law fit for avalanche sizes or durations
%
% Variables:
%   values - Positive scalar vector (sizes or durations)
%   config - Struct with fields:
%     .powerLawFitMethod - 'clauset', 'plfit2023', or 'hybrid' (default 'clauset')
%     .clausetPlfitPath   - Path to Clauset toolbox MATLAB Code folder
%     .plfit2023Path      - Path containing plfit2023.m (hybrid / plfit2023)
%     .gofThreshold       - For plfit2023 and hybrid (default 0.8)
%     .runClausetPlpva    - If true, run plpva on Clauset tail (clauset / hybrid)
%
% Goal:
%   Unified interface for Clauset plfit, legacy plfit2023, and hybrid fitting.
%
%   hybrid: plfit2023 selects xmax; data with x <= xmax are passed to plfit for
%   KS-optimal xmin and exponent (Clauset).
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
  case 'hybrid'
    fitResult = fit_power_law_hybrid(values, config);
  otherwise
    error(['Unknown powerLawFitMethod "%s". Use ''clauset'', ''plfit2023'', ', ...
      'or ''hybrid''.'], method);
end
end

function fitResult = fit_power_law_hybrid(values, config)
% FIT_POWER_LAW_HYBRID - plfit2023 xmax, then Clauset plfit on truncated data
%
% Variables:
%   values - Positive avalanche sizes or durations
%   config - clausetPlfitPath, plfit2023Path, gofThreshold, runClausetPlpva
%
% Goal:
%   Use plfit2023 to set an upper cutoff (xmax), then run plfit on x <= xmax
%   to estimate xmin and the power-law exponent.

fitResult = struct('exponent', nan, 'fitMin', nan, 'fitMax', nan, ...
  'decades', nan, 'method', 'hybrid', 'logLikelihood', nan, ...
  'pValue', nan, 'ksGof', nan, 'nTail', 0, ...
  'fitMaxPlfit2023', nan, 'fitMinPlfit2023', nan, 'exponentPlfit2023', nan);

boundsResult = fit_power_law_plfit2023(values, config);
if ~isfinite(boundsResult.fitMax) || boundsResult.fitMax <= 0
  return;
end

xmax = boundsResult.fitMax;
fitResult.fitMaxPlfit2023 = xmax;
fitResult.fitMinPlfit2023 = boundsResult.fitMin;
fitResult.exponentPlfit2023 = boundsResult.exponent;

truncatedValues = values(values <= xmax);
if numel(truncatedValues) < 2 || numel(unique(truncatedValues)) < 2
  return;
end

clausetResult = fit_power_law_clauset(truncatedValues, config);
if ~isfinite(clausetResult.exponent) || ~isfinite(clausetResult.fitMin)
  return;
end

fitResult.exponent = clausetResult.exponent;
fitResult.fitMin = clausetResult.fitMin;
fitResult.fitMax = xmax;
fitResult.logLikelihood = clausetResult.logLikelihood;
fitResult.pValue = clausetResult.pValue;
fitResult.ksGof = clausetResult.ksGof;

fitValues = round(truncatedValues(:));
fitValues = fitValues(fitValues > 0);
tailMask = fitValues >= fitResult.fitMin & fitValues <= fitResult.fitMax;
fitResult.nTail = sum(tailMask);

if fitResult.fitMax > fitResult.fitMin
  fitResult.decades = log10(fitResult.fitMax / fitResult.fitMin);
end
end

function fitResult = fit_power_law_clauset(values, config)
% FIT_POWER_LAW_CLAUSET - Clauset et al. plfit (KS-optimal xmin)

fitResult = struct('exponent', nan, 'fitMin', nan, 'fitMax', nan, ...
  'decades', nan, 'method', 'clauset', 'logLikelihood', nan, ...
  'pValue', nan, 'ksGof', nan, 'nTail', 0);

clausetPath = '';
if isfield(config, 'clausetPlfitPath')
  clausetPath = config.clausetPlfitPath;
end
[clausetPath, ~] = resolve_power_law_paths(clausetPath, '');
setup_clauset_plfit_path(clausetPath);

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

setup_plfit2023_path(config);

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
% SETUP_CLAUSET_PLFIT_PATH - Prepend Clauset plfit and numeric zeta support

supportPath = fullfile(fileparts(mfilename('fullpath')), 'clauset_support');
if exist(supportPath, 'dir')
  addpath(supportPath, '-begin');
end

if ~exist(clausetPlfitPath, 'dir')
  error('Clauset plfit path not found: %s', clausetPlfitPath);
end
addpath(clausetPlfitPath, '-begin');

if exist('plfit', 'file') ~= 2
  error('plfit.m not found after addpath(%s).', clausetPlfitPath);
end
if exist('zeta', 'file') ~= 2
  error(['zeta.m not found. Add Symbolic Math Toolbox, include zeta.m in the ', ...
    'Clauset toolbox, or use criticality/clauset_support/zeta.m.']);
end
end

function setup_plfit2023_path(config)
% SETUP_PLFIT2023_PATH - Ensure plfit2023.m is on the path

if exist('plfit2023', 'file') == 2
  return;
end

plfit2023Path = '';
if isfield(config, 'plfit2023Path')
  plfit2023Path = config.plfit2023Path;
end
[~, plfit2023Path] = resolve_power_law_paths('', plfit2023Path);
addpath(plfit2023Path, '-begin');

if exist('plfit2023', 'file') ~= 2
  error('plfit2023.m not found after addpath(%s).', plfit2023Path);
end
end
