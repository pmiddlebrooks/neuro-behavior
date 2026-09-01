function [d2Val, d2Err, exitStatus] = compute_d2_from_pop_trace(popTrace, pOrder, critType, binSize, d2Method, d2Opts)
% COMPUTE_D2_FROM_POP_TRACE Euclidean or KL-rate d2 from a population trace
%
% Variables:
%   popTrace  - Population spike-count / SDF vector for one window
%   pOrder    - AR model order
%   critType  - tRG beta (usually 2)
%   binSize   - Bin duration in seconds (deltaT for KL-rate d2)
%   d2Method  - 'euclidean' (Yule-Walker + getFixedPointDistance2) or 'kl'
%               (Sooter et al. calc_db KL-rate, bits/s)
%   d2Opts    - Optional struct: klFitMethod, klErrBars, klParallel
%
% Goal:
%   One call site for both d2 implementations. KL error bars follow S2.5 and
%   require MaxLikelihood; they are off unless klErrBars is true.
%
% Returns:
%   d2Val      - Distance to criticality
%   d2Err      - KL Hessian error (NaN for Euclidean or when error bars off)
%   exitStatus - 0 = ok; 1 = failed / explosive KL model

if nargin < 5 || isempty(d2Method)
  d2Method = 'euclidean';
end
if nargin < 6 || isempty(d2Opts) || ~isstruct(d2Opts)
  d2Opts = struct();
end

d2Val = nan;
d2Err = nan;
exitStatus = 1;

d2Method = normalize_d2_method(d2Method);
popTrace = double(popTrace(:));
if numel(popTrace) < (pOrder + 2)
  return;
end
popStd = nanstd(popTrace);
if ~any(isfinite(popTrace)) || ~(popStd > 0)
  return;
end

if strcmp(d2Method, 'kl')
  [d2Val, d2Err, exitStatus] = compute_kl_d2(popTrace, pOrder, critType, binSize, d2Opts);
else
  try
    [varphi, ~] = myYuleWalker3(popTrace, pOrder);
    d2Val = getFixedPointDistance2(pOrder, critType, varphi);
    exitStatus = 0;
  catch
    d2Val = nan;
    exitStatus = 1;
  end
end
end

function d2Method = normalize_d2_method(d2Method)
d2Method = lower(strtrim(char(d2Method)));
switch d2Method
  case {'kl', 'klrate', 'sooter', 'calc_db'}
    d2Method = 'kl';
  otherwise
    d2Method = 'euclidean';
end
end

function [d2Val, d2Err, exitStatus] = compute_kl_d2(popTrace, pOrder, critType, binSize, d2Opts)
add_prox_crit_toolkit_path();

klFitMethod = 'MaxLikelihood';
if isfield(d2Opts, 'klFitMethod') && ~isempty(d2Opts.klFitMethod)
  klFitMethod = char(d2Opts.klFitMethod);
end
klErrBars = false;
if isfield(d2Opts, 'klErrBars') && ~isempty(d2Opts.klErrBars)
  klErrBars = logical(d2Opts.klErrBars);
end
klParallel = false;
if isfield(d2Opts, 'klParallel') && ~isempty(d2Opts.klParallel)
  klParallel = logical(d2Opts.klParallel);
end
if klErrBars && ~strcmp(klFitMethod, 'MaxLikelihood')
  error('KL d2 error bars require klFitMethod = ''MaxLikelihood''.');
end

dbopt = struct();
dbopt.fit_method = klFitMethod;
dbopt.with_err_bars = klErrBars;
dbopt.with_QC = false;
dbopt.with_parallel = klParallel;

try
  [~, dbVal, sdVal, ~, ~, ~, ~, exitStatus] = evalc( ...
    'calc_db(popTrace, pOrder, binSize, critType, dbopt)');
  d2Val = dbVal;
  d2Err = sdVal;
catch
  d2Val = nan;
  d2Err = nan;
  exitStatus = 1;
end

if ~isreal(d2Val) || ~isfinite(d2Val)
  d2Val = nan;
end
if ~isreal(d2Err) || ~isfinite(d2Err) || d2Err < 0
  d2Err = nan;
end
end
