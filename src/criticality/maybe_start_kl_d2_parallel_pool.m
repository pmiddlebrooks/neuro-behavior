function maybe_start_kl_d2_parallel_pool(d2Method, klErrBars, klParallel, nWorkers)
% MAYBE_START_KL_D2_PARALLEL_POOL Start parpool for calc_db error-bar gradient
%
% Variables:
%   d2Method   - 'euclidean' or 'kl'
%   klErrBars  - If true, S2.5 Hessian error bars (the parfor lives here)
%   klParallel - If true, use parallel workers inside calc_db
%   nWorkers   - Optional worker count. [] or omitted keeps an existing pool,
%                or starts min(3, feature('numcores')) if none. Capped at
%                available cores. If a pool exists with a different size,
%                it is deleted and restarted at nWorkers.
%
% Goal:
%   calc_db only parallelizes the error-bar gradient, so a pool is needed
%   only for KL d2 with both klErrBars and klParallel.

if nargin < 3
  return;
end
if ~strcmpi(strtrim(char(d2Method)), 'kl')
  return;
end
if ~logical(klErrBars) || ~logical(klParallel)
  return;
end
if nargin < 4
  nWorkers = [];
end

nCores = feature('numcores');
requestedWorkers = resolve_kl_d2_n_workers(nWorkers, nCores);

currentPool = gcp('nocreate');
if isempty(currentPool)
  if isempty(requestedWorkers)
    requestedWorkers = min(3, nCores);
  end
  parpool('local', requestedWorkers);
  fprintf('Started parallel pool with %d workers for calc_db gradient error bars\n', ...
    requestedWorkers);
elseif ~isempty(requestedWorkers) && currentPool.NumWorkers ~= requestedWorkers
  delete(currentPool);
  parpool('local', requestedWorkers);
  fprintf('Resized parallel pool to %d workers for calc_db gradient error bars\n', ...
    requestedWorkers);
else
  fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
end
end

function requestedWorkers = resolve_kl_d2_n_workers(nWorkers, nCores)
% RESOLVE_KL_D2_N_WORKERS Validate nWorkers and cap at available cores
%
% Variables:
%   nWorkers - Requested worker count; [] means use helper default
%   nCores   - feature('numcores')
%
% Goal:
%   Return a positive integer worker count, or [] if the caller left it unset.

requestedWorkers = [];
if isempty(nWorkers)
  return;
end
nWorkers = round(double(nWorkers(1)));
if ~isfinite(nWorkers) || nWorkers < 1
  error('nWorkers must be a positive integer when runParallel is true.');
end
requestedWorkers = min(nWorkers, nCores);
if requestedWorkers < nWorkers
  fprintf('nWorkers=%d exceeds %d cores; using %d workers\n', ...
    nWorkers, nCores, requestedWorkers);
end
end
