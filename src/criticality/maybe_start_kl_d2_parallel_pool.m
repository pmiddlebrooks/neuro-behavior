function maybe_start_kl_d2_parallel_pool(d2Method, klErrBars, klParallel)
% MAYBE_START_KL_D2_PARALLEL_POOL Start parpool for calc_db error-bar gradient
%
% Variables:
%   d2Method   - 'euclidean' or 'kl'
%   klErrBars  - If true, S2.5 Hessian error bars (the parfor lives here)
%   klParallel - If true, use parallel workers inside calc_db
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

currentPool = gcp('nocreate');
if isempty(currentPool)
  parpool('local', min(3, feature('numcores')));
  fprintf('Started parallel pool for calc_db gradient error bars\n');
else
  fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
end
end
