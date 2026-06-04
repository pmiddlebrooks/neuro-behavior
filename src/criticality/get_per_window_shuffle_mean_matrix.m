function colMean = get_per_window_shuffle_mean_matrix(permMat, results)
% GET_PER_WINDOW_SHUFFLE_MEAN_MATRIX - Per-window shuffle/surrogate summary
%
% Variables:
%   permMat  - [numWindows x numShuffleCols] permutation or surrogate values
%   results  - Optional results struct with .params (useSubsampling, nShuffles, nSubsamples)
%
% Goal:
%   Standard: mean across columns per window.
%   Subsampling: mean across subsamples of (mean shuffles/surrogates per subsample).

colMean = [];
if isempty(permMat)
  return;
end

permMat = permMat(:, :);
if size(permMat, 2) <= 1
  colMean = permMat(:);
  return;
end

[useSubsampling, nShuffles, nSubsamples] = resolve_shuffle_params_from_results(results, size(permMat, 2));
nCols = size(permMat, 2);

if useSubsampling && nShuffles > 0 && nCols == nShuffles * nSubsamples
  perSubsampleMeans = nan(size(permMat, 1), nSubsamples);
  for s = 1:nSubsamples
    colStart = (s - 1) * nShuffles + 1;
    colEnd = s * nShuffles;
    perSubsampleMeans(:, s) = nanmean(permMat(:, colStart:colEnd), 2);
  end
  colMean = nanmean(perSubsampleMeans, 2);
else
  colMean = nanmean(permMat, 2);
end
end

function [useSubsampling, nShuffles, nSubsamples] = resolve_shuffle_params_from_results(results, nCols)
% RESOLVE_SHUFFLE_PARAMS_FROM_RESULTS - Infer shuffle column layout from params

useSubsampling = false;
nShuffles = nCols;
nSubsamples = 1;

if nargin < 2 || isempty(nCols)
  nCols = 1;
end

if nargin < 1 || isempty(results) || ~isstruct(results)
  return;
end

if isfield(results, 'params')
  params = results.params;
elseif isfield(results, 'useSubsampling')
  params = results;
else
  params = struct();
end

if isfield(params, 'useSubsampling')
  useSubsampling = logical(params.useSubsampling);
end
if isfield(params, 'nShuffles') && ~isempty(params.nShuffles)
  nShuffles = params.nShuffles;
elseif isfield(results, 'nShuffles') && ~isempty(results.nShuffles)
  nShuffles = results.nShuffles;
end
if isfield(params, 'nSurrogates') && ~isempty(params.nSurrogates)
  nShuffles = params.nSurrogates;
end
if isfield(params, 'nSubsamples') && ~isempty(params.nSubsamples)
  nSubsamples = params.nSubsamples;
end

if useSubsampling && nShuffles > 0 && mod(nCols, nShuffles) == 0
  nSubsamples = nCols / nShuffles;
end
end
