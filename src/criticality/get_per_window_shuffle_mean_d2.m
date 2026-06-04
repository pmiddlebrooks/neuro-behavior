function shufVec = get_per_window_shuffle_mean_d2(results, areaIdx, useLog10D2)
% GET_PER_WINDOW_SHUFFLE_MEAN_D2 - Per-window summary of shuffled d2
%
% Variables:
%   results      - Output from criticality_ar_analysis
%   areaIdx      - Area index
%   useLog10D2   - If true, apply log10 to positive values only
%
% Goal:
%   Standard mode: mean across permutations per window.
%   Subsampling mode: mean across subsamples of (mean permutations per subsample).

if nargin < 3 || isempty(useLog10D2)
  useLog10D2 = false;
end

shufVec = [];
if ~isfield(results, 'd2Permuted') || areaIdx > numel(results.d2Permuted) ...
    || isempty(results.d2Permuted{areaIdx})
  return;
end

d2Perm = results.d2Permuted{areaIdx};
if useLog10D2
  d2Perm = log10_safe_numeric(d2Perm);
end

shufVec = get_per_window_shuffle_mean_matrix(d2Perm, results);
end

function y = log10_safe_numeric(x)
% LOG10_SAFE_NUMERIC - log10 with NaN for non-positive values

validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end
