function xZ = zscore_columns_finite(xIn)
% ZSCORE_COLUMNS_FINITE Per-column z-score with finite output (zero-variance columns -> zeros).
%
% Variables:
%   - xIn: [nRows x nCols] numeric matrix
% Goal:
%   Match zscore(...,0,1) when std > 0; avoid NaN/Inf when a column is constant or near-constant.
% Returns:
%   - xZ: same size as xIn, all finite

x = double(xIn);
mu = mean(x, 1);
sigma = std(x, 0, 1);
sigma(~isfinite(sigma)) = eps;
sigma = max(sigma, eps);
xZ = (x - mu) ./ sigma;
end
