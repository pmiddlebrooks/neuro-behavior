function [binEdges, xMin, xMax] = build_shared_histogram_bin_edges(allVals, nBinsTarget, forcedLimits)
% BUILD_SHARED_HISTOGRAM_BIN_EDGES - Shared histogram bin edges and x-limits
%
% Variables:
%   allVals       - Pooled values used to set bin range
%   nBinsTarget   - Target number of bins (default 28)
%   forcedLimits  - Optional [xMin, xMax] to include in limits
%
% Goal:
%   Consistent histogram x-axis padding and bin edges across figures.

if nargin < 2 || isempty(nBinsTarget)
  nBinsTarget = 28;
end
if nargin < 3
  forcedLimits = [];
end

if isempty(allVals)
  if ~isempty(forcedLimits)
    xMin = forcedLimits(1);
    xMax = forcedLimits(2);
  else
    xMin = 0;
    xMax = 1;
  end
else
  xMin = min(allVals);
  xMax = max(allVals);
  if ~isempty(forcedLimits)
    xMin = min(xMin, forcedLimits(1));
    xMax = max(xMax, forcedLimits(2));
  end
end

xSpan = xMax - xMin;
if xSpan <= 0 || ~isfinite(xSpan)
  pad = max(0.5, abs(xMin) * 0.05 + eps);
  xMin = xMin - pad;
  xMax = xMax + pad;
else
  pad = 0.03 * xSpan;
  xMin = xMin - pad;
  xMax = xMax + pad;
end

nBins = max(8, round(nBinsTarget));
binEdges = linspace(xMin, xMax, nBins + 1);
end
