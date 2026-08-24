function overlay_avalanche_alternative_ccdfs(ax, values, fitInfo, plotConfig)
% OVERLAY_AVALANCHE_ALTERNATIVE_CCDFS - Exponential / lognormal / truncated-PL CCDFs
%
% Variables:
%   ax         - Axes handle (same units as values)
%   values     - Observed sizes or durations (plot units)
%   fitInfo    - Fit struct with .fitMin and .tailComparison (native fit units)
%   plotConfig - Uses fitLineWidth; optional .drawAlternativeFits (default true)
%
% Goal:
%   Overlay alternative-model CCDFs on the empirical CCDF, scaled so x matches
%   the plotted units (e.g. duration ms vs duration bins).

if nargin < 4 || isempty(plotConfig)
  plotConfig = struct();
end
if ~isfield(plotConfig, 'drawAlternativeFits') || isempty(plotConfig.drawAlternativeFits)
  plotConfig.drawAlternativeFits = true;
end
if ~plotConfig.drawAlternativeFits
  return;
end
if ~isstruct(fitInfo) || ~isfield(fitInfo, 'tailComparison') || ~isstruct(fitInfo.tailComparison)
  return;
end
comparison = fitInfo.tailComparison;
if ~isfield(comparison, 'xGrid') || isempty(comparison.xGrid)
  return;
end

values = values(:);
values = values(isfinite(values) & values > 0);
if numel(values) < 2
  return;
end
fitMin = nan;
if isfield(fitInfo, 'fitMin')
  fitMin = fitInfo.fitMin;
end
if ~(isfinite(fitMin) && fitMin > 0)
  return;
end

xScale = 1;
if isfield(comparison, 'fitMin') && isfinite(comparison.fitMin) && comparison.fitMin > 0
  xScale = fitMin / comparison.fitMin;
end
xPlot = comparison.xGrid * xScale;
yAtMin = mean(values >= min(xPlot));
if ~(isfinite(yAtMin) && yAtMin > 0)
  return;
end

lineWidth = 1.5;
if isfield(plotConfig, 'fitLineWidth') && isfinite(plotConfig.fitLineWidth)
  lineWidth = max(1, plotConfig.fitLineWidth * 0.7);
end

plot_ccdf_curve(ax, xPlot, comparison.ccdfExponential, yAtMin, ...
  [0.20, 0.45, 0.75], '--', lineWidth, 'Exp');
plot_ccdf_curve(ax, xPlot, comparison.ccdfLognormal, yAtMin, ...
  [0.55, 0.35, 0.75], ':', lineWidth, 'Lognormal');
plot_ccdf_curve(ax, xPlot, comparison.ccdfTruncatedPowerLaw, yAtMin, ...
  [0.15, 0.60, 0.40], '-.', lineWidth, 'Trunc. PL');
end

function plot_ccdf_curve(ax, xPlot, ccdfModel, yAtMin, lineColor, lineStyle, lineWidth, displayName)
if isempty(ccdfModel) || numel(ccdfModel) ~= numel(xPlot)
  return;
end
valid = isfinite(xPlot) & isfinite(ccdfModel) & ccdfModel > 0 & xPlot > 0;
if nnz(valid) < 2
  return;
end
y0 = ccdfModel(find(valid, 1, 'first'));
if ~(isfinite(y0) && y0 > 0)
  return;
end
yPlot = ccdfModel * (yAtMin / y0);
plot(ax, xPlot(valid), yPlot(valid), lineStyle, 'Color', lineColor, ...
  'LineWidth', lineWidth, 'DisplayName', displayName);
end
