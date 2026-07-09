function add_manuscript_scatter_trendline(ax, x, y, plotConfig)
% ADD_MANUSCRIPT_SCATTER_TRENDLINE - Linear fit overlay for scatter plots
%
% Variables:
%   ax         - Axes handle
%   x, y       - Data vectors (finite values only)
%   plotConfig - From fill_manuscript_plot_config
%
% Goal:
%   Add a simple linear trend line when at least three finite points exist.

if nargin < 4 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if numel(x) < 3 || numel(y) < 3
  return;
end

plotColors = manuscript_plot_colors();
fitCoeffs = polyfit(x(:), y(:), 1);
xFit = linspace(min(x), max(x), 100);
plot(ax, xFit, polyval(fitCoeffs, xFit), '-', ...
  'Color', plotColors.trendLine, 'LineWidth', plotConfig.lineWidth, ...
  'HandleVisibility', 'off');
end
