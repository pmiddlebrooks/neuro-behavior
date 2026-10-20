function h = scatter_manuscript_open(ax, x, y, plotConfig, edgeColor, displayName)
% SCATTER_MANUSCRIPT_OPEN - Open circles in the same color as filled points
%
% Variables:
%   ax          - Axes handle
%   x, y        - Data vectors
%   plotConfig  - From fill_manuscript_plot_config
%   edgeColor   - RGB row for the circle edge
%   displayName - Legend label (optional)
%
% Goal:
%   Draw engaged sessions as open circles that match the non-engaged color.

if nargin < 4 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 5 || isempty(edgeColor)
  edgeColor = manuscript_plot_colors().data;
end
if nargin < 6
  displayName = '';
end

lineWidth = 1.25;
if isfield(plotConfig, 'lineWidth') && ~isempty(plotConfig.lineWidth)
  lineWidth = max(plotConfig.lineWidth, 1.25);
end

scatterArgs = { ...
  ax, x, y, plotConfig.scatterMarkerSize, edgeColor, ...
  'Marker', 'o', ...
  'MarkerFaceColor', 'none', ...
  'MarkerEdgeColor', edgeColor, ...
  'LineWidth', lineWidth};
if ~isempty(displayName)
  scatterArgs = [scatterArgs, {'DisplayName', displayName}]; %#ok<AGROW>
end
h = scatter(scatterArgs{:});
end
