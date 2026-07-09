function h = scatter_manuscript_open(ax, x, y, plotConfig, edgeColor, displayName)
% SCATTER_MANUSCRIPT_OPEN - Open-circle scatter points with colored edge
%
% Variables:
%   ax           - Axes handle
%   x, y         - Data vectors
%   plotConfig   - From fill_manuscript_plot_config
%   edgeColor    - RGB row (default: manuscript data blue)
%   displayName  - Legend label (optional)
%
% Goal:
%   Consistent open-marker scatter styling for d2 correlation figures.

if nargin < 4 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 5 || isempty(edgeColor)
  edgeColor = manuscript_plot_colors().data;
end
if nargin < 6
  displayName = '';
end

scatterArgs = {...
  ax, x, y, plotConfig.scatterMarkerSize, ...
  'Marker', 'o', ...
  'MarkerFaceColor', 'none', ...
  'MarkerEdgeColor', edgeColor, ...
  'LineWidth', plotConfig.scatterLineWidth};
if ~isempty(displayName)
  scatterArgs = [scatterArgs, {'DisplayName', displayName}]; %#ok<AGROW>
end
h = scatter(scatterArgs{:});
end
