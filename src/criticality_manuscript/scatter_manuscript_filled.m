function h = scatter_manuscript_filled(ax, x, y, plotConfig, faceColor, displayName)
% SCATTER_MANUSCRIPT_FILLED - Translucent filled scatter points
%
% Variables:
%   ax           - Axes handle
%   x, y         - Data vectors
%   plotConfig   - From fill_manuscript_plot_config
%   faceColor    - RGB row (default: manuscript data blue)
%   displayName  - Legend label (optional)
%
% Goal:
%   Consistent scatter styling for d2 correlation figures.

if nargin < 4 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 5 || isempty(faceColor)
  faceColor = manuscript_plot_colors().data;
end
if nargin < 6
  displayName = '';
end

scatterArgs = {...
  ax, x, y, plotConfig.scatterMarkerSize, faceColor, 'filled', ...
  'MarkerFaceAlpha', plotConfig.markerFaceAlpha, ...
  'MarkerEdgeColor', 'none'};
if ~isempty(displayName)
  scatterArgs = [scatterArgs, {'DisplayName', displayName}]; %#ok<AGROW>
end
h = scatter(scatterArgs{:});
end
