function apply_manuscript_axes_style(ax, plotConfig, xLabelText, yLabelText, titleText, textInterpreter)
% APPLY_MANUSCRIPT_AXES_STYLE - Thicker axes and larger fonts (manuscript style)
%
% Variables:
%   ax              - Axes handle
%   plotConfig      - From fill_manuscript_plot_config
%   xLabelText      - X label ('' or omit to skip)
%   yLabelText      - Y label ('' or omit to skip)
%   titleText       - Title ('' or omit to skip)
%   textInterpreter - Optional interpreter for labels/title (default 'none')
%
% Goal:
%   Consistent axis formatting across criticality manuscript and task figures.

if nargin < 2 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 6 || isempty(textInterpreter)
  textInterpreter = 'none';
end

set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
  'Box', 'off', 'TickDir', 'out');
if nargin >= 3 && ~isempty(xLabelText)
  xlabel(ax, xLabelText, 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', textInterpreter);
end
if nargin >= 4 && ~isempty(yLabelText)
  ylabel(ax, yLabelText, 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', textInterpreter);
end
if nargin >= 5 && ~isempty(titleText)
  title(ax, titleText, 'FontSize', plotConfig.titleFontSize, 'Interpreter', textInterpreter);
end
end
