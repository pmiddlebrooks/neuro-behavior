function plotConfig = fill_manuscript_plot_config(plotConfig)
% FILL_MANUSCRIPT_PLOT_CONFIG - Default fonts and line widths for manuscript figures
%
% Variables:
%   plotConfig - Partial or empty struct; missing fields receive defaults
%
% Goal:
%   Shared axis/scatter/histogram styling for criticality manuscript and
%   task engagement figures (reach, interval, spontaneous).

if nargin < 1 || isempty(plotConfig)
  plotConfig = struct();
end
if ~isfield(plotConfig, 'axisLabelFontSize') || isempty(plotConfig.axisLabelFontSize)
  plotConfig.axisLabelFontSize = 14;
end
if ~isfield(plotConfig, 'tickLabelFontSize') || isempty(plotConfig.tickLabelFontSize)
  plotConfig.tickLabelFontSize = 12;
end
if ~isfield(plotConfig, 'titleFontSize') || isempty(plotConfig.titleFontSize)
  plotConfig.titleFontSize = 13;
end
if ~isfield(plotConfig, 'sgtitleFontSize') || isempty(plotConfig.sgtitleFontSize)
  plotConfig.sgtitleFontSize = 14;
end
if ~isfield(plotConfig, 'axesLineWidth') || isempty(plotConfig.axesLineWidth)
  plotConfig.axesLineWidth = 1.5;
end
if ~isfield(plotConfig, 'markerSize') || isempty(plotConfig.markerSize)
  plotConfig.markerSize = 6;
end
if ~isfield(plotConfig, 'scatterMarkerSize') || isempty(plotConfig.scatterMarkerSize)
  plotConfig.scatterMarkerSize = 48;
end
if ~isfield(plotConfig, 'scatterLineWidth') || isempty(plotConfig.scatterLineWidth)
  plotConfig.scatterLineWidth = 2;
end
if ~isfield(plotConfig, 'markerFaceAlpha') || isempty(plotConfig.markerFaceAlpha)
  plotConfig.markerFaceAlpha = 0.55;
end
if ~isfield(plotConfig, 'lineWidth') || isempty(plotConfig.lineWidth)
  plotConfig.lineWidth = 1.5;
end
if ~isfield(plotConfig, 'errorCapSize') || isempty(plotConfig.errorCapSize)
  plotConfig.errorCapSize = 8;
end
if ~isfield(plotConfig, 'legendFontSize') || isempty(plotConfig.legendFontSize)
  plotConfig.legendFontSize = 11;
end
if ~isfield(plotConfig, 'shuffleMarkerSize') || isempty(plotConfig.shuffleMarkerSize)
  plotConfig.shuffleMarkerSize = 4;
end
if ~isfield(plotConfig, 'histogramFaceAlpha') || isempty(plotConfig.histogramFaceAlpha)
  plotConfig.histogramFaceAlpha = 0.45;
end
if ~isfield(plotConfig, 'histogramShuffleFaceAlpha') || isempty(plotConfig.histogramShuffleFaceAlpha)
  plotConfig.histogramShuffleFaceAlpha = 0.4;
end
end
