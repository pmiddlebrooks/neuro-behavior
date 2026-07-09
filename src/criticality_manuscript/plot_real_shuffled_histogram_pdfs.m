function plot_real_shuffled_histogram_pdfs(ax, realVals, shuffledVals, binEdges, xMin, xMax, plotConfig, drawZeroRef)
% PLOT_REAL_SHUFFLED_HISTOGRAM_PDFS - Overlapping PDFs for data vs shuffled d2
%
% Variables:
%   ax            - Axes handle
%   realVals      - Window-wise data values
%   shuffledVals  - Per-window shuffled mean values (may be empty)
%   binEdges      - Histogram edges
%   xMin, xMax    - X limits
%   plotConfig    - From fill_manuscript_plot_config
%   drawZeroRef   - If true, draw dashed vertical line at x = 0
%
% Goal:
%   Manuscript-style overlapping histogram PDFs (real vs shuffled).

if nargin < 7 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 8 || isempty(drawZeroRef)
  drawZeroRef = false;
end

plotColors = manuscript_plot_colors();
hold(ax, 'on');

histogram(ax, realVals, binEdges, 'Normalization', 'pdf', ...
  'FaceColor', plotColors.data, 'FaceAlpha', plotConfig.histogramFaceAlpha, ...
  'EdgeColor', 'none', 'DisplayName', sprintf('Data (n=%d)', numel(realVals)));

if ~isempty(shuffledVals)
  histogram(ax, shuffledVals, binEdges, 'Normalization', 'pdf', ...
    'FaceColor', plotColors.shuffled, 'FaceAlpha', plotConfig.histogramShuffleFaceAlpha, ...
    'EdgeColor', 'none', 'DisplayName', sprintf('Shuffled mean (n=%d)', numel(shuffledVals)));
end

if drawZeroRef
  xline(ax, 0, '--', 'Color', plotColors.refLine, 'LineWidth', plotConfig.lineWidth, ...
    'HandleVisibility', 'off');
end

xlim(ax, [xMin, xMax]);
grid(ax, 'on');
legend(ax, 'Location', 'northeast', 'FontSize', plotConfig.legendFontSize);
hold(ax, 'off');
end
