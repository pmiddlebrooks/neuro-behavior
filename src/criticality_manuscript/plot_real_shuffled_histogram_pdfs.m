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
%   When shuffled values are present, ylim is set from the observed PDF
%   peak so the tighter shuffled distribution does not squash the data.

if nargin < 7 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end
if nargin < 8 || isempty(drawZeroRef)
  drawZeroRef = false;
end

plotColors = manuscript_plot_colors();
hold(ax, 'on');
hasShuffle = ~isempty(shuffledVals) && any(isfinite(shuffledVals(:)));

histogram(ax, realVals, binEdges, 'Normalization', 'pdf', ...
  'FaceColor', plotColors.data, 'FaceAlpha', plotConfig.histogramFaceAlpha, ...
  'EdgeColor', 'none', 'DisplayName', sprintf('Data (n=%d)', numel(realVals)));

if hasShuffle
  histogram(ax, shuffledVals, binEdges, 'Normalization', 'pdf', ...
    'FaceColor', plotColors.shuffled, 'FaceAlpha', plotConfig.histogramShuffleFaceAlpha, ...
    'EdgeColor', 'none', 'DisplayName', sprintf('Shuffled mean (n=%d)', numel(shuffledVals)));
end

if drawZeroRef
  xline(ax, 0, '--', 'Color', plotColors.refLine, 'LineWidth', plotConfig.lineWidth, ...
    'HandleVisibility', 'off');
end

xlim(ax, [xMin, xMax]);
% Shuffled d2 is typically peaked; scale y to observed density so data stays visible
if hasShuffle
  yMaxObs = histogram_pdf_peak(realVals, binEdges);
  if yMaxObs > 0
    ylim(ax, [0, yMaxObs * 1.12]);
  end
end
ax.Clipping = 'on';
grid(ax, 'on');
legend(ax, 'Location', 'northeast', 'FontSize', plotConfig.legendFontSize);
hold(ax, 'off');
end

function yMax = histogram_pdf_peak(vals, binEdges)
% HISTOGRAM_PDF_PEAK - Max PDF bar height for vals on binEdges
%
% Variables:
%   vals     - Sample values
%   binEdges - Histogram edges (same as plotted bars)
%
% Goal:
%   Match MATLAB histogram Normalization='pdf' bar heights.

vals = vals(isfinite(vals));
if isempty(vals) || numel(binEdges) < 2
  yMax = 0;
  return;
end
counts = histcounts(vals(:), binEdges);
binWidths = diff(binEdges(:)');
nObs = sum(counts);
if nObs < 1
  yMax = 0;
  return;
end
pdfHeights = counts ./ (nObs .* binWidths);
yMax = max(pdfHeights);
if isempty(yMax) || ~isfinite(yMax)
  yMax = 0;
end
end
