function fig = plot_session_ei_summary(summary, plotTitle, yLabelText, metricNamesToPlot, parentFig, plotConfig)
% PLOT_SESSION_EI_SUMMARY - Bar summary of mean +/- SEM for combined, E, and I
%
% Variables:
%   summary           - From init_session_ei_summary / set_session_ei_summary_population
%   plotTitle         - Figure title
%   yLabelText        - Y-axis label (e.g. 'log_{10}(d2)')
%   metricNamesToPlot - Optional subset of summary.metricNames (default: all)
%   parentFig         - Optional existing figure to replot into
%   plotConfig        - From fill_manuscript_plot_config
%
% Goal:
%   Combined = black, excitatory = blue, inhibitory = red.

if nargin < 4
  metricNamesToPlot = [];
end
if nargin < 5
  parentFig = [];
end
if nargin < 6 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

if isempty(metricNamesToPlot)
  metricIdx = 1:numel(summary.metricNames);
else
  metricNamesToPlot = metricNamesToPlot(:)';
  metricIdx = zeros(1, numel(metricNamesToPlot));
  for m = 1:numel(metricNamesToPlot)
    idx = find(strcmp(summary.metricNames, metricNamesToPlot{m}), 1);
    if isempty(idx)
      error('plot_session_ei_summary:UnknownMetric', ...
        'Unknown metric "%s" in E/I summary.', metricNamesToPlot{m});
    end
    metricIdx(m) = idx;
  end
end

populations = summary.populations;
popLabels = {'Combined', 'Excitatory', 'Inhibitory'};
popColors = [0 0 0; 0 0.4470 0.7410; 0.8500 0.1000 0.1000];

metricNames = summary.metricNames(metricIdx);
metricLabels = summary.metricLabels(metricIdx);
nMetrics = numel(metricNames);

if nargin >= 5 && ~isempty(parentFig) && isgraphics(parentFig)
  fig = parentFig;
  clf(fig);
else
  fig = figure('Name', 'Session E/I summary', 'Color', 'w', ...
    'Position', [140 140 max(420, 220 * nMetrics) 420]);
end
ax = axes(fig); %#ok<LAXES>
hold(ax, 'on');

groupWidth = min(0.8, 0.25 * nMetrics);
barWidth = groupWidth / numel(populations);
xCenters = 1:nMetrics;
legendHandles = gobjects(numel(populations), 1);

for m = 1:nMetrics
  metricName = metricNames{m};
  for p = 1:numel(populations)
    popName = populations{p};
    stats = summary.(popName).(metricName);
    xPos = xCenters(m) + (p - (numel(populations) + 1) / 2) * barWidth;
    if ~isfinite(stats.mean)
      continue;
    end
    hBar = bar(ax, xPos, stats.mean, barWidth, ...
      'FaceColor', popColors(p, :), 'EdgeColor', 'none', 'FaceAlpha', 0.9);
    if ~isgraphics(legendHandles(p))
      legendHandles(p) = hBar;
    end
    if isfinite(stats.sem) && stats.sem > 0
      errorbar(ax, xPos, stats.mean, stats.sem, 'Color', popColors(p, :), ...
        'LineStyle', 'none', 'LineWidth', plotConfig.lineWidth, ...
        'CapSize', plotConfig.errorCapSize, 'HandleVisibility', 'off');
    end
  end
end

set(ax, 'XTick', xCenters, 'XTickLabel', metricLabels);
apply_manuscript_axes_style(ax, plotConfig, '', yLabelText, '', 'tex');
title(ax, plotTitle, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
legendMask = isgraphics(legendHandles);
legend(ax, legendHandles(legendMask), popLabels(legendMask), ...
  'Location', 'best', 'FontSize', plotConfig.legendFontSize);
grid(ax, 'on');
hold(ax, 'off');
end
