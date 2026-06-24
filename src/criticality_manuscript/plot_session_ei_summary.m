function fig = plot_session_ei_summary(summary, plotTitle, yLabelText)
% PLOT_SESSION_EI_SUMMARY - Bar summary of mean +/- SEM for combined, E, and I
%
% Variables:
%   summary     - From init_session_ei_summary / set_session_ei_summary_population
%   plotTitle   - Figure title
%   yLabelText  - Y-axis label (e.g. 'log_{10}(d2)')
%
% Goal:
%   Combined = black, excitatory = blue, inhibitory = red.

populations = summary.populations;
popLabels = {'Combined', 'Excitatory', 'Inhibitory'};
popColors = [0 0 0; 0 0.4470 0.7410; 0.8500 0.1000 0.1000];

nMetrics = numel(summary.metricNames);
fig = figure('Name', 'Session E/I summary', 'Color', 'w');
ax = axes(fig); %#ok<LAXES>
hold(ax, 'on');

groupWidth = min(0.8, 0.25 * nMetrics);
barWidth = groupWidth / numel(populations);
xCenters = 1:nMetrics;

for m = 1:nMetrics
  metricName = summary.metricNames{m};
  for p = 1:numel(populations)
    popName = populations{p};
    stats = summary.(popName).(metricName);
    xPos = xCenters(m) + (p - (numel(populations) + 1) / 2) * barWidth;
    if ~isfinite(stats.mean)
      continue;
    end
    bar(ax, xPos, stats.mean, barWidth, ...
      'FaceColor', popColors(p, :), 'EdgeColor', 'none', 'FaceAlpha', 0.9);
    if isfinite(stats.sem) && stats.sem > 0
      errorbar(ax, xPos, stats.mean, stats.sem, 'Color', popColors(p, :), ...
        'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 8);
    end
  end
end

set(ax, 'XTick', xCenters, 'XTickLabel', summary.metricLabels, 'Box', 'off');
ylabel(ax, yLabelText);
title(ax, plotTitle, 'Interpreter', 'none');
legend(ax, popLabels, 'Location', 'best');
grid(ax, 'on');
hold(ax, 'off');
end
