function summary = init_session_ei_summary(metricNames, metricLabels)
% INIT_SESSION_EI_SUMMARY - Empty summary struct for E/I comparison plots

if nargin < 2 || isempty(metricLabels)
  metricLabels = metricNames;
end

summary = struct();
summary.metricNames = metricNames(:)';
summary.metricLabels = metricLabels(:)';
summary.populations = {'all', 'excitatory', 'inhibitory'};
for p = 1:numel(summary.populations)
  popName = summary.populations{p};
  summary.(popName) = struct();
  for m = 1:numel(summary.metricNames)
    metricName = summary.metricNames{m};
    summary.(popName).(metricName) = struct('mean', nan, 'sem', nan, 'n', 0);
  end
end
end
