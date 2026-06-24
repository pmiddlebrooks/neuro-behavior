function summary = set_session_ei_summary_population(summary, cellType, metricValues)
% SET_SESSION_EI_SUMMARY_POPULATION - Store mean/SEM for one population
%
% Variables:
%   metricValues - Struct with one field per metric name; each value is a numeric vector

popName = normalize_ei_population_name(cellType);
if ~isfield(summary, popName)
  return;
end

for m = 1:numel(summary.metricNames)
  metricName = summary.metricNames{m};
  if ~isfield(metricValues, metricName)
    continue;
  end
  summary.(popName).(metricName) = mean_sem_values(metricValues.(metricName));
end
end

function popName = normalize_ei_population_name(cellType)
if isempty(cellType) || strcmpi(cellType, 'all')
  popName = 'all';
else
  popName = lower(strtrim(cellType));
end
end

function stats = mean_sem_values(values)
values = values(:);
values = values(isfinite(values));
stats = struct('mean', nan, 'sem', nan, 'n', 0);
if isempty(values)
  return;
end
stats.mean = mean(values);
stats.n = numel(values);
if stats.n > 1
  stats.sem = std(values, 0) / sqrt(stats.n);
else
  stats.sem = 0;
end
end
