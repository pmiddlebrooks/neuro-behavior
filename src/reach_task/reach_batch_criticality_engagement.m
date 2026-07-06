%%
% Reach Batch Criticality Metrics by Engagement
%
% Runs reach_criticality_metrics_engagement on every session in
% reach_session_list.m, collects per-session engaged vs non-engaged metrics,
% and tests differences across sessions.
%
% Variables (configure in this section):
%   sessions       - Cell of session names; default reach_session_list()
%   makePlots      - If true, create per-session figures (default false)
%   plotBatchSummary - If true, plot across-session engaged vs non-engaged summary
%   saveFigure     - Export batch summary figure
%   analyses       - Subset of {'d2','kurtosis','avalanches'}
%   nMinNeurons    - Minimum neurons in analysis area(s)
%   useSubsampling - If true, metrics = mean across neuron subsamples
%   nSubsamples, nNeuronsSubsample, minNeuronsMultiple - subsampling settings
%   brainArea, collectStart/End, reachBuffer, minNonEngagedWindow, ...
%
% Goal:
%   Compare criticality metrics between engaged and non-engaged conditions
%   across reach sessions with paired statistics (Wilcoxon signed-rank).

%% Configuration
sessions = reach_session_list();
makePlots = false;           % per-session figures from reach_criticality_metrics_engagement
plotBatchSummary = true;     % across-session engaged vs non-engaged summary
saveFigure = true;

analyses = {'d2', 'kurtosis', 'avalanches'};

collectStart = 0;
collectEnd = [];             % full session when empty

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

% Neuron count and optional subsampling (same pattern as session_d2_distributions)
nMinNeurons = 50;
useSubsampling = true;
nSubsamples = 10;
nNeuronsSubsample = 50;
minNeuronsMultiple = 1.25;

reachBuffer = 2;
minNonEngagedWindow = 25;
absorbSingleReaches = 1;
d2Window = 30;
prgWindow = 30;
useLog10D2 = true;

enableCircularPermutations = true;
nShuffles = 5;

% Paths (neuro-behavior src is assumed already on the MATLAB path)
paths = get_paths();

%% Session opts (shared)
opts = neuro_behavior_options();
opts.firingRateCheckTime = [];
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.minFiringRate = 0.1;
opts.maxFiringRate = 100;
opts.analyses = analyses;
opts.brainArea = brainArea;
opts.brainAreaCombinations = brainAreaCombinations;
opts.nMinNeurons = nMinNeurons;
opts.useSubsampling = useSubsampling;
opts.nSubsamples = nSubsamples;
opts.nNeuronsSubsample = nNeuronsSubsample;
opts.minNeuronsMultiple = minNeuronsMultiple;
opts.reachBuffer = reachBuffer;
opts.minNonEngagedWindow = minNonEngagedWindow;
opts.absorbSingleReaches = absorbSingleReaches;
opts.d2Window = d2Window;
opts.prgWindow = prgWindow;
opts.useLog10D2 = useLog10D2;
opts.enableCircularPermutations = enableCircularPermutations;
opts.nShuffles = nShuffles;
opts.makePlots = makePlots;
opts.saveFigure = false;

nSessions = numel(sessions);
fprintf('\n=== Reach batch criticality by engagement ===\n');
fprintf('Sessions: %d\n', nSessions);
fprintf('Analyses: %s\n', strjoin(analyses, ', '));
fprintf('nMinNeurons: %d\n', nMinNeurons);
if useSubsampling
  fprintf('Subsampling: %d subsets x %d neurons (min neurons x %.2f)\n', ...
    nSubsamples, nNeuronsSubsample, minNeuronsMultiple);
else
  fprintf('Subsampling: off\n');
end
fprintf('makePlots (per session): %d\n', makePlots);

%% Run each session
sessionResults = cell(nSessions, 1);
sessionOk = false(nSessions, 1);
sessionErrors = cell(nSessions, 1);

for iSess = 1:nSessions
  sessionName = sessions{iSess};
  fprintf('\n========== Session %d/%d: %s ==========\n', iSess, nSessions, sessionName);
  try
    sessionResults{iSess} = reach_criticality_metrics_engagement(sessionName, opts);
    sessionOk(iSess) = true;
  catch ME
    sessionOk(iSess) = false;
    sessionErrors{iSess} = ME;
    warning('reach_batch_criticality_engagement:SessionFailed', ...
      'Session %s failed: %s', sessionName, ME.message);
  end
end

fprintf('\nCompleted %d/%d sessions successfully.\n', sum(sessionOk), nSessions);
if any(~sessionOk)
  fprintf('Failed sessions:\n');
  failedSessIdx = find(~sessionOk);
  failedSessIdx = failedSessIdx(:);
  for k = 1:numel(failedSessIdx)
    iSess = failedSessIdx(k);
    fprintf('  %s — %s\n', sessions{iSess}, get_session_error_message(sessionErrors, iSess));
  end
end

%% Collect per-session engaged / non-engaged metrics
batchTable = build_batch_engagement_table(sessions, sessionResults, sessionOk, useLog10D2);
print_batch_engagement_table(batchTable);

% Across-session stats (engaged vs non-engaged)
statsTable = run_batch_engagement_stats(batchTable);
print_batch_engagement_stats(statsTable);

% Batch summary figure
figBatch = gobjects(0);
if plotBatchSummary
  figBatch = plot_batch_engagement_summary(batchTable, statsTable, useLog10D2);
  if saveFigure && isgraphics(figBatch)
    saveDir = fullfile(paths.dropPath, 'reach_task', 'results', 'batch_engagement');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    areaTag = matlab.lang.makeValidName(brainArea);
    if isempty(areaTag)
      areaTag = 'all_areas';
    end
    plotBase = fullfile(saveDir, sprintf('reach_batch_criticality_engagement_%s', areaTag));
    exportgraphics(figBatch, [plotBase, '.png'], 'Resolution', 300);
    exportgraphics(figBatch, [plotBase, '.eps'], 'ContentType', 'vector');
    fprintf('\nSaved batch figure: %s\n', plotBase);
  end
end

%% Package outputs in workspace
batchResults = struct();
batchResults.sessions = sessions;
batchResults.sessionOk = sessionOk;
batchResults.sessionErrors = sessionErrors;
batchResults.sessionResults = sessionResults;
batchResults.batchTable = batchTable;
batchResults.statsTable = statsTable;
batchResults.figBatch = figBatch;
batchResults.config = opts;

fprintf('\n=== Batch done ===\n');

%% -------------------------------------------------------------------------
%% Local functions
%% -------------------------------------------------------------------------

function msg = get_session_error_message(sessionErrors, iSess)
% GET_SESSION_ERROR_MESSAGE - Safe message extraction from sessionErrors cell
%
% Variables:
%   sessionErrors - Cell array; failed sessions may hold MException objects
%   iSess         - Session index
%
% Goal:
%   Avoid isempty() on MException (can error); return message or empty string.

msg = '';
if nargin < 2 || isempty(iSess) || ~isscalar(iSess) || ~isfinite(iSess)
  return;
end
iSess = round(double(iSess));
nErrors = numel(sessionErrors);
if iSess < 1 || iSess > nErrors
  return;
end
errEntry = sessionErrors{iSess};
if isa(errEntry, 'MException')
  msg = errEntry.message;
elseif ischar(errEntry) || isstring(errEntry)
  msg = char(string(errEntry));
end
end

function batchTable = build_batch_engagement_table(sessions, sessionResults, sessionOk, useLog10D2)
% BUILD_BATCH_ENGAGEMENT_TABLE - One row per successful session
%
% Goal:
%   Session-level means for engaged and non-engaged: d2, kappa, D_JS, tau, alpha.

metricNames = {'d2', 'decades', 'kappa', 'djs', 'tau', 'alpha', 'dcc'};
nSessions = numel(sessions);

batchTable = table();
batchTable.sessionName = sessions(:);
batchTable.ok = sessionOk(:);

for m = 1:numel(metricNames)
  name = metricNames{m};
  batchTable.(['engaged_' name]) = nan(nSessions, 1);
  batchTable.(['nonEngaged_' name]) = nan(nSessions, 1);
  batchTable.(['total_' name]) = nan(nSessions, 1);
end
batchTable.engagedDurSec = nan(nSessions, 1);
batchTable.nonEngagedDurSec = nan(nSessions, 1);
batchTable.nEngagedWindows = nan(nSessions, 1);
batchTable.nNonEngagedWindows = nan(nSessions, 1);

for iSess = 1:nSessions
  if ~sessionOk(iSess) || ~isstruct(sessionResults{iSess})
    continue;
  end
  out = sessionResults{iSess};

  if isfield(out, 'summary') && ~isempty(out.summary)
    for m = 1:numel(out.summary.metrics)
      metric = out.summary.metrics{m};
      name = metric.name;
      engCol = ['engaged_' name];
      nonCol = ['nonEngaged_' name];
      totCol = ['total_' name];
      if ~ismember(engCol, batchTable.Properties.VariableNames)
        continue;
      end
      % class order: Total=1, Engaged=2, Non-engaged=3
      if numel(metric.stats) >= 3
        batchTable.(totCol)(iSess) = metric.stats(1).mean;
        batchTable.(engCol)(iSess) = metric.stats(2).mean;
        batchTable.(nonCol)(iSess) = metric.stats(3).mean;
      end
    end
  end

  if isfield(out, 'avalanches') && ~isempty(out.avalanches) ...
      && isfield(out.avalanches, 'byClass')
    for avMetric = {'tau', 'alpha', 'decades', 'dcc'}
      name = avMetric{1};
      batchTable.(['total_' name])(iSess) = first_area_avalanche_metric( ...
        out.avalanches.byClass.total, name);
      batchTable.(['engaged_' name])(iSess) = first_area_avalanche_metric( ...
        out.avalanches.byClass.engaged, name);
      batchTable.(['nonEngaged_' name])(iSess) = first_area_avalanche_metric( ...
        out.avalanches.byClass.nonEngaged, name);
    end
  end

  if isfield(out, 'durations')
    if isfield(out.durations, 'd2') && ~isempty(out.durations.d2)
      batchTable.engagedDurSec(iSess) = out.durations.d2.engagedSec;
      batchTable.nonEngagedDurSec(iSess) = out.durations.d2.nonEngagedSec;
      batchTable.nEngagedWindows(iSess) = out.durations.d2.nEngagedWindows;
      batchTable.nNonEngagedWindows(iSess) = out.durations.d2.nNonEngagedWindows;
    elseif isfield(out.durations, 'kurtosis') && ~isempty(out.durations.kurtosis)
      batchTable.engagedDurSec(iSess) = out.durations.kurtosis.engagedSec;
      batchTable.nonEngagedDurSec(iSess) = out.durations.kurtosis.nonEngagedSec;
      batchTable.nEngagedWindows(iSess) = out.durations.kurtosis.nEngagedWindows;
      batchTable.nNonEngagedWindows(iSess) = out.durations.kurtosis.nNonEngagedWindows;
    elseif isfield(out.durations, 'avalanches') && ~isempty(out.durations.avalanches)
      batchTable.engagedDurSec(iSess) = out.durations.avalanches.engagedSec;
      batchTable.nonEngagedDurSec(iSess) = out.durations.avalanches.nonEngagedSec;
    end
  end
end

batchTable.Properties.UserData.useLog10D2 = useLog10D2;
end

function value = first_area_avalanche_metric(avClassResult, metricName)
% FIRST_AREA_AVALANCHE_METRIC - Exponent from first area with avalanches

value = nan;
if isempty(avClassResult) || ~isfield(avClassResult, 'byArea')
  return;
end
for a = 1:numel(avClassResult.byArea)
  avData = avClassResult.byArea{a};
  if isstruct(avData) && isfield(avData, 'hasAvalanches') && avData.hasAvalanches
    if isfield(avData, metricName) && isfinite(avData.(metricName))
      value = avData.(metricName);
      return;
    end
    if strcmp(metricName, 'decades') && isfield(avData, 'sizeFitInfo') ...
        && isstruct(avData.sizeFitInfo) && isfield(avData.sizeFitInfo, 'decades') ...
        && isfinite(avData.sizeFitInfo.decades)
      value = avData.sizeFitInfo.decades;
      return;
    end
  end
end
end

function print_batch_engagement_table(batchTable)
% PRINT_BATCH_ENGAGEMENT_TABLE - Session-level engaged / non-engaged means

fprintf('\n=== Per-session engaged vs non-engaged means ===\n');
okRows = batchTable.ok;
if ~any(okRows)
  fprintf('  No successful sessions.\n');
  return;
end

metricPairs = {
  'd2', 'engaged_d2', 'nonEngaged_d2'
  'decades', 'engaged_decades', 'nonEngaged_decades'
  'kappa', 'engaged_kappa', 'nonEngaged_kappa'
  'djs', 'engaged_djs', 'nonEngaged_djs'
  'tau', 'engaged_tau', 'nonEngaged_tau'
  'alpha', 'engaged_alpha', 'nonEngaged_alpha'
  'dcc', 'engaged_dcc', 'nonEngaged_dcc'
  };

for iSess = find(okRows)'
  fprintf('  %s\n', batchTable.sessionName{iSess});
  for m = 1:size(metricPairs, 1)
    eng = batchTable.(metricPairs{m, 2})(iSess);
    non = batchTable.(metricPairs{m, 3})(iSess);
    if ~isfinite(eng) && ~isfinite(non)
      continue;
    end
    fprintf('    %-6s  engaged=%8.4f   non-engaged=%8.4f\n', ...
      metricPairs{m, 1}, eng, non);
  end
end
end

function statsTable = run_batch_engagement_stats(batchTable)
% RUN_BATCH_ENGAGEMENT_STATS - Paired engaged vs non-engaged across sessions
%
% Goal:
%   For each metric, use sessions with finite values in both conditions.
%   Wilcoxon signed-rank (signrank) and paired t-test when n >= 2.

metricDefs = {
  'd2', 'engaged_d2', 'nonEngaged_d2', 'd2'
  'decades', 'engaged_decades', 'nonEngaged_decades', 'decades'
  'kappa', 'engaged_kappa', 'nonEngaged_kappa', '\kappa'
  'djs', 'engaged_djs', 'nonEngaged_djs', 'D_{JS}'
  'tau', 'engaged_tau', 'nonEngaged_tau', '\tau'
  'alpha', 'engaged_alpha', 'nonEngaged_alpha', '\alpha'
  'dcc', 'engaged_dcc', 'nonEngaged_dcc', 'dcc'
  };

nMetrics = size(metricDefs, 1);
metricName = metricDefs(:, 1);
metricLabel = metricDefs(:, 4);
n = nan(nMetrics, 1);
engagedMean = nan(nMetrics, 1);
engagedSem = nan(nMetrics, 1);
nonEngagedMean = nan(nMetrics, 1);
nonEngagedSem = nan(nMetrics, 1);
meanDiff = nan(nMetrics, 1);
semDiff = nan(nMetrics, 1);
pSignrank = nan(nMetrics, 1);
pTtest = nan(nMetrics, 1);

okRows = batchTable.ok;
for m = 1:nMetrics
  eng = batchTable.(metricDefs{m, 2})(okRows);
  non = batchTable.(metricDefs{m, 3})(okRows);
  valid = isfinite(eng) & isfinite(non);
  eng = eng(valid);
  non = non(valid);
  n(m) = numel(eng);
  if n(m) == 0
    continue;
  end
  engagedMean(m) = mean(eng);
  nonEngagedMean(m) = mean(non);
  if n(m) > 1
    engagedSem(m) = std(eng) / sqrt(n(m));
    nonEngagedSem(m) = std(non) / sqrt(n(m));
  else
    engagedSem(m) = 0;
    nonEngagedSem(m) = 0;
  end
  diffs = eng - non;
  meanDiff(m) = mean(diffs);
  if n(m) > 1
    semDiff(m) = std(diffs) / sqrt(n(m));
  else
    semDiff(m) = 0;
  end

  if n(m) >= 2
    try
      pSignrank(m) = signrank(eng, non);
    catch
      pSignrank(m) = nan;
    end
    try
      [~, pTtest(m)] = ttest(eng, non);
    catch
      pTtest(m) = nan;
    end
  end
end

statsTable = table(metricName, metricLabel, n, engagedMean, engagedSem, ...
  nonEngagedMean, nonEngagedSem, meanDiff, semDiff, pSignrank, pTtest);
end

function print_batch_engagement_stats(statsTable)
% PRINT_BATCH_ENGAGEMENT_STATS - Command-window paired stats summary

fprintf('\n=== Across-session stats: engaged vs non-engaged ===\n');
fprintf('%-8s  n=%-3s  engaged mean+/-SEM   non-eng mean+/-SEM   diff+/-SEM      p_signrank  p_ttest\n', ...
  'metric', '');
for m = 1:height(statsTable)
  row = statsTable(m, :);
  if row.n == 0 || ~isfinite(row.engagedMean)
    fprintf('%-8s  no paired data\n', row.metricName{1});
    continue;
  end
  fprintf('%-8s  n=%-3d  %7.4f +/- %-7.4f  %7.4f +/- %-7.4f  %7.4f +/- %-6.4f  %-10s  %-10s\n', ...
    row.metricName{1}, row.n, ...
    row.engagedMean, row.engagedSem, ...
    row.nonEngagedMean, row.nonEngagedSem, ...
    row.meanDiff, row.semDiff, ...
    format_p_value(row.pSignrank), format_p_value(row.pTtest));
end
fprintf('diff = engaged - non-engaged; signrank = Wilcoxon signed-rank (paired)\n');
end

function pStr = format_p_value(pVal)
if ~isfinite(pVal)
  pStr = 'n/a';
elseif pVal < 0.001
  pStr = sprintf('%.2e', pVal);
else
  pStr = sprintf('%.4f', pVal);
end
end

function fig = plot_batch_engagement_summary(batchTable, statsTable, useLog10D2)
% PLOT_BATCH_ENGAGEMENT_SUMMARY - Across-session engaged vs non-engaged bars
%
% Goal:
%   One axes per metric with mean +/- SEM across sessions and paired session lines.

metricDefs = {
  'd2', 'engaged_d2', 'nonEngaged_d2'
  'decades', 'engaged_decades', 'nonEngaged_decades'
  'kappa', 'engaged_kappa', 'nonEngaged_kappa'
  'djs', 'engaged_djs', 'nonEngaged_djs'
  'tau', 'engaged_tau', 'nonEngaged_tau'
  'alpha', 'engaged_alpha', 'nonEngaged_alpha'
  'dcc', 'engaged_dcc', 'nonEngaged_dcc'
  };
metricLabels = {'d2', 'decades', '\kappa Kurtosis', 'D_{JS}', 'Av \tau', 'Av \alpha', 'Av DCC'};
if useLog10D2
  metricLabels{1} = 'log_{10}(d2)';
end

% Keep metrics that have at least one paired session
keep = false(size(metricDefs, 1), 1);
for m = 1:size(metricDefs, 1)
  row = statsTable(strcmp(statsTable.metricName, metricDefs{m, 1}), :);
  keep(m) = ~isempty(row) && row.n > 0;
end
metricDefs = metricDefs(keep, :);
metricLabels = metricLabels(keep);
nMetrics = size(metricDefs, 1);
if nMetrics == 0
  warning('reach_batch_criticality_engagement:NoMetrics', ...
    'No metrics available for batch summary plot.');
  fig = gobjects(0);
  return;
end

engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];
okRows = batchTable.ok;

fig = figure('Color', 'w', 'Name', 'Reach batch engagement summary', ...
  'Position', [100 100 280 * nMetrics 420]);
tileLayout = tiledlayout(fig, 1, nMetrics, 'TileSpacing', 'compact', 'Padding', 'compact');

for m = 1:nMetrics
  ax = nexttile(tileLayout);
  hold(ax, 'on');

  eng = batchTable.(metricDefs{m, 2})(okRows);
  non = batchTable.(metricDefs{m, 3})(okRows);
  valid = isfinite(eng) & isfinite(non);
  eng = eng(valid);
  non = non(valid);
  n = numel(eng);

  % Paired session lines
  for i = 1:n
    plot(ax, [1, 2], [eng(i), non(i)], '-', 'Color', [0.7, 0.7, 0.7], ...
      'LineWidth', 1, 'HandleVisibility', 'off');
  end
  scatter(ax, ones(n, 1), eng, 36, engagedColor, 'filled', ...
    'MarkerFaceAlpha', 0.7, 'HandleVisibility', 'off');
  scatter(ax, 2 * ones(n, 1), non, 36, nonEngagedColor, 'filled', ...
    'MarkerFaceAlpha', 0.7, 'HandleVisibility', 'off');

  engMean = mean(eng);
  nonMean = mean(non);
  if n > 1
    engSem = std(eng) / sqrt(n);
    nonSem = std(non) / sqrt(n);
  else
    engSem = 0;
    nonSem = 0;
  end
  errorbar(ax, 1, engMean, engSem, 'o', 'Color', engagedColor, ...
    'MarkerFaceColor', engagedColor, 'LineWidth', 1.5, 'CapSize', 10, ...
    'MarkerSize', 8, 'DisplayName', 'Engaged');
  errorbar(ax, 2, nonMean, nonSem, 'o', 'Color', nonEngagedColor, ...
    'MarkerFaceColor', nonEngagedColor, 'LineWidth', 1.5, 'CapSize', 10, ...
    'MarkerSize', 8, 'DisplayName', 'Non-engaged');

  row = statsTable(strcmp(statsTable.metricName, metricDefs{m, 1}), :);
  pText = 'n/a';
  if ~isempty(row) && isfinite(row.pSignrank)
    if row.pSignrank < 0.001
      pText = sprintf('p=%.1e', row.pSignrank);
    else
      pText = sprintf('p=%.3f', row.pSignrank);
    end
  end

  set(ax, 'XTick', [1, 2], 'XTickLabel', {'Engaged', 'Non-eng'}, ...
    'FontSize', 12, 'LineWidth', 1.5, 'Box', 'off', 'TickDir', 'out');
  xlim(ax, [0.5, 2.5]);
  xtickangle(ax, 20);
  ylabel(ax, metricLabels{m}, 'FontSize', 14, 'Interpreter', 'tex');
  title(ax, sprintf('%s (n=%d, %s)', metricLabels{m}, n, pText), ...
    'FontSize', 12, 'Interpreter', 'tex');
  % grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayout, 'Reach sessions: engaged vs non-engaged (mean \pm SEM; paired lines)', ...
  'FontSize', 14, 'Interpreter', 'tex');
end
