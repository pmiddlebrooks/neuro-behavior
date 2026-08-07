%%
% Criticality Per Subsample Size Across Tasks (Manuscript)
%
% Sweeps nNeuronsSubsample for d2 / avalanche / PRG across session types,
% then plots summary curves (and optional trend summaries) without per-session
% figures unless plotIndividualSessions is true.
%
% Variables (edit Configuration below):
%   sessionTypes, analyses, nNeuronsSubsampleList
%   plotResults, plotIndividualSessions, saveBatchResults
%   collectStart/End, d2Window, prgWindow, brainArea, ...
%
% Summary strategy:
%   1) Primary: mean ± SEM of each metric vs nNeuronsSubsample across sessions
%      (faint per-session curves colored by task).
%   2) Secondary: per-session OLS slope and relative change first→last N,
%      summarized by task.

%% Configuration
sessionTypes = default_manuscript_session_types();
sessionTypes = order_manuscript_session_types(sessionTypes);
analyses = {'d2', 'av', 'prg'};  % any subset of {'d2','av','prg'}
nNeuronsSubsampleList = 20:5:60;

dataSource = 'spikes';
collectStart = 0;
collectEnd = 60 * 60;
% collectEnd = [];  % [] = full session
d2Window = 30;   % [] = one window over the collect duration
prgWindow = 30;  % [] = one block over the collect duration

setup_criticality_manuscript_paths('criticality_per_subsample_size_batch');
paths = get_paths();

brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

plotResults = true;
plotIndividualSessions = false;
saveBatchResults = true;
batchResultsFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
  'criticality_per_subsample_size_batch.mat');

useLog10D2 = true;
nSubsamples = 25;
minNeuronsMultiple = 1.1;
nMinNeurons = 20;
enablePermutations = false;
nShuffles = 0;

avalancheDetectionMode = 'fixedBinMedian';
powerLawFitMethod = 'plfit2023';
gofThreshold = 0.1;

prgMethod = 'pca';
binSize = 0.05;
cvThreshold = 5;
cutoffDivisors = [1, 2, 4, 8, 16, 32];
finalCutoffDivisor = 16;
kappaAxisMax = 20;
enableSurrogates = false;
nSurrogates = 0;
surrogateMethod = 'isi';

firingRateCheckTime = [];
minFiringRate = 0.05;
maxFiringRate = 100;

% Pack opts for helpers / session calls
opts = struct( ...
  'sessionTypes', {sessionTypes}, ...
  'analyses', {analyses}, ...
  'nNeuronsSubsampleList', nNeuronsSubsampleList, ...
  'dataSource', dataSource, ...
  'collectStart', collectStart, ...
  'collectEnd', collectEnd, ...
  'd2Window', d2Window, ...
  'prgWindow', prgWindow, ...
  'brainArea', brainArea, ...
  'brainAreaCombinations', {brainAreaCombinations}, ...
  'plotResults', plotResults, ...
  'plotIndividualSessions', plotIndividualSessions, ...
  'saveBatchResults', saveBatchResults, ...
  'batchResultsFile', batchResultsFile, ...
  'useLog10D2', useLog10D2, ...
  'nSubsamples', nSubsamples, ...
  'minNeuronsMultiple', minNeuronsMultiple, ...
  'nMinNeurons', nMinNeurons, ...
  'enablePermutations', enablePermutations, ...
  'nShuffles', nShuffles, ...
  'avalancheDetectionMode', avalancheDetectionMode, ...
  'powerLawFitMethod', powerLawFitMethod, ...
  'gofThreshold', gofThreshold, ...
  'prgMethod', prgMethod, ...
  'binSize', binSize, ...
  'cvThreshold', cvThreshold, ...
  'cutoffDivisors', cutoffDivisors, ...
  'finalCutoffDivisor', finalCutoffDivisor, ...
  'kappaAxisMax', kappaAxisMax, ...
  'enableSurrogates', enableSurrogates, ...
  'nSurrogates', nSurrogates, ...
  'surrogateMethod', surrogateMethod, ...
  'firingRateCheckTime', firingRateCheckTime, ...
  'minFiringRate', minFiringRate, ...
  'maxFiringRate', maxFiringRate);

%% Run batch
fprintf('\n=== Criticality per subsample size (batch) ===\n');
fprintf('Session types: %s\n', strjoin(sessionTypes, ', '));
fprintf('Analyses: %s\n', strjoin(analyses, ', '));
fprintf('nNeuronsSubsampleList: [%s]\n', strjoin(string(nNeuronsSubsampleList), ' '));
fprintf('plotIndividualSessions: %d | plotResults: %d\n', ...
  plotIndividualSessions, plotResults);

sessionTable = build_session_table(sessionTypes);
numSessions = size(sessionTable, 1);
fprintf('Total sessions: %d\n', numSessions);
if numSessions == 0
  error('No sessions found for the requested session types.');
end

sessionOpts = build_session_opts_from_batch(opts);

batchResults = repmat(struct( ...
  'sessionType', '', ...
  'sessionName', '', ...
  'subjectName', '', ...
  'label', '', ...
  'success', false, ...
  'skipReason', '', ...
  'nUnits', nan, ...
  'sweep', [], ...
  'trend', []), numSessions, 1);

for iSess = 1:numSessions
  sessionType = sessionTable.sessionType{iSess};
  sessionName = sessionTable.sessionName{iSess};
  subjectName = sessionTable.subjectName{iSess};
  label = sessionTable.label{iSess};

  fprintf('\n######## Session %d/%d: %s / %s ########\n', ...
    iSess, numSessions, sessionType, sessionName);

  batchResults(iSess).sessionType = sessionType;
  batchResults(iSess).sessionName = sessionName;
  batchResults(iSess).subjectName = subjectName;
  batchResults(iSess).label = label;

  sessOpts = sessionOpts;
  sessOpts.plotResults = plotIndividualSessions;

  try
    sessOut = criticality_per_subsample_size(sessionType, sessionName, subjectName, sessOpts);
    batchResults(iSess).success = true;
    batchResults(iSess).nUnits = sessOut.nUnits;
    batchResults(iSess).sweep = sessOut.sweep;
    batchResults(iSess).trend = sessOut.trend;
  catch ME
    batchResults(iSess).success = false;
    batchResults(iSess).skipReason = ME.message;
    warning('Session failed (%s / %s): %s', sessionType, sessionName, ME.message);
  end
end

nOk = sum([batchResults.success]);
fprintf('\n=== Batch done: %d/%d sessions succeeded ===\n', nOk, numSessions);

summary = aggregate_subsample_batch_summary(batchResults, opts);
%%
if plotResults
  plot_subsample_batch_summary(summary, opts);
  plot_subsample_batch_trend_summary(summary, opts);
end

if saveBatchResults
  batchMeta = struct('opts', opts, 'sessionTable', sessionTable, 'paths', paths);
  save(batchResultsFile, 'batchResults', 'summary', 'batchMeta', '-v7.3');
  fprintf('Saved batch results: %s\n', batchResultsFile);
end

%% Local functions
function sessionOpts = build_session_opts_from_batch(opts)
% BUILD_SESSION_OPTS_FROM_BATCH - Pass through analysis options to session fn
sessionOpts = struct();
copyFields = {'analyses', 'nNeuronsSubsampleList', 'dataSource', 'collectStart', ...
  'collectEnd', 'd2Window', 'prgWindow', 'brainArea', 'brainAreaCombinations', ...
  'useLog10D2', 'nSubsamples', 'minNeuronsMultiple', 'nMinNeurons', ...
  'enablePermutations', 'nShuffles', 'avalancheDetectionMode', 'powerLawFitMethod', ...
  'gofThreshold', 'prgMethod', 'binSize', 'cvThreshold', 'cutoffDivisors', ...
  'finalCutoffDivisor', 'kappaAxisMax', 'enableSurrogates', 'nSurrogates', ...
  'surrogateMethod', 'firingRateCheckTime', 'minFiringRate', 'maxFiringRate'};
for i = 1:numel(copyFields)
  f = copyFields{i};
  if isfield(opts, f)
    sessionOpts.(f) = opts.(f);
  end
end
sessionOpts.plotResults = false;
end

function sessionTable = build_session_table(sessionTypes)
sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
labelCol = {};

for t = 1:length(sessionTypes)
  sessionType = sessionTypes{t};
  entries = get_sessions_for_type(sessionType);
  for i = 1:numel(entries)
    sessionTypeCol{end+1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end+1, 1} = entries(i).sessionName; %#ok<AGROW>
    if isfield(entries, 'subjectName')
      subjectNameCol{end+1, 1} = entries(i).subjectName; %#ok<AGROW>
    else
      subjectNameCol{end+1, 1} = ''; %#ok<AGROW>
    end
    labelCol{end+1, 1} = entries(i).sessionName; %#ok<AGROW>
  end
end

sessionTable = table(sessionTypeCol, sessionNameCol, subjectNameCol, labelCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'subjectName', 'label'});
end

function entries = get_sessions_for_type(sessionType)
entries = manuscript_sessions_for_type(sessionType);
end

function summary = aggregate_subsample_batch_summary(batchResults, opts)
% AGGREGATE_SUBSAMPLE_BATCH_SUMMARY - Stack session sweeps; mean/SEM by N & task

nList = opts.nNeuronsSubsampleList(:);
specs = metric_specs_for_analyses(opts.analyses);
sessionTypes = opts.sessionTypes;

summary = struct();
summary.nNeuronsSubsample = nList;
summary.sessionTypes = sessionTypes;
summary.analyses = opts.analyses;
summary.metrics = specs;
summary.byMetric = struct();

okMask = [batchResults.success];
okResults = batchResults(okMask);

for m = 1:numel(specs)
  fieldName = specs(m).field;
  nSess = numel(okResults);
  nSizes = numel(nList);
  matAll = nan(nSess, nSizes);
  sessionTypeList = cell(nSess, 1);
  slopeAll = nan(nSess, 1);
  relAll = nan(nSess, 1);
  rhoAll = nan(nSess, 1);

  for i = 1:nSess
    sessionTypeList{i} = okResults(i).sessionType;
    sw = okResults(i).sweep;
    if isempty(sw) || ~isfield(sw, fieldName)
      continue;
    end
    y = sw.(fieldName)(:);
    nHere = sw.nNeuronsSubsample(:);
    for k = 1:nSizes
      idx = find(nHere == nList(k), 1);
      if ~isempty(idx) && idx <= numel(y)
        matAll(i, k) = y(idx);
      end
    end
    if isfield(okResults(i), 'trend') && ~isempty(okResults(i).trend) ...
        && isfield(okResults(i).trend, fieldName)
      tr = okResults(i).trend.(fieldName);
      slopeAll(i) = tr.slope;
      relAll(i) = tr.relativeChange;
      rhoAll(i) = tr.rhoSpearman;
    end
  end

  entry = struct();
  entry.field = fieldName;
  entry.label = specs(m).label;
  entry.family = specs(m).family;
  entry.values = matAll;
  entry.sessionTypes = sessionTypeList;
  entry.slope = slopeAll;
  entry.relativeChange = relAll;
  entry.rhoSpearman = rhoAll;
  entry.meanAcrossSessions = mean(matAll, 1, 'omitnan');
  entry.semAcrossSessions = sem_omitnan(matAll, 1);
  entry.nSessionsPerSize = sum(isfinite(matAll), 1);

  entry.byType = struct();
  for t = 1:numel(sessionTypes)
    typeKey = matlab.lang.makeValidName(sessionTypes{t});
    typeMask = strcmp(sessionTypeList, sessionTypes{t});
    matT = matAll(typeMask, :);
    typeEntry = struct();
    typeEntry.values = matT;
    typeEntry.meanAcrossSessions = mean(matT, 1, 'omitnan');
    typeEntry.semAcrossSessions = sem_omitnan(matT, 1);
    typeEntry.nSessionsPerSize = sum(isfinite(matT), 1);
    typeEntry.slope = slopeAll(typeMask);
    typeEntry.relativeChange = relAll(typeMask);
    typeEntry.rhoSpearman = rhoAll(typeMask);
    entry.byType.(typeKey) = typeEntry;
  end

  summary.byMetric.(fieldName) = entry;
end
end

function s = sem_omitnan(mat, dim)
n = sum(isfinite(mat), dim);
sd = std(mat, 0, dim, 'omitnan');
s = sd ./ sqrt(max(n, 1));
s(n < 1) = nan;
s(n == 1) = 0;
end

function specs = metric_specs_for_analyses(analyses)
specs = struct('field', {}, 'label', {}, 'family', {});
if any(strcmpi(analyses, 'd2'))
  specs(end+1) = struct('field', 'd2Mean', 'label', 'd_2', 'family', 'd2'); %#ok<AGROW>
end
if any(strcmpi(analyses, 'av'))
  specs(end+1) = struct('field', 'tau', 'label', '\tau', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'alpha', 'label', '\alpha', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'paramSD', 'label', '1/(\sigma\nu z)', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'dcc', 'label', 'DCC', 'family', 'av'); %#ok<AGROW>
  specs(end+1) = struct('field', 'decades', 'label', 'decades', 'family', 'av'); %#ok<AGROW>
end
if any(strcmpi(analyses, 'prg'))
  specs(end+1) = struct('field', 'kappaMean', 'label', '\kappa', 'family', 'prg'); %#ok<AGROW>
  specs(end+1) = struct('field', 'djsMean', 'label', 'D_{JS}', 'family', 'prg'); %#ok<AGROW>
end
end

function plot_subsample_batch_summary(summary, opts)
% PLOT_SUBSAMPLE_BATCH_SUMMARY - Metric vs N: session curves + mean±SEM by task

n = summary.nNeuronsSubsample(:);
families = unique({summary.metrics.family}, 'stable');

for f = 1:numel(families)
  fam = families{f};
  famSpecs = summary.metrics(strcmp({summary.metrics.family}, fam));
  if isempty(famSpecs)
    continue;
  end
  nMet = numel(famSpecs);
  nCol = min(3, nMet);
  nRow = ceil(nMet / nCol);
  figure('Name', sprintf('Subsample size batch — %s', upper(fam)), ...
    'Color', 'w', 'Position', [60 60 440*nCol 340*nRow]);

  for m = 1:nMet
    subplot(nRow, nCol, m);
    hold on;
    entry = summary.byMetric.(famSpecs(m).field);

    for i = 1:size(entry.values, 1)
      y = entry.values(i, :);
      ok = isfinite(n') & isfinite(y);
      if ~any(ok)
        continue;
      end
      c = colors_for_tasks(entry.sessionTypes{i})';
      cFaint = 0.35 * c + 0.65;
      plot(n(ok), y(ok), '-', 'Color', cFaint, 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
    end

    for t = 1:numel(opts.sessionTypes)
      typeKey = matlab.lang.makeValidName(opts.sessionTypes{t});
      if ~isfield(entry.byType, typeKey)
        continue;
      end
      te = entry.byType.(typeKey);
      c = colors_for_tasks(opts.sessionTypes{t})';
      yMean = te.meanAcrossSessions(:);
      ySem = te.semAcrossSessions(:);
      ok = isfinite(n) & isfinite(yMean);
      if ~any(ok)
        continue;
      end
      errorbar(n(ok), yMean(ok), ySem(ok), '-o', 'Color', c, ...
        'MarkerFaceColor', c, 'LineWidth', 1.8, 'CapSize', 4, ...
        'DisplayName', opts.sessionTypes{t});
    end

    hold off;
    grid on;
    xlabel('nNeuronsSubsample');
    ylabel(famSpecs(m).label, 'Interpreter', 'tex');
    title(famSpecs(m).label, 'Interpreter', 'tex');
    xlim([min(n)-2, max(n)+2]);
    if m == 1
      legend('Location', 'best', 'Interpreter', 'none');
    end
  end
  sgtitle(sprintf('%s metrics vs subsample size (mean \\pm SEM across sessions)', ...
    upper(fam)), 'Interpreter', 'tex');
end
end

function plot_subsample_batch_trend_summary(summary, opts)
% PLOT_SUBSAMPLE_BATCH_TREND_SUMMARY - Per-session slopes & relative change by task

families = unique({summary.metrics.family}, 'stable');

for f = 1:numel(families)
  fam = families{f};
  famSpecs = summary.metrics(strcmp({summary.metrics.family}, fam));
  if isempty(famSpecs)
    continue;
  end
  nMet = numel(famSpecs);
  figure('Name', sprintf('Subsample trend summary — %s', upper(fam)), ...
    'Color', 'w', 'Position', [80 80 260*nMet 520]);

  for m = 1:nMet
    entry = summary.byMetric.(famSpecs(m).field);

    subplot(2, nMet, m);
    plot_task_swarm(entry, opts.sessionTypes, 'slope', ...
      sprintf('slope(%s vs N)', famSpecs(m).label));
    if m == 1
      ylabel('OLS slope (metric / neuron)');
    end

    subplot(2, nMet, nMet + m);
    plot_task_swarm(entry, opts.sessionTypes, 'relativeChange', ...
      sprintf('\\Delta_{rel}(%s)', famSpecs(m).label));
    if m == 1
      ylabel('(y_{Nmax}-y_{Nmin}) / |y_{Nmin}|');
    end
  end
  sgtitle(sprintf(['%s: per-session sensitivity to subsample size\n', ...
    '(slope = linear trend; \\Delta_{rel} = relative change first\\rightarrowlast N)'], ...
    upper(fam)), 'Interpreter', 'tex');
end
end

function plot_task_swarm(entry, sessionTypes, fieldName, titleStr)
% PLOT_TASK_SWARM - Jittered points + mean±SEM per task for one scalar summary
hold on;
yRef = 0;
plot([-0.5, numel(sessionTypes)+0.5], [yRef yRef], 'k:', 'HandleVisibility', 'off');

for t = 1:numel(sessionTypes)
  typeKey = matlab.lang.makeValidName(sessionTypes{t});
  if ~isfield(entry.byType, typeKey)
    continue;
  end
  vals = entry.byType.(typeKey).(fieldName);
  vals = vals(isfinite(vals));
  c = colors_for_tasks(sessionTypes{t})';
  if isempty(vals)
    continue;
  end
  xJit = t + 0.12 * (rand(size(vals)) - 0.5);
  scatter(xJit, vals, 28, 'filled', ...
    'MarkerFaceColor', c, 'MarkerEdgeColor', c, ...
    'MarkerFaceAlpha', 0.65, 'HandleVisibility', 'off');
  mu = mean(vals);
  se = std(vals) / sqrt(numel(vals));
  if numel(vals) == 1
    se = 0;
  end
  errorbar(t, mu, se, 'o', 'Color', c, 'MarkerFaceColor', c, ...
    'LineWidth', 1.6, 'CapSize', 8, 'DisplayName', sessionTypes{t});
end
hold off;
xlim([0.4, numel(sessionTypes) + 0.6]);
set(gca, 'XTick', 1:numel(sessionTypes), 'XTickLabel', sessionTypes, ...
  'TickLabelInterpreter', 'none');
xtickangle(25);
grid on;
title(titleStr, 'Interpreter', 'tex');
end
