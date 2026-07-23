function out = reach_criticality_d2_accuracy_across_sessions(opts)
% REACH_CRITICALITY_D2_ACCURACY_ACROSS_SESSIONS - Batch pre/post d2 by outcome
%
% Variables:
%   opts - Options struct. Fields:
%     .sessions            - Cell of reach session names (default reach_session_list)
%     .runBatch            - If true, call reach_criticality_d2_accuracy per session
%     .plotResults         - If true, plot across-session summary
%     .saveFigure          - Export summary figure(s)
%     .saveBatchResults    - Save batch .mat
%     .batchResultsFile    - Path for batch .mat (default under dropPath)
%     Plus all fields accepted by reach_criticality_d2_accuracy (brainArea,
%     d2Window, pre/postReachBuffer, useLog10D2, collectStart/End, ...).
%     Per-session makePlots/saveFigure are forced off in the batch loop.
%
% Goal:
%   Run reach_criticality_d2_accuracy across reach sessions (no per-session
%   figures). Aggregate session-mean d2 for correct vs error (pre- and
%   post-reach), plot session means plus across-session mean +/- SEM, and
%   test correct vs error with paired Wilcoxon signed-rank (and paired t-test)
%   across sessions, separately for pre- and post-reach.
%
% Usage:
%   opts = reach_criticality_d2_accuracy_across_sessions();
%   out = reach_criticality_d2_accuracy_across_sessions(opts);
%
% Returns:
%   With no inputs: default options struct.
%   Otherwise: batchResults, batchTable, statsByArea, figHandles, paths, ...

setup_reach_criticality_d2_accuracy_across_sessions_paths();

if nargin == 0
  out = fill_d2_accuracy_across_sessions_opts(struct());
  return;
end
if nargin < 1 || isempty(opts)
  opts = struct();
end
opts = fill_d2_accuracy_across_sessions_opts(opts);
paths = get_paths();

if isempty(opts.batchResultsFile)
  areaTag = format_areas_label_local(opts.brainArea);
  if isempty(areaTag)
    areaTag = 'all_areas';
  end
  opts.batchResultsFile = fullfile(paths.dropPath, 'reach_task', 'results', ...
    'd2_accuracy_across_sessions', ...
    sprintf('reach_criticality_d2_accuracy_across_sessions_%s.mat', areaTag));
end

sessions = opts.sessions;
if isempty(sessions)
  sessions = reach_session_list();
  opts.sessions = sessions;
end
nSessions = numel(sessions);

fprintf('\n=== Reach criticality d2 accuracy across sessions ===\n');
fprintf('Sessions: %d\n', nSessions);
if isempty(opts.brainArea)
  fprintf('Brain area: (all areas)\n');
else
  fprintf('Brain area: %s\n', char(opts.brainArea));
end
fprintf('d2Window=%.2fs | preReachBuffer=%d ms | postReachBuffer=%d ms\n', ...
  opts.d2Window, opts.preReachBuffer, opts.postReachBuffer);
fprintf('useLog10D2: %d | runBatch: %d | plotResults: %d\n', ...
  opts.useLog10D2, opts.runBatch, opts.plotResults);

if opts.runBatch
  [batchResults, sessionOk, sessionErrors] = run_d2_accuracy_session_batch(sessions, opts);
  batchMeta = pack_d2_accuracy_across_sessions_meta(opts);
  if opts.saveBatchResults
    batchDir = fileparts(opts.batchResultsFile);
    if ~isempty(batchDir) && ~exist(batchDir, 'dir')
      mkdir(batchDir);
    end
    save(opts.batchResultsFile, 'batchResults', 'sessionOk', 'sessionErrors', ...
      'batchMeta', 'sessions', '-v7.3');
    fprintf('\nSaved batch results: %s\n', opts.batchResultsFile);
  end
else
  if ~isfile(opts.batchResultsFile)
    error('reach_criticality_d2_accuracy_across_sessions:NoBatchFile', ...
      'Batch file not found: %s. Set runBatch true to compute.', opts.batchResultsFile);
  end
  loaded = load(opts.batchResultsFile);
  batchResults = loaded.batchResults;
  sessionOk = loaded.sessionOk;
  if isfield(loaded, 'sessionErrors')
    sessionErrors = loaded.sessionErrors;
  else
    sessionErrors = cell(nSessions, 1);
  end
  if isfield(loaded, 'sessions') && ~isempty(loaded.sessions)
    sessions = loaded.sessions;
    opts.sessions = sessions;
    nSessions = numel(sessions);
  end
  if isfield(loaded, 'batchMeta')
    batchMeta = loaded.batchMeta;
  else
    batchMeta = pack_d2_accuracy_across_sessions_meta(opts);
  end
  fprintf('\nLoaded batch results: %s\n', opts.batchResultsFile);
end

fprintf('\nCompleted %d/%d sessions successfully.\n', sum(sessionOk), nSessions);
if any(~sessionOk)
  fprintf('Failed sessions:\n');
  failedIdx = find(~sessionOk);
  for k = 1:numel(failedIdx)
    iSess = failedIdx(k);
    fprintf('  %s — %s\n', sessions{iSess}, ...
      get_session_error_message(sessionErrors, iSess));
  end
end

[batchTable, areaNames] = build_d2_accuracy_across_sessions_table( ...
  sessions, batchResults, sessionOk, opts.brainArea);
statsByArea = run_d2_accuracy_across_sessions_stats(batchTable, areaNames);
print_d2_accuracy_across_sessions_stats(statsByArea, opts.useLog10D2);

figHandles = struct();
figHandles.summary = gobjects(0);
if opts.plotResults
  figHandles.summary = plot_d2_accuracy_across_sessions( ...
    batchTable, statsByArea, areaNames, opts);
  if opts.saveFigure && isgraphics(figHandles.summary)
    saveDir = fullfile(paths.dropPath, 'reach_task', 'results', ...
      'd2_accuracy_across_sessions');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    areaTag = format_areas_label_local(opts.brainArea);
    if isempty(areaTag)
      areaTag = 'all_areas';
    end
    plotBase = fullfile(saveDir, sprintf( ...
      'reach_criticality_d2_accuracy_across_sessions_%s_W%.1f_pre%d_post%d', ...
      areaTag, opts.d2Window, opts.preReachBuffer, opts.postReachBuffer));
    exportgraphics(figHandles.summary, [plotBase, '.png'], 'Resolution', 300);
    exportgraphics(figHandles.summary, [plotBase, '.eps'], 'ContentType', 'vector');
    fprintf('\nSaved figure: %s\n', plotBase);
  end
end

fprintf('\n=== Done ===\n');

out = struct();
out.sessions = sessions;
out.sessionOk = sessionOk;
out.sessionErrors = sessionErrors;
out.batchResults = batchResults;
out.batchTable = batchTable;
out.statsByArea = statsByArea;
out.areaNames = areaNames;
out.batchMeta = batchMeta;
out.figHandles = figHandles;
out.config = opts;
out.paths = paths;
if isfield(opts, 'batchResultsFile')
  out.batchResultsFile = opts.batchResultsFile;
end
end

%% -------------------------------------------------------------------------
%% Defaults and paths
%% -------------------------------------------------------------------------

function opts = fill_d2_accuracy_across_sessions_opts(opts)
% FILL_D2_ACCURACY_ACROSS_SESSIONS_OPTS - Defaults for across-session batch

sessionDefaults = reach_criticality_d2_accuracy();
preserveCollectEndEmpty = isfield(opts, 'collectEnd') && isempty(opts.collectEnd);
preserveD2WindowEmpty = isfield(opts, 'd2Window') && isempty(opts.d2Window);
preserveBrainAreaEmpty = isfield(opts, 'brainArea') && isempty(opts.brainArea);

opts = merge_struct_defaults(opts, sessionDefaults);

if ~isfield(opts, 'sessions') || isempty(opts.sessions)
  opts.sessions = reach_session_list();
end
if ~isfield(opts, 'runBatch') || isempty(opts.runBatch)
  opts.runBatch = true;
end
if ~isfield(opts, 'plotResults') || isempty(opts.plotResults)
  opts.plotResults = true;
end
if ~isfield(opts, 'saveBatchResults') || isempty(opts.saveBatchResults)
  opts.saveBatchResults = false;
end
if ~isfield(opts, 'batchResultsFile')
  opts.batchResultsFile = '';
end
if ~isfield(opts, 'plotConfig') || isempty(opts.plotConfig)
  opts.plotConfig = struct();
end
if exist('fill_manuscript_plot_config', 'file')
  opts.plotConfig = fill_manuscript_plot_config(opts.plotConfig);
else
  opts.plotConfig = fill_fallback_plot_config(opts.plotConfig);
end

% Per-session figures are intentionally off for the across-sessions script
opts.makePlots = false;
opts.saveFigure = logical(opts.saveFigure);
if preserveCollectEndEmpty
  opts.collectEnd = [];
end
if preserveD2WindowEmpty
  opts.d2Window = [];
end
if preserveBrainAreaEmpty
  opts.brainArea = '';
end
end

function plotConfig = fill_fallback_plot_config(plotConfig)
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
if ~isfield(plotConfig, 'lineWidth') || isempty(plotConfig.lineWidth)
  plotConfig.lineWidth = 1.5;
end
if ~isfield(plotConfig, 'errorCapSize') || isempty(plotConfig.errorCapSize)
  plotConfig.errorCapSize = 8;
end
if ~isfield(plotConfig, 'scatterMarkerSize') || isempty(plotConfig.scatterMarkerSize)
  plotConfig.scatterMarkerSize = 48;
end
if ~isfield(plotConfig, 'legendFontSize') || isempty(plotConfig.legendFontSize)
  plotConfig.legendFontSize = 11;
end
end

function setup_reach_criticality_d2_accuracy_across_sessions_paths()
% SETUP_REACH_CRITICALITY_D2_ACCURACY_ACROSS_SESSIONS_PATHS - Add src paths

scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_']) || ~isfolder(fullfile(scriptDir, '..', 'reach_task'))
  resolved = which('reach_criticality_d2_accuracy_across_sessions');
  if ~isempty(resolved)
    scriptDir = fileparts(resolved);
  end
end
srcPath = fullfile(scriptDir, '..');
pathDirs = {
  srcPath
  fullfile(srcPath, 'reach_task')
  fullfile(srcPath, 'criticality', 'scripts')
  fullfile(srcPath, 'criticality', 'analyses')
  fullfile(srcPath, 'criticality_manuscript')
  fullfile(srcPath, 'session_prep', 'data_prep')
  fullfile(srcPath, 'session_prep', 'utils')
  fullfile(srcPath, 'data_prep')
  fullfile(srcPath, 'sliding_window_prep', 'utils')
  fullfile(srcPath, 'criticality')
  };
for iDir = 1:numel(pathDirs)
  if isfolder(pathDirs{iDir})
    addpath(pathDirs{iDir});
  end
end
end

function batchMeta = pack_d2_accuracy_across_sessions_meta(opts)
batchMeta = struct( ...
  'brainArea', opts.brainArea, ...
  'd2Window', opts.d2Window, ...
  'preReachBuffer', opts.preReachBuffer, ...
  'postReachBuffer', opts.postReachBuffer, ...
  'useLog10D2', opts.useLog10D2, ...
  'collectStart', opts.collectStart, ...
  'collectEnd', opts.collectEnd, ...
  'nMinNeurons', opts.nMinNeurons);
end

%% -------------------------------------------------------------------------
%% Batch loop
%% -------------------------------------------------------------------------

function [batchResults, sessionOk, sessionErrors] = run_d2_accuracy_session_batch(sessions, opts)
% RUN_D2_ACCURACY_SESSION_BATCH - Call reach_criticality_d2_accuracy per session
%
% Variables:
%   sessions - Cell of session names
%   opts     - Shared options (makePlots forced false)
%
% Goal:
%   Collect per-session outputs without generating individual figures.

nSessions = numel(sessions);
sessionOpts = opts;
sessionOpts.makePlots = false;
sessionOpts.saveFigure = false;
sessionOpts.saveResults = false;

batchResults = cell(nSessions, 1);
sessionOk = false(nSessions, 1);
sessionErrors = cell(nSessions, 1);

for iSess = 1:nSessions
  sessionName = sessions{iSess};
  fprintf('\n========== Session %d/%d: %s ==========\n', iSess, nSessions, sessionName);
  try
    batchResults{iSess} = reach_criticality_d2_accuracy(sessionName, sessionOpts);
    sessionOk(iSess) = true;
  catch ME
    sessionOk(iSess) = false;
    sessionErrors{iSess} = ME;
    warning('reach_criticality_d2_accuracy_across_sessions:SessionFailed', ...
      'Session %s failed: %s', sessionName, ME.message);
  end
end
end

function msg = get_session_error_message(sessionErrors, iSess)
% GET_SESSION_ERROR_MESSAGE - Safe message from sessionErrors cell

msg = '';
if nargin < 2 || isempty(iSess) || ~isscalar(iSess) || ~isfinite(iSess)
  return;
end
iSess = round(double(iSess));
if iSess < 1 || iSess > numel(sessionErrors)
  return;
end
errEntry = sessionErrors{iSess};
if isa(errEntry, 'MException')
  msg = errEntry.message;
elseif ischar(errEntry) || isstring(errEntry)
  msg = char(string(errEntry));
end
end

%% -------------------------------------------------------------------------
%% Aggregation and statistics
%% -------------------------------------------------------------------------

function [batchTable, areaNames] = build_d2_accuracy_across_sessions_table( ...
    sessions, batchResults, sessionOk, brainArea)
% BUILD_D2_ACCURACY_ACROSS_SESSIONS_TABLE - Session-mean correct/error d2
%
% Variables:
%   sessions     - Session name cell
%   batchResults - Cell of reach_criticality_d2_accuracy outputs
%   sessionOk    - Logical success flags
%   brainArea    - Preferred area name; '' keeps all areas found
%
% Goal:
%   One row per session with pre/post meanCorrect and meanError per area.

areaNames = collect_batch_area_names(batchResults, sessionOk, brainArea);
nSessions = numel(sessions);
nAreas = numel(areaNames);

batchTable = table();
batchTable.sessionName = sessions(:);
batchTable.ok = sessionOk(:);

for a = 1:nAreas
  areaKey = matlab.lang.makeValidName(areaNames{a});
  batchTable.(['pre_meanCorrect_' areaKey]) = nan(nSessions, 1);
  batchTable.(['pre_meanError_' areaKey]) = nan(nSessions, 1);
  batchTable.(['pre_nCorrect_' areaKey]) = nan(nSessions, 1);
  batchTable.(['pre_nError_' areaKey]) = nan(nSessions, 1);
  batchTable.(['pre_pRanksum_' areaKey]) = nan(nSessions, 1);
  batchTable.(['post_meanCorrect_' areaKey]) = nan(nSessions, 1);
  batchTable.(['post_meanError_' areaKey]) = nan(nSessions, 1);
  batchTable.(['post_nCorrect_' areaKey]) = nan(nSessions, 1);
  batchTable.(['post_nError_' areaKey]) = nan(nSessions, 1);
  batchTable.(['post_pRanksum_' areaKey]) = nan(nSessions, 1);
end

for iSess = 1:nSessions
  if ~sessionOk(iSess) || ~isstruct(batchResults{iSess})
    continue;
  end
  sessOut = batchResults{iSess};
  for a = 1:nAreas
    areaName = areaNames{a};
    areaKey = matlab.lang.makeValidName(areaName);
    stPre = find_area_stats(sessOut.preReach.statsByArea, areaName);
    stPost = find_area_stats(sessOut.postReach.statsByArea, areaName);
    if ~isempty(stPre)
      batchTable.(['pre_meanCorrect_' areaKey])(iSess) = stPre.meanCorrect;
      batchTable.(['pre_meanError_' areaKey])(iSess) = stPre.meanError;
      batchTable.(['pre_nCorrect_' areaKey])(iSess) = stPre.nCorrect;
      batchTable.(['pre_nError_' areaKey])(iSess) = stPre.nError;
      batchTable.(['pre_pRanksum_' areaKey])(iSess) = stPre.pRanksum;
    end
    if ~isempty(stPost)
      batchTable.(['post_meanCorrect_' areaKey])(iSess) = stPost.meanCorrect;
      batchTable.(['post_meanError_' areaKey])(iSess) = stPost.meanError;
      batchTable.(['post_nCorrect_' areaKey])(iSess) = stPost.nCorrect;
      batchTable.(['post_nError_' areaKey])(iSess) = stPost.nError;
      batchTable.(['post_pRanksum_' areaKey])(iSess) = stPost.pRanksum;
    end
  end
end
end

function areaNames = collect_batch_area_names(batchResults, sessionOk, brainArea)
% COLLECT_BATCH_AREA_NAMES - Union of analyzed areas (or preferred brainArea)

areaNames = {};
if ~isempty(brainArea)
  areaNames = {char(brainArea)};
  return;
end

for iSess = 1:numel(batchResults)
  if ~sessionOk(iSess) || ~isstruct(batchResults{iSess})
    continue;
  end
  sessOut = batchResults{iSess};
  if ~isfield(sessOut, 'preReach') || ~isfield(sessOut.preReach, 'statsByArea')
    continue;
  end
  for iA = 1:numel(sessOut.preReach.statsByArea)
    name = sessOut.preReach.statsByArea(iA).areaName;
    if ~any(strcmp(areaNames, name))
      areaNames{end + 1} = name; %#ok<AGROW>
    end
  end
end
end

function st = find_area_stats(statsByArea, areaName)
% FIND_AREA_STATS - Match stats struct by areaName

st = [];
if isempty(statsByArea)
  return;
end
for i = 1:numel(statsByArea)
  if strcmp(statsByArea(i).areaName, areaName)
    st = statsByArea(i);
    return;
  end
end
end

function statsByArea = run_d2_accuracy_across_sessions_stats(batchTable, areaNames)
% RUN_D2_ACCURACY_ACROSS_SESSIONS_STATS - Paired correct vs error across sessions
%
% Variables:
%   batchTable - Session-level means from build_d2_accuracy_across_sessions_table
%   areaNames  - Areas to test
%
% Goal:
%   For pre- and post-reach separately, test whether session-mean d2 differs
%   between correct and error (Wilcoxon signed-rank + paired t-test).

nAreas = numel(areaNames);
emptyRow = struct( ...
  'areaName', '', ...
  'timing', '', ...
  'n', nan, ...
  'meanCorrect', nan, ...
  'semCorrect', nan, ...
  'meanError', nan, ...
  'semError', nan, ...
  'meanDiff', nan, ...
  'semDiff', nan, ...
  'pSignrank', nan, ...
  'pTtest', nan, ...
  'sessionMeanCorrect', [], ...
  'sessionMeanError', [], ...
  'sessionNames', {{}});
statsByArea = repmat(emptyRow, nAreas, 2);  % rows: areas; cols: pre=1, post=2
timingTags = {'pre', 'post'};
timingLabels = {'pre-reach', 'post-reach'};

okRows = batchTable.ok;
for a = 1:nAreas
  areaName = areaNames{a};
  areaKey = matlab.lang.makeValidName(areaName);
  for t = 1:2
    corrCol = [timingTags{t}, '_meanCorrect_', areaKey];
    errCol = [timingTags{t}, '_meanError_', areaKey];
    corrVals = batchTable.(corrCol)(okRows);
    errVals = batchTable.(errCol)(okRows);
    sessNames = batchTable.sessionName(okRows);
    valid = isfinite(corrVals) & isfinite(errVals);
    corrVals = corrVals(valid);
    errVals = errVals(valid);
    sessNames = sessNames(valid);

    row = emptyRow;
    row.areaName = areaName;
    row.timing = timingLabels{t};
    row.n = numel(corrVals);
    row.sessionMeanCorrect = corrVals;
    row.sessionMeanError = errVals;
    row.sessionNames = sessNames;
    if row.n == 0
      statsByArea(a, t) = row;
      continue;
    end

    row.meanCorrect = mean(corrVals);
    row.meanError = mean(errVals);
    if row.n > 1
      row.semCorrect = std(corrVals) / sqrt(row.n);
      row.semError = std(errVals) / sqrt(row.n);
    else
      row.semCorrect = 0;
      row.semError = 0;
    end
    diffs = corrVals - errVals;
    row.meanDiff = mean(diffs);
    if row.n > 1
      row.semDiff = std(diffs) / sqrt(row.n);
    else
      row.semDiff = 0;
    end

    if row.n >= 2
      try
        row.pSignrank = signrank(corrVals, errVals);
      catch
        row.pSignrank = nan;
      end
      try
        [~, row.pTtest] = ttest(corrVals, errVals);
      catch
        row.pTtest = nan;
      end
    end
    statsByArea(a, t) = row;
  end
end
end

function print_d2_accuracy_across_sessions_stats(statsByArea, useLog10D2)
% PRINT_D2_ACCURACY_ACROSS_SESSIONS_STATS - Command-window paired stats

if useLog10D2
  metricLabel = 'log10(d2)';
else
  metricLabel = 'd2';
end

fprintf('\n=== Across-session %s: correct vs error (paired) ===\n', metricLabel);
fprintf(['%-8s  %-10s  n   mean_corr+/-SEM   mean_err+/-SEM    ', ...
  'diff+/-SEM      p_signrank  p_ttest\n'], 'Area', 'Timing');
for a = 1:size(statsByArea, 1)
  for t = 1:size(statsByArea, 2)
    st = statsByArea(a, t);
    if st.n == 0 || ~isfinite(st.meanCorrect)
      fprintf('%-8s  %-10s  no paired session means\n', st.areaName, st.timing);
      continue;
    end
    fprintf(['%-8s  %-10s  %-3d %7.4f+/-%-7.4f %7.4f+/-%-7.4f ', ...
      '%+7.4f+/-%-6.4f  %-10s  %-10s\n'], ...
      st.areaName, st.timing, st.n, ...
      st.meanCorrect, st.semCorrect, st.meanError, st.semError, ...
      st.meanDiff, st.semDiff, ...
      format_p_value(st.pSignrank), format_p_value(st.pTtest));
  end
end
fprintf(['diff = correct - error; units = session means; ', ...
  'signrank = Wilcoxon signed-rank (paired across sessions)\n']);
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

%% -------------------------------------------------------------------------
%% Plotting
%% -------------------------------------------------------------------------

function fig = plot_d2_accuracy_across_sessions(batchTable, statsByArea, areaNames, opts)
% PLOT_D2_ACCURACY_ACROSS_SESSIONS - Pre/post panels: session means + grand mean
%
% Variables:
%   batchTable  - Session-level means
%   statsByArea - Across-session paired stats (areas x [pre, post])
%   areaNames   - Areas to plot (columns)
%   opts        - Includes useLog10D2, plotConfig, buffers, brainArea
%
% Goal:
%   Two rows (pre, post) x nAreas columns. Each panel: Correct vs Error with
%   one marker per session (paired gray lines), plus across-session mean+/-SEM.

plotConfig = opts.plotConfig;
nAreas = numel(areaNames);
if nAreas == 0
  warning('reach_criticality_d2_accuracy_across_sessions:NoAreas', ...
    'No areas available for across-session plot.');
  fig = gobjects(0);
  return;
end

if opts.useLog10D2
  yLabelStr = 'log_{10}(d2)';
  yLabInterpreter = 'tex';
else
  yLabelStr = 'd2';
  yLabInterpreter = 'none';
end

correctColor = [0.15, 0.45, 0.75];
errorColor = [0.85, 0.33, 0.1];
sessionMarkerSize = plotConfig.scatterMarkerSize;
meanMarkerSize = max(8, plotConfig.scatterMarkerSize / 4);

fig = figure('Color', 'w', 'Name', 'Across-session d2: correct vs error', ...
  'NumberTitle', 'off');
if exist('position_figure_full_monitor', 'file')
  position_figure_full_monitor(fig);
else
  set(fig, 'Position', [80, 80, max(420, 320 * nAreas), 720]);
end

nRow = 2;
useTight = exist('tight_subplot', 'file');
if useTight
  ha = tight_subplot(nRow, nAreas, [0.08 0.06], [0.08 0.12], [0.08 0.04]);
else
  ha = gobjects(nRow * nAreas, 1);
  for ii = 1:(nRow * nAreas)
    ha(ii) = subplot(nRow, nAreas, ii);
  end
end

rowTitles = {'Pre-reach', 'Post-reach'};
allVals = [];
for a = 1:nAreas
  for t = 1:2
    st = statsByArea(a, t);
    allVals = [allVals; st.sessionMeanCorrect(:); st.sessionMeanError(:)]; %#ok<AGROW>
  end
end
allVals = allVals(isfinite(allVals));
if isempty(allVals)
  yLimGlobal = [0, 1];
elseif numel(unique(allVals)) == 1
  yPad = max(0.05, 0.05 * abs(allVals(1)));
  yLimGlobal = [allVals(1) - yPad, allVals(1) + yPad];
else
  yPad = 0.12 * (max(allVals) - min(allVals));
  yLimGlobal = [min(allVals) - yPad, max(allVals) + yPad];
end

for rowIdx = 1:nRow
  for colIdx = 1:nAreas
    ax = ha((rowIdx - 1) * nAreas + colIdx);
    st = statsByArea(colIdx, rowIdx);
    hold(ax, 'on');

    corrVals = st.sessionMeanCorrect(:);
    errVals = st.sessionMeanError(:);
    n = numel(corrVals);
    for i = 1:n
      plot(ax, [1, 2], [corrVals(i), errVals(i)], '-', ...
        'Color', [0.75, 0.75, 0.75], 'LineWidth', 1, 'HandleVisibility', 'off');
    end
    if n > 0
      scatter(ax, ones(n, 1), corrVals, sessionMarkerSize, correctColor, 'filled', ...
        'MarkerFaceAlpha', 0.7, 'DisplayName', 'session mean (correct)');
      scatter(ax, 2 * ones(n, 1), errVals, sessionMarkerSize, errorColor, 'filled', ...
        'MarkerFaceAlpha', 0.7, 'DisplayName', 'session mean (error)');
    end

    if n >= 1 && isfinite(st.meanCorrect)
      errorbar(ax, 1, st.meanCorrect, st.semCorrect, 'o', ...
        'Color', correctColor, 'MarkerFaceColor', correctColor, ...
        'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
        'MarkerSize', meanMarkerSize, ...
        'DisplayName', 'across-session mean \pm SEM');
      errorbar(ax, 2, st.meanError, st.semError, 'o', ...
        'Color', errorColor, 'MarkerFaceColor', errorColor, ...
        'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
        'MarkerSize', meanMarkerSize, 'HandleVisibility', 'off');
    end

    pText = format_p_value(st.pSignrank);
    titleStr = sprintf('%s | n=%d sessions | p_{signrank}=%s', ...
      st.areaName, st.n, pText);
    if exist('apply_manuscript_axes_style', 'file')
      apply_manuscript_axes_style(ax, plotConfig, '', '', titleStr, 'tex');
    else
      title(ax, titleStr, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'tex');
      set(ax, 'FontSize', plotConfig.tickLabelFontSize, ...
        'LineWidth', plotConfig.axesLineWidth, 'Box', 'off', 'TickDir', 'out');
    end
    set(ax, 'XTick', [1, 2], 'XTickLabel', {'Correct', 'Error'}, ...
      'XTickLabelMode', 'manual', 'TickLabelInterpreter', 'none');
    xlim(ax, [0.5, 2.5]);
    ylim(ax, yLimGlobal);
    if colIdx == 1
      ylabel(ax, sprintf('%s\n%s', rowTitles{rowIdx}, yLabelStr), ...
        'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', yLabInterpreter);
    end
    if rowIdx == 1 && colIdx == nAreas
      legend(ax, 'Location', 'best', 'FontSize', plotConfig.legendFontSize);
    end
    hold(ax, 'off');
  end
end

if ~isempty(opts.brainArea)
  areaTitle = char(opts.brainArea);
else
  areaTitle = strjoin(areaNames, ', ');
end
sgtitle(fig, sprintf([ ...
  'Across sessions: pre/post-reach d2 correct vs error | %s | ', ...
  'W=%.1fs, preBuf=%d ms, postBuf=%d ms'], ...
  areaTitle, opts.d2Window, opts.preReachBuffer, opts.postReachBuffer), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'none', 'FontWeight', 'bold');
end

function label = format_areas_label_local(areaNames)
% FORMAT_AREAS_LABEL_LOCAL - Filesystem-safe area tag

if isempty(areaNames)
  label = '';
  return;
end
if iscell(areaNames)
  label = strjoin(areaNames(:)', '_');
else
  label = char(areaNames);
end
label = matlab.lang.makeValidName(label);
end
