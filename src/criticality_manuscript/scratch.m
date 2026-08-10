%% Load one spontaneous session — put behavior labels in bhvID
%
% Uses the same load path as manuscript analyses (load_session_data →
% load_spontaneous_data). Behavior codes are already on dataStruct.bhvID at
% opts.fsBhv; copy them into workspace variable bhvID.

setup_criticality_manuscript_paths('criticality_multiple_metrics_across_tasks');
paths = get_paths();

sessionType = 'spontaneous';
subjectName = 'ag25290';
sessionName = 'ag112321_1';  % from spontaneous_session_list
sessionName = 'ag112321_1';  % from spontaneous_session_list
dataSource = 'spikes';
collectStart = 0;
collectEnd = [];  % [] = full session

loadOpts = neuro_behavior_options();
loadOpts.collectStart = collectStart;
loadOpts.collectEnd = collectEnd;

loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});

bhvID = [];
if isfield(dataStruct, 'bhvID') && ~isempty(dataStruct.bhvID)
  bhvID = dataStruct.bhvID(:);
end

fprintf('\n=== Spontaneous session loaded ===\n');
fprintf('Subject / session: %s / %s\n', subjectName, sessionName);
if isempty(bhvID)
  warning('scratch:NoBhvID', 'No behavior labels found (bhvID is empty).');
else
  fsBhv = [];
  if isfield(dataStruct, 'fsBhv') && ~isempty(dataStruct.fsBhv)
    fsBhv = dataStruct.fsBhv;
  elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'fsBhv')
    fsBhv = dataStruct.opts.fsBhv;
  end
  fprintf('bhvID: %d samples', numel(bhvID));
  if ~isempty(fsBhv)
    fprintf(' (fsBhv = %.3g Hz, ~%.1f s)\n', fsBhv, numel(bhvID) / fsBhv);
  else
    fprintf('\n');
  end
  fprintf('Unique behavior codes: %s\n', mat2str(unique(bhvID)'));
end

%% Behavior-label pie charts across all spontaneous sessions
%
% Load every session in spontaneous_session_list, compute frame proportions
% per behavior code, and plot one pie per session. Colors are fixed by code
% via colors_for_behaviors so the same label matches across panels.

setup_criticality_manuscript_paths('criticality_multiple_metrics_across_tasks');
paths = get_paths();

sessionType = 'spontaneous';
dataSource = 'spikes';
collectStart = 0;
collectEnd = [];  % [] = full session
binSizePop = 0.05;  % s; population activity bin width
brainAreaPop = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();
sessionList = spontaneous_session_list();
nSess = numel(sessionList);

loadOptsBase = neuro_behavior_options();
loadOptsBase.collectStart = collectStart;
loadOptsBase.collectEnd = collectEnd;

fprintf('\n=== Spontaneous behavior pies (%d sessions) ===\n', nSess);
if isempty(collectEnd)
  fprintf('collectEnd = [] (full session); printing loaded durations.\n');
end
fprintf('popActivity area: %s | binSize: %.3g s\n', brainAreaPop, binSizePop);

sessionBhv = repmat(struct( ...
  'subjectName', '', 'sessionName', '', 'label', '', ...
  'bhvID', [], 'codeToName', [], 'durationSec', nan, 'fsBhv', nan, ...
  'popActivity', [], 'popTime', [], ...
  'success', false, 'skipReason', ''), nSess, 1);

allCodes = [];
for iSess = 1:nSess
  subjectName = sessionList(iSess).subjectName;
  sessionName = sessionList(iSess).sessionName;
  sessionBhv(iSess).subjectName = subjectName;
  sessionBhv(iSess).sessionName = sessionName;
  sessionBhv(iSess).label = sessionName;

  fprintf('Session %d/%d: %s / %s\n', iSess, nSess, subjectName, sessionName);
  try
    loadOpts = loadOptsBase;
    loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
    dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});
    durationSec = scratch_session_duration_sec(dataStruct, collectStart);
    sessionBhv(iSess).durationSec = durationSec;
    if isempty(collectEnd)
      if isfinite(durationSec)
        fprintf('  Duration: %.1f s (%.2f min)\n', durationSec, durationSec / 60);
      else
        fprintf('  Duration: unknown\n');
      end
    end

    [popActivity, popTime] = scratch_session_pop_activity( ...
      dataStruct, brainAreaPop, brainAreaCombinations, binSizePop, collectStart);
    sessionBhv(iSess).popActivity = popActivity;
    sessionBhv(iSess).popTime = popTime;
    if ~isempty(popActivity)
      fprintf('  popActivity: %d bins (%.1f s)\n', numel(popActivity), ...
        popTime(end) - popTime(1) + binSizePop);
    end

    if ~isfield(dataStruct, 'bhvID') || isempty(dataStruct.bhvID)
      sessionBhv(iSess).skipReason = 'empty bhvID';
      warning('scratch:NoBhvID', 'No bhvID for %s; skipping pie.', sessionName);
      continue;
    end
    bhvID = dataStruct.bhvID(:);
    sessionBhv(iSess).bhvID = bhvID;
    sessionBhv(iSess).codeToName = scratch_bhv_code_to_name_map(dataStruct);
    if isfield(dataStruct, 'fsBhv') && ~isempty(dataStruct.fsBhv)
      sessionBhv(iSess).fsBhv = dataStruct.fsBhv;
    elseif isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'fsBhv')
      sessionBhv(iSess).fsBhv = dataStruct.opts.fsBhv;
    else
      sessionBhv(iSess).fsBhv = loadOptsBase.fsBhv;
    end
    sessionBhv(iSess).success = true;
    allCodes = [allCodes; unique(bhvID)]; %#ok<AGROW>
  catch ME
    sessionBhv(iSess).skipReason = ME.message;
    warning('scratch:SpontaneousPieLoadFailed', 'Failed %s: %s', sessionName, ME.message);
  end
end

if isempty(collectEnd)
  fprintf('\n--- Session durations (full collect) ---\n');
  for iSess = 1:nSess
    dur = sessionBhv(iSess).durationSec;
    if isfinite(dur)
      fprintf('  %s / %s: %.1f s (%.2f min)\n', ...
        sessionBhv(iSess).subjectName, sessionBhv(iSess).sessionName, dur, dur / 60);
    else
      fprintf('  %s / %s: unknown (%s)\n', ...
        sessionBhv(iSess).subjectName, sessionBhv(iSess).sessionName, ...
        sessionBhv(iSess).skipReason);
    end
  end
end

okMask = [sessionBhv.success];
nOk = sum(okMask);
if nOk < 1
  error('scratch:NoSpontaneousBhv', 'No spontaneous sessions with behavior labels.');
end

allCodes = unique(allCodes(:));
allCodes = allCodes(isfinite(allCodes));
[codeColors, codeNames] = scratch_behavior_code_colors_and_names(allCodes, sessionBhv(okMask));

nCol = min(4, nOk);
nRow = ceil(nOk / nCol);
fig = figure('Color', 'w', 'Name', 'Spontaneous behavior proportions');
position_scratch_figure_full_monitor(fig);
tiled = tiledlayout(fig, nRow, nCol, 'TileSpacing', 'compact', 'Padding', 'compact');

okIdx = find(okMask);
for iPlot = 1:nOk
  iSess = okIdx(iPlot);
  bhvID = sessionBhv(iSess).bhvID;
  counts = zeros(numel(allCodes), 1);
  for iCode = 1:numel(allCodes)
    counts(iCode) = sum(bhvID == allCodes(iCode));
  end
  present = counts > 0;
  if ~any(present)
    continue;
  end

  ax = nexttile(tiled);
  pieCounts = counts(present);
  pieColors = codeColors(present, :);
  pieNames = codeNames(present);
  pieProps = 100 * pieCounts / sum(pieCounts);
  hPie = pie(ax, pieCounts);
  % pie returns [patch; text; patch; text; ...]
  for iSlice = 1:numel(pieCounts)
    patchIdx = 2 * iSlice - 1;
    textIdx = 2 * iSlice;
    set(hPie(patchIdx), 'FaceColor', pieColors(iSlice, :), 'EdgeColor', [1 1 1], ...
      'LineWidth', 0.5);
    if pieProps(iSlice) >= 4
      set(hPie(textIdx), 'String', sprintf('%.0f%%', pieProps(iSlice)), ...
        'FontSize', 8);
    else
      set(hPie(textIdx), 'String', '');
    end
  end
  title(ax, sessionBhv(iSess).label, 'Interpreter', 'none', 'FontSize', 11);
end

sgtitle(tiled, 'Spontaneous behavior label proportions', 'FontWeight', 'bold');

% Shared legend (same colors across sessions)
figLeg = figure('Color', 'w', 'Name', 'Behavior label legend', ...
  'Position', [120 120 360 max(220, 28 * numel(allCodes) + 60)]);
axLeg = axes(figLeg);
hold(axLeg, 'on');
axis(axLeg, 'off');
for iCode = 1:numel(allCodes)
  y = numel(allCodes) - iCode + 1;
  plot(axLeg, 0.1, y, 's', 'MarkerSize', 12, 'MarkerFaceColor', codeColors(iCode, :), ...
    'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
  text(axLeg, 0.25, y, sprintf('%s (code %g)', codeNames{iCode}, allCodes(iCode)), ...
    'FontSize', 11, 'Interpreter', 'none', 'VerticalAlignment', 'middle');
end
xlim(axLeg, [0 3]);
ylim(axLeg, [0.5, numel(allCodes) + 0.5]);
title(axLeg, 'Behavior colors (shared across sessions)', 'FontSize', 12, 'Interpreter', 'none');
hold(axLeg, 'off');

fprintf('Plotted pies for %d/%d sessions (%d unique behavior codes).\n', ...
  nOk, nSess, numel(allCodes));

% Population activity + ethogram — one session per cell, nRows x 2 tiling
hasPop = arrayfun(@(s) ~isempty(s.popActivity) && ~isempty(s.popTime), sessionBhv);
nPop = sum(hasPop);
if nPop < 1
  warning('scratch:NoPopActivity', 'No sessions with popActivity to plot.');
else
  popIdx = find(hasPop);

  % Shared code→color map for ethograms (same colors as pies)
  codeColorMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
  for iCode = 1:numel(allCodes)
    codeColorMap(allCodes(iCode)) = codeColors(iCode, :);
  end

  nCols = 2;
  nRows = ceil(nPop / nCols);
  % Each session uses two subplot rows (popActivity above ethogram)
  nSubRows = nRows * 2;

  figPop = figure('Color', 'w', 'Name', 'Spontaneous popActivity + ethogram');
  position_scratch_figure_full_monitor(figPop);

  for iPlot = 1:nPop
    iSess = popIdx(iPlot);
    row = ceil(iPlot / nCols);
    col = mod(iPlot - 1, nCols) + 1;
    popSubRow = (row - 1) * 2 + 1;
    ethSubRow = (row - 1) * 2 + 2;
    popAxIdx = (popSubRow - 1) * nCols + col;
    ethAxIdx = (ethSubRow - 1) * nCols + col;

    tSess = sessionBhv(iSess).popTime;
    tMaxSess = max(tSess);
    if isfinite(sessionBhv(iSess).durationSec)
      tMaxSess = max(tMaxSess, sessionBhv(iSess).durationSec);
    end

    axPop = subplot(nSubRows, nCols, popAxIdx, 'Parent', figPop);
    plot(axPop, tSess, sessionBhv(iSess).popActivity, ...
      'Color', [0.15 0.15 0.15], 'LineWidth', 0.6);
    xlim(axPop, [collectStart, tMaxSess]);
    ylabel(axPop, 'pop', 'FontSize', 9);
    title(axPop, sprintf('%s (%.1f min)', sessionBhv(iSess).label, ...
      sessionBhv(iSess).durationSec / 60), 'Interpreter', 'none', 'FontSize', 10);
    box(axPop, 'off');
    set(axPop, 'XTickLabel', []);

    axEth = subplot(nSubRows, nCols, ethAxIdx, 'Parent', figPop);
    scratch_plot_behavior_ethogram(axEth, sessionBhv(iSess), codeColorMap, ...
      collectStart, tMaxSess);
    xlabel(axEth, 'Time (s)', 'FontSize', 9);
    linkaxes([axPop, axEth], 'x');
  end

  sgtitle(figPop, sprintf( ...
    'Population activity + ethogram (%s, bin=%.0f ms) — %d sessions (%d x %d)', ...
    brainAreaPop, binSizePop * 1000, nPop, nRows, nCols), ...
    'FontWeight', 'bold', 'Interpreter', 'none');
  fprintf('Plotted popActivity + ethogram for %d/%d sessions (%d x %d tiles).\n', ...
    nPop, nSess, nRows, nCols);
end

%% Semicircle: d2 distributions + popActivity + TaskMatrix ethogram
%
% Runs session_d2_distributions for semicircle session(s). Produces:
%   1) d2 vs shuffled density
%   2) d2 vs mean popActivity scatter
%   3) timeline: mean pop | d2 | TaskMatrix ethogram
%
% Set runAllSemicircleSessions=true to loop semicircle_session_list(); otherwise
% uses the single subjectName/sessionName below.

setup_criticality_manuscript_paths('session_d2_distributions');
paths = get_paths();

runAllSemicircleSessions = true;
% Single-session override when runAllSemicircleSessions = false:
subjectName = 'AS1';
sessionName = 'AS1_0618_WellLearned';
% sessionName = 'AS1_0623_TransitionAfterCompletedTrial_80';
% sessionName = 'AS1_0624_PoorlyLearned';

if runAllSemicircleSessions
  semiSessions = semicircle_session_list();
else
  semiSessions = struct('subjectName', subjectName, 'sessionName', sessionName);
end

fprintf('\n=== Semicircle d2 / popActivity / ethogram (%d session(s)) ===\n', ...
  numel(semiSessions));

for iSemi = 1:numel(semiSessions)
  sessionType = 'semicircle';
  subjectName = semiSessions(iSemi).subjectName;
  sessionName = semiSessions(iSemi).sessionName;
  dataSource = 'spikes';

  collectStart = 0;
  collectEnd = [];
  d2Window = 1 * 60;
  brainArea = 'M23M56';
  brainAreaCombinations = default_manuscript_brain_area_combinations();
  useLog10D2 = true;
  useSubsampling = false;
  nSubsamples = 20;
  nNeuronsSubsample = 40;
  minNeuronsMultiple = 1.1;
  nPermutations = 5;
  plotD2PopActivity = true;
  plotD2Timeline = true;
  useRelativeTime = false;
  binSize = 0.03;
  saveFigure = false;
  plotConfig = fill_manuscript_plot_config();
  splitExcitatoryInhibitory = false;
  widthCutoff = 0.35;

  fprintf('\n##### Semicircle %d/%d: %s / %s #####\n', ...
    iSemi, numel(semiSessions), subjectName, sessionName);

  try
    session_d2_distributions;
  catch ME
    warning('scratch:SemicircleD2Failed', ...
      'Failed for %s / %s: %s', subjectName, sessionName, ME.message);
    for st = 1:min(5, numel(ME.stack))
      fprintf('  %s (line %d)\n', ME.stack(st).name, ME.stack(st).line);
    end
  end
end
clear runAllSemicircleSessions semiSessions iSemi
fprintf('\n=== Semicircle d2 / popActivity / ethogram: done ===\n');

%% Batch: d2 vs windowSize — one plot per reach session
%
% Keeps the default spontaneous / interval examples fixed and swaps the reach
% session through reach_session_list(), running criticality_d2_vs_windowSize
% and saving a uniquely named PNG for each.

setup_criticality_manuscript_paths('criticality_d2_vs_windowSize');

reachSessions = reach_session_list();
nReach = numel(reachSessions);

spontaneousEx = struct( ...
  'sessionType', 'spontaneous', ...
  'subjectName', 'ag25290', ...
  'sessionName', 'ag112321_1', ...
  'displayLabel', 'spontaneous');
intervalEx = struct( ...
  'sessionType', 'interval', ...
  'subjectName', 'ey9166', ...
  'sessionName', 'ey9166_2026_04_03', ...
  'displayLabel', 'interval');

closeFigure = true;   % avoid stacking figures across the batch

fprintf('\n=== Batch d2 vs windowSize across %d reach sessions ===\n', nReach);
for iReach = 1:nReach
  reachName = reachSessions{iReach};
  fprintf('\n##### Reach %d/%d: %s #####\n', iReach, nReach, reachName);

  exampleSessions = spontaneousEx;
  exampleSessions(2) = intervalEx;
  exampleSessions(3) = struct( ...
    'sessionType', 'reach', ...
    'subjectName', '', ...
    'sessionName', reachName, ...
    'displayLabel', 'reach');
  figureTag = reachName;

  try
    criticality_d2_vs_windowSize;
  catch ME
    warning('scratch:D2VsWindowSizeBatchFailed', ...
      'Failed for reach session %s: %s', reachName, ME.message);
  end

  % Clear so the next iteration can re-set; also drop large intermediates
  clear exampleSessions figureTag results exampleResults
end
clear closeFigure spontaneousEx intervalEx reachSessions nReach iReach reachName
fprintf('\n=== Batch d2 vs windowSize: done ===\n');

%% Scratch: PRG across tasks — sweep finalCutoffDivisor x surrogateMethod
%
% Nested loops call criticality_prg_across_tasks for each combo, then build a
% 2x2 summary across cutoffs (kappa / D_JS for isi and circular surrogates).

setup_criticality_manuscript_paths('criticality_prg_across_tasks');
paths = get_paths();

finalCutoffDivisors = [4, 8, 16, 32];
surrogateMethods = {'isi', 'circular'};

baseOpts = criticality_prg_across_tasks();
baseOpts.runBatch = true;
baseOpts.plotResults = false;
baseOpts.saveBatchResults = false;
baseOpts.cutoffDivisors = unique([baseOpts.cutoffDivisors, finalCutoffDivisors], 'stable');

nSurr = numel(surrogateMethods);
nCut = numel(finalCutoffDivisors);
sweepOut = cell(nSurr, nCut);

for iSurr = 1:nSurr
  for iCut = 1:nCut
    opts = baseOpts;
    opts.surrogateMethod = surrogateMethods{iSurr};
    opts.finalCutoffDivisor = finalCutoffDivisors(iCut);
    opts.batchResultsFile = fullfile(paths.dropPath, 'criticality_manuscript', ...
      sprintf('criticality_prg_across_tasks_batch_N%d_%s.mat', ...
      opts.finalCutoffDivisor, opts.surrogateMethod));

    fprintf('\n##### Sweep %d/%d: N/%d, surrogate=%s #####\n', ...
      (iSurr - 1) * nCut + iCut, nSurr * nCut, ...
      opts.finalCutoffDivisor, opts.surrogateMethod);

    sweepOut{iSurr, iCut} = criticality_prg_across_tasks(opts);
  end
end

% 2x2 summary: kappa / D_JS vs finalCutoffDivisor for each surrogate method
areaName = resolve_sweep_summary_area(sweepOut, baseOpts);
plotConfig = fill_manuscript_plot_config();
saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end

fig = figure('Color', 'w', 'Name', 'PRG cutoff x surrogate summary');
position_scratch_figure_full_monitor(fig);
tileLayout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% (1,1) kappa isi | (1,2) kappa circular
% (2,1) Djs isi   | (2,2) Djs circular
panelSpecs = { ...
  'isi', 'kappaMean', '\kappa (isi)'
  'circular', 'kappaMean', '\kappa (circular)'
  'isi', 'djsMean', 'D_{JS} (isi)'
  'circular', 'djsMean', 'D_{JS} (circular)'
  };

for iPanel = 1:size(panelSpecs, 1)
  ax = nexttile(tileLayout);
  hold(ax, 'on');
  surrName = panelSpecs{iPanel, 1};
  metricField = panelSpecs{iPanel, 2};
  yLabelText = panelSpecs{iPanel, 3};
  iSurr = find(strcmp(surrogateMethods, surrName), 1);

  plot_prg_cutoff_sweep_panel(ax, sweepOut(iSurr, :), finalCutoffDivisors, ...
    areaName, metricField, plotConfig);

  xlabel(ax, 'finalCutoffDivisor (N / divisor)', 'FontSize', plotConfig.axisLabelFontSize);
  ylabel(ax, yLabelText, 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', 'tex');
  title(ax, sprintf('%s — %s', areaName, yLabelText), ...
    'FontSize', plotConfig.titleFontSize, 'Interpreter', 'tex');
  set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
    'Box', 'off', 'TickDir', 'out', 'XTick', finalCutoffDivisors);
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayout, sprintf('PRG across cutoffs — %s', areaName), ...
  'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

plotBase = sprintf('criticality_prg_cutoff_surrogate_summary_%s_%s', ...
  char(baseOpts.prgMethod), matlab.lang.makeValidName(areaName));
exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
exportgraphics(fig, fullfile(saveDir, [plotBase, '.eps']), 'ContentType', 'vector');
fprintf('\nSaved summary figure: %s\n', fullfile(saveDir, plotBase));

%% 2x2 summary: sessions on x-axis, N values as markers (within-task lines)
nMarkers = {'o', 's', 'd', '^'};
if numel(nMarkers) < nCut
  error('scratch:TooManyCutoffs', 'Add more marker styles for %d cutoffs.', nCut);
end
nMarkers = nMarkers(1:nCut);

figSess = figure('Color', 'w', 'Name', 'PRG sessions x N summary');
position_scratch_figure_full_monitor(figSess);
tileLayoutSess = tiledlayout(figSess, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for iPanel = 1:size(panelSpecs, 1)
  ax = nexttile(tileLayoutSess);
  hold(ax, 'on');
  surrName = panelSpecs{iPanel, 1};
  metricField = panelSpecs{iPanel, 2};
  yLabelText = panelSpecs{iPanel, 3};
  iSurr = find(strcmp(surrogateMethods, surrName), 1);

  plot_prg_session_by_n_panel(ax, sweepOut(iSurr, :), finalCutoffDivisors, ...
    nMarkers, areaName, metricField, plotConfig);

  xlabel(ax, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
  ylabel(ax, yLabelText, 'FontSize', plotConfig.axisLabelFontSize, 'Interpreter', 'tex');
  title(ax, sprintf('%s — %s', areaName, yLabelText), ...
    'FontSize', plotConfig.titleFontSize, 'Interpreter', 'tex');
  set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
    'Box', 'off', 'TickDir', 'out');
  grid(ax, 'on');
  hold(ax, 'off');
end

sgtitle(tileLayoutSess, sprintf('PRG across sessions by N — %s', areaName), ...
  'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold');

plotBaseSess = sprintf('criticality_prg_session_n_summary_%s_%s', ...
  char(baseOpts.prgMethod), matlab.lang.makeValidName(areaName));
exportgraphics(figSess, fullfile(saveDir, [plotBaseSess, '.png']), 'Resolution', 300);
exportgraphics(figSess, fullfile(saveDir, [plotBaseSess, '.eps']), 'ContentType', 'vector');
fprintf('Saved session-by-N summary figure: %s\n', fullfile(saveDir, plotBaseSess));

%% Neuron counts across task sessions (load-option dependent)
%
% Loops the same session lists / loading options as
% criticality_multiple_metrics_across_tasks, prints unit counts for brainArea
% (and component areas when brainArea is a compound merge), then plots counts
% with the multimetric across-tasks session layout.

setup_criticality_manuscript_paths('criticality_multiple_metrics_across_tasks');
paths = get_paths();
plotConfig = fill_manuscript_plot_config();

sessionTypes = default_manuscript_session_types();
sessionTypes = {'spontaneous'};
collectStart = 0;
collectEnd = [];  % [] = full session (matches multiple_metrics)
collectEnd = 120 * 60;
dataSource = 'spikes';
brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

% Loading filters — same defaults as criticality_ar_across_tasks when
% multiple_metrics does not override them
firingRateCheckTime = [];  % [] = check rate over the loaded collect window
minFiringRate = 0.1;
maxFiringRate = 200;

fprintf('\n=== Neuron counts across tasks ===\n');
fprintf('Session types: %s\n', strjoin(sessionTypes, ', '));
if isempty(collectEnd)
  fprintf('Collect window: [%.1f, full] s\n', collectStart);
else
  fprintf('Collect window: [%.1f, %.1f] s\n', collectStart, collectEnd);
end
fprintf('Brain area: %s\n', brainArea);
fprintf('Firing-rate filter: [%.3g, %.3g] Hz (checkTime=%s)\n', ...
  minFiringRate, maxFiringRate, mat2str(firingRateCheckTime));

combo = scratch_lookup_brain_area_combo(brainArea, brainAreaCombinations);
componentAreas = {};
if ~isempty(combo)
  componentAreas = cellstr(combo.areas(:));
  fprintf('Compound area %s = %s\n', brainArea, strjoin(componentAreas, '+'));
end

sessionTable = scratch_build_session_table(sessionTypes);
numSessions = size(sessionTable, 1);
fprintf('Total sessions: %d\n', numSessions);

nNeuronsTarget = nan(numSessions, 1);
nNeuronsByComponent = nan(numSessions, max(numel(componentAreas), 1));
sessionOk = false(numSessions, 1);
skipReason = repmat({''}, numSessions, 1);

loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = firingRateCheckTime;
loadOpts.collectStart = collectStart;
loadOpts.collectEnd = collectEnd;
loadOpts.minFiringRate = minFiringRate;
loadOpts.maxFiringRate = maxFiringRate;

for iSess = 1:numSessions
  sessionType = sessionTable.sessionType{iSess};
  sessionName = sessionTable.sessionName{iSess};
  subjectName = sessionTable.subjectName{iSess};

  fprintf('\n%s\n', repmat('-', 1, 72));
  fprintf('Session %d/%d [%s]: %s\n', iSess, numSessions, sessionType, sessionName);

  try
    loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
    dataStruct = load_session_data(sessionType, dataSource, loadArgs{:});
    [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
      dataStruct, brainArea, brainAreaCombinations);
    if ~areaOk
      skipReason{iSess} = sprintf('Brain area "%s" not available', brainArea);
      fprintf('  Skip: %s\n', skipReason{iSess});
      continue;
    end

    nTarget = scratch_count_neurons_in_area(dataStruct, brainArea);
    nNeuronsTarget(iSess) = nTarget;
    fprintf('  %s: %d neurons\n', brainArea, nTarget);

    for iComp = 1:numel(componentAreas)
      nComp = scratch_count_neurons_in_area(dataStruct, componentAreas{iComp});
      nNeuronsByComponent(iSess, iComp) = nComp;
      fprintf('    %s: %d neurons\n', componentAreas{iComp}, nComp);
    end

    sessionOk(iSess) = true;
  catch ME
    skipReason{iSess} = ME.message;
    warning('scratch:NeuronCountFailed', 'Failed (%s / %s): %s', ...
      sessionType, sessionName, ME.message);
  end
end

nOk = sum(sessionOk);
fprintf('\n=== Neuron counts done: %d/%d sessions OK ===\n', nOk, numSessions);

countTable = sessionTable;
countTable.nNeurons = nNeuronsTarget;
countTable.success = sessionOk;
countTable.skipReason = skipReason;
for iComp = 1:numel(componentAreas)
  countTable.(matlab.lang.makeValidName(['n_', componentAreas{iComp}])) = ...
    nNeuronsByComponent(:, iComp);
end

plot_neuron_counts_across_tasks(countTable, sessionTypes, brainArea, ...
  componentAreas, collectStart, collectEnd, minFiringRate, maxFiringRate, ...
  paths, plotConfig);

%% Local helpers
function areaName = resolve_sweep_summary_area(sweepOut, baseOpts)
% RESOLVE_SWEEP_SUMMARY_AREA - Prefer opts.brainArea, else first available area

if ~isempty(baseOpts.brainArea)
  areaName = char(baseOpts.brainArea);
  return;
end
for i = 1:numel(sweepOut)
  if isempty(sweepOut{i}) || ~isfield(sweepOut{i}, 'areasToPlot')
    continue;
  end
  if ~isempty(sweepOut{i}.areasToPlot)
    areaName = sweepOut{i}.areasToPlot{1};
    return;
  end
end
error('scratch:NoArea', 'No areas available in sweep results.');
end

function plot_prg_cutoff_sweep_panel(ax, plotDataByCutoff, finalCutoffDivisors, ...
    areaName, metricField, plotConfig)
% PLOT_PRG_CUTOFF_SWEEP_PANEL - One line per session across cutoffs, colored by task
%
% Variables:
%   ax                  - Target axes
%   plotDataByCutoff    - 1 x nCutoff cell of criticality_prg_across_tasks outputs
%   finalCutoffDivisors - Cutoff values (x-axis)
%   areaName            - Area to extract
%   metricField         - 'kappaMean' or 'djsMean'
%   plotConfig          - Manuscript plot styling

nCut = numel(finalCutoffDivisors);
sessionTypes = {};
for iCut = 1:nCut
  if isempty(plotDataByCutoff{iCut}) || ~isfield(plotDataByCutoff{iCut}, 'plotData')
    continue;
  end
  sessionTypes = plotDataByCutoff{iCut}.plotData.sessionTypes;
  break;
end
if isempty(sessionTypes)
  title(ax, 'No data');
  return;
end

legendHandles = gobjects(0);
legendLabels = {};

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  taskColor = colors_for_tasks(sessionType);

  nSessions = [];
  for iCut = 1:nCut
    nSessions = max_session_count(nSessions, plotDataByCutoff{iCut}, typeKey, areaName, metricField);
  end
  if isempty(nSessions) || nSessions < 1
    continue;
  end

  for iSess = 1:nSessions
    xVals = nan(1, nCut);
    yVals = nan(1, nCut);
    for iCut = 1:nCut
      yVal = get_session_metric(plotDataByCutoff{iCut}, typeKey, areaName, metricField, iSess);
      if isfinite(yVal)
        xVals(iCut) = finalCutoffDivisors(iCut);
        yVals(iCut) = yVal;
      end
    end
    valid = isfinite(xVals) & isfinite(yVals);
    if ~any(valid)
      continue;
    end
    hLine = plot(ax, xVals(valid), yVals(valid), '-o', ...
      'Color', taskColor, ...
      'MarkerFaceColor', taskColor, ...
      'MarkerSize', plotConfig.markerSize, ...
      'LineWidth', plotConfig.lineWidth, ...
      'HandleVisibility', 'off');
    if iSess == 1
      set(hLine, 'HandleVisibility', 'on', 'DisplayName', sessionType);
      legendHandles(end + 1) = hLine; %#ok<AGROW>
      legendLabels{end + 1} = sessionType; %#ok<AGROW>
    end
  end
end

if ~isempty(legendHandles)
  legend(ax, legendHandles, legendLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize);
end
end

function plot_prg_session_by_n_panel(ax, plotDataByCutoff, finalCutoffDivisors, ...
    nMarkers, areaName, metricField, plotConfig)
% PLOT_PRG_SESSION_BY_N_PANEL - Sessions on x; N values as markers with task lines
%
% Variables:
%   ax                  - Target axes
%   plotDataByCutoff    - 1 x nCutoff cell of criticality_prg_across_tasks outputs
%   finalCutoffDivisors - N/divisor values (one series each)
%   nMarkers            - Marker styles, one per cutoff
%   areaName            - Area to extract
%   metricField         - 'kappaMean' or 'djsMean'
%   plotConfig          - Manuscript plot styling
%
% Goal:
%   Match multimetric across-tasks layout: sessions along x, grouped by task,
%   with a within-task connecting line per N value.

nCut = numel(finalCutoffDivisors);
sessionTypes = {};
for iCut = 1:nCut
  if isempty(plotDataByCutoff{iCut}) || ~isfield(plotDataByCutoff{iCut}, 'plotData')
    continue;
  end
  sessionTypes = plotDataByCutoff{iCut}.plotData.sessionTypes;
  break;
end
if isempty(sessionTypes)
  title(ax, 'No data');
  return;
end

xCursor = 0;
xticksCenters = [];
xtickLabels = {};
legendHandles = gobjects(0);
legendLabels = {};

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  typeKey = matlab.lang.makeValidName(sessionType);
  taskColor = colors_for_tasks(sessionType);
  lineColor = 0.55 * taskColor + 0.45 * [1 1 1];

  nSessions = [];
  for iCut = 1:nCut
    nSessions = max_session_count(nSessions, plotDataByCutoff{iCut}, typeKey, areaName, metricField);
  end
  if isempty(nSessions) || nSessions < 1
    continue;
  end

  xPos = xCursor + (1:nSessions);
  sessLabels = get_session_xtick_labels(plotDataByCutoff, typeKey, nSessions, sessionType);

  for iCut = 1:nCut
    yVals = nan(1, nSessions);
    for iSess = 1:nSessions
      yVals(iSess) = get_session_metric(plotDataByCutoff{iCut}, typeKey, areaName, ...
        metricField, iSess);
    end
    valid = isfinite(xPos) & isfinite(yVals);
    if sum(valid) >= 2
      plot(ax, xPos(valid), yVals(valid), '-', ...
        'Color', lineColor, ...
        'LineWidth', max(1, plotConfig.lineWidth - 0.25), ...
        'HandleVisibility', 'off');
    end
    if any(valid)
      hMark = plot(ax, xPos(valid), yVals(valid), nMarkers{iCut}, ...
        'Color', taskColor, ...
        'MarkerFaceColor', taskColor, ...
        'MarkerSize', plotConfig.markerSize, ...
        'LineWidth', plotConfig.lineWidth, ...
        'HandleVisibility', 'off');
      if t == 1
        set(hMark, 'HandleVisibility', 'on', ...
          'DisplayName', sprintf('N/%d', finalCutoffDivisors(iCut)));
        legendHandles(end + 1) = hMark; %#ok<AGROW>
        legendLabels{end + 1} = sprintf('N/%d', finalCutoffDivisors(iCut)); %#ok<AGROW>
      end
    end
  end

  for iSess = 1:nSessions
    xticksCenters(end + 1) = xPos(iSess); %#ok<AGROW>
    xtickLabels{end + 1} = sessLabels{iSess}; %#ok<AGROW>
  end
  xCursor = xPos(end) + 1.5;
end

if ~isempty(xticksCenters)
  xlim(ax, [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8]);
  set(ax, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45);
end
if ~isempty(legendHandles)
  legend(ax, legendHandles, legendLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize);
end
end

function sessLabels = get_session_xtick_labels(plotDataByCutoff, typeKey, nSessions, sessionType)
% GET_SESSION_XTICK_LABELS - Prefer sessionNames/labels from first available cutoff run
%
% Tick text uses up to the first 8 characters of sessionName.

sessLabels = cell(1, nSessions);
for iSess = 1:nSessions
  sessLabels{iSess} = sprintf('%s-%d', sessionType, iSess);
end
for iCut = 1:numel(plotDataByCutoff)
  if isempty(plotDataByCutoff{iCut}) || ~isfield(plotDataByCutoff{iCut}, 'plotData')
    continue;
  end
  plotData = plotDataByCutoff{iCut}.plotData;
  if ~isfield(plotData.byType, typeKey)
    continue;
  end
  typeData = plotData.byType.(typeKey);
  if isfield(typeData, 'sessionNames') && numel(typeData.sessionNames) >= nSessions
    for iSess = 1:nSessions
      sessLabels{iSess} = truncate_session_tick_label(typeData.sessionNames{iSess});
    end
    return;
  end
  if isfield(typeData, 'sessionLabels') && numel(typeData.sessionLabels) >= nSessions
    for iSess = 1:nSessions
      sessLabels{iSess} = truncate_session_tick_label(typeData.sessionLabels{iSess});
    end
    return;
  end
end
end

function label = truncate_session_tick_label(sessionName)
% TRUNCATE_SESSION_TICK_LABEL - First 8 characters of sessionName for xticks

label = char(sessionName);
if numel(label) > 8
  label = label(1:8);
end
end

function nSessions = max_session_count(nSessions, runOut, typeKey, areaName, metricField)
% MAX_SESSION_COUNT - Largest session count seen for this type/area/metric

if isempty(runOut) || ~isfield(runOut, 'plotData')
  return;
end
plotData = runOut.plotData;
if ~isfield(plotData.byType, typeKey)
  return;
end
areaIdx = find(strcmp(plotData.areas, areaName), 1);
if isempty(areaIdx)
  return;
end
typeData = plotData.byType.(typeKey);
if ~isfield(typeData, metricField) || areaIdx > numel(typeData.(metricField))
  return;
end
metricVec = typeData.(metricField){areaIdx};
nHere = numel(metricVec);
if isempty(nSessions)
  nSessions = nHere;
else
  nSessions = max(nSessions, nHere);
end
end

function yVal = get_session_metric(runOut, typeKey, areaName, metricField, iSess)
% GET_SESSION_METRIC - Session iSess metric at one cutoff run

yVal = nan;
if isempty(runOut) || ~isfield(runOut, 'plotData')
  return;
end
plotData = runOut.plotData;
if ~isfield(plotData.byType, typeKey)
  return;
end
areaIdx = find(strcmp(plotData.areas, areaName), 1);
if isempty(areaIdx)
  return;
end
typeData = plotData.byType.(typeKey);
if ~isfield(typeData, metricField) || areaIdx > numel(typeData.(metricField))
  return;
end
metricVec = typeData.(metricField){areaIdx};
if iSess > numel(metricVec)
  return;
end
yVal = metricVec(iSess);
end

function position_scratch_figure_full_monitor(fig)
% POSITION_SCRATCH_FIGURE_FULL_MONITOR - Size figure to fill a monitor

monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
  targetPos = monitorPositions(end, :);
else
  targetPos = monitorPositions(1, :);
end
set(fig, 'Units', 'pixels', 'Position', targetPos);
end

function sessionTable = scratch_build_session_table(sessionTypes)
% SCRATCH_BUILD_SESSION_TABLE - Flatten session lists from each session type

sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
labelCol = {};

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  entries = scratch_get_sessions_for_type(sessionType);
  for i = 1:numel(entries)
    sessionTypeCol{end+1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end+1, 1} = entries(i).sessionName; %#ok<AGROW>
    if isfield(entries, 'subjectName')
      subjectNameCol{end+1, 1} = entries(i).subjectName; %#ok<AGROW>
    else
      subjectNameCol{end+1, 1} = ''; %#ok<AGROW>
    end
    labelCol{end+1, 1} = truncate_session_tick_label(entries(i).sessionName); %#ok<AGROW>
  end
end

sessionTable = table(sessionTypeCol, sessionNameCol, subjectNameCol, labelCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'subjectName', 'label'});
end

function entries = scratch_get_sessions_for_type(sessionType)
% SCRATCH_GET_SESSIONS_FOR_TYPE - Struct array with subjectName and sessionName
entries = manuscript_sessions_for_type(sessionType);
end

function combo = scratch_lookup_brain_area_combo(brainArea, combinations)
% SCRATCH_LOOKUP_BRAIN_AREA_COMBO - Match compound-area definition by name

combo = [];
if isempty(brainArea) || isempty(combinations)
  return;
end
brainArea = char(brainArea);
for i = 1:numel(combinations)
  entry = combinations{i};
  if isstruct(entry) && isfield(entry, 'name') && strcmpi(entry.name, brainArea)
    combo = entry;
    return;
  end
end
end

function nUnits = scratch_count_neurons_in_area(dataStruct, areaName)
% SCRATCH_COUNT_NEURONS_IN_AREA - Units in one area after load / FR filter

nUnits = nan;
if ~isfield(dataStruct, 'areas') || ~isfield(dataStruct, 'idMatIdx')
  return;
end
areaIdx = find(strcmp(dataStruct.areas, areaName), 1);
if isempty(areaIdx)
  return;
end
nUnits = numel(dataStruct.idMatIdx{areaIdx});
end

function plot_neuron_counts_across_tasks(countTable, sessionTypes, brainArea, ...
    componentAreas, collectStart, collectEnd, minFiringRate, maxFiringRate, ...
    paths, plotConfig)
% PLOT_NEURON_COUNTS_ACROSS_TASKS - Session-grouped neuron counts (multimetric layout)
%
% Variables:
%   countTable       - Table with sessionType, label, nNeurons, success, optional n_* cols
%   sessionTypes     - Task order on the x-axis
%   brainArea        - Primary / combined area name
%   componentAreas   - Cell of component area names ({} if not compound)
%   collectStart/End - Loaded window (for title)
%   min/maxFiringRate - FR filter (for title)
%   paths, plotConfig - Paths and manuscript styling

if nargin < 10 || isempty(plotConfig)
  plotConfig = fill_manuscript_plot_config();
end

okMask = countTable.success;
if ~any(okMask)
  warning('scratch:NoNeuronCounts', 'No successful sessions to plot.');
  return;
end

% Series to plot: combined first, then components (like d2/tau/alpha markers)
seriesNames = {brainArea};
seriesMarkers = {'o'};
seriesFill = true;
for iComp = 1:numel(componentAreas)
  seriesNames{end+1} = componentAreas{iComp}; %#ok<AGROW>
  markers = {'s', 'd', '^', 'v'};
  seriesMarkers{end+1} = markers{min(iComp, numel(markers))}; %#ok<AGROW>
  seriesFill(end+1) = false; %#ok<AGROW>
end
nSeries = numel(seriesNames);
xOffsets = linspace(-0.15, 0.15, max(nSeries, 1));
if nSeries == 1
  xOffsets = 0;
end

yBySeries = cell(1, nSeries);
yBySeries{1} = countTable.nNeurons;
for iComp = 1:numel(componentAreas)
  fieldName = matlab.lang.makeValidName(['n_', componentAreas{iComp}]);
  if ismember(fieldName, countTable.Properties.VariableNames)
    yBySeries{1 + iComp} = countTable.(fieldName);
  else
    yBySeries{1 + iComp} = nan(height(countTable), 1);
  end
end

fig = figure('Color', 'w', 'Name', sprintf('Neuron counts — %s', brainArea));
position_scratch_figure_full_monitor(fig);
ax = axes(fig);
hold(ax, 'on');

legendHandles = gobjects(0);
legendLabels = {};
xCursor = 0;
xticksCenters = [];
xtickLabels = {};

for t = 1:numel(sessionTypes)
  sessionType = sessionTypes{t};
  rowMask = okMask & strcmp(countTable.sessionType, sessionType);
  if ~any(rowMask)
    continue;
  end
  taskColor = colors_for_tasks(sessionType)';
  rowIdx = find(rowMask);
  nSess = numel(rowIdx);
  xPos = xCursor + (1:nSess);

  for iSeries = 1:nSeries
    yVals = yBySeries{iSeries}(rowIdx);
    xSeries = xPos + xOffsets(iSeries);
    valid = isfinite(xSeries) & isfinite(yVals(:)');
    if sum(valid) >= 2
      plot(ax, xSeries(valid), yVals(valid), '-', ...
        'Color', taskColor, ...
        'LineWidth', max(1, plotConfig.lineWidth - 0.25), ...
        'HandleVisibility', 'off');
    end
    faceColor = taskColor;
    if ~seriesFill(iSeries)
      faceColor = 'none';
    end
    if any(valid)
      hMark = plot(ax, xSeries(valid), yVals(valid), seriesMarkers{iSeries}, ...
        'Color', taskColor, ...
        'MarkerFaceColor', faceColor, ...
        'MarkerSize', plotConfig.markerSize, ...
        'LineWidth', plotConfig.lineWidth, ...
        'HandleVisibility', 'off');
      if t == 1
        set(hMark, 'HandleVisibility', 'on', 'DisplayName', seriesNames{iSeries});
        legendHandles(end+1) = hMark; %#ok<AGROW>
        legendLabels{end+1} = seriesNames{iSeries}; %#ok<AGROW>
      end
    end
  end

  for iSess = 1:nSess
    xticksCenters(end+1) = xPos(iSess); %#ok<AGROW>
    xtickLabels{end+1} = char(countTable.label{rowIdx(iSess)}); %#ok<AGROW>
  end
  xCursor = xPos(end) + 1.5;
end

if isempty(xticksCenters)
  warning('scratch:EmptyNeuronCountPlot', 'No sessions to place on x-axis.');
  close(fig);
  return;
end

xlim(ax, [min(xticksCenters) - 0.8, max(xticksCenters) + 0.8]);
set(ax, 'XTick', xticksCenters, 'XTickLabel', xtickLabels, 'XTickLabelRotation', 45, ...
  'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
  'Box', 'off', 'TickDir', 'out');
ylabel(ax, 'Number of neurons', 'FontSize', plotConfig.axisLabelFontSize);
xlabel(ax, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
if isempty(collectEnd)
  collectTag = sprintf('[%.0f, full] s', collectStart);
else
  collectTag = sprintf('[%.0f, %.0f] s', collectStart, collectEnd);
end
title(ax, sprintf('Neuron counts — %s | collect %s | FR [%.2g, %.2g] Hz', ...
  brainArea, collectTag, minFiringRate, maxFiringRate), ...
  'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
grid(ax, 'on');
if ~isempty(legendHandles)
  legend(ax, legendHandles, legendLabels, 'Location', 'best', ...
    'FontSize', plotConfig.legendFontSize, 'Interpreter', 'none');
end
hold(ax, 'off');

% saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
% if ~exist(saveDir, 'dir')
%   mkdir(saveDir);
% end
% if isempty(collectEnd)
%   winTag = 'full';
% else
%   winTag = sprintf('%.0f-%.0f', collectStart, collectEnd);
% end
% plotBase = sprintf('neuron_counts_across_tasks_%s_win%s_fr%.2g-%.2g', ...
%   matlab.lang.makeValidName(brainArea), winTag, minFiringRate, maxFiringRate);
% exportgraphics(fig, fullfile(saveDir, [plotBase, '.png']), 'Resolution', 300);
% fprintf('Saved neuron-count figure: %s\n', fullfile(saveDir, plotBase));
end

function codeToName = scratch_bhv_code_to_name_map(dataStruct)
% SCRATCH_BHV_CODE_TO_NAME_MAP - Map behavior codes to names from dataBhv

codeToName = containers.Map('KeyType', 'double', 'ValueType', 'char');
if ~isfield(dataStruct, 'dataBhv') || isempty(dataStruct.dataBhv)
  return;
end
dataBhv = dataStruct.dataBhv;
if ~ismember('ID', dataBhv.Properties.VariableNames)
  return;
end
hasName = ismember('Name', dataBhv.Properties.VariableNames);
ids = dataBhv.ID(:);
for i = 1:numel(ids)
  code = double(ids(i));
  if ~isfinite(code) || isKey(codeToName, code)
    continue;
  end
  if hasName
    nameVal = dataBhv.Name(i);
    if iscell(nameVal)
      nameVal = nameVal{1};
    end
    codeToName(code) = char(nameVal);
  else
    codeToName(code) = sprintf('code_%g', code);
  end
end
end

function durationSec = scratch_session_duration_sec(dataStruct, collectStart)
% SCRATCH_SESSION_DURATION_SEC - Loaded collect window length (s)

durationSec = nan;
if nargin < 2 || isempty(collectStart)
  collectStart = 0;
end
if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
    && ~isempty(dataStruct.spikeData.collectEnd)
  startVal = collectStart;
  if isfield(dataStruct.spikeData, 'collectStart') && ~isempty(dataStruct.spikeData.collectStart)
    startVal = dataStruct.spikeData.collectStart;
  end
  durationSec = dataStruct.spikeData.collectEnd - startVal;
  return;
end
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
    && ~isempty(dataStruct.opts.collectEnd)
  startVal = collectStart;
  if isfield(dataStruct.opts, 'collectStart') && ~isempty(dataStruct.opts.collectStart)
    startVal = dataStruct.opts.collectStart;
  end
  durationSec = dataStruct.opts.collectEnd - startVal;
  return;
end
if isfield(dataStruct, 'spikeTimes') && ~isempty(dataStruct.spikeTimes)
  durationSec = max(dataStruct.spikeTimes) - collectStart;
end
end

function [popActivity, popTime] = scratch_session_pop_activity(dataStruct, brainArea, ...
    brainAreaCombinations, binSize, collectStart)
% SCRATCH_SESSION_POP_ACTIVITY - Binned population spike count for one area
%
% Variables:
%   dataStruct              - Loaded session
%   brainArea               - Area or compound name (e.g. M23M56)
%   brainAreaCombinations   - Manuscript area merges
%   binSize                 - Bin width (s)
%   collectStart            - Analysis start (s)
%
% Returns:
%   popActivity - Sum of spikes across neurons per bin
%   popTime     - Bin-center times (s)

popActivity = [];
popTime = [];
if nargin < 5 || isempty(collectStart)
  collectStart = 0;
end
if nargin < 4 || isempty(binSize)
  binSize = 0.05;
end

[dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
  dataStruct, brainArea, brainAreaCombinations);
if ~areaOk
  return;
end
areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx) && isfield(dataStruct, 'areasToTest') && ~isempty(dataStruct.areasToTest)
  areaIdx = dataStruct.areasToTest(1);
end
if isempty(areaIdx)
  return;
end
if ~isfield(dataStruct, 'idLabel') || areaIdx > numel(dataStruct.idLabel) ...
    || isempty(dataStruct.idLabel{areaIdx})
  return;
end

durationSec = scratch_session_duration_sec(dataStruct, collectStart);
if ~isfinite(durationSec) || durationSec <= 0
  return;
end
timeRange = [collectStart, collectStart + durationSec];
neuronIDs = dataStruct.idLabel{areaIdx};
aDataMat = bin_spikes(dataStruct.spikeTimes, dataStruct.spikeClusters, ...
  neuronIDs, timeRange, binSize);
if isempty(aDataMat)
  return;
end
popActivity = sum(aDataMat, 2);
popTime = collectStart + ((0:numel(popActivity)-1)' + 0.5) * binSize;
end

function scratch_plot_behavior_ethogram(ax, sessionRec, codeColorMap, tMin, tMax)
% SCRATCH_PLOT_BEHAVIOR_ETHOGRAM - Colored behavior runs aligned to session time
%
% Variables:
%   ax           - Target axes (thin strip under popActivity)
%   sessionRec   - Session struct with .bhvID and .fsBhv
%   codeColorMap - containers.Map code → RGB (shared across sessions)
%   tMin, tMax   - Shared x-limits (s)

hold(ax, 'on');
bhvID = sessionRec.bhvID;
fsBhv = sessionRec.fsBhv;
if isempty(bhvID) || ~(isfinite(fsBhv) && fsBhv > 0)
  text(ax, mean([tMin tMax]), 0.5, 'no behavior labels', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.5 0.5 0.5]);
  xlim(ax, [tMin, tMax]);
  ylim(ax, [0 1]);
  set(ax, 'YTick', [], 'Box', 'off');
  hold(ax, 'off');
  return;
end

bhvID = bhvID(:);
nFrame = numel(bhvID);
% bhvID is relative to the loaded collect window; align to shared time axis
frameStarts = tMin + ((0:nFrame-1)' ) / fsBhv;
frameEnds = tMin + (1:nFrame)' / fsBhv;

% Merge contiguous identical labels into runs
runCode = bhvID(1);
runStart = frameStarts(1);
for i = 2:nFrame
  if bhvID(i) ~= runCode
    scratch_fill_ethogram_run(ax, runStart, frameStarts(i), runCode, codeColorMap);
    runCode = bhvID(i);
    runStart = frameStarts(i);
  end
end
scratch_fill_ethogram_run(ax, runStart, frameEnds(end), runCode, codeColorMap);

xlim(ax, [tMin, tMax]);
ylim(ax, [0 1]);
ylabel(ax, 'bhv', 'FontSize', 9);
set(ax, 'YTick', [], 'Box', 'off', 'TickDir', 'out');
hold(ax, 'off');
end

function scratch_fill_ethogram_run(ax, tStart, tEnd, code, codeColorMap)
% SCRATCH_FILL_ETHOGRAM_RUN - One colored rectangle for a behavior bout

if ~(isfinite(tStart) && isfinite(tEnd)) || tEnd <= tStart
  return;
end
if isKey(codeColorMap, double(code))
  faceColor = codeColorMap(double(code));
else
  c = colors_for_behaviors(double(code));
  if size(c, 1) == 1 && size(c, 2) == 3
    faceColor = c;
  else
    faceColor = [0.7 0.7 0.7];
  end
end
fill(ax, [tStart, tEnd, tEnd, tStart], [0, 0, 1, 1], faceColor, ...
  'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function [codeColors, codeNames] = scratch_behavior_code_colors_and_names(allCodes, sessionBhvOk)
% SCRATCH_BEHAVIOR_CODE_COLORS_AND_NAMES - Shared colors/names across sessions
%
% Prefer colors_for_behaviors for known B-SOiD codes; fill any missing codes
% with distinguishable colors so every label still has a stable color.

allCodes = allCodes(:);
nCode = numel(allCodes);
codeColors = nan(nCode, 3);
codeNames = cell(nCode, 1);

for iCode = 1:nCode
  code = allCodes(iCode);
  codeNames{iCode} = sprintf('code_%g', code);
  for iSess = 1:numel(sessionBhvOk)
    map = sessionBhvOk(iSess).codeToName;
    if isa(map, 'containers.Map') && isKey(map, code)
      codeNames{iCode} = map(code);
      break;
    end
  end

  c = colors_for_behaviors(code);
  if size(c, 1) == 1 && size(c, 2) == 3
    codeColors(iCode, :) = c;
  end
end

missing = any(~isfinite(codeColors), 2);
if any(missing)
  nMissing = sum(missing);
  if exist('distinguishable_colors', 'file') == 2
    fillColors = distinguishable_colors(nMissing, [1 1 1]);
  else
    fillColors = lines(nMissing);
  end
  codeColors(missing, :) = fillColors;
end
end
