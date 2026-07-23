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
