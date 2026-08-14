function out = semicircle_reward_session_performance(subjectName, sessionName, opts)
% SEMICIRCLE_REWARD_SESSION_PERFORMANCE - Behavioral summary for one semicircle session
%
% Variables:
%   subjectName - Subject folder (e.g. 'AS1')
%   sessionName - Session .mat basename (e.g. 'AS1_0618_WellLearned')
%   opts        - Options struct. Fields:
%       .collectStart, .collectEnd - Analysis window (s); empty collectEnd = session end
%       .minNonEngagedWindow - Min gap for non-engaged segments (default 30)
%       .eventBufferBefore   - Buffer before each beam break (s; default 1)
%       .eventBufferAfter    - Buffer after each beam break (s; default 1)
%       .absorbSingleEvents  - Merge isolated events into non-engaged (default true)
%       .movAvgWinSec        - Moving-average window for rates/accuracy (default 30)
%       .rateBinSec          - Time bin for rate/accuracy traces (default 1)
%       .ieiHistBinSec       - Bin width for beam-break interval histogram (default 1)
%       .ieiHistMaxSec       - Histogram x-limit (s); [] = 99th percentile of IEIs
%       .makePlots           - Create figure (default true)
%       .saveFigure          - Save PNG/EPS (default false)
%       .outputDir           - Save directory (default dropPath/semicircle_reward_task/results)
%
% Goal:
%   Load rewarded / unrewarded choice-port beam breaks (TaskMatrix), shade
%   engaged/non-engaged segments (same definition as
%   semicircle_criticality_metrics_engagement), mark correct (green) and error
%   (red) events, plot running reward rate and accuracy, and histogram
%   inter-beam-break intervals to help choose minNonEngagedWindow when the
%   distribution is bimodal.
%
% Returns:
%   With no inputs: default options struct.
%   Otherwise: struct with events, segments, accuracy, beamBreakIntervals, figHandle.

setup_semicircle_reward_session_performance_paths();

if nargin == 0
  out = fill_semicircle_reward_session_performance_opts(struct());
  return;
end
if nargin < 2 || isempty(subjectName) || isempty(sessionName)
  error('semicircle_reward_session_performance:MissingSession', ...
    'subjectName and sessionName are required.');
end
if nargin < 3 || isempty(opts)
  opts = struct();
end
opts = fill_semicircle_reward_session_performance_opts(opts);

paths = get_paths();
dataFile = fullfile(paths.semicircleDataPath, subjectName, [sessionName, '.mat']);
if ~isfile(dataFile)
  error('semicircle_reward_session_performance:FileNotFound', ...
    'Semicircle data file not found: %s', dataFile);
end

fprintf('\n=== Semicircle reward session performance ===\n');
fprintf('Session: %s / %s\n', subjectName, sessionName);
fprintf('Loading: %s\n', dataFile);

dataS = load(dataFile, 'TaskMatrix');
if ~isfield(dataS, 'TaskMatrix') || isempty(dataS.TaskMatrix)
  error('semicircle_reward_session_performance:NoTaskMatrix', ...
    'TaskMatrix missing in %s', dataFile);
end
sessionEnd = max(dataS.TaskMatrix(:, 8));

[eventTimesAll, eventTypesAll, trials] = load_semicircle_beam_break_events( ...
  paths, subjectName, sessionName);

collectStart = opts.collectStart;
collectEnd = opts.collectEnd;
if isempty(collectEnd)
  collectEnd = sessionEnd;
end
if isempty(collectStart)
  collectStart = 0;
end

eventInCollect = eventTimesAll >= collectStart & eventTimesAll <= collectEnd;
eventTimes = eventTimesAll(eventInCollect);
eventTypes = eventTypesAll(eventInCollect);

fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
  collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('Beam breaks in collect window: %d\n', numel(eventTimes));

[engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
  collectStart, collectEnd, eventTimes, opts.minNonEngagedWindow, ...
  opts.eventBufferBefore, opts.eventBufferAfter, opts.absorbSingleEvents);

isCorrect = eventTypes == "correct";
isError = eventTypes == "error";
accuracyStats = summarize_semicircle_accuracy(isCorrect, isError);
print_semicircle_accuracy(accuracyStats);

beamBreakIntervals = diff(sort(eventTimes(:)));
print_beam_break_interval_summary(beamBreakIntervals, opts.minNonEngagedWindow);

out = struct();
out.subjectName = subjectName;
out.sessionName = sessionName;
out.config = opts;
out.collectStart = collectStart;
out.collectEnd = collectEnd;
out.eventTimes = eventTimes;
out.eventTypes = eventTypes;
out.trials = trials;
out.isCorrect = isCorrect;
out.isError = isError;
out.segments = struct('engaged', engagedSegs, 'nonEngaged', nonEngagedSegs);
out.accuracyStats = accuracyStats;
out.beamBreakIntervals = beamBreakIntervals;
out.figHandle = gobjects(0);

if opts.makePlots
  out.figHandle = plot_semicircle_reward_session_performance( ...
    eventTimes, isCorrect, isError, engagedSegs, nonEngagedSegs, ...
    beamBreakIntervals, subjectName, sessionName, collectStart, collectEnd, opts);
  if opts.saveFigure && isgraphics(out.figHandle)
    saveDir = opts.outputDir;
    if isempty(saveDir)
      saveDir = fullfile(paths.dropPath, 'semicircle_reward_task', 'results', ...
        matlab.lang.makeValidName(subjectName), ...
        matlab.lang.makeValidName(sessionName));
    end
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    plotBase = fullfile(saveDir, sprintf('semicircle_reward_session_performance_%s', ...
      matlab.lang.makeValidName(sessionName)));
    exportgraphics(out.figHandle, [plotBase, '.png'], 'Resolution', 300);
    exportgraphics(out.figHandle, [plotBase, '.eps'], 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', plotBase);
  end
end
end

%% -------------------------------------------------------------------------
%% Defaults and paths
%% -------------------------------------------------------------------------

function opts = fill_semicircle_reward_session_performance_opts(opts)
% FILL_SEMICIRCLE_REWARD_SESSION_PERFORMANCE_OPTS - Default behavioral summary options

if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
  opts.collectStart = 0;
end
if ~isfield(opts, 'collectEnd')
  opts.collectEnd = [];
end
if ~isfield(opts, 'minNonEngagedWindow') || isempty(opts.minNonEngagedWindow)
  opts.minNonEngagedWindow = 30;
end
% Prefer eventBufferBefore/After; accept reachBuffer* aliases used elsewhere
if (~isfield(opts, 'eventBufferBefore') || isempty(opts.eventBufferBefore)) ...
    && isfield(opts, 'reachBufferBefore') && ~isempty(opts.reachBufferBefore)
  opts.eventBufferBefore = opts.reachBufferBefore;
end
if (~isfield(opts, 'eventBufferAfter') || isempty(opts.eventBufferAfter)) ...
    && isfield(opts, 'reachBufferAfter') && ~isempty(opts.reachBufferAfter)
  opts.eventBufferAfter = opts.reachBufferAfter;
end
[opts.eventBufferBefore, opts.eventBufferAfter] = resolve_engagement_buffer_pair( ...
  opts, 'eventBufferBefore', 'eventBufferAfter', 'reachBuffer', 1);
if ~isfield(opts, 'absorbSingleEvents') || isempty(opts.absorbSingleEvents)
  if isfield(opts, 'absorbSingleReaches') && ~isempty(opts.absorbSingleReaches)
    opts.absorbSingleEvents = opts.absorbSingleReaches;
  else
    opts.absorbSingleEvents = true;
  end
end
if ~isfield(opts, 'movAvgWinSec') || isempty(opts.movAvgWinSec)
  opts.movAvgWinSec = 30;
end
if ~isfield(opts, 'rateBinSec') || isempty(opts.rateBinSec)
  opts.rateBinSec = 1;
end
if ~isfield(opts, 'ieiHistBinSec') || isempty(opts.ieiHistBinSec)
  opts.ieiHistBinSec = 1;
end
if ~isfield(opts, 'ieiHistMaxSec')
  opts.ieiHistMaxSec = [];
end
if ~isfield(opts, 'makePlots') || isempty(opts.makePlots)
  opts.makePlots = true;
end
if ~isfield(opts, 'saveFigure') || isempty(opts.saveFigure)
  opts.saveFigure = false;
end
if ~isfield(opts, 'outputDir')
  opts.outputDir = '';
end
end

function setup_semicircle_reward_session_performance_paths()
% SETUP_SEMICIRCLE_REWARD_SESSION_PERFORMANCE_PATHS - Add neuro-behavior paths

scriptDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(scriptDir, '..');
addpath(srcPath);
addpath(fullfile(srcPath, 'semicircle_reward_task'));
addpath(fullfile(srcPath, 'criticality'));
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
end

%% -------------------------------------------------------------------------
%% Accuracy and IEI summaries
%% -------------------------------------------------------------------------

function accuracyStats = summarize_semicircle_accuracy(isCorrect, isError)
% SUMMARIZE_SEMICIRCLE_ACCURACY - Correct/error counts and percent correct

accuracyStats = struct();
accuracyStats.nCorrect = sum(isCorrect);
accuracyStats.nError = sum(isError);
accuracyStats.nTotal = accuracyStats.nCorrect + accuracyStats.nError;
if accuracyStats.nTotal > 0
  accuracyStats.accuracyPct = 100 * accuracyStats.nCorrect / accuracyStats.nTotal;
else
  accuracyStats.accuracyPct = nan;
end
end

function print_semicircle_accuracy(accuracyStats)
% PRINT_SEMICIRCLE_ACCURACY - Command-window accuracy summary

fprintf('\n--- Accuracy ---\n');
if accuracyStats.nTotal == 0
  fprintf('No rewarded/unrewarded beam breaks in collect window\n');
  return;
end
fprintf('Accuracy: %.1f%% (%d correct/rewarded, %d error/unrewarded; n=%d)\n', ...
  accuracyStats.accuracyPct, accuracyStats.nCorrect, accuracyStats.nError, ...
  accuracyStats.nTotal);
end

function print_beam_break_interval_summary(beamBreakIntervals, minNonEngagedWindow)
% PRINT_BEAM_BREAK_INTERVAL_SUMMARY - IEI stats for minNonEngagedWindow choice

fprintf('\n--- Beam-break intervals (for minNonEngagedWindow) ---\n');
if isempty(beamBreakIntervals)
  fprintf('Fewer than 2 events; no intervals to summarize\n');
  return;
end
iei = beamBreakIntervals(isfinite(beamBreakIntervals) & beamBreakIntervals > 0);
if isempty(iei)
  fprintf('No positive finite intervals\n');
  return;
end
fracAbove = 100 * mean(iei >= minNonEngagedWindow);
fprintf(['n=%d IEIs | median=%.1f s | p25=%.1f | p75=%.1f | max=%.1f s | ', ...
  '%.1f%% >= minNonEngagedWindow (%.1f s)\n'], ...
  numel(iei), median(iei), prctile(iei, 25), prctile(iei, 75), max(iei), ...
  fracAbove, minNonEngagedWindow);
end

%% -------------------------------------------------------------------------
%% Plotting
%% -------------------------------------------------------------------------

function fig = plot_semicircle_reward_session_performance(eventTimes, isCorrect, isError, ...
    engagedSegs, nonEngagedSegs, beamBreakIntervals, subjectName, sessionName, ...
    collectStart, collectEnd, opts)
% PLOT_SEMICIRCLE_REWARD_SESSION_PERFORMANCE - Engagement, rates, accuracy, IEI hist
%
% Variables:
%   eventTimes, isCorrect, isError - Beam-break onsets and outcomes
%   engagedSegs, nonEngagedSegs - Engagement interval structs (.start, .end)
%   beamBreakIntervals - Consecutive inter-event intervals (s)
%   subjectName, sessionName, collectStart, collectEnd, opts
%
% Goal:
%   Four-panel figure: (1) engagement shading + green/red beam breaks,
%   (2) moving reward rate, (3) moving accuracy, (4) IEI histogram with
%   minNonEngagedWindow marker.

engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];
correctColor = [0.0, 0.65, 0.2];
errorColor = [0.85, 0.15, 0.15];

rewardTimesSec = eventTimes(isCorrect);
timeBinsSec = (collectStart:opts.rateBinSec:collectEnd)';
if numel(timeBinsSec) < 2
  timeBinsSec = [collectStart; collectStart + opts.rateBinSec];
end
binEdges = [timeBinsSec; timeBinsSec(end) + opts.rateBinSec];
smoothWinBins = max(1, round(opts.movAvgWinSec / opts.rateBinSec));

rewardCounts = histcounts(rewardTimesSec, binEdges);
rewardsPerMin = rewardCounts(:) * (60 / opts.rateBinSec);
rewardsPerMinSmooth = movmean(rewardsPerMin, smoothWinBins);

correctCounts = histcounts(eventTimes(isCorrect), binEdges);
errorCounts = histcounts(eventTimes(isError), binEdges);
correctSmooth = movmean(correctCounts(:), smoothWinBins);
errorSmooth = movmean(errorCounts(:), smoothWinBins);
denom = correctSmooth + errorSmooth;
accuracySmooth = nan(size(denom));
validAcc = denom > 0;
accuracySmooth(validAcc) = 100 * correctSmooth(validAcc) ./ denom(validAcc);

fig = figure('Color', 'w', ...
  'Name', sprintf('%s %s semicircle performance', subjectName, sessionName), ...
  'Units', 'pixels');
layout = tiledlayout(fig, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Panel 1: engagement + beam breaks
axEvents = nexttile(layout);
hold(axEvents, 'on');
yMin = 0;
yMax = 1;
hNon = add_segment_patches(axEvents, nonEngagedSegs, nonEngagedColor, yMin, yMax);
hEng = add_segment_patches(axEvents, engagedSegs, engagedColor, yMin, yMax);

hErr = gobjects(0);
hCorr = gobjects(0);
for iEvent = 1:numel(eventTimes)
  x = eventTimes(iEvent);
  if isCorrect(iEvent)
    hLine = plot(axEvents, [x, x], [yMin, yMax], 'Color', correctColor, ...
      'LineWidth', 1.0, 'HandleVisibility', 'off');
    if isempty(hCorr)
      hCorr = hLine;
      set(hCorr, 'HandleVisibility', 'on', 'DisplayName', 'Correct/rewarded');
    end
  elseif isError(iEvent)
    hLine = plot(axEvents, [x, x], [yMin, yMax], 'Color', errorColor, ...
      'LineWidth', 1.0, 'HandleVisibility', 'off');
    if isempty(hErr)
      hErr = hLine;
      set(hErr, 'HandleVisibility', 'on', 'DisplayName', 'Error/unrewarded');
    end
  else
    plot(axEvents, [x, x], [yMin, yMax], 'Color', [0.45, 0.45, 0.45], ...
      'LineWidth', 0.75, 'HandleVisibility', 'off');
  end
end

legendHandles = gobjects(0);
legendLabels = {};
if ~isempty(hNon)
  set(hNon, 'HandleVisibility', 'on', 'DisplayName', ...
    sprintf('Non-engaged (n=%d)', numel(nonEngagedSegs)));
  legendHandles(end + 1) = hNon; %#ok<AGROW>
  legendLabels{end + 1} = get(hNon, 'DisplayName'); %#ok<AGROW>
end
if ~isempty(hEng)
  set(hEng, 'HandleVisibility', 'on', 'DisplayName', ...
    sprintf('Engaged (n=%d)', numel(engagedSegs)));
  legendHandles(end + 1) = hEng; %#ok<AGROW>
  legendLabels{end + 1} = get(hEng, 'DisplayName'); %#ok<AGROW>
end
if ~isempty(hCorr)
  legendHandles(end + 1) = hCorr; %#ok<AGROW>
  legendLabels{end + 1} = 'Correct/rewarded'; %#ok<AGROW>
end
if ~isempty(hErr)
  legendHandles(end + 1) = hErr; %#ok<AGROW>
  legendLabels{end + 1} = 'Error/unrewarded'; %#ok<AGROW>
end

xlim(axEvents, [collectStart, collectEnd]);
ylim(axEvents, [yMin, yMax]);
yticks(axEvents, []);
xlabel(axEvents, 'Session time (s)');
ylabel(axEvents, 'Engagement');
title(axEvents, sprintf('%s / %s — beam breaks and engagement', ...
  subjectName, sessionName), 'Interpreter', 'none');
if ~isempty(legendHandles)
  legend(axEvents, legendHandles, legendLabels, 'Location', 'best');
end
grid(axEvents, 'on');
hold(axEvents, 'off');

%% Panel 2: reward rate
axReward = nexttile(layout);
plot(axReward, timeBinsSec, rewardsPerMinSmooth, 'k-', 'LineWidth', 1.2);
xlabel(axReward, 'Session time (s)');
ylabel(axReward, sprintf('Rewards/min (%.0f s moving avg)', opts.movAvgWinSec));
title(axReward, 'Running reward rate (rewarded beam breaks)');
xlim(axReward, [collectStart, collectEnd]);
grid(axReward, 'on');

%% Panel 3: accuracy
axAcc = nexttile(layout);
plot(axAcc, timeBinsSec, accuracySmooth, 'k-', 'LineWidth', 1.2);
xlabel(axAcc, 'Session time (s)');
ylabel(axAcc, sprintf('Accuracy %% (%.0f s moving window)', opts.movAvgWinSec));
title(axAcc, 'Running accuracy (rewarded / (rewarded + unrewarded))');
xlim(axAcc, [collectStart, collectEnd]);
ylim(axAcc, [0, 100]);
grid(axAcc, 'on');

%% Panel 4: beam-break interval histogram
axIei = nexttile(layout);
hold(axIei, 'on');
iei = beamBreakIntervals(isfinite(beamBreakIntervals) & beamBreakIntervals > 0);
if isempty(iei)
  text(axIei, 0.5, 0.5, 'Fewer than 2 beam breaks', ...
    'Units', 'normalized', 'HorizontalAlignment', 'center');
  title(axIei, 'Beam-break intervals');
else
  if isempty(opts.ieiHistMaxSec) || ~isfinite(opts.ieiHistMaxSec)
    histMaxSec = max(prctile(iei, 99), opts.minNonEngagedWindow * 1.25);
  else
    histMaxSec = opts.ieiHistMaxSec;
  end
  histMaxSec = max(histMaxSec, opts.ieiHistBinSec);
  edges = 0:opts.ieiHistBinSec:(histMaxSec + opts.ieiHistBinSec);
  histogram(axIei, iei, edges, 'FaceColor', [0.35, 0.45, 0.65], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.85);
  hMin = xline(axIei, opts.minNonEngagedWindow, 'r--', 'LineWidth', 1.5, ...
    'Label', sprintf('minNonEngagedWindow = %.1f s', opts.minNonEngagedWindow), ...
    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'top');
  set(hMin, 'Interpreter', 'none');
  xlim(axIei, [0, histMaxSec]);
  xlabel(axIei, 'Inter-beam-break interval (s)');
  ylabel(axIei, 'Count');
  title(axIei, sprintf( ...
    ['Beam-break intervals (n=%d; median=%.1f s) — use bimodality to set ', ...
    'minNonEngagedWindow'], numel(iei), median(iei)));
  legend(axIei, hMin, 'Location', 'best');
end
grid(axIei, 'on');
hold(axIei, 'off');

fit_figure_on_screen(fig, 1100, 980);
end

function h = add_segment_patches(ax, segs, colorVal, yMin, yMax)
% ADD_SEGMENT_PATCHES - Shade engagement intervals; return one legend handle

h = gobjects(0);
for i = 1:numel(segs)
  t0 = segs(i).start;
  t1 = segs(i).end;
  hi = patch(ax, [t0, t1, t1, t0], [yMin, yMin, yMax, yMax], colorVal, ...
    'FaceAlpha', 0.35, 'EdgeColor', 'none', 'HandleVisibility', 'off');
  if isempty(h)
    h = hi;
  end
end
end

function fit_figure_on_screen(fig, prefWidth, prefHeight)
% FIT_FIGURE_ON_SCREEN - Place figure on screen at preferred size

monitorPositions = get(0, 'MonitorPositions');
screen = monitorPositions(1, :);
width = min(prefWidth, screen(3) - 80);
height = min(prefHeight, screen(4) - 100);
left = screen(1) + max(40, (screen(3) - width) / 2);
bottom = screen(2) + max(40, (screen(4) - height) / 2);
set(fig, 'Position', [left, bottom, width, height]);
end

%% -------------------------------------------------------------------------
%% Engagement segments (same definition as semicircle / reach engagement)
%% -------------------------------------------------------------------------

function [engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
    collectStart, collectEnd, eventTimes, minNonEngagedWindow, bufferBefore, bufferAfter, ...
    absorbSingleEvents)
% DEFINE_REACH_ENGAGEMENT_SEGMENTS - Continuous engaged / non-engaged intervals

if nargin < 5 || isempty(bufferBefore)
  bufferBefore = 0;
end
if nargin < 6 || isempty(bufferAfter)
  bufferAfter = bufferBefore;
end
if nargin < 7 || isempty(absorbSingleEvents)
  absorbSingleEvents = true;
end
bufferBefore = max(0, bufferBefore);
bufferAfter = max(0, bufferAfter);

eventTimes = sort(eventTimes(:));
eventTimes = eventTimes(eventTimes >= collectStart & eventTimes <= collectEnd);

occupied = merge_reach_buffer_intervals( ...
  eventTimes, bufferBefore, bufferAfter, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if absorbSingleEvents && ~isempty(occupied)
  absorbedMask = get_absorbed_single_reach_occupied_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, bufferBefore, bufferAfter);
end

nonEngagedSegs = struct('start', {}, 'end', {});
cursor = collectStart;
iOcc = 1;
while iOcc <= numel(occupied)
  if absorbedMask(iOcc)
    gapStart = cursor;
    if iOcc < numel(occupied)
      gapEnd = occupied(iOcc + 1).start;
    else
      gapEnd = collectEnd;
    end
    if (gapEnd - gapStart) >= minNonEngagedWindow
      nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
      nonEngagedSegs(end).end = gapEnd;
    end
    cursor = gapEnd;
  else
    gapStart = cursor;
    gapEnd = occupied(iOcc).start;
    if (gapEnd - gapStart) >= minNonEngagedWindow
      nonEngagedSegs(end + 1).start = gapStart; %#ok<AGROW>
      nonEngagedSegs(end).end = gapEnd;
    end
    cursor = occupied(iOcc).end;
  end
  iOcc = iOcc + 1;
end
if (collectEnd - cursor) >= minNonEngagedWindow
  nonEngagedSegs(end + 1).start = cursor;
  nonEngagedSegs(end).end = collectEnd;
end

engagedSegs = complement_segments(collectStart, collectEnd, nonEngagedSegs);
end

function absorbedMask = get_absorbed_single_reach_occupied_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, bufferBefore, bufferAfter)
% GET_ABSORBED_SINGLE_REACH_OCCUPIED_MASK - Isolated single events to absorb

if nargin < 6 || isempty(bufferAfter)
  bufferAfter = bufferBefore;
end
occupied = merge_reach_buffer_intervals( ...
  eventTimes, bufferBefore, bufferAfter, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if isempty(occupied)
  return;
end

for iOcc = 1:numel(occupied)
  nEventsInOcc = sum(eventTimes >= occupied(iOcc).start & eventTimes <= occupied(iOcc).end);
  if nEventsInOcc ~= 1
    continue;
  end
  if iOcc == 1
    gapBeforeStart = collectStart;
  else
    gapBeforeStart = occupied(iOcc - 1).end;
  end
  gapBeforeEnd = occupied(iOcc).start;
  gapAfterStart = occupied(iOcc).end;
  if iOcc == numel(occupied)
    gapAfterEnd = collectEnd;
  else
    gapAfterEnd = occupied(iOcc + 1).start;
  end
  if (gapBeforeEnd - gapBeforeStart) >= minNonEngagedWindow && ...
      (gapAfterEnd - gapAfterStart) >= minNonEngagedWindow
    absorbedMask(iOcc) = true;
  end
end
end

function occupied = merge_reach_buffer_intervals(eventTimes, bufferBefore, bufferAfter, ...
    collectStart, collectEnd)
% MERGE_REACH_BUFFER_INTERVALS - Union of [event-before, event+after]

if nargin < 3 || isempty(bufferAfter)
  bufferAfter = bufferBefore;
end
bufferBefore = max(0, bufferBefore);
bufferAfter = max(0, bufferAfter);

occupied = struct('start', {}, 'end', {});
if isempty(eventTimes)
  return;
end

starts = max(collectStart, eventTimes - bufferBefore);
ends = min(collectEnd, eventTimes + bufferAfter);
valid = ends > starts;
starts = starts(valid);
ends = ends(valid);
if isempty(starts)
  return;
end

[starts, ord] = sort(starts);
ends = ends(ord);

occupied(1).start = starts(1);
occupied(1).end = ends(1);
for i = 2:numel(starts)
  if starts(i) <= occupied(end).end
    occupied(end).end = max(occupied(end).end, ends(i));
  else
    occupied(end + 1).start = starts(i); %#ok<AGROW>
    occupied(end).end = ends(i);
  end
end
end

function engagedSegs = complement_segments(collectStart, collectEnd, nonEngagedSegs)
% COMPLEMENT_SEGMENTS - Intervals in collect window not in nonEngagedSegs

engagedSegs = struct('start', {}, 'end', {});
if isempty(nonEngagedSegs)
  engagedSegs(1).start = collectStart;
  engagedSegs(1).end = collectEnd;
  return;
end

starts = [nonEngagedSegs.start];
ends = [nonEngagedSegs.end];
[starts, ord] = sort(starts);
ends = ends(ord);

cursor = collectStart;
for i = 1:numel(starts)
  if starts(i) > cursor + eps
    engagedSegs(end + 1).start = cursor; %#ok<AGROW>
    engagedSegs(end).end = starts(i);
  end
  cursor = max(cursor, ends(i));
end
if cursor < collectEnd - eps
  engagedSegs(end + 1).start = cursor;
  engagedSegs(end).end = collectEnd;
end
end
