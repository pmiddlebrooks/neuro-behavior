function out = reach_session_performance(sessionName, opts)
% REACH_SESSION_PERFORMANCE - Behavioral summary for one reach-task session
%
% Variables:
%   sessionName - Reach session folder/file stem (e.g. 'AB19_31-Mar-2026 15_46_45_NeuroBeh')
%   opts        - Options struct. Fields:
%       .collectStart, .collectEnd - Analysis window (s); empty collectEnd = session end
%       .minNonEngagedWindow - Min gap for non-engaged segments (default 30)
%       .reachBuffer         - Buffer around reaches (s; default 1)
%       .absorbSingleReaches - Merge isolated reaches into non-engaged (default true)
%       .movAvgWinSec        - Moving-average window for rates/accuracy (default 30)
%       .rateBinSec          - Time bin for rate/accuracy traces (default 1)
%       .makePlots           - Create figure (default true)
%       .saveFigure          - Save PNG/EPS (default false)
%       .outputDir           - Save directory (default session results folder)
%
% Goal:
%   Load reach behavior, shade engaged/non-engaged segments (same definition as
%   reach_criticality_metrics_engagement), mark correct (green) and error (red)
%   reaches, plot 30 s running reward rate and accuracy, and print Block 1 / 2
%   accuracy with correct and error counts.
%
% Returns:
%   With no inputs: default options struct.
%   Otherwise: struct with reaches, segments, blockStats, figHandle.

setup_reach_session_performance_paths();

if nargin == 0
  out = fill_reach_session_performance_opts(struct());
  return;
end
if nargin < 1 || isempty(sessionName)
  error('reach_session_performance:MissingSession', 'sessionName is required.');
end
if nargin < 2 || isempty(opts)
  opts = struct();
end
opts = fill_reach_session_performance_opts(opts);

paths = get_paths();
reachDataFile = fullfile(paths.reachDataPath, [sessionName, '.mat']);
if ~isfile(reachDataFile)
  error('reach_session_performance:FileNotFound', ...
    'Reach data file not found: %s', reachDataFile);
end

fprintf('\n=== Reach session performance ===\n');
fprintf('Session: %s\n', sessionName);
fprintf('Loading: %s\n', reachDataFile);
dataR = load(reachDataFile);

reachStartAll = dataR.R(:, 1) / 1000;
reachClassAll = dataR.Block(:, 3);
sessionEnd = round(min(dataR.R(end, 1) + 5000, max(dataR.CSV(:, 1) * 1000)) / 1000);

collectStart = opts.collectStart;
collectEnd = opts.collectEnd;
if isempty(collectEnd)
  collectEnd = sessionEnd;
end
if isempty(collectStart)
  collectStart = 0;
end

reachInCollect = reachStartAll >= collectStart & reachStartAll <= collectEnd;
reachStart = reachStartAll(reachInCollect);
reachClass = reachClassAll(reachInCollect);

fprintf('Collect window: [%.1f, %.1f] s (%.1f min)\n', ...
  collectStart, collectEnd, (collectEnd - collectStart) / 60);
fprintf('Reaches in collect window: %d\n', numel(reachStart));

[engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
  collectStart, collectEnd, reachStart, opts.minNonEngagedWindow, opts.reachBuffer, ...
  opts.absorbSingleReaches);

isCorrect = ismember(reachClass, [2, 4, 6]);
isError = ismember(reachClass, [1, 3, 5]);
blockStats = summarize_reach_block_accuracy(reachClass, isCorrect, isError);
print_reach_block_accuracy(blockStats);

out = struct();
out.sessionName = sessionName;
out.config = opts;
out.collectStart = collectStart;
out.collectEnd = collectEnd;
out.reachStart = reachStart;
out.reachClass = reachClass;
out.isCorrect = isCorrect;
out.isError = isError;
out.segments = struct('engaged', engagedSegs, 'nonEngaged', nonEngagedSegs);
out.blockStats = blockStats;
out.figHandle = gobjects(0);

if opts.makePlots
  out.figHandle = plot_reach_session_performance( ...
    reachStart, reachClass, isCorrect, isError, engagedSegs, nonEngagedSegs, ...
    sessionName, collectStart, collectEnd, opts);
  if opts.saveFigure && isgraphics(out.figHandle)
    saveDir = opts.outputDir;
    if isempty(saveDir)
      [~, dataBaseName, ~] = fileparts(sessionName);
      saveDir = fullfile(paths.dropPath, 'reach_task', 'results', dataBaseName);
    end
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    plotBase = fullfile(saveDir, sprintf('reach_session_performance_%s', ...
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

function opts = fill_reach_session_performance_opts(opts)
% FILL_REACH_SESSION_PERFORMANCE_OPTS - Default behavioral summary options

if ~isfield(opts, 'collectStart') || isempty(opts.collectStart)
  opts.collectStart = 0;
end
if ~isfield(opts, 'collectEnd')
  opts.collectEnd = [];
end
if ~isfield(opts, 'minNonEngagedWindow') || isempty(opts.minNonEngagedWindow)
  opts.minNonEngagedWindow = 30;
end
if ~isfield(opts, 'reachBuffer') || isempty(opts.reachBuffer)
  opts.reachBuffer = 1;
end
if ~isfield(opts, 'absorbSingleReaches') || isempty(opts.absorbSingleReaches)
  opts.absorbSingleReaches = true;
end
if ~isfield(opts, 'movAvgWinSec') || isempty(opts.movAvgWinSec)
  opts.movAvgWinSec = 30;
end
if ~isfield(opts, 'rateBinSec') || isempty(opts.rateBinSec)
  opts.rateBinSec = 1;
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

function setup_reach_session_performance_paths()
% SETUP_REACH_SESSION_PERFORMANCE_PATHS - Add neuro-behavior paths

scriptDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(scriptDir, '..');
addpath(srcPath);
addpath(fullfile(srcPath, 'reach_task'));
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
end

%% -------------------------------------------------------------------------
%% Block accuracy
%% -------------------------------------------------------------------------

function blockStats = summarize_reach_block_accuracy(reachClass, isCorrect, isError)
% SUMMARIZE_REACH_BLOCK_ACCURACY - Correct/error counts and accuracy by block
%
% Variables:
%   reachClass - Reach class labels (1/2 = Block 1 error/rew; 3/4 = Block 2)
%   isCorrect, isError - Logical masks for outcome

blockStats = struct();
blockStats.block1 = struct('nCorrect', 0, 'nError', 0, 'nTotal', 0, 'accuracyPct', nan);
blockStats.block2 = struct('nCorrect', 0, 'nError', 0, 'nTotal', 0, 'accuracyPct', nan);

block1Mask = ismember(reachClass, [1, 2]);
block2Mask = ismember(reachClass, [3, 4]);
blockStats.block1 = fill_block_accuracy_struct(isCorrect & block1Mask, isError & block1Mask);
blockStats.block2 = fill_block_accuracy_struct(isCorrect & block2Mask, isError & block2Mask);
end

function stats = fill_block_accuracy_struct(correctMask, errorMask)
% FILL_BLOCK_ACCURACY_STRUCT - Counts and percent correct for one block

stats = struct();
stats.nCorrect = sum(correctMask);
stats.nError = sum(errorMask);
stats.nTotal = stats.nCorrect + stats.nError;
if stats.nTotal > 0
  stats.accuracyPct = 100 * stats.nCorrect / stats.nTotal;
else
  stats.accuracyPct = nan;
end
end

function print_reach_block_accuracy(blockStats)
% PRINT_REACH_BLOCK_ACCURACY - Command-window Block 1 / 2 accuracy summary

fprintf('\n--- Accuracy by block ---\n');
print_one_block_accuracy('Block 1', blockStats.block1);
print_one_block_accuracy('Block 2', blockStats.block2);
end

function print_one_block_accuracy(blockName, stats)
% PRINT_ONE_BLOCK_ACCURACY - One block line for console

if stats.nTotal == 0
  fprintf('%s: no reaches\n', blockName);
  return;
end
fprintf('%s: accuracy %.1f%% (%d correct, %d error; n=%d)\n', ...
  blockName, stats.accuracyPct, stats.nCorrect, stats.nError, stats.nTotal);
end

%% -------------------------------------------------------------------------
%% Plotting
%% -------------------------------------------------------------------------

function fig = plot_reach_session_performance(reachStart, reachClass, isCorrect, isError, ...
    engagedSegs, nonEngagedSegs, sessionName, collectStart, collectEnd, opts)
% PLOT_REACH_SESSION_PERFORMANCE - Engagement, reward rate, and accuracy traces
%
% Variables:
%   reachStart, reachClass, isCorrect, isError - Reach onsets and outcomes
%   engagedSegs, nonEngagedSegs - Engagement interval structs (.start, .end)
%   sessionName, collectStart, collectEnd, opts
%
% Goal:
%   Three-panel figure: (1) engagement shading + green/red reaches,
%   (2) 30 s moving reward rate, (3) 30 s moving accuracy.

engagedColor = [0.15, 0.45, 0.75];
nonEngagedColor = [0.85, 0.35, 0.15];
correctColor = [0.0, 0.65, 0.2];
errorColor = [0.85, 0.15, 0.15];

rewardTimesSec = reachStart(isCorrect);
timeBinsSec = (collectStart:opts.rateBinSec:collectEnd)';
if numel(timeBinsSec) < 2
  timeBinsSec = [collectStart; collectStart + opts.rateBinSec];
end
binEdges = [timeBinsSec; timeBinsSec(end) + opts.rateBinSec];
smoothWinBins = max(1, round(opts.movAvgWinSec / opts.rateBinSec));

rewardCounts = histcounts(rewardTimesSec, binEdges);
rewardsPerMin = rewardCounts(:) * (60 / opts.rateBinSec);
rewardsPerMinSmooth = movmean(rewardsPerMin, smoothWinBins);

correctCounts = histcounts(reachStart(isCorrect), binEdges);
errorCounts = histcounts(reachStart(isError), binEdges);
correctSmooth = movmean(correctCounts(:), smoothWinBins);
errorSmooth = movmean(errorCounts(:), smoothWinBins);
denom = correctSmooth + errorSmooth;
accuracySmooth = nan(size(denom));
validAcc = denom > 0;
accuracySmooth(validAcc) = 100 * correctSmooth(validAcc) ./ denom(validAcc);

fig = figure('Color', 'w', 'Name', sprintf('%s reach performance', sessionName), ...
  'Units', 'pixels');
layout = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Panel 1: engagement + reaches
axReach = nexttile(layout);
hold(axReach, 'on');
yMin = 0;
yMax = 1;
hNon = add_segment_patches(axReach, nonEngagedSegs, nonEngagedColor, yMin, yMax);
hEng = add_segment_patches(axReach, engagedSegs, engagedColor, yMin, yMax);

hErr = gobjects(0);
hCorr = gobjects(0);
for iReach = 1:numel(reachStart)
  x = reachStart(iReach);
  if isCorrect(iReach)
    hLine = plot(axReach, [x, x], [yMin, yMax], 'Color', correctColor, ...
      'LineWidth', 1.0, 'HandleVisibility', 'off');
    if isempty(hCorr)
      hCorr = hLine;
      set(hCorr, 'HandleVisibility', 'on', 'DisplayName', 'Correct');
    end
  elseif isError(iReach)
    hLine = plot(axReach, [x, x], [yMin, yMax], 'Color', errorColor, ...
      'LineWidth', 1.0, 'HandleVisibility', 'off');
    if isempty(hErr)
      hErr = hLine;
      set(hErr, 'HandleVisibility', 'on', 'DisplayName', 'Error');
    end
  else
    plot(axReach, [x, x], [yMin, yMax], 'Color', [0.45, 0.45, 0.45], ...
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
  legendLabels{end + 1} = 'Correct'; %#ok<AGROW>
end
if ~isempty(hErr)
  legendHandles(end + 1) = hErr; %#ok<AGROW>
  legendLabels{end + 1} = 'Error'; %#ok<AGROW>
end

block2Mask = ismember(reachClass, [3, 4]);
if any(block2Mask)
  block2StartTime = reachStart(find(block2Mask, 1, 'first'));
  hBlock2 = plot(axReach, [block2StartTime, block2StartTime], [yMin, yMax], 'k--', ...
    'LineWidth', 1.5, 'DisplayName', 'Block 2 start');
  legendHandles(end + 1) = hBlock2; %#ok<AGROW>
  legendLabels{end + 1} = 'Block 2 start'; %#ok<AGROW>
end

xlim(axReach, [collectStart, collectEnd]);
ylim(axReach, [yMin, yMax]);
yticks(axReach, []);
xlabel(axReach, 'Session time (s)');
ylabel(axReach, 'Engagement');
title(axReach, sprintf('%s — reaches and engagement', sessionName), 'Interpreter', 'none');
if ~isempty(legendHandles)
  legend(axReach, legendHandles, legendLabels, 'Location', 'best');
end
grid(axReach, 'on');
hold(axReach, 'off');

%% Panel 2: reward rate
axReward = nexttile(layout);
plot(axReward, timeBinsSec, rewardsPerMinSmooth, 'k-', 'LineWidth', 1.2);
xlabel(axReward, 'Session time (s)');
ylabel(axReward, sprintf('Rewards/min (%.0f s moving avg)', opts.movAvgWinSec));
title(axReward, 'Running reward rate (correct reaches)');
xlim(axReward, [collectStart, collectEnd]);
grid(axReward, 'on');

%% Panel 3: accuracy
axAcc = nexttile(layout);
plot(axAcc, timeBinsSec, accuracySmooth, 'k-', 'LineWidth', 1.2);
xlabel(axAcc, 'Session time (s)');
ylabel(axAcc, sprintf('Accuracy %% (%.0f s moving window)', opts.movAvgWinSec));
title(axAcc, 'Running accuracy (correct / (correct + error))');
xlim(axAcc, [collectStart, collectEnd]);
ylim(axAcc, [0, 100]);
grid(axAcc, 'on');

fit_figure_on_screen(fig, 1100, 820);
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
%% Engagement segments (same definition as reach_criticality_metrics_engagement)
%% -------------------------------------------------------------------------

function [engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
    collectStart, collectEnd, reachStart, minNonEngagedWindow, reachBuffer, absorbSingleReaches)
% DEFINE_REACH_ENGAGEMENT_SEGMENTS - Continuous engaged / non-engaged intervals

if nargin < 5 || isempty(reachBuffer)
  reachBuffer = 0;
end
if nargin < 6 || isempty(absorbSingleReaches)
  absorbSingleReaches = true;
end
reachBuffer = max(0, reachBuffer);

reachStart = sort(reachStart(:));
reachStart = reachStart(reachStart >= collectStart & reachStart <= collectEnd);

occupied = merge_reach_buffer_intervals(reachStart, reachBuffer, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if absorbSingleReaches && ~isempty(occupied)
  absorbedMask = get_absorbed_single_reach_occupied_mask( ...
    reachStart, collectStart, collectEnd, minNonEngagedWindow, reachBuffer);
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
    reachStart, collectStart, collectEnd, minNonEngagedWindow, reachBuffer)
% GET_ABSORBED_SINGLE_REACH_OCCUPIED_MASK - Isolated single reaches to absorb

occupied = merge_reach_buffer_intervals(reachStart, reachBuffer, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if isempty(occupied)
  return;
end

for iOcc = 1:numel(occupied)
  nReachesInOcc = sum(reachStart >= occupied(iOcc).start & reachStart <= occupied(iOcc).end);
  if nReachesInOcc ~= 1
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

function occupied = merge_reach_buffer_intervals(reachStart, reachBuffer, collectStart, collectEnd)
% MERGE_REACH_BUFFER_INTERVALS - Union of [reach-buffer, reach+buffer]

occupied = struct('start', {}, 'end', {});
if isempty(reachStart)
  return;
end

starts = max(collectStart, reachStart - reachBuffer);
ends = min(collectEnd, reachStart + reachBuffer);
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
