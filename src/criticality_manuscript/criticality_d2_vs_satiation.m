%%
% Criticality: d2 vs satiation vs engagement (Manuscript)
%
% Windowed d2 across task sessions (interval, semicircle, reach — not
% spontaneous). For each non-overlapping d2 window, records (1) cumulative
% rewards through the window end (satiation proxy) and (2) the fraction of
% the window overlapping continuous engaged segments. Marker fill (engaged)
% uses the same full-containment rule as engagement d2/PRG: the window must
% lie fully inside an engaged segment (straddlers are not marked engaged).
% Segment definition matches interval / semicircle / reach engagement analyses.
%
% Variables (configure in this section):
%   sessionTypes   - Task types; spontaneous is dropped if present
%   collectStart, collectEnd - Analysis range (s); [] collectEnd = session end
%   d2Window       - Non-overlapping window length (s)
%   binSize        - Spike bin width (s) for d2
%   brainArea, brainAreaCombinations - Area selection
%   splitByEngagement - If true, marker fill = windows fully inside engaged
%                       segments (straddlers / non-engaged stay open)
%   engagementBufferBefore / After - Seconds around each event = engaged
%   minNonEngagedWindow, absorbSingleEvents - Engaged-segment definition
%                        (gaps >= minNonEngagedWindow without events are non-engaged)
%   useLog10D2     - Plot / correlate log10(d2)
%   useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple
%   makePlots, saveFigure, closeFigure, plotConfig
%
% Plot:
%   nTasks x 2 scatters (one row per task, task color): d2 vs cumulative
%   rewards | d2 vs engagement fraction. Shared y-limits; filled = engaged
%   when splitByEngagement.
%
% Stats (rule out satiation as the stronger d2 predictor):
%   Spearman rho(d2, reward) vs rho(d2, engagement); Steiger Z on those
%   dependent correlations; Spearman partial correlations; OLS unique R^2
%   after session-mean-centering (removes between-session confounds).
%
% Goal:
%   Show that window d2 tracks engagement more than cumulative reward.

%% Paths and configuration
setup_criticality_manuscript_paths('criticality_d2_vs_satiation');
paths = get_paths();

sessionTypes = order_manuscript_session_types({'interval', 'semicircle', 'reach'});
collectStart = 10;
collectEnd = [];
d2Window = 30;
binSize = 0.04;
brainArea = 'M23M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

splitByEngagement = true;
engagementBufferBefore = 3;
engagementBufferAfter = 1;
minNonEngagedWindow = 30;
absorbSingleEvents = true;

useLog10D2 = true;
useSubsampling = false;
nSubsamples = 40;
nNeuronsSubsample = 45;
minNeuronsMultiple = 1.1;
nMinNeurons = 20;

minLeaveSec = 0.1;   % interval trial parsing
makePlots = true;
saveFigure = true;
closeFigure = false;
if ~exist('plotConfig', 'var') || isempty(plotConfig)
  plotConfig = struct();
end
plotConfig = fill_manuscript_plot_config(plotConfig);

sessionTypes = filter_to_task_session_types(sessionTypes);
if isempty(sessionTypes)
  error('criticality_d2_vs_satiation:NoTasks', ...
    'Need interval, semicircle, and/or reach (spontaneous has no rewards).');
end

[engagementBufferBefore, engagementBufferAfter] = resolve_engagement_buffer_pair( ...
  struct('engagementBufferBefore', engagementBufferBefore, ...
  'engagementBufferAfter', engagementBufferAfter), ...
  'engagementBufferBefore', 'engagementBufferAfter', 'engagementBuffer', 1);

fprintf('\n=== criticality_d2_vs_satiation ===\n');
fprintf('Session types: %s\n', strjoin(sessionTypes, ', '));
fprintf('d2 windows: %.0f s; binSize: %.3f s; brainArea: %s\n', ...
  d2Window, binSize, brainArea);
fprintf(['splitByEngagement: %d; eventBuffer before=%.3g s after=%.3g s; ', ...
  'minNonEngagedWindow=%.1f s; absorbSingleEvents=%d\n'], ...
  splitByEngagement, engagementBufferBefore, engagementBufferAfter, ...
  minNonEngagedWindow, absorbSingleEvents);
fprintf('useLog10D2: %d\n', useLog10D2);
if useSubsampling
  fprintf('Subsampling: %d x %d neurons\n', nSubsamples, nNeuronsSubsample);
else
  fprintf('Subsampling: off\n');
end

sessionTable = build_task_session_table(sessionTypes);
numSessions = height(sessionTable);
fprintf('Sessions: %d\n', numSessions);

analysisConfig = build_d2_analysis_config(d2Window, binSize, useLog10D2, ...
  useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple, nMinNeurons);

loadOpts = neuro_behavior_options();
loadOpts.firingRateCheckTime = [];
loadOpts.collectStart = collectStart;
loadOpts.collectEnd = collectEnd;
loadOpts.minFiringRate = 0.1;
loadOpts.maxFiringRate = 150;

windowRows = [];
sessionStats = [];

for s = 1:numSessions
  sessionType = sessionTable.sessionType{s};
  sessionName = sessionTable.sessionName{s};
  subjectName = sessionTable.subjectName{s};
  fprintf('\n--- %d/%d [%s] %s ---\n', s, numSessions, sessionType, sessionName);

  try
    loadArgs = build_session_load_args(sessionType, sessionName, loadOpts, subjectName);
    dataStruct = load_session_data(sessionType, 'spikes', loadArgs{:});
    [dataStruct, areaOk] = apply_manuscript_brain_area_selection( ...
      dataStruct, brainArea, brainAreaCombinations);
    if ~areaOk
      fprintf('  Brain area "%s" not available; skipping.\n', brainArea);
      continue;
    end

    collectStartUsed = collectStart;
    collectEndUsed = collectEnd;
    if isfield(dataStruct, 'spikeData') && isfield(dataStruct.spikeData, 'collectEnd') ...
        && ~isempty(dataStruct.spikeData.collectEnd)
      collectStartUsed = dataStruct.spikeData.collectStart;
      collectEndUsed = dataStruct.spikeData.collectEnd;
    elseif isempty(collectEndUsed)
      collectEndUsed = max(dataStruct.spikeTimes);
    end

    [eventTimes, rewardTimes] = load_task_event_and_reward_times( ...
      sessionType, paths, subjectName, sessionName, dataStruct, minLeaveSec);
    eventInCollect = eventTimes >= collectStartUsed & eventTimes <= collectEndUsed;
    eventTimes = eventTimes(eventInCollect);
    rewardInCollect = rewardTimes >= collectStartUsed & rewardTimes <= collectEndUsed;
    rewardTimes = rewardTimes(rewardInCollect);
    fprintf('  Events: %d; rewards: %d\n', numel(eventTimes), numel(rewardTimes));

    sessionConfig = analysisConfig;
    sessionDuration = collectEndUsed - collectStartUsed;
    if sessionDuration < (d2Window - 1)
      fprintf('  Session %.1f s < d2Window; using full session window.\n', sessionDuration);
      sessionConfig.slidingWindowSize = sessionDuration;
      sessionConfig.stepSize = sessionDuration;
    end
    winLenSec = sessionConfig.slidingWindowSize;

    arResults = criticality_ar_analysis(dataStruct, sessionConfig);
    if ~isempty(brainArea)
      arResults = filter_ar_results_to_brain_area(arResults, brainArea);
    end
    if isempty(arResults.areas) || isempty(arResults.d2) || isempty(arResults.d2{1})
      fprintf('  No d2 windows; skipping.\n');
      continue;
    end

    sessRows = windows_for_session(arResults, winLenSec, eventTimes, rewardTimes, ...
      collectStartUsed, collectEndUsed, engagementBufferBefore, engagementBufferAfter, ...
      minNonEngagedWindow, absorbSingleEvents, useLog10D2, ...
      sessionType, sessionName, subjectName, s);
    if isempty(sessRows)
      fprintf('  No finite d2 windows; skipping.\n');
      continue;
    end
    nEng = sum([sessRows.isEngaged]);
    fprintf('  Windows: %d (%d engaged); median rewards at window end = %.0f\n', ...
      numel(sessRows), nEng, median([sessRows.nRewards]));

    windowRows = [windowRows; sessRows(:)]; %#ok<AGROW>
    sessionStats = [sessionStats; session_predictor_stats(sessRows)]; %#ok<AGROW>
  catch ME
    if is_skippable_session_error(ME)
      fprintf('  Skip: %s\n', ME.message);
      continue;
    end
    rethrow(ME);
  end
end

if isempty(windowRows)
  error('criticality_d2_vs_satiation:NoWindows', 'No valid d2 windows were collected.');
end

pooledStats = compare_reward_vs_engagement(windowRows, true);
print_predictor_comparison(pooledStats, 'Pooled windows (session-mean-centered OLS)');
print_per_session_rho_summary(sessionStats);

results = struct();
results.sessionTypes = sessionTypes;
results.windowRows = windowRows;
results.sessionStats = sessionStats;
results.pooledStats = pooledStats;
results.d2Window = d2Window;
results.brainArea = brainArea;
results.useLog10D2 = useLog10D2;
results.splitByEngagement = splitByEngagement;
results.engagementBufferBefore = engagementBufferBefore;
results.engagementBufferAfter = engagementBufferAfter;
results.minNonEngagedWindow = minNonEngagedWindow;
results.absorbSingleEvents = absorbSingleEvents;

if makePlots
  fig = plot_d2_vs_reward_and_engagement(windowRows, pooledStats, sessionTypes, ...
    splitByEngagement, useLog10D2, brainArea, d2Window, plotConfig);
  if saveFigure && isgraphics(fig)
    saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
    if ~exist(saveDir, 'dir')
      mkdir(saveDir);
    end
    areaTag = matlab.lang.makeValidName(char(string(brainArea)));
    if useLog10D2
      logTag = '_log10';
    else
      logTag = '';
    end
    if useSubsampling
      sampTag = sprintf('_subsamp%d', nNeuronsSubsample);
    else
      sampTag = '';
    end
    plotBase = fullfile(saveDir, sprintf( ...
      'criticality_d2_vs_satiation_%s_win%.0fs%s%s', ...
      areaTag, d2Window, sampTag, logTag));
    exportgraphics(fig, [plotBase, '.png'], 'Resolution', 300);
    exportgraphics(fig, [plotBase, '.eps'], 'ContentType', 'vector');
    fprintf('Saved figure: %s\n', plotBase);
  end
  if closeFigure && isgraphics(fig)
    close(fig);
  end
end

fprintf('\n=== criticality_d2_vs_satiation: done ===\n');

%% Local functions

function sessionTypes = filter_to_task_session_types(sessionTypes)
% FILTER_TO_TASK_SESSION_TYPES - Keep interval / semicircle / reach only

keepMask = false(size(sessionTypes));
for i = 1:numel(sessionTypes)
  if is_manuscript_engagement_session_type(sessionTypes{i})
    keepMask(i) = true;
  else
    fprintf('Dropping %s (no task rewards / engagement events).\n', sessionTypes{i});
  end
end
sessionTypes = sessionTypes(keepMask);
end

function sessionTable = build_task_session_table(sessionTypes)
% BUILD_TASK_SESSION_TABLE - Flatten manuscript session lists

sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
for t = 1:numel(sessionTypes)
  entries = manuscript_sessions_for_type(sessionTypes{t});
  for i = 1:numel(entries)
    sessionTypeCol{end + 1, 1} = sessionTypes{t}; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = entries(i).sessionName; %#ok<AGROW>
    if isfield(entries, 'subjectName')
      subjectNameCol{end + 1, 1} = entries(i).subjectName; %#ok<AGROW>
    else
      subjectNameCol{end + 1, 1} = ''; %#ok<AGROW>
    end
  end
end
sessionTable = table(sessionTypeCol, sessionNameCol, subjectNameCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'subjectName'});
end

function analysisConfig = build_d2_analysis_config(d2Window, binSize, useLog10D2, ...
    useSubsampling, nSubsamples, nNeuronsSubsample, minNeuronsMultiple, nMinNeurons)
% BUILD_D2_ANALYSIS_CONFIG - Non-overlapping AR / d2 settings

analysisConfig = struct();
analysisConfig.slidingWindowSize = d2Window;
analysisConfig.stepSize = d2Window;
analysisConfig.binSize = binSize;
analysisConfig.useOptimalBinWindowFunction = false;
analysisConfig.analyzeD2 = true;
analysisConfig.analyzeMrBr = false;
analysisConfig.pcaFlag = 0;
analysisConfig.pcaFirstFlag = 1;
analysisConfig.nDim = 4;
analysisConfig.enablePermutations = false;
analysisConfig.nShuffles = 1;
analysisConfig.normalizeD2 = false;
analysisConfig.useLog10D2 = useLog10D2;
analysisConfig.makePlots = false;
analysisConfig.saveData = false;
analysisConfig.pOrder = 10;
analysisConfig.critType = 2;
analysisConfig.minSpikesPerBin = 2.5;
analysisConfig.minBinsPerWindow = 1000;
analysisConfig.maxSpikesPerBin = 100;
analysisConfig.nMinNeurons = nMinNeurons;
analysisConfig.useSubsampling = useSubsampling;
analysisConfig.nSubsamples = nSubsamples;
analysisConfig.nNeuronsSubsample = nNeuronsSubsample;
analysisConfig.minNeuronsMultiple = minNeuronsMultiple;
end

function [eventTimes, rewardTimes] = load_task_event_and_reward_times( ...
    sessionType, paths, subjectName, sessionName, dataStruct, minLeaveSec)
% LOAD_TASK_EVENT_AND_REWARD_TIMES - Engagement events and rewarded outcomes
%
% Goal:
%   Events = all task responses used for engagement (reaches, interval beam
%   breaks, or semicircle TaskMatrix times: trial start, choice poke,
%   leave/enter home).
%   Rewards = successful / rewarded outcomes only (satiation count).

sessionType = lower(strtrim(char(sessionType)));
if strcmp(sessionType, 'reach')
  eventTimes = dataStruct.reachStart(:);
  reachClass = dataStruct.reachClass(:);
  nUse = min(numel(eventTimes), numel(reachClass));
  eventTimes = eventTimes(1:nUse);
  reachClass = reachClass(1:nUse);
  rewardTimes = eventTimes(ismember(reachClass, [2, 4, 6]));
elseif strcmp(sessionType, 'semicircle')
  [eventTimes, eventTypes] = load_semicircle_beam_break_events( ...
    paths, subjectName, sessionName);
  rewardTimes = eventTimes(eventTypes == "correct");
elseif strcmp(sessionType, 'interval')
  [eventTimes, eventTypes] = load_interval_beam_break_events( ...
    paths, subjectName, sessionName, minLeaveSec);
  rewardTimes = eventTimes(eventTypes == "correct");
else
  error('load_task_event_and_reward_times:BadType', 'Unsupported type %s', sessionType);
end
eventTimes = eventTimes(isfinite(eventTimes));
rewardTimes = rewardTimes(isfinite(rewardTimes));
end

function rows = windows_for_session(arResults, winLenSec, eventTimes, rewardTimes, ...
    collectStart, collectEnd, bufferBefore, bufferAfter, minNonEngagedWindow, ...
    absorbSingleEvents, useLog10D2, sessionType, sessionName, subjectName, sessionIdx)
% WINDOWS_FOR_SESSION - Per-window d2, cumulative rewards, engagement fraction
%
% startS from criticality_ar_analysis is absolute window-center time (s).
% engagementFrac = fraction of window overlapping continuous engaged segments.
% isEngaged = window fully inside an engaged segment (same as engagement d2).

rows = struct('sessionIdx', {}, 'sessionType', {}, 'sessionName', {}, ...
  'subjectName', {}, 'd2', {}, 'nRewards', {}, 'engagementFrac', {}, 'isEngaged', {}, ...
  'winStart', {}, 'winEnd', {});

centerAbs = arResults.startS{1}(:);
d2Vec = arResults.d2{1}(:);
nWin = min(numel(centerAbs), numel(d2Vec));
if nWin < 1
  return;
end
centerAbs = centerAbs(1:nWin);
d2Vec = d2Vec(1:nWin);
if useLog10D2
  d2Vec = log10_safe_numeric(d2Vec);
end

winStartAbs = centerAbs - winLenSec / 2;
winEndAbs = centerAbs + winLenSec / 2;
[engagedSegs, ~] = define_reach_engagement_segments( ...
  collectStart, collectEnd, eventTimes, minNonEngagedWindow, bufferBefore, bufferAfter, ...
  absorbSingleEvents);
engFrac = window_segment_overlap_fraction(winStartAbs, winEndAbs, engagedSegs);
isEngaged = window_fully_inside_segments(winStartAbs, winEndAbs, engagedSegs);

rewardTimes = sort(rewardTimes(:));
for w = 1:nWin
  if ~isfinite(d2Vec(w))
    continue;
  end
  row = struct();
  row.sessionIdx = sessionIdx;
  row.sessionType = sessionType;
  row.sessionName = sessionName;
  row.subjectName = subjectName;
  row.d2 = d2Vec(w);
  row.nRewards = sum(rewardTimes <= winEndAbs(w));
  row.engagementFrac = engFrac(w);
  row.isEngaged = logical(isEngaged(w));
  row.winStart = winStartAbs(w);
  row.winEnd = winEndAbs(w);
  rows(end + 1) = row; %#ok<AGROW>
end
end

function engFrac = window_segment_overlap_fraction(winStartAbs, winEndAbs, segs)
% WINDOW_SEGMENT_OVERLAP_FRACTION - Fraction of each window in segment union

engFrac = zeros(size(winStartAbs));
if isempty(segs)
  return;
end
for w = 1:numel(winStartAbs)
  winDur = winEndAbs(w) - winStartAbs(w);
  if winDur <= 0
    continue;
  end
  overlapSec = 0;
  for iSeg = 1:numel(segs)
    overlapSec = overlapSec + max(0, ...
      min(winEndAbs(w), segs(iSeg).end) - max(winStartAbs(w), segs(iSeg).start));
  end
  engFrac(w) = min(1, overlapSec / winDur);
end
end

function isInside = window_fully_inside_segments(winStartAbs, winEndAbs, segments)
% WINDOW_FULLY_INSIDE_SEGMENTS - True if [winStart, winEnd] lies in one segment

isInside = false(size(winStartAbs));
if isempty(segments)
  return;
end
segStarts = [segments.start];
segEnds = [segments.end];
for w = 1:numel(winStartAbs)
  isInside(w) = any(winStartAbs(w) >= segStarts & winEndAbs(w) <= segEnds);
end
end

function [engagedSegs, nonEngagedSegs] = define_reach_engagement_segments( ...
    collectStart, collectEnd, eventTimes, minNonEngagedWindow, bufferBefore, bufferAfter, ...
    absorbSingleEvents)
% DEFINE_REACH_ENGAGEMENT_SEGMENTS - Continuous engaged / non-engaged intervals
%
% Variables:
%   collectStart, collectEnd - Session analysis bounds (s)
%   eventTimes               - Beam-break / reach times in collect window (s)
%   minNonEngagedWindow      - Minimum gap without events (s)
%   bufferBefore, bufferAfter - Event neighborhood excluded from non-engaged gaps
%   absorbSingleEvents       - Merge isolated single events into non-engaged gaps
%
% Goal:
%   Non-engaged = gaps >= minNonEngagedWindow without events (after buffers).
%   Engaged = complement (includes all event buffers and short gaps).

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

occupied = merge_event_buffer_intervals( ...
  eventTimes, bufferBefore, bufferAfter, collectStart, collectEnd);
absorbedMask = false(1, numel(occupied));
if absorbSingleEvents && ~isempty(occupied)
  absorbedMask = get_absorbed_single_event_occupied_mask( ...
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

function absorbedMask = get_absorbed_single_event_occupied_mask( ...
    eventTimes, collectStart, collectEnd, minNonEngagedWindow, bufferBefore, bufferAfter)
% GET_ABSORBED_SINGLE_EVENT_OCCUPIED_MASK - Isolated single events to absorb

if nargin < 6 || isempty(bufferAfter)
  bufferAfter = bufferBefore;
end
occupied = merge_event_buffer_intervals( ...
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

function occupied = merge_event_buffer_intervals(eventTimes, bufferBefore, bufferAfter, ...
    collectStart, collectEnd)
% MERGE_EVENT_BUFFER_INTERVALS - Union of [event-before, event+after] in collect window

if nargin < 3 || isempty(bufferAfter)
  bufferAfter = bufferBefore;
end
bufferBefore = max(0, bufferBefore);
bufferAfter = max(0, bufferAfter);

occupied = struct('start', {}, 'end', {});
if isempty(eventTimes)
  return;
end

starts = max(collectStart, eventTimes(:) - bufferBefore);
ends = min(collectEnd, eventTimes(:) + bufferAfter);
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

function stats = session_predictor_stats(sessRows)
% SESSION_PREDICTOR_STATS - Within-session Spearman rhos

d2 = [sessRows.d2]';
nRewards = [sessRows.nRewards]';
engFrac = [sessRows.engagementFrac]';
stats = struct();
stats.sessionType = sessRows(1).sessionType;
stats.sessionName = sessRows(1).sessionName;
stats.nWindows = numel(d2);
[stats.rhoReward, stats.pReward] = spearman_corr(d2, nRewards);
[stats.rhoEngagement, stats.pEngagement] = spearman_corr(d2, engFrac);
end

function stats = compare_reward_vs_engagement(windowRows, meanCenterBySession)
% COMPARE_REWARD_VS_ENGAGEMENT - Which predictor uniquely tracks d2
%
% Goal:
%   Spearman (zero-order + partial), Steiger test of dependent correlations,
%   and unique R^2 from OLS. Optional session-mean-centering for regression
%   so cumulative reward is not just "later in longer sessions".

if nargin < 2 || isempty(meanCenterBySession)
  meanCenterBySession = true;
end

d2 = [windowRows.d2]';
nRewards = double([windowRows.nRewards]');
engFrac = [windowRows.engagementFrac]';
sessionIdx = [windowRows.sessionIdx]';

stats = struct();
stats.nWindows = numel(d2);
[stats.rhoReward, stats.pReward] = spearman_corr(d2, nRewards);
[stats.rhoEngagement, stats.pEngagement] = spearman_corr(d2, engFrac);
[stats.rhoRewardEng] = spearman_corr(nRewards, engFrac);
[stats.steigerZ, stats.steigerP] = steiger_z_dependent_corr( ...
  stats.rhoEngagement, stats.rhoReward, stats.rhoRewardEng, stats.nWindows);

[stats.partialRhoRewardGivenEng, stats.partialPRewardGivenEng] = ...
  spearman_partial_corr(d2, nRewards, engFrac);
[stats.partialRhoEngGivenReward, stats.partialPEngGivenReward] = ...
  spearman_partial_corr(d2, engFrac, nRewards);

d2Reg = d2;
rewReg = nRewards;
engReg = engFrac;
if meanCenterBySession
  [d2Reg, rewReg, engReg] = mean_center_by_session(d2, nRewards, engFrac, sessionIdx);
end
ols = ols_unique_r_squared(d2Reg, rewReg, engReg);
stats.betaReward = ols.betaReward;
stats.betaEngagement = ols.betaEngagement;
stats.r2RewardOnly = ols.r2RewardOnly;
stats.r2EngagementOnly = ols.r2EngagementOnly;
stats.r2Full = ols.r2Full;
stats.uniqueR2Reward = ols.uniqueR2Reward;
stats.uniqueR2Engagement = ols.uniqueR2Engagement;
stats.meanCentered = meanCenterBySession;
end

function print_predictor_comparison(stats, titleStr)
% PRINT_PREDICTOR_COMPARISON - Console summary of satiation vs engagement

fprintf('\n--- %s ---\n', titleStr);
fprintf('n windows = %d\n', stats.nWindows);
fprintf('Spearman  d2 vs cumulative rewards:  rho = % .3f  p = %.3g\n', ...
  stats.rhoReward, stats.pReward);
fprintf('Spearman  d2 vs engagement fraction: rho = % .3f  p = %.3g\n', ...
  stats.rhoEngagement, stats.pEngagement);
fprintf('Spearman  rewards vs engagement:     rho = % .3f\n', stats.rhoRewardEng);
fprintf('Steiger Z ( |rho_eng| vs |rho_reward| direction: eng - reward ): Z = %.2f  p = %.3g\n', ...
  stats.steigerZ, stats.steigerP);
fprintf('Partial Spearman  d2 vs reward | engagement:    rho = % .3f  p = %.3g\n', ...
  stats.partialRhoRewardGivenEng, stats.partialPRewardGivenEng);
fprintf('Partial Spearman  d2 vs engagement | reward:    rho = % .3f  p = %.3g\n', ...
  stats.partialRhoEngGivenReward, stats.partialPEngGivenReward);
fprintf('OLS unique R^2 (session-centered=%d): engagement = %.3f; reward = %.3f; full = %.3f\n', ...
  stats.meanCentered, stats.uniqueR2Engagement, stats.uniqueR2Reward, stats.r2Full);
if abs(stats.uniqueR2Engagement) >= abs(stats.uniqueR2Reward) ...
    && abs(stats.partialRhoEngGivenReward) >= abs(stats.partialRhoRewardGivenEng)
  fprintf(['Interpretation: engagement uniquely accounts for at least as much d2 ', ...
    'variance as cumulative reward (satiation), after controlling for the other.\n']);
else
  fprintf(['Interpretation: cumulative reward is not clearly weaker than engagement ', ...
    'in this sample; inspect partial rhos and unique R^2.\n']);
end
end

function print_per_session_rho_summary(sessionStats)
% PRINT_PER_SESSION_RHO_SUMMARY - Mean within-session Spearman rhos

if isempty(sessionStats)
  return;
end
rhoR = [sessionStats.rhoReward]';
rhoE = [sessionStats.rhoEngagement]';
valid = isfinite(rhoR) & isfinite(rhoE);
rhoR = rhoR(valid);
rhoE = rhoE(valid);
fprintf('\n--- Within-session Spearman (n sessions = %d) ---\n', numel(rhoR));
if isempty(rhoR)
  return;
end
fprintf('mean rho(d2, reward)      = % .3f  (SEM %.3f)\n', mean(rhoR), std(rhoR) / sqrt(numel(rhoR)));
fprintf('mean rho(d2, engagement)  = % .3f  (SEM %.3f)\n', mean(rhoE), std(rhoE) / sqrt(numel(rhoE)));
if numel(rhoR) >= 6
  pWil = signrank(rhoE, rhoR);
  fprintf('Wilcoxon signed-rank (rho_eng vs rho_reward): p = %.3g\n', pWil);
end
end

function fig = plot_d2_vs_reward_and_engagement(windowRows, pooledStats, sessionTypes, ...
    splitByEngagement, useLog10D2, brainArea, d2Window, plotConfig)
% PLOT_D2_VS_REWARD_AND_ENGAGEMENT - Per-task  n x 2  scatters (reward | engagement)

d2 = [windowRows.d2]';
nRewards = double([windowRows.nRewards]');
engFrac = [windowRows.engagementFrac]';
isEngaged = [windowRows.isEngaged]';
types = {windowRows.sessionType};

plotTypes = order_manuscript_session_types(sessionTypes);
keepType = false(size(plotTypes));
for t = 1:numel(plotTypes)
  keepType(t) = any(strcmpi(types, plotTypes{t}));
end
plotTypes = plotTypes(keepType);
nTasks = numel(plotTypes);
if nTasks < 1
  error('criticality_d2_vs_satiation:NoTaskData', 'No task windows to plot.');
end

fig = figure('Color', 'w', 'Name', 'd2 vs satiation vs engagement', 'Units', 'pixels');
screenSize = get(0, 'ScreenSize');
figHeightPx = round(min(0.90 * screenSize(4), 0.32 * screenSize(4) * nTasks + 80));
set(fig, 'Position', [round(0.06 * screenSize(3)), round(0.08 * screenSize(4)), ...
  round(0.88 * screenSize(3)), figHeightPx]);
tl = tiledlayout(fig, nTasks, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

d2YLabel = get_d2_axis_label(useLog10D2);
if useLog10D2
  labelInterp = 'tex';
else
  labelInterp = 'none';
end
yLim = padded_ylim(d2);

for t = 1:nTasks
  typeName = plotTypes{t};
  typeColor = colors_for_tasks(typeName);
  typeMask = strcmpi(types, typeName);
  d2T = d2(typeMask);
  rewT = nRewards(typeMask);
  engT = engFrac(typeMask);
  isEngT = isEngaged(typeMask);
  [rhoRew, pRew] = spearman_corr(d2T, rewT);
  [rhoEng, pEng] = spearman_corr(d2T, engT);

  axReward = nexttile(tl);
  hold(axReward, 'on');
  scatter_one_task(axReward, rewT, d2T, isEngT, typeColor, typeName, ...
    splitByEngagement, plotConfig);
  add_manuscript_scatter_trendline(axReward, rewT, d2T, plotConfig);
  if ~isempty(yLim)
    ylim(axReward, yLim);
  end
  apply_manuscript_axes_style(axReward, plotConfig, ...
    'Cumulative rewards (through window end)', d2YLabel, ...
    sprintf('%s  |  satiation  \\rho=%.2f (p=%.2g)', typeName, rhoRew, pRew), ...
    labelInterp);
  grid(axReward, 'on');
  if splitByEngagement && t == 1
    legend(axReward, 'Location', 'best', 'FontSize', plotConfig.legendFontSize);
  end
  hold(axReward, 'off');

  axEng = nexttile(tl);
  hold(axEng, 'on');
  scatter_one_task(axEng, engT, d2T, isEngT, typeColor, typeName, ...
    splitByEngagement, plotConfig);
  add_manuscript_scatter_trendline(axEng, engT, d2T, plotConfig);
  if ~isempty(yLim)
    ylim(axEng, yLim);
  end
  xlim(axEng, [-0.02, 1.02]);
  apply_manuscript_axes_style(axEng, plotConfig, 'Engagement fraction of window', ...
    '', sprintf('%s  |  engagement  \\rho=%.2f (p=%.2g)', typeName, rhoEng, pEng), ...
    'none');
  grid(axEng, 'on');
  hold(axEng, 'off');
end

sgtitle(tl, sprintf( ...
  ['%s  |  %.0fs windows  |  pooled partial  reward|eng \\rho=%.2f,  eng|reward \\rho=%.2f', ...
  '  |  unique R^2  eng=%.3f  reward=%.3f'], ...
  brainArea, d2Window, pooledStats.partialRhoRewardGivenEng, ...
  pooledStats.partialRhoEngGivenReward, pooledStats.uniqueR2Engagement, ...
  pooledStats.uniqueR2Reward), ...
  'FontSize', plotConfig.sgtitleFontSize, 'Interpreter', 'tex');
end

function scatter_one_task(ax, xVals, d2, isEngaged, typeColor, typeName, ...
    splitByEngagement, plotConfig)
% SCATTER_ONE_TASK - One task's windows; filled = engaged if split

if splitByEngagement
  if any(isEngaged)
    scatter_manuscript_filled(ax, xVals(isEngaged), d2(isEngaged), ...
      plotConfig, typeColor, 'engaged');
  end
  if any(~isEngaged)
    scatter_manuscript_open(ax, xVals(~isEngaged), d2(~isEngaged), ...
      plotConfig, typeColor, 'non-engaged');
  end
else
  scatter_manuscript_filled(ax, xVals, d2, plotConfig, typeColor, typeName);
end
end

function yLim = padded_ylim(vals)
% PADDED_YLIM - [ymin ymax] with small padding

yLim = [];
vals = vals(isfinite(vals));
if isempty(vals)
  return;
end
yMin = min(vals);
yMax = max(vals);
if yMax > yMin
  pad = 0.05 * (yMax - yMin);
else
  pad = max(0.05 * abs(yMax), 1e-6);
end
yLim = [yMin - pad, yMax + pad];
end

function yLabelText = get_d2_axis_label(useLog10D2)
if useLog10D2
  yLabelText = 'log_{10}(d2)';
else
  yLabelText = 'd2';
end
end

function y = log10_safe_numeric(x)
validMask = isfinite(x) & x > 0;
y = nan(size(x));
y(validMask) = log10(x(validMask));
end

function [rho, pVal] = spearman_corr(x, y)
% SPEARMAN_CORR - Spearman rho; NaN if too few finite pairs

rho = nan;
pVal = nan;
ok = isfinite(x) & isfinite(y);
if sum(ok) < 5
  return;
end
[rho, pVal] = corr(x(ok), y(ok), 'Type', 'Spearman');
end

function [rho, pVal] = spearman_partial_corr(y, x, z)
% SPEARMAN_PARTIAL_CORR - Spearman partial correlation y vs x | z

rho = nan;
pVal = nan;
ok = isfinite(y) & isfinite(x) & isfinite(z);
if sum(ok) < 6
  return;
end
[rho, pVal] = partialcorr(y(ok), x(ok), z(ok), 'Type', 'Spearman');
end

function [zVal, pVal] = steiger_z_dependent_corr(rXy, rXz, rYz, nObs)
% STEIGER_Z_DEPENDENT_CORR - Compare r(y,x) vs r(y,z) sharing y
%
% Steiger (1980) test of two dependent correlations. Here x = engagement,
% z = reward, y = d2, so positive Z means engagement correlates more with d2.

zVal = nan;
pVal = nan;
if nObs < 6 || ~all(isfinite([rXy, rXz, rYz]))
  return;
end
rXy = max(min(rXy, 0.999), -0.999);
rXz = max(min(rXz, 0.999), -0.999);
rYz = max(min(rYz, 0.999), -0.999);
meanR2 = (rXy^2 + rXz^2) / 2;
if abs(1 - meanR2) < 1e-9
  return;
end
fVal = (1 - rYz) / (2 * (1 - meanR2));
hVal = (1 - fVal * meanR2) / (1 - meanR2);
denom = 2 * (1 - rYz) * hVal;
if denom <= 0
  return;
end
zVal = (rXy - rXz) * sqrt((nObs - 3) / denom);
pVal = 2 * (1 - 0.5 * (1 + erf(abs(zVal) / sqrt(2))));
end

function [yC, x1C, x2C] = mean_center_by_session(y, x1, x2, sessionIdx)
% MEAN_CENTER_BY_SESSION - Subtract per-session means

yC = nan(size(y));
x1C = nan(size(x1));
x2C = nan(size(x2));
uSess = unique(sessionIdx);
for i = 1:numel(uSess)
  m = sessionIdx == uSess(i);
  yC(m) = y(m) - mean(y(m), 'omitnan');
  x1C(m) = x1(m) - mean(x1(m), 'omitnan');
  x2C(m) = x2(m) - mean(x2(m), 'omitnan');
end
end

function ols = ols_unique_r_squared(y, xReward, xEng)
% OLS_UNIQUE_R_SQUARED - Standardized unique R^2 for two predictors

ols = struct('betaReward', nan, 'betaEngagement', nan, ...
  'r2RewardOnly', nan, 'r2EngagementOnly', nan, 'r2Full', nan, ...
  'uniqueR2Reward', nan, 'uniqueR2Engagement', nan);
ok = isfinite(y) & isfinite(xReward) & isfinite(xEng);
if sum(ok) < 8
  return;
end
y = y(ok);
xReward = xReward(ok);
xEng = xEng(ok);
y = y - mean(y);
xReward = xReward - mean(xReward);
xEng = xEng - mean(xEng);
ssY = sum(y.^2);
if ssY <= 0
  return;
end
bRew = xReward \ y;
bEng = xEng \ y;
bFull = [xReward, xEng] \ y;
ols.r2RewardOnly = 1 - sum((y - xReward * bRew).^2) / ssY;
ols.r2EngagementOnly = 1 - sum((y - xEng * bEng).^2) / ssY;
residFull = y - [xReward, xEng] * bFull;
ols.r2Full = 1 - sum(residFull.^2) / ssY;
ols.uniqueR2Reward = max(0, ols.r2Full - ols.r2EngagementOnly);
ols.uniqueR2Engagement = max(0, ols.r2Full - ols.r2RewardOnly);
sdY = std(y);
if sdY > 0
  ols.betaReward = bFull(1) * std(xReward) / sdY;
  ols.betaEngagement = bFull(2) * std(xEng) / sdY;
end
end

function results = filter_ar_results_to_brain_area(results, brainArea)
% FILTER_AR_RESULTS_TO_BRAIN_AREA - Keep one area in AR results struct

if isempty(brainArea) || ~isfield(results, 'areas')
  return;
end
areaIdx = find(strcmp(results.areas, brainArea), 1);
if isempty(areaIdx)
  results.areas = {};
  return;
end
cellFields = {'d2', 'd2Normalized', 'startS', 'd2Permuted', 'mrBrPermuted', ...
  'd2PermutedMean', 'd2PermutedSEM', 'popActivityWindows', 'popActivityFull', ...
  'd2Subsamples', 'd2NormalizedSubsamples'};
results.areas = results.areas(areaIdx);
for f = 1:numel(cellFields)
  fieldName = cellFields{f};
  if isfield(results, fieldName) && numel(results.(fieldName)) >= areaIdx
    results.(fieldName) = results.(fieldName)(areaIdx);
  end
end
end

function tf = is_skippable_session_error(ME)
msg = ME.message;
id = ME.identifier;
tf = contains(msg, 'No valid areas to process') ...
  || contains(msg, 'insufficient neurons') ...
  || contains(msg, 'for subsampling') ...
  || contains(msg, 'not available') ...
  || contains(id, 'TooFewNeurons');
end

function [eventTimes, eventTypes] = load_interval_beam_break_events( ...
    paths, subjectName, sessionName, minLeaveSec)
% LOAD_INTERVAL_BEAM_BREAK_EVENTS - Correct/error times from interval CSV

sessionDir = fullfile(paths.intervalDataPath, subjectName, sessionName);
if ~exist(sessionDir, 'dir')
  error('criticality_d2_vs_satiation:SessionNotFound', ...
    'Interval session directory not found: %s', sessionDir);
end
csvFiles = dir(fullfile(sessionDir, 'revised_interval_*.csv'));
if isempty(csvFiles)
  error('criticality_d2_vs_satiation:NoCsv', ...
    'No revised_interval_*.csv in %s', sessionDir);
end
[~, newestIdx] = max([csvFiles.datenum]);
csvPath = fullfile(sessionDir, csvFiles(newestIdx).name);
rawTable = readtable(csvPath, 'TextType', 'string');
varNames = lower(string(rawTable.Properties.VariableNames));
timeCol = find(contains(varNames, 'timestamp'), 1);
eventCol = find(strcmp(varNames, 'event') | contains(varNames, 'event'), 1);
valueCol = find(strcmp(varNames, 'value') | contains(varNames, 'value'), 1);
timestampMs = double(rawTable{:, timeCol});
eventNames = string(rawTable{:, eventCol});
eventValues = double(rawTable{:, valueCol});
validRows = ~isnan(timestampMs) & eventNames ~= "" & ~ismissing(eventNames);
logTable = table(timestampMs(validRows), eventNames(validRows), eventValues(validRows), ...
  'VariableNames', {'timestampMs', 'event', 'value'});
logTable = sortrows(logTable, 'timestampMs');
trials = extract_interval_trials(logTable, minLeaveSec);
eventTimes = trials.outcomeTimeSec(:);
eventTypes = trials.type(:);
[eventTimes, ord] = sort(eventTimes);
eventTypes = eventTypes(ord);
end

function trials = extract_interval_trials(logTable, minLeaveSec)
% EXTRACT_INTERVAL_TRIALS - ERROR / REWARD outcomes (interval_session_performance)

minLeaveMs = minLeaveSec * 1000;
leavePending = false;
initialExitMs = NaN;
leaveTimeMs = NaN;
timerArmed = false;
beamState = 0;
firstRewardSeen = false;
sessionOriginMs = min(logTable.timestampMs);
trialTypes = strings(0, 1);
pokeTimesSec = [];
outcomeTimesSec = [];
eventNames = logTable.event;
timestampsMs = logTable.timestampMs;
eventValues = logTable.value;
leaveConfirmStartMs = NaN;

for eventIdx = 1:height(logTable)
  eventTimeMs = timestampsMs(eventIdx);
  eventName = eventNames(eventIdx);
  eventValue = eventValues(eventIdx);
  if leavePending && beamState == 0 && (eventTimeMs - leaveConfirmStartMs) >= minLeaveMs
    leaveTimeMs = initialExitMs;
    timerArmed = true;
    leavePending = false;
  end
  if eventName == "B"
    beamState = eventValue;
    if eventValue == 0
      leavePending = true;
      initialExitMs = eventTimeMs;
      leaveConfirmStartMs = eventTimeMs;
    else
      leavePending = false;
    end
  elseif eventName == "ERROR"
    if timerArmed && ~isnan(leaveTimeMs)
      trialTypes(end + 1, 1) = "error"; %#ok<AGROW>
      pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
      outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
    end
    timerArmed = false;
    leaveTimeMs = NaN;
  elseif eventName == "REWARD"
    if ~firstRewardSeen
      firstRewardSeen = true;
    elseif timerArmed && ~isnan(leaveTimeMs)
      trialTypes(end + 1, 1) = "correct"; %#ok<AGROW>
      pokeTimesSec(end + 1, 1) = (eventTimeMs - leaveTimeMs) / 1000; %#ok<AGROW>
      outcomeTimesSec(end + 1, 1) = (eventTimeMs - sessionOriginMs) / 1000; %#ok<AGROW>
    end
    timerArmed = false;
    leaveTimeMs = NaN;
  end
end
trials = table(trialTypes, pokeTimesSec, outcomeTimesSec, ...
  'VariableNames', {'type', 'pokeTimeSec', 'outcomeTimeSec'});
end
