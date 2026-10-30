%%
% Criticality LFP Spontaneous vs Task (Manuscript)
%
% Same session layout as criticality_spontaneous_vs_task.m for one subject:
% spontaneous sessions first, then that subject's sessions of the selected
% tasks. Signals are LFP, not spikes.
%
% The spike script is left unchanged. LFP d2 uses criticality_lfp_analysis,
% and avalanches use the Klaus negative-peak rule from session_lfp_criticality.
%
% Spontaneous sessions drop sleep (lfp_sleep_detector). d2 windows are tiled
% inside each awake bout, and each bout has its own negative-peak threshold.
% Task sessions, when splitByEngagement is true, are split the same way as
% criticality_spontaneous_vs_task: engaged and non-engaged segments only.
% d2 windows are center-packed inside each segment. Each segment has its own
% avalanche threshold. Plots use filled markers for non-engaged (and for
% spontaneous) and open circles for engaged.
%
% Plots:
%   Raw LFP (bandpass, default 1-100 Hz) — d2, tau, and alpha on one axes.
%     d2 is the session mean of window-wise raw-LFP d2. Tau and alpha come
%     from negative-peak avalanches and are scaled onto the d2 axis.
%     Color is the task color from colors_for_tasks.
%   Separated metrics — the same raw-LFP d2, tau, and alpha, each on its
%     own axes (native units). A horizontal line marks the within-task mean.
%   Band-power d2 — only when analyzeBandPower is true. One shared axis for
%     every band. Color is the band (same palette as session_lfp_criticality,
%     after the raw-LFP color). Marker shape is the task, so a band keeps
%     one color across tasks.
%
% Variables:
%   subjectName, taskTypes, collectStart, collectEnd, brainArea
%   d2Method    - 'euclidean' (Yule-Walker + getFixedPointDistance2) or 'kl'
%                 (Sooter et al. KL-rate d2). The LFP path has been Euclidean.
%   klFitMethod, klErrBars, klParallel - Used only when d2Method = 'kl'
%   d2Window, d2Step, rawD2BinSize, useLog10D2, pOrder, critType
%   lfpHighpassHz, lfpLowpassHz, nSdCutoff, bands, analyzeBandPower
%   splitByEngagement, engagementBufferBefore, engagementBufferAfter,
%   minNonEngagedWindow, absorbSingleEvents, minTimeNonEngaged
%   restrictToAwake - Spontaneous only: drop sleep (default true)
%   powerLawFitMethod, useSessionCache, forceRecompute,
%   forceRecomputeSessionTypes, forceRecomputeSubject, plotResults

%% Configuration
subjectName = 'ey9166';
taskTypes = {'interval', 'semicircle'};  % one or more of 'reach', 'interval', 'semicircle'
taskTypes = {'interval'};  % one or more of 'reach', 'interval', 'semicircle'

collectStart = 0;
collectEnd = 120 * 60;  % [] = through the end of the loaded LFP

brainArea = 'M56';
brainAreaCombinations = default_manuscript_brain_area_combinations();

d2Window = 30;
d2Step = 30;
rawD2BinSize = 0.005;  % 100 Hz lowpass -> 5 ms mean bins, snapped to samples
useLog10D2 = false;
d2Method = 'euclidean';         % 'euclidean' or 'kl'
% prox_crit_toolkit / Sooter et al. S2.5 (used when d2Method = 'kl')
klFitMethod = 'MaxLikelihood';  % required for error bars
klErrBars = false;
runParallel = true;
nWorkers = 5;  % used only when runParallel is true; capped at feature('numcores')
klParallel = runParallel;
pOrder = 10;
critType = 2;

lfpHighpassHz = 1;
lfpLowpassHz = 100;
nSdCutoff = 2.5;
bands = {'alpha', [8 13]; ...
  'beta', [13 30]; ...
  'lowGamma', [30 50]; ...
  'highGamma', [50 80]};
analyzeBandPower = true;  % band-power envelopes, d2, and the band plot

% Engagement timing matches criticality_spontaneous_vs_task
splitByEngagement = true;  % tasks: engaged (open) and non-engaged (filled)
engagementBufferBefore = 3;  % s before each engagement event
engagementBufferAfter = 1;   % s after each engagement event
minNonEngagedWindow = 30;    % min gap (s) for a non-engaged segment
absorbSingleEvents = true;   % merge isolated single events into non-engaged gaps
minTimeNonEngaged = 180;     % blank non-engaged metrics below this total (s); 0 = off
restrictToAwake = true;      % spontaneous: drop sleep

powerLawFitMethod = 'clauset';
compareTailModels = false;
runClausetPlpva = false;
gofThreshold = 0.8;
useSessionCache = true;
forceRecompute = false;   % true: reprocess and overwrite per-session cache
forceRecomputeSessionTypes = {};  % e.g. {'spontaneous'} recomputes only that type
forceRecomputeSubject = '';       % e.g. 'ey9166'; with session types set, only this subject
plotResults = true;
plotSeparatedMetrics = true;

% Setup
setup_criticality_manuscript_paths('criticality_lfp_spontaneous_vs_task');
paths = get_paths();

taskTypes = normalize_lfp_task_types(taskTypes);
if isempty(subjectName)
  error('Set subjectName to a subject in the spontaneous and task session lists.');
end
sessionTypes = order_manuscript_session_types([{'spontaneous'}, taskTypes]);

if isempty(collectStart)
  collectStart = 0;
end
if isempty(d2Window) || isempty(d2Step) || d2Window <= 0 || d2Step <= 0
  error('d2Window and d2Step must be positive durations in seconds.');
end
[d2Method, klFitMethod, klErrBars, klParallel] = normalize_kl_d2_options( ...
  d2Method, klFitMethod, klErrBars, klParallel);
if ~strcmp(d2Method, 'kl')
  runParallel = false;
end

plotConfig = fill_manuscript_plot_config();
taskFileTag = strjoin(cellfun(@(t) matlab.lang.makeValidName(char(t)), ...
  taskTypes, 'UniformOutput', false), '_');
plotConfig.fileTag = sprintf('lfp_spont_vs_%s_%s_d2%s', taskFileTag, ...
  matlab.lang.makeValidName(char(subjectName)), ...
  matlab.lang.makeValidName(char(d2Method)));
if strcmp(d2Method, 'kl')
  plotConfig.fileTag = sprintf('%s_%s', plotConfig.fileTag, ...
    matlab.lang.makeValidName(char(klFitMethod)));
  if klErrBars
    plotConfig.fileTag = [plotConfig.fileTag, '_klerr']; %#ok<AGROW>
  end
end

fprintf('\n=== LFP Spontaneous vs Task ===\n');
fprintf('Subject: %s\n', char(subjectName));
fprintf('Tasks: %s (spontaneous sessions plotted first)\n', strjoin(taskTypes, ', '));
fprintf('Area: %s\n', brainArea);
fprintf('d2Method: %s\n', d2Method);
if strcmp(d2Method, 'kl')
  fprintf('KL fit: %s; klErrBars=%d; klParallel=%d; nWorkers=%d\n', ...
    klFitMethod, klErrBars, klParallel, nWorkers);
end
fprintf('LFP bandpass: %.0f-%.0f Hz; raw d2 bin %.0f ms\n', ...
  lfpHighpassHz, lfpLowpassHz, rawD2BinSize * 1000);
fprintf('d2 windows: %.0f s, step %.0f s\n', d2Window, d2Step);
fprintf('Band-power envelopes: %d\n', analyzeBandPower);
fprintf('restrictToAwake: %d (spontaneous sleep removed when true)\n', restrictToAwake);
fprintf('splitByEngagement: %d\n', splitByEngagement);
if splitByEngagement
  fprintf(['engagementBuffer: before=%.3g s, after=%.3g s; minNonEngagedWindow: %.1f s; ', ...
    'absorbSingleEvents: %d\n'], engagementBufferBefore, engagementBufferAfter, ...
    minNonEngagedWindow, absorbSingleEvents);
  fprintf('minTimeNonEngaged: %.1f s (blank non-engaged below this; 0 = off)\n', ...
    minTimeNonEngaged);
  fprintf(['Task d2 windows are center-packed inside each engaged and ', ...
    'non-engaged segment. Each segment has its own avalanche threshold.\n']);
end
fprintf('Avalanches: negative peaks at %.1f SD (Klaus nLFP)\n', nSdCutoff);
if isempty(collectEnd)
  fprintf('Collect window: [%.1f, full] s\n', collectStart);
else
  fprintf('Collect window: [%.1f, %.1f] s\n', collectStart, collectEnd);
end

sessionTable = build_lfp_subject_session_table(sessionTypes, subjectName);
if ~any(strcmp(sessionTable.sessionType, 'spontaneous'))
  error('No spontaneous sessions for subject "%s".', subjectName);
end
fprintf('Sessions: %d\n', height(sessionTable));
fprintf('useSessionCache: %d; forceRecompute: %d\n', useSessionCache, forceRecompute);
if ~isempty(forceRecomputeSessionTypes)
  fprintf('forceRecomputeSessionTypes: %s\n', strjoin(cellstr(forceRecomputeSessionTypes), ', '));
end
if ~isempty(forceRecomputeSubject)
  fprintf('forceRecomputeSubject: %s\n', strjoin(cellstr(forceRecomputeSubject), ', '));
end

cacheOpts = struct( ...
  'useSessionCache', useSessionCache, ...
  'forceRecompute', forceRecompute, ...
  'forceRecomputeSessionTypes', {forceRecomputeSessionTypes}, ...
  'forceRecomputeSubject', {forceRecomputeSubject});

maybe_start_kl_d2_parallel_pool(d2Method, klErrBars, klParallel, nWorkers);

lfpCleanParams = struct();
lfpCleanParams.skipArtifactClean = true;
lfpCleanParams.highpassFreq = lfpHighpassHz;
lfpCleanParams.lowpassFreq = lfpLowpassHz;

[clausetPlfitPath, plfit2023Path] = resolve_power_law_paths();
fitConfig = struct();
fitConfig.powerLawFitMethod = powerLawFitMethod;
fitConfig.clausetPlfitPath = clausetPlfitPath;
fitConfig.plfit2023Path = plfit2023Path;
fitConfig.gofThreshold = gofThreshold;
fitConfig.runClausetPlpva = runClausetPlpva;
fitConfig.compareTailModels = compareTailModels;

cacheParams = struct();
cacheParams.brainArea = char(brainArea);
cacheParams.collectStart = collectStart;
cacheParams.collectEnd = collectEnd;
cacheParams.d2Window = d2Window;
cacheParams.d2Step = d2Step;
cacheParams.rawD2BinSize = rawD2BinSize;
cacheParams.useLog10D2 = logical(useLog10D2);
cacheParams.pOrder = pOrder;
cacheParams.critType = critType;
cacheParams.d2Method = d2Method;
cacheParams.klFitMethod = klFitMethod;
cacheParams.klErrBars = logical(klErrBars);
cacheParams.lfpHighpassHz = lfpHighpassHz;
cacheParams.lfpLowpassHz = lfpLowpassHz;
cacheParams.nSdCutoff = nSdCutoff;
cacheParams.bands = bands;
cacheParams.analyzeBandPower = logical(analyzeBandPower);
cacheParams.powerLawFitMethod = powerLawFitMethod;
cacheParams.compareTailModels = logical(compareTailModels);
cacheParams.restrictToAwake = logical(restrictToAwake);
cacheParams.splitByEngagement = logical(splitByEngagement);
if splitByEngagement
  cacheParams.engagementBufferBefore = engagementBufferBefore;
  cacheParams.engagementBufferAfter = engagementBufferAfter;
  cacheParams.minNonEngagedWindow = minNonEngagedWindow;
  cacheParams.absorbSingleEvents = logical(absorbSingleEvents);
  cacheParams.minTimeNonEngaged = minTimeNonEngaged;
else
  cacheParams.engagementBufferBefore = 0;
  cacheParams.engagementBufferAfter = 0;
  cacheParams.minNonEngagedWindow = 0;
  cacheParams.absorbSingleEvents = false;
  cacheParams.minTimeNonEngaged = 0;
end

nSessions = height(sessionTable);
sessionResults = repmat(empty_lfp_session_result(), nSessions, 1);

for iSession = 1:nSessions
  sessionType = sessionTable.sessionType{iSession};
  sessionName = sessionTable.sessionName{iSession};
  subjectForSession = sessionTable.subjectName{iSession};
  fprintf('\n%s\n', repmat('=', 1, 72));
  fprintf('Session %d/%d [%s]: %s\n', iSession, nSessions, sessionType, sessionName);

  sessionResults(iSession).sessionType = sessionType;
  sessionResults(iSession).sessionName = sessionName;
  sessionResults(iSession).subjectName = subjectForSession;
  sessionResults(iSession).label = sessionTable.label{iSession};

  cacheSubject = subjectForSession;
  if isempty(cacheSubject)
    cacheSubject = subjectName;
  end
  if ~skip_manuscript_session_cache_load(cacheOpts, sessionType, cacheSubject)
    cached = try_load_lfp_svt_cache(paths, subjectName, sessionType, sessionName, cacheParams);
    if ~isempty(cached)
      cached = add_missing_lfp_result_fields(cached);
      sessionResults(iSession) = cached;
      sessionResults(iSession).sessionType = sessionType;
      sessionResults(iSession).sessionName = sessionName;
      sessionResults(iSession).subjectName = subjectForSession;
      sessionResults(iSession).label = sessionTable.label{iSession};
      sessionResults(iSession).success = true;
      fprintf('  Loaded cached LFP summary.\n');
      continue;
    end
  end

  try
    engConfig = struct( ...
      'splitByEngagement', splitByEngagement, ...
      'bufferBefore', engagementBufferBefore, ...
      'bufferAfter', engagementBufferAfter, ...
      'minNonEngagedWindow', minNonEngagedWindow, ...
      'absorbSingleEvents', absorbSingleEvents, ...
      'minTimeNonEngaged', minTimeNonEngaged, ...
      'restrictToAwake', restrictToAwake);
    summary = run_lfp_session_summary(sessionType, sessionName, cacheSubject, ...
      collectStart, collectEnd, brainArea, brainAreaCombinations, bands, ...
      analyzeBandPower, lfpCleanParams, d2Window, d2Step, rawD2BinSize, useLog10D2, ...
      pOrder, critType, d2Method, klFitMethod, klErrBars, klParallel, nSdCutoff, ...
      fitConfig, engConfig);
    sessionResults(iSession) = copy_lfp_summary_metrics(sessionResults(iSession), summary);
    sessionResults(iSession).success = true;
    fprintf('  Raw d2 = %.3g (n=%d); tau = %.3f; alpha = %.3f; avalanches = %d\n', ...
      summary.rawD2Mean, summary.rawNWindows, summary.tau, summary.alpha, summary.nAvalanches);
    if summary.engagementSplit
      fprintf('  Engaged d2 = %.3g (n=%d); tau = %.3f; alpha = %.3f; avalanches = %d\n', ...
        summary.engagedRawD2Mean, summary.engagedNWindows, summary.engagedTau, ...
        summary.engagedAlpha, summary.engagedNAvalanches);
    end
    if useSessionCache
      save_lfp_svt_cache(paths, subjectName, sessionType, sessionName, cacheParams, ...
        sessionResults(iSession));
    end
  catch ME
    sessionResults(iSession).success = false;
    sessionResults(iSession).skipReason = ME.message;
    fprintf('  Skipping session: %s\n', ME.message);
  end
end

nOk = sum([sessionResults.success]);
fprintf('\nFinished %d/%d sessions.\n', nOk, nSessions);
if nOk == 0
  error('No LFP sessions completed for subject "%s".', subjectName);
end

if plotResults
  plot_lfp_raw_metrics_across_sessions(sessionResults, sessionTypes, brainArea, ...
    useLog10D2, lfpHighpassHz, lfpLowpassHz, paths, plotConfig);
  if plotSeparatedMetrics
    plot_lfp_separated_metrics(sessionResults, sessionTypes, brainArea, ...
      useLog10D2, lfpHighpassHz, lfpLowpassHz, paths, plotConfig);
  end
  if analyzeBandPower
    plot_lfp_band_d2_across_sessions(sessionResults, sessionTypes, brainArea, ...
      useLog10D2, paths, plotConfig);
  end
end

fprintf('\n=== Done ===\n');

%% Local functions

function taskTypes = normalize_lfp_task_types(taskTypes)
% NORMALIZE_LFP_TASK_TYPES - Cell of reach, interval, and/or semicircle
if ischar(taskTypes) || isstring(taskTypes)
  taskTypes = cellstr(taskTypes);
end
taskTypes = cellfun(@(taskName) lower(strtrim(char(taskName))), taskTypes(:), ...
  'UniformOutput', false);
taskTypes = taskTypes(~cellfun(@isempty, taskTypes));
allowed = {'reach', 'interval', 'semicircle'};
unknown = taskTypes(~ismember(taskTypes, allowed));
if isempty(taskTypes) || ~isempty(unknown)
  error('taskTypes must be one or more of: %s.', strjoin(allowed, ', '));
end
end

function sessionTable = build_lfp_subject_session_table(sessionTypes, subjectName)
% BUILD_LFP_SUBJECT_SESSION_TABLE - One subject's sessions in plot order
sessionTypeCol = {};
sessionNameCol = {};
subjectNameCol = {};
labelCol = {};
for iType = 1:numel(sessionTypes)
  sessionType = sessionTypes{iType};
  entries = manuscript_sessions_for_type(sessionType);
  for iEntry = 1:numel(entries)
    sessionTypeCol{end + 1, 1} = sessionType; %#ok<AGROW>
    sessionNameCol{end + 1, 1} = entries(iEntry).sessionName; %#ok<AGROW>
    if isfield(entries, 'subjectName')
      subjectNameCol{end + 1, 1} = entries(iEntry).subjectName; %#ok<AGROW>
    else
      subjectNameCol{end + 1, 1} = ''; %#ok<AGROW>
    end
    labelCol{end + 1, 1} = entries(iEntry).sessionName; %#ok<AGROW>
  end
end
sessionTable = table(sessionTypeCol, sessionNameCol, subjectNameCol, labelCol, ...
  'VariableNames', {'sessionType', 'sessionName', 'subjectName', 'label'});
sessionTable = filter_manuscript_session_table_by_subject(sessionTable, subjectName);
end

function result = empty_lfp_session_result()
% EMPTY_LFP_SESSION_RESULT - One session row before analysis
result = struct();
result.sessionType = '';
result.sessionName = '';
result.subjectName = '';
result.label = '';
result.success = false;
result.skipReason = '';
result.rawD2Mean = nan;
result.rawD2Sem = nan;
result.bandNames = {};
result.bandD2Mean = [];
result.bandD2Sem = [];
result.tau = nan;
result.alpha = nan;
result.nAvalanches = 0;
result.engagementSplit = false;
result.nonEngagedSec = nan;
result.engagedRawD2Mean = nan;
result.engagedRawD2Sem = nan;
result.engagedNWindows = 0;
result.engagedBandD2Mean = [];
result.engagedBandD2Sem = [];
result.engagedTau = nan;
result.engagedAlpha = nan;
result.engagedNAvalanches = 0;
end

function summary = run_lfp_session_summary(sessionType, sessionName, subjectName, ...
    collectStart, collectEnd, brainArea, brainAreaCombinations, bands, ...
    analyzeBandPower, lfpCleanParams, d2Window, d2Step, rawD2BinSize, useLog10D2, ...
    pOrder, critType, d2Method, klFitMethod, klErrBars, klParallel, nSdCutoff, ...
    fitConfig, engConfig)
% RUN_LFP_SESSION_SUMMARY - Raw d2, optional band-power d2, and negative-peak avalanches
%
% Variables:
%   sessionType, sessionName, subjectName - Session identity
%   collectStart, collectEnd - Analysis window (s)
%   brainArea - Area or combined-area name
%   bands - Band names and frequency edges for binned power
%   analyzeBandPower - If true, compute band-power envelopes and their d2
%   nSdCutoff - Negative-peak threshold in SD
%   engConfig - splitByEngagement, buffers, and restrictToAwake
%
% Goal:
%   Spontaneous sessions drop sleep: d2 windows and avalanche bouts stay
%   inside awake epochs. Task sessions with splitByEngagement analyze
%   engaged and non-engaged segments only.

opts = neuro_behavior_options();
opts.collectStart = collectStart;
opts.collectEnd = collectEnd;
opts.firingRateCheckTime = [];

loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectName);
loadArgs = [loadArgs, {'lfpCleanParams', lfpCleanParams, 'bands', bands, ...
  'computeLfpBands', analyzeBandPower}];
dataStruct = load_session_data(sessionType, 'lfp', loadArgs{:});
dataStruct.sessionType = sessionType;
dataStruct.sessionName = sessionName;
dataStruct.subjectName = subjectName;
dataStruct.dataType = sessionType;

if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectStart') ...
    && ~isempty(dataStruct.opts.collectStart)
  collectStart = dataStruct.opts.collectStart;
end
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectEnd') ...
    && ~isempty(dataStruct.opts.collectEnd)
  collectEnd = dataStruct.opts.collectEnd;
end

dataStruct = prepare_area_lfp_for_criticality( ...
  dataStruct, brainArea, brainAreaCombinations, collectStart, collectEnd, ...
  bands, lfpCleanParams, rawD2BinSize, analyzeBandPower);
dataStruct.sessionType = sessionType;
dataStruct.sessionName = sessionName;
dataStruct.subjectName = subjectName;

d2Config = struct();
d2Config.slidingWindowSize = d2Window;
d2Config.stepSize = d2Step;
d2Config.analyzeD2 = true;
d2Config.analyzeDFA = false;
d2Config.enablePermutations = false;
d2Config.makePlots = false;
d2Config.saveResults = false;
d2Config.plotBinnedEnvelopes = false;
d2Config.plotRawLfp = false;
d2Config.binnedSignalField = 'binnedPower';
d2Config.analyzeBandPower = analyzeBandPower;
d2Config.pOrder = pOrder;
d2Config.critType = critType;
d2Config.d2Method = d2Method;
d2Config.klFitMethod = klFitMethod;
d2Config.klErrBars = klErrBars;
d2Config.klParallel = klParallel;
d2Config.minSegmentLength = 50;
d2Config.restrictToAwake = logical(engConfig.restrictToAwake);

signal = dataStruct.lfpPerArea(:, 1);
fs = dataStruct.opts.fsLfp;
timeOrigin = dataStruct.opts.collectStart;
timeEnd = timeOrigin + numel(signal) / fs;
doSplit = engConfig.splitByEngagement && is_manuscript_engagement_session_type(sessionType);

if ~doSplit
  d2Results = criticality_lfp_analysis(dataStruct, d2Config);
  summary = summarize_lfp_d2(d2Results, bands, useLog10D2, analyzeBandPower);
  av = pool_nlfp_avalanches(dataStruct, signal, nSdCutoff, engConfig.restrictToAwake);
  av = fit_nlfp_avalanche_exponents(av, fitConfig);
  summary.tau = av.tau;
  summary.alpha = av.alpha;
  summary.nAvalanches = av.nAvalanches;
  summary = attach_empty_engaged_metrics(summary);
  return;
end

eventTimes = load_lfp_engagement_event_times(dataStruct, sessionType, subjectName, sessionName);
eventTimes = eventTimes(eventTimes >= timeOrigin & eventTimes <= timeEnd);
if isempty(eventTimes)
  error('No engagement events in the collect window.');
end
[engagedSegs, nonEngagedSegs] = define_engagement_segments( ...
  timeOrigin, timeEnd, eventTimes, engConfig.minNonEngagedWindow, ...
  engConfig.bufferBefore, engConfig.bufferAfter, engConfig.absorbSingleEvents);
fprintf('  Engagement events: %d; engaged segments: %d (%.1f s); non-engaged: %d (%.1f s)\n', ...
  numel(eventTimes), numel(engagedSegs), sum_segment_durations(engagedSegs), ...
  numel(nonEngagedSegs), sum_segment_durations(nonEngagedSegs));

summary = summarize_lfp_class(dataStruct, signal, d2Config, engagedSegs, ...
  bands, useLog10D2, analyzeBandPower, d2Window, nSdCutoff, fitConfig, 'Engaged');
engagedSummary = summary;
summary = summarize_lfp_class(dataStruct, signal, d2Config, nonEngagedSegs, ...
  bands, useLog10D2, analyzeBandPower, d2Window, nSdCutoff, fitConfig, 'Non-engaged');
summary.engagementSplit = true;
summary.nonEngagedSec = sum_segment_durations(nonEngagedSegs);
summary.engagedRawD2Mean = engagedSummary.rawD2Mean;
summary.engagedRawD2Sem = engagedSummary.rawD2Sem;
summary.engagedNWindows = engagedSummary.rawNWindows;
summary.engagedBandD2Mean = engagedSummary.bandD2Mean;
summary.engagedBandD2Sem = engagedSummary.bandD2Sem;
summary.engagedTau = engagedSummary.tau;
summary.engagedAlpha = engagedSummary.alpha;
summary.engagedNAvalanches = engagedSummary.nAvalanches;

if engConfig.minTimeNonEngaged > 0 && summary.nonEngagedSec < engConfig.minTimeNonEngaged
  fprintf('  Non-engaged time %.1f s is below %.1f s; non-engaged metrics blanked.\n', ...
    summary.nonEngagedSec, engConfig.minTimeNonEngaged);
  summary.rawD2Mean = nan;
  summary.rawD2Sem = nan;
  summary.rawNWindows = 0;
  summary.tau = nan;
  summary.alpha = nan;
  summary.nAvalanches = 0;
  if ~isempty(summary.bandD2Mean)
    summary.bandD2Mean(:) = nan;
    summary.bandD2Sem(:) = nan;
  end
end
end

function summary = summarize_lfp_class(dataStruct, signal, d2Config, segments, ...
    bands, useLog10D2, analyzeBandPower, d2Window, nSdCutoff, fitConfig, classLabel)
% SUMMARIZE_LFP_CLASS - d2 and avalanches inside one engagement class
%
% Variables:
%   segments - Struct array with .start and .end (s)
%   d2Window - Center-packed window length (s)
%
% Goal:
%   Pack non-overlapping d2 windows in the middle of each segment and give
%   each segment its own negative-peak threshold.

fprintf('  %s: %d segment(s)\n', classLabel, numel(segments));
centerTimes = center_pack_windows_in_segments(segments, d2Window);
classConfig = d2Config;
classConfig.restrictToAwake = false;
classConfig.windowCenterTimes = centerTimes;
if isempty(centerTimes)
  fprintf('    No d2 window fits inside these segments.\n');
  summary = empty_class_d2_summary(bands, analyzeBandPower);
else
  d2Results = criticality_lfp_analysis(dataStruct, classConfig);
  summary = summarize_lfp_d2(d2Results, bands, useLog10D2, analyzeBandPower);
end
timeOrigin = dataStruct.opts.collectStart;
av = pool_nlfp_on_segments(signal, 1 / dataStruct.opts.fsLfp, timeOrigin, segments, nSdCutoff);
av = fit_nlfp_avalanche_exponents(av, fitConfig);
summary.tau = av.tau;
summary.alpha = av.alpha;
summary.nAvalanches = av.nAvalanches;
end

function summary = empty_class_d2_summary(bands, analyzeBandPower)
% EMPTY_CLASS_D2_SUMMARY - NaN d2 when no window fits in the class
summary = struct();
summary.rawD2Mean = nan;
summary.rawD2Sem = nan;
summary.rawNWindows = 0;
if analyzeBandPower
  nBands = size(bands, 1);
  summary.bandNames = cell(1, nBands);
  for iBand = 1:nBands
    summary.bandNames{iBand} = char(bands{iBand, 1});
  end
  summary.bandD2Mean = nan(1, nBands);
  summary.bandD2Sem = nan(1, nBands);
else
  summary.bandNames = {};
  summary.bandD2Mean = [];
  summary.bandD2Sem = [];
end
end

function summary = attach_empty_engaged_metrics(summary)
% ATTACH_EMPTY_ENGAGED_METRICS - No engagement overlay for this session
summary.engagementSplit = false;
summary.nonEngagedSec = nan;
summary.engagedRawD2Mean = nan;
summary.engagedRawD2Sem = nan;
summary.engagedNWindows = 0;
summary.engagedBandD2Mean = [];
summary.engagedBandD2Sem = [];
summary.engagedTau = nan;
summary.engagedAlpha = nan;
summary.engagedNAvalanches = 0;
end

function eventTimes = load_lfp_engagement_event_times(dataStruct, sessionType, subjectName, sessionName)
% LOAD_LFP_ENGAGEMENT_EVENT_TIMES - Reach onsets, interval outcomes, or semicircle times
%
% Goal:
%   Use the same event clocks as the spike engagement modules.
paths = get_paths();
switch lower(strtrim(char(sessionType)))
  case 'reach'
    if ~isfield(dataStruct, 'reachStart') || isempty(dataStruct.reachStart)
      error('Reach session has no reachStart times.');
    end
    eventTimes = dataStruct.reachStart(:);
  case 'interval'
    eventTimes = load_interval_beam_break_events(paths, subjectName, sessionName, 0.1);
  case 'semicircle'
    eventTimes = load_semicircle_beam_break_events(paths, subjectName, sessionName);
  otherwise
    eventTimes = [];
end
eventTimes = double(eventTimes(:));
eventTimes = eventTimes(isfinite(eventTimes));
end

function totalSec = sum_segment_durations(segments)
% SUM_SEGMENT_DURATIONS - Total length of a segment list (s)
if isempty(segments)
  totalSec = 0;
  return;
end
totalSec = sum([segments.end] - [segments.start]);
end

function result = copy_lfp_summary_metrics(result, summary)
% COPY_LFP_SUMMARY_METRICS - Write analysis fields onto one session row
metricFields = {'rawD2Mean', 'rawD2Sem', 'bandNames', 'bandD2Mean', 'bandD2Sem', ...
  'tau', 'alpha', 'nAvalanches', 'engagementSplit', 'nonEngagedSec', ...
  'engagedRawD2Mean', 'engagedRawD2Sem', 'engagedNWindows', ...
  'engagedBandD2Mean', 'engagedBandD2Sem', 'engagedTau', 'engagedAlpha', ...
  'engagedNAvalanches'};
for iField = 1:numel(metricFields)
  fieldName = metricFields{iField};
  if isfield(summary, fieldName)
    result.(fieldName) = summary.(fieldName);
  end
end
end

function result = add_missing_lfp_result_fields(result)
% ADD_MISSING_LFP_RESULT_FIELDS - Older caches gain the engagement fields
template = empty_lfp_session_result();
fieldNames = fieldnames(template);
for iField = 1:numel(fieldNames)
  if ~isfield(result, fieldNames{iField})
    result.(fieldNames{iField}) = template.(fieldNames{iField});
  end
end
end

function summary = summarize_lfp_d2(d2Results, bands, useLog10D2, analyzeBandPower)
% SUMMARIZE_LFP_D2 - Mean and SEM of window-wise d2 for raw LFP and each band
summary = struct();
rawVals = nan;
if isfield(d2Results, 'd2Lfp') && ~isempty(d2Results.d2Lfp) && ~isempty(d2Results.d2Lfp{1})
  rawVals = d2Results.d2Lfp{1}{1}(:);
end
if useLog10D2
  rawVals = log10_positive(rawVals);
end
[summary.rawD2Mean, summary.rawD2Sem, summary.rawNWindows] = mean_sem_finite(rawVals);

if nargin < 4 || isempty(analyzeBandPower) || ~logical(analyzeBandPower)
  summary.bandNames = {};
  summary.bandD2Mean = [];
  summary.bandD2Sem = [];
  return;
end

nBands = size(bands, 1);
summary.bandNames = cell(1, nBands);
summary.bandD2Mean = nan(1, nBands);
summary.bandD2Sem = nan(1, nBands);
for iBand = 1:nBands
  summary.bandNames{iBand} = char(bands{iBand, 1});
  bandVals = nan;
  if isfield(d2Results, 'd2') && ~isempty(d2Results.d2) && ~isempty(d2Results.d2{1}) ...
      && numel(d2Results.d2{1}) >= iBand && ~isempty(d2Results.d2{1}{iBand})
    bandVals = d2Results.d2{1}{iBand}(:);
  end
  if useLog10D2
    bandVals = log10_positive(bandVals);
  end
  [summary.bandD2Mean(iBand), summary.bandD2Sem(iBand)] = mean_sem_finite(bandVals);
end
end

function values = log10_positive(values)
% LOG10_POSITIVE - log10, with NaN where the value is not positive
values = values(:);
out = nan(size(values));
keep = isfinite(values) & values > 0;
out(keep) = log10(values(keep));
values = out;
end

function [mu, sem, n] = mean_sem_finite(values)
% MEAN_SEM_FINITE - Mean and SEM of finite samples
values = values(:);
values = values(isfinite(values));
n = numel(values);
if n == 0
  mu = nan;
  sem = nan;
  return;
end
mu = mean(values);
if n > 1
  sem = std(values) / sqrt(n);
else
  sem = 0;
end
end

function av = pool_nlfp_avalanches(dataStruct, signal, nSdCutoff, restrictToAwake)
% POOL_NLFP_AVALANCHES - Negative-peak avalanches, one threshold per awake bout
%
% Variables:
%   dataStruct - Prepared LFP session (identity and opts.collectStart)
%   signal - Zero-mean raw LFP column
%   nSdCutoff - Peak cutoff in SD
%   restrictToAwake - Spontaneous sessions drop sleep when true
%
% Goal:
%   Task sessions use one threshold on the collect window. Spontaneous
%   sessions detect negative peaks inside each awake bout, then pool sizes
%   (threshold units) and durations (seconds) for one power-law fit.

signal = signal(:);
fs = dataStruct.opts.fsLfp;
samplePeriodSec = 1 / fs;
timeOrigin = 0;
if isfield(dataStruct, 'opts') && isfield(dataStruct.opts, 'collectStart') ...
    && ~isempty(dataStruct.opts.collectStart)
  timeOrigin = dataStruct.opts.collectStart;
end
timeEnd = timeOrigin + numel(signal) * samplePeriodSec;
awakeConfig = struct('restrictToAwake', logical(restrictToAwake));
[segments, applyFilter] = resolve_spontaneous_awake_segments( ...
  dataStruct, awakeConfig, [timeOrigin, timeEnd]);

if ~applyFilter
  av = detect_klaus_avalanches(signal, samplePeriodSec, nSdCutoff, 'nlfp');
  av.durationSec = av.durations(:) * av.deltaT;
  return;
end

if isempty(segments)
  fprintf('  No awake epochs; negative-peak avalanches left empty.\n');
else
  fprintf('  Negative-peak avalanches: %d awake bout(s), own threshold each\n', numel(segments));
end
av = pool_nlfp_on_segments(signal, samplePeriodSec, timeOrigin, segments, nSdCutoff);
end

function av = pool_nlfp_on_segments(signal, samplePeriodSec, timeOrigin, segments, nSdCutoff)
% POOL_NLFP_ON_SEGMENTS - One negative-peak threshold per segment, then pool
%
% Variables:
%   signal - Zero-mean LFP column
%   samplePeriodSec - Sample period (s)
%   timeOrigin - Time of sample 1 (s)
%   segments - Struct array with .start and .end (s)
%   nSdCutoff - Peak cutoff in SD
%
% Goal:
%   Sizes stay in each segment's threshold units. Durations are converted to
%   seconds with that segment's delta-t before pooling.

signal = signal(:);
sizes = [];
durationSec = [];
nEvents = 0;
for iSeg = 1:numel(segments)
  piece = lfp_samples_inside_segment(signal, samplePeriodSec, timeOrigin, segments(iSeg));
  if numel(piece) < 3
    continue;
  end
  segAv = detect_klaus_avalanches(piece, samplePeriodSec, nSdCutoff, 'nlfp');
  sizes = [sizes; segAv.sizes(:)]; %#ok<AGROW>
  if isfield(segAv, 'deltaT') && isfinite(segAv.deltaT) && segAv.deltaT > 0
    durationSec = [durationSec; segAv.durations(:) * segAv.deltaT]; %#ok<AGROW>
  end
  nEvents = nEvents + segAv.nEvents;
end

av = detect_klaus_avalanches([], samplePeriodSec, nSdCutoff, 'nlfp');
av.sizes = sizes;
av.durationSec = durationSec;
av.nEvents = nEvents;
av.nAvalanches = numel(sizes);
end

function piece = lfp_samples_inside_segment(signal, samplePeriodSec, timeOrigin, segment)
% LFP_SAMPLES_INSIDE_SEGMENT - Samples whose interval lies inside one bout
piece = signal([], 1);
if ~(isfinite(samplePeriodSec) && samplePeriodSec > 0)
  return;
end
iStart = ceil((segment.start - timeOrigin) / samplePeriodSec + 1 - 1e-9);
iEnd = floor((segment.end - timeOrigin) / samplePeriodSec);
iStart = max(1, iStart);
iEnd = min(numel(signal), iEnd);
if iEnd >= iStart
  piece = signal(iStart:iEnd);
end
end

function av = fit_nlfp_avalanche_exponents(av, fitConfig)
% FIT_NLFP_AVALANCHE_EXPONENTS - Tau from sizes, alpha from duration in seconds
av.tau = nan;
av.alpha = nan;
if ~isfield(av, 'durationSec')
  if isfield(av, 'durations') && isfield(av, 'deltaT') && isfinite(av.deltaT)
    av.durationSec = av.durations(:) * av.deltaT;
  else
    av.durationSec = [];
  end
end
if numel(av.sizes) >= 2
  sizeFit = fit_avalanche_power_law(av.sizes, fitConfig);
  av.tau = sizeFit.exponent;
end
if numel(av.durationSec) >= 2
  durFit = fit_avalanche_power_law(av.durationSec, fitConfig);
  av.alpha = durFit.exponent;
end
if ~isfield(av, 'nAvalanches')
  av.nAvalanches = numel(av.sizes);
end
end

function plot_lfp_raw_metrics_across_sessions(sessionResults, sessionTypes, brainArea, ...
    useLog10D2, lfpHighpassHz, lfpLowpassHz, paths, plotConfig)
% PLOT_LFP_RAW_METRICS_ACROSS_SESSIONS - Raw-LFP d2, tau, and alpha by session
%
% Goal:
%   Match the spontaneous-vs-task marker plot. d2 stays in native units.
%   Tau and alpha are scaled onto that axis. Color is the task color.

ok = [sessionResults.success];
if ~any(ok)
  return;
end
if useLog10D2
  d2Label = 'log_{10}(d2)';
else
  d2Label = 'd2';
end

[xBySession, xtickLabels] = session_plot_positions(sessionResults, sessionTypes);
d2Mean = [sessionResults.rawD2Mean];
d2Sem = [sessionResults.rawD2Sem];
tau = [sessionResults.tau];
alpha = [sessionResults.alpha];
engagedD2 = lfp_engaged_series(sessionResults, 'd2');
engagedTau = lfp_engaged_series(sessionResults, 'tau');
engagedAlpha = lfp_engaged_series(sessionResults, 'alpha');
yLim = padded_ylim([d2Mean(:); d2Mean(:) + d2Sem(:); d2Mean(:) - d2Sem(:); engagedD2(:)]);
if isempty(yLim)
  warning('criticality_lfp_spontaneous_vs_task:NoRawD2', ...
    'Skipping the raw LFP plot: no finite d2 values.');
  return;
end
tauRange = [tau(:); engagedTau(:)];
alphaRange = [alpha(:); engagedAlpha(:)];
[tauPlot, tauSemPlot] = scale_onto_interval(tau, zeros(size(tau)), yLim, tauRange);
[alphaPlot, alphaSemPlot] = scale_onto_interval(alpha, zeros(size(alpha)), yLim, alphaRange);
[engagedTauPlot, ~] = scale_onto_interval(engagedTau, zeros(size(engagedTau)), yLim, tauRange);
[engagedAlphaPlot, ~] = scale_onto_interval(engagedAlpha, zeros(size(engagedAlpha)), yLim, alphaRange);

fig = figure('Color', 'w', 'Name', 'LFP raw d2, tau, alpha');
position_figure_full_monitor(fig);
ax = axes(fig);
hold(ax, 'on');
metricNames = {'d2', 'tau', 'alpha'};
metricMarkers = {'o', 's', 'd'};
metricFilled = [true, false, false];
yByMetric = {d2Mean, tauPlot, alphaPlot};
semByMetric = {d2Sem, tauSemPlot, alphaSemPlot};
yEngByMetric = {engagedD2, engagedTauPlot, engagedAlphaPlot};
xOffsets = [-0.18, 0, 0.18];

for iType = 1:numel(sessionTypes)
  sessionType = sessionTypes{iType};
  typeMask = ok & strcmp({sessionResults.sessionType}, sessionType);
  if ~any(typeMask)
    continue;
  end
  taskColor = colors_for_tasks(sessionType);
  idx = find(typeMask);
  for iMetric = 1:numel(metricNames)
    xLine = [];
    yLine = [];
    for iRow = 1:numel(idx)
      xMetric = xBySession(idx(iRow)) + xOffsets(iMetric);
      yMetric = yByMetric{iMetric}(idx(iRow));
      if isfinite(xMetric) && isfinite(yMetric)
        xLine(end + 1) = xMetric; %#ok<AGROW>
        yLine(end + 1) = yMetric; %#ok<AGROW>
      end
    end
    if numel(xLine) >= 2
      plot(ax, xLine, yLine, '-', 'Color', taskColor, 'LineWidth', plotConfig.lineWidth, ...
        'HandleVisibility', 'off');
    end
  end
  for iRow = 1:numel(idx)
    for iMetric = 1:numel(metricNames)
      xMetric = xBySession(idx(iRow)) + xOffsets(iMetric);
      yMetric = yByMetric{iMetric}(idx(iRow));
      ySem = semByMetric{iMetric}(idx(iRow));
      if isfinite(xMetric) && isfinite(yMetric)
        if ~isfinite(ySem)
          ySem = 0;
        end
        faceColor = taskColor;
        if ~metricFilled(iMetric)
          faceColor = 'none';
        end
        errorbar(ax, xMetric, yMetric, ySem, metricMarkers{iMetric}, ...
          'Color', taskColor, 'MarkerFaceColor', faceColor, ...
          'MarkerSize', plotConfig.scatterMarkerSize / 5, ...
          'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
          'HandleVisibility', 'off');
      end
      yEng = yEngByMetric{iMetric}(idx(iRow));
      if isfinite(xMetric) && isfinite(yEng) && sessionResults(idx(iRow)).engagementSplit
        errorbar(ax, xMetric, yEng, 0, 'o', ...
          'Color', taskColor, 'MarkerFaceColor', 'none', ...
          'MarkerSize', plotConfig.scatterMarkerSize / 5, ...
          'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
          'HandleVisibility', 'off');
      end
    end
  end
end

legendHandles = gobjects(numel(metricNames), 1);
for iMetric = 1:numel(metricNames)
  faceColor = [0.2 0.2 0.2];
  if ~metricFilled(iMetric)
    faceColor = 'none';
  end
  legendHandles(iMetric) = plot(ax, nan, nan, metricMarkers{iMetric}, ...
    'Color', [0.2 0.2 0.2], 'MarkerFaceColor', faceColor, ...
    'MarkerSize', plotConfig.scatterMarkerSize / 5, 'LineWidth', plotConfig.lineWidth, ...
    'DisplayName', metricNames{iMetric});
end
if any([sessionResults.engagementSplit])
  legendHandles(end + 1) = plot(ax, nan, nan, 'o', ...
    'Color', [0.2 0.2 0.2], 'MarkerFaceColor', 'none', ...
    'MarkerSize', plotConfig.scatterMarkerSize / 5, 'LineWidth', plotConfig.lineWidth, ...
    'DisplayName', 'engaged'); %#ok<AGROW>
end

xlim(ax, [0.4, max(xBySession(ok)) + 0.6]);
ylim(ax, yLim);
xticks(ax, xBySession(ok));
xticklabels(ax, xtickLabels(ok));
xtickangle(ax, 45);
ylabel(ax, sprintf('Raw LFP %s', d2Label), 'FontSize', plotConfig.axisLabelFontSize);
xlabel(ax, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
titleText = sprintf(['%s | raw LFP %.0f-%.0f Hz | filled circle = d2, ', ...
  'open square = tau, open diamond = alpha (scaled onto d2)'], ...
  brainArea, lfpHighpassHz, lfpLowpassHz);
if any([sessionResults.engagementSplit])
  titleText = [titleText, ' | open circles = engaged'];
end
title(ax, titleText, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
legend(ax, legendHandles, 'Location', 'best', ...
  'FontSize', plotConfig.legendFontSize);
set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'TickDir', 'out', 'Box', 'off');
grid(ax, 'on');

save_lfp_svt_figure(fig, paths, plotConfig, 'raw_d2_tau_alpha');
end

function plot_lfp_band_d2_across_sessions(sessionResults, sessionTypes, brainArea, ...
    useLog10D2, paths, plotConfig)
% PLOT_LFP_BAND_D2_ACROSS_SESSIONS - Band-power d2 on one shared axis
%
% Goal:
%   Every band uses the same y-axis. Color identifies the band and is shared
%   across tasks. Marker shape identifies the task.

ok = [sessionResults.success];
if ~any(ok)
  return;
end
bandNames = {};
for iSession = 1:numel(sessionResults)
  if sessionResults(iSession).success && ~isempty(sessionResults(iSession).bandNames)
    bandNames = sessionResults(iSession).bandNames;
    break;
  end
end
nBands = numel(bandNames);
if nBands == 0
  return;
end
if useLog10D2
  d2Label = 'log_{10}(d2)';
else
  d2Label = 'd2';
end

[xBySession, xtickLabels] = session_plot_positions(sessionResults, sessionTypes);
bandColors = lfp_band_colors(nBands);
xOffsets = linspace(-0.22, 0.22, nBands);
if nBands == 1
  xOffsets = 0;
end

yAll = [];
for iSession = 1:numel(sessionResults)
  if ~sessionResults(iSession).success
    continue;
  end
  yAll = [yAll; sessionResults(iSession).bandD2Mean(:); ...
    sessionResults(iSession).bandD2Mean(:) + sessionResults(iSession).bandD2Sem(:); ...
    sessionResults(iSession).bandD2Mean(:) - sessionResults(iSession).bandD2Sem(:)]; %#ok<AGROW>
  if sessionResults(iSession).engagementSplit && ~isempty(sessionResults(iSession).engagedBandD2Mean)
    yAll = [yAll; sessionResults(iSession).engagedBandD2Mean(:)]; %#ok<AGROW>
  end
end
yLim = padded_ylim(yAll);
if isempty(yLim)
  warning('criticality_lfp_spontaneous_vs_task:NoBandD2', ...
    'Skipping the band-power plot: no finite d2 values.');
  return;
end

fig = figure('Color', 'w', 'Name', 'LFP band-power d2');
position_figure_full_monitor(fig);
ax = axes(fig);
hold(ax, 'on');

for iType = 1:numel(sessionTypes)
  sessionType = sessionTypes{iType};
  typeMask = ok & strcmp({sessionResults.sessionType}, sessionType);
  if ~any(typeMask)
    continue;
  end
  marker = marker_for_session_type(sessionType);
  idx = find(typeMask);
  for iBand = 1:nBands
    xLine = [];
    yLine = [];
    for iRow = 1:numel(idx)
      means = sessionResults(idx(iRow)).bandD2Mean;
      if numel(means) < iBand || ~isfinite(means(iBand))
        continue;
      end
      xLine(end + 1) = xBySession(idx(iRow)) + xOffsets(iBand); %#ok<AGROW>
      yLine(end + 1) = means(iBand); %#ok<AGROW>
    end
    if numel(xLine) >= 2
      plot(ax, xLine, yLine, '-', 'Color', bandColors(iBand, :), ...
        'LineWidth', max(plotConfig.lineWidth - 0.4, 0.8), 'HandleVisibility', 'off');
    end
    for iRow = 1:numel(idx)
      xBand = xBySession(idx(iRow)) + xOffsets(iBand);
      means = sessionResults(idx(iRow)).bandD2Mean;
      sems = sessionResults(idx(iRow)).bandD2Sem;
      if numel(means) >= iBand && isfinite(means(iBand))
        ySem = 0;
        if numel(sems) >= iBand && isfinite(sems(iBand))
          ySem = sems(iBand);
        end
        errorbar(ax, xBand, means(iBand), ySem, ...
          marker, 'Color', bandColors(iBand, :), 'MarkerFaceColor', bandColors(iBand, :), ...
          'MarkerSize', plotConfig.scatterMarkerSize / 5, ...
          'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
          'HandleVisibility', 'off');
      end
      engagedMeans = sessionResults(idx(iRow)).engagedBandD2Mean;
      if sessionResults(idx(iRow)).engagementSplit && numel(engagedMeans) >= iBand ...
          && isfinite(engagedMeans(iBand))
        plot(ax, xBand, engagedMeans(iBand), ...
          marker, 'Color', bandColors(iBand, :), 'MarkerFaceColor', 'none', ...
          'MarkerSize', plotConfig.scatterMarkerSize / 5, ...
          'LineWidth', plotConfig.lineWidth, 'HandleVisibility', 'off');
      end
    end
  end
end

bandHandles = gobjects(nBands, 1);
for iBand = 1:nBands
  bandHandles(iBand) = plot(ax, nan, nan, 'o', 'Color', bandColors(iBand, :), ...
    'MarkerFaceColor', bandColors(iBand, :), 'MarkerSize', plotConfig.scatterMarkerSize / 5, ...
    'LineWidth', plotConfig.lineWidth, 'DisplayName', bandNames{iBand});
end
presentTypes = sessionTypes(cellfun(@(sessionType) any(ok & strcmp({sessionResults.sessionType}, sessionType)), ...
  sessionTypes));
shapeHandles = gobjects(numel(presentTypes), 1);
for iType = 1:numel(presentTypes)
  shapeHandles(iType) = plot(ax, nan, nan, marker_for_session_type(presentTypes{iType}), ...
    'Color', [0.25 0.25 0.25], 'MarkerFaceColor', [0.25 0.25 0.25], ...
    'MarkerSize', plotConfig.scatterMarkerSize / 5, 'LineWidth', plotConfig.lineWidth, ...
    'DisplayName', presentTypes{iType});
end

xlim(ax, [0.4, max(xBySession(ok)) + 0.6]);
ylim(ax, yLim);
xticks(ax, xBySession(ok));
xticklabels(ax, xtickLabels(ok));
xtickangle(ax, 45);
ylabel(ax, sprintf('Band-power %s', d2Label), 'FontSize', plotConfig.axisLabelFontSize);
xlabel(ax, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
titleText = sprintf('%s | band-power d2 | color = band, shape = task', brainArea);
if any([sessionResults.engagementSplit])
  titleText = [titleText, ' | open markers = engaged'];
end
title(ax, titleText, 'FontSize', plotConfig.titleFontSize, 'Interpreter', 'none');
legend(ax, [bandHandles; shapeHandles], 'Location', 'best', ...
  'FontSize', plotConfig.legendFontSize);
set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'TickDir', 'out', 'Box', 'off');
grid(ax, 'on');

save_lfp_svt_figure(fig, paths, plotConfig, 'band_d2');
end

function plot_lfp_separated_metrics(sessionResults, sessionTypes, brainArea, ...
    useLog10D2, lfpHighpassHz, lfpLowpassHz, paths, plotConfig)
% PLOT_LFP_SEPARATED_METRICS - Raw d2, tau, and alpha each on their own axes
%
% Variables:
%   sessionResults - Per-session raw d2, tau, and alpha
%   sessionTypes   - Plot order (spontaneous first)
%   brainArea      - Area name for the figure title
%   useLog10D2     - If true, the d2 panel is already on a log10 scale
%   lfpHighpassHz, lfpLowpassHz - Bandpass printed in the title
%   paths, plotConfig - Save location and manuscript styling
%
% Goal:
%   Same session-level values as the combined raw-LFP plot, in native units.
%   Within each task, a horizontal line marks the mean from the first session
%   to the last. d2 is a filled circle; tau is an open square; alpha is an
%   open diamond. Color is the task color.

if useLog10D2
  d2Label = 'log_{10}(d2)';
else
  d2Label = 'd2';
end
panelField = {'rawD2Mean', 'tau', 'alpha'};
panelSemField = {'rawD2Sem', '', ''};
panelTitle = {'D2', 'Avalanche Sizes', 'Avalanche Durations'};
panelLabel = {d2Label, 'tau', 'alpha'};
panelMarker = {'o', 's', 'd'};
panelFilled = [true, false, false];

[xBySession, xtickLabels] = session_plot_positions(sessionResults, sessionTypes);
ok = [sessionResults.success];
if ~any(ok)
  return;
end

fig = figure('Color', 'w', 'Name', sprintf('Separated metrics — %s', brainArea));
position_figure_full_monitor(fig);

for iPanel = 1:numel(panelField)
  ax = subplot(1, numel(panelField), iPanel, 'Parent', fig);
  hold(ax, 'on');
  [yValues, ySems] = lfp_session_metric_vectors(sessionResults, ...
    panelField{iPanel}, panelSemField{iPanel});
  engagedValues = lfp_engaged_series(sessionResults, panelField{iPanel});
  yLimVals = [yValues(:); yValues(:) + ySems(:); yValues(:) - ySems(:); engagedValues(:)];

  for iType = 1:numel(sessionTypes)
    sessionType = sessionTypes{iType};
    idx = find(ok & strcmp({sessionResults.sessionType}, sessionType));
    if isempty(idx)
      continue;
    end
    taskColor = colors_for_tasks(sessionType);
    xPos = xBySession(idx);
    yPos = yValues(idx);
    semPos = ySems(idx);
    yMeanTask = mean(yPos(isfinite(yPos)));
    if isfinite(yMeanTask)
      plot(ax, [xPos(1), xPos(end)], [yMeanTask, yMeanTask], '-', ...
        'Color', taskColor, 'LineWidth', 2, 'HandleVisibility', 'off');
    end
    faceColor = taskColor;
    if ~panelFilled(iPanel)
      faceColor = 'none';
    end
    errorbar(ax, xPos, yPos, semPos, panelMarker{iPanel}, ...
      'Color', taskColor, 'MarkerFaceColor', faceColor, ...
      'MarkerSize', plotConfig.scatterMarkerSize / 5, ...
      'LineWidth', plotConfig.lineWidth, 'CapSize', plotConfig.errorCapSize, ...
      'HandleVisibility', 'off');
    yEng = engagedValues(idx);
    engOk = isfinite(yEng) & arrayfun(@(row) sessionResults(row).engagementSplit, idx);
    if any(engOk)
      plot(ax, xPos(engOk), yEng(engOk), 'o', ...
        'Color', taskColor, 'MarkerFaceColor', 'none', ...
        'MarkerSize', plotConfig.scatterMarkerSize / 5, ...
        'LineWidth', plotConfig.lineWidth, 'HandleVisibility', 'off');
    end
  end

  yLimPlot = padded_ylim(yLimVals);
  if ~isempty(yLimPlot)
    ylim(ax, yLimPlot);
  end
  if any(ok)
    xlim(ax, [min(xBySession(ok)) - 0.8, max(xBySession(ok)) + 0.8]);
    set(ax, 'XTick', xBySession(ok), 'XTickLabel', xtickLabels(ok), ...
      'XTickLabelRotation', 45);
  end
  xlabel(ax, 'Session', 'FontSize', plotConfig.axisLabelFontSize);
  ylabel(ax, panelLabel{iPanel}, 'FontSize', plotConfig.axisLabelFontSize, ...
    'Interpreter', 'tex');
  title(ax, panelTitle{iPanel}, 'FontSize', plotConfig.titleFontSize);
  set(ax, 'FontSize', plotConfig.tickLabelFontSize, 'LineWidth', plotConfig.axesLineWidth, ...
    'Box', 'off', 'TickDir', 'out');
  hold(ax, 'off');
end

titleText = sprintf('Separated metrics — %s | raw LFP %.0f-%.0f Hz', ...
  brainArea, lfpHighpassHz, lfpLowpassHz);
if any([sessionResults.engagementSplit])
  titleText = [titleText, ' | open circles = engaged'];
end
sgtitle(fig, titleText, ...
  'FontSize', plotConfig.sgtitleFontSize, 'FontWeight', 'bold', 'Interpreter', 'none');
save_lfp_svt_figure(fig, paths, plotConfig, 'separated_metrics');
end

function values = lfp_engaged_series(sessionResults, metricName)
% LFP_ENGAGED_SERIES - Engaged-class metric aligned with sessionResults
%
% Variables:
%   metricName - 'd2', 'tau', 'alpha', or the raw field names used by the
%                separated-metrics panels
values = nan(1, numel(sessionResults));
for iSession = 1:numel(sessionResults)
  if ~isfield(sessionResults(iSession), 'engagementSplit') ...
      || ~sessionResults(iSession).engagementSplit
    continue;
  end
  switch char(metricName)
    case {'d2', 'rawD2Mean'}
      values(iSession) = sessionResults(iSession).engagedRawD2Mean;
    case {'tau'}
      values(iSession) = sessionResults(iSession).engagedTau;
    case {'alpha'}
      values(iSession) = sessionResults(iSession).engagedAlpha;
  end
end
end

function [yValues, ySems] = lfp_session_metric_vectors(sessionResults, valueField, semField)
% LFP_SESSION_METRIC_VECTORS - One row of session means and SEMs
nSessions = numel(sessionResults);
yValues = nan(1, nSessions);
ySems = zeros(1, nSessions);
for iSession = 1:nSessions
  yValues(iSession) = sessionResults(iSession).(valueField);
  if ~isempty(semField) && isfield(sessionResults, semField) ...
      && ~isempty(sessionResults(iSession).(semField))
    ySems(iSession) = sessionResults(iSession).(semField);
  end
end
ySems(~isfinite(ySems)) = 0;
end

function [xBySession, labels] = session_plot_positions(sessionResults, sessionTypes)
% SESSION_PLOT_POSITIONS - Spontaneous first, then tasks, with a gap between groups
nSessions = numel(sessionResults);
xBySession = nan(1, nSessions);
labels = cell(1, nSessions);
xCursor = 0;
groupGap = 1.5;
for iType = 1:numel(sessionTypes)
  idx = find(strcmp({sessionResults.sessionType}, sessionTypes{iType}) & [sessionResults.success]);
  if isempty(idx)
    continue;
  end
  xBySession(idx) = xCursor + (1:numel(idx));
  for iRow = 1:numel(idx)
    labels{idx(iRow)} = sessionResults(idx(iRow)).label;
  end
  xCursor = xBySession(idx(end)) + groupGap;
end
end

function [yPlot, semPlot] = scale_onto_interval(values, sems, yLim, rangeValues)
% SCALE_ONTO_INTERVAL - Map a series onto yLim; SEM uses the same gain
%
% Variables:
%   rangeValues - Optional samples that set the source range. Defaults to values.
yPlot = nan(size(values));
semPlot = zeros(size(values));
if nargin < 4 || isempty(rangeValues)
  rangeValues = values;
end
finiteMask = isfinite(values);
rangeMask = isfinite(rangeValues);
if ~any(rangeMask) || numel(yLim) ~= 2
  return;
end
srcMin = min(rangeValues(rangeMask));
srcMax = max(rangeValues(rangeMask));
if srcMax <= srcMin
  yPlot(finiteMask) = mean(yLim);
  return;
end
gain = (yLim(2) - yLim(1)) / (srcMax - srcMin);
yPlot(finiteMask) = yLim(1) + (values(finiteMask) - srcMin) * gain;
semPlot = abs(gain) * sems;
semPlot(~isfinite(semPlot)) = 0;
end

function yLim = padded_ylim(values)
% PADDED_YLIM - Finite-data limits with a small margin
values = values(isfinite(values));
if isempty(values)
  yLim = [];
  return;
end
yMin = min(values);
yMax = max(values);
if yMax <= yMin
  pad = max(abs(yMin) * 0.05, 0.05);
  yLim = [yMin - pad, yMax + pad];
  return;
end
pad = 0.08 * (yMax - yMin);
yLim = [yMin - pad, yMax + pad];
end

function colors = lfp_band_colors(nBands)
% LFP_BAND_COLORS - Band RGB rows matching session_lfp_criticality after raw LFP
%
% Row 1 of that palette is raw LFP (near black). Bands use the following rows.

palette = [ ...
  0.20 0.45 0.82; ...
  0.90 0.49 0.13; ...
  0.16 0.62 0.32; ...
  0.55 0.25 0.72; ...
  0.84 0.22 0.35];
if nBands <= size(palette, 1)
  colors = palette(1:nBands, :);
  return;
end
colors = lines(nBands);
end

function marker = marker_for_session_type(sessionType)
% MARKER_FOR_SESSION_TYPE - One shape per task, shared across bands
switch lower(strtrim(char(sessionType)))
  case 'spontaneous'
    marker = 'o';
  case 'interval'
    marker = 's';
  case 'reach'
    marker = '^';
  case 'semicircle'
    marker = 'd';
  otherwise
    marker = 'v';
end
end

function save_lfp_svt_figure(fig, paths, plotConfig, plotName)
% SAVE_LFP_SVT_FIGURE - PNG and EPS under dropPath/criticality_manuscript
saveDir = fullfile(paths.dropPath, 'criticality_manuscript');
if ~exist(saveDir, 'dir')
  mkdir(saveDir);
end
fileBase = sprintf('criticality_lfp_spont_vs_task_%s_%s', plotConfig.fileTag, plotName);
exportgraphics(fig, fullfile(saveDir, [fileBase, '.png']), 'Resolution', 300);
exportgraphics(fig, fullfile(saveDir, [fileBase, '.eps']), 'ContentType', 'vector');
fprintf('Saved figure: %s\n', fullfile(saveDir, fileBase));
end

function position_figure_full_monitor(fig)
% POSITION_FIGURE_FULL_MONITOR - Use the last monitor when more than one is attached
monitorPositions = get(0, 'MonitorPositions');
if size(monitorPositions, 1) >= 2
  targetPos = monitorPositions(end, :);
else
  targetPos = monitorPositions(1, :);
end
set(fig, 'Units', 'pixels', 'Position', targetPos);
end

function cached = try_load_lfp_svt_cache(paths, subjectName, sessionType, sessionName, cacheParams)
% TRY_LOAD_LFP_SVT_CACHE - Saved session summary when the config matches
cached = [];
cacheFile = lfp_svt_cache_file(paths, subjectName, sessionType, sessionName, cacheParams);
if ~isfile(cacheFile)
  return;
end
loaded = load(cacheFile, 'summary', 'cacheParams');
if ~isfield(loaded, 'summary') || ~isfield(loaded, 'cacheParams')
  return;
end
storedParams = fill_lfp_cache_d2_defaults(loaded.cacheParams);
requestedParams = fill_lfp_cache_d2_defaults(cacheParams);
if ~is_manuscript_engagement_session_type(sessionType)
  storedParams = clear_engagement_cache_fields(storedParams);
  requestedParams = clear_engagement_cache_fields(requestedParams);
end
if ~isequal(fieldnames(orderfields(storedParams)), fieldnames(orderfields(requestedParams))) ...
    || ~isequaln(orderfields(storedParams), orderfields(requestedParams))
  return;
end
cached = loaded.summary;
end

function save_lfp_svt_cache(paths, subjectName, sessionType, sessionName, cacheParams, summary)
% SAVE_LFP_SVT_CACHE - Write one session summary for this LFP config
cacheFile = lfp_svt_cache_file(paths, subjectName, sessionType, sessionName, cacheParams);
cacheDir = fileparts(cacheFile);
if ~exist(cacheDir, 'dir')
  mkdir(cacheDir);
end
save(cacheFile, 'summary', 'cacheParams', '-v7.3');
fprintf('  Saved LFP cache:\n    %s\n', cacheFile);
end

function cacheFile = lfp_svt_cache_file(paths, subjectName, sessionType, sessionName, cacheParams)
% LFP_SVT_CACHE_FILE - Path encoding area, window, bandpass, cutoff, and bands
if isempty(cacheParams.collectEnd)
  collectTag = sprintf('c%.0ftofull', cacheParams.collectStart);
else
  collectTag = sprintf('c%.0fto%.0f', cacheParams.collectStart, cacheParams.collectEnd);
end
bandTag = strjoin(cacheParams.bands(:, 1), '_');
fileName = sprintf('%s_%s_%s_%s_d2w%.0f_step%.0f_bin%dms_nsd%.1f_hp%.0f_lp%.0f_%s_%s%s%s%s.mat', ...
  matlab.lang.makeValidName(char(sessionType)), ...
  matlab.lang.makeValidName(char(sessionName)), ...
  matlab.lang.makeValidName(char(cacheParams.brainArea)), ...
  collectTag, cacheParams.d2Window, cacheParams.d2Step, ...
  round(cacheParams.rawD2BinSize * 1000), cacheParams.nSdCutoff, ...
  cacheParams.lfpHighpassHz, cacheParams.lfpLowpassHz, ...
  matlab.lang.makeValidName(bandTag), ...
  matlab.lang.makeValidName(char(cacheParams.powerLawFitMethod)), ...
  lfp_band_power_file_tag(cacheParams), ...
  lfp_engagement_file_tag(cacheParams, sessionType), ...
  lfp_d2_method_file_tag(cacheParams));
cacheFile = fullfile(paths.dropPath, 'criticality_manuscript', 'lfp_spontaneous_vs_task', ...
  matlab.lang.makeValidName(char(subjectName)), fileName);
end

function tag = lfp_band_power_file_tag(cacheParams)
% LFP_BAND_POWER_FILE_TAG - Extra filename piece when band power is off
%
% Goal:
%   Caches written before this option included band-power d2 and keep their names.
tag = '';
if isfield(cacheParams, 'analyzeBandPower') && ~logical(cacheParams.analyzeBandPower)
  tag = '_noband';
end
end

function tag = lfp_engagement_file_tag(cacheParams, sessionType)
% LFP_ENGAGEMENT_FILE_TAG - Extra filename piece when a task session is split
%
% Goal:
%   Spontaneous caches, and caches written before engagement splits, keep
%   their names. Sleep removal for spontaneous is already in restrictToAwake.
tag = '';
if nargin >= 2 && ~is_manuscript_engagement_session_type(sessionType)
  return;
end
if ~isfield(cacheParams, 'splitByEngagement') || ~logical(cacheParams.splitByEngagement)
  return;
end
tag = sprintf('_eng_b%.0fa%.0f_gap%.0f', cacheParams.engagementBufferBefore, ...
  cacheParams.engagementBufferAfter, cacheParams.minNonEngagedWindow);
if isfield(cacheParams, 'absorbSingleEvents') && logical(cacheParams.absorbSingleEvents)
  tag = [tag, '_abs'];
end
if isfield(cacheParams, 'minTimeNonEngaged') && cacheParams.minTimeNonEngaged > 0
  tag = sprintf('%s_ne%.0f', tag, cacheParams.minTimeNonEngaged);
end
end

function params = clear_engagement_cache_fields(params)
% CLEAR_ENGAGEMENT_CACHE_FIELDS - Spontaneous summaries do not depend on engagement buffers
params.splitByEngagement = false;
params.engagementBufferBefore = 0;
params.engagementBufferAfter = 0;
params.minNonEngagedWindow = 0;
params.absorbSingleEvents = false;
params.minTimeNonEngaged = 0;
end

function tag = lfp_d2_method_file_tag(cacheParams)
% LFP_D2_METHOD_FILE_TAG - Extra filename piece for KL; empty for Euclidean
%
% Goal:
%   Caches written before the d2Method option are Euclidean and keep their
%   original names.

tag = '';
d2Method = 'euclidean';
if isfield(cacheParams, 'd2Method') && ~isempty(cacheParams.d2Method)
  d2Method = lower(strtrim(char(cacheParams.d2Method)));
end
if strcmp(d2Method, 'euclidean')
  return;
end
tag = ['_d2', matlab.lang.makeValidName(d2Method)];
klFitMethod = 'MaxLikelihood';
if isfield(cacheParams, 'klFitMethod') && ~isempty(cacheParams.klFitMethod)
  klFitMethod = char(cacheParams.klFitMethod);
end
tag = sprintf('%s_%s', tag, matlab.lang.makeValidName(klFitMethod));
if isfield(cacheParams, 'klErrBars') && logical(cacheParams.klErrBars)
  tag = [tag, '_klerr'];
end
end

function params = fill_lfp_cache_d2_defaults(params)
% FILL_LFP_CACHE_D2_DEFAULTS - Missing d2Method matches the original Euclidean fit
if ~isfield(params, 'd2Method') || isempty(params.d2Method)
  params.d2Method = 'euclidean';
end
if ~isfield(params, 'klFitMethod') || isempty(params.klFitMethod)
  params.klFitMethod = 'MaxLikelihood';
end
if ~isfield(params, 'klErrBars') || isempty(params.klErrBars)
  params.klErrBars = false;
end
if ~isfield(params, 'analyzeBandPower') || isempty(params.analyzeBandPower)
  params.analyzeBandPower = true;
end
if ~isfield(params, 'restrictToAwake') || isempty(params.restrictToAwake)
  params.restrictToAwake = true;
end
if ~isfield(params, 'splitByEngagement') || isempty(params.splitByEngagement)
  params.splitByEngagement = false;
end
if ~logical(params.splitByEngagement)
  params.engagementBufferBefore = 0;
  params.engagementBufferAfter = 0;
  params.minNonEngagedWindow = 0;
  params.absorbSingleEvents = false;
  params.minTimeNonEngaged = 0;
end
end
