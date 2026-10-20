function sleepResult = lfp_sleep_detector(sessionType, subjectName, sessionName, varargin)
% LFP_SLEEP_DETECTOR - Awake and sleep epochs from bimodal delta power
%
% Variables:
%   sessionType      - 'spontaneous', 'interval', 'reach', 'semicircle', 'schall'
%   subjectName      - Subject folder; '' when the session type has none
%   sessionName      - Session folder
%   brainArea        - Area whose delta power is scored (default 'M56')
%   minSegmentLength - Shortest kept awake or sleep bout, in seconds (default 120)
%   windowSec        - Sliding-window length for delta power (default 4)
%   stepSec          - Step between window centers, in seconds (default 1)
%   smoothSec        - Running-average length applied before the threshold (default 30).
%                      0 uses the sliding-window power as-is.
%   deltaBand        - [fLow fHigh] in Hz (default [0.5 4])
%   collectStart     - Analysis start (s from session onset; default 0)
%   collectEnd       - Analysis end (s); [] = through the loaded LFP
%   makePlot         - If true (default), plot power, the threshold, and the histogram
%
% Goal:
%   Mark sleep in a spontaneous session so those epochs can be left out of
%   criticality analyses. Delta power is high in sleep and low in wake, and
%   the split is the valley between those two modes. A running average, then
%   a minimum bout length, keeps brief dips and surges from flipping state.
%
% Returns:
%   sleepResult - Struct with:
%     .centerTimes, .deltaPower, .deltaPowerSmooth
%     .sleepThreshold, .sleepThresholdLog10, .thresholdMethod
%     .isSleepRaw  - Above threshold, before the minimum bout length
%     .isSleep     - Final label at each window center
%     .sleepSegments, .awakeSegments - [n x 2] absolute start and end (s)
%     .sleepFraction, .nShortRunsAbsorbed
%     .histogramCenters, .histogramCounts, .modeLog10

if nargin < 3
    error('lfp_sleep_detector:BadInput', ...
        'Usage: lfp_sleep_detector(sessionType, subjectName, sessionName, ...)');
end

sessionType = char(sessionType);
subjectName = char(subjectName);
sessionName = char(sessionName);
if isempty(sessionType) || isempty(sessionName)
    error('lfp_sleep_detector:BadInput', ...
        'sessionType and sessionName must be non-empty.');
end

params = parse_sleep_detector_params(varargin{:});
setup_sleep_detector_paths();

fprintf('\n=== LFP sleep detector ===\n');
fprintf('Session [%s]: %s\n', sessionType, sessionName);
fprintf('Area: %s | delta [%.2f %.2f] Hz\n', ...
    params.brainArea, params.deltaBand(1), params.deltaBand(2));
fprintf('Window %.1f s, step %.1f s, running average %.1f s, min bout %.1f s\n', ...
    params.windowSec, params.stepSec, params.smoothSec, params.minSegmentLength);

lfpStruct = load_delta_lfp(sessionType, subjectName, sessionName, params);
[centerTimes, deltaPower, tStart, tEnd] = sliding_delta_power( ...
    lfpStruct, params.brainArea, params.windowSec, params.stepSec);
deltaPowerSmooth = running_average_power(deltaPower, params.stepSec, params.smoothSec);

[thresholdLog, thresholdMethod, histCenters, histCounts, modeLog] = ...
    delta_bimodal_threshold(deltaPowerSmooth);

logSmooth = log10(deltaPowerSmooth);
isSleepRaw = logSmooth > thresholdLog;
[isSleep, nShortRunsAbsorbed] = enforce_min_segment_length( ...
    isSleepRaw, logSmooth, thresholdLog, params.stepSec, params.minSegmentLength);

[sleepSegments, awakeSegments] = segments_from_labels( ...
    centerTimes, isSleep, params.stepSec, tStart, tEnd);

sessionDuration = tEnd - tStart;
if sessionDuration > 0 && ~isempty(sleepSegments)
    sleepFraction = sum(sleepSegments(:, 2) - sleepSegments(:, 1)) / sessionDuration;
else
    sleepFraction = 0;
end

sleepResult = struct();
sleepResult.sessionType = sessionType;
sleepResult.subjectName = subjectName;
sleepResult.sessionName = sessionName;
sleepResult.brainArea = params.brainArea;
sleepResult.centerTimes = centerTimes;
sleepResult.deltaPower = deltaPower;
sleepResult.deltaPowerSmooth = deltaPowerSmooth;
sleepResult.sleepThreshold = 10 .^ thresholdLog;
sleepResult.sleepThresholdLog10 = thresholdLog;
sleepResult.thresholdMethod = thresholdMethod;
sleepResult.modeLog10 = modeLog;
sleepResult.isSleepRaw = isSleepRaw;
sleepResult.isSleep = isSleep;
sleepResult.sleepSegments = sleepSegments;
sleepResult.awakeSegments = awakeSegments;
sleepResult.sleepFraction = sleepFraction;
sleepResult.nShortRunsAbsorbed = nShortRunsAbsorbed;
sleepResult.histogramCenters = histCenters;
sleepResult.histogramCounts = histCounts;
sleepResult.minSegmentLength = params.minSegmentLength;
sleepResult.windowSec = params.windowSec;
sleepResult.stepSec = params.stepSec;
sleepResult.smoothSec = params.smoothSec;
sleepResult.deltaBand = params.deltaBand;
sleepResult.collectStart = tStart;
sleepResult.collectEnd = tEnd;

fprintf('Threshold: log10(delta) = %.3f (%s)\n', thresholdLog, thresholdMethod);
fprintf('Modes (log10): %.3f and %.3f\n', modeLog(1), modeLog(2));
fprintf('Absorbed %d bout(s) shorter than %.0f s\n', ...
    nShortRunsAbsorbed, params.minSegmentLength);
fprintf('Sleep: %d epoch(s), %.1f%% of %.1f s | Awake: %d epoch(s)\n', ...
    size(sleepSegments, 1), 100 * sleepFraction, sessionDuration, size(awakeSegments, 1));

if params.makePlot
    plot_sleep_detection(sleepResult);
end
end

function params = parse_sleep_detector_params(varargin)
% PARSE_SLEEP_DETECTOR_PARAMS - Name-value settings for the detector
%
% Variables:
%   varargin - Name-value pairs (see lfp_sleep_detector)
%
% Goal:
%   Fill defaults and reject window settings that cannot form a time base.
%
% Returns:
%   params - Struct of detector settings

parser = inputParser;
parser.addParameter('brainArea', 'M56', @(c) (ischar(c) || isstring(c)) && ~isempty(c));
parser.addParameter('minSegmentLength', 60, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('windowSec', 4, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('stepSec', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('smoothSec', 30, @(x) isnumeric(x) && isscalar(x) && x >= 0);
parser.addParameter('deltaBand', [0.5 4], @(x) isnumeric(x) && numel(x) == 2 && x(1) > 0 && x(2) > x(1));
parser.addParameter('collectStart', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
parser.addParameter('collectEnd', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parser.addParameter('makePlot', true, @(x) (islogical(x) || isnumeric(x)) && isscalar(x));
parser.parse(varargin{:});
params = parser.Results;
params.brainArea = char(params.brainArea);
params.deltaBand = double(params.deltaBand(:)');
params.makePlot = logical(params.makePlot);

if params.stepSec > params.windowSec
    error('lfp_sleep_detector:BadWindow', ...
        'Need 0 < stepSec <= windowSec.');
end
if ~isempty(params.collectEnd) && params.collectEnd <= params.collectStart
    error('lfp_sleep_detector:BadWindow', ...
        'collectEnd must be greater than collectStart.');
end
end

function setup_sleep_detector_paths()
% SETUP_SLEEP_DETECTOR_PATHS - Add the loaders this function calls
%
% Goal:
%   Make the function runnable without a prior path setup. srcPath is the
%   folder that contains this file.

srcPath = fileparts(mfilename('fullpath'));
addpath(srcPath);
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
addpath(fullfile(srcPath, 'criticality', 'scripts'));
end

function lfpStruct = load_delta_lfp(sessionType, subjectName, sessionName, params)
% LOAD_DELTA_LFP - Load one session of LFP and its delta-band power
%
% Variables:
%   sessionType, subjectName, sessionName - Session identity
%   params - Detector settings (collect window and delta band)
%
% Goal:
%   Load per-area LFP, skip artifact cleaning, and keep only the delta band.
%   Delta is not highpass-filtered away.
%
% Returns:
%   lfpStruct - Output of load_session_data for dataSource 'lfp'

opts = neuro_behavior_options();
opts.collectStart = params.collectStart;
opts.collectEnd = params.collectEnd;
opts.firingRateCheckTime = [];

lfpCleanParams = struct();
lfpCleanParams.skipArtifactClean = true;
bands = {'delta', params.deltaBand};

loadArgs = [build_session_load_args(sessionType, sessionName, opts, subjectName), ...
    {'lfpCleanParams', lfpCleanParams, 'bands', bands}];
lfpStruct = load_session_data(sessionType, 'lfp', loadArgs{:});

if ~isfield(lfpStruct, 'binnedPower') || isempty(lfpStruct.binnedPower)
    error('lfp_sleep_detector:NoDelta', ...
        'Loaded LFP has no binned delta power.');
end
end

function [centerTimes, deltaPower, tStart, tEnd] = sliding_delta_power( ...
    lfpStruct, brainArea, windowSec, stepSec)
% SLIDING_DELTA_POWER - Mean delta power on a shared window grid
%
% Variables:
%   lfpStruct  - LFP struct with binnedPower and timePoints
%   brainArea  - Area name
%   windowSec  - Full window length (s)
%   stepSec    - Step between window centers (s)
%
% Goal:
%   Same center times as session_lfp_power: each point is the mean delta
%   power in [center - window/2, center + window/2).
%
% Returns:
%   centerTimes - Window centers (s, absolute session time)
%   deltaPower  - Mean delta power at each center
%   tStart      - Start of the loaded LFP (s)
%   tEnd        - End of the loaded LFP (s)

areaIdx = find(strcmp(lfpStruct.areas, brainArea), 1);
if isempty(areaIdx)
    error('lfp_sleep_detector:MissingArea', ...
        'Area "%s" is not in the LFP areas (%s).', ...
        brainArea, strjoin(lfpStruct.areas, ', '));
end

tStart = 0;
if isfield(lfpStruct, 'opts') && isfield(lfpStruct.opts, 'collectStart') ...
        && ~isempty(lfpStruct.opts.collectStart)
    tStart = lfpStruct.opts.collectStart;
end
nSamples = size(lfpStruct.lfpPerArea, 1);
tEnd = tStart + nSamples / lfpStruct.opts.fsLfp;

firstCenter = tStart + windowSec / 2;
lastCenter = tEnd - windowSec / 2;
centerTimes = (firstCenter:stepSec:lastCenter)';
if isempty(centerTimes)
    error('lfp_sleep_detector:NoWindows', ...
        'No windows fit in %.1f s. Shorten windowSec.', tEnd - tStart);
end

binTimes = tStart + lfpStruct.timePoints{areaIdx}{1}(:);
powerVals = lfpStruct.binnedPower{areaIdx}{1}(:);
deltaPower = window_mean_at_centers(binTimes, powerVals, centerTimes, windowSec / 2);
if all(~isfinite(deltaPower) | deltaPower <= 0)
    error('lfp_sleep_detector:NoDelta', ...
        'Delta power is empty for area %s.', brainArea);
end
end

function deltaPowerSmooth = running_average_power(deltaPower, stepSec, smoothSec)
% RUNNING_AVERAGE_POWER - Centered running mean of windowed delta power
%
% Variables:
%   deltaPower - Sliding-window delta power
%   stepSec    - Spacing of deltaPower (s)
%   smoothSec  - Running-average length (s). 0 or < stepSec leaves the series unchanged.
%
% Goal:
%   Smooth brief swings before the histogram and the threshold. Nonpositive
%   samples are ignored, then gaps are filled from the nearest finite sample.
%
% Returns:
%   deltaPowerSmooth - Same size as deltaPower, all finite and positive

deltaPower = deltaPower(:);
deltaPower(~isfinite(deltaPower) | deltaPower <= 0) = nan;
nSmooth = max(1, round(smoothSec / stepSec));
if nSmooth <= 1
    deltaPowerSmooth = deltaPower;
else
    deltaPowerSmooth = movmean(deltaPower, nSmooth, 'omitnan');
end
if all(~isfinite(deltaPowerSmooth))
    error('lfp_sleep_detector:NoDelta', ...
        'Running average of delta power is all non-finite.');
end
deltaPowerSmooth = fillmissing(deltaPowerSmooth, 'nearest');
end

function [thresholdLog, method, histCenters, histCounts, modeLog] = ...
    delta_bimodal_threshold(deltaPowerSmooth)
% DELTA_BIMODAL_THRESHOLD - Split high (sleep) and low (wake) delta power
%
% Variables:
%   deltaPowerSmooth - Positive delta power, one value per window
%
% Goal:
%   Put the threshold in the valley between the two modes of log10 power.
%   If the histogram has no clear valley, use the midpoint between two
%   fitted modes. Sleep is every sample above the threshold.
%
% Returns:
%   thresholdLog - Threshold on log10(delta power)
%   method        - 'valley' or 'twoMeans'
%   histCenters   - Histogram bin centers (log10 power)
%   histCounts    - Smoothed bin counts
%   modeLog       - [lowMode highMode] on log10 power

logPower = log10(deltaPowerSmooth(:));
logPower = logPower(isfinite(logPower));
powerRange = max(logPower) - min(logPower);
if numel(logPower) < 10 || powerRange < 0.05
    error('lfp_sleep_detector:NotBimodal', ...
        'Delta power has too little range to separate sleep from wake.');
end

nBins = 60;
[rawCounts, edges] = histcounts(logPower, nBins);
histCenters = ((edges(1:end-1) + edges(2:end)) / 2)';
rawCounts = rawCounts(:);
smoothCounts = movmean(rawCounts, 5);

[foundValley, thresholdLog, modeLog] = valley_between_modes(histCenters, smoothCounts);
histCounts = rawCounts;
if foundValley
    method = 'valley';
    return
end

[thresholdLog, modeLog, modesOk] = two_means_threshold(logPower, modeLog);
if ~modesOk
    error('lfp_sleep_detector:NotBimodal', ...
        ['Delta power is not clearly bimodal (no valley, and the two fitted ', ...
        'modes are not separated). Check the area, the delta band, or smoothSec.']);
end
method = 'twoMeans';
fprintf(['Delta histogram had no clear valley. Threshold is the midpoint ', ...
    'between two fitted modes.\n']);
end

function [foundValley, thresholdLog, modeLog] = valley_between_modes(histCenters, histCounts)
% VALLEY_BETWEEN_MODES - Antimode between the two tallest separated peaks
%
% Variables:
%   histCenters - Bin centers (log10 power)
%   histCounts  - Smoothed counts
%
% Goal:
%   Accept a valley only when the peaks are separated and the counts dip
%   between them. The low-power peak is wake and the high-power peak is sleep.
%
% Returns:
%   foundValley  - True when the valley is usable
%   thresholdLog - Bin center at the valley
%   modeLog      - [lowPeak highPeak]

foundValley = false;
thresholdLog = nan;
modeLog = [nan nan];
nBins = numel(histCounts);
if nBins < 5
    return
end

isPeak = false(nBins, 1);
isPeak(2:end-1) = histCounts(2:end-1) >= histCounts(1:end-2) ...
    & histCounts(2:end-1) > histCounts(3:end);
peakIdx = find(isPeak);
if numel(peakIdx) < 2
    return
end

[~, order] = sort(histCounts(peakIdx), 'descend');
peakIdx = peakIdx(order);
firstIdx = peakIdx(1);
secondIdx = [];
minSeparation = 0.2;
for k = 2:numel(peakIdx)
    if abs(histCenters(peakIdx(k)) - histCenters(firstIdx)) >= minSeparation
        secondIdx = peakIdx(k);
        break
    end
end
if isempty(secondIdx)
    return
end

iLeft = min(firstIdx, secondIdx);
iRight = max(firstIdx, secondIdx);
modeLog = sort(histCenters([iLeft iRight]));
span = histCounts(iLeft:iRight);
[valleyCount, rel] = min(span);
peakHeight = min(histCounts(iLeft), histCounts(iRight));
tol = max(1, 0.02 * peakHeight);
atFloor = span <= valleyCount + tol;
runStart = rel;
while runStart > 1 && atFloor(runStart - 1)
    runStart = runStart - 1;
end
runEnd = rel;
while runEnd < numel(atFloor) && atFloor(runEnd + 1)
    runEnd = runEnd + 1;
end
valleyIdx = iLeft + round((runStart + runEnd) / 2) - 1;
if valleyIdx <= iLeft || valleyIdx >= iRight || valleyCount >= 0.85 * peakHeight
    return
end

foundValley = true;
thresholdLog = histCenters(valleyIdx);
end

function [thresholdLog, modeLog, modesOk] = two_means_threshold(logPower, initModes)
% TWO_MEANS_THRESHOLD - Midpoint between two 1D clusters of log delta power
%
% Variables:
%   logPower  - Finite log10(delta power) samples
%   initModes - [low high] peak locations when the histogram found two peaks;
%               empty or non-finite falls back to the 25th and 75th percentiles
%
% Goal:
%   Fallback when the histogram valley is unclear. Both clusters must hold
%   at least 5% of the samples and sit at least 0.2 log10 units apart.
%
% Returns:
%   thresholdLog - Midpoint between the low and high cluster centers
%   modeLog      - [lowCenter highCenter]
%   modesOk      - True when the split is usable

logPower = sort(logPower(:));
nSamples = numel(logPower);
cLow = logPower(max(1, round(0.25 * nSamples)));
cHigh = logPower(max(1, round(0.75 * nSamples)));
if nargin >= 2 && numel(initModes) == 2 && all(isfinite(initModes))
    cLow = min(initModes);
    cHigh = max(initModes);
end
for iPass = 1:40
    inHigh = abs(logPower - cHigh) < abs(logPower - cLow);
    if ~any(inHigh) || ~any(~inHigh)
        break
    end
    newLow = mean(logPower(~inHigh));
    newHigh = mean(logPower(inHigh));
    if abs(newLow - cLow) < 1e-6 && abs(newHigh - cHigh) < 1e-6
        cLow = newLow;
        cHigh = newHigh;
        break
    end
    cLow = newLow;
    cHigh = newHigh;
end

fracHigh = mean(inHigh);
modesOk = cHigh > cLow && (cHigh - cLow) >= 0.2 ...
    && fracHigh >= 0.05 && fracHigh <= 0.95;
thresholdLog = (cLow + cHigh) / 2;
modeLog = [cLow cHigh];
end

function [isSleep, nAbsorbed] = enforce_min_segment_length( ...
    isSleep, logPower, thresholdLog, stepSec, minSegmentLength)
% ENFORCE_MIN_SEGMENT_LENGTH - Drop awake and sleep bouts that are too short
%
% Variables:
%   isSleep          - Logical label at each window (true = sleep)
%   logPower         - log10 delta power, used only to break ties
%   thresholdLog     - Sleep threshold on logPower
%   stepSec          - Seconds per label
%   minSegmentLength - Minimum bout length (s)
%
% Goal:
%   Repeatedly fold the shortest too-short bout into its neighbor. A brief
%   dip during sleep becomes sleep, and a brief surge during wake becomes
%   wake. The loop stops when every remaining bout is long enough, or the
%   session is a single bout.
%
% Returns:
%   isSleep   - Labels after short bouts are absorbed
%   nAbsorbed - How many short bouts were absorbed

isSleep = isSleep(:);
logPower = logPower(:);
minBins = max(1, round(minSegmentLength / stepSec));
nAbsorbed = 0;
nLabels = numel(isSleep);

for iPass = 1:nLabels
    [starts, ends, states] = label_runs(isSleep);
    if numel(starts) <= 1
        break
    end
    lengths = ends - starts + 1;
    short = find(lengths < minBins);
    if isempty(short)
        break
    end

    shortest = min(lengths(short));
    candidates = short(lengths(short) == shortest);
    if numel(candidates) == 1
        runIdx = candidates;
    else
        margin = zeros(numel(candidates), 1);
        for k = 1:numel(candidates)
            idx = starts(candidates(k)):ends(candidates(k));
            margin(k) = mean(abs(logPower(idx) - thresholdLog), 'omitnan');
        end
        [~, kBest] = min(margin);
        runIdx = candidates(kBest);
    end

    if runIdx == 1
        newState = states(2);
    else
        newState = states(runIdx - 1);
    end
    isSleep(starts(runIdx):ends(runIdx)) = newState;
    nAbsorbed = nAbsorbed + 1;
end
end

function [starts, ends, states] = label_runs(isSleep)
% LABEL_RUNS - Contiguous blocks of one sleep/wake label
%
% Variables:
%   isSleep - Logical column
%
% Goal:
%   Return the first index, last index, and label of each run.
%
% Returns:
%   starts, ends - Index columns
%   states       - Logical column, true where the run is sleep

isSleep = isSleep(:);
changeIdx = find(diff(isSleep) ~= 0) + 1;
starts = [1; changeIdx];
ends = [changeIdx - 1; numel(isSleep)];
states = isSleep(starts);
end

function [sleepSegments, awakeSegments] = segments_from_labels( ...
    centerTimes, isSleep, stepSec, tStart, tEnd)
% SEGMENTS_FROM_LABELS - Absolute start and end of each awake and sleep bout
%
% Variables:
%   centerTimes - Window centers (s)
%   isSleep     - Final label at each center
%   stepSec     - Spacing of centers (s)
%   tStart      - Start of the loaded LFP (s)
%   tEnd        - End of the loaded LFP (s)
%
% Goal:
%   Cover the whole loaded interval. Each center stands for one step, and
%   the first and last bouts extend to tStart and tEnd.
%
% Returns:
%   sleepSegments, awakeSegments - [n x 2] start and end (s). Empty is 0 x 2.

halfStep = stepSec / 2;
[starts, ends, states] = label_runs(isSleep);
nRuns = numel(starts);
allSegs = zeros(nRuns, 2);
for r = 1:nRuns
    allSegs(r, 1) = centerTimes(starts(r)) - halfStep;
    allSegs(r, 2) = centerTimes(ends(r)) + halfStep;
end
allSegs(1, 1) = tStart;
allSegs(end, 2) = tEnd;

sleepSegments = allSegs(states, :);
awakeSegments = allSegs(~states, :);
if isempty(sleepSegments)
    sleepSegments = zeros(0, 2);
end
if isempty(awakeSegments)
    awakeSegments = zeros(0, 2);
end
end

function yMean = window_mean_at_centers(binTimes, values, centerTimes, halfWin)
% WINDOW_MEAN_AT_CENTERS - Mean of samples whose times fall in each window
%
% Variables:
%   binTimes    - Sample times (s), ascending
%   values      - Sample values, same length as binTimes
%   centerTimes - Window centers (s)
%   halfWin     - Half window (s)
%
% Goal:
%   Average delta-power bins inside [center - halfWin, center + halfWin).

yMean = nan(numel(centerTimes), 1);
if isempty(binTimes) || isempty(values)
    return
end
nKeep = min(numel(binTimes), numel(values));
binTimes = binTimes(1:nKeep);
values = values(1:nKeep);
[binTimes, sortIdx] = sort(binTimes);
values = values(sortIdx);
finite = isfinite(values);
values(~finite) = 0;
cs = [0; cumsum(values)];
csCount = [0; cumsum(finite)];
iLeft = lookup_first_bin_at_or_after(binTimes, centerTimes - halfWin);
iRight = lookup_first_bin_at_or_after(binTimes, centerTimes + halfWin);
nIn = csCount(iRight) - csCount(iLeft);
valid = nIn > 0;
sumIn = cs(iRight(valid)) - cs(iLeft(valid));
yMean(valid) = sumIn ./ nIn(valid);
end

function idx = lookup_first_bin_at_or_after(sampleTimes, queryTimes)
% LOOKUP_FIRST_BIN_AT_OR_AFTER - First index with time >= each query
%
% Variables:
%   sampleTimes - Ascending sample times (duplicates allowed)
%   queryTimes  - Query times, one per window
%
% Goal:
%   Left edge of a half-open window. Queries after the last sample return
%   numel(sampleTimes)+1 so a difference of edges is a count.

nTimes = numel(sampleTimes);
queryTimes = queryTimes(:);
idx = repmat(nTimes + 1, numel(queryTimes), 1);
if nTimes == 0
    return
end
sampleTimes = sampleTimes(:);
for iQuery = 1:numel(queryTimes)
    q = queryTimes(iQuery);
    if sampleTimes(1) >= q
        idx(iQuery) = 1;
        continue
    end
    if sampleTimes(nTimes) < q
        continue
    end
    lo = 1;
    hi = nTimes;
    while lo < hi
        mid = floor((lo + hi) / 2);
        if sampleTimes(mid) < q
            lo = mid + 1;
        else
            hi = mid;
        end
    end
    idx(iQuery) = lo;
end
end

function plot_sleep_detection(sleepResult)
% PLOT_SLEEP_DETECTION - Delta power, sleep bouts, and the bimodal histogram
%
% Variables:
%   sleepResult - Output of lfp_sleep_detector
%
% Goal:
%   Top: log10 delta power, the threshold, and final sleep bouts.
%   Bottom: the histogram the threshold was taken from.

logRaw = log10(sleepResult.deltaPower);
logSmooth = log10(sleepResult.deltaPowerSmooth);
logRaw(~isfinite(logRaw)) = nan;

yVals = [logRaw; logSmooth; sleepResult.sleepThresholdLog10];
yVals = yVals(isfinite(yVals));
yPad = 0.05 * (max(yVals) - min(yVals));
if yPad == 0
    yPad = 0.1;
end
yLim = [min(yVals) - yPad, max(yVals) + yPad];

fig = figure('Color', 'w', 'Position', [120 80 960 640], ...
    'Name', sprintf('LFP sleep | %s | %s', sleepResult.sessionName, sleepResult.brainArea));
tileLayout = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

axTime = nexttile(tileLayout);
hold(axTime, 'on');
for iSeg = 1:size(sleepResult.sleepSegments, 1)
    t0 = sleepResult.sleepSegments(iSeg, 1);
    t1 = sleepResult.sleepSegments(iSeg, 2);
    patch(axTime, [t0 t1 t1 t0], [yLim(1) yLim(1) yLim(2) yLim(2)], ...
        [0.45 0.20 0.70], 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
end
plot(axTime, sleepResult.centerTimes, logRaw, 'Color', [0.65 0.65 0.65], ...
    'LineWidth', 0.8, 'DisplayName', 'Window delta');
plot(axTime, sleepResult.centerTimes, logSmooth, 'Color', [0.45 0.20 0.70], ...
    'LineWidth', 1.4, 'DisplayName', 'Running average');
yline(axTime, sleepResult.sleepThresholdLog10, '--', 'Color', [0.15 0.15 0.15], ...
    'LineWidth', 1.1, 'DisplayName', 'Sleep threshold');
hold(axTime, 'off');
ylim(axTime, yLim);
ylabel(axTime, 'log_{10}(delta power)', 'Interpreter', 'tex');
title(axTime, sprintf('%s | sleep %.1f%% | min bout %.0f s', ...
    sleepResult.brainArea, 100 * sleepResult.sleepFraction, ...
    sleepResult.minSegmentLength), 'Interpreter', 'none');
legend(axTime, 'Location', 'northeast');
set(axTime, 'Box', 'off', 'TickDir', 'out', 'FontSize', 11);
grid(axTime, 'on');

axHist = nexttile(tileLayout);
bar(axHist, sleepResult.histogramCenters, sleepResult.histogramCounts, 1, ...
    'FaceColor', [0.75 0.75 0.82], 'EdgeColor', 'none');
hold(axHist, 'on');
xline(axHist, sleepResult.sleepThresholdLog10, '--', 'Color', [0.15 0.15 0.15], ...
    'LineWidth', 1.1);
if all(isfinite(sleepResult.modeLog10))
    xline(axHist, sleepResult.modeLog10(1), ':', 'Color', [0.20 0.40 0.85], 'LineWidth', 1.1);
    xline(axHist, sleepResult.modeLog10(2), ':', 'Color', [0.45 0.20 0.70], 'LineWidth', 1.1);
end
hold(axHist, 'off');
xlabel(axHist, 'log_{10}(delta power)', 'Interpreter', 'tex');
ylabel(axHist, 'Windows');
title(axHist, sprintf('Threshold method: %s', sleepResult.thresholdMethod), ...
    'Interpreter', 'none');
set(axHist, 'Box', 'off', 'TickDir', 'out', 'FontSize', 11);

sgtitle(tileLayout, sprintf('%s | %s | shaded = sleep', ...
    sleepResult.sessionName, sleepResult.sessionType), 'Interpreter', 'none');
end
