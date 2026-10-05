function av = detect_klaus_avalanches(signal, samplePeriodSec, nSd, mode, deltaT)
% DETECT_KLAUS_AVALANCHES - Avalanche sizes and durations, Klaus / Plenz nLFP rule
%
% Variables:
%   signal          - Column vector. Continuous LFP ('nlfp') or one binned
%                     power band ('binnedPower').
%   samplePeriodSec - Seconds per sample. For 'nlfp' this is 1/fs. For
%                     'binnedPower' this is the power-bin width and is also
%                     the avalanche bin Δt.
%   nSd             - Threshold in standard deviations (Klaus-style cutoff;
%                     session_lfp_criticality uses 2.5).
%   mode            - 'nlfp' or 'binnedPower'
%   deltaT          - Optional bin width (s) for 'nlfp'. [] uses the mean
%                     inter-event interval, the Plenz / Klaus default.
%
% Goal:
%   Continuous LFP ('nlfp'), as in Klaus, Plenz, and colleagues:
%     1. Zero-mean the trace.
%     2. Mark negative peaks at or below -nSd * SD (nLFP events).
%     3. Event size = |peak| / threshold, so the smallest event has size 1.
%     4. Bin events at Δt = mean inter-event interval (unless deltaT is set).
%     5. An avalanche is a run of occupied bins bounded by empty bins.
%     6. Size = sum of event sizes in the run. Duration = number of bins.
%   Binned band power ('binnedPower') uses the same cutoff on the discrete
%   series: bins at or above mean + nSd * SD are occupied, size is the sum of
%   (power / threshold) along each run, and Δt is the power-bin width.
%
% Returns:
%   av - Struct with sizes, durations (bins), durationsSec, deltaT, threshold,
%        nEvents, nAvalanches, fractionActive, and mode.

if nargin < 4 || isempty(mode)
    mode = 'nlfp';
end
if nargin < 5
    deltaT = [];
end

mode = normalize_klaus_mode(mode);
av = empty_klaus_avalanche_result(mode);

signal = signal(:);
signal = signal(isfinite(signal));
if numel(signal) < 3 || ~(isfinite(samplePeriodSec) && samplePeriodSec > 0)
    return;
end
if ~(isfinite(nSd) && nSd > 0)
    return;
end

switch mode
    case 'nlfp'
        av = detect_nlfp_avalanches(signal, samplePeriodSec, nSd, deltaT);
    case 'binnedPower'
        av = detect_binned_power_avalanches(signal, samplePeriodSec, nSd);
    otherwise
        error('detect_klaus_avalanches:BadMode', ...
            'mode must be ''nlfp'' or ''binnedPower'' (got %s).', mode);
end
end

function av = detect_nlfp_avalanches(signal, samplePeriodSec, nSd, deltaT)
% DETECT_NLFP_AVALANCHES - Negative-peak avalanches on a continuous LFP trace
%
% Variables:
%   signal          - Continuous LFP samples
%   samplePeriodSec - 1/fs (seconds)
%   nSd             - Negative-peak cutoff in SD
%   deltaT          - Bin width (s); [] = mean inter-event interval
%
% Goal:
%   Klaus / Plenz nLFP detection on one trace, then empty-bin avalanche grouping.

av = empty_klaus_avalanche_result('nlfp');

signal = signal - mean(signal);
sigma = std(signal);
if ~(isfinite(sigma) && sigma > 0)
    return;
end

threshold = nSd * sigma;
% Peaks of -signal at or above the threshold are LFP troughs at or below -nSd.
[~, peakLocs] = findpeaks(-signal, 'MinPeakHeight', threshold);
peakLocs = peakLocs(:);
if numel(peakLocs) < 2
    av.threshold = threshold;
    av.nEvents = numel(peakLocs);
    return;
end

eventSizes = abs(signal(peakLocs)) / threshold;
eventTimes = (peakLocs - 1) * samplePeriodSec;

if isempty(deltaT) || ~(isfinite(deltaT) && deltaT > 0)
    deltaT = mean(diff(eventTimes));
end
if ~(isfinite(deltaT) && deltaT > 0)
    return;
end

binIdx = floor((eventTimes - eventTimes(1)) / deltaT) + 1;
nBins = max(binIdx);
activity = accumarray(binIdx, eventSizes, [nBins, 1]);
[sizes, durations] = avalanche_runs_from_activity(activity);

av.sizes = sizes;
av.durations = durations;
av.durationsSec = durations * deltaT;
av.deltaT = deltaT;
av.threshold = threshold;
av.nEvents = numel(peakLocs);
av.nAvalanches = numel(sizes);
av.fractionActive = mean(activity > 0);
end

function av = detect_binned_power_avalanches(signal, binWidthSec, nSd)
% DETECT_BINNED_POWER_AVALANCHES - Suprathreshold runs in one binned power band
%
% Variables:
%   signal      - Binned band power (one value per bin)
%   binWidthSec - Power bin width (s); this is Δt
%   nSd         - Cutoff above the mean, in SD of the binned series
%
% Goal:
%   Apply the same nSd cutoff Klaus uses for nLFP, on an already-binned
%   positive power trace. Occupied bins are those at or above mean + nSd * SD.

av = empty_klaus_avalanche_result('binnedPower');
av.deltaT = binWidthSec;

mu = mean(signal);
sigma = std(signal);
if ~(isfinite(sigma) && sigma > 0)
    return;
end

threshold = mu + nSd * sigma;
if ~(isfinite(threshold) && threshold > 0)
    return;
end

active = signal >= threshold;
activity = zeros(size(signal));
activity(active) = signal(active) / threshold;
[sizes, durations] = avalanche_runs_from_activity(activity);

av.sizes = sizes;
av.durations = durations;
av.durationsSec = durations * binWidthSec;
av.threshold = threshold;
av.nEvents = sum(active);
av.nAvalanches = numel(sizes);
av.fractionActive = mean(active);
end

function [sizes, durations] = avalanche_runs_from_activity(activity)
% AVALANCHE_RUNS_FROM_ACTIVITY - Sum of activity and length of each occupied run
%
% Variables:
%   activity - Non-negative time series; empty bins are exactly 0
%
% Goal:
%   Group consecutive nonzero bins into avalanches. Size is the sum of
%   activity; duration is the number of bins in the run.

activity = activity(:);
occupied = activity > 0;
edges = diff([false; occupied; false]);
runStarts = find(edges == 1);
runEnds = find(edges == -1) - 1;
nRuns = numel(runStarts);
sizes = zeros(nRuns, 1);
durations = zeros(nRuns, 1);
for iRun = 1:nRuns
    sizes(iRun) = sum(activity(runStarts(iRun):runEnds(iRun)));
    durations(iRun) = runEnds(iRun) - runStarts(iRun) + 1;
end
end

function mode = normalize_klaus_mode(mode)
% NORMALIZE_KLAUS_MODE - Canonical 'nlfp' or 'binnedPower'

mode = lower(strtrim(char(mode)));
mode = strrep(mode, '-', '');
mode = strrep(mode, '_', '');
switch mode
    case {'nlfp', 'raw', 'rawlfp', 'continuous'}
        mode = 'nlfp';
    case {'binnedpower', 'power', 'bandpower', 'binned'}
        mode = 'binnedPower';
    otherwise
        error('detect_klaus_avalanches:BadMode', ...
            'mode must be ''nlfp'' or ''binnedPower'' (got %s).', mode);
end
end

function av = empty_klaus_avalanche_result(mode)
% EMPTY_KLAUS_AVALANCHE_RESULT - Result struct before any events are found

av = struct();
av.mode = mode;
av.sizes = zeros(0, 1);
av.durations = zeros(0, 1);
av.durationsSec = zeros(0, 1);
av.deltaT = nan;
av.threshold = nan;
av.nEvents = 0;
av.nAvalanches = 0;
av.fractionActive = nan;
end
