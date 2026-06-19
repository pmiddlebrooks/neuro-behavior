%% Waveforms sandbox
% Load spike waveforms for the session selected in the workspace.
%
% Prerequisites:
%   Run src/data_prep/choose_task_and_session.m first so these variables exist:
%     sessionType, sessionName, subjectName (required for spontaneous/interval)
%     paths, opts (optional; defaults are created if missing)
%
% Goal:
%   Open waveforms.mat from the same folder as cluster_info.tsv / spike_times.npy.

%% Paths
scriptDir = fileparts(mfilename('fullpath'));
if exist(scriptDir, 'dir')
    addpath(scriptDir);
end

if ~exist('paths', 'var') || isempty(paths)
    paths = get_paths;
end
if ~exist('opts', 'var') || isempty(opts)
    opts = neuro_behavior_options;
end

%% Session info from workspace
if ~exist('sessionType', 'var') || isempty(sessionType)
    error(['Define sessionType in the workspace — run ', ...
        'src/data_prep/choose_task_and_session.m first.']);
end
if ~exist('sessionName', 'var') || isempty(sessionName)
    error(['Define sessionName in the workspace — run ', ...
        'src/data_prep/choose_task_and_session.m first.']);
end

if any(strcmpi(sessionType, {'spontaneous', 'interval'}))
    if ~exist('subjectName', 'var') || isempty(subjectName)
        error(['subjectName is required for %s sessions. ', ...
            'Set it in choose_task_and_session.m.'], sessionType);
    end
else
    error(['waveforms_sandbox currently supports spontaneous and interval sessions only. ', ...
        'Got sessionType = %s.'], sessionType);
end

%% Resolve session folder and load waveforms
sessionFolder = get_waveform_session_folder(sessionType, paths, subjectName, sessionName);
waveformsFile = fullfile(sessionFolder, 'waveforms.mat');

if ~isfile(waveformsFile)
    error('waveforms.mat not found in %s', sessionFolder);
end

fprintf('Loading waveforms from:\n  %s\n', waveformsFile);
waveformData = load(waveformsFile);

if ~isfield(waveformData, 'sp_waveforms')
    error('Expected variable sp_waveforms in %s', waveformsFile);
end

sp_waveforms = waveformData.sp_waveforms;
nUnits = numel(sp_waveforms);
fprintf('%d waveform entries loaded (fsSpike = %g Hz)\n', nUnits, opts.fsSpike);

ci = readtable(fullfile(sessionFolder, 'cluster_info.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

spikeTimes = double(readNPY(fullfile(sessionFolder, 'spike_times.npy'))) / opts.fsSpike;
spikeClusters = readNPY(fullfile(sessionFolder, 'spike_clusters.npy'));

collectStart = 0;
if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
    collectStart = opts.collectStart;
end
collectEnd = spikeTimes(end);
if isfield(opts, 'collectEnd') && ~isempty(opts.collectEnd)
    collectEnd = opts.collectEnd;
end
recordingDurationSec = collectEnd - collectStart;

%% Waveform width and firing rate per unit
unitIds = zeros(nUnits, 1);
waveformWidthMs = nan(nUnits, 1);
fwhmMs = nan(nUnits, 1);
unitFiringRateHz = nan(nUnits, 1);

for iUnit = 1:nUnits
    unitId = sp_waveforms(iUnit).unitID;
    meanWf = get_unit_mean_waveform(sp_waveforms(iUnit), ci);
    if isempty(meanWf)
        continue;
    end

    unitIds(iUnit) = unitId;
    waveformWidthMs(iUnit) = compute_peak_to_trough(meanWf, opts.fsSpike);
    fwhmMs(iUnit) = compute_fwhm(meanWf, opts.fsSpike);

    unitSpikeTimes = spikeTimes(spikeClusters == unitId);
    unitSpikeTimes = unitSpikeTimes(unitSpikeTimes >= collectStart & unitSpikeTimes <= collectEnd);
    unitFiringRateHz(iUnit) = numel(unitSpikeTimes) / recordingDurationSec;
end

validWidth = ~isnan(waveformWidthMs);
validScatter = validWidth & ~isnan(unitFiringRateHz);
fprintf('%d / %d units with valid peak-to-trough width\n', sum(validWidth), nUnits);
fprintf('Firing rates computed over %.1f–%.1f s (duration = %.1f s)\n', ...
    collectStart, collectEnd, recordingDurationSec);

widthEdges = 0.02:0.02:1.2;
figure('Name', 'waveforms_sandbox_widths');
histogram(waveformWidthMs(validWidth), widthEdges, ...
    'FaceColor', [0.35 0.55 0.85], 'EdgeColor', 'w');
xlabel('Peak-to-trough width (ms)');
ylabel('Unit count');
title(sprintf('%s | %s — mean waveform widths (n = %d)', ...
    subjectName, sessionName, sum(validWidth)));
grid on;

figure('Name', 'waveforms_sandbox_fr_vs_width');
scatter(waveformWidthMs(validScatter), unitFiringRateHz(validScatter), ...
    18, [0.25 0.25 0.25], 'filled', 'MarkerFaceAlpha', 0.55);
xlabel('Peak-to-trough width (ms)');
ylabel('Firing rate (Hz)');
title(sprintf('%s | %s — firing rate vs waveform width (n = %d)', ...
    subjectName, sessionName, sum(validScatter)));
grid on;

%% Quick look (edit below for your analyses)
% Example: plot mean waveform for the first unit on its primary channel
if nUnits > 0
    meanWf = get_unit_mean_waveform(sp_waveforms(1), ci);
    tMs = (0:numel(meanWf) - 1) / opts.fsSpike * 1e3;

    figure('Name', 'waveforms_sandbox_example');
    plot(tMs, meanWf, 'k', 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('Amplitude (a.u.)');
    title(sprintf('%s | %s | unit %d', subjectName, sessionName, sp_waveforms(1).unitID));
    grid on;
end

function sessionFolder = get_waveform_session_folder(sessionType, paths, subjectName, sessionName)
% GET_WAVEFORM_SESSION_FOLDER - Resolve folder containing waveforms.mat
%
% Variables:
%   sessionType - 'spontaneous' or 'interval'
%   paths       - struct from get_paths
%   subjectName - subject folder under the task data root
%   sessionName - session folder under subject
%
% Goal:
%   Return the spike-data session folder (same layout as load_data.m).

switch lower(sessionType)
    case 'spontaneous'
        basePath = paths.spontaneousDataPath;
    case 'interval'
        basePath = paths.intervalDataPath;
    otherwise
        error('Unsupported sessionType: %s', sessionType);
end

sessionFolder = fullfile(basePath, subjectName, sessionName);

if ~isfolder(sessionFolder)
    error('Session folder not found: %s', sessionFolder);
end
end

function meanWf = get_unit_mean_waveform(unitEntry, ci)
% GET_UNIT_MEAN_WAVEFORM - Mean waveform on the cluster's assigned channel
%
% Variables:
%   unitEntry - one element of sp_waveforms
%   ci        - cluster_info table for the session
%
% Goal:
%   Return the 1 x nSamples mean waveform used for width metrics.

ciIdx = find(ci.cluster_id == unitEntry.unitID, 1, 'first');
if isempty(ciIdx)
    meanWf = [];
    return;
end

chIdx = find(unitEntry.channels == ci.ch(ciIdx), 1, 'first');
if isempty(chIdx)
    chIdx = 1;
end

mw = unitEntry.mean_wf;
if ndims(mw) == 3
    meanWf = reshape(mw(1, chIdx, :), 1, []);
else
    meanWf = mw(chIdx, :);
end
end

function fwhm = compute_fwhm(waveform, sampleRate)
% COMPUTE_FWHM - Full width at half maximum of a spike waveform (ms)
%
% Variables:
%   waveform   - 1D spike waveform
%   sampleRate - sampling rate in Hz
%
% Goal:
%   Return FWHM in milliseconds (NaN if crossings are ambiguous).

[~, peakIdx] = max(abs(waveform));
peakVal = waveform(peakIdx);
halfMax = 0.5 * peakVal;

crossings = [];
for i = 1:numel(waveform) - 1
    if (waveform(i) - halfMax) * (waveform(i + 1) - halfMax) < 0
        x1 = i;
        y1 = waveform(i);
        y2 = waveform(i + 1);
        crossings(end + 1) = x1 + (halfMax - y1) / (y2 - y1); %#ok<AGROW>
    end
end

pre = crossings(crossings < peakIdx);
post = crossings(crossings > peakIdx);
if isempty(pre) || isempty(post)
    fwhm = NaN;
    return;
end

fwhm = (post(1) - pre(end)) / sampleRate * 1e3;
end

function ptDuration = compute_peak_to_trough(waveform, sampleRate)
% COMPUTE_PEAK_TO_TROUGH - Peak-to-trough duration of a spike waveform (ms)
%
% Variables:
%   waveform   - 1D spike waveform
%   sampleRate - sampling rate in Hz
%
% Goal:
%   Return spike width in milliseconds for excitatory vs inhibitory screening.

[~, peakIdx] = max(abs(waveform));
peakVal = waveform(peakIdx);

if peakVal < 0
    postSegment = waveform(peakIdx + 1:end);
    [~, relIdx] = max(postSegment);
    troughIdx = peakIdx;
    peakIdx = peakIdx + relIdx;
else
    postSegment = waveform(peakIdx + 1:end);
    [~, relIdx] = min(postSegment);
    troughIdx = peakIdx + relIdx;
end

ptDuration = abs(peakIdx - troughIdx) / sampleRate * 1e3;
end
