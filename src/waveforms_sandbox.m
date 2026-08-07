%% Waveforms sandbox — spontaneous waveform vs neuronID overlap
% Opens an example spontaneous session and reports how waveforms.mat unitIDs
% match the analysis neuronIDs from load_spike_times (cluster_info good/mua
% + area + firing-rate filters).
%
% Goal:
%   Print:
%     - waveforms accounted for in neuronIDs
%     - waveforms with no matching neuronID
%     - neuronIDs with no corresponding waveform

%% Paths

paths = get_paths;
opts = neuro_behavior_options;

% Example spontaneous session
sessionType = 'spontaneous';
subjectName = 'ag25290';
sessionName = 'ag112321_1';

opts.subjectName = subjectName;
opts.sessionName = sessionName;
opts.collectStart = 0;
opts.collectEnd = [];

fprintf('\n=== Waveform vs neuronID overlap ===\n');
fprintf('Session: %s | %s (%s)\n', subjectName, sessionName, sessionType);

%% Load waveforms (sp_waveforms.unitID)
waveformContext = load_kilosort_waveform_context(sessionType, paths, subjectName, sessionName, opts);
waveformUnitIds = unique(waveformContext.unitIds(:));
waveformUnitIds = waveformUnitIds(~isnan(waveformUnitIds));

fprintf('Waveforms file:\n  %s\n', waveformContext.waveformsFile);
fprintf('Waveform entries: %d (unique unitIDs: %d)\n', ...
    waveformContext.nUnits, numel(waveformUnitIds));

%% Load analysis neuronIDs (same path as criticality manuscript pipeline)
spikeData = load_spike_times(sessionType, paths, sessionName, opts);
neuronIds = unique(spikeData.neuronIDs(:));

fprintf('neuronIDs (after area / quality / FR filters): %d\n', numel(neuronIds));

%% Overlap
wfInNeuronIds = ismember(waveformUnitIds, neuronIds);
neuronIdsWithWf = ismember(neuronIds, waveformUnitIds);

nWfMatched = sum(wfInNeuronIds);
nWfUnmatched = sum(~wfInNeuronIds);
nNeuronMissingWf = sum(~neuronIdsWithWf);

fprintf('\n--- Match summary ---\n');
fprintf('Waveforms accounted for in neuronIDs:     %d / %d\n', ...
    nWfMatched, numel(waveformUnitIds));
fprintf('Waveforms NOT in neuronIDs:               %d / %d\n', ...
    nWfUnmatched, numel(waveformUnitIds));
fprintf('neuronIDs with no corresponding waveform: %d / %d\n', ...
    nNeuronMissingWf, numel(neuronIds));

if nWfUnmatched > 0
    fprintf('\nExample waveform unitIDs missing from neuronIDs (up to 20):\n');
    disp(waveformUnitIds(~wfInNeuronIds).');
end
if nNeuronMissingWf > 0
    fprintf('Example neuronIDs missing waveforms (up to 20):\n');
    missingIds = neuronIds(~neuronIdsWithWf);
    disp(missingIds(1:min(20, end)).');
end

function waveformContext = load_kilosort_waveform_context(sessionType, paths, subjectName, sessionName, opts)
% LOAD_KILOSORT_WAVEFORM_CONTEXT - Waveforms from spontaneous/interval session folder

sessionFolder = get_kilosort_session_folder(sessionType, paths, subjectName, sessionName);
waveformsFile = fullfile(sessionFolder, 'waveforms.mat');
if ~isfile(waveformsFile)
    error('waveforms.mat not found in %s', sessionFolder);
end

waveformData = load(waveformsFile);
if ~isfield(waveformData, 'sp_waveforms')
    error('Expected variable sp_waveforms in %s', waveformsFile);
end

ci = readtable(fullfile(sessionFolder, 'cluster_info.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

spikeTimes = double(readNPY(fullfile(sessionFolder, 'spike_times.npy'))) / opts.fsSpike;
spikeClusters = readNPY(fullfile(sessionFolder, 'spike_clusters.npy'));

[collectStart, collectEnd] = get_collect_window(opts, spikeTimes(end));
spWaveforms = waveformData.sp_waveforms;
nUnits = numel(spWaveforms);

unitIds = zeros(nUnits, 1);
meanWaveforms = cell(nUnits, 1);
for iUnit = 1:nUnits
    unitIds(iUnit) = spWaveforms(iUnit).unitID;
    meanWaveforms{iUnit} = get_kilosort_unit_mean_waveform(spWaveforms(iUnit), ci);
end

waveformContext = struct();
waveformContext.source = 'kilosort';
waveformContext.sessionLabel = sprintf('%s | %s', subjectName, sessionName);
waveformContext.waveformsFile = waveformsFile;
waveformContext.fsSpike = opts.fsSpike;
waveformContext.wfSampleOffset = 0;
waveformContext.nUnits = nUnits;
waveformContext.unitIds = unitIds;
waveformContext.meanWaveforms = meanWaveforms;
waveformContext.spikeTimes = spikeTimes;
waveformContext.spikeClusters = spikeClusters;
waveformContext.collectStart = collectStart;
waveformContext.collectEnd = collectEnd;
end

function waveformContext = load_reach_waveform_context(paths, sessionName, opts)
% LOAD_REACH_WAVEFORM_CONTEXT - Waveforms from reach_task/data/WaveformDATA

waveformsFile = get_reach_waveform_file(paths, sessionName);
wfData = load(waveformsFile, 'WFs');
if ~isfield(wfData, 'WFs') || isempty(wfData.WFs)
    error('Expected non-empty variable WFs in %s', waveformsFile);
end

reachDataFile = fullfile(paths.reachDataPath, [sessionName, '.mat']);
if ~isfile(reachDataFile)
    error('Reach session file not found: %s', reachDataFile);
end
reachData = load(reachDataFile, 'CSV', 'idchan', 'R');

idchan = orient_idchan_units_as_rows(reachData.idchan);
wfs = wfData.WFs;
nUnits = min(size(wfs, 1), size(idchan, 1));

unitIds = idchan(1:nUnits, 1);
meanWaveforms = cell(nUnits, 1);
for iUnit = 1:nUnits
    meanWf = mean(wfs(iUnit, :, :), 3);
    meanWf = meanWf(:)';
    if all(meanWf == 0) || ~any(isfinite(meanWf))
        meanWaveforms{iUnit} = [];
    else
        meanWaveforms{iUnit} = meanWf;
    end
end

spikeTimes = reachData.CSV(:, 1);
spikeClusters = reachData.CSV(:, 2);
defaultCollectEnd = round(min(reachData.R(end, 1) + 5000, max(reachData.CSV(:, 1) * 1000)) / 1000);
[collectStart, collectEnd] = get_collect_window(opts, defaultCollectEnd);

waveformContext = struct();
waveformContext.source = 'reach';
waveformContext.sessionLabel = sessionName;
waveformContext.waveformsFile = waveformsFile;
waveformContext.fsSpike = opts.fsSpike;
waveformContext.wfSampleOffset = -60;  % matches GetWFs_AmpChange wfWin(1)
waveformContext.nUnits = nUnits;
waveformContext.unitIds = unitIds;
waveformContext.meanWaveforms = meanWaveforms;
waveformContext.spikeTimes = spikeTimes;
waveformContext.spikeClusters = spikeClusters;
waveformContext.collectStart = collectStart;
waveformContext.collectEnd = collectEnd;
end

function [collectStart, collectEnd] = get_collect_window(opts, defaultCollectEnd)
% GET_COLLECT_WINDOW - Analysis window from opts or session duration

collectStart = 0;
if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
    collectStart = opts.collectStart;
end

collectEnd = defaultCollectEnd;
if isfield(opts, 'collectEnd') && ~isempty(opts.collectEnd)
    collectEnd = opts.collectEnd;
end
end

function tMs = waveform_context_time_ms(waveformContext, nSamples)
% WAVEFORM_CONTEXT_TIME_MS - Time axis in ms for plotted mean waveforms

sampleIdx = waveformContext.wfSampleOffset + (0:nSamples - 1);
tMs = sampleIdx / waveformContext.fsSpike * 1e3;
end

function sessionFolder = get_kilosort_session_folder(sessionType, paths, subjectName, sessionName)
% GET_KILOSORT_SESSION_FOLDER - Folder with cluster_info.tsv and waveforms.mat

switch lower(sessionType)
    case 'spontaneous'
        basePath = paths.spontaneousDataPath;
    case 'interval'
        basePath = paths.intervalDataPath;
    otherwise
        error('Unsupported Kilosort session type: %s', sessionType);
end

sessionFolder = fullfile(basePath, subjectName, sessionName);
if ~isfolder(sessionFolder)
    error('Session folder not found: %s', sessionFolder);
end
end

function waveformsFile = get_reach_waveform_file(paths, sessionName)
% GET_REACH_WAVEFORM_FILE - Resolve *_Neural_WFs.mat from a reach session name

waveformDataDir = fullfile(paths.reachDataPath, 'WaveformDATA');
if ~isfolder(waveformDataDir)
    error('Reach waveform folder not found: %s', waveformDataDir);
end

if contains(sessionName, '_NeuroBeh')
    wfSessionName = strrep(sessionName, '_NeuroBeh', '_Neural_WFs');
else
    error(['Reach sessionName must contain ''_NeuroBeh'' for waveform lookup. Got: %s'], sessionName);
end

waveformsFile = fullfile(waveformDataDir, [wfSessionName, '.mat']);
if ~isfile(waveformsFile)
    error('Reach waveform file not found: %s', waveformsFile);
end
end

function idchan = orient_idchan_units_as_rows(idchan)
% ORIENT_IDCHAN_UNITS_AS_ROWS - Ensure one row per unit (Nx7)

if size(idchan, 1) < size(idchan, 2)
    idchan = idchan';
end
end

function meanWf = get_kilosort_unit_mean_waveform(unitEntry, ci)
% GET_KILOSORT_UNIT_MEAN_WAVEFORM - Mean waveform on the cluster's assigned channel

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
