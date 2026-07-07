function [unitWidths, meta] = get_session_unit_widths(dataStruct, paths)
% GET_SESSION_UNIT_WIDTHS - Map cluster id to peak-to-trough width (ms)
%
% Variables:
%   dataStruct - Session struct from load_session_data
%   paths      - Struct from get_paths
%
% Goal:
%   Load waveforms once for spontaneous/interval (Kilosort) or reach sessions.
%
% Returns:
%   unitWidths - containers.Map from unit id to width (ms)
%   meta       - Struct with .source and .waveformsFile

opts = neuro_behavior_options();
if isfield(dataStruct, 'opts') && isstruct(dataStruct.opts) && isfield(dataStruct.opts, 'fsSpike')
    opts.fsSpike = dataStruct.opts.fsSpike;
end

if ~isfield(dataStruct, 'sessionType') || isempty(dataStruct.sessionType)
    error('dataStruct.sessionType is required for waveform loading.');
end
if ~isfield(dataStruct, 'sessionName') || isempty(dataStruct.sessionName)
    error('dataStruct.sessionName is required for waveform loading.');
end

switch lower(dataStruct.sessionType)
    case 'spontaneous'
        if ~isfield(dataStruct, 'subjectName') || isempty(dataStruct.subjectName)
            error('dataStruct.subjectName is required for spontaneous waveform loading.');
        end
        sessionFolder = get_kilosort_session_folder(paths, dataStruct.subjectName, dataStruct.sessionName, 'spontaneous');
        [unitWidths, meta] = load_kilosort_unit_widths(sessionFolder, opts.fsSpike);

    case 'interval'
        if ~isfield(dataStruct, 'subjectName') || isempty(dataStruct.subjectName)
            error('dataStruct.subjectName is required for interval waveform loading.');
        end
        sessionFolder = get_kilosort_session_folder(paths, dataStruct.subjectName, dataStruct.sessionName, 'interval');
        [unitWidths, meta] = load_kilosort_unit_widths(sessionFolder, opts.fsSpike);

    case 'reach'
        [unitWidths, meta] = load_reach_unit_widths(dataStruct, paths, opts.fsSpike);

    otherwise
        error(['Waveform loading supports spontaneous, interval, and reach sessions. ', ...
            'Got sessionType = %s.'], dataStruct.sessionType);
end
end

function sessionFolder = get_kilosort_session_folder(paths, subjectName, sessionName, sessionType)
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

function [unitWidths, meta] = load_kilosort_unit_widths(sessionFolder, fsSpike)
% LOAD_KILOSORT_UNIT_WIDTHS - Widths from waveforms.mat / sp_waveforms

waveformsFile = fullfile(sessionFolder, 'waveforms.mat');
if ~isfile(waveformsFile)
    error('waveforms.mat not found in %s (required for E/I split).', sessionFolder);
end

waveformData = load(waveformsFile);
if ~isfield(waveformData, 'sp_waveforms')
    error('Expected variable sp_waveforms in %s', waveformsFile);
end

ci = readtable(fullfile(sessionFolder, 'cluster_info.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');
unitWidths = build_kilosort_unit_width_lookup(waveformData.sp_waveforms, ci, fsSpike);

meta = struct('source', 'kilosort', 'waveformsFile', waveformsFile);
end

function [unitWidths, meta] = load_reach_unit_widths(dataStruct, paths, fsSpike)
% LOAD_REACH_UNIT_WIDTHS - Widths from reach_task/data/WaveformDATA/*_Neural_WFs.mat

waveformsFile = get_reach_waveform_file(paths, dataStruct.sessionName);
wfData = load(waveformsFile, 'WFs');
if ~isfield(wfData, 'WFs') || isempty(wfData.WFs)
    error('Expected non-empty variable WFs in %s', waveformsFile);
end

idchan = get_reach_idchan(dataStruct, paths);
unitWidths = build_reach_unit_width_lookup(wfData.WFs, idchan, fsSpike);

meta = struct('source', 'reach', 'waveformsFile', waveformsFile);
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

function idchan = get_reach_idchan(dataStruct, paths)
% GET_REACH_IDCHAN - Unit metadata table for row alignment with WFs

if isfield(dataStruct, 'dataR') && isstruct(dataStruct.dataR) && isfield(dataStruct.dataR, 'idchan')
    idchan = dataStruct.dataR.idchan;
elseif isfield(dataStruct, 'idchan')
    idchan = dataStruct.idchan;
else
    reachDataFile = fullfile(paths.reachDataPath, [dataStruct.sessionName, '.mat']);
    if ~isfile(reachDataFile)
        error('Reach session file not found: %s', reachDataFile);
    end
    reachData = load(reachDataFile, 'idchan');
    idchan = reachData.idchan;
end

idchan = orient_idchan_units_as_rows(idchan);
end

function idchan = orient_idchan_units_as_rows(idchan)
% ORIENT_IDCHAN_UNITS_AS_ROWS - Ensure one row per unit (Nx7)

if size(idchan, 1) < size(idchan, 2)
    idchan = idchan';
end
end

function unitWidths = build_kilosort_unit_width_lookup(spWaveforms, ci, fsSpike)
% BUILD_KILOSORT_UNIT_WIDTH_LOOKUP - Map cluster id to peak-to-trough width (ms)

unitWidths = containers.Map('KeyType', 'double', 'ValueType', 'double');
for iUnit = 1:numel(spWaveforms)
    meanWf = get_kilosort_unit_mean_waveform(spWaveforms(iUnit), ci);
    if isempty(meanWf)
        continue;
    end
    unitWidths(spWaveforms(iUnit).unitID) = compute_peak_to_trough(meanWf, fsSpike);
end
end

function unitWidths = build_reach_unit_width_lookup(wfs, idchan, fsSpike)
% BUILD_REACH_UNIT_WIDTH_LOOKUP - Map reach unit id to width from mean WFs

idchan = orient_idchan_units_as_rows(idchan);
nUnits = min(size(wfs, 1), size(idchan, 1));
unitWidths = containers.Map('KeyType', 'double', 'ValueType', 'double');

for iUnit = 1:nUnits
    meanWf = mean(wfs(iUnit, :, :), 3);
    meanWf = meanWf(:)';
    if isempty(meanWf) || all(meanWf == 0) || ~any(isfinite(meanWf))
        continue;
    end
    unitWidths(idchan(iUnit, 1)) = compute_peak_to_trough(meanWf, fsSpike);
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
