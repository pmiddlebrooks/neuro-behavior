function [dataStruct, meta] = apply_session_cell_type_filter(dataStruct, paths, cellType, widthCutoff)
% APPLY_SESSION_CELL_TYPE_FILTER - Keep excitatory or inhibitory units by waveform width
%
% Variables:
%   dataStruct   - Session struct from load_session_data (must include idLabel, idMatIdx)
%   paths        - Struct from get_paths
%   cellType     - 'excitatory' or 'inhibitory'
%   widthCutoff  - Peak-to-trough width threshold in ms (narrow <= cutoff = inhibitory)
%
% Goal:
%   Load waveforms.mat, match unit IDs to analysis neurons, and subset each area.
%
% Returns:
%   meta - Struct with cellType, widthCutoff, unitWidths, keptPerArea, nKept, nTotal

if nargin < 4 || isempty(widthCutoff)
    widthCutoff = 0.035;
end

validateattributes(widthCutoff, {'numeric'}, {'scalar', 'positive', 'finite'});

cellType = lower(strtrim(cellType));
if ~ismember(cellType, {'excitatory', 'inhibitory'})
    error('cellType must be ''excitatory'' or ''inhibitory''.');
end

if ~isfield(dataStruct, 'sessionType') || isempty(dataStruct.sessionType)
    error('dataStruct.sessionType is required for waveform-based cell-type filtering.');
end
if ~isfield(dataStruct, 'subjectName') || isempty(dataStruct.subjectName)
    error('dataStruct.subjectName is required for waveform-based cell-type filtering.');
end
if ~isfield(dataStruct, 'sessionName') || isempty(dataStruct.sessionName)
    error('dataStruct.sessionName is required for waveform-based cell-type filtering.');
end
if ~isfield(dataStruct, 'idLabel') || ~isfield(dataStruct, 'idMatIdx')
    error('dataStruct must contain idLabel and idMatIdx.');
end

opts = neuro_behavior_options();
if isfield(dataStruct, 'opts') && isstruct(dataStruct.opts) && isfield(dataStruct.opts, 'fsSpike')
    opts.fsSpike = dataStruct.opts.fsSpike;
end

sessionFolder = get_waveform_session_folder(dataStruct.sessionType, paths, ...
    dataStruct.subjectName, dataStruct.sessionName);
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
unitWidths = build_unit_width_lookup(waveformData.sp_waveforms, ci, opts.fsSpike);

numAreas = numel(dataStruct.areas);
keptPerArea = zeros(numAreas, 1);
totalPerArea = zeros(numAreas, 1);

for a = 1:numAreas
    neuronIds = dataStruct.idLabel{a}(:)';
    areaIdx = dataStruct.idMatIdx{a}(:)';
    totalPerArea(a) = numel(neuronIds);

    keepMask = select_units_by_cell_type(neuronIds, unitWidths, cellType, widthCutoff);
    keptPerArea(a) = sum(keepMask);

    dataStruct.idLabel{a} = neuronIds(keepMask);
    dataStruct.idMatIdx{a} = areaIdx(keepMask);
end

meta = struct();
meta.cellType = cellType;
meta.widthCutoff = widthCutoff;
meta.sessionFolder = sessionFolder;
meta.unitWidths = unitWidths;
meta.keptPerArea = keptPerArea;
meta.totalPerArea = totalPerArea;
meta.nKept = sum(keptPerArea);
meta.nTotal = sum(totalPerArea);

fprintf('  Waveform E/I filter (%s): kept %d / %d units (cutoff = %.3f ms)\n', ...
    cellType, meta.nKept, meta.nTotal, widthCutoff);
for a = 1:numAreas
    fprintf('    %s: %d / %d\n', dataStruct.areas{a}, keptPerArea(a), totalPerArea(a));
end
end

function sessionFolder = get_waveform_session_folder(sessionType, paths, subjectName, sessionName)
% GET_WAVEFORM_SESSION_FOLDER - Folder with cluster_info.tsv and waveforms.mat

switch lower(sessionType)
    case 'spontaneous'
        basePath = paths.spontaneousDataPath;
    case 'interval'
        basePath = paths.intervalDataPath;
    otherwise
        error(['Waveform-based E/I split supports spontaneous and interval sessions. ', ...
            'Got sessionType = %s.'], sessionType);
end

sessionFolder = fullfile(basePath, subjectName, sessionName);
if ~isfolder(sessionFolder)
    error('Session folder not found: %s', sessionFolder);
end
end

function unitWidths = build_unit_width_lookup(spWaveforms, ci, fsSpike)
% BUILD_UNIT_WIDTH_LOOKUP - Map cluster id to peak-to-trough width (ms)

unitWidths = containers.Map('KeyType', 'double', 'ValueType', 'double');
for iUnit = 1:numel(spWaveforms)
    meanWf = get_unit_mean_waveform(spWaveforms(iUnit), ci);
    if isempty(meanWf)
        continue;
    end
    unitWidths(spWaveforms(iUnit).unitID) = compute_peak_to_trough(meanWf, fsSpike);
end
end

function keepMask = select_units_by_cell_type(unitIds, unitWidths, cellType, widthCutoff)
% SELECT_UNITS_BY_CELL_TYPE - Logical mask for excitatory or inhibitory units

keepMask = false(size(unitIds));
for i = 1:numel(unitIds)
    unitId = unitIds(i);
    if ~isKey(unitWidths, unitId)
        continue;
    end
    widthMs = unitWidths(unitId);
    if isnan(widthMs)
        continue;
    end
    if strcmp(cellType, 'inhibitory')
        keepMask(i) = widthMs <= widthCutoff;
    else
        keepMask(i) = widthMs > widthCutoff;
    end
end
end

function meanWf = get_unit_mean_waveform(unitEntry, ci)
% GET_UNIT_MEAN_WAVEFORM - Mean waveform on the cluster's assigned channel

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
