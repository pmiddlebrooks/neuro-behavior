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
%   Load session waveforms, match unit IDs to analysis neurons, and subset each area.
%   Spontaneous/interval: waveforms.mat in the spike session folder (Kilosort/Phy).
%   Reach: *_Neural_WFs.mat in paths.reachDataPath/WaveformDATA (from GetWFs_AmpChange.m).
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
if ~isfield(dataStruct, 'sessionName') || isempty(dataStruct.sessionName)
    error('dataStruct.sessionName is required for waveform-based cell-type filtering.');
end
if ~isfield(dataStruct, 'idLabel') || ~isfield(dataStruct, 'idMatIdx')
    error('dataStruct must contain idLabel and idMatIdx.');
end

[unitWidths, waveformMeta] = get_session_unit_widths(dataStruct, paths);

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
meta.unitWidths = unitWidths;
meta.keptPerArea = keptPerArea;
meta.totalPerArea = totalPerArea;
meta.nKept = sum(keptPerArea);
meta.nTotal = sum(totalPerArea);
meta.waveformSource = waveformMeta.source;
meta.waveformsFile = waveformMeta.waveformsFile;

fprintf('  Waveform E/I filter (%s, %s): kept %d / %d units (cutoff = %.3f ms)\n', ...
    cellType, waveformMeta.source, meta.nKept, meta.nTotal, widthCutoff);
for a = 1:numAreas
    fprintf('    %s: %d / %d\n', dataStruct.areas{a}, keptPerArea(a), totalPerArea(a));
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
