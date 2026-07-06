function dataStructRun = prepare_session_data_for_cell_type(dataStruct, paths, cellType, widthCutoff, splitExcitatoryInhibitory)
% PREPARE_SESSION_DATA_FOR_CELL_TYPE - Copy session data and optional E/I filter
%
% Variables:
%   dataStruct                  - Base session struct after brain-area validation
%   paths                       - From get_paths
%   cellType                    - '', 'all', 'excitatory', or 'inhibitory'
%   widthCutoff                 - Waveform width threshold (ms)
%   splitExcitatoryInhibitory   - Whether E/I split mode is active
%
% Goal:
%   When split is on, 'all' keeps E+I combined; E and I apply waveform filter.
%   Reach waveforms load from reach_task/data/WaveformDATA (see apply_session_cell_type_filter).

dataStructRun = copy_neuron_selection(dataStruct);

if ~splitExcitatoryInhibitory || isempty(cellType) || strcmpi(cellType, 'all')
  if splitExcitatoryInhibitory && strcmpi(cellType, 'all')
    fprintf('\n--- Cell type: %s ---\n', cell_type_label(cellType));
  end
  return;
end

fprintf('\n--- Cell type: %s ---\n', cell_type_label(cellType));
[dataStructRun, ~] = apply_session_cell_type_filter(dataStructRun, paths, cellType, widthCutoff);
end

function dataStructCopy = copy_neuron_selection(dataStruct)
% COPY_NEURON_SELECTION - Deep copy of per-area neuron id fields
%
% Variables:
%   dataStruct - Session struct with idLabel and idMatIdx cell arrays
%
% Goal: Copy neuron selections without sharing cell arrays with the source struct

dataStructCopy = dataStruct;
nAreas = numel(dataStruct.areas);
dataStructCopy.idLabel = cell(1, nAreas);
dataStructCopy.idMatIdx = cell(1, nAreas);
for a = 1:nAreas
  dataStructCopy.idLabel{a} = dataStruct.idLabel{a}(:)';
  dataStructCopy.idMatIdx{a} = dataStruct.idMatIdx{a}(:)';
end
end
