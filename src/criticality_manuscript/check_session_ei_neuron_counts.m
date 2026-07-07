function checkResult = check_session_ei_neuron_counts(dataStruct, paths, widthCutoff, brainArea, brainAreaCombinations, nMinNeurons)
% CHECK_SESSION_EI_NEURON_COUNTS - Pre-check E/I counts before split analyses
%
% Variables:
%   dataStruct             - Session struct after load_session_data
%   paths                  - Struct from get_paths
%   widthCutoff            - Peak-to-trough width threshold in ms
%   brainArea              - Analysis area (single or merged)
%   brainAreaCombinations  - Merged-area definitions
%   nMinNeurons            - Minimum neurons required per E and I population
%
% Goal:
%   Load waveforms once, count excitatory and inhibitory units in the analysis
%   area, print counts, and return whether both populations are large enough.
%
% Returns:
%   checkResult - Struct with fields:
%       .isOk, .nExcitatory, .nInhibitory, .nUnclassified, .nMinNeurons, .brainArea

if nargin < 6 || isempty(nMinNeurons)
    error('nMinNeurons is required.');
end

dataStructCheck = dataStruct;
[dataStructCheck, areaOk] = apply_manuscript_brain_area_selection( ...
    dataStructCheck, brainArea, brainAreaCombinations);
if ~areaOk
    error('Brain area "%s" not available for E/I pre-check.', brainArea);
end

neuronIds = collect_analysis_neuron_ids(dataStructCheck, brainArea);
[unitWidths, waveformMeta] = get_session_unit_widths(dataStructCheck, paths);

nExcitatory = 0;
nInhibitory = 0;
nUnclassified = 0;
for i = 1:numel(neuronIds)
    unitId = neuronIds(i);
    if ~isKey(unitWidths, unitId)
        nUnclassified = nUnclassified + 1;
        continue;
    end
    widthMs = unitWidths(unitId);
    if ~isfinite(widthMs)
        nUnclassified = nUnclassified + 1;
        continue;
    end
    if widthMs <= widthCutoff
        nInhibitory = nInhibitory + 1;
    else
        nExcitatory = nExcitatory + 1;
    end
end

isOk = nExcitatory >= nMinNeurons && nInhibitory >= nMinNeurons;

fprintf('\n=== E/I neuron pre-check (%s) ===\n', waveformMeta.source);
fprintf('  Waveforms: %s\n', waveformMeta.waveformsFile);
if isempty(brainArea)
    fprintf('  Analysis area: all areas\n');
else
    fprintf('  Analysis area: %s\n', brainArea);
end
fprintf('  Width cutoff: %.3f ms (narrow <= cutoff = inhibitory)\n', widthCutoff);
fprintf('  Excitatory units: %d\n', nExcitatory);
fprintf('  Inhibitory units: %d\n', nInhibitory);
if nUnclassified > 0
    fprintf('  Unclassified (no waveform): %d\n', nUnclassified);
end
fprintf('  Required per population: >= %d\n', nMinNeurons);

if ~isOk
    fprintf('\nInsufficient neurons for E/I split analysis.\n');
    if nExcitatory < nMinNeurons
        fprintf('  Excitatory: %d < %d (minimum)\n', nExcitatory, nMinNeurons);
    end
    if nInhibitory < nMinNeurons
        fprintf('  Inhibitory: %d < %d (minimum)\n', nInhibitory, nMinNeurons);
    end
    fprintf('Exiting without running analysis.\n');
end

checkResult = struct( ...
    'isOk', isOk, ...
    'nExcitatory', nExcitatory, ...
    'nInhibitory', nInhibitory, ...
    'nUnclassified', nUnclassified, ...
    'nMinNeurons', nMinNeurons, ...
    'brainArea', brainArea);
end

function neuronIds = collect_analysis_neuron_ids(dataStruct, brainArea)
% COLLECT_ANALYSIS_NEURON_IDS - Unit ids in the selected analysis area(s)

neuronIds = [];
if ~isfield(dataStruct, 'idLabel') || isempty(dataStruct.idLabel)
    return;
end

if isempty(brainArea)
    for areaIdx = 1:numel(dataStruct.idLabel)
        neuronIds = [neuronIds, dataStruct.idLabel{areaIdx}(:)']; %#ok<AGROW>
    end
    return;
end

areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
    return;
end
neuronIds = dataStruct.idLabel{areaIdx}(:)';
end
