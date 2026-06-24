function [dataStruct, areaOk] = apply_manuscript_brain_area_selection(dataStruct, brainArea, combinations, mergeCombined)
% APPLY_MANUSCRIPT_BRAIN_AREA_SELECTION - Validate and optionally merge brain areas
%
% Variables:
%   dataStruct     - Session struct from load_session_data / load_sliding_window_data
%   brainArea      - Single area or merged-area name; '' = all areas
%   combinations   - Cell array from default_manuscript_brain_area_combinations()
%   mergeCombined  - If true, create merged area and set areasToTest (default true)
%
% Goal:
%   Support single-area restriction and user-defined multi-area merges (e.g. M23+M56).
%
% Returns:
%   areaOk - false when brainArea is missing from the session

if nargin < 3 || isempty(combinations)
  combinations = default_manuscript_brain_area_combinations();
end
if nargin < 4 || isempty(mergeCombined)
  mergeCombined = true;
end

areaOk = true;
if isempty(brainArea)
  return;
end

combo = lookup_manuscript_brain_area_combination(brainArea, combinations);
if ~isempty(combo)
  if ~session_has_brain_areas(dataStruct, combo.areas)
    areaOk = false;
    return;
  end
  if mergeCombined
    dataStruct = maybe_add_combined_brain_area(dataStruct, combo.name, combo.areas);
    areaIdx = find(strcmp(dataStruct.areas, combo.name), 1);
    dataStruct.areasToTest = areaIdx;
  else
    fprintf('  Brain area %s (combined: %s)\n', combo.name, strjoin(combo.areas, '+'));
  end
  return;
end

areaIdx = find(strcmp(dataStruct.areas, brainArea), 1);
if isempty(areaIdx)
  areaOk = false;
  return;
end

if mergeCombined
  dataStruct.areasToTest = areaIdx;
  fprintf('  Restricting analysis to area: %s\n', brainArea);
end
end

function combo = lookup_manuscript_brain_area_combination(brainArea, combinations)
combo = [];
if isempty(brainArea) || isempty(combinations)
  return;
end

brainArea = char(brainArea);
for i = 1:numel(combinations)
  entry = combinations{i};
  if isstruct(entry) && isfield(entry, 'name') && strcmpi(entry.name, brainArea)
    combo = entry;
    return;
  end
end
end

function tf = session_has_brain_areas(dataStruct, areaNames)
tf = false;
if isempty(areaNames) || ~isfield(dataStruct, 'areas')
  return;
end

areaNames = cellstr(areaNames(:));
for i = 1:numel(areaNames)
  if ~any(strcmp(dataStruct.areas, areaNames{i}))
    return;
  end
end
tf = true;
end

function dataStruct = maybe_add_combined_brain_area(dataStruct, combinedName, sourceAreas)
if isempty(combinedName) || isempty(sourceAreas)
  return;
end

combinedName = char(combinedName);
sourceAreas = cellstr(sourceAreas(:)');
areas = dataStruct.areas;

if any(strcmp(areas, combinedName))
  return;
end

sourceIdx = zeros(1, numel(sourceAreas));
for i = 1:numel(sourceAreas)
  sourceIdx(i) = find(strcmp(areas, sourceAreas{i}), 1);
  if isempty(sourceIdx(i))
    return;
  end
end

mergedIdx = [];
mergedLabel = [];
for i = 1:numel(sourceIdx)
  mergedIdx = [mergedIdx; dataStruct.idMatIdx{sourceIdx(i)}(:)]; %#ok<AGROW>
  if isfield(dataStruct, 'idLabel')
    mergedLabel = [mergedLabel; dataStruct.idLabel{sourceIdx(i)}(:)]; %#ok<AGROW>
  end
end

areas{end+1} = combinedName;
dataStruct.areas = areas;
dataStruct.idMatIdx{end+1} = mergedIdx;
if isfield(dataStruct, 'idLabel')
  dataStruct.idLabel{end+1} = mergedLabel;
end

countsStr = cell(1, numel(sourceAreas));
for i = 1:numel(sourceAreas)
  countsStr{i} = sprintf('%s: %d', sourceAreas{i}, numel(dataStruct.idMatIdx{sourceIdx(i)}));
end
fprintf('  Added combined area %s (%d neurons; %s)\n', ...
  combinedName, numel(mergedIdx), strjoin(countsStr, ', '));
end
