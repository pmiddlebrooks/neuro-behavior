function combinations = default_manuscript_brain_area_combinations()
% DEFAULT_MANUSCRIPT_BRAIN_AREA_COMBINATIONS - Preset merged-area definitions
%
% Goal:
%   Provide common M23+M56 merges. Add custom entries in each script via
%   brainAreaCombinations, e.g. struct('name', 'DSVS', 'areas', {{'DS', 'VS'}}).
%
% Returns:
%   combinations - Cell array of struct with fields .name and .areas (cell of char)

combinations = {
  struct('name', 'M23M56', 'areas', {{'M23', 'M56'}})
  struct('name', 'M2356', 'areas', {{'M23', 'M56'}})  % legacy alias
};
end
