function [clausetPlfitPath, plfit2023Path] = resolve_power_law_paths(clausetPlfitPath, plfit2023Path)
% RESOLVE_POWER_LAW_PATHS - Locate Clauset plfit and plfit2023 toolboxes
%
% Variables:
%   clausetPlfitPath - Optional override; used if the directory exists
%   plfit2023Path    - Optional override; used if the directory exists
%
% Goal:
%   Resolve toolbox paths from the neuro-behavior repo layout, not from
%   mfilename (which points at Editor temp copies when using Live Editor).
%
% Returns:
%   clausetPlfitPath - Folder containing plfit.m
%   plfit2023Path    - Folder containing plfit2023.m

if nargin < 1
  clausetPlfitPath = '';
end
if nargin < 2
  plfit2023Path = '';
end

toolboxesRoot = get_toolboxes_root();

if isempty(clausetPlfitPath) || ~is_valid_plfit_dir(clausetPlfitPath)
  clausetPlfitPath = find_first_existing_dir({
    fullfile(toolboxesRoot, 'Power-Law-Fit-Distribution-MATLAB-main', 'MATLAB Code')
    fullfile(toolboxesRoot, 'power-law', 'MATLAB Code')
    });
end

if isempty(plfit2023Path) || ~is_valid_plfit2023_dir(plfit2023Path)
  plfit2023Path = find_first_existing_dir({
    fullfile(toolboxesRoot, 'criticality_shew')
    fullfile(toolboxesRoot, 'd2_criticality', 'power-law range estimation')
    fullfile(toolboxesRoot, 'd_beta_code', 'power-law range estimation')
    });
end

if isempty(clausetPlfitPath)
  error(['Clauset plfit toolbox not found. Set clausetPlfitPath or install under ', ...
    '%s/Power-Law-Fit-Distribution-MATLAB-main/MATLAB Code'], toolboxesRoot);
end
% plfit2023 is optional here; setup_plfit2023_path errors if required and missing
end

function toolboxesRoot = get_toolboxes_root()
% GET_TOOLBOXES_ROOT - Projects/toolboxes sibling of neuro-behavior repo

fitterFile = which('fit_avalanche_power_law');
if isempty(fitterFile)
  error(['fit_avalanche_power_law must be on the path. Add src/criticality ', ...
    'before calling resolve_power_law_paths.']);
end

criticalityDir = fileparts(fitterFile);
srcDir = fileparts(criticalityDir);
projectRoot = fileparts(srcDir);

candidateRoots = {
  fullfile(fileparts(projectRoot), 'toolboxes')
  fullfile(projectRoot, 'toolboxes')
  };

for i = 1:numel(candidateRoots)
  if exist(candidateRoots{i}, 'dir')
    toolboxesRoot = candidateRoots{i};
    return;
  end
end

error(['Could not find toolboxes directory. Expected ', ...
  'Projects/toolboxes next to neuro-behavior or neuro-behavior/toolboxes.']);
end

function tf = is_valid_plfit_dir(dirPath)
tf = ~isempty(dirPath) && exist(dirPath, 'dir') == 7 ...
  && exist(fullfile(dirPath, 'plfit.m'), 'file') == 2;
end

function tf = is_valid_plfit2023_dir(dirPath)
tf = ~isempty(dirPath) && exist(dirPath, 'dir') == 7 ...
  && exist(fullfile(dirPath, 'plfit2023.m'), 'file') == 2;
end

function dirPath = find_first_existing_dir(candidates)
dirPath = '';
for i = 1:numel(candidates)
  if ischar(candidates{i}) && exist(candidates{i}, 'dir')
    dirPath = candidates{i};
    return;
  end
end
end
