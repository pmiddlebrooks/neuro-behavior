function toolkitSrc = add_prox_crit_toolkit_path()
% ADD_PROX_CRIT_TOOLKIT_PATH Add Sooter/Shew d2 toolboxes to the MATLAB path
%
% Goal:
%   Resolve Projects/toolboxes (sibling of neuro-behavior) or
%   neuro-behavior/toolboxes without using '..' in fileparts().

thisFile = mfilename('fullpath');
criticalityDir = fileparts(thisFile);
srcRoot = fileparts(criticalityDir);
projectRoot = fileparts(srcRoot);
projectsRoot = fileparts(projectRoot);
candidateRoots = {
  fullfile(projectsRoot, 'toolboxes')
  fullfile(projectRoot, 'toolboxes')
  };

shewPath = '';
toolkitSrc = '';
for i = 1:numel(candidateRoots)
  shewCandidate = fullfile(candidateRoots{i}, 'criticality_shew');
  klCandidate = fullfile(candidateRoots{i}, 'prox_crit_toolkit', 'src');
  if isempty(shewPath) && exist(shewCandidate, 'dir')
    shewPath = shewCandidate;
  end
  if isempty(toolkitSrc) && exist(klCandidate, 'dir') ...
      && exist(fullfile(klCandidate, 'calc_db.m'), 'file')
    toolkitSrc = klCandidate;
  end
end

if ~isempty(shewPath)
  addpath(shewPath);
end
if isempty(toolkitSrc)
  error(['prox_crit_toolkit not found. Expected calc_db.m under ', ...
    '%s\\prox_crit_toolkit\\src'], candidateRoots{1});
end
addpath(toolkitSrc);
end
