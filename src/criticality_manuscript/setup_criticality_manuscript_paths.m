function srcPath = setup_criticality_manuscript_paths(callerName)
% SETUP_CRITICALITY_MANUSCRIPT_PATHS - Add neuro-behavior paths for manuscript scripts
%
% Variables:
%   callerName - Optional m-file name for Editor_* path resolution
%
% Returns:
%   srcPath - Path to neuro-behavior/src
%
% Goal:
%   Shared addpath block for criticality_manuscript across-task runners.

if nargin < 1 || isempty(callerName)
  callerName = 'setup_criticality_manuscript_paths';
end

scriptDir = fileparts(mfilename('fullpath'));
if contains(scriptDir, [filesep 'Editor_' filesep])
  resolved = which(callerName);
  if ~isempty(resolved)
    scriptDir = fileparts(resolved);
  end
end
srcPath = fullfile(scriptDir, '..');
addpath(srcPath);
addpath(fullfile(srcPath, 'reach_task'));
addpath(fullfile(srcPath, 'schall'));
addpath(fullfile(srcPath, 'spontaneous'));
addpath(fullfile(srcPath, 'interval_timing_task'));
addpath(fullfile(srcPath, 'criticality', 'scripts'));
addpath(fullfile(srcPath, 'criticality', 'analyses'));
addpath(fullfile(srcPath, 'session_prep', 'data_prep'));
addpath(fullfile(srcPath, 'session_prep', 'utils'));
addpath(fullfile(srcPath, 'data_prep'));
addpath(fullfile(srcPath, 'sliding_window_prep', 'utils'));
addpath(fullfile(srcPath, 'criticality'));
addpath(scriptDir);
end
