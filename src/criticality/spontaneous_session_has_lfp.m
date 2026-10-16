function tf = spontaneous_session_has_lfp(sessionType, subjectName, sessionName)
% SPONTANEOUS_SESSION_HAS_LFP - True when the session folder contains lfp.mat
%
% Variables:
%   sessionType  - Session type. Only 'spontaneous' can be true.
%   subjectName  - Subject folder under the spontaneous data root
%   sessionName  - Session folder under the subject
%
% Goal:
%   Sleep exclusion uses the processed LFP file. A missing file means the
%   caller should keep the whole session.
%
% Returns:
%   tf - Logical

tf = false;
if nargin < 3 || isempty(sessionType) || isempty(subjectName) || isempty(sessionName)
  return;
end
if ~strcmpi(strtrim(char(sessionType)), 'spontaneous')
  return;
end
try
  paths = get_paths();
catch
  return;
end
if ~isstruct(paths) || ~isfield(paths, 'spontaneousDataPath') ...
    || isempty(paths.spontaneousDataPath)
  return;
end
matPath = fullfile(paths.spontaneousDataPath, char(subjectName), char(sessionName), 'lfp.mat');
tf = isfile(matPath);
end
