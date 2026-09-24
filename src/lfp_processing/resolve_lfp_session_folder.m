function sessionFolder = resolve_lfp_session_folder(sessionType, subjectName, sessionName, paths)
% RESOLVE_LFP_SESSION_FOLDER - Spike-data folder that should receive lfp.csv
%
% Variables:
%   sessionType - 'spontaneous' or 'interval' (other kilosort tasks also ok)
%   subjectName - Subject folder under the task data root
%   sessionName - Session folder under subject
%   paths       - Struct from get_paths
%
% Goal:
%   Return the session directory that already holds spike sorting files.

if nargin < 4 || isempty(paths)
    paths = get_paths;
end
if isempty(subjectName)
    error('resolve_lfp_session_folder:MissingSubject', ...
        'subjectName is required for %s sessions.', sessionType);
end

switch lower(sessionType)
    case 'spontaneous'
        basePath = paths.spontaneousDataPath;
    case 'interval'
        basePath = paths.intervalDataPath;
    case 'semicircle'
        basePath = paths.semicircleDataPath;
    otherwise
        error('resolve_lfp_session_folder:BadType', ...
            'Unsupported sessionType "%s". Use spontaneous or interval.', sessionType);
end

sessionFolder = fullfile(basePath, subjectName, sessionName);
if ~isfolder(sessionFolder)
    error('resolve_lfp_session_folder:NotFound', ...
        'Session folder not found: %s', sessionFolder);
end
end
