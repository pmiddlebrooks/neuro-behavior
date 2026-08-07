function loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectName)
% BUILD_SESSION_LOAD_ARGS - Name-value args for session data loaders
%
% Variables:
%   sessionType  - 'reach', 'spontaneous', 'interval', 'semicircle', 'schall', or 'hong'
%   sessionName  - Session identifier
%   opts         - Options struct passed to the loader
%   subjectName  - Subject folder for spontaneous/interval/semicircle; use '' otherwise
%
% Goal:
%   Return varargin cell for load_session_data (or legacy load_sliding_window_data).

if nargin < 4
    subjectName = '';
end

loadArgs = {'sessionName', sessionName, 'opts', opts};

needsSubject = any(strcmpi(sessionType, {'spontaneous', 'interval', 'semicircle'}));
if needsSubject
    if isempty(subjectName)
        error(['subjectName must be set in the workspace for %s sessions ', ...
            '(e.g. subjectName = ''AS1''; sessionName = ''AS1_0618_WellLearned'').'], ...
            sessionType);
    end
    loadArgs = [loadArgs, {'subjectName', subjectName}]; %#ok<AGROW>
end

end
