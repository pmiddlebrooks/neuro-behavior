function loadArgs = build_session_load_args(sessionType, sessionName, opts, subjectName)
% BUILD_SESSION_LOAD_ARGS - Name-value args for session data loaders
%
% Variables:
%   sessionType  - 'reach', 'spontaneous', 'interval', 'schall', or 'hong'
%   sessionName  - Session identifier
%   opts         - Options struct passed to the loader
%   subjectName  - Subject folder for spontaneous/interval; use '' otherwise
%
% Goal:
%   Return varargin cell for load_session_data (or legacy load_sliding_window_data).

if nargin < 4
    subjectName = '';
end

loadArgs = {'sessionName', sessionName, 'opts', opts};

if strcmpi(sessionType, 'spontaneous') || strcmpi(sessionType, 'interval')
    if isempty(subjectName)
        error(['subjectName must be set in the workspace for %s sessions ', ...
            '(e.g. subjectName = ''ey9166''; sessionName = ''ey9166_2026_04_03'').'], ...
            sessionType);
    end
    loadArgs = [loadArgs, {'subjectName', subjectName}]; %#ok<AGROW>
end

end
