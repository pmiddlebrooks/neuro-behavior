function sessions = spontaneous_session_list()
% SPONTANEOUS_SESSION_LIST - Subject/session pairs for spontaneous recordings
%
% Variables:
%   (none)
%
% Goal:
%   Return a struct array with fields subjectName and sessionName for each
%   session (required by load_spontaneous_data / load_sliding_window_data).
%
% Returns:
%   sessions - Struct array: sessions(i).subjectName, sessions(i).sessionName

sessions = [
    struct('subjectName', 'ag25290', 'sessionName', 'ag112221')   % good
    struct('subjectName', 'ag25290', 'sessionName', 'ag112321_1')   % good
    struct('subjectName', 'ag25290', 'sessionName', 'ag112721')   % 
    struct('subjectName', 'ag25290', 'sessionName', 'ag112821')   % 
    struct('subjectName', 'ag25290', 'sessionName', 'ag112921')   % 
    struct('subjectName', 'ey4152', 'sessionName', 'ey042822')     % good M56, bad DS
    struct('subjectName', 'ey4152', 'sessionName', 'ey042922')     % good M56, bad DS
    struct('subjectName', 'kw7193', 'sessionName', 'kw092821')     % bad M56, good DS
    struct('subjectName', 'kw7193', 'sessionName', 'kw092921')     % ok, few M56
    ];

% I have these sessions but they generally have too few neurons for
% reliable population analyses:
    % struct('subjectName', 'ag25290', 'sessionName', 'ag112221')   % good
    % struct('subjectName', 'ag25290', 'sessionName', 'ag112321_2')   % good
    % struct('subjectName', 'ey4152', 'sessionName', 'ey042922')   % good
    % struct('subjectName', 'kw7193', 'sessionName', 'kw092821')     % bad M56, good DS

end
