function sessions = interval_session_list()
% INTERVAL_SESSION_LIST - Subject/session pairs for interval timing task
%
% Variables:
%   (none)
%
% Goal:
%   Return a struct array with fields subjectName and sessionName for each
%   session (required by load_interval_data / load_sliding_window_data).
%
% Returns:
%   sessions - Struct array: sessions(i).subjectName, sessions(i).sessionName

sessions = [
    struct('subjectName', 'ey9166', 'sessionName', 'ey9166_2026_04_03')
    % struct('subjectName', 'ey9166', 'sessionName', 'ey9166_2026_04_04')
    % struct('subjectName', 'ey9166', 'sessionName', 'ey9166_2026_04_07')
    % struct('subjectName', 'ey9166', 'sessionName', 'ey9166_2026_04_09')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_19')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_20')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_21')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_22')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_22')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_25')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_26')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_27')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_05_28')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_06_01')
    % struct('subjectName', 'ey9387', 'sessionName', 'ey9387_2026_06_02')
    ];

end
