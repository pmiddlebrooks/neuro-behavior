function sessions = semicircle_session_list()
% SEMICIRCLE_SESSION_LIST - Subject/session pairs for semicircle reward task
%
% Variables:
%   (none)
%
% Goal:
%   Return a struct array with fields subjectName and sessionName for each
%   session (required by load_semicircle_data / load_session_data).
%
% Returns:
%   sessions - Struct array: sessions(i).subjectName, sessions(i).sessionName
%
% Notes:
%   Session .mat files live under
%   paths.semicircleDataPath/<subjectName>/<sessionName>.mat

sessions = [
    struct('subjectName', 'AS1', 'sessionName', 'AS1_0618_WellLearned')
    struct('subjectName', 'AS1', 'sessionName', 'AS1_0623_TransitionAfterCompletedTrial_80')
    struct('subjectName', 'AS1', 'sessionName', 'AS1_0624_PoorlyLearned')
    ];

end
