function sessionDate = parse_session_recording_date(sessionName)
% PARSE_SESSION_RECORDING_DATE - Extract YYYY-MM-DD from a session folder name
%
% Variables:
%   sessionName - Session folder, e.g. 'ey9166_2026_04_09'
%
% Goal:
%   Return the recording date as 'YYYY-MM-DD' to match np1-lfp_*.raw titles.
%   Uses the last YYYY_MM_DD token in the name.

if nargin < 1 || isempty(sessionName)
    error('parse_session_recording_date:MissingName', 'sessionName is required.');
end

dateTokens = regexp(char(sessionName), '(\d{4})_(\d{2})_(\d{2})', 'tokens');
if isempty(dateTokens)
    error('parse_session_recording_date:NoDate', ...
        'Could not parse YYYY_MM_DD from sessionName "%s".', sessionName);
end

lastTok = dateTokens{end};
sessionDate = sprintf('%s-%s-%s', lastTok{1}, lastTok{2}, lastTok{3});
end
