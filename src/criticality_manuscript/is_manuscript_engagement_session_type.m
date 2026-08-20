function tf = is_manuscript_engagement_session_type(sessionType)
% IS_MANUSCRIPT_ENGAGEMENT_SESSION_TYPE - Tasks with engaged/non-engaged splits
%
% Variables:
%   sessionType - Session type string
%
% Goal:
%   Centralize which tasks have engagement modules. Reach uses reach onsets;
%   interval uses reward/error beam breaks; semicircle uses TaskMatrix times
%   (trial start, choice poke, leave/enter home).
%
% Returns:
%   tf - True if interval, reach, or semicircle

tf = any(strcmpi(strtrim(char(sessionType)), {'interval', 'reach', 'semicircle'}));
end
