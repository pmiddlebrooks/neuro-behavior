function [sessionName, subjectName] = unpack_session_list_entry(sessions, sessionIdx)
% UNPACK_SESSION_LIST_ENTRY - sessionName and subjectName from a session list
%
% Variables:
%   sessions    - Cell array of session names, or struct array with fields
%                 subjectName and sessionName
%   sessionIdx  - Index into sessions
%
% Goal:
%   Normalize session list formats for batch scripts.
%
% Returns:
%   sessionName - Session identifier string
%   subjectName - Subject folder name ('' when not required)

if isstruct(sessions)
  sessionName = sessions(sessionIdx).sessionName;
  if isfield(sessions, 'subjectName')
    subjectName = sessions(sessionIdx).subjectName;
  else
    subjectName = '';
  end
elseif iscell(sessions)
  sessionName = sessions{sessionIdx};
  subjectName = '';
else
  error('sessions must be a struct array or cell array of session names.');
end

end
