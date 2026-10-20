function rowIdx = find_overlay_session_row(sessionTable, sessionType, sessionName)
% FIND_OVERLAY_SESSION_ROW - Row of one session in an engagement overlay table
%
% Variables:
%   sessionTable - Table with sessionType and sessionName columns
%   sessionType  - Task / session type to match
%   sessionName  - Session identifier to match
%
% Returns:
%   rowIdx - First matching row, or [] when the session is absent

rowIdx = [];
if isempty(sessionTable) || ~istable(sessionTable) || height(sessionTable) == 0
  return;
end
if ~ismember('sessionType', sessionTable.Properties.VariableNames) ...
    || ~ismember('sessionName', sessionTable.Properties.VariableNames)
  return;
end

sessionName = char(sessionName);
typeMatch = strcmp(cellstr(sessionTable.sessionType), char(sessionType));
nameMatch = strcmp(cellstr(sessionTable.sessionName), sessionName);
rowIdx = find(typeMatch & nameMatch, 1);
end
