function sessionTable = filter_manuscript_session_table_by_subject(sessionTable, subjectName)
% FILTER_MANUSCRIPT_SESSION_TABLE_BY_SUBJECT Keep rows for one subject
%
% Variables:
%   sessionTable - Table with sessionType, sessionName, subjectName
%   subjectName  - Subject id as it appears in session lists (e.g. ey9166, Y15, AS1)
%
% Goal:
%   Match struct-list subjectName when present; otherwise match reach-style
%   session names that start with Subject_ or Subject/.

if isempty(sessionTable) || isempty(subjectName)
  return;
end

want = lower(strtrim(char(subjectName)));
keep = false(height(sessionTable), 1);
for i = 1:height(sessionTable)
  rowSubject = '';
  if ismember('subjectName', sessionTable.Properties.VariableNames)
    rowSubject = sessionTable.subjectName{i};
  end
  rowSession = sessionTable.sessionName{i};
  keep(i) = session_entry_matches_subject(want, rowSubject, rowSession);
end

sessionTable = sessionTable(keep, :);
end

function tf = session_entry_matches_subject(want, rowSubject, rowSession)
% SESSION_ENTRY_MATCHES_SUBJECT True if this row belongs to want

if ~isempty(rowSubject) && strcmpi(strtrim(char(rowSubject)), want)
  tf = true;
  return;
end

sessionName = strtrim(char(rowSession));
if isempty(sessionName)
  tf = false;
  return;
end

prefixUnderscore = [want, '_'];
prefixSlash = [want, '/'];
tf = strncmpi(sessionName, prefixUnderscore, numel(prefixUnderscore)) ...
  || strncmpi(sessionName, prefixSlash, numel(prefixSlash)) ...
  || strcmpi(sessionName, want);
end
