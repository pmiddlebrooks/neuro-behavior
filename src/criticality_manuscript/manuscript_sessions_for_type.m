function entries = manuscript_sessions_for_type(sessionType)
% MANUSCRIPT_SESSIONS_FOR_TYPE - Subject/session entries for manuscript batches
%
% Variables:
%   sessionType - 'spontaneous', 'interval', 'reach', 'semicircle', or 'schall'
%
% Goal:
%   Single source of truth for across-task manuscript session lists. Returns a
%   struct array with .subjectName and .sessionName (subjectName may be '').
%
% Returns:
%   entries - Struct array used by build_*_session_table helpers

switch lower(strtrim(char(sessionType)))
  case 'spontaneous'
    entries = spontaneous_session_list();

  case 'interval'
    entries = interval_session_list();

  case 'semicircle'
    entries = semicircle_session_list();

  case 'reach'
    names = reach_session_list();
    entries = struct('subjectName', {}, 'sessionName', {});
    for i = 1:numel(names)
      entries(i).subjectName = '';
      entries(i).sessionName = names{i};
    end

  case 'schall'
    names = schall_session_list();
    entries = struct('subjectName', {}, 'sessionName', {});
    for i = 1:numel(names)
      parts = strsplit(names{i}, '/');
      if numel(parts) >= 2
        entries(i).subjectName = parts{1};
        entries(i).sessionName = parts{2};
      else
        entries(i).subjectName = '';
        entries(i).sessionName = names{i};
      end
    end

  otherwise
    error('manuscript_sessions_for_type:UnknownType', ...
      'Unknown sessionType: %s', sessionType);
end

if ~isstruct(entries) || ~isfield(entries, 'sessionName')
  error('manuscript_sessions_for_type:BadList', ...
    'Session list for %s must return a struct array with sessionName.', sessionType);
end
end
