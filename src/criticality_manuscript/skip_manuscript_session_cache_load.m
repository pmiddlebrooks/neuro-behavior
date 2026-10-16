function tf = skip_manuscript_session_cache_load(opts, sessionType, subjectName)
% SKIP_MANUSCRIPT_SESSION_CACHE_LOAD - True if cache should not be read
%
% Variables:
%   opts        - Batch options (useSessionCache, forceRecompute,
%                 forceRecomputeSessionTypes, forceRecomputeSubject)
%   sessionType - Session type being loaded. Optional. When it is listed in
%                 forceRecomputeSessionTypes, only that type is recomputed.
%   subjectName - Subject folder for this session. Optional. When
%                 forceRecomputeSubject is set, a listed session type is
%                 recomputed only for that subject.
%
% Goal:
%   forceRecompute skips every cached session. forceRecomputeSessionTypes
%   skips cache loads for the named types (for example {'spontaneous'}) and
%   still loads the others. forceRecomputeSubject narrows that list to one
%   subject, so the other animals in that session type keep their cache.
%   Saving a replacement file is unchanged.

if nargin < 3
  subjectName = '';
end

tf = ~use_manuscript_session_cache(opts);
if nargin < 1 || ~isstruct(opts)
  return;
end
if isfield(opts, 'forceRecompute') && opts.forceRecompute
  tf = true;
  return;
end
if nargin < 2 || isempty(sessionType)
  return;
end
if ~isfield(opts, 'forceRecomputeSessionTypes') || isempty(opts.forceRecomputeSessionTypes)
  return;
end
typeList = cellstr(opts.forceRecomputeSessionTypes);
if ~any(strcmpi(strtrim(typeList), strtrim(char(sessionType))))
  return;
end
subjectList = force_recompute_subject_list(opts);
if isempty(subjectList)
  tf = true;
  return;
end
tf = any(strcmpi(subjectList, strtrim(char(subjectName))));
end

function subjectList = force_recompute_subject_list(opts)
% FORCE_RECOMPUTE_SUBJECT_LIST - Subject names that narrow a type recompute
%
% Variables:
%   opts - Batch options. forceRecomputeSubject may be a char or a cell.
%
% Returns:
%   subjectList - Cell of non-empty subject names. Empty means every subject.

subjectList = {};
if ~isfield(opts, 'forceRecomputeSubject') || isempty(opts.forceRecomputeSubject)
  return;
end
subjectList = cellstr(opts.forceRecomputeSubject);
subjectList = strtrim(subjectList(:));
subjectList = subjectList(~cellfun(@isempty, subjectList));
end
