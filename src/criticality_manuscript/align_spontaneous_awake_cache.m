function cacheParams = align_spontaneous_awake_cache(cacheParams, sessionType, subjectName, sessionName)
% ALIGN_SPONTANEOUS_AWAKE_CACHE - Match the cache key to sleep exclusion
%
% Variables:
%   cacheParams - Output of make_manuscript_session_cache_params
%   sessionType, subjectName, sessionName - Session identity
%
% Goal:
%   When restrictToAwake is on but this spontaneous session has no lfp.mat,
%   the analysis keeps the whole session. Store that result as not awake-only
%   so it is not reused after an lfp.mat file is added.
%
% Returns:
%   cacheParams - Input struct, with awakeOnly cleared when LFP is absent

if ~isstruct(cacheParams) || ~isfield(cacheParams, 'awakeOnly') || ~logical(cacheParams.awakeOnly)
  return;
end
if ~spontaneous_session_has_lfp(sessionType, subjectName, sessionName)
  cacheParams.awakeOnly = false;
end
end
