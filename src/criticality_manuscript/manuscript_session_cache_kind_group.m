function group = manuscript_session_cache_kind_group(kind)
% MANUSCRIPT_SESSION_CACHE_KIND_GROUP - 'd2', 'av', or 'prg' for cache keys

kind = lower(strtrim(char(kind)));
if any(strcmp(kind, {'av', 'engagement_av'}))
  group = 'av';
elseif any(strcmp(kind, {'prg', 'engagement_prg'}))
  group = 'prg';
else
  group = 'd2';
end
end
