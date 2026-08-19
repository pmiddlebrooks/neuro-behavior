function tf = skip_manuscript_session_cache_load(opts)
% SKIP_MANUSCRIPT_SESSION_CACHE_LOAD - True if cache should not be read
%
% forceRecompute skips loading but still allows saving a replacement file.

tf = ~use_manuscript_session_cache(opts);
if nargin >= 1 && isstruct(opts) && isfield(opts, 'forceRecompute') && opts.forceRecompute
  tf = true;
end
end
