function tf = use_manuscript_session_cache(opts)
% USE_MANUSCRIPT_SESSION_CACHE - True unless caller disables per-session cache

tf = true;
if nargin < 1 || ~isstruct(opts)
  return;
end
if isfield(opts, 'useSessionCache') && ~isempty(opts.useSessionCache)
  tf = logical(opts.useSessionCache);
end
end
