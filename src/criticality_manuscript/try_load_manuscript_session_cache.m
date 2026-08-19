function [payload, didLoad] = try_load_manuscript_session_cache(cacheFile, cacheParams)
% TRY_LOAD_MANUSCRIPT_SESSION_CACHE - Load payload if file exists and params match

payload = [];
didLoad = false;
if nargin < 1 || isempty(cacheFile) || ~isfile(cacheFile)
  return;
end
if nargin < 2 || isempty(cacheParams)
  return;
end

try
  loaded = load(cacheFile, 'payload', 'cacheParams');
catch
  return;
end
if ~isfield(loaded, 'payload') || ~isfield(loaded, 'cacheParams')
  return;
end
if ~manuscript_session_cache_params_match(cacheParams, loaded.cacheParams)
  return;
end

payload = loaded.payload;
didLoad = true;
end
