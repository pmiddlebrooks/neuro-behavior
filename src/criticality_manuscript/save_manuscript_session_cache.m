function save_manuscript_session_cache(cacheFile, payload, cacheParams)
% SAVE_MANUSCRIPT_SESSION_CACHE - Write per-session results next to matching params

if nargin < 3 || isempty(cacheFile) || isempty(cacheParams)
  error('save_manuscript_session_cache:MissingArgs', ...
    'cacheFile, payload, and cacheParams are required.');
end

cacheDir = fileparts(cacheFile);
if ~exist(cacheDir, 'dir')
  mkdir(cacheDir);
end

save(cacheFile, 'payload', 'cacheParams', '-v7.3');
fprintf('  Saved session cache:\n    %s\n', cacheFile);
end
