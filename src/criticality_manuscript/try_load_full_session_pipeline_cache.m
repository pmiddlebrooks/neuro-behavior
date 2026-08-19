function payload = try_load_full_session_pipeline_cache(kind, sessionType, ...
    sessionName, subjectName, opts)
% TRY_LOAD_FULL_SESSION_PIPELINE_CACHE - Load AR / AV / PRG session cache
%
% Used by engagement split to reuse full-session d2/PRG windows without
% recomputing those pipelines.

payload = [];
if skip_manuscript_session_cache_load(opts)
  return;
end
cacheParams = make_manuscript_session_cache_params(kind, opts);
cacheFile = manuscript_session_cache_filepath( ...
  sessionType, sessionName, subjectName, cacheParams);
[payload, didLoad] = try_load_manuscript_session_cache(cacheFile, cacheParams);
if didLoad
  if ~full_session_cache_can_split(kind, payload)
    fprintf('  Full-session %s cache lacks window times; cannot split by engagement.\n', kind);
    payload = [];
    return;
  end
  fprintf('  Using full-session %s cache for engagement split:\n    %s\n', ...
    kind, cacheFile);
end
end

function tf = full_session_cache_can_split(kind, payload)
tf = isstruct(payload);
if ~tf
  return;
end
kindGroup = manuscript_session_cache_kind_group(kind);
if strcmp(kindGroup, 'd2')
  tf = isfield(payload, 'd2') && isfield(payload, 'startS') && ~isempty(payload.startS);
elseif strcmp(kindGroup, 'prg')
  tf = isfield(payload, 'kappa') && isfield(payload, 'windowStartS') ...
    && ~isempty(payload.windowStartS);
else
  tf = false;
end
end
