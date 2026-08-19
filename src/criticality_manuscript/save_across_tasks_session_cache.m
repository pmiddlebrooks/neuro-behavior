function save_across_tasks_session_cache(results, sessionType, sessionName, subjectName, ...
    opts, kind, cellType)
% SAVE_ACROSS_TASKS_SESSION_CACHE - Store one pipeline/cell-type session result
% Used by AR / AV / PRG across-task batches.

if ~use_manuscript_session_cache(opts)
  return;
end
cacheParams = make_manuscript_session_cache_params(kind, opts);
cacheParams.cellType = cellType;
cacheFile = manuscript_session_cache_filepath(sessionType, sessionName, subjectName, cacheParams);
save_manuscript_session_cache(cacheFile, results, cacheParams);
end
