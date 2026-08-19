function [batchByCell, missingIdx] = try_load_across_tasks_session_cache( ...
    batchByCell, sessionIdx, cellTypesToRun, sessionType, sessionName, subjectName, opts, kind)
% TRY_LOAD_ACROSS_TASKS_SESSION_CACHE - Fill batch entries from per-session cache
%
% Returns:
%   missingIdx - Cell-type indices that still need analysis

missingIdx = 1:numel(cellTypesToRun);
if skip_manuscript_session_cache_load(opts)
  return;
end

missingIdx = [];
for iCell = 1:numel(cellTypesToRun)
  cacheParams = make_manuscript_session_cache_params(kind, opts);
  cacheParams.cellType = cellTypesToRun{iCell};
  cacheFile = manuscript_session_cache_filepath(sessionType, sessionName, subjectName, cacheParams);
  [payload, didLoad] = try_load_manuscript_session_cache(cacheFile, cacheParams);
  if didLoad
    batchByCell{iCell}(sessionIdx).results = payload;
    batchByCell{iCell}(sessionIdx).success = true;
    fprintf('  Loaded cached %s results (%s).\n', kind, cell_type_label(cellTypesToRun{iCell}));
  else
    missingIdx(end + 1) = iCell; %#ok<AGROW>
  end
end
end
