function save_engagement_pipeline_caches(sessionOut, sessionType, sessionName, ...
    subjectName, opts, analysesToSave)
% SAVE_ENGAGEMENT_PIPELINE_CACHES - Write one engagement result file per pipeline
%
% Variables:
%   sessionOut - Engagement module output (d2 / avalanches / kurtosis fields)
%   analysesToSave - Cell of analyses computed this run (not already cached)

if ~use_manuscript_session_cache(opts) || isempty(analysesToSave)
  return;
end

for i = 1:numel(analysesToSave)
  analysisName = analysesToSave{i};
  payload = engagement_pipeline_payload(sessionOut, analysisName);
  if isempty(payload)
    continue;
  end
  kind = engagement_pipeline_cache_kind(analysisName);
  cacheParams = make_manuscript_session_cache_params(kind, opts);
  cacheFile = manuscript_session_cache_filepath( ...
    sessionType, sessionName, subjectName, cacheParams);
  save_manuscript_session_cache(cacheFile, payload, cacheParams);
end
end

function payload = engagement_pipeline_payload(sessionOut, analysisName)
payload = [];
switch lower(char(analysisName))
  case 'd2'
    if isfield(sessionOut, 'd2')
      payload = sessionOut.d2;
    end
  case 'avalanches'
    if isfield(sessionOut, 'avalanches')
      payload = sessionOut.avalanches;
    end
  case 'kurtosis'
    if isfield(sessionOut, 'kurtosis')
      payload = sessionOut.kurtosis;
    end
end
end
