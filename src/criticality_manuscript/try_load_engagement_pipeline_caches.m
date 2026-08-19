function [loaded, missingAnalyses] = try_load_engagement_pipeline_caches( ...
    sessionType, sessionName, subjectName, opts, analyses)
% TRY_LOAD_ENGAGEMENT_PIPELINE_CACHES - Load d2 / AV / PRG engagement files
%
% Variables:
%   sessionType, sessionName, subjectName - Session identity
%   opts - Engagement batch options (used for cache param matching)
%   analyses - Cell of {'d2','avalanches','kurtosis'} requested this run
%
% Goal:
%   Return cached payloads for pipelines that match; list analyses still needed.

loaded = struct('d2', [], 'avalanches', [], 'kurtosis', []);
missingAnalyses = cellstr(analyses(:)');
if skip_manuscript_session_cache_load(opts)
  return;
end

missingAnalyses = {};
for i = 1:numel(analyses)
  analysisName = analyses{i};
  kind = engagement_pipeline_cache_kind(analysisName);
  cacheParams = make_manuscript_session_cache_params(kind, opts);
  cacheFile = manuscript_session_cache_filepath( ...
    sessionType, sessionName, subjectName, cacheParams);
  [payload, didLoad] = try_load_manuscript_session_cache(cacheFile, cacheParams);
  if didLoad
    loaded.(engagement_pipeline_result_field(analysisName)) = payload;
    fprintf('  Loaded cached engagement %s:\n    %s\n', analysisName, cacheFile);
  else
    missingAnalyses{end + 1} = analysisName; %#ok<AGROW>
  end
end
end

function fieldName = engagement_pipeline_result_field(analysisName)
switch lower(char(analysisName))
  case 'd2'
    fieldName = 'd2';
  case 'avalanches'
    fieldName = 'avalanches';
  case 'kurtosis'
    fieldName = 'kurtosis';
  otherwise
    fieldName = lower(char(analysisName));
end
end
