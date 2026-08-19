function kind = engagement_pipeline_cache_kind(analysisName)
% ENGAGEMENT_PIPELINE_CACHE_KIND - Per-pipeline cache tag for engagement splits

switch lower(strtrim(char(analysisName)))
  case 'd2'
    kind = 'engagement_d2';
  case 'avalanches'
    kind = 'engagement_av';
  case 'kurtosis'
    kind = 'engagement_prg';
  otherwise
    error('engagement_pipeline_cache_kind:UnknownAnalysis', ...
      'Unknown engagement analysis "%s".', char(analysisName));
end
end
