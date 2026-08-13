function avConfigClass = attach_engagement_class_av_thresholds(avConfig, dataStruct, ...
    areasToAnalyze, segments, sharedCollectByArea, classLabel)
% ATTACH_ENGAGEMENT_CLASS_AV_THRESHOLDS - Class-specific sharedThresholdByArea
%
% Variables:
%   avConfig             - Base AV analysis config
%   dataStruct           - Session data
%   areasToAnalyze       - Area indices
%   segments             - Engaged or non-engaged segments (.start, .end)
%   sharedCollectByArea  - Collect-range thresholds (total class + neuron subsets)
%   classLabel           - 'engaged' / 'nonEngaged' (for logging)
%
% Goal:
%   Total uses the collect-range cutoff. Engaged and non-engaged each get a
%   cutoff from that class's pooled pop activity when avWindow is empty.
%   When avWindow is set, reuse collect-range shared info (neuron subsets);
%   per-tile cutoffs are attached later in extract_pooled_area_avalanches.

avConfigClass = avConfig;
if nargin < 6 || isempty(classLabel)
  classLabel = 'class';
end

if use_local_av_window_thresholds(avConfig) || isempty(segments)
  avConfigClass.sharedThresholdByArea = sharedCollectByArea;
  return;
end

avConfigClass.sharedThresholdByArea = prepare_segment_class_av_thresholds_by_area( ...
  dataStruct, areasToAnalyze, avConfig, segments, sharedCollectByArea, classLabel);
end
