function avConfig = mark_per_segment_av_config(avConfig)
% MARK_PER_SEGMENT_AV_CONFIG - One threshold per segment; ignore avWindow tiles
%
% Variables:
%   avConfig - Avalanche analysis config (modified copy returned)
%
% Goal:
%   Engagement classes treat each continuous segment as one analysis window
%   and compute that segment's population cutoff from its own activity.
%   avWindow still applies to the total (full-session) class.

avConfig.perSegmentThreshold = true;
avConfig.avWindow = [];
end
