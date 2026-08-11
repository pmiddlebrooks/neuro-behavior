function tf = use_local_av_window_thresholds(config)
% USE_LOCAL_AV_WINDOW_THRESHOLDS - True when avWindow requests per-window cutoffs
%
% Variables:
%   config - AV config; .avWindow empty = full collect, shared threshold
%            (falls back to set_manuscript_av_window override)
%
% Goal:
%   avWindow set to a positive scalar means tile time and recompute the
%   population cutoff from each window's own pop activity.

if nargin < 1
  config = struct();
end
avWindow = resolve_effective_av_window(config);
tf = ~isempty(avWindow);
end
