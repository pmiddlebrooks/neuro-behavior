function set_manuscript_av_window(avWindow)
% SET_MANUSCRIPT_AV_WINDOW - Session-wide avWindow override for AV callers
%
% Variables:
%   avWindow - Tile length (s), or [] to clear. Used when a caller config
%              does not yet carry .avWindow (e.g. interval engagement locals).
%
% Goal:
%   Let manuscript scripts (multiple_metrics / av_across_tasks) publish the
%   requested avalanche analysis window so shared detectors can tile and
%   recompute per-window thresholds.

if nargin < 1
  avWindow = [];
end
get_manuscript_av_window(avWindow);
end
