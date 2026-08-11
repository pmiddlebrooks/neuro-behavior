function avWindow = resolve_effective_av_window(config)
% RESOLVE_EFFECTIVE_AV_WINDOW - Config avWindow, else manuscript override
%
% Variables:
%   config - AV config that may include .avWindow
%
% Returns:
%   avWindow - Positive scalar seconds, or [] for full-collect shared threshold
%
% Goal:
%   If config has an .avWindow field, that value wins (including [] = full
%   collect). Otherwise use set_manuscript_av_window (for callers that never
%   copy avWindow onto their local AV config).

avWindow = [];
if nargin >= 1 && isstruct(config) && isfield(config, 'avWindow')
  avWindow = config.avWindow;
else
  avWindow = get_manuscript_av_window();
end
if ~(isnumeric(avWindow) && isscalar(avWindow) && isfinite(avWindow) && avWindow > 0)
  avWindow = [];
end
end
