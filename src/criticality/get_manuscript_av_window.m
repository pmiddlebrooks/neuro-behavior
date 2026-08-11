function avWindow = get_manuscript_av_window(avWindowToStore)
% GET_MANUSCRIPT_AV_WINDOW - Read (or set) the manuscript avWindow override
%
% Variables:
%   avWindowToStore - Optional; when provided, updates the persistent value
%
% Returns:
%   avWindow - Current override ([] = none / full-collect shared threshold)

persistent storedAvWindow
if nargin >= 1
  storedAvWindow = avWindowToStore;
end
if isempty(storedAvWindow)
  avWindow = [];
else
  avWindow = storedAvWindow;
end
end
