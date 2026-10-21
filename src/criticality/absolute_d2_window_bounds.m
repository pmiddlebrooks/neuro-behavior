function [winStartAbs, winEndAbs] = absolute_d2_window_bounds(startS, collectStart, winLen)
% ABSOLUTE_D2_WINDOW_BOUNDS - Window edges in absolute session time
%
% Variables:
%   startS       - Window centers from criticality_ar_analysis (one per window)
%   collectStart - Collect-window start (s)
%   winLen       - Window length (s)
%
% Goal:
%   criticality_ar_analysis stores absolute centers. If every center sits
%   before collectStart, treat them as offsets from that start instead.
%
% Returns:
%   winStartAbs, winEndAbs - Column vectors, [center - winLen/2, center + winLen/2]

centers = startS(:);
if ~isempty(collectStart) && isfinite(collectStart) && collectStart > 1 ...
    && ~isempty(centers) && all(isfinite(centers)) && max(centers) < (collectStart - 1)
  centers = centers + collectStart;
end
winStartAbs = centers - winLen / 2;
winEndAbs = centers + winLen / 2;
end
