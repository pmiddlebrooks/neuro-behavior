function colorRgb = colors_for_tasks(sessionType)
% COLORS_FOR_TASKS - Task-specific RGB colors for manuscript figures
%
% Variables:
%   sessionType - 'spontaneous', 'interval', 'reach', or 'schall'
%
% Returns:
%   colorRgb - 1x3 RGB in [0, 1]
%
% Goal:
%   Consistent task colors across criticality_manuscript plots.

switch lower(strtrim(char(sessionType)))
  case 'spontaneous'
    colorRgb = [0.15, 0.45, 0.85];
  case 'interval'
    colorRgb = [0.90, 0.40, 0.15];
  case 'reach'
    colorRgb = [0.20, 0.65, 0.30];
  otherwise
    colorRgb = [0.35, 0.35, 0.35];
end
colorRgb = colorRgb(:)';
end
