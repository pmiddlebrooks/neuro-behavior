function figNumber = figure_number_for_task(sessionType, plotKind, cellType)
% FIGURE_NUMBER_FOR_TASK - Stable MATLAB figure numbers per task and plot role
%
% Variables:
%   sessionType - 'spontaneous', 'interval', 'reach', or 'schall'
%   plotKind    - 'distributions', 'ei_summary', or 'bin_size_sweep'
%   cellType    - '', 'all', 'excitatory', or 'inhibitory' (distributions only)
%
% Returns:
%   figNumber - Integer passed to figure(figNumber) to reuse the same window
%
% Goal:
%   Replot new session data in the same figure handle for each task.

if nargin < 2 || isempty(plotKind)
  plotKind = 'distributions';
end
if nargin < 3
  cellType = '';
end

switch lower(strtrim(char(sessionType)))
  case 'spontaneous'
    baseNumber = 8100;
  case 'interval'
    baseNumber = 8200;
  case 'reach'
    baseNumber = 8300;
  case 'schall'
    baseNumber = 8400;
  otherwise
    baseNumber = 8900;
end

switch lower(strtrim(char(plotKind)))
  case 'ei_summary'
    figNumber = baseNumber + 10;
  case 'bin_size_sweep'
    figNumber = baseNumber + 20;
  otherwise
    figNumber = baseNumber + cell_type_figure_offset(cellType);
end
end

function offset = cell_type_figure_offset(cellType)
if isempty(cellType) || strcmpi(cellType, 'all')
  offset = 0;
elseif strcmpi(cellType, 'excitatory')
  offset = 1;
elseif strcmpi(cellType, 'inhibitory')
  offset = 2;
else
  offset = 0;
end
end
