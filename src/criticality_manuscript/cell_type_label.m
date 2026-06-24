function label = cell_type_label(cellType)
% CELL_TYPE_LABEL - Display label for command-window messages

if isempty(cellType)
  label = 'all units';
elseif strcmpi(cellType, 'all')
  label = 'combined (E+I)';
else
  label = cellType;
end
end
