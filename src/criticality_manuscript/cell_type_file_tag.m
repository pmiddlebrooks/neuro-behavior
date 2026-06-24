function tag = cell_type_file_tag(cellType)
% CELL_TYPE_FILE_TAG - Filename suffix for per-population runs

if isempty(cellType)
  tag = '';
elseif strcmpi(cellType, 'all')
  tag = '_combined';
else
  tag = ['_' cellType];
end
end
