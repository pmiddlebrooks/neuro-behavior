function out = merge_struct_defaults(userStruct, defaultsStruct)
% MERGE_STRUCT_DEFAULTS - Fill missing fields in userStruct from defaultsStruct

if nargin < 1 || isempty(userStruct)
  out = defaultsStruct;
  return;
end
out = userStruct;
defaultFields = fieldnames(defaultsStruct);
for i = 1:numel(defaultFields)
  fieldName = defaultFields{i};
  if ~isfield(out, fieldName) || isempty(out.(fieldName))
    out.(fieldName) = defaultsStruct.(fieldName);
  end
end
end
