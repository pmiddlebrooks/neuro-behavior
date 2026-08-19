function tf = manuscript_session_cache_params_match(requested, stored)
% MANUSCRIPT_SESSION_CACHE_PARAMS_MATCH - True if pipeline cache keys agree
%
% Shared keys: kind, brain area, collect range, bin size, subsampling.
% Window keys are pipeline-specific (d2Window / avWindow / prgWindow).

tf = false;
if ~isstruct(requested) || ~isstruct(stored)
  return;
end

coreFields = {'kind', 'brainArea', 'collectStart', 'collectEnd', ...
  'binSizeD2', 'nSubsamples', 'nNeuronsSubsample'};
for i = 1:numel(coreFields)
  fieldName = coreFields{i};
  if ~cache_values_equal(get_cache_field(requested, fieldName), ...
      get_cache_field(stored, fieldName))
    return;
  end
end

kindGroup = manuscript_session_cache_kind_group(get_cache_field(requested, 'kind'));
if strcmp(kindGroup, 'av')
  windowField = 'avWindow';
elseif strcmp(kindGroup, 'prg')
  windowField = 'prgWindow';
else
  windowField = 'd2Window';
end
if ~cache_values_equal(get_cache_field(requested, windowField), ...
    get_cache_field(stored, windowField))
  return;
end

if isfield(requested, 'cellType') && isfield(stored, 'cellType')
  if ~cache_values_equal(requested.cellType, stored.cellType)
    return;
  end
end

tf = true;
end

function val = get_cache_field(s, fieldName)
if isfield(s, fieldName)
  val = s.(fieldName);
else
  val = [];
end
end

function tf = cache_values_equal(a, b)
if isempty(a) && isempty(b)
  tf = true;
  return;
end
if ischar(a) || isstring(a)
  a = char(a);
end
if ischar(b) || isstring(b)
  b = char(b);
end
tf = isequaln(a, b);
end
