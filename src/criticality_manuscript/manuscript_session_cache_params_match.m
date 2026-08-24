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

if ~pca_cache_params_match(requested, stored)
  return;
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

function tf = pca_cache_params_match(requested, stored)
% PCA_CACHE_PARAMS_MATCH - Treat missing pca fields as pcaFlag off (legacy caches)
reqFlag = cache_pca_flag(requested);
stoFlag = cache_pca_flag(stored);
if ~reqFlag && ~stoFlag
  tf = true;
  return;
end
tf = cache_values_equal(reqFlag, stoFlag) ...
  && cache_values_equal(cache_pca_first_flag(requested), cache_pca_first_flag(stored)) ...
  && cache_values_equal(cache_pca_n_dim(requested), cache_pca_n_dim(stored));
end

function tf = cache_pca_flag(s)
tf = false;
if isfield(s, 'pcaFlag') && ~isempty(s.pcaFlag)
  tf = logical(s.pcaFlag);
end
end

function tf = cache_pca_first_flag(s)
tf = true;
if isfield(s, 'pcaFirstFlag') && ~isempty(s.pcaFirstFlag)
  tf = logical(s.pcaFirstFlag);
end
end

function nDim = cache_pca_n_dim(s)
nDim = 0;
if isfield(s, 'nDim') && ~isempty(s.nDim) && isfinite(s.nDim)
  nDim = s.nDim;
end
end
