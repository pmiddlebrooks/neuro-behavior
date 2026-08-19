function cacheFile = manuscript_session_cache_filepath(sessionType, sessionName, subjectName, cacheParams)
% MANUSCRIPT_SESSION_CACHE_FILEPATH - Per-session cache .mat under dropPath
%
% Layout:
%   <dropPath>/criticality_manuscript/<task>/<subject?>/<session>/<file>.mat
% File name includes pipeline, brain area, collect range, that pipeline's
% window and bin size, subsample count, and subsample size.

if nargin < 3
  subjectName = '';
end
if nargin < 4 || isempty(cacheParams)
  error('manuscript_session_cache_filepath:MissingParams', 'cacheParams is required.');
end

paths = get_paths();
taskTag = matlab.lang.makeValidName(char(sessionType));
sessTag = matlab.lang.makeValidName(char(sessionName));
sessDir = fullfile(paths.dropPath, 'criticality_manuscript', taskTag);
if ~isempty(subjectName)
  sessDir = fullfile(sessDir, matlab.lang.makeValidName(char(subjectName)));
end
sessDir = fullfile(sessDir, sessTag);

kindTag = matlab.lang.makeValidName(char(cacheParams.kind));
kindGroup = manuscript_session_cache_kind_group(cacheParams.kind);
areaTag = 'allareas';
if isfield(cacheParams, 'brainArea') && ~isempty(cacheParams.brainArea)
  areaTag = matlab.lang.makeValidName(char(cacheParams.brainArea));
end
cellTag = '';
if isfield(cacheParams, 'cellType') && ~isempty(cacheParams.cellType)
  cellTag = ['_', matlab.lang.makeValidName(char(cacheParams.cellType))];
end

collectStart = 0;
if isfield(cacheParams, 'collectStart') && ~isempty(cacheParams.collectStart)
  collectStart = cacheParams.collectStart;
end
if isfield(cacheParams, 'collectEnd') && ~isempty(cacheParams.collectEnd) ...
    && isnumeric(cacheParams.collectEnd) && isfinite(cacheParams.collectEnd)
  collectTag = sprintf('c%.0fto%.0f', collectStart, cacheParams.collectEnd);
else
  collectTag = sprintf('c%.0ftofull', collectStart);
end

windowTag = pipeline_window_tag(kindGroup, cacheParams);
binTag = format_pipeline_bin_tag(kindGroup, cacheParams);
subTag = 'suboff';
if isfield(cacheParams, 'nNeuronsSubsample') && cacheParams.nNeuronsSubsample > 0
  nSubsamples = 0;
  if isfield(cacheParams, 'nSubsamples') && ~isempty(cacheParams.nSubsamples)
    nSubsamples = cacheParams.nSubsamples;
  end
  subTag = sprintf('sub%dx%d', nSubsamples, cacheParams.nNeuronsSubsample);
end

fileName = sprintf('%s%s_%s_%s_%s_%s_%s.mat', ...
  kindTag, cellTag, areaTag, collectTag, windowTag, binTag, subTag);
cacheFile = fullfile(sessDir, fileName);
end

function windowTag = pipeline_window_tag(kindGroup, cacheParams)
if strcmp(kindGroup, 'av')
  windowTag = format_window_or_full_tag(get_cache_field(cacheParams, 'avWindow'), 'avw');
elseif strcmp(kindGroup, 'prg')
  windowTag = format_window_or_full_tag(get_cache_field(cacheParams, 'prgWindow'), 'prgw');
else
  windowTag = format_window_or_full_tag(get_cache_field(cacheParams, 'd2Window'), 'd2w');
end
end

function val = get_cache_field(s, fieldName)
if isfield(s, fieldName)
  val = s.(fieldName);
else
  val = [];
end
end

function tag = format_window_or_full_tag(windowSec, prefix)
if isempty(windowSec) || ~(isnumeric(windowSec) && isscalar(windowSec) && isfinite(windowSec))
  tag = sprintf('%sfull', prefix);
else
  tag = sprintf('%s%.0fs', prefix, windowSec);
end
end

function binTag = format_pipeline_bin_tag(kindGroup, cacheParams)
if strcmp(kindGroup, 'av')
  prefix = 'binAv';
elseif strcmp(kindGroup, 'prg')
  prefix = 'binPrg';
else
  prefix = 'binD2';
end
if isfield(cacheParams, 'binSizeD2') && ~isempty(cacheParams.binSizeD2) ...
    && isnumeric(cacheParams.binSizeD2) && isfinite(cacheParams.binSizeD2)
  binTag = sprintf('%s_%dms', prefix, round(cacheParams.binSizeD2 * 1000));
else
  binTag = sprintf('%s_na', prefix);
end
end
