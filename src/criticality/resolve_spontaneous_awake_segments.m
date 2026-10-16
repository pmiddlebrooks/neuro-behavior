function [segments, applyFilter] = resolve_spontaneous_awake_segments(dataStruct, config, timeRange)
% RESOLVE_SPONTANEOUS_AWAKE_SEGMENTS - Awake epochs for a spontaneous session
%
% Variables:
%   dataStruct - Loaded session (sessionType, subjectName, sessionName)
%   config     - Analysis config. restrictToAwake (default true) turns the
%                filter on. sleepBrainArea overrides the delta-power area
%                (default 'M56').
%   timeRange  - [start end] analysis interval in seconds
%
% Goal:
%   Call lfp_sleep_detector once per spontaneous session and return awake
%   bouts clipped to timeRange. Task sessions are not filtered. Results are
%   reused across AR / AV / PRG calls in the same MATLAB session, and saved
%   under dropPath so a later pipeline does not reload the LFP.
%
% Returns:
%   segments    - Struct array with .start and .end (s). Empty when none.
%   applyFilter - True only for spontaneous sessions with the filter on.

segments = struct('start', {}, 'end', {});
applyFilter = false;
if ~is_spontaneous_session(dataStruct) || ~spontaneous_restrict_to_awake(config)
  return;
end
if nargin < 3 || numel(timeRange) < 2 || ~(timeRange(2) > timeRange(1))
  error('resolve_spontaneous_awake_segments:BadTimeRange', ...
    'timeRange must be [start end] with end > start.');
end

applyFilter = true;
[sessionType, subjectName, sessionName] = session_identity(dataStruct);
detectorParams = sleep_detector_call_params(config, timeRange);
cacheKey = sleep_cache_key(sessionType, subjectName, sessionName, detectorParams);

[awakeMatrix, didLoad] = load_cached_awake_matrix(cacheKey, detectorParams);
if ~didLoad
  awakeMatrix = detect_awake_matrix(sessionType, subjectName, sessionName, detectorParams);
  store_cached_awake_matrix(cacheKey, detectorParams, awakeMatrix, ...
    sessionType, subjectName, sessionName);
else
  fprintf('Awake epochs (cached): %s / %s\n', subjectName, sessionName);
end

segments = clip_awake_matrix(awakeMatrix, timeRange);
if isscalar(segments) ...
    && segments(1).start <= timeRange(1) + 1 ...
    && segments(1).end >= timeRange(2) - 1
  % Entirely awake: keep the analysis grid aligned to the collect window.
  segments(1).start = timeRange(1);
  segments(1).end = timeRange(2);
end
end

function tf = is_spontaneous_session(dataStruct)
% IS_SPONTANEOUS_SESSION - True when this recording is a spontaneous session
tf = isstruct(dataStruct) && isfield(dataStruct, 'sessionType') ...
  && strcmpi(strtrim(char(dataStruct.sessionType)), 'spontaneous');
end

function tf = spontaneous_restrict_to_awake(config)
% SPONTANEOUS_RESTRICT_TO_AWAKE - True unless the caller turns the filter off
%
% Variables:
%   config - Analysis or cache options. Missing restrictToAwake means on.
%
% Goal:
%   Spontaneous criticality comparisons drop sleep unless restrictToAwake
%   is explicitly false.

tf = true;
if isstruct(config) && isfield(config, 'restrictToAwake') && ~isempty(config.restrictToAwake)
  tf = logical(config.restrictToAwake);
end
end

function [sessionType, subjectName, sessionName] = session_identity(dataStruct)
% SESSION_IDENTITY - Names lfp_sleep_detector needs
sessionType = strtrim(char(dataStruct.sessionType));
sessionName = '';
if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
  sessionName = strtrim(char(dataStruct.sessionName));
end
subjectName = '';
if isfield(dataStruct, 'subjectName') && ~isempty(dataStruct.subjectName)
  subjectName = strtrim(char(dataStruct.subjectName));
end
if isempty(sessionName) || isempty(subjectName)
  error('resolve_spontaneous_awake_segments:MissingIdentity', ...
    'Spontaneous awake filtering needs dataStruct.subjectName and dataStruct.sessionName.');
end
end

function detectorParams = sleep_detector_call_params(config, timeRange)
% SLEEP_DETECTOR_CALL_PARAMS - Settings stored with the awake-epoch cache
detectorParams = struct();
detectorParams.brainArea = 'M56';
if isstruct(config) && isfield(config, 'sleepBrainArea') && ~isempty(config.sleepBrainArea)
  detectorParams.brainArea = char(config.sleepBrainArea);
end
detectorParams.collectStart = timeRange(1);
detectorParams.collectEnd = timeRange(2);
detectorParams.minSegmentLength = [];
if isstruct(config) && isfield(config, 'sleepMinSegmentLength') ...
    && ~isempty(config.sleepMinSegmentLength)
  detectorParams.minSegmentLength = config.sleepMinSegmentLength;
end
end

function cacheKey = sleep_cache_key(sessionType, subjectName, sessionName, detectorParams)
% SLEEP_CACHE_KEY - Identity plus collect window and delta-power area
cacheKey = sprintf('%s|%s|%s|%.3f|%.3f|%s', ...
  sessionType, subjectName, sessionName, ...
  detectorParams.collectStart, detectorParams.collectEnd, detectorParams.brainArea);
if ~isempty(detectorParams.minSegmentLength)
  cacheKey = sprintf('%s|min%.3f', cacheKey, detectorParams.minSegmentLength);
end
end

function [awakeMatrix, didLoad] = load_cached_awake_matrix(cacheKey, detectorParams)
% LOAD_CACHED_AWAKE_MATRIX - Memory cache, then the dropPath file
%
% An empty matrix is a valid hit (no awake time), so didLoad is separate.
awakeMatrix = zeros(0, 2);
didLoad = false;
memoryCache = awake_memory_cache();
if isKey(memoryCache, cacheKey)
  awakeMatrix = memoryCache(cacheKey);
  didLoad = true;
  return;
end
cacheFile = awake_cache_file(cacheKey);
if isempty(cacheFile) || ~isfile(cacheFile)
  return;
end
try
  loaded = load(cacheFile, 'awakeMatrix', 'detectorParams');
catch
  return;
end
if ~isfield(loaded, 'awakeMatrix') || ~isfield(loaded, 'detectorParams')
  return;
end
if ~isequaln(loaded.detectorParams, detectorParams)
  return;
end
awakeMatrix = loaded.awakeMatrix;
didLoad = true;
store_awake_memory(cacheKey, awakeMatrix);
end

function store_cached_awake_matrix(cacheKey, detectorParams, awakeMatrix, ...
    sessionType, subjectName, sessionName)
% STORE_CACHED_AWAKE_MATRIX - Remember awake epochs for later pipelines
store_awake_memory(cacheKey, awakeMatrix);
cacheFile = awake_cache_file(cacheKey, sessionType, subjectName, sessionName);
if isempty(cacheFile)
  return;
end
cacheDir = fileparts(cacheFile);
if ~isfolder(cacheDir)
  mkdir(cacheDir);
end
try
  save(cacheFile, 'awakeMatrix', 'detectorParams');
catch ME
  fprintf('Could not save awake-epoch cache (%s).\n', ME.message);
end
end

function memoryCache = awake_memory_cache()
% AWAKE_MEMORY_CACHE - Persistent map shared by AR, AV, and PRG
persistent memoryMap
if isempty(memoryMap) || ~isa(memoryMap, 'containers.Map')
  memoryMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
end
memoryCache = memoryMap;
end

function store_awake_memory(cacheKey, awakeMatrix)
% STORE_AWAKE_MEMORY - Write one entry into the persistent map
% containers.Map is a handle; the write updates the persistent map.
memoryCache = awake_memory_cache();
memoryCache(cacheKey) = awakeMatrix;
if ~isKey(memoryCache, cacheKey)
  error('resolve_spontaneous_awake_segments:CacheWrite', ...
    'Could not store awake epochs for this session.');
end
end

function cacheFile = awake_cache_file(cacheKey, sessionType, subjectName, sessionName)
% AWAKE_CACHE_FILE - dropPath file for one session and collect window
cacheFile = '';
if nargin < 2
  sessionType = '';
  subjectName = '';
  sessionName = '';
end
try
  paths = get_paths();
catch
  return;
end
if ~isstruct(paths) || ~isfield(paths, 'dropPath') || isempty(paths.dropPath)
  return;
end
if isempty(sessionName)
  % Lookup by key only: scan is unnecessary; the save path uses the names.
  % Callers that load pass the key and also need the names. Encode them.
  parts = strsplit(cacheKey, '|');
  if numel(parts) < 3
    return;
  end
  sessionType = parts{1};
  subjectName = parts{2};
  sessionName = parts{3};
end
sessDir = fullfile(paths.dropPath, 'criticality_manuscript', 'sleep_awake', ...
  matlab.lang.makeValidName(sessionType), ...
  matlab.lang.makeValidName(subjectName), ...
  matlab.lang.makeValidName(sessionName));
cacheFile = fullfile(sessDir, [matlab.lang.makeValidName(cacheKey), '.mat']);
end

function awakeMatrix = detect_awake_matrix(sessionType, subjectName, sessionName, detectorParams)
% DETECT_AWAKE_MATRIX - Run lfp_sleep_detector and keep the awake intervals
callArgs = {'brainArea', detectorParams.brainArea, ...
  'collectStart', detectorParams.collectStart, ...
  'collectEnd', detectorParams.collectEnd, ...
  'makePlot', false};
if ~isempty(detectorParams.minSegmentLength)
  callArgs = [callArgs, {'minSegmentLength', detectorParams.minSegmentLength}];
end
sleepResult = lfp_sleep_detector(sessionType, subjectName, sessionName, callArgs{:});
awakeMatrix = zeros(0, 2);
if isfield(sleepResult, 'awakeSegments') && ~isempty(sleepResult.awakeSegments)
  awakeMatrix = sleepResult.awakeSegments;
end
end

function segments = clip_awake_matrix(awakeMatrix, timeRange)
% CLIP_AWAKE_MATRIX - Keep the overlap of each bout with the analysis interval
segments = struct('start', {}, 'end', {});
if isempty(awakeMatrix)
  return;
end
for iRow = 1:size(awakeMatrix, 1)
  segStart = max(awakeMatrix(iRow, 1), timeRange(1));
  segEnd = min(awakeMatrix(iRow, 2), timeRange(2));
  if segEnd > segStart
    segments(end + 1).start = segStart; %#ok<AGROW>
    segments(end).end = segEnd;
  end
end
end
