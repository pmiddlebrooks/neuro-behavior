%%
% Print Saved Datasets (Manuscript Session Cache)
%
% Walks dropPath/criticality_manuscript per-session pipeline caches written by
% criticality_multiple_metrics_across_tasks (and the AR / AV / PRG / engagement
% across-task runners). Each cache .mat stores cacheParams that distinguish
% datasets: kind, brainArea, collect window, pipeline window, bin size,
% subsampling, PCA, and d2 method.
%
% Goal:
%   Print the distinct configuration-variable sets that have saved data.
%   Session names are not listed; only unique config tuples and how many
%   cache files share each tuple.

setup_criticality_manuscript_paths('print_saved_datasets');
paths = get_paths();
cacheRoot = fullfile(paths.dropPath, 'criticality_manuscript');

fprintf('\n=== Saved manuscript datasets ===\n');
fprintf('Cache root: %s\n', cacheRoot);
if ~exist(cacheRoot, 'dir')
  error('print_saved_datasets:NoCacheRoot', 'Cache folder not found: %s', cacheRoot);
end

matList = dir(fullfile(cacheRoot, '**', '*.mat'));
nMat = numel(matList);
fprintf('Found %d .mat files (recursive).\n', nMat);

fileRecords = struct('key', {}, 'kind', {}, 'taskType', {}, 'display', {});
nSkippedNoParams = 0;
nSkippedLoadError = 0;
nSkippedFigures = 0;

for iFile = 1:nMat
  matPath = fullfile(matList(iFile).folder, matList(iFile).name);
  taskType = task_type_from_cache_path(matPath, cacheRoot);
  if strcmpi(taskType, 'figures')
    nSkippedFigures = nSkippedFigures + 1;
    continue;
  end

  if ~mat_file_has_variable(matPath, 'cacheParams')
    nSkippedNoParams = nSkippedNoParams + 1;
    continue;
  end

  try
    loaded = load(matPath, 'cacheParams');
  catch
    nSkippedLoadError = nSkippedLoadError + 1;
    continue;
  end
  if ~isfield(loaded, 'cacheParams') || ~isstruct(loaded.cacheParams)
    nSkippedNoParams = nSkippedNoParams + 1;
    continue;
  end

  cacheParams = loaded.cacheParams;
  displayRow = summarize_cache_params(cacheParams);
  fileRecords(end + 1).key = config_key_from_display(displayRow); %#ok<AGROW>
  fileRecords(end).kind = displayRow.kind;
  fileRecords(end).taskType = taskType;
  fileRecords(end).display = displayRow;

  if nMat > 40 && mod(iFile, 50) == 0
    fprintf('  scanned %d / %d files\n', iFile, nMat);
  end
end

nCache = numel(fileRecords);
fprintf('\nCache files with cacheParams: %d\n', nCache);
fprintf('Skipped (no cacheParams): %d\n', nSkippedNoParams);
fprintf('Skipped (load error): %d\n', nSkippedLoadError);
if nSkippedFigures > 0
  fprintf('Skipped (figures folder): %d\n', nSkippedFigures);
end
if nCache == 0
  fprintf('\nNo session-cache datasets found.\n');
  return;
end

print_unique_variable_sets(fileRecords);
print_unique_config_combinations(fileRecords);

fprintf('\n=== Done ===\n');

%% Local functions

function taskType = task_type_from_cache_path(matPath, cacheRoot)
% TASK_TYPE_FROM_CACHE_PATH - First folder under the cache root (task name)

relPath = strrep(matPath, [cacheRoot filesep], '');
relPath = strrep(relPath, [cacheRoot '/'], '');
parts = strsplit(relPath, {filesep, '/'});
if isempty(parts) || isempty(parts{1})
  taskType = '';
else
  taskType = parts{1};
end
end

function tf = mat_file_has_variable(matPath, varName)
% MAT_FILE_HAS_VARIABLE - True if the .mat lists varName (no full load)

tf = false;
try
  info = whos('-file', matPath);
catch
  return;
end
if isempty(info)
  return;
end
tf = any(strcmp({info.name}, varName));
end

function displayRow = summarize_cache_params(cacheParams)
% SUMMARIZE_CACHE_PARAMS - Canonical config fields used to tag caches
%
% Variables:
%   cacheParams - Struct saved next to payload in session cache files
%
% Goal:
%   One row of the keys from make_manuscript_session_cache_params, with
%   missing/legacy fields filled the same way as cache matching.

kind = char(get_cache_char(cacheParams, 'kind', ''));
kindGroup = 'd2';
if ~isempty(kind)
  kindGroup = manuscript_session_cache_kind_group(kind);
end

displayRow = struct();
displayRow.kind = kind;
displayRow.cellType = char(get_cache_char(cacheParams, 'cellType', ''));
displayRow.brainArea = char(get_cache_char(cacheParams, 'brainArea', 'allareas'));
displayRow.collectStart = get_cache_numeric(cacheParams, 'collectStart', 0);
displayRow.collectEnd = get_cache_numeric(cacheParams, 'collectEnd', []);
if strcmp(kindGroup, 'av')
  displayRow.windowField = 'avWindow';
  displayRow.windowSec = get_cache_numeric(cacheParams, 'avWindow', []);
elseif strcmp(kindGroup, 'prg')
  displayRow.windowField = 'prgWindow';
  displayRow.windowSec = get_cache_numeric(cacheParams, 'prgWindow', []);
else
  displayRow.windowField = 'd2Window';
  displayRow.windowSec = get_cache_numeric(cacheParams, 'd2Window', []);
end
displayRow.binSize = get_cache_numeric(cacheParams, 'binSizeD2', []);
displayRow.nSubsamples = get_cache_numeric(cacheParams, 'nSubsamples', 0);
displayRow.nNeuronsSubsample = get_cache_numeric(cacheParams, 'nNeuronsSubsample', 0);
displayRow.pcaFlag = logical(get_cache_numeric(cacheParams, 'pcaFlag', 0));
displayRow.pcaFirstFlag = true;
if isfield(cacheParams, 'pcaFirstFlag') && ~isempty(cacheParams.pcaFirstFlag)
  displayRow.pcaFirstFlag = logical(cacheParams.pcaFirstFlag);
end
displayRow.nDim = get_cache_numeric(cacheParams, 'nDim', 0);
displayRow.d2Method = 'euclidean';
if strcmp(kindGroup, 'd2')
  displayRow.d2Method = char(get_cache_char(cacheParams, 'd2Method', 'euclidean'));
end
displayRow.klFitMethod = 'MaxLikelihood';
displayRow.klErrBars = false;
if strcmp(displayRow.d2Method, 'kl')
  displayRow.klFitMethod = char(get_cache_char(cacheParams, 'klFitMethod', 'MaxLikelihood'));
  if isfield(cacheParams, 'klErrBars') && ~isempty(cacheParams.klErrBars)
    displayRow.klErrBars = logical(cacheParams.klErrBars);
  end
end
end

function key = config_key_from_display(displayRow)
% CONFIG_KEY_FROM_DISPLAY - Stable identity string for one config tuple

key = strjoin({ ...
  displayRow.kind, ...
  displayRow.cellType, ...
  displayRow.brainArea, ...
  format_numeric_token(displayRow.collectStart), ...
  format_numeric_token(displayRow.collectEnd), ...
  displayRow.windowField, ...
  format_numeric_token(displayRow.windowSec), ...
  format_numeric_token(displayRow.binSize), ...
  format_numeric_token(displayRow.nSubsamples), ...
  format_numeric_token(displayRow.nNeuronsSubsample), ...
  format_logical_token(displayRow.pcaFlag), ...
  format_logical_token(displayRow.pcaFirstFlag), ...
  format_numeric_token(displayRow.nDim), ...
  displayRow.d2Method, ...
  displayRow.klFitMethod, ...
  format_logical_token(displayRow.klErrBars)}, '|');
end

function print_unique_variable_sets(fileRecords)
% PRINT_UNIQUE_VARIABLE_SETS - Distinct values observed for each config field

fprintf('\n--- Distinct values (any pipeline) ---\n');
print_unique_field_values('kind', {fileRecords.kind});
print_unique_field_values('cellType', cellfun(@(d) d.cellType, {fileRecords.display}, ...
  'UniformOutput', false));
print_unique_field_values('brainArea', cellfun(@(d) d.brainArea, {fileRecords.display}, ...
  'UniformOutput', false));
print_unique_field_values('collectStart', cellfun(@(d) format_numeric_token(d.collectStart), ...
  {fileRecords.display}, 'UniformOutput', false));
print_unique_field_values('collectEnd', cellfun(@(d) format_collect_end_token(d.collectEnd), ...
  {fileRecords.display}, 'UniformOutput', false));
print_unique_field_values('d2Window', window_tokens_for_field(fileRecords, 'd2Window'));
print_unique_field_values('avWindow', window_tokens_for_field(fileRecords, 'avWindow'));
print_unique_field_values('prgWindow', window_tokens_for_field(fileRecords, 'prgWindow'));
print_unique_field_values('binSize', cellfun(@(d) format_bin_token(d.binSize), ...
  {fileRecords.display}, 'UniformOutput', false));
print_unique_field_values('subsampling', cellfun(@(d) format_subsample_token(d), ...
  {fileRecords.display}, 'UniformOutput', false));
print_unique_field_values('pca', cellfun(@(d) format_pca_token(d), ...
  {fileRecords.display}, 'UniformOutput', false));
isD2 = cellfun(@(kindName) strcmp(manuscript_session_cache_kind_group(kindName), 'd2'), ...
  {fileRecords.kind});
if any(isD2)
  print_unique_field_values('d2Method', cellfun(@(d) format_d2_method_token(d), ...
    {fileRecords(isD2).display}, 'UniformOutput', false));
end
print_unique_field_values('taskType', {fileRecords.taskType});
end

function tokens = window_tokens_for_field(fileRecords, windowField)
tokens = {};
for i = 1:numel(fileRecords)
  displayRow = fileRecords(i).display;
  if strcmp(displayRow.windowField, windowField)
    tokens{end + 1} = format_window_token(displayRow.windowSec); %#ok<AGROW>
  end
end
end

function print_unique_field_values(fieldName, values)
if isempty(values)
  return;
end
values = values(:)';
values = cellfun(@char, values, 'UniformOutput', false);
values = strtrim(values);
values(cellfun(@isempty, values)) = {'(empty)'};
values = unique(values);
fprintf('  %-14s %s\n', [fieldName, ':'], strjoin(values, ', '));
end

function print_unique_config_combinations(fileRecords)
% PRINT_UNIQUE_CONFIG_COMBINATIONS - One line per unique cacheParams tuple

allKeys = {fileRecords.key};
uniqueKeys = unique(allKeys, 'stable');
kindList = {fileRecords.kind};
uniqueKinds = unique(kindList, 'stable');

fprintf('\n--- Unique configurations (%d) ---\n', numel(uniqueKeys));
for iKind = 1:numel(uniqueKinds)
  kindName = uniqueKinds{iKind};
  kindMask = strcmp(kindList, kindName);
  kindKeys = unique(allKeys(kindMask), 'stable');
  fprintf('\n  %s  (%d configs)\n', kindName, numel(kindKeys));
  for iCfg = 1:numel(kindKeys)
    thisKey = kindKeys{iCfg};
    matchIdx = find(strcmp(allKeys, thisKey));
    displayRow = fileRecords(matchIdx(1)).display;
    nFiles = numel(matchIdx);
    taskTypes = unique({fileRecords(matchIdx).taskType});
    fprintf('    %s\n', format_config_line(displayRow, nFiles, taskTypes));
  end
end
end

function lineText = format_config_line(displayRow, nFiles, taskTypes)
% FORMAT_CONFIG_LINE - Single summary line for one saved config set

bits = {};
if ~isempty(displayRow.cellType)
  bits{end + 1} = sprintf('cellType=%s', displayRow.cellType); %#ok<AGROW>
end
bits{end + 1} = sprintf('brainArea=%s', nonempty_or(displayRow.brainArea, 'allareas')); %#ok<AGROW>
bits{end + 1} = sprintf('collect=%s', format_collect_range( ...
  displayRow.collectStart, displayRow.collectEnd)); %#ok<AGROW>
bits{end + 1} = sprintf('%s=%s', displayRow.windowField, ...
  format_window_token(displayRow.windowSec)); %#ok<AGROW>
bits{end + 1} = sprintf('binSize=%s', format_bin_token(displayRow.binSize)); %#ok<AGROW>
bits{end + 1} = sprintf('sub=%s', format_subsample_token(displayRow)); %#ok<AGROW>
bits{end + 1} = sprintf('pca=%s', format_pca_token(displayRow)); %#ok<AGROW>
if ~isempty(displayRow.kind) && strcmp(manuscript_session_cache_kind_group(displayRow.kind), 'd2')
  bits{end + 1} = sprintf('d2Method=%s', format_d2_method_token(displayRow)); %#ok<AGROW>
end
bits{end + 1} = sprintf('nFiles=%d', nFiles); %#ok<AGROW>
bits{end + 1} = sprintf('tasks=%s', strjoin(taskTypes, ',')); %#ok<AGROW>
lineText = strjoin(bits, '  ');
end

function text = format_collect_range(collectStart, collectEnd)
if isempty(collectEnd) || ~(isnumeric(collectEnd) && isfinite(collectEnd))
  text = sprintf('%s-full', format_numeric_token(collectStart));
else
  text = sprintf('%s-%ss', format_numeric_token(collectStart), format_numeric_token(collectEnd));
end
end

function text = format_collect_end_token(collectEnd)
if isempty(collectEnd) || ~(isnumeric(collectEnd) && isfinite(collectEnd))
  text = 'full';
else
  text = format_numeric_token(collectEnd);
end
end

function text = format_window_token(windowSec)
if isempty(windowSec) || ~(isnumeric(windowSec) && isfinite(windowSec))
  text = 'full';
else
  text = sprintf('%gs', windowSec);
end
end

function text = format_bin_token(binSize)
if isempty(binSize) || ~(isnumeric(binSize) && isfinite(binSize))
  text = 'na';
else
  text = sprintf('%gs', binSize);
end
end

function text = format_subsample_token(displayRow)
if isempty(displayRow.nNeuronsSubsample) || displayRow.nNeuronsSubsample <= 0
  text = 'off';
else
  nSub = displayRow.nSubsamples;
  if isempty(nSub)
    nSub = 0;
  end
  text = sprintf('%dx%d', nSub, displayRow.nNeuronsSubsample);
end
end

function text = format_pca_token(displayRow)
if ~displayRow.pcaFlag
  text = 'off';
  return;
end
orderTag = 'first';
if ~displayRow.pcaFirstFlag
  orderTag = 'last';
end
text = sprintf('on_%s%d', orderTag, displayRow.nDim);
end

function text = format_d2_method_token(displayRow)
if strcmp(displayRow.d2Method, 'kl')
  errTag = 'off';
  if displayRow.klErrBars
    errTag = 'on';
  end
  text = sprintf('kl/%s/err=%s', displayRow.klFitMethod, errTag);
else
  text = displayRow.d2Method;
end
end

function text = format_numeric_token(value)
if isempty(value) || ~(isnumeric(value) && isscalar(value) && isfinite(value))
  text = '[]';
elseif value == floor(value)
  text = sprintf('%.0f', value);
else
  text = sprintf('%g', value);
end
end

function text = format_logical_token(value)
if logical(value)
  text = '1';
else
  text = '0';
end
end

function text = nonempty_or(value, fallback)
text = char(value);
if isempty(text)
  text = fallback;
end
end

function value = get_cache_char(cacheParams, fieldName, defaultVal)
value = defaultVal;
if isfield(cacheParams, fieldName) && ~isempty(cacheParams.(fieldName))
  value = char(cacheParams.(fieldName));
end
end

function value = get_cache_numeric(cacheParams, fieldName, defaultVal)
value = defaultVal;
if ~isfield(cacheParams, fieldName) || isempty(cacheParams.(fieldName))
  return;
end
fieldVal = cacheParams.(fieldName);
if isnumeric(fieldVal) && isscalar(fieldVal)
  value = fieldVal;
end
end
