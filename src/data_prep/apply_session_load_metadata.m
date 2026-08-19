function [opts, metadata] = apply_session_load_metadata(sessionType, subjectName, sessionName, opts)
% APPLY_SESSION_LOAD_METADATA - Merge per-session metadata into load opts
%
% Variables:
%   sessionType - 'interval', 'semicircle', 'reach', 'spontaneous', 'schall', ...
%   subjectName - Subject folder / id ('' when unused)
%   sessionName - Session identifier
%   opts        - Loading options (neuro_behavior_options or equivalent)
%
% Goal:
%   Look up <task>_session_metadata(subjectName, sessionName) and apply
%   overrides. collectStartMin floors collectStart; collectEndMax caps a
%   finite collectEnd; other nonempty fields copy onto opts.
%
% Returns:
%   opts     - Options with session metadata applied
%   metadata - Raw metadata struct (empty if none)

if nargin < 4 || isempty(opts)
  opts = struct();
end
if nargin < 2
  subjectName = '';
end
if nargin < 3
  sessionName = '';
end

metadata = lookup_session_load_metadata(sessionType, subjectName, sessionName);
if isempty(metadata) || isempty(fieldnames(metadata))
  return;
end

skipFields = {'collectStartMin', 'collectEndMax', 'notes', 'comment'};
changedMsg = {};

% Hard overrides first, then min/max so floors/caps still apply
fn = fieldnames(metadata);
for iField = 1:numel(fn)
  fieldName = fn{iField};
  if any(strcmp(fieldName, skipFields))
    continue;
  end
  fieldVal = metadata.(fieldName);
  if isempty(fieldVal)
    continue;
  end
  oldVal = [];
  if isfield(opts, fieldName)
    oldVal = opts.(fieldName);
  end
  opts.(fieldName) = fieldVal;
  changedMsg{end+1} = sprintf('%s %s -> %s', fieldName, ...
    format_opt_value(oldVal), format_opt_value(fieldVal)); %#ok<AGROW>
end

if isfield(metadata, 'collectStartMin') && ~isempty(metadata.collectStartMin)
  oldStart = [];
  if isfield(opts, 'collectStart')
    oldStart = opts.collectStart;
  end
  if isempty(oldStart)
    opts.collectStart = metadata.collectStartMin;
  else
    opts.collectStart = max(oldStart, metadata.collectStartMin);
  end
  if isempty(oldStart) || opts.collectStart ~= oldStart
    changedMsg{end+1} = sprintf('collectStart %s -> %.3g (min=%.3g)', ...
      format_opt_value(oldStart), opts.collectStart, metadata.collectStartMin); %#ok<AGROW>
  end
end

if isfield(metadata, 'collectEndMax') && ~isempty(metadata.collectEndMax)
  oldEnd = [];
  if isfield(opts, 'collectEnd')
    oldEnd = opts.collectEnd;
  end
  if ~isempty(oldEnd) && isfinite(oldEnd)
    opts.collectEnd = min(oldEnd, metadata.collectEndMax);
    if opts.collectEnd ~= oldEnd
      changedMsg{end+1} = sprintf('collectEnd %.3g -> %.3g (max=%.3g)', ...
        oldEnd, opts.collectEnd, metadata.collectEndMax); %#ok<AGROW>
    end
  end
end

if ~isempty(changedMsg)
  fprintf('Session metadata (%s / %s / %s): %s\n', ...
    char(sessionType), format_opt_value(subjectName), char(sessionName), ...
    strjoin(changedMsg, '; '));
end
end

function metadata = lookup_session_load_metadata(sessionType, subjectName, sessionName)
% LOOKUP_SESSION_LOAD_METADATA - Dispatch to the task-specific metadata function

metadata = struct();
sessionType = lower(strtrim(char(sessionType)));
srcPath = fullfile(fileparts(mfilename('fullpath')), '..');
switch sessionType
  case 'interval'
    fcnName = 'interval_session_metadata';
    taskDir = fullfile(srcPath, 'interval_timing_task');
  case 'semicircle'
    fcnName = 'semicircle_session_metadata';
    taskDir = fullfile(srcPath, 'semicircle_reward_task');
  case 'reach'
    fcnName = 'reach_session_metadata';
    taskDir = fullfile(srcPath, 'reach_task');
  case 'spontaneous'
    fcnName = 'spontaneous_session_metadata';
    taskDir = fullfile(srcPath, 'spontaneous');
  case 'schall'
    fcnName = 'schall_session_metadata';
    taskDir = fullfile(srcPath, 'schall');
  otherwise
    return;
end

if exist(fcnName, 'file') ~= 2 && exist(taskDir, 'dir')
  addpath(taskDir);
end
if exist(fcnName, 'file') ~= 2
  return;
end
metadata = feval(fcnName, subjectName, sessionName);
if ~isstruct(metadata)
  metadata = struct();
end
end

function text = format_opt_value(value)
% FORMAT_OPT_VALUE - Short string for metadata apply log

if nargin < 1 || isempty(value)
  text = '[]';
elseif isnumeric(value) && isscalar(value)
  text = sprintf('%.3g', value);
elseif ischar(value)
  text = value;
elseif isstring(value) && isscalar(value)
  text = char(value);
else
  text = class(value);
end
end
