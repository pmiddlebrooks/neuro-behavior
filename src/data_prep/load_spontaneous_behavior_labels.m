function dataStruct = load_spontaneous_behavior_labels(dataStruct, opts)
% LOAD_SPONTANEOUS_BEHAVIOR_LABELS - Attach bhvID from behavior_labels CSV
%
% Variables:
%   dataStruct - Output of load_spontaneous_data / load_session_data
%   opts       - Load options (uses collectStart, collectEnd, fsBhv, dataPath)
%
% Goal:
%   Call this after a spike-only spontaneous load when behavior labels are
%   needed. load_spontaneous_data does not read behavior CSVs.

if nargin < 1 || isempty(dataStruct)
  error('load_spontaneous_behavior_labels:MissingData', 'dataStruct is required.');
end
if nargin < 2 || isempty(opts)
  if isfield(dataStruct, 'opts') && ~isempty(dataStruct.opts)
    opts = dataStruct.opts;
  else
    error('load_spontaneous_behavior_labels:MissingOpts', 'opts is required.');
  end
end

if ~isfield(opts, 'dataPath') || isempty(opts.dataPath) ...
    || ~isfield(opts, 'sessionName') || isempty(opts.sessionName)
  error('load_spontaneous_behavior_labels:MissingSession', ...
    'opts.dataPath and opts.sessionName must be set (after load_spontaneous_data).');
end
if ~isfield(opts, 'fsBhv') || isempty(opts.fsBhv)
  error('behavior needs a sampling frequency, opts.fsBhv, in the opts struct');
end

sessionFolder = fullfile(opts.dataPath, opts.sessionName);
if ~behavior_labels_available(sessionFolder)
  warning('load_spontaneous_behavior_labels:NoBehaviorLabels', ...
    'No behavior_labels CSV in %s; skipping behavior data.', sessionFolder);
  dataStruct = attach_empty_behavior_fields(dataStruct, opts);
  return;
end

dataBhv = load_data(opts, 'behavior');

collectStartBhv = 0;
if isfield(opts, 'collectStart') && ~isempty(opts.collectStart)
  collectStartBhv = opts.collectStart;
end
collectEndBhv = opts.collectEnd;
if isempty(collectEndBhv) && isfield(dataStruct, 'spikeData') ...
    && isfield(dataStruct.spikeData, 'collectEnd')
  collectEndBhv = dataStruct.spikeData.collectEnd;
end

[bhvID, bhvTimeOrigin] = build_bhv_id_vector( ...
  dataBhv, collectStartBhv, collectEndBhv, opts.fsBhv);
dataBhv.StartFrame = abs_time_to_collect_frame( ...
  dataBhv.StartTime, collectStartBhv, 1 / opts.fsBhv, 'round');

dataStruct.bhvID = bhvID;
dataStruct.bhvTimeOrigin = bhvTimeOrigin;
dataStruct.dataBhv = dataBhv;
dataStruct.fsBhv = opts.fsBhv;
if isfield(dataStruct, 'opts')
  dataStruct.opts = opts;
end
end

function tf = behavior_labels_available(sessionFolder)
csvFiles = dir(fullfile(sessionFolder, 'behavior_labels*.csv'));
tf = ~isempty(csvFiles);
end

function dataStruct = attach_empty_behavior_fields(dataStruct, opts)
dataStruct.bhvID = [];
dataStruct.bhvTimeOrigin = [];
dataStruct.dataBhv = [];
if isfield(opts, 'fsBhv') && ~isempty(opts.fsBhv)
  dataStruct.fsBhv = opts.fsBhv;
else
  dataStruct.fsBhv = [];
end
end
