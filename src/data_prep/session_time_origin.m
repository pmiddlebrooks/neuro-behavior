function t0 = session_time_origin(dataStruct)
% SESSION_TIME_ORIGIN - Absolute session time of loaded collect-window start
%
% Variables:
%   dataStruct - Session struct (bhvTimeOrigin / spikeData / opts)
%
% Goal:
%   Resolve where frame/bin index 1 and relative t=0 map in absolute session
%   time. Prefer explicit bhvTimeOrigin, then spikeData.collectStart, then
%   opts.collectStart.
%
% Returns:
%   t0 - Absolute start time in seconds (default 0)

t0 = 0;
if nargin < 1 || isempty(dataStruct) || ~isstruct(dataStruct)
  return;
end

if isfield(dataStruct, 'bhvTimeOrigin') && ~isempty(dataStruct.bhvTimeOrigin) ...
    && isfinite(dataStruct.bhvTimeOrigin)
  t0 = dataStruct.bhvTimeOrigin;
  return;
end

if isfield(dataStruct, 'spikeData') && isstruct(dataStruct.spikeData) ...
    && isfield(dataStruct.spikeData, 'collectStart') ...
    && ~isempty(dataStruct.spikeData.collectStart) ...
    && isfinite(dataStruct.spikeData.collectStart)
  t0 = dataStruct.spikeData.collectStart;
  return;
end

if isfield(dataStruct, 'opts') && isstruct(dataStruct.opts) ...
    && isfield(dataStruct.opts, 'collectStart') ...
    && ~isempty(dataStruct.opts.collectStart) ...
    && isfinite(dataStruct.opts.collectStart)
  t0 = dataStruct.opts.collectStart;
end
end
