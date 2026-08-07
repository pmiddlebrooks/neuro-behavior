function [bhvID, bhvTimeOrigin] = build_bhv_id_vector(dataBhv, collectStart, collectEnd, fsBhv)
% BUILD_BHV_ID_VECTOR - Behavior code per frame for absolute collect window
%
% Variables:
%   dataBhv       - Table/struct with absolute StartTime (s) and ID per bout;
%                   Dur (s) used when present so the last bout is not extended
%                   past its labeled end
%   collectStart  - Absolute session start of analysis window (s)
%   collectEnd    - Absolute session end of analysis window (s)
%   fsBhv         - Behavior sampling rate (Hz)
%
% Goal:
%   Build bhvID covering [collectStart, collectEnd] in absolute session time.
%   Frame 1 maps to absolute time collectStart (bhvTimeOrigin).
%
% Returns:
%   bhvID         - [nFrames x 1] behavior codes (0 where unlabeled)
%   bhvTimeOrigin - Absolute time (s) of bhvID(1); equals collectStart

if nargin < 2 || isempty(collectStart)
  collectStart = 0;
end
if nargin < 3 || isempty(collectEnd)
  error('build_bhv_id_vector:collectEnd', 'collectEnd is required.');
end
if nargin < 4 || isempty(fsBhv) || ~(isfinite(fsBhv) && fsBhv > 0)
  error('build_bhv_id_vector:fsBhv', 'fsBhv must be a positive finite rate.');
end
if collectEnd <= collectStart
  error('build_bhv_id_vector:timeRange', ...
    'collectEnd (%.3f) must be greater than collectStart (%.3f).', collectEnd, collectStart);
end

bhvTimeOrigin = collectStart;
bhvBinSize = 1 / fsBhv;
nBhvBins = max(1, ceil((collectEnd - collectStart) / bhvBinSize));
bhvID = zeros(nBhvBins, 1);

if isempty(dataBhv) || (~istable(dataBhv) && ~isstruct(dataBhv))
  return;
end
if isstruct(dataBhv)
  if ~isfield(dataBhv, 'StartTime') || ~isfield(dataBhv, 'ID')
    return;
  end
  startTimes = dataBhv.StartTime(:);
  ids = dataBhv.ID(:);
  if isfield(dataBhv, 'Dur') && ~isempty(dataBhv.Dur)
    durs = dataBhv.Dur(:);
  else
    durs = [];
  end
else
  if ~ismember('StartTime', dataBhv.Properties.VariableNames) ...
      || ~ismember('ID', dataBhv.Properties.VariableNames)
    return;
  end
  startTimes = dataBhv.StartTime(:);
  ids = dataBhv.ID(:);
  if ismember('Dur', dataBhv.Properties.VariableNames)
    durs = dataBhv.Dur(:);
  else
    durs = [];
  end
end

nBout = numel(startTimes);
if nBout < 1
  return;
end
if ~isempty(durs) && numel(durs) ~= nBout
  durs = [];
end

% Absolute bout boundaries -> collect-window frame indices
startFrames = abs_time_to_collect_frame(startTimes, collectStart, bhvBinSize, 'round');
if ~isempty(durs)
  endFrames = abs_time_to_collect_frame(startTimes + durs, collectStart, bhvBinSize, 'round');
else
  endFrames = [startFrames(2:end) - 1; nBhvBins];
end

for i = 1:nBout
  iStart = max(1, startFrames(i));
  iEnd = min(nBhvBins, endFrames(i));
  if iStart <= iEnd
    bhvID(iStart:iEnd) = ids(i);
  end
end
end
