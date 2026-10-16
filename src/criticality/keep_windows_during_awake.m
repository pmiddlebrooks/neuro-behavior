function windowTimes = keep_windows_during_awake(dataStruct, config, windowTimes, ...
    windowSize, stepSize, anchor, timeRange)
% KEEP_WINDOWS_DURING_AWAKE - Place analysis windows inside awake epochs
%
% Variables:
%   dataStruct  - Loaded session
%   config      - Analysis config (restrictToAwake, optional sleep settings)
%   windowTimes - Centers or starts already laid out on the full collect range
%   windowSize  - Window length (s). The whole window must sit in one awake bout.
%   stepSize    - Step between windows (s)
%   anchor      - 'center' (time is the window midpoint) or 'start'
%   timeRange   - [start end] of the collect interval (s)
%
% Goal:
%   For spontaneous sessions, drop sleep. Windows are tiled inside each awake
%   bout so a window never mixes sleep and wake. Other session types keep
%   windowTimes unchanged. A session that is awake throughout keeps the
%   original grid, because the single awake bout is snapped to timeRange.
%
% Returns:
%   windowTimes - Row vector of centers or starts to analyze

if nargin < 7
  error('keep_windows_during_awake:BadInput', ...
    'Usage: keep_windows_during_awake(dataStruct, config, windowTimes, windowSize, stepSize, anchor, timeRange)');
end

[segments, applyFilter] = resolve_spontaneous_awake_segments(dataStruct, config, timeRange);
if ~applyFilter
  return;
end

anchor = lower(strtrim(char(anchor)));
if ~isscalar(windowSize) || ~isscalar(stepSize) || ~isfinite(windowSize) || ~isfinite(stepSize) ...
    || windowSize <= 0 || stepSize <= 0
  error('keep_windows_during_awake:BadWindow', ...
    'Awake filtering needs a positive scalar window size and step.');
end
windowTimes = tile_windows_in_segments(segments, windowSize, stepSize, anchor);
nWindows = numel(windowTimes);
sessionName = '';
if isfield(dataStruct, 'sessionName') && ~isempty(dataStruct.sessionName)
  sessionName = char(dataStruct.sessionName);
end
if nWindows < 1
  boutDurations = zeros(1, numel(segments));
  for iSeg = 1:numel(segments)
    boutDurations(iSeg) = segments(iSeg).end - segments(iSeg).start;
  end
  error('keep_windows_during_awake:NoAwakeWindows', ...
    ['No %.1f s windows fit inside awake epochs for %s. ', ...
    'Awake bouts (s): %s. Use a shorter window or check lfp_sleep_detector.'], ...
    windowSize, sessionName, mat2str(boutDurations, 4));
end

fprintf('Awake-only windows: %d x %.1f s (%d awake epoch(s), sleep excluded)\n', ...
  nWindows, windowSize, numel(segments));
end

function windowTimes = tile_windows_in_segments(segments, windowSize, stepSize, anchor)
% TILE_WINDOWS_IN_SEGMENTS - Non-overlapping or stepped windows fully inside bouts
%
% Variables:
%   segments   - Struct array with .start and .end (s)
%   windowSize - Window length (s)
%   stepSize   - Step (s)
%   anchor     - 'center' or 'start'
%
% Returns:
%   windowTimes - Row vector. Empty when nothing fits.

windowTimes = zeros(1, 0);
if isempty(segments) || ~isscalar(windowSize) || ~isfinite(windowSize) || windowSize <= 0
  return;
end
if ~isscalar(stepSize) || ~isfinite(stepSize) || stepSize <= 0
  return;
end

edgeTol = 1e-3;
for iSeg = 1:numel(segments)
  segStart = segments(iSeg).start;
  segEnd = segments(iSeg).end;
  if strcmp(anchor, 'start')
    windowTimes = [windowTimes, starts_inside_segment(segStart, segEnd, windowSize, stepSize, edgeTol)]; %#ok<AGROW>
  else
    windowTimes = [windowTimes, centers_inside_segment(segStart, segEnd, windowSize, stepSize, edgeTol)]; %#ok<AGROW>
  end
end
end

function centers = centers_inside_segment(segStart, segEnd, windowSize, stepSize, edgeTol)
% CENTERS_INSIDE_SEGMENT - Midpoints whose [center ± window/2] stays in the bout
firstCenter = segStart + windowSize / 2;
lastCenter = segEnd - windowSize / 2;
centers = zeros(1, 0);
if lastCenter + edgeTol < firstCenter
  return;
end
centers = firstCenter:stepSize:lastCenter;
if isempty(centers)
  centers = firstCenter;
end
end

function starts = starts_inside_segment(segStart, segEnd, windowSize, stepSize, edgeTol)
% STARTS_INSIDE_SEGMENT - Starts whose [start, start+window] stays in the bout
starts = zeros(1, 0);
windowStart = segStart;
while windowStart + windowSize <= segEnd + edgeTol
  starts(end + 1) = windowStart; %#ok<AGROW>
  windowStart = windowStart + stepSize;
end
end
