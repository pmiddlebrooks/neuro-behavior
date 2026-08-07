function frameIdx = abs_time_to_collect_frame(absTime, collectStart, frameSize, roundMode)
% ABS_TIME_TO_COLLECT_FRAME - Map absolute session time to collect-window frame
%
% Variables:
%   absTime      - Absolute session time(s) in seconds
%   collectStart - Absolute start of collect window (s); default 0
%   frameSize    - Frame/bin width in seconds
%   roundMode    - 'round' (default) or 'floor'
%
% Goal:
%   Convert absolute bout/event times to 1-based frame indices where frame 1
%   corresponds to collectStart (matches neural matrices / bhvID vectors).
%
% Returns:
%   frameIdx - Same size as absTime

if nargin < 2 || isempty(collectStart)
  collectStart = 0;
end
if nargin < 3 || isempty(frameSize) || ~(isfinite(frameSize) && frameSize > 0)
  error('abs_time_to_collect_frame:frameSize', 'frameSize must be positive.');
end
if nargin < 4 || isempty(roundMode)
  roundMode = 'round';
end

relTime = absTime - collectStart;
switch lower(roundMode)
  case 'floor'
    frameIdx = 1 + floor(relTime ./ frameSize);
  case 'round'
    frameIdx = 1 + round(relTime ./ frameSize);
  otherwise
    error('abs_time_to_collect_frame:roundMode', ...
      'roundMode must be ''round'' or ''floor''.');
end
end
