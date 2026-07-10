function collectEnd = clamp_collect_end_to_session(collectEnd, sessionEnd, collectStart)
% CLAMP_COLLECT_END_TO_SESSION - Cap collectEnd only when session is shorter
%
% Variables:
%   collectEnd   - Requested end time (s); [] means use full session
%   sessionEnd   - Available session end time (s)
%   collectStart - Analysis start time (s)
%
% Goal:
%   If collectEnd is empty, use sessionEnd. If the session is meaningfully
%   shorter than collectEnd, clamp to sessionEnd. Near-equality (e.g. last
%   spike at 2699.95 vs collectEnd 2700) does not trigger clamping.

if nargin < 3 || isempty(collectStart)
  collectStart = 0;
end
if isempty(sessionEnd) || ~isfinite(sessionEnd)
  error('clamp_collect_end_to_session:InvalidSessionEnd', ...
    'Could not determine session end time.');
end
if isempty(collectEnd)
  collectEnd = sessionEnd;
  return;
end

% Only clamp when session is clearly shorter than requested collectEnd
toleranceSec = 1;
if sessionEnd < (collectEnd - toleranceSec)
  fprintf('  Session ends at %.1f s (shorter than collectEnd %.1f s); analyzing full session.\n', ...
    sessionEnd, collectEnd);
  collectEnd = sessionEnd;
end
if collectEnd <= collectStart
  error('clamp_collect_end_to_session:InvalidCollectWindow', ...
    'collectEnd (%.1f) must be greater than collectStart (%.1f).', collectEnd, collectStart);
end
end
