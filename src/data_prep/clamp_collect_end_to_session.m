function collectEnd = clamp_collect_end_to_session(collectEnd, sessionEnd, collectStart)
% CLAMP_COLLECT_END_TO_SESSION - Cap collectEnd only when session is shorter
%
% Variables:
%   collectEnd   - Requested end time (s); [] or <= collectStart means full session
%   sessionEnd   - Available session end time (s)
%   collectStart - Analysis start time (s)
%
% Goal:
%   If collectEnd is empty or not after collectStart, use sessionEnd. If the
%   session is meaningfully shorter than collectEnd, clamp to sessionEnd.
%   Near-equality (e.g. last spike at 2699.95 vs collectEnd 2700) does not
%   trigger clamping.

if nargin < 3 || isempty(collectStart)
  collectStart = 0;
end
if isempty(sessionEnd) || ~isfinite(sessionEnd)
  error('clamp_collect_end_to_session:InvalidSessionEnd', ...
    'Could not determine session end time.');
end
% [] or a window that does not extend past collectStart means full session
if isempty(collectEnd) || collectEnd <= collectStart
  collectEnd = sessionEnd;
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
    ['collectEnd (%.1f) must be greater than collectStart (%.1f). ', ...
     'Session end is %.1f s (often caused by no spikes in the collect window).'], ...
    collectEnd, collectStart, sessionEnd);
end
end
