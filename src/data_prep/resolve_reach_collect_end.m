function collectEnd = resolve_reach_collect_end(collectEnd, sessionEnd, collectStart)
% RESOLVE_REACH_COLLECT_END - Resolve reach collectEnd; [] omits final 180 s
%
% Variables:
%   collectEnd   - Requested end time (s); [] means full session minus tail
%   sessionEnd   - Available reach session end time (s)
%   collectStart - Analysis start time (s)
%
% Goal:
%   When collectEnd is empty, analyze through sessionEnd - 180 s (omit the
%   last 3 minutes of the reach recording). Otherwise clamp to sessionEnd
%   via clamp_collect_end_to_session.

if nargin < 3 || isempty(collectStart)
  collectStart = 0;
end

reachTailOmitSec = 180;

if isempty(collectEnd)
  if (sessionEnd - collectStart) > reachTailOmitSec
    collectEnd = sessionEnd - reachTailOmitSec;
    fprintf(['  Reach collectEnd=[]: omitting last %.0f s ', ...
      '(collectEnd=%.1f s of sessionEnd=%.1f s).\n'], ...
      reachTailOmitSec, collectEnd, sessionEnd);
  else
    warning('resolve_reach_collect_end:ShortSession', ...
      ['Reach session (%.1f s from collectStart) is not longer than %.0f s tail omit; ', ...
      'using full sessionEnd=%.1f s.'], ...
      sessionEnd - collectStart, reachTailOmitSec, sessionEnd);
    collectEnd = sessionEnd;
  end
end

collectEnd = clamp_collect_end_to_session(collectEnd, sessionEnd, collectStart);
end
