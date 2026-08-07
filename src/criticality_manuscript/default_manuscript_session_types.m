function sessionTypes = default_manuscript_session_types()
% DEFAULT_MANUSCRIPT_SESSION_TYPES - Default task set for across-task scripts
%
% Variables:
%   (none)
%
% Goal:
%   One editable list for manuscript batch defaults, already in plot order:
%   spontaneous -> interval -> semicircle -> reach.
%   Engagement-only pipelines still filter to tasks with engagement modules.
%
% Returns:
%   sessionTypes - Cell array of sessionType strings

sessionTypes = order_manuscript_session_types( ...
  {'spontaneous', 'interval', 'semicircle', 'reach'});
end
