function metadata = semicircle_session_metadata(subjectName, sessionName)
% SEMICIRCLE_SESSION_METADATA - Per-session loading-option overrides
%
% Variables:
%   subjectName - Subject folder (e.g. 'AS1')
%   sessionName - Session identifier (e.g. 'AS1_0618_WellLearned')
%
% Goal:
%   Return metadata that apply_session_load_metadata merges into load opts.
%   Empty / omitted fields leave the caller's options unchanged.
%
% Returns:
%   metadata - Struct of overrides. Recognized fields (all optional):
%     .collectStartMin - Floor for opts.collectStart (seconds, not minutes):
%                        collectStart = max(requested, collectStartMin)
%     .collectEndMax   - Cap for opts.collectEnd (seconds); ignored if collectEnd is []
%     .collectStart, .collectEnd - Seconds, same as neuro_behavior_options
%     .minFiringRate, .maxFiringRate, ...
%                      - Any neuro_behavior_options field; nonempty values override
%     .notes           - Comment only; not copied onto opts

metadata = struct();
subjectName = normalize_metadata_string(subjectName);
sessionName = normalize_metadata_string(sessionName);

switch subjectName
  case 'AS1'
    switch sessionName
      case 'AS1_0618_WellLearned'
        % no overrides

      case 'AS1_0623_TransitionAfterCompletedTrial_80'
        % no overrides

      case 'AS1_0624_PoorlyLearned'
        % no overrides

      otherwise
        % no overrides
    end

  otherwise
    % no overrides
end
end

function value = normalize_metadata_string(value)
% NORMALIZE_METADATA_STRING - Char vector for switch matching

if nargin < 1 || isempty(value)
  value = '';
else
  value = char(value);
end
end
