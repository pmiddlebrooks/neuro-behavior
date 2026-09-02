function metadata = spontaneous_session_metadata(subjectName, sessionName)
% SPONTANEOUS_SESSION_METADATA - Per-session loading-option overrides
%
% Variables:
%   subjectName - Subject folder (e.g. 'ag25290')
%   sessionName - Session identifier (e.g. 'ag112321_1')
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
  case 'ag25290'
    switch sessionName
      case 'ag112221'
        % no overrides

      case 'ag112321_1'
        % no overrides

      case 'ag112321_2'
        % no overrides

      case 'ag112721'
        % no overrides

      case 'ag112821'
        % no overrides

      case 'ag112921'
        % no overrides

      otherwise
        % no overrides
    end

  case 'ey4152'
    switch sessionName
      case 'ey042822'
        % no overrides

      case 'ey042922'
        % no overrides

      otherwise
        % no overrides
    end

  case 'kw7193'
    switch sessionName
      case 'kw092821'
        % no overrides

      case 'kw092921'
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
