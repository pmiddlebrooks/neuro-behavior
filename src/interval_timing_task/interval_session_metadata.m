function metadata = interval_session_metadata(subjectName, sessionName)
% INTERVAL_SESSION_METADATA - Per-session loading-option overrides
%
% Variables:
%   subjectName - Subject folder (e.g. 'ey9166')
%   sessionName - Session identifier (e.g. 'ey9166_2026_04_03')
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
  case 'ey9166'
    switch sessionName
      case 'ey9166_2026_04_02'
        % no overrides

      case 'ey9166_2026_04_03'
        metadata.collectStartMin = 10;  % seconds

      case 'ey9166_2026_04_04'
        % no overrides

      case 'ey9166_2026_04_07'
        % no overrides

      case 'ey9166_2026_04_09'
        metadata.collectStartMin = 610;  % seconds

      otherwise
        % no overrides
    end

  case 'ey9387'
    switch sessionName
      case 'ey9387_2026_05_19'
        % no overrides

      case 'ey9387_2026_05_20'
        % no overrides

      case 'ey9387_2026_05_21'
        % no overrides

      case 'ey9387_2026_05_22'
        % no overrides

      case 'ey9387_2026_05_25'
        % no overrides

      case 'ey9387_2026_05_26'
        % no overrides

      case 'ey9387_2026_05_27'
        % no overrides

      case 'ey9387_2026_05_28'
        % no overrides

      case 'ey9387_2026_06_01'
        % no overrides

      case 'ey9387_2026_06_02'
        % no overrides

      case 'ey9387_2026_06_05'
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
